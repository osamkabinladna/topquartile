import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Type
import re
from collections import defaultdict
import warnings

from topquartile.modules.datamodule.transforms.covariate import CovariateTransform
from topquartile.modules.datamodule.transforms.label import LabelTransform
from topquartile.modules.datamodule.partitions import (
    BasePurgedTimeSeriesPartition,
    PurgedTimeSeriesPartition,
    PurgedGroupTimeSeriesPartition,
)


class DataLoader:
    def __init__(
            self,
            data_id: str,
            covariate_transform: Optional[
                List[Tuple[Type[CovariateTransform], Dict]]] = None,
            label_transform: Optional[
                List[Tuple[Type[LabelTransform], Dict]]] = None,
            cols2drop: Optional[List[str]] = None,
            prediction_length: int = 20,
            partition_class: Type[BasePurgedTimeSeriesPartition] = PurgedTimeSeriesPartition,
            partition_kwargs: Optional[Dict] = None,
            label_per_partition: bool = False,
    ):
        self.data_id = data_id
        self.covariate_transform_config = covariate_transform or []
        self.label_transform_config = label_transform or []
        self.cols2drop = cols2drop or ["NEWS_SENTIMENT_DAILY_AVG"]
        self.prediction_length = prediction_length
        self.label_per_partition = label_per_partition

        if not issubclass(partition_class, BasePurgedTimeSeriesPartition):
            raise ValueError(
                "partition_class must inherit from BasePurgedTimeSeriesPartition"
            )
        self.partitioner: BasePurgedTimeSeriesPartition = partition_class(**(partition_kwargs or {}))

        if not issubclass(partition_class, BasePurgedTimeSeriesPartition):
            raise ValueError("partition_class must inherit from BasePurgedTimeSeriesPartition")
        if partition_kwargs is None:
            partition_kwargs = {}
        self.partitioner: BasePurgedTimeSeriesPartition = partition_class(**partition_kwargs)


        self.data: Optional[pd.DataFrame] = None
        self.tickernames: List[str] = []
        self.required_covariates: set[str] = set()
        self.preds: Optional[pd.DataFrame] = None

        try:
            root_path = Path(__file__).resolve().parent.parent.parent
        except NameError:
             root_path = Path().resolve()

        self.covariates_path = root_path / "data" / f"{self.data_id}.csv"


    def load_preds(self) -> pd.DataFrame:
        if self.data is None:
            print("Data not loaded. Processing data...")
            try:
                self._process_data()
            except FileNotFoundError:
                print(f"ERROR: Data file not found at {self.covariates_path}")
                raise
            except Exception as e:
                print(f"Error during _process_data: {e}")
                raise

        if self.data is None or self.data.empty:
             raise ValueError("Data could not be loaded or is empty after processing.")

        print(f"Separating last {self.prediction_length} rows per ticker for predictions.")
        self.data = self.data.sort_index()

        self.preds = (
            self.data.groupby("ticker", group_keys=False)
            .tail(self.prediction_length)
            .copy()
        )
        if not isinstance(self.data.index, pd.DatetimeIndex):
             self.data.index = pd.to_datetime(self.data.index)
        if not isinstance(self.preds.index, pd.DatetimeIndex):
             self.preds.index = pd.to_datetime(self.preds.index)

        remaining_index = self.data.index.difference(self.preds.index)
        self.data = self.data.loc[remaining_index]

        print(f"Predictions shape: {self.preds.shape}, Remaining data shape: {self.data.shape}")
        return self.preds

    def _process_data(self):
        self._load_data()
        self.transform_data()
        print("Data processing complete.")

    def transform_data(self):
        if self.data is None:
            self._load_data()
            if self.data is None: # Still None after trying to load
                raise ValueError("Data could not be loaded in transform_data.")


        for TransformClass, params in self.covariate_transform_config:
            if not issubclass(TransformClass, CovariateTransform):
                raise ValueError(
                    "Invalid transform in covariate_transform_config: must subclass CovariateTransform"
                )
            print(f" Applying {TransformClass.__name__} with params {params}")
            transformer = TransformClass(df=self.data, **params)
            self.data = transformer.transform()
            self.required_covariates.update(transformer.required_base)

        if not self.label_per_partition:
            for TransformClass, params in self.label_transform_config:
                if not issubclass(TransformClass, LabelTransform):
                    raise ValueError(
                        "Invalid transform in label_transform_config: must subclass LabelTransform"
                    )
                print(f" Applying {TransformClass.__name__} with params {params} (globally)")
                transformer = TransformClass(df=self.data, **params)
                self.data = transformer.transform()
        return self.data

    def _load_data(self) -> pd.DataFrame:
        print(f"Reading data from: {self.covariates_path}")
        try:
             if not self.covariates_path.is_file():
                 raise FileNotFoundError(f"Data file not found at {self.covariates_path}")

             ticker_header_df = pd.read_csv(self.covariates_path, nrows=4, header=None)
             ticker_row_index = 3
             for i, row in enumerate(ticker_header_df.values):
                 if any(':' in str(cell) for cell in row if pd.notna(cell)):
                     ticker_row_index = i
                     break

             ticker_df = pd.read_csv(self.covariates_path, skiprows=ticker_row_index, nrows=1, header=None)
             raw_tickernames = [str(col) for col in ticker_df.iloc[0] if pd.notna(col) and not str(col).lower().startswith("unnamed")]

             covariates_header_row = ticker_row_index + 2
             covariates = pd.read_csv(
                 self.covariates_path, skiprows=covariates_header_row, index_col=0, low_memory=False
             )
        except Exception as e:
             print(f"Error reading CSV file structure: {e}")
             raise

        covariates.dropna(inplace=True, axis=0, how="all")
        covariates.dropna(inplace=True, axis=1, how="all")
        covariates.index = pd.to_datetime(covariates.index, errors='coerce', format='mixed')
        covariates = covariates[covariates.index.notna()]

        num_tickers = len(raw_tickernames)
        print(f"Found {num_tickers} raw ticker names.")

        col_dict: dict[int, list[str]] = defaultdict(list)
        for col in covariates.columns:
            number = self._get_number(col)
            if number < num_tickers :
                 col_dict[number].append(col)

        covlist: list[pd.DataFrame] = [pd.DataFrame(index=covariates.index.copy()) for _ in range(num_tickers)]


        for number in range(num_tickers):
            cols_for_ticker = col_dict.get(number)
            if cols_for_ticker:
                df_part = covariates[cols_for_ticker].copy()
                df_part.columns = [self._get_base_col_name(col) for col in cols_for_ticker]
                covlist[number] = df_part

        processed_tickernames = [re.sub(r'\s.*', '', ticker).strip() for ticker in raw_tickernames]

        first_occurrence: dict[str, int] = {}
        unique_indices: list[int] = []
        unique_tickernames_list: list[str] = []
        for idx, ticker in enumerate(processed_tickernames):
            if ticker not in first_occurrence:
                first_occurrence[ticker] = idx
                unique_indices.append(idx)
                unique_tickernames_list.append(ticker)

        unique_covlist = [covlist[idx] for idx in unique_indices if idx < len(covlist)]
        self.tickernames = unique_tickernames_list

        final_covlist = []
        for idx, cov_df in enumerate(unique_covlist):
             if not cov_df.empty:
                 cov_copy = cov_df.copy()
                 cov_copy["ticker"] = self.tickernames[idx]
                 final_covlist.append(cov_copy)

        if not final_covlist:
             print("Warning: No dataframes to concatenate after processing.")
             self.data = pd.DataFrame()
        else:
             self.data = pd.concat(final_covlist)
             self.data['ticker'] = self.data['ticker'].astype('category')
             self.data.sort_index(inplace=True)
        if self.data.empty:
            print("Warning: Data is empty after loading and initial processing.")
        return self.data

    def _get_base_col_name(self, col_name: str) -> str:
         return str(col_name).split('.')[0]

    def _get_number(self, col_name: str) -> int:
        match = re.match(r"^(.*?)(?:\.(\d+))?$", str(col_name))
        if match and match.group(2) is not None:
            try:
                return int(match.group(2))
            except ValueError:
                return -1
        return -1

    def _partition_data(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if self.data is None or self.data.empty:
            print("Data not available for partitioning. Attempting to process...")
            try:
                 self._process_data()
            except Exception as e:
                 print(f"Failed to process data for partitioning: {e}")
                 return []
            if self.data is None or self.data.empty:
                 print("Data is still empty after processing. Cannot partition.")
                 return []

        self.data = self.data.sort_index()
        self.data.index.set_names('TickerIndex', level='ticker', inplace=True)
        self.data.index.set_names('DateIndex', level='Dates', inplace=True)

        n_splits = self.partitioner.get_n_splits()
        fold_buckets = [
            {"train": [], "test": []} for _ in range(n_splits)
        ]

        data_grouped_by_ticker = self.data.groupby("ticker")
        print(f"Partitioning data using {self.partitioner.__class__.__name__} for {n_splits} splits across {len(data_grouped_by_ticker)} tickers.")

        for ticker, df_ticker in data_grouped_by_ticker:
            df_ticker = df_ticker.sort_index()

            min_samples_for_splitter = n_splits
            if isinstance(self.partitioner, PurgedTimeSeriesPartition) and self.partitioner.test_size is not None:
                min_samples_for_splitter = self.partitioner.test_size * n_splits + self.partitioner.gap * (n_splits -1)
                if self.partitioner.max_train_size is not None:
                     min_samples_for_splitter = max(min_samples_for_splitter, self.partitioner.test_size + self.partitioner.gap + 1)


            if len(df_ticker) < min_samples_for_splitter :
                 warnings.warn(
                     f"Ticker {ticker} has only {len(df_ticker)} rows. "
                     f"This might be too short for {n_splits} splits with current settings. Skipping or may result in fewer splits."
                 )


            groups = None
            if isinstance(self.partitioner, PurgedGroupTimeSeriesPartition):
                if not isinstance(df_ticker.index, pd.DatetimeIndex):
                    warnings.warn(f"Ticker {ticker} index is not DatetimeIndex. Cannot generate groups for PurgedGroupTimeSeriesPartition. Skipping ticker.")
                    continue
                groups = df_ticker.index.normalize()
                if len(groups) == 0:
                    warnings.warn(f"Ticker {ticker} resulted in 0 groups. Skipping ticker for PurgedGroupTimeSeriesPartition.")
                    continue
                print(f" Using date groups for ticker {ticker} with PurgedGroupTimeSeriesPartition.")


            try:
                ticker_splits = list(self.partitioner.split(X=df_ticker, groups=groups))

                if not ticker_splits:
                    warnings.warn(f"Ticker {ticker} produced no splits. This might be due to short series or partitioner settings. Skipping this ticker.")
                    continue

                if len(ticker_splits) != n_splits:
                     warnings.warn(
                         f"Ticker {ticker} produced {len(ticker_splits)} splits, but partitioner is configured for {n_splits} splits. "
                         f"This can happen with short series or specific purge/embargo settings. Folds will be constructed with available splits."
                     )

                for fold_id, (train_idx, test_idx) in enumerate(ticker_splits):
                     if fold_id >= n_splits:
                         break

                     if train_idx.size == 0 or test_idx.size == 0:
                         warnings.warn(f"Ticker {ticker}, fold {fold_id} resulted in empty train or test set. Skipping this part.")
                         continue

                     train_df_part = df_ticker.iloc[train_idx].copy()
                     test_df_part = df_ticker.iloc[test_idx].copy()

                     if self.label_per_partition and self.label_transform_config:
                         current_train_data = train_df_part
                         for TransformClass, params in self.label_transform_config:
                             if not issubclass(TransformClass, LabelTransform):
                                 raise ValueError("Invalid transform in label_transform_config.")
                             transformer = TransformClass(df=current_train_data, **params)
                             current_train_data = transformer.transform()
                         train_df_part = current_train_data

                         current_test_data = test_df_part
                         for TransformClass, params in self.label_transform_config:
                             if not issubclass(TransformClass, LabelTransform):
                                 raise ValueError("Invalid transform in label_transform_config.")
                             transformer = TransformClass(df=current_test_data, **params)
                             current_test_data = transformer.transform()
                         test_df_part = current_test_data

                     fold_buckets[fold_id]["train"].append(train_df_part)
                     fold_buckets[fold_id]["test"].append(test_df_part)

            except ValueError as ve:
                 warnings.warn(f"Error partitioning ticker {ticker}: {ve}. Skipping this ticker.")
                 continue
            except Exception as e:
                 warnings.warn(f"Unexpected error partitioning ticker {ticker}: {e}. Skipping this ticker.")
                 continue


        cv_folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for fold_id, bucket in enumerate(fold_buckets):
            if bucket["train"] and bucket["test"]:
                train_df_fold = pd.concat(bucket["train"]).sort_index()
                test_df_fold = pd.concat(bucket["test"]).sort_index()
                cv_folds.append((train_df_fold, test_df_fold))
                print(f"Fold {fold_id}: Train shape {train_df_fold.shape}, Test shape {test_df_fold.shape}")
            else:
                 warnings.warn(f"Fold {fold_id} will be skipped as it contained no data after processing all tickers.")

        if not cv_folds:
            warnings.warn("Partitioning did not result in any valid CV folds. Check data length, ticker data, and partitioner settings.")

        print(f"Partitioning complete. Generated {len(cv_folds)} CV folds.")
        return cv_folds

    def get_cv_folds(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Method to get cross-validation folds.
        Ensures data is processed and then partitioned.
        """
        if self.data is None:
            print("Data not yet processed. Processing now...")
            self._process_data()
            if self.data is None or self.data.empty:
                 raise ValueError("Data could not be loaded or is empty after processing, cannot generate folds.")

        return self._partition_data()