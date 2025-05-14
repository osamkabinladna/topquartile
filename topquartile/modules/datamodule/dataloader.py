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
        partition_params: Optional[Dict] = None,
    ):
        self.data_id = data_id
        self.covariate_transform_config = covariate_transform or []
        self.label_transform_config = label_transform or []
        self.cols2drop = cols2drop or ["NEWS_SENTIMENT_DAILY_AVG"]
        self.prediction_length = prediction_length

        if not issubclass(partition_class, BasePurgedTimeSeriesPartition):
            raise ValueError(
                "partition_class must inherit from BasePurgedTimeSeriesPartition"
            )
        self.partitioner: BasePurgedTimeSeriesPartition = partition_class(
            **(partition_params or {})
        )

        self.data: Optional[pd.DataFrame] = None
        self.tickernames: List[str] = []
        self.required_covariates: set[str] = set()
        self.preds: Optional[pd.DataFrame] = None

        try:
            root_path = Path(__file__).resolve().parent.parent.parent
        except NameError:
             print("Warning: __file__ not defined. Assuming current working directory structure.")
             root_path = Path(".") # Or provide a specific path to your project root

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
        self.transform_data()
        self._impute_columns()
        print("Data processing complete.")

    def transform_data(self):
        if self.data is None:
            self._load_data()

        for TransformClass, params in self.covariate_transform_config:
            if not issubclass(TransformClass, CovariateTransform):
                raise ValueError(
                    "Invalid transform in covariate_transform_config: must subclass CovariateTransform"
                )
            print(f" Applying {TransformClass.__name__} with params {params}")
            transformer = TransformClass(df=self.data, **params)
            self.data = transformer.transform()
            self.required_covariates.update(transformer.required_base)

        for TransformClass, params in self.label_transform_config:
            if not issubclass(TransformClass, LabelTransform):
                raise ValueError(
                    "Invalid transform in label_transform_config: must subclass LabelTransform"
                )
            print(f" Applying {TransformClass.__name__} with params {params}")
            if 'target_column' in params and params['target_column'] not in self.data.columns:
                 print(f"Warning: Target column '{params['target_column']}' not found for {TransformClass.__name__}. Skipping or creating dummy.")
                 if 'PX_LAST' in self.data.columns:
                     self.data[params['target_column']] = self.data['PX_LAST']
                 else:
                     self.data[params['target_column']] = 0

            transformer = TransformClass(df=self.data, **params)
            self.data = transformer.transform()
        return self.data

    def _load_data(self) -> pd.DataFrame:
        print(f"Reading data from: {self.covariates_path}")
        try:
             if not self.covariates_path.is_file():
                 raise FileNotFoundError(f"Data file not found at {self.covariates_path}")

             ticker_header_df = pd.read_csv(self.covariates_path, nrows=4, header=None) # Read first few rows to find tickers
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
        print(covariates.index.name)
        # covariates.dropna(axis=0, subset=[covariates.index.name], inplace=True)

        num_tickers = len(raw_tickernames)
        print(f"Found {num_tickers} raw ticker names.")

        col_dict: dict[int, list[str]] = defaultdict(list)
        for col in covariates.columns:
            number = self._get_number(col)
            if number < num_tickers :
                 col_dict[number].append(col)


        covlist: list[pd.DataFrame] = [pd.DataFrame(index=covariates.index) for _ in range(num_tickers)]

        for number in range(num_tickers):
            cols = col_dict.get(number, [])
            if cols:
                df_part = covariates[cols].copy()
                df_part.columns = [self._get_base_col_name(col) for col in cols]
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

        print(f"Using {len(self.tickernames)} unique tickers.")
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


        print(f"Data loaded. Shape: {self.data.shape}")
        return self.data

    def _get_base_col_name(self, col_name: str) -> str:
         return str(col_name).split('.')[0]

    def _get_number(self, col_name: str) -> int:
        """
        thank you chat gpt, i could never in a million years figure out how to do this
        """
        match = re.match(r"^(.*?)(?:\.(\d+))?$", str(col_name))
        if match and match.group(2) is not None:
            try:
                return int(match.group(2))
            except ValueError:
                return 0
        return 0

    def _impute_columns(self):
        print("Imputing/dropping columns based on missingness...")
        if self.data is None or self.data.empty:
            print("No data to impute.")
            return

        # Ensure required covariates exist before checking missingness
        valid_required_covariates = [col for col in self.required_covariates if col in self.data.columns]
        if not valid_required_covariates:
            print("Warning: No required covariates found in data for imputation thresholding.")
            missing_value_threshold = len(self.data) # Set threshold high to avoid dropping based on this
        else:
             max_missing_in_required = self.data[valid_required_covariates].isna().sum().max()
             missing_value_threshold = max_missing_in_required


        missing_value_all = self.data.isna().sum()
        columns_to_drop_missingness = missing_value_all[missing_value_all > missing_value_threshold].index.tolist()

        combined_cols_to_drop = list(set(columns_to_drop_missingness + self.cols2drop))

        final_cols_to_drop = [
             col for col in combined_cols_to_drop
             if col in self.data.columns and col not in self.required_covariates or col in self.cols2drop
        ]


        if columns_to_drop_missingness:
             print(f" Columns with missingness > threshold ({missing_value_threshold}): {columns_to_drop_missingness}")
        if self.cols2drop:
             print(f" Explicitly dropping columns: {self.cols2drop}")

        if final_cols_to_drop:
            print(f" Final columns to drop: {final_cols_to_drop}")
            self.data = self.data.drop(columns=final_cols_to_drop, errors="ignore")
            print(f"Data shape after dropping columns: {self.data.shape}")
        else:
            print("No columns dropped based on missingness criteria or explicit list.")


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


        print(f"Partitioning data using {self.partitioner.__class__.__name__} with {self.partitioner.n_splits} splits.")
        self.data = self.data.sort_index()

        fold_buckets = [
            {"train": [], "test": []} for _ in range(self.partitioner.n_splits)
        ]

        data_grouped_by_ticker = self.data.groupby("ticker")
        print(f"Found {len(data_grouped_by_ticker)} tickers for partitioning.")

        for ticker, df_ticker in data_grouped_by_ticker:
            df_ticker = df_ticker.sort_index()

            if len(df_ticker) < self.partitioner.n_splits:
                 warnings.warn(f"Ticker {ticker} has only {len(df_ticker)} rows, fewer than n_splits={self.partitioner.n_splits}. Skipping this ticker for partitioning.")
                 continue

            if isinstance(self.partitioner, PurgedGroupTimeSeriesPartition):
                groups = df_ticker.index.normalize()
                print(f" Using date groups for ticker {ticker} with PurgedGroupTimeSeriesPartition.")
            else:
                groups = None # Standard time series split doesn't need explicit groups


            try:
                splits = list(self.partitioner.split(df_ticker, groups=groups))
                if len(splits) != self.partitioner.n_splits:
                     warnings.warn(
                         f"Ticker {ticker} produced {len(splits)} splits but partitioner is configured for {self.partitioner.n_splits}. This might happen with short series or large purge/embargo values."
                         )
                     if not splits: continue

                for fold_id, (train_idx, test_idx) in enumerate(splits):
                     if fold_id >= self.partitioner.n_splits: break # Ensure we don't exceed bucket count

                     # Check if indices are valid before iloc
                     if train_idx.size > 0 and test_idx.size > 0:
                          fold_buckets[fold_id]["train"].append(df_ticker.iloc[train_idx])
                          fold_buckets[fold_id]["test"].append(df_ticker.iloc[test_idx])

            except Exception as e:
                 warnings.warn(f"Error partitioning ticker {ticker}: {e}. Skipping this ticker.")
                 continue


        cv_folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for fold_id, bucket in enumerate(fold_buckets):
            if bucket["train"] and bucket["test"]:
                train_df = pd.concat(bucket["train"]).sort_index()
                test_df = pd.concat(bucket["test"]).sort_index()
                cv_folds.append((train_df, test_df))
            else:
                 print(f" Fold {fold_id} skipped as it contained no data after processing all tickers.")

        print(f"Partitioning complete. Generated {len(cv_folds)} folds.")
        return cv_folds