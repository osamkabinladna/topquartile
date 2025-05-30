import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Type
import re
from collections import defaultdict
import warnings
import numpy as np

from topquartile.modules.datamodule.transforms.covariate import CovariateTransform
from topquartile.modules.datamodule.transforms.label import LabelTransform
from topquartile.modules.datamodule.partitions import (
    BasePurgedTimeSeriesPartition,
    PurgedTimeSeriesPartition,
    PurgedGroupTimeSeriesPartition,
)

import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


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
            self.root_path = Path(__file__).resolve().parent.parent.parent
        except NameError:
            self.root_path = Path().resolve().parent.parent.parent

        self.covariates_path = self.root_path / "data" / f"{self.data_id}.csv"

    def load_preds(self) -> pd.DataFrame:
        # TODO: Revise this for later prediction
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

        preds_group_key = 'ticker'
        if isinstance(self.data.index, pd.MultiIndex) and 'ticker' in self.data.index.names:
            pass
        elif 'ticker' not in self.data.columns:
            if isinstance(self.data.index, pd.MultiIndex) and 'TickerIndex' in self.data.index.names:
                preds_group_key = 'TickerIndex'
            else:
                raise ValueError("load_preds: 'ticker' not found as column or index level for grouping.")

        self.preds = (
            self.data.groupby(preds_group_key, group_keys=False)
            .tail(self.prediction_length)
            .copy()
        )

        if not isinstance(self.data.index, pd.DatetimeIndex):
            if not (isinstance(self.data.index, pd.MultiIndex) and isinstance(self.data.index.get_level_values(-1),
                                                                              pd.DatetimeIndex)):
                pass

        if not isinstance(self.preds.index, pd.DatetimeIndex):
            if not (isinstance(self.preds.index, pd.MultiIndex) and isinstance(self.preds.index.get_level_values(-1),
                                                                               pd.DatetimeIndex)):
                pass

        remaining_index = self.data.index.difference(self.preds.index)
        self.data = self.data.loc[remaining_index]

        print(f"Predictions shape: {self.preds.shape}, Remaining data shape: {self.data.shape}")
        return self.preds

    def _process_data(self):
        self._load_data()
        self.transform_covariates()
        print("Data processing complete.")

    def transform_covariates(self):
        if self.data is None:
            self._load_data()
            if self.data is None:
                raise ValueError("Data could not be loaded in transform_data.")
    
        if not isinstance(self.data.index, pd.MultiIndex) or set(self.data.index.names) != {'ticker', 'Dates'}:
            print(f"Fixing index: current index names = {self.data.index.names}")

            if not isinstance(self.data.index, pd.MultiIndex):
                if 'Dates' in self.data.columns and 'ticker' in self.data.columns:
                    self.data = self.data.set_index(['ticker', 'Dates'])
                elif 'ticker' in self.data.columns:
                    self.data.index.name = 'Dates'
                    self.data = self.data.set_index('ticker', append=True)
                    self.data = self.data.reorder_levels(['ticker', 'Dates'])
                else:
                    raise ValueError("Cannot set MultiIndex — missing 'ticker' column.")
            else:
                if len(self.data.index.levels) == 2:
                    self.data.index = self.data.index.set_names(['ticker', 'Dates'])
                else:
                    raise ValueError(f"Expected 2-level MultiIndex, got {self.data.index.names}")

            self.data = self.data.sort_index()

        for TransformClass, params in self.covariate_transform_config:
            if not issubclass(TransformClass, CovariateTransform):
                raise ValueError(
                    "Invalid transform in covariate_transform_config: must subclass CovariateTransform"
                )
            print(f" Applying {TransformClass.__name__} with params {params}")
            transformer = TransformClass(df=self.data, **params)
            self.data = transformer.transform()
            self.required_covariates.update(transformer.required_base)

        if not self.label_per_partition and self.label_transform_config:
            print("Applying label transformations globally to the dataset (before partitioning).")

            if not (isinstance(self.data.index, pd.MultiIndex) and \
                    list(self.data.index.names) == ['ticker', 'Dates']):

                if 'ticker' not in self.data.columns and \
                        (not isinstance(self.data.index, pd.MultiIndex) or 'ticker' not in self.data.index.names):
                    raise ValueError(
                        "To prepare for global label transformation, 'ticker' must be a column or an index level.")

                if not isinstance(self.data.index, pd.MultiIndex):
                    original_date_index_name = self.data.index.name
                    if original_date_index_name is None:
                        original_date_index_name = 'TemporaryDateNameGlobal'
                        self.data.index.name = original_date_index_name

                    self.data = self.data.set_index('ticker', append=True)
                    self.data = self.data.reorder_levels(['ticker', original_date_index_name])

                self.data.index = self.data.index.set_names(['ticker', 'Dates'])
                print(
                    f"Standardized self.data index to MultiIndex: {self.data.index.names} for global label transform.")

            for TransformClass, params in self.label_transform_config:
                if not issubclass(TransformClass, LabelTransform):
                    raise ValueError(
                        "Invalid transform in label_transform_config: must subclass LabelTransform"
                    )
                print(f" Applying {TransformClass.__name__} with params {params} (globally)")
                transformer = TransformClass(df=self.data, root_path=self.root_path, **params)
                self.data = transformer.transform()

            self.data.index = self.data.index.set_names(['ticker', 'Dates'])
            self.data = self._ffill_covariates()
            self.data = self._fill_dividends()
            self.data = self.data.replace('#NAME?', np.nan)
            self.data = self.data.apply(pd.to_numeric, errors='ignore')

        return self.data

    def transform_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.label_per_partition:
            for TransformClass, params in self.label_transform_config:
                if not issubclass(TransformClass, LabelTransform):
                    raise ValueError(
                        "Invalid transform in label_transform_config: must subclass LabelTransform"
                    )
                print(f" Applying {TransformClass.__name__} with params {params} (via transform_labels method)")
                transformer = TransformClass(df=df, **params)
                df = transformer.transform()
        else:
            warnings.warn("transform_labels called, but label_per_partition is True. "
                          "Label transformations are handled within _partition_data for each fold.")
        return df

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
            raw_tickernames = [str(col) for col in ticker_df.iloc[0] if
                               pd.notna(col) and not str(col).lower().startswith("unnamed")]

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
        if covariates.index.name is None:
            covariates.index.name = "Dates"

        covariates = covariates[covariates.index.notna()]

        num_tickers = len(raw_tickernames)
        print(f"Found {num_tickers} raw ticker names.")

        col_dict: dict[int, list[str]] = defaultdict(list)
        for col in covariates.columns:
            number = self._get_number(col)
            if number < num_tickers:
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

    def _fill_dividends(self):
        """
        Bloomberg gives NaNs for non dividend paying companies,
        it should be zero instead
        """
        df_copy = self.data.copy()
        col = 'DVD_SH_12M'

        zero_div = (
            df_copy[col]
            .groupby(level='ticker', observed=False)
            .transform(lambda s: s.isna().all())
        )

        df_copy.loc[zero_div, col] = 0.0
        self.data = df_copy
        return df_copy


    def _ffill_covariates(self):
        df_copy = self.data.copy()
        df_copy = df_copy.sort_index(level='Dates')

        ffill_features = [
            "TOTAL_EQUITY", "BOOK_VAL_PER_SH", "REVENUE_PER_SH", "RETURN_COM_EQY",
            "TOT_DEBT_TO_TOT_ASSET", "TOT_DEBT_TO_TOT_EQY",
            "BS_TOT_LIAB2", "BS_TOT_ASSET", "IS_EPS"
        ]

        df_copy[ffill_features] = (
            df_copy.groupby(level='ticker', observed=False)[ffill_features]
            .ffill()
        )

        self.data = df_copy
        return df_copy


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

        if not (isinstance(self.data.index, pd.MultiIndex) and \
                list(self.data.index.names) == ['ticker', 'Dates']):
            if 'ticker' not in self.data.columns and (
                    not isinstance(self.data.index, pd.MultiIndex) or 'ticker' not in self.data.index.names):
                raise ValueError("To prepare for partitioning, 'ticker' must be a column or an index level.")
            if not isinstance(self.data.index, pd.MultiIndex):
                original_date_index_name = self.data.index.name if self.data.index.name is not None else 'TemporaryDateNamePartition'
                self.data.index.name = original_date_index_name
                self.data = self.data.set_index('ticker', append=True)
                self.data = self.data.reorder_levels(['ticker', original_date_index_name])
            self.data.index = self.data.index.set_names(['ticker', 'Dates'])

        self.data.index.set_names('TickerIndex', level='ticker', inplace=True)
        self.data.index.set_names('DateIndex', level='Dates', inplace=True)

        n_splits = self.partitioner.get_n_splits()
        fold_buckets = [
            {"train": [], "test": []} for _ in range(n_splits)
        ]

        data_grouped_by_ticker = self.data.groupby(level="TickerIndex", observed=False)
        print(
            f"Partitioning data using {self.partitioner.__class__.__name__} for {n_splits} splits across {len(data_grouped_by_ticker)} tickers.")

        for ticker, df_ticker in data_grouped_by_ticker:
            df_ticker = df_ticker.sort_index()

            min_samples_for_splitter = n_splits
            if isinstance(self.partitioner, PurgedTimeSeriesPartition) and self.partitioner.test_size is not None:
                min_samples_for_splitter = self.partitioner.test_size * n_splits + self.partitioner.gap * (n_splits - 1)
                if self.partitioner.max_train_size is not None:
                    min_samples_for_splitter = max(min_samples_for_splitter,
                                                   self.partitioner.test_size + self.partitioner.gap + 1)

            if len(df_ticker) < min_samples_for_splitter:
                warnings.warn(
                    f"Ticker {ticker} has only {len(df_ticker)} rows. "
                    f"This might be too short for {n_splits} splits with current settings. Skipping or may result in fewer splits."
                )

            groups = None
            if isinstance(self.partitioner, PurgedGroupTimeSeriesPartition):
                date_level_values = df_ticker.index.get_level_values('DateIndex')
                if not isinstance(date_level_values, pd.DatetimeIndex):
                    warnings.warn(
                        f"Ticker {ticker} DateIndex level is not DatetimeIndex. Cannot generate groups for PurgedGroupTimeSeriesPartition. Skipping ticker.")
                    continue
                groups = date_level_values.normalize()
                if len(groups) == 0:
                    warnings.warn(
                        f"Ticker {ticker} resulted in 0 groups (or group length mismatch). Skipping ticker for PurgedGroupTimeSeriesPartition.")
                    continue
                print(f" Using date groups for ticker {ticker} with PurgedGroupTimeSeriesPartition.")

            try:
                ticker_splits = list(self.partitioner.split(X=df_ticker, groups=groups))

                for fold_id, (train_idx, test_idx) in enumerate(ticker_splits):
                    if fold_id >= n_splits:
                        break

                    if train_idx.size == 0 or test_idx.size == 0:
                        warnings.warn(
                            f"Ticker {ticker}, fold {fold_id} resulted in empty train or test set. Skipping this part.")
                        continue

                    train_df_part = df_ticker.iloc[train_idx].copy()
                    test_df_part = df_ticker.iloc[test_idx].copy()

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

                if self.label_per_partition and self.label_transform_config:
                    print(f"Applying label transformations for Fold {fold_id} (train set)")
                    current_train_data = train_df_fold  # Has ['TickerIndex', 'DateIndex']
                    for TransformClass, params in self.label_transform_config:
                        if not issubclass(TransformClass, LabelTransform):
                            raise ValueError(
                                "Invalid transform in label_transform_config (must subclass LabelTransform)")
                        print(
                            f"  Applying {TransformClass.__name__} with params {params} to train data of Fold {fold_id}")
                        transformer = TransformClass(df=current_train_data, **params)
                        current_train_data = transformer.transform()
                    train_df_fold = current_train_data

                    print(f"Applying label transformations for Fold {fold_id} (test set)")
                    current_test_data = test_df_fold
                    for TransformClass, params in self.label_transform_config:
                        if not issubclass(TransformClass, LabelTransform):
                            raise ValueError(
                                "Invalid transform in label_transform_config (must subclass LabelTransform)")
                        print(
                            f"  Applying {TransformClass.__name__} with params {params} to test data of Fold {fold_id}")
                        transformer = TransformClass(df=current_test_data, **params)
                        current_test_data = transformer.transform()
                    test_df_fold = current_test_data

                cv_folds.append((train_df_fold, test_df_fold))
                print(f"Fold {fold_id}: Train shape {train_df_fold.shape}, Test shape {test_df_fold.shape}")
            else:
                warnings.warn(f"Fold {fold_id} will be skipped as it contained no data after processing all tickers.")

        if not cv_folds:
            warnings.warn(
                "Partitioning did not result in any valid CV folds. Check data length, ticker data, and partitioner settings.")

        print(f"Partitioning complete. Generated {len(cv_folds)} CV folds.")
        return cv_folds

    def get_cv_folds(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if self.data is None:
            print("Data not yet processed. Processing now...")
            self._process_data()
            if self.data is None or self.data.empty:
                raise ValueError("Data could not be loaded or is empty after processing, cannot generate folds.")

        return self._partition_data()
#%%
