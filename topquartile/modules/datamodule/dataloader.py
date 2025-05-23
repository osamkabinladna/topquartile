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

        # When label_per_partition=False, self.data.index.names are ['ticker', 'Dates'] here
        # due to reversion in transform_covariates.
        # If label_per_partition=True, self.data.index.names are whatever _load_data and covariate transforms set them to
        # (e.g., DatetimeIndex + 'ticker' column, or possibly already ['ticker', 'Dates'] from a covariate transform).
        # load_preds typically is called independently or before get_cv_folds.
        # The groupby('ticker') here assumes 'ticker' is a column or an index level named 'ticker'.

        # If self.data has MultiIndex ['ticker', 'Dates'], groupby('ticker') works on the level.
        # If self.data has 'ticker' column, it works on the column.
        # This part should be robust to these states.

        preds_group_key = 'ticker'
        if isinstance(self.data.index, pd.MultiIndex) and 'ticker' in self.data.index.names:
            pass  # groupby('ticker') will use the level name
        elif 'ticker' not in self.data.columns:
            # This could happen if index is ['TickerIndex', 'DateIndex'] and no 'ticker' column
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
            # This check might be too simple if MultiIndex is expected
            if not (isinstance(self.data.index, pd.MultiIndex) and isinstance(self.data.index.get_level_values(-1),
                                                                              pd.DatetimeIndex)):
                # If not a simple DatetimeIndex, and not a MultiIndex ending in DatetimeIndex, then convert.
                # This part may need more nuanced handling depending on expected index structures.
                # For now, we assume if it's MultiIndex, it's correctly typed.
                # If it was ['ticker','Dates'] or ['TickerIndex','DateIndex'], it's fine.
                pass  # self.data.index = pd.to_datetime(self.data.index) # This would fail for MultiIndex

        if not isinstance(self.preds.index, pd.DatetimeIndex):
            if not (isinstance(self.preds.index, pd.MultiIndex) and isinstance(self.preds.index.get_level_values(-1),
                                                                               pd.DatetimeIndex)):
                pass  # self.preds.index = pd.to_datetime(self.preds.index) # This would fail for MultiIndex

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

            # Step 1: Ensure self.data is a MultiIndex with names ['ticker', 'Dates'].
            # This standardizes self.data's index structure before specific renames.
            if not (isinstance(self.data.index, pd.MultiIndex) and \
                    list(self.data.index.names) == ['ticker', 'Dates']):

                # If not already the target MultiIndex ['ticker', 'Dates'], try to construct it.
                # This typically assumes self.data has a DatetimeIndex and a 'ticker' column
                # if it's not already a MultiIndex.
                if 'ticker' not in self.data.columns and \
                        (not isinstance(self.data.index, pd.MultiIndex) or 'ticker' not in self.data.index.names):
                    raise ValueError(
                        "To prepare for global label transformation, 'ticker' must be a column or an index level.")

                if not isinstance(self.data.index, pd.MultiIndex):
                    # Case: self.data has a DatetimeIndex (e.g., index_col=0 from CSV) and 'ticker' is a column.
                    original_date_index_name = self.data.index.name
                    # Ensure the date index level has a name before using it in reorder_levels
                    if original_date_index_name is None:
                        original_date_index_name = 'TemporaryDateNameGlobal'  # Assign a temporary name
                        self.data.index.name = original_date_index_name

                    self.data = self.data.set_index('ticker', append=True)
                    # Index levels are now (original_date_index_name, 'ticker')
                    self.data = self.data.reorder_levels(['ticker', original_date_index_name])
                    # Index levels are now ('ticker', original_date_index_name)

                # At this point, self.data is guaranteed to be a MultiIndex.
                # Ensure its level names are exactly ['ticker', 'Dates'].
                self.data.index = self.data.index.set_names(['ticker', 'Dates'])
                print(
                    f"Standardized self.data index to MultiIndex: {self.data.index.names} for global label transform.")

            # Step 2: Temporarily rename from ['ticker', 'Dates'] to ['TickerIndex', 'DateIndex']
            # This is what the LabelTransform is expecting.
            self.data.index = self.data.index.set_names(['TickerIndex', 'DateIndex'])
            # print(f"DEBUG: Temporarily renamed index for global label transform to {list(self.data.index.names)}")

            for TransformClass, params in self.label_transform_config:
                if not issubclass(TransformClass, LabelTransform):
                    raise ValueError(
                        "Invalid transform in label_transform_config: must subclass LabelTransform"
                    )
                print(f" Applying {TransformClass.__name__} with params {params} (globally)")
                transformer = TransformClass(df=self.data, **params)  # df has ['TickerIndex', 'DateIndex']
                self.data = transformer.transform()

            # Step 3: Revert index names to ['ticker', 'Dates'].
            # This is crucial for the "don't touch" renaming lines in _partition_data
            # which expect to find levels named 'ticker' and 'Dates'.
            self.data.index = self.data.index.set_names(['ticker', 'Dates'])
            # print(f"DEBUG: Reverted index names for _partition_data to {list(self.data.index.names)}")

        return self.data

    def transform_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.label_per_partition:
            for TransformClass, params in self.label_transform_config:
                if not issubclass(TransformClass, LabelTransform):
                    raise ValueError(
                        "Invalid transform in label_transform_config: must subclass LabelTransform"
                    )
                print(f" Applying {TransformClass.__name__} with params {params} (via transform_labels method)")
                # This external call would also need the rename-transform-revert logic
                # if df does not have ['TickerIndex', 'DateIndex'] and transform expects it.
                # For simplicity, assuming this method is called with df in the expected state by the transform.
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
        # Ensure index has a default name if read_csv(index_col=0) doesn't provide one from header
        if covariates.index.name is None:
            covariates.index.name = "Dates"  # Default name after loading

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
            self.data.sort_index(inplace=True)  # Sorts by current index (DatetimeIndex, e.g. 'Dates')
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

        # The following lines are the "don't touch" part for renaming.
        # They expect self.data to have a MultiIndex with levels named 'ticker' and 'Dates'.
        # This is ensured by the revert step in transform_covariates if global labels were applied,
        # or by covariate transforms, or by the standardization step in transform_covariates.
        # If self.data comes straight from _load_data (no global labels, no prior multiindex conversion by covariate transforms),
        # it needs to be converted to MultiIndex ['ticker', 'Dates'] first.
        if not (isinstance(self.data.index, pd.MultiIndex) and \
                list(self.data.index.names) == ['ticker', 'Dates']):
            # This block ensures data is MultiIndex ['ticker', 'Dates'] before the critical set_names
            if 'ticker' not in self.data.columns and (
                    not isinstance(self.data.index, pd.MultiIndex) or 'ticker' not in self.data.index.names):
                raise ValueError("To prepare for partitioning, 'ticker' must be a column or an index level.")
            if not isinstance(self.data.index, pd.MultiIndex):
                original_date_index_name = self.data.index.name if self.data.index.name is not None else 'TemporaryDateNamePartition'
                self.data.index.name = original_date_index_name
                self.data = self.data.set_index('ticker', append=True)
                self.data = self.data.reorder_levels(['ticker', original_date_index_name])
            self.data.index = self.data.index.set_names(['ticker', 'Dates'])
            # print(f"DEBUG: Standardized index in _partition_data to {list(self.data.index.names)}")

        self.data.index.set_names('TickerIndex', level='ticker', inplace=True)
        self.data.index.set_names('DateIndex', level='Dates', inplace=True)
        # Now self.data.index.names are ['TickerIndex', 'DateIndex']

        n_splits = self.partitioner.get_n_splits()
        fold_buckets = [
            {"train": [], "test": []} for _ in range(n_splits)
        ]

        # Group by the level name 'TickerIndex'
        data_grouped_by_ticker = self.data.groupby(level="TickerIndex")
        print(
            f"Partitioning data using {self.partitioner.__class__.__name__} for {n_splits} splits across {len(data_grouped_by_ticker)} tickers.")

        for ticker, df_ticker in data_grouped_by_ticker:
            df_ticker = df_ticker.sort_index()  # df_ticker inherits TickerIndex, DateIndex

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
                # df_ticker.index is already a MultiIndex here. We need the 'DateIndex' level for groups.
                date_level_values = df_ticker.index.get_level_values('DateIndex')
                if not isinstance(date_level_values, pd.DatetimeIndex):
                    warnings.warn(
                        f"Ticker {ticker} DateIndex level is not DatetimeIndex. Cannot generate groups for PurgedGroupTimeSeriesPartition. Skipping ticker.")
                    continue
                groups = date_level_values.normalize()
                if len(groups) == 0:  # Should be len(groups) != len(df_ticker) or specific checks
                    warnings.warn(
                        f"Ticker {ticker} resulted in 0 groups (or group length mismatch). Skipping ticker for PurgedGroupTimeSeriesPartition.")
                    continue
                print(f" Using date groups for ticker {ticker} with PurgedGroupTimeSeriesPartition.")

            try:
                # X for splitter should be based on values, groups on index
                ticker_splits = list(self.partitioner.split(X=df_ticker, groups=groups))

                if not ticker_splits:
                    warnings.warn(
                        f"Ticker {ticker} produced no splits. This might be due to short series or partitioner settings. Skipping this ticker.")
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
                        warnings.warn(
                            f"Ticker {ticker}, fold {fold_id} resulted in empty train or test set. Skipping this part.")
                        continue

                    train_df_part = df_ticker.iloc[train_idx].copy()  # train_df_part has ['TickerIndex', 'DateIndex']
                    test_df_part = df_ticker.iloc[test_idx].copy()  # test_df_part has ['TickerIndex', 'DateIndex']

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
                train_df_fold = pd.concat(bucket["train"]).sort_index()  # Retains ['TickerIndex', 'DateIndex']
                test_df_fold = pd.concat(bucket["test"]).sort_index()  # Retains ['TickerIndex', 'DateIndex']

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
                    current_test_data = test_df_fold  # Has ['TickerIndex', 'DateIndex']
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