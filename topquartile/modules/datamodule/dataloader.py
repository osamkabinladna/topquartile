import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Type, Generator
import re
from collections import defaultdict
import warnings


class CovariateTransform:
    def __init__(self, df, **kwargs):
        self.df = df
        self.required_base = set()

    def transform(self):
        return self.df


class LabelTransform:  # Placeholder base class
    def __init__(self, df, **kwargs):
        self.df = df

    def transform(self):
        return self.df


# --- Start: PurgedGroupTimeSeriesSplit Class ---
# (Included directly here for completeness, you might want to import it)


# --- End: PurgedGroupTimeSeriesSplit Class ---


class DataLoader:
    def __init__(self, data_id: str,
                 covariate_transform: Optional[List[Tuple[Type[CovariateTransform], Dict]]] = None,
                 label_transform: Optional[List[Tuple[Type[LabelTransform], Dict]]] = None,
                 cols2drop: Optional[List[str]] = 'NEWS_SENTIMENT_DAILY_AVG',
                 prediction_length: int = 20,
                 # --- CV Parameters ---
                 n_splits: int = 5,
                 cv_group_gap: int = 0,  # Gap between train/test in terms of groups (e.g., days)
                 cv_max_train_group_size: int = np.inf,
                 cv_max_test_group_size: int = np.inf,
                 cv_verbose: bool = False
                 ):
        self.data_id = data_id
        self.covariate_transform = covariate_transform

        self.covariate_transform_config = covariate_transform if covariate_transform else []
        self.label_transform_config = label_transform if label_transform else []

        self.data = None
        self.labels = None  # Consider defining how labels are created/used
        self.preds = None  # Stores data held out for final prediction
        self.required_covariates = set()

        # Ensure cols2drop is a list
        if isinstance(cols2drop, str):
            self.cols2drop = [cols2drop]
        elif cols2drop is None:
            self.cols2drop = []
        else:
            self.cols2drop = cols2drop

        self.prediction_length = prediction_length

        # --- CV Parameters ---
        self.n_splits = n_splits
        self.cv_group_gap = cv_group_gap
        self.cv_max_train_group_size = cv_max_train_group_size
        self.cv_max_test_group_size = cv_max_test_group_size
        self.cv_verbose = cv_verbose

        # --- File Path ---
        # Ensure Path(__file__) works correctly depending on execution context
        try:
            # Assumes the script is run from a location where this relative path makes sense
            script_dir = Path(__file__).resolve().parent
            root_path = script_dir.parent.parent  # Adjust based on your actual project structure
            self.covariates_path = root_path / 'data' / f'{self.data_id}.csv'
        except NameError:
            # Fallback if __file__ is not defined (e.g., interactive session)
            warnings.warn(
                "Could not determine script path using __file__. Using current working directory for data path.")
            root_path = Path.cwd()  # Or provide an explicit path
            self.covariates_path = root_path / 'data' / f'{self.data_id}.csv'  # Adjust as needed

        if not self.covariates_path.is_file():
            warnings.warn(f"Data file not found at expected path: {self.covariates_path}")

    def _transform_data(self):
        if self.data is None:
            self._load_data()
            if self.data is None or self.data.empty:
                raise RuntimeError(
                    "Data loading failed or resulted in empty DataFrame. Cannot proceed with transformations.")

        print("Applying covariate transforms...")
        for TransformClass, params in self.covariate_transform_config:
            if not issubclass(TransformClass, CovariateTransform):
                warnings.warn(
                    f"Warning: Invalid transform type in config: {TransformClass}. Must be a subclass of CovariateTransform. Skipping.")
                continue
            try:
                print(f" - Applying {TransformClass.__name__} with params {params}")
                transformer_instance = TransformClass(df=self.data.copy(), **params)  # Operate on copy
                self.data = transformer_instance.transform()
                self.required_covariates.update(
                    getattr(transformer_instance, 'required_base', set()))  # Safely get required base
                print(f"   Data shape after transform: {self.data.shape}")


            except Exception as e:
                raise ValueError(f"Error applying {TransformClass.__name__}: {e}")

        print("Applying label transforms...")
        for TransformClass, params in self.label_transform_config:
            if not issubclass(TransformClass, LabelTransform):
                warnings.warn(
                    f"Warning: Invalid transform type in config: {TransformClass}. Must be a subclass of LabelTransform. Skipping.")
                continue
            try:
                print(f" - Applying {TransformClass.__name__} with params {params}")
                transformer_instance = TransformClass(df=self.data.copy(), **params)  # Operate on copy
                self.data = transformer_instance.transform()
                print(f"   Data shape after transform: {self.data.shape}")
            except Exception as e:
                raise ValueError(f"Error applying {TransformClass.__name__}: {e}")

        # Assign labels if a 'label' column was created by transforms
        if 'label' in self.data.columns:
            self.labels = self.data['label']

        print("Data transformation complete.")
        return self.data

    def _process_data(self):
        """Loads, transforms, and imputes data."""
        print("Processing data...")
        if self.data is None:
            self._load_data()
            if self.data is None or self.data.empty:
                raise RuntimeError("Data loading failed or resulted in empty DataFrame.")

        self._transform_data()  # Apply defined transformations
        # self._impute_columns() # Call imputation after transforms might be better

        # Drop specified columns *after* transforms, in case transforms create/rely on them
        print(f"Dropping specified columns: {self.cols2drop}")
        cols_actually_dropped = [col for col in self.cols2drop if col in self.data.columns]
        if len(cols_actually_dropped) < len(self.cols2drop):
            warnings.warn(
                f"Not all columns in cols2drop were found. Found and dropped: {cols_actually_dropped}. Missing: {list(set(self.cols2drop) - set(cols_actually_dropped))}")
        self.data.drop(columns=cols_actually_dropped, inplace=True, errors='ignore')

        # Crucial for time series splitting: ensure data is sorted by index (time)
        print("Sorting data by time index...")
        self.data.sort_index(inplace=True)

        print(f"Data processing complete. Final data shape: {self.data.shape}")
        return self.data

    def _load_data(self) -> pd.DataFrame:
        """Loads data from the CSV, handling the specific header format."""
        print(f"Loading data from: {self.covariates_path}")
        if not self.covariates_path.is_file():
            raise FileNotFoundError(f"Data file not found: {self.covariates_path}")

        try:
            ticker_df = pd.read_csv(self.covariates_path,
                                    skiprows=3, nrows=1, header=None, low_memory=False)  # Read just the ticker row
            raw_tickernames = ticker_df.iloc[0].dropna().tolist()
            # Clean ticker names (remove ':1', etc.) and handle potential duplicates cleanly
            tickernames = []
            seen_tickers = set()
            ticker_map = {}  # Maps raw name (like 'IMJS IJ EQUITY:1') to clean name ('IMJS')
            valid_indices = []  # Store column indices corresponding to unique tickers
            current_index = 0  # CSV column index, assuming first column is date
            for i, raw_name in enumerate(raw_tickernames):
                # Skip empty strings or potential placeholder columns
                if not isinstance(raw_name, str) or not raw_name.strip():
                    current_index += 1  # Assuming unnamed cols might exist here
                    continue

                # Heuristic: find the position of the number suffix like '.1'
                num_suffix_match = re.search(r'\.(\d+)$', raw_name)
                if num_suffix_match:
                    base_name = raw_name[:num_suffix_match.start()]
                    col_num_for_dict = int(num_suffix_match.group(1))
                else:
                    # Assume it's the first instance (like '.0')
                    base_name = raw_name
                    col_num_for_dict = 0

                # Another common pattern: 'TICKER XYZ EQUITY:1'
                equity_suffix_match = re.match(r'^(.+?)\s*:?\d+$', base_name)  # Non-greedy match for ticker name
                if equity_suffix_match:
                    clean_ticker = equity_suffix_match.group(1).strip()
                else:
                    clean_ticker = base_name.strip()  # Use base name if no equity suffix

                # Use first occurrence for unique ticker list
                if clean_ticker not in seen_tickers:
                    seen_tickers.add(clean_ticker)
                    tickernames.append(clean_ticker)
                    valid_indices.append(col_num_for_dict)  # Store the *original* column number (.0, .1 etc)
                    ticker_map[raw_name] = clean_ticker  # Map raw name to clean name

            print(f"Found {len(tickernames)} unique tickers.")

            # Read the main data, skipping header rows, use index_col=0 for dates
            covariates = pd.read_csv(self.covariates_path, skiprows=5, index_col=0, low_memory=False)
            covariates.dropna(inplace=True, axis=0, how='all')  # Drop rows with all NaN
            # Drop columns with all NaN *before* processing might be safer
            covariates.dropna(inplace=True, axis=1, how='all')

            covariates.index = pd.to_datetime(covariates.index, format='mixed', errors='coerce')
            covariates.dropna(axis=0, subset=[covariates.index.name],
                              inplace=True)  # Drop rows where date parsing failed

            # Reconstruct dataframes per ticker using the cleaned names and original column numbers
            all_ticker_dfs = []
            processed_cols = set()  # Keep track of columns already added

            for i, clean_ticker in enumerate(tickernames):
                # Find all original columns that map to this clean ticker
                original_col_nums = []
                base_metric_names = set()
                col_mapping_for_ticker = {}  # Maps base metric name to full column name (e.g., PX_LAST to PX_LAST.1)

                for col in covariates.columns:
                    num_suffix_match = re.search(r'\.(\d+)$', col)
                    if num_suffix_match:
                        base_metric = col[:num_suffix_match.start()]
                        col_num = int(num_suffix_match.group(1))
                    else:
                        # Assume .0 if no suffix
                        base_metric = col
                        col_num = 0

                    # Check if this column number corresponds to the current unique ticker index
                    # This logic assumes columns are grouped like PX_LAST.0, VOL.0, PX_LAST.1, VOL.1 ...
                    # We need a robust way to link column numbers (0, 1, 2...) to tickers.
                    # Let's refine the column grouping based on the suffix number.

                # Alternative: Group columns by suffix first
                col_dict = defaultdict(list)
                for col in covariates.columns:
                    number = self._get_number(col)
                    col_dict[number].append(col)

                # Assuming valid_indices correctly holds the column group numbers for unique tickers
                ticker_col_group_index = valid_indices[i]
                cols_for_this_ticker = col_dict.get(ticker_col_group_index, [])

                if not cols_for_this_ticker:
                    warnings.warn(
                        f"No columns found for ticker {clean_ticker} with expected group index {ticker_col_group_index}. Skipping.")
                    continue

                ticker_df = covariates[cols_for_this_ticker].copy()

                # Rename columns to remove the suffix (e.g., 'PX_LAST.1' -> 'PX_LAST')
                rename_map = {col: self._get_base_name(col) for col in ticker_df.columns}
                ticker_df.rename(columns=rename_map, inplace=True)

                # Add ticker identifier
                ticker_df['ticker'] = clean_ticker
                all_ticker_dfs.append(ticker_df)

            if not all_ticker_dfs:
                raise ValueError("Could not extract data for any tickers. Check CSV format and ticker parsing logic.")

            self.data = pd.concat(all_ticker_dfs)
            self.tickernames = tickernames  # Store the unique ticker names found

            print(f"Data loaded. Shape: {self.data.shape}")
            # Ensure index is sorted after concatenation
            self.data.sort_index(inplace=True)

        except FileNotFoundError:
            raise
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            import traceback
            traceback.print_exc()  # Print stack trace for debugging
            self.data = pd.DataFrame()  # Return empty dataframe on error
            self.tickernames = []

        return self.data

    def _get_number(self, col_name):
        """Extracts the numerical suffix (e.g., .0, .1) from a column name."""
        match = re.search(r'\.(\d+)$', col_name)  # Search for suffix at the end
        if match:
            return int(match.group(1))
        else:
            return 0  # Default to 0 if no suffix

    def _get_base_name(self, col_name):
        """Removes the numerical suffix from a column name."""
        match = re.search(r'\.(\d+)$', col_name)
        if match:
            return col_name[:match.start()]
        else:
            return col_name

    def _impute_columns(self):
        """
        Imputes columns inplace. (Placeholder - Needs specific imputation strategy)
        """
        warnings.warn(
            "'_impute_columns' method is called but has no specific imputation logic implemented. Skipping imputation.")
        # Example: Forward fill
        # print("Applying forward fill imputation...")
        # self.data.fillna(method='ffill', inplace=True)
        # Example: Drop columns with high missing %
        # missing_pct = self.data.isna().mean()
        # cols_to_drop_impute = missing_pct[missing_pct > 0.5].index # Drop cols with > 50% missing
        # if not cols_to_drop_impute.empty:
        #    print(f"Dropping columns due to high missing values (>50%): {cols_to_drop_impute.tolist()}")
        #    self.data.drop(columns=cols_to_drop_impute, inplace=True)

    def load_preds(self, group_by_col='ticker'):
        """Separates the final `prediction_length` periods for prediction/evaluation."""
        if self.data is None or self.data.empty:
            print("Attempting to process data before loading predictions...")
            self._process_data()
            if self.data is None or self.data.empty:
                raise ValueError("Data is not loaded or is empty. Cannot separate prediction data.")

        if group_by_col not in self.data.columns:
            raise ValueError(f"Grouping column '{group_by_col}' not found in data columns: {self.data.columns}")

        print(f"Separating last {self.prediction_length} periods per '{group_by_col}' for prediction set...")

        # Ensure data is sorted for tail to work correctly per group
        self.data.sort_index(inplace=True)

        # Use groupby().tail() to get the last N rows for each group
        self.preds = self.data.groupby(group_by_col, group_keys=False).tail(self.prediction_length)

        # Remove these prediction rows from the main data
        remaining_index = self.data.index.difference(self.preds.index)
        self.data = self.data.loc[remaining_index]

        print(f"Training/CV data shape: {self.data.shape}")
        print(f"Prediction data shape: {self.preds.shape}")

        if self.data.empty:
            warnings.warn(
                "The main data is empty after separating the prediction set. Check prediction_length vs data size.")
        if self.preds.empty:
            warnings.warn("The prediction set is empty. Check prediction_length.")

        return self.preds

    # --- Implementation of PurgedGroupTimeSeriesSplit ---
    def _partition_data(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Partitions the data using PurgedGroupTimeSeriesSplit.

        Yields:
            Generator[Tuple[np.ndarray, np.ndarray], None, None]:
            A generator yielding tuples of (train_indices, test_indices) for each split.
            Indices refer to the integer position (.iloc) of rows in the *current* self.data DataFrame
            (after processing and potentially after load_preds separation).
        """
        if self.data is None or self.data.empty:
            print("Data not processed. Running _process_data() first.")
            self._process_data()  # Ensure data is loaded and processed
            if self.data is None or self.data.empty:
                raise ValueError("Data is still empty after processing. Cannot partition.")

        if self.data.index.nlevels > 1:
            warnings.warn(
                "Data index seems to be a MultiIndex. Grouping for CV will use the primary level (level 0). Ensure this is the desired time dimension.")
            time_index = self.data.index.get_level_values(0)
        else:
            time_index = self.data.index

        # --- Define Groups ---
        # We need a consistent group identifier for each *time period* across all tickers.
        # Using the date part of the index is a common strategy.
        # Ensure the index is DatetimeIndex
        if not isinstance(time_index, pd.DatetimeIndex):
            try:
                print("Converting index to DatetimeIndex for grouping...")
                # Attempt conversion, assuming it's convertible
                time_index = pd.to_datetime(time_index, errors='coerce')
                # Drop rows where conversion failed (NaT)
                nat_rows = time_index.isna()
                if nat_rows.any():
                    warnings.warn(f"Dropping {nat_rows.sum()} rows where index could not be converted to datetime.")
                    self.data = self.data[~nat_rows]
                    time_index = self.data.index  # Re-assign after dropping
                if not isinstance(self.data.index, pd.DatetimeIndex):
                    # If the main index wasn't updated, try updating it directly
                    self.data.index = time_index

            except Exception as e:
                raise TypeError(
                    f"Data index must be a DatetimeIndex or convertible to one for time series grouping. Error: {e}")

        # Group by day. You could change 'D' to 'W' (week), 'M' (month), etc.
        # Using integer representation of period for sortable groups
        groups = time_index.to_period('D').astype(int)
        print(f"Grouping data by day for PurgedGroupTimeSeriesSplit. Found {len(np.unique(groups))} unique days.")

        # --- Instantiate the Splitter ---
        cv_splitter = PurgedGroupTimeSeriesSplit(
            n_splits=self.n_splits,
            group_gap=self.cv_group_gap,
            max_train_group_size=self.cv_max_train_group_size,
            max_test_group_size=self.cv_max_test_group_size,
            verbose=self.cv_verbose
        )

        print(f"Generating {self.n_splits} splits with group_gap={self.cv_group_gap}...")
        # --- Generate and Yield Splits ---
        # The split method requires X and groups. We only need X for shape/indexing.
        # It yields integer indices corresponding to the rows of the input data (self.data).
        split_count = 0
        for train_idx, test_idx in cv_splitter.split(X=self.data, groups=groups):
            if len(train_idx) > 0 and len(test_idx) > 0:
                print(f"  Split {split_count + 1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
                yield train_idx, test_idx
                split_count += 1
            else:
                warnings.warn(
                    f"Split {split_count + 1} resulted in empty train ({len(train_idx)}) or test ({len(test_idx)}) set. Skipping this split.")
                continue  # Skip this iteration if a set is empty

        if split_count == 0:
            warnings.warn(
                "PurgedGroupTimeSeriesSplit generated no valid splits. Check data size, n_splits, group_gap, and group sizes.")
        elif split_count < self.n_splits:
            warnings.warn(
                f"PurgedGroupTimeSeriesSplit generated only {split_count} valid splits, less than the requested n_splits={self.n_splits}.")

    # Example of how to use the partition method
    def get_cv_splits(self):
        """
        Calls _partition_data and returns a list of (train_df, test_df) tuples.
        Note: This loads the full dataframes for each split into memory.
        For large datasets, using the generator directly might be better.
        """
        if self.data is None:
            self._process_data()  # Ensure data is ready

        splits = []
        for train_idx, test_idx in self._partition_data():
            # Use .iloc for integer-based indexing
            train_df = self.data.iloc[train_idx].copy()
            test_df = self.data.iloc[test_idx].copy()
            splits.append((train_df, test_df))
        return splits