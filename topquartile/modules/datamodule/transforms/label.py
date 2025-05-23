import pandas as pd
import yfinance as yf  # Added import for yfinance
from abc import ABC, abstractmethod  # Added imports for ABC


class LabelTransform(ABC):
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.df = df

    @abstractmethod
    def transform(self) -> pd.DataFrame:
        """Transforms the DataFrame to add label(s)."""
        raise NotImplementedError


class ExcessReturnTransform(LabelTransform):
    def __init__(self, df: pd.DataFrame, label_duration: int,
                 index_ticker: str = "^JKSE", price_column: str = 'PX_LAST',
                 ticker_level_name: str = 'TickerIndex',
                 date_level_name: str = 'DateIndex'):
        """
        :param df: dataframe to be transformed
        :param label_duration: asset holding period for return calculation
        :param index_ticker: ticker of market index
        :param price_column: column name of price
        :param ticker_level_name: Multiindex name for ticker
        :param date_level_name: Multiindex name for date
        """
        super().__init__(df)
        self.label_duration = label_duration
        self.index_ticker = index_ticker
        self.price_column = price_column
        self.ticker_level_name = ticker_level_name
        self.date_level_name = date_level_name

        self.stock_return_col_name = f'{self.label_duration}d_stock_return'
        self.index_return_col_name = 'INDEX_RETURN'
        self.excess_return_col_name = 'EXCESS_RETURN'

    def _calculate_returns(self, series: pd.Series) -> pd.Series:
        """Calculates percentage returns over a future period."""
        future_price = series.shift(-self.label_duration)
        returns = ((future_price - series) / series) * 100
        return returns

    def _get_index_returns(self) -> pd.Series:
        """Downloads index data and calculates its future returns."""
        if self.date_level_name not in self.df.index.names:
            raise ValueError(f"Date level '{self.date_level_name}' not found in DataFrame index.")

        unique_dates = self.df.index.get_level_values(self.date_level_name).unique()

        if not pd.api.types.is_datetime64_any_dtype(unique_dates.dtype):
            try:
                unique_dates = pd.to_datetime(unique_dates)
            except Exception as e:
                raise ValueError(
                    f"Could not convert unique dates from level '{self.date_level_name}' "
                    f"for yfinance download. Check data. Error: {e}"
                ) from e

        unique_dates = unique_dates.sort_values()
        if unique_dates.empty:
            raise ValueError(f"No dates found in the DataFrame's '{self.date_level_name}' index level.")

        start_date = unique_dates.min()
        required_end_date = unique_dates.max()

        try:
            index_data = yf.download(self.index_ticker, start=start_date, end=required_end_date + pd.Timedelta(days=1),
                                     progress=False, auto_adjust=False)
        except Exception as e:
            raise ConnectionError(f"Failed to download index data for {self.index_ticker}: {e}") from e

        if index_data.empty:
            raise ValueError(
                f"No data downloaded for index {self.index_ticker} for the date range {start_date} to {required_end_date}.")

        index_data.index = pd.to_datetime(index_data.index)
        price_col_yf = 'Close'

        if price_col_yf not in index_data.columns:
            raise KeyError(
                f"'{price_col_yf}' column not found in downloaded index data. Available columns: {index_data.columns}")

        price_data_selection = index_data[price_col_yf]

        if isinstance(price_data_selection, pd.DataFrame):
            if price_data_selection.shape[1] == 1:
                price_series = price_data_selection.iloc[:, 0]
            else:
                raise TypeError(
                    f"Selection of '{price_col_yf}' yielded DataFrame with multiple columns: {price_data_selection.columns}. Cannot proceed.")
        elif isinstance(price_data_selection, pd.Series):
            price_series = price_data_selection
        else:
            raise TypeError(f"Selection of '{price_col_yf}' yielded unexpected type: {type(price_data_selection)}")

        index_returns = self._calculate_returns(price_series)

        if not isinstance(index_returns, pd.Series):
            raise TypeError(
                f"_calculate_returns function unexpectedly returned a {type(index_returns)}, expected Series.")

        index_returns.name = self.index_return_col_name

        aligned_index_returns = index_returns.reindex(unique_dates)

        nan_count = aligned_index_returns.isnull().sum()
        if pd.api.types.is_scalar(nan_count):
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN values found in index returns after aligning to DataFrame dates. ")
        else:
            raise TypeError(f"Calculation of NaN count failed. Expected scalar, got {type(nan_count)}. ")
        return aligned_index_returns


    def transform(self) -> pd.DataFrame:
        df_copy = self.df.copy()

        is_multiindex = isinstance(df_copy.index, pd.MultiIndex)
        ticker_in_index = is_multiindex and self.ticker_level_name in df_copy.index.names
        date_in_index = is_multiindex and self.date_level_name in df_copy.index.names

        ticker_in_cols = self.ticker_level_name in df_copy.columns
        current_date_index_name = df_copy.index.name if not is_multiindex else None

        if not (ticker_in_index and date_in_index):
            if ticker_in_cols and (current_date_index_name == self.date_level_name or date_in_index):
                if not is_multiindex and current_date_index_name == self.date_level_name:
                    df_copy = df_copy.set_index([self.ticker_level_name], append=True)
                elif is_multiindex and date_in_index and not ticker_in_index:
                    df_copy = df_copy.set_index(self.ticker_level_name, append=True)
                else:
                    df_copy = df_copy.reset_index().set_index([self.ticker_level_name, self.date_level_name])

                if df_copy.index.names != [self.ticker_level_name, self.date_level_name]:
                    try:
                        df_copy = df_copy.reorder_levels([self.ticker_level_name, self.date_level_name])
                    except KeyError as e:
                        raise ValueError(
                            f"Failed to reorder levels to [{self.ticker_level_name}, {self.date_level_name}]. Current levels: {df_copy.index.names}. Error: {e}")

            elif not (ticker_in_index and date_in_index):
                raise ValueError(
                    f"ExcessReturnTransform expects '{self.ticker_level_name}' and '{self.date_level_name}' "
                    f"to be in the index or '{self.ticker_level_name}' in columns and '{self.date_level_name}' as index name. "
                    f"Current index: {df_copy.index.names if is_multiindex else [df_copy.index.name]}, Columns: {df_copy.columns.tolist()}"
                )

        df_copy = df_copy.sort_index(level=[self.ticker_level_name, self.date_level_name])

        date_level_values = df_copy.index.get_level_values(self.date_level_name)
        if not pd.api.types.is_datetime64_any_dtype(date_level_values.dtype):
            try:
                current_levels_arrays = [df_copy.index.get_level_values(name) for name in df_copy.index.names]
                date_level_idx_for_array = df_copy.index.names.index(self.date_level_name)

                current_levels_arrays[date_level_idx_for_array] = pd.to_datetime(date_level_values)

                df_copy.index = pd.MultiIndex.from_arrays(current_levels_arrays, names=df_copy.index.names)
            except Exception as e:
                raise ValueError(
                    f"Could not convert the '{self.date_level_name}' index level to datetime. "
                    f"Please check the data. Error: {e}"
                ) from e

        df_copy[self.stock_return_col_name] = (
            df_copy.groupby(level=self.ticker_level_name, group_keys=False)[self.price_column]
            .apply(self._calculate_returns)
        )

        index_returns_series = self._get_index_returns()

        df_copy = df_copy.join(index_returns_series, on=self.date_level_name)
        df_copy[self.excess_return_col_name] = df_copy[self.stock_return_col_name] - df_copy[self.index_return_col_name]

        return df_copy

class BinaryLabelTransform(ExcessReturnTransform):
    def __init__(self, df: pd.DataFrame, label_duration: int, quantile: float,
                 index_ticker: str = "^JKSE", price_column: str = 'PX_LAST',
                 ticker_level_name: str = 'TickerIndex',
                 date_level_name: str = 'DateIndex'):
        """
        :param df: dataframe to be transformed
        :param label_duration: asset holding period
        :param quantile: quantile of excess returns to define the positive class
        :param index_ticker: ticker of market index
        :param price_column: column name of price
        :param ticker_level_name: Multiindex name for ticker
        :param date_level_name: Multiindex name for date
        """
        super().__init__(df, label_duration, index_ticker, price_column,
                         ticker_level_name, date_level_name)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("Quantile must be between 0.0 and 1.0.")
        self.quantile = quantile
        self.label_col_name = 'label'

    def _assign_label(self, group: pd.DataFrame) -> pd.Series:
        """
        TODO: Current implementation calculates quantiles over all returns before partitions, maybe we want to
        create a flag to calculate it per partition?

        Assigns a binary label based on whether the excess return is in the top quantile.
        Operates on a group (typically grouped by date).
        """
        valid_returns = group[self.excess_return_col_name].dropna()

        if valid_returns.empty:
            return pd.Series(pd.NA, index=group.index, dtype='Int64')

        quantile_threshold = valid_returns.quantile(self.quantile)

        if pd.isna(quantile_threshold):
            return pd.Series(pd.NA, index=group.index, dtype='Int64')

        label = (group[self.excess_return_col_name] >= quantile_threshold)
        label = label.astype(float).where(group[self.excess_return_col_name].notna()).astype('Int64')
        return label

    def transform(self) -> pd.DataFrame:
        """
        Calculates excess returns using the parent class's transform.
        Then, adds a binary label based on the quantile of these excess returns.
        """
        df_with_excess_returns = super().transform()

        if isinstance(df_with_excess_returns.index, pd.MultiIndex) and \
                self.date_level_name in df_with_excess_returns.index.names:
            df_with_excess_returns[self.label_col_name] = (
                df_with_excess_returns.groupby(level=self.date_level_name, group_keys=False)
                .apply(self._assign_label)
            )
        else:
            print(f"Warning: DataFrame is not MultiIndexed by '{self.date_level_name}'. ")
            df_with_excess_returns[self.label_col_name] = self._assign_label(df_with_excess_returns)

        return df_with_excess_returns


class NaryLabelTransform(ExcessReturnTransform):
    """
    Transforms asset returns into N-ary labels based on quantiles of excess returns.
    Label 1 represents the highest quantile of excess returns.
    """

    def __init__(self, df: pd.DataFrame, label_duration: int, n_labels: int,
                 index_ticker: str = "^JKSE", price_column: str = 'PX_LAST',
                 ticker_level_name: str = 'ticker',
                 date_level_name: str = 'Dates'):
        """
        Initializes the NaryLabelTransform.

        :param df: DataFrame to be transformed.
        :param label_duration: Asset holding period for return calculation.
        :param n_labels: The number of distinct labels (groups) to create.
        :param index_ticker: Ticker of the market index.
        :param price_column: Column name of the price data.
        :param ticker_level_name: MultiIndex name for the ticker level.
        :param date_level_name: MultiIndex name for the date level.
        """
        super().__init__(df, label_duration, index_ticker, price_column,
                         ticker_level_name, date_level_name)
        if not isinstance(n_labels, int) or n_labels < 1:
            raise ValueError("n_labels must be a positive integer (>= 1).")
        self.n_labels = n_labels
        self.label_col_name = f'n-ary-label'

    def _assign_label(self, group: pd.DataFrame) -> pd.Series:
        """
        Assigns an N-ary label based on quantiles of excess returns within a group.
        Operates on a group (typically grouped by date).
        Label 1 is for the highest returns.

        :param group: A DataFrame group (e.g., data for a single date).
        :return: A Series containing the N-ary labels for the group.
        """
        labels_series = pd.Series(pd.NA, index=group.index, dtype='Int64')

        valid_returns = group[self.excess_return_col_name].dropna()

        if valid_returns.empty:
            return labels_series

        try:
            qcut_labels_raw = pd.qcut(valid_returns, q=self.n_labels, labels=False, duplicates='drop')

            if len(qcut_labels_raw) == 0:
                return labels_series

            num_actual_bins = qcut_labels_raw.max() + 1
            final_labels = num_actual_bins - qcut_labels_raw

            labels_series.loc[valid_returns.index] = final_labels

        except ValueError as e:
            print(f"Warning: Could not assign N-ary labels for a group due to a pd.qcut error: {e}. "
                  f"Assigning NA to this group's labels.")

        return labels_series.astype('Int64')

    def transform(self) -> pd.DataFrame:
        """
        Calculates excess returns using the parent class's transform method.
        Then, adds an N-ary label column based on the quantiles of these excess returns.

        :return: DataFrame with excess returns and the N-ary label column.
        """
        df_with_excess_returns = super().transform()

        if isinstance(df_with_excess_returns.index, pd.MultiIndex) and \
                self.date_level_name in df_with_excess_returns.index.names:
            df_with_excess_returns[self.label_col_name] = (
                df_with_excess_returns.groupby(level=self.date_level_name, group_keys=False)
                .apply(self._assign_label)
            )
        else:
            print(f"Warning: DataFrame is not MultiIndexed by '{self.date_level_name}' or this level "
                  f"is not present in the index. Applying N-ary labeling to the entire DataFrame as one group.")
            df_with_excess_returns[self.label_col_name] = self._assign_label(df_with_excess_returns)

        return df_with_excess_returns