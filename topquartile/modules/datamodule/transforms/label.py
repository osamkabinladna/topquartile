import pandas as pd
import yfinance as yf
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import warnings
from pandas.tseries.offsets import BDay

class LabelTransform(ABC):
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.df = df.copy()

    @abstractmethod
    def transform(self) -> pd.DataFrame:
        raise NotImplementedError


class ExcessReturnTransform(LabelTransform):
    def __init__(self, df: pd.DataFrame, label_duration: int,
                 index_ticker: str = "^JKSE", price_column: str = 'PX_LAST',
                 ticker_level_name: str = 'ticker',
                 date_level_name: str = 'Dates'):
        """
        Calculates excess returns for assets compared to a market index (IHSG Default).

        :param df: dataframe to be transformed
        :param label_duration: asset holding period for return calculation
        :param index_ticker: ticker of market index
        :param price_column: column name of asset price
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
        future_price = series.shift(-self.label_duration)
        returns = ((future_price - series) / series) * 100
        return returns

    def _get_index_returns(self) -> pd.Series:
        df_for_dates = self.df
        unique_dates = df_for_dates.index.get_level_values(self.date_level_name).unique()

        if not pd.api.types.is_datetime64_any_dtype(unique_dates.dtype):
            try:
                unique_dates = pd.to_datetime(unique_dates)
            except Exception as e:
                raise ValueError(
                    f"Could not convert unique dates from level '{self.date_level_name}' for yfinance download. Check data. Error: {e}") from e

        unique_dates = unique_dates.sort_values()
        if unique_dates.empty:
            raise ValueError(f"No dates found in the DataFrame's '{self.date_level_name}' index level.")

        start_date = unique_dates.min()
        required_end_date_for_prices = unique_dates.max()
        if self.label_duration > 0:
            buffer_days = pd.Timedelta(days=self.label_duration * 2 + 30)  # A generous buffer
            fetch_end_date = required_end_date_for_prices + buffer_days
        else:
            fetch_end_date = required_end_date_for_prices

        try:
            index_data = yf.download(self.index_ticker, start=start_date, end=fetch_end_date,
                                     progress=False, auto_adjust=False)
        except Exception as e:
            raise ConnectionError(
                f"Failed to download index data for {self.index_ticker} from {start_date} to {fetch_end_date}: {e}")

        if index_data.empty:
            raise ValueError(
                f"No data downloaded for index {self.index_ticker} from {start_date} to {fetch_end_date}. Check ticker and date range.")

        index_data.index = pd.to_datetime(index_data.index)
        price_col_yf = 'Close'

        if price_col_yf not in index_data.columns:
            raise KeyError(
                f"Column '{price_col_yf}' not found in downloaded index data. Available columns: {index_data.columns}")

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
            print(
                f"Info: {nan_count} NaN values in aligned index returns (total {len(aligned_index_returns)}). This is expected for last {self.label_duration} periods or due to missing market data on specific dates.")
        else:
            raise TypeError(
                f"Calculation of NaN count failed. Expected scalar, got {type(nan_count)}. This might indicate aligned_index_returns is not a Series.")

        return aligned_index_returns

    def transform(self) -> pd.DataFrame:
        df_transformed = self.df

        df_transformed = df_transformed.sort_index(level=[self.ticker_level_name, self.date_level_name])

        date_level_values = df_transformed.index.get_level_values(self.date_level_name)
        if not pd.api.types.is_datetime64_any_dtype(date_level_values.dtype):
            try:
                new_levels = list(df_transformed.index.levels)
                date_level_idx = df_transformed.index.names.index(self.date_level_name)
                new_levels[date_level_idx] = pd.to_datetime(date_level_values.unique())

                current_levels = [df_transformed.index.get_level_values(name) for name in df_transformed.index.names]
                current_levels[date_level_idx] = pd.to_datetime(current_levels[date_level_idx])
                df_transformed.index = pd.MultiIndex.from_arrays(current_levels, names=df_transformed.index.names)
                df_transformed = df_transformed.sort_index(
                    level=[self.ticker_level_name, self.date_level_name])  # Re-sort after potential reindexing

            except Exception as e:
                raise ValueError(
                    f"Could not convert the '{self.date_level_name}' index level to datetime. Please check the data. Error: {e}") from e

        df_transformed[self.stock_return_col_name] = df_transformed.groupby(
            level=self.ticker_level_name, group_keys=False
        )[self.price_column].apply(self._calculate_returns)

        index_returns_series = self._get_index_returns()
        df_transformed = df_transformed.join(index_returns_series, on=self.date_level_name)

        df_transformed[self.excess_return_col_name] = df_transformed[self.stock_return_col_name] - df_transformed[
            self.index_return_col_name]

        self.df = df_transformed
        return self.df


class BinaryLabelTransform(ExcessReturnTransform):
    def __init__(self, df: pd.DataFrame, label_duration: int, quantile: float,
                 index_ticker: str = "^JKSE", price_column: str = 'PX_LAST',
                 ticker_level_name: str = 'ticker',
                 date_level_name: str = 'Dates'):
        """
        Generates binary labels based on whether an asset's excess return
        is in the top quantile for a given date.

        :param df: dataframe to be transformed
        :param label_duration: asset holding period
        :param quantile: quantile of excess returns to define the top performers (e.g., 0.8 for top 20%)
        :param index_ticker: ticker of market index
        :param price_column: column name of price
        :param ticker_level_name: Multiindex name for ticker
        :param date_level_name: Multiindex name for date
        """
        super().__init__(df, label_duration, index_ticker, price_column,
                         ticker_level_name, date_level_name)
        if not (0 < quantile < 1):
            raise ValueError("Quantile must be between 0 and 1 (exclusive).")
        self.quantile = quantile
        self.label_col_name = 'label'

    def _assign_label(self, group: pd.DataFrame) -> pd.Series:
        """
        Assigns a binary label to each stock in the group for a given date.
        Label is 1 if excess return >= quantile_threshold, 0 otherwise.
        NaN excess returns result in NA labels.
        """
        excess_return_col = self.excess_return_col_name

        valid_returns = group[excess_return_col].dropna()
        if valid_returns.empty:
            return pd.Series(pd.NA, index=group.index, dtype='Int64')

        quantile_threshold = valid_returns.quantile(self.quantile)

        if pd.isna(quantile_threshold):
            return pd.Series(pd.NA, index=group.index, dtype='Int64')

        labels = pd.Series(pd.NA, index=group.index, dtype='Int64')

        condition = group[excess_return_col] >= quantile_threshold
        labels = labels.mask(group[excess_return_col].notna(), condition.astype(float).astype('Int64'))

        return labels

    def transform(self) -> pd.DataFrame:
        """
        First, calculates excess returns using the parent's transform method.
        Then, assigns binary labels based on the quantile of these excess returns.
        """
        df_with_excess_returns = super().transform()

        df_with_excess_returns[self.label_col_name] = df_with_excess_returns.groupby(
            level=self.date_level_name, group_keys=False
        ).apply(self._assign_label)

        self.df = df_with_excess_returns
        return self.df

