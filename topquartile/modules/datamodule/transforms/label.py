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


class BinaryLabelTransform(LabelTransform):
    def __init__(self, df: pd.DataFrame, label_duration: int, quantile: float,
                 index_ticker: str = "^JKSE", price_column: str = 'PX_LAST',
                 ticker_level_name: str = 'ticker',
                 date_level_name: str = 'Dates'):
        """
        :param df: dataframe to be transformed
        :param label_duration: asset holding period
        :param quantile: quantile of labels
        :param index_ticker: ticker of market index
        :param price_column: column name of price
        :param ticker_level_name: Multiindex name for ticker
        :param date_level_name: Multiindex name for date
        """
        super().__init__(df)
        self.label_duration = label_duration
        self.quantile = quantile
        self.index_ticker = index_ticker
        self.price_column = price_column
        self.label_col_name = 'label'
        self.ticker_level_name = ticker_level_name
        self.date_level_name = date_level_name


    def _calculate_returns(self, series: pd.Series) -> pd.Series:
        future_price = series.shift(-self.label_duration)
        returns = ((future_price - series) / series) * 100
        return returns

    def _get_index_returns(self) -> pd.Series:
        unique_dates = self.df.index.get_level_values(self.date_level_name).unique()
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
        required_end_date = unique_dates.max()
        try:
            from pandas.tseries.offsets import BDay
            download_end_date = required_end_date + BDay(1)
        except ImportError:
            print("Warning: pandas.tseries.offsets not available. Using max date directly.")
            download_end_date = required_end_date

        print(f"Attempting yfinance download for {self.index_ticker} from {start_date} to {download_end_date}")
        try:
            index_data = yf.download(self.index_ticker, start=start_date, end=download_end_date,
                                     progress=False, auto_adjust=False)
        except Exception as e:
            raise ConnectionError(f"Failed to download index data for {self.index_ticker}: {e}")

        if index_data.empty:
            raise ValueError(f"No data downloaded from yfinance for {self.index_ticker} in the specified range.")

        index_data.index = pd.to_datetime(index_data.index)
        price_col_yf = 'Close'

        if price_col_yf not in index_data.columns:
            raise KeyError(
                f"Column '{price_col_yf}' not found in downloaded yfinance data for {self.index_ticker}. Available columns: {index_data.columns.tolist()}")

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

        index_returns_raw = self._calculate_returns(price_series)

        if not isinstance(index_returns_raw, pd.Series):
            raise TypeError(
                f"_calculate_returns function unexpectedly returned a {type(index_returns_raw)}, expected Series.")

        raw_return_dates = index_returns_raw.index
        nans_before_reindex = index_returns_raw.isnull().sum()

        if index_returns_raw.index.tz is None and unique_dates.tz is not None:
            print(
                "Warning: Raw index dates are timezone-naive, but target dates have timezone. Localizing target dates.")
            try:
                unique_dates = unique_dates.tz_localize(None)
            except TypeError:
                pass
        elif index_returns_raw.index.tz is not None and unique_dates.tz is None:
            print(
                "Warning: Raw index dates have timezone, but target dates are naive. Removing timezone from raw dates.")
            raw_return_dates = index_returns_raw.index.tz_localize(None)
            index_returns_raw.index = raw_return_dates

        aligned_index_returns = index_returns_raw.reindex(unique_dates)
        aligned_index_returns.name = 'INDEX_RETURN'

        total_nans_after_reindex = aligned_index_returns.isnull().sum()
        dates_introduced_by_reindex = unique_dates.difference(raw_return_dates)

        nans_at_introduced_dates = 0
        if not dates_introduced_by_reindex.empty:
            nans_at_introduced_dates = aligned_index_returns.loc[dates_introduced_by_reindex].isnull().sum()
        else:
            pass

        if nans_at_introduced_dates > 0:
            print(
                f"   -> {nans_at_introduced_dates} NaN(s) were introduced because the corresponding date(s) were missing in the downloaded '{self.index_ticker}' price data index.")
        elif total_nans_after_reindex > nans_before_reindex:
            print(
                f"   -> Total NaNs increased from {nans_before_reindex} to {total_nans_after_reindex}, but not attributed to missing dates. Manual check recommended.")
        else:
            print(
                f"   ->  No NaNs seem to have been introduced *solely* due to the reindexing process itself for missing dates.")
            if total_nans_after_reindex > 0:
                print(
                    f"      Remaining {total_nans_after_reindex} NaNs likely originate from yfinance data gaps or the forward return calculation window.")
        return aligned_index_returns

    def _assign_label(self, group: pd.DataFrame) -> pd.Series:
        excess_return_col = 'EXCESS_RETURN'
        valid_returns = group[excess_return_col].dropna()
        if valid_returns.empty:
            return pd.Series(pd.NA, index=group.index, dtype='Int64')
        quantile_threshold = valid_returns.quantile(self.quantile)
        if pd.isna(quantile_threshold):
            return pd.Series(pd.NA, index=group.index, dtype='Int64')
        label = (group[excess_return_col] >= quantile_threshold)
        label = label.astype(float).where(group[excess_return_col].notna()).astype('Int64')
        return label

    def transform(self) -> pd.DataFrame:
        df_copy = self.df.copy()
        df_copy = df_copy.sort_index(level=[self.ticker_level_name, self.date_level_name])

        date_level_values = df_copy.index.get_level_values(self.date_level_name)
        if not pd.api.types.is_datetime64_any_dtype(date_level_values.dtype):
            try:
                new_levels = [
                    df_copy.index.get_level_values(self.ticker_level_name),
                    pd.to_datetime(date_level_values)
                ]
                new_names = [self.ticker_level_name, self.date_level_name]
                df_copy.index = pd.MultiIndex.from_arrays(new_levels, names=new_names)
            except Exception as e:
                raise ValueError(f"Could not convert the '{self.date_level_name}' index level to datetime. Please check the data. Error: {e}") from e

        stock_return_col = f'{self.label_duration}d_stock_return'
        df_copy[stock_return_col] = df_copy.groupby(level=self.ticker_level_name, group_keys=False, observed=False)[self.price_column].apply(self._calculate_returns)
        index_returns_series = self._get_index_returns()
        df_copy = df_copy.join(index_returns_series, on=self.date_level_name)
        excess_return_col = 'EXCESS_RETURN'
        df_copy[excess_return_col] = df_copy[stock_return_col] - df_copy['INDEX_RETURN']
        df_copy[self.label_col_name] = df_copy.groupby(level=self.date_level_name, group_keys=False).apply(self._assign_label)

        return df_copy

