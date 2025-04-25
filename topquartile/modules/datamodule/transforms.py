import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import warnings
import yfinance as yf

class CovariateTransform(ABC):
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.df = df

    @abstractmethod
    def transform(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def group_transform(self, group: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TechnicalCovariateTransform(CovariateTransform):
    def __init__(self, df: pd.DataFrame, sma: Optional[List[int]] = None, ema: Optional[List[int]] = None,
                 rsi: Optional[List[int]] = None, macd: bool = False, macd_signal: bool = False,
                 macd_histogram: bool = False, obv: bool = False, roc: Optional[List[int]] = None,
                 volatility: Optional[List[int]] = None, volume_sma: Optional[List[int]] = None,
                 volume_std: Optional[List[int]] = None, vroc: Optional[List[int]] = None,
                 price_gap: Optional[List[int]] = None, price_vs_sma: Optional[List[int]] = None,
                 turnover: Optional[List[int]] = None, beta: Optional[List[int]] = None):
        """
        :param df: DataFrame containing covariates
        :param sma: List of window sizes for Simple Moving Average
        :param ema: List of window sizes for Exponential Moving Average
        :param rsi: List of window sizes for Relative Strength Index
        :param macd: Calculate MACD (12-ema - 26-ema). Required for signal/histogram
        :param macd_signal: Calculate MACD signal line (9-ema of MACD)
        :param macd_histogram: Calculate MACD histogram (MACD - signal)
        :param obv: Calculate On-Balance Volume. Requires 'VOLUME'
        :param roc: List of periods for Price Rate of Change
        :param volatility: List of window sizes for rolling standard deviation of daily returns
        :param volume_sma: List of window sizes for Simple Moving Average of Volume. Requires 'VOLUME'
        :param volume_std: List of window sizes for Rolling Standard Deviation of Volume. Requires 'VOLUME'
        :param vroc: List of periods for Volume Rate of Change. Requires 'VOLUME'
        :param price_gap: List of window sizes for Price - SMA(window)
        :param price_vs_sma: List of window sizes for Price / SMA(window)
        :param turnover: List of window sizes for Turnover calculation (Placeholder - Requires definition)
        :param beta: List of window sizes for Beta calculation
        """
        super().__init__(df)

        self.sma = sma
        self.ema = ema
        self.rsi = rsi
        self.macd = macd
        self.macd_signal = macd_signal
        self.macd_histogram = macd_histogram
        self.obv = obv
        self.roc = roc
        self.volatility = volatility
        self.volume_sma = volume_sma
        self.volume_std = volume_std
        self.vroc = vroc
        self.price_gap = price_gap
        self.price_vs_sma = price_vs_sma
        self.turnover = turnover
        self.beta = beta
        self.required_base = set()

        self.required_base.update(['PX_LAST'])
        if any([self.obv, self.volume_sma, self.volume_std, self.vroc]):
            self.required_base.update('VOLUME')

        missing_base = [col for col in self.required_base if col not in self.df.columns]
        if missing_base:
            raise ValueError(f"Missing required base columns in DataFrame: {missing_base}")


    def group_transform(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_index()
        group = self._add_sma(group)
        group = self._add_ema(group)
        group = self._add_rsi(group)
        group = self._add_macd(group)
        group = self._add_obv(group)
        group = self._add_roc(group)
        group = self._add_volatility(group)
        group = self._add_volume_sma(group)
        group = self._add_volume_std(group)
        group = self._add_vroc(group)
        group = self._add_price_gap(group)
        group = self._add_price_vs_sma(group)
        group = self._add_turnover(group)
        group = self._add_beta(group)

        return group

    def transform(self) -> pd.DataFrame:
        transformed_df = self.df.groupby('ticker', group_keys=True).apply(self.group_transform)
        return transformed_df.sort_index()


    def _add_sma(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.sma is not None:
            for window in self.sma:
                group_df[f'sma_{window}'] = group_df['PX_LAST'].rolling(window=window, min_periods=window).mean()
        return group_df

    def _add_ema(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.ema is not None:
            for window in self.ema:
                group_df[f'ema_{window}'] = group_df['PX_LAST'].ewm(span=window, adjust=False, min_periods=window).mean()
        return group_df

    def _add_rsi(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.rsi is not None:
            for window in self.rsi:
                delta = group_df['PX_LAST'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(com=window - 1, adjust=False, min_periods=window).mean()
                avg_loss = loss.ewm(com=window - 1, adjust=False, min_periods=window).mean()

                rs = avg_gain / avg_loss
                rsi_val = 100.0 - (100.0 / (1.0 + rs))
                rsi_val[avg_loss == 0] = 100.0
                rsi_val[(avg_gain == 0) & (avg_loss == 0)] = np.nan
                group_df[f'rsi_{window}'] = rsi_val

        return group_df

    def _add_macd(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.macd:
            ema12 = group_df['PX_LAST'].ewm(span=12, adjust=False, min_periods=12).mean()
            ema26 = group_df['PX_LAST'].ewm(span=26, adjust=False, min_periods=26).mean()
            group_df['macd'] = ema12 - ema26

            if self.macd_signal:
                group_df['macd_signal'] = group_df['macd'].ewm(span=9, adjust=False, min_periods=9).mean()

                if self.macd_histogram:
                    group_df['macd_histogram'] = group_df['macd'] - group_df['macd_signal']
            elif self.macd_histogram:
                signal = group_df['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
                group_df['macd_histogram'] = group_df['macd'] - signal

        return group_df

    def _add_obv(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.obv:
            if 'VOLUME' not in group_df.columns:
                warnings.warn(f"Skipping OBV for group: 'VOLUME' column not found.", UserWarning)
                return group_df
            price_diff = group_df['PX_LAST'].diff()
            volume = group_df['VOLUME']
            signed_volume = (np.sign(price_diff) * volume).fillna(0)
            group_df['obv'] = signed_volume.cumsum()
        return group_df

    def _add_price_gap(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.price_gap is not None:
            for window in self.price_gap:
                sma = group_df['PX_LAST'].rolling(window=window, min_periods=window).mean()
                group_df[f'price_gap_{window}'] = group_df['PX_LAST'] - sma
        return group_df

    def _add_price_vs_sma(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.price_vs_sma is not None:
            for window in self.price_vs_sma:
                sma = group_df['PX_LAST'].rolling(window=window, min_periods=window).mean()
                group_df[f'price_div_sma_{window}'] = (group_df['PX_LAST'] / sma).replace([np.inf, -np.inf], np.nan)
        return group_df

    def _add_roc(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.roc is not None:
            for window in self.roc:
                shifted_price = group_df['PX_LAST'].shift(window)
                group_df[f'roc_{window}'] = ((group_df['PX_LAST'] / shifted_price) - 1).replace([np.inf, -np.inf], np.nan) * 100
        return group_df

    def _add_volatility(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.volatility is not None:
            daily_return = group_df['PX_LAST'].pct_change(fill_method=None)
            for window in self.volatility:
                group_df[f'volatility_{window}'] = daily_return.rolling(window=window, min_periods=window).std()
        return group_df

    def _add_volume_sma(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.volume_sma:
            if 'VOLUME' not in group_df.columns:
                warnings.warn(f"Skipping Volume SMA for group: 'VOLUME' column not found.", UserWarning)
                return group_df
            for window in self.volume_sma:
                group_df[f'volume_sma_{window}'] = group_df['VOLUME'].rolling(window=window, min_periods=window).mean()
        return group_df

    def _add_volume_std(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.volume_std is not None:
            if 'VOLUME' not in group_df.columns:
                warnings.warn(f"Skipping Volume StDev for group: 'VOLUME' column not found.", UserWarning)
                return group_df
            for window in self.volume_std:
                group_df[f'volume_std_{window}'] = group_df['VOLUME'].rolling(window=window, min_periods=window).std()
        return group_df

    def _add_vroc(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.vroc is not None:
            if 'VOLUME' not in group_df.columns:
                warnings.warn(f"Skipping VROC for group: 'VOLUME' column not found.", UserWarning)
                return group_df
            for window in self.vroc:
                shifted_volume = group_df['VOLUME'].shift(window)
                group_df[f'vroc_{window}'] = ((group_df['VOLUME'] / shifted_volume) - 1).replace([np.inf, -np.inf], np.nan) * 100
        return group_df

    def _add_turnover(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.turnover is not None:
            warnings.warn(f"Turnover calculation not implemented. Requires additional data (Shares Outstanding).", UserWarning)
            raise NotImplementedError
        return group_df

    def _add_beta(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.beta:
            raise NotImplementedError
        return group_df


class FundamentalCovariateTransform(CovariateTransform):
    def __init__(self, df, pe_ratio: bool = False, earnings_yield: bool = False, debt_to_assets: bool = False,
                 pe_band: Optional[Tuple[List[int], List[int]]] = None, debt_to_capital: bool = False, equity_ratio: bool = False, market_to_book: bool = False,
                 adjusted_roic: bool = False, operating_efficiency: bool = False, levered_roa: bool = False, eps_growth: bool = False):
        """
        :param df: dataframe of covariates including fundamental data and price/market cap
        :param pe_ratio: Calculate Price to Earnings ratio
        :param earnings_yield: Calculate Earnings Yield (reciprocal of PE)
        :param debt_to_assets: Calculate Debt to Assets ratio
        :param pe_band: Tuple containing (list of window sizes, list of quantiles [0-100]) for PE Ratio rolling quantiles
        :param debt_to_capital: Calculate Debt to Capital ratio
        :param equity_ratio: Calculate Equity Ratio (inverse of Debt to Assets based on Liabilities)
        :param market_to_book: Calculate Market to Book ratio
        :param adjusted_roic: Calculate Adjusted ROIC (Operating ROIC - WACC)
        :param operating_efficiency: Calculate Operating Efficiency (Operating Margin * Sales Growth)
        :param levered_roa: Calculate Levered ROA (ROA * (1 + Debt/Equity))
        :param eps_growth: Calculate Earnings Per Share growth (period over period)
        """
        super().__init__(df)

        self.pe_ratio = pe_ratio
        self.earnings_yield = earnings_yield
        self.debt_to_assets = debt_to_assets
        self.pe_band = pe_band
        self.debt_to_capital = debt_to_capital
        self.equity_ratio = equity_ratio
        self.market_to_book = market_to_book
        self.adjusted_roic = adjusted_roic
        self.operating_efficiency = operating_efficiency
        self.levered_roa = levered_roa
        self.eps_growth = eps_growth

        self.required_base = set()
        if self.pe_ratio or self.earnings_yield or self.pe_band:
            self.required_base.update(['PX_LAST', 'IS_EPS'])
        if self.debt_to_assets or self.debt_to_capital or self.equity_ratio or self.market_to_book:
            self.required_base.update(['BS_TOT_ASSET', 'BS_TOTAL_LIABILITIES'])
        if self.market_to_book:
            self.required_base.add('CUR_MKT_CAP')
        if self.adjusted_roic:
            self.required_base.update(['OPERATING_ROIC', 'WACC'])
        if self.operating_efficiency:
            self.required_base.update(['OPER_MARGIN', 'SALES_GROWTH'])
        if self.levered_roa:
            self.required_base.update(['RETURN_ON_ASSET', 'TOT_DEBT_TO_TOT_EQY'])
        if self.eps_growth:
            self.required_base.add('IS_EPS')

        if self.pe_band is not None:
            self.required_base.update(['PX_LAST', 'IS_EPS'])
            if not self.pe_ratio:
                warnings.warn("pe_band requested without pe_ratio=True. PE Ratio will be calculated but not added unless pe_ratio=True.", UserWarning)

        missing_base = [col for col in self.required_base if col not in self.df.columns]
        if missing_base:
            raise ValueError(f"Missing required base columns in DataFrame for fundamental calculations: {missing_base}")

    def group_transform(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_index()

        if self.pe_band is not None or self.pe_ratio:
            group = self._add_pe_ratio(group, add_column=self.pe_ratio)

        group = self._add_pe_band(group)
        group = self._add_earnings_yield(group)
        group = self._add_debt_to_assets(group)
        group = self._add_debt_to_capital(group)
        group = self._add_equity_ratio(group)
        group = self._add_market_to_book(group)
        group = self._add_adjusted_roic(group)
        group = self._add_operating_efficiency(group)
        group = self._add_levered_roa(group)
        group = self._add_eps_growth(group)

        return group

    def transform(self) -> pd.DataFrame:
        transformed_df = self.df.groupby(level='ticker', group_keys=False).apply(self.group_transform)
        return transformed_df.sort_index()


    def _add_pe_ratio(self, group_df: pd.DataFrame, add_column: bool = True) -> pd.DataFrame:
        if self.pe_ratio or self.pe_band is not None:
            safe_eps = group_df['IS_EPS'].replace(0, np.nan)
            safe_eps[safe_eps < 0] = np.nan
            pe_col = group_df['PX_LAST'] / safe_eps
            pe_col = pe_col.replace([np.inf, -np.inf], np.nan)
            if add_column:
                group_df['pe_ratio'] = pe_col
            else:
                group_df['_temp_pe_ratio'] = pe_col
        return group_df

    def _add_pe_band(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.pe_band is not None:
            pe_col_name = 'pe_ratio' if 'pe_ratio' in group_df.columns else '_temp_pe_ratio'
            if pe_col_name not in group_df.columns:
                warnings.warn(f"Skipping PE Band for group '{group_df['ticker'].iloc[0]}': Base PE Ratio not calculated.", UserWarning)
                return group_df

            if not (isinstance(self.pe_band, tuple) and len(self.pe_band) == 2 and
                    isinstance(self.pe_band[0], list) and isinstance(self.pe_band[1], list)):
                raise ValueError("pe_band parameter must be a tuple containing two lists: ([windows], [quantiles])")

            windows, quantiles = self.pe_band
            for window in windows:
                for q_percent in quantiles:
                    if not 0 <= q_percent <= 100:
                        warnings.warn(f"Skipping PE Band quantile {q_percent}. Quantiles should be between 0 and 100.", UserWarning)
                        continue
                    q_decimal = q_percent / 100.0
                    try:
                        group_df[f'pe_band_{window}_{q_percent}'] = group_df[pe_col_name].rolling(window=window, min_periods=max(1, int(window*q_decimal))).quantile(q_decimal, interpolation='linear')
                    except Exception as e:
                        warnings.warn(f"Could not calculate pe_band_{window}_{q_percent} for group '{group_df['ticker'].iloc[0]}'. Error: {e}", UserWarning)

            if '_temp_pe_ratio' in group_df.columns:
                group_df = group_df.drop(columns=['_temp_pe_ratio'])

        return group_df

    def _add_earnings_yield(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.earnings_yield:
            safe_price = group_df['PX_LAST'].replace(0, np.nan)
            group_df['earnings_yield'] = (group_df['IS_EPS'] / safe_price).replace([np.inf, -np.inf], np.nan)
        return group_df

    def _add_debt_to_assets(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.debt_to_assets:
            safe_assets = group_df['BS_TOT_ASSET'].replace(0, np.nan)
            group_df['debt_to_assets'] = (group_df['BS_TOTAL_LIABILITIES'] / safe_assets).replace([np.inf, -np.inf], np.nan)
        return group_df

    def _add_debt_to_capital(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.debt_to_capital:
            total_equity = group_df['BS_TOT_ASSET'] - group_df['BS_TOTAL_LIABILITIES']
            total_capital = group_df['BS_TOTAL_LIABILITIES'] + total_equity
            safe_capital = total_capital.replace(0, np.nan)
            group_df['debt_to_capital'] = (group_df['BS_TOTAL_LIABILITIES'] / safe_capital).replace([np.inf, -np.inf], np.nan)
        return group_df

    def _add_equity_ratio(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.equity_ratio:
            total_equity = group_df['BS_TOT_ASSET'] - group_df['BS_TOTAL_LIABILITIES']
            safe_assets = group_df['BS_TOT_ASSET'].replace(0, np.nan)
            group_df['equity_ratio'] = (total_equity / safe_assets).replace([np.inf, -np.inf], np.nan)
        return group_df

    def _add_market_to_book(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.market_to_book:
            book_value = group_df['BS_TOT_ASSET'] - group_df['BS_TOTAL_LIABILITIES']
            safe_book_value = book_value.replace(0, np.nan)
            safe_book_value[safe_book_value < 0] = np.nan
            group_df['market_to_book'] = (group_df['CUR_MKT_CAP'] / safe_book_value).replace([np.inf, -np.inf], np.nan)
        return group_df

    def _add_adjusted_roic(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.adjusted_roic:
            group_df['adjusted_roic'] = group_df['OPERATING_ROIC'] - group_df['WACC']
        return group_df

    def _add_operating_efficiency(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.operating_efficiency:
            group_df['operating_efficiency'] = group_df['OPER_MARGIN'] * group_df['SALES_GROWTH']
        return group_df

    def _add_levered_roa(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.levered_roa:
            leverage_factor = (1 + group_df['TOT_DEBT_TO_TOT_EQY'].fillna(0))
            group_df['levered_roa'] = group_df['RETURN_ON_ASSET'] * leverage_factor
        return group_df

    def _add_eps_growth(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.eps_growth:
            eps_prior = group_df['IS_EPS'].shift(1)
            safe_eps_prior = eps_prior.replace(0, np.nan)
            safe_eps_prior[safe_eps_prior < 0] = np.nan

            growth = (group_df['IS_EPS'] - safe_eps_prior) / safe_eps_prior
            group_df['eps_growth'] = growth.replace([np.inf, -np.inf], np.nan)
        return group_df


class LabelTransform:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.df = df

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
                raise ValueError(f"Could not convert unique dates from level '{self.date_level_name}' for yfinance download. Check data. Error: {e}") from e

        unique_dates = unique_dates.sort_values()
        if unique_dates.empty:
            raise ValueError(f"No dates found in the DataFrame's '{self.date_level_name}' index level.")

        start_date = unique_dates.min()
        required_end_date = unique_dates.max()

        try:
            index_data = yf.download(self.index_ticker, start=start_date, end=required_end_date,
                progress=False, auto_adjust=False)
        except Exception as e:
            raise ConnectionError(f"Failed to download index data for {self.index_ticker}: {e}")

        index_data.index = pd.to_datetime(index_data.index)
        price_col_yf = 'Close'

        price_data_selection = index_data[price_col_yf]
        if isinstance(price_data_selection, pd.DataFrame):
            if price_data_selection.shape[1] == 1:
                price_series = price_data_selection.iloc[:, 0]
            else:
                raise TypeError(f"Selection of '{price_col_yf}' yielded DataFrame with multiple columns: {price_data_selection.columns}. Cannot proceed.")
        elif isinstance(price_data_selection, pd.Series):
            price_series = price_data_selection
        else:
            raise TypeError(f"Selection of '{price_col_yf}' yielded unexpected type: {type(price_data_selection)}")

        index_returns = self._calculate_returns(price_series)

        if not isinstance(index_returns, pd.Series):
            raise TypeError(f"_calculate_returns function unexpectedly returned a {type(index_returns)}, expected Series.")

        index_returns.name = 'INDEX_RETURN'
        aligned_index_returns = index_returns.reindex(unique_dates)

        nan_count = aligned_index_returns.isnull().sum()
        if pd.api.types.is_scalar(nan_count):
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN values found in index returns after aligning.")
        else:
            raise TypeError(f"Calculation of NaN count failed. Expected scalar, got {type(nan_count)}. This might indicate aligned_index_returns is not a Series.")

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
        df_copy[stock_return_col] = df_copy.groupby(level=self.ticker_level_name, group_keys=False)[self.price_column].apply(self._calculate_returns)
        index_returns_series = self._get_index_returns()
        df_copy = df_copy.join(index_returns_series, on=self.date_level_name)
        excess_return_col = 'EXCESS_RETURN'
        df_copy[excess_return_col] = df_copy[stock_return_col] - df_copy['INDEX_RETURN']
        df_copy[self.label_col_name] = df_copy.groupby(level=self.date_level_name, group_keys=False).apply(self._assign_label)

        return df_copy