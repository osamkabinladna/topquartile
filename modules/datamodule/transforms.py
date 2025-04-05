import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

class CovariateTransform(ABC):
    def __init__(self, df:pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df.copy()
        self.original_columns = list(df.columns)
        self.included_features = ['PX_LAST']

    def include_features(self, feature: Union[str, List[str]]):
        if feature not in self.original_columns:
            raise ValueError(f"Feature {feature} not found in dataframe")
        if feature not in self.included_features:
            if isinstance(feature, list):
                for feat in feature:
                    self.included_features.append(feat)
            else:
                self.included_features.append(feature)

    @abstractmethod
    def transform(self) -> pd.DataFrame:
        pass

class TechnicalCovariateTransform(CovariateTransform):
    def __init__(self, df: pd.DataFrame,  sma: Optional[List] = None, ema: Optional[List] = None, rsi: Optional[List] = None, macd: bool = False,
                 macd_signal: bool = False, volume_sma: Optional[List] = None, volume_std: Optional[List] = None, vroc: Optional[List] = None,
                 macd_histogram: bool = False, obv: bool = False, roc: Optional[List] = None, volatility: Optional[List] = None,
                 price_gap: Optional[List] = None, price_vs_sma: Optional[List] = None, turnover: Optional[List] = None, beta: Optional[List] = None):
        """
        :param df: dataframe containing covariates
        :param sma: simple moving average
        :param ema: exponential moving average
        :param rsi: relative strength index
        :param macd: mean average convergence divergence
        :param macd_signal: mean average convergence divergence signal
        :param macd_histogram: mean average convergence divergence signal
        :param obv: on balance volume
        :param roc: price rate of change
        :param volatility: std of price
        :param volume: trading volume
        :param price_gap: difference between current price and simple moving average
        :param price_vs_sma: current price divided by simple moving average
        :param turnover: turnover
        :param beta: beta to market (JKSE)
        """
        self.df = df
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
        self.price_gap = price_gap
        self.price_vs_sma = price_vs_sma
        self.turnover = turnover
        self.beta = beta
        self.vroc = vroc


    def _add_sma(self):
        if self.sma is not None:
            for window in self.sma:
                self.df[f'sma_{window}'] = self.df['PX_LAST'].rolling(window=window).mean()

    def _add_ema(self):
        if self.ema is not None:
            for window in self.sma:
                self.df[f'sma_{window}'] = self.df['PX_LAST'].ewm(span=window, adjust=False).mean()

    def _add_rsi(self):
        if self.rsi is not None:
            for window in self.rsi:
                delta = self.df['PX_LAST'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                rs = avg_gain / avg_loss
                self.df[f'rsi_{window}'] = 100 - (100 / (1 + rs))

    def _add_macd(self):
        if self.macd_histogram or self.macd_signal and not self.macd_histogram:
            raise ValueError('macd required for computation of macd_histogram and macd_signal')

        if self.macd:
            ema12 = self.df['PX_LAST'].ewm(span=12, adjust=False).mean()
            ema26 = self.df['PX_LAST'].ewm(span=26, adjust=False).mean()
            self.df[f'macd'] = ema12 - ema26

            if self.macd_signal:
                self.df[f'macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()

            if self.macd_histogram:
                self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']

    def _add_obv(self):
        if self.obv:
            self.df['obv'] = (np.sign(self.df['PX_LAST'].diff()) * self.df['VOLUME']).fillna(0).cumsum()
            self.include_features('VOLUME')

    def _add_price_gap(self):
        if self.price_gap is not None:
            for window in self.price_gap:
                self.df[f'price_gap_{window}'] = self.df['PX_LAST'] - self.df['PX_LAST'].rolling(window=window).mean()

    def _add_price_vs_sma(self):
        if self.price_vs_sma is not None:
            for window in self.price_vs_sma:
                self.df[f'price_gap_{window}'] = self.df['PX_LAST'] / self.df['PX_LAST'].rolling(window=window).mean()

    def _add_roc(self):
        if self.roc is not None:
            for window in self.roc:
                self.df[f'ROC_{window}'] = ((self.df['PX_LAST'] / self.df['PX_LAST'].shift(window)) - 1) * 100

    def _add_volatility(self):
        if self.volatility is not None:
            self.df['daily_return'] = self.df['PX_LAST'].pct_change()
            for window in self.volatility:
                self.df[f'volatility_{window}'] = self.df['daily_return'].rolling(window=window).std()

    def _add_volume_sma(self):
        if self.volume_sma is not None:
            for window in self.volume_sma:
                self.df['volume_sma'] = self.df['VOLUME'].rolling(window=window).mean()
            self.include_features('VOLUME')

    def _add_volume_std(self):
        if self.volume_std is not None:
            for window in self.volume_std:
                self.df['volume_std'] = self.df['VOLUME'].rolling(window=window).std()
            self.include_features('VOLUME')

    def _add_vroc(self):
        if self.vroc is not None:
            for window in self.vroc:
                self.df[f'vroc_{window}'] = ((self.df['VOLUME'] / self.df['VOLUME'].shift(window)) - 1) * 100
            self.include_features('VOLUME')

    # TODO: Figure an ez way to calculate this
    def _add_beta(self):
        if self.beta is not None:
            raise NotImplementedError


class FundamentalCovariateTransform(CovariateTransform):
    def __init__(self, df, pe_ratio: bool = False, earnings_yield: bool = False, debt_to_assets: bool = False,
                 pe_band: Optional[Tuple[List, List]] = None, debt_to_capital: bool = False, equity_ratio: bool = False, market_to_book: bool = False,
                 adjusted_roic: bool = False, operating_efficiency: bool = False, levered_roa: bool = False, eps_growth: bool = False,):
        """
        :param df: dataframe of covariates
        :param pe_ratio: price earnings ratio
        :param earnings_yield: earnings yield
        :param debt_to_assets: debt to assets
        :param pe_band (list of windows, list of quartiles): price earnings band
        :param debt_to_capital:  debt to capital
        :param equity_ratio: equity ratio
        :param market_to_book: market to book ratio
        :param adjusted_roic: adjusted roic
        :param operating_efficiency: operating efficiency
        :param levered_roa: levered roa
        :param eps_growth: earnings per share growth
        """
        self.df = df
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

    def _add_pe_ratio(self):
        if self.pe_ratio:
            self.df['pe_ratio'] = self.df['PX_LAST'] / self.df['IS_EPS']
            self.include_features(['IS_EPS'])

    def _add_pe_band(self):
        if not self.pe_ratio:
            raise ValueError('pe_ratio required for computation of pe_band; set pe_ratio=True')
        if self.pe_band is not None:
            for window in self.pe_band[0]:
                for q in self.pe_band[1]:
                    self.df[f'pe_band_{window}_{q}'] = self.df['pe_ratio'].rolling(window).quantile(q/100)

    def earnings_yield(self):
        if self.earnings_yield:
            self.df['Earnings_Yield'] = self.df['IS_EPS'] / self.df['PX_LAST'].replace(0, np.nan)
            self.include_features(['IS_EPS'])

    def _add_debt_to_assets(self):
        if self.debt_to_assets:
            self.df['Debt_to_Assets'] = self.df['BS_TOTAL_LIABILITIES'] / self.df['BS_TOT_ASSET'].replace(0, np.nan)
            self.include_features(['BS_TOT_ASSET', 'BS_TOTAL_LIABILITIES'])

    def _add_debt_to_capital(self):
        if self.debt_to_capital:
            self.df['Debt_to_Capital'] = self.df['BS_TOTAL_LIABILITIES'] / (
                    self.df['BS_TOTAL_LIABILITIES'] + (self.df['BS_TOT_ASSET'] - self.df['BS_TOTAL_LIABILITIES']))
            self.include_features(['BS_TOT_ASSET', 'BS_TOTAL_LIABILITIES'])

    def _add_equity_ratio(self):
        if self.equity_ratio:
            self.df['equity_ratio'] = (self.df['BS_TOT_ASSET'] - self.df['BS_TOTAL_LIABILITIES']) / self.df['BS_TOT_ASSET'].replace(0, np.nan)
            self.include_features(['BS_TOT_ASSET', 'BS_TOTAL_LIABILITIES'])

    def _market_to_book(self):
        if self.market_to_book:
            self.df['Market_to_Book'] = self.df['CUR_MKT_CAP'] / ((self.df['BS_TOT_ASSET'] - self.df['BS_TOTAL_LIABILITIES']).replace(0, np.nan))
            self.include_features(['BS_TOT_ASSET', 'BS_TOTAL_LIABILITIES', 'CUR_MKT_CAP'])

    def _add_adjusted_roic(self):
        if self.adjusted_roic:
            self.df['Adjusted_ROIC'] = self.df['OPERATING_ROIC'] - self.df['WACC']
            self.include_features(['OPERATING_ROIC', 'WACC'])

    def _add_operating_efficiency(self):
        if self.operating_efficiency:
            self.df['Operating_Efficiency'] = self.df['OPER_MARGIN'] * self.df['SALES_GROWTH']
            self.include_features(['OPER_MARGIN', 'SALES_GROWTH'])

    def _add_levered_roa(self):
        if self.levered_roa:
            self.df['Levered_ROA'] = self.df['RETURN_ON_ASSET'] * (1 + self.df['TOT_DEBT_TO_TOT_EQY'])
            self.include_features(['RETURN_ON_ASSET', 'TOT_DEBT_TO_TOT_EQY'])

    # TODO: This is very dubious
    def _add_eps_growth(self):
        if self.eps_growth:
            self.df['EPS_Growth'] = (self.df['IS_EPS'] - self.df['IS_EPS'].shift(1)) / self.df['IS_EPS'].shift(1)
