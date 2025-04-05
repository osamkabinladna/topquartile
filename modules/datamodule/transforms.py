import pandas as pd
import numpy as np
from typing import List

class TechnicalCovariateTransform():
    def __init__(self, df: pd.DataFrame,  sma: List = None, ema: List = None, rsi: List = None, macd: bool = None,
                 macd_signal: bool = None, volume_sma: List = None, volume_std: List = None, vroc: List = None,
                 macd_histogram: bool = None, obv: bool=None, roc: List = None, volatility: List = None,
                 price_gap: List = None, price_vs_sma: List = None, turnover: List = None, beta: List = None):
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
        self.features_used = ['PX_LAST']

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
        if self.macd_histogram is not None or self.macd_signal is not None and self.macd_histogram is None:
            raise ValueError('macd required for computation of macd_histogram and macd_signal')

        if self.macd is not None:
            ema12 = self.df['PX_LAST'].ewm(span=12, adjust=False).mean()
            ema26 = self.df['PX_LAST'].ewm(span=26, adjust=False).mean()
            self.df[f'macd'] = ema12 - ema26

            if self.macd_signal is not None:
                self.df[f'macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()

            if self.macd_histogram is not None:
                self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']

    def _add_obv(self):
        if self.obv is not None:
            self.df['obv'] = (np.sign(self.df['PX_LAST'].diff()) * self.df['VOLUME']).fillna(0).cumsum()

        if "VOLUME" not in self.features_used:
            self.features_used.append('VOLUME')


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

    def _add_volume_std(self):
        if self.volume_sma is not None:
            for window in self.volume_std:
                self.df['volume_std'] = self.df['VOLUME'].rolling(window=window).std()

    def _add_vroc(self):
        if self.vroc is not None:
            for window in self.vroc:
                self.df[f'vroc_{window}'] = ((self.df['VOLUME'] / self.df['VOLUME'].shift(window)) - 1) * 100

    # TODO: Figure an ez way to calculate this
    def _add_beta(self):
        if self.beta is not None:
            raise NotImplementedError





