import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import warnings
import yfinance as yf
from tsfresh.feature_extraction.feature_calculators import autocorrelation
from tsfresh.feature_extraction.feature_calculators import approximate_entropy
from tsfresh.feature_extraction.feature_calculators import ar_coefficient
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from scipy.stats import entropy
from scipy.signal import welch
import itertools
from scipy.signal import find_peaks_cwt
from statsmodels.tsa.stattools import pacf
from antropy import perm_entropy
from antropy import sample_entropy
from tsfresh.feature_extraction.feature_calculators import spkt_welch_density
from pathlib import Path


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
    sma: Optional[List[int]]
    ema: Optional[List[int]]
    rsi: Optional[List[int]]
    macd: bool
    macd_signal: bool
    macd_histogram: bool
    obv: bool
    roc: Optional[List[int]]
    volatility: Optional[List[int]]
    volume_sma: Optional[List[int]]
    volume_std: Optional[List[int]]
    vroc: Optional[List[int]]
    price_gap: Optional[List[int]]
    price_vs_sma: Optional[List[int]]
    momentum_change: bool
    ultimate: bool
    awesome: bool
    max_return: Optional[List[int]]
    cmo: Optional[List[int]]
    trix: Optional[List[int]]
    atr: bool
    plus_di: bool
    minus_di: bool
    bb: bool
    ulcer: bool
    mean_price_volatility: Optional[List[int]]
    force_index: bool
    mfi: bool
    mass_index: bool
    cci: Optional[List[int]]
    stc: bool
    amih_l: bool
    kyle_l: bool
    corwin_schultz: bool
    approximate_entropy: bool
    adfuller: bool
    binned_entropy: bool
    cid_ce: bool
    count_above_mean: bool
    count_below_mean: bool
    energy_ratio_chunks: bool
    fft_aggregated: bool
    first_location_maximum: bool
    first_location_minimum: bool
    fourier_entropy: bool
    index_mass_quantile: Optional[float]
    kurtosis: bool
    last_location_of_maximum: bool
    lempel_ziv_complexity: bool
    linear_trend_timewise: bool
    longest_strike_above_mean: bool
    longest_strike_below_mean: bool
    mean_change: bool
    mean_abs_change: bool
    mean_second_derivative_central: bool
    number_cwt_peaks: bool
    permutation_entropy: bool
    sample_entropy: bool
    skewness: bool
    spkt_welch_density: bool
    time_reversal_asymmetry_statistic: bool
    variation_coefficient: bool
    turnover: Optional[List[int]]
    beta: Optional[List[int]]

    def __init__(self, df: pd.DataFrame,
                 sma: Optional[List[int]] = None, ema: Optional[List[int]] = None,
                 rsi: Optional[List[int]] = None, macd: bool = False, macd_signal: bool = False,
                 macd_histogram: bool = False, obv: bool = False, roc: Optional[List[int]] = None,
                 volatility: Optional[List[int]] = None, volume_sma: Optional[List[int]] = None,
                 volume_std: Optional[List[int]] = None, vroc: Optional[List[int]] = None,
                 price_gap: Optional[List[int]] = None,
                 price_vs_sma: Optional[List[int]] = None,
                 momentum_change: bool = False,
                 price_ratio=None,
                 ultimate: bool = False,  # TBC KN
                 awesome: bool = False,  # TBC KN
                 max_return: Optional[List[int]] = None,  # TBC KN
                 cmo: Optional[List[int]] = None,  # TBC KN
                 trix: Optional[List[int]] = None,  # TBC KN
                 atr: bool = False,  # TBC KN
                 plus_di: bool = False,  # TBC KN
                 minus_di: bool = False,  # TBC KN
                 bb: bool = False,  # TBC KN
                 ulcer: bool = False,  # TBC KN
                 mean_price_volatility: Optional[List[int]] = None,
                 force_index: bool = False,
                 mfi: bool = False,
                 mass_index: bool = False,
                 cci: Optional[List[int]] = None,
                 stc: bool = False,
                 amih_l: bool = False,
                 kyle_l: bool = False,
                 corwin_schultz: bool = False,
                 approximate_entropy: bool = False,
                 adfuller: bool = False,
                 binned_entropy: bool = False,
                 cid_ce: bool = False,
                 count_above_mean: bool = False,
                 count_below_mean: bool = False,
                 energy_ratio_chunks: bool = False,
                 fft_aggregated: bool = False,
                 first_location_maximum: bool = False,
                 first_location_minimum: bool = False,
                 fourier_entropy: bool = False,
                 index_mass_quantile: Optional[float] = None,
                 kurtosis: bool = False,
                 last_location_of_maximum: bool = False,
                 lempel_ziv_complexity: bool = False,
                 linear_trend_timewise: bool = False,
                 longest_strike_above_mean: bool = False,
                 longest_strike_below_mean: bool = False,
                 mean_change: bool = False,
                 mean_abs_change: bool = False,
                 mean_second_derivative_central: bool = False,
                 number_cwt_peaks: bool = False,
                 permutation_entropy: bool = False,
                 sample_entropy: bool = False,
                 skewness: bool = False,
                 spkt_welch_density: bool = False,
                 time_reversal_asymmetry_statistic: bool = False,
                 variation_coefficient: bool = False,
                 autocorrelation: Optional[List[int]] = None,
                 agg_autocorrelation: Optional[List[int]] = None,
                 ar_coefficient: Optional[List[int]] = None,
                 partial_autocorrelation: Optional[List[int]] = None,
                 turnover: Optional[List[int]] = None,
                 adfuller_window: int = 50,
                 beta: Optional[List[int]] = None):
        """
        :param df: DataFrame containing covariates
        :param sma: List of window sizes for Simple Moving Average
        :param ema: List of window sizes for Exponential Moving Average
        :param rsi: List of window sizes for Relative Strength Index
        :param macd: Calculate MACD (12-ema - 26-ema). Required for signal/histogram
        :param macd_signal: Calculate MACD signal line (9-ema of MACD)
        :param macd_histogram: Calculate MACD histogram (MACD - signal)
        :param obv: Calculate On-Balance Volume. Requires 'VOLUME,'
        :param roc: List of periods for Price Rate of Change
        :param volatility: List of window sizes for rolling standard deviation of daily returns
        :param volume_sma: List of window sizes for Simple Moving Average of Volume. Requires 'VOLUME,'
        :param volume_std: List of window sizes for Rolling Standard Deviation of Volume. Requires 'VOLUME'
        :param vroc: List of periods for Volume Rate of Change. Requires 'VOLUME'
        :param price_gap: List of window sizes for Price - SMA(window)
        :param price_vs_sma: List of window sizes for Price / SMA(window)
        :param turnover: List of window sizes for Turnover calculation (Placeholder - Requires definition)
        :param beta: List of window sizes for Beta calculation
        :param momentum_change: Calculate momentum change ROC6m - ROC6mp
        :param ultimate_oscillator: Calculate Ultimate Oscillator ((uses BP/TR over 7/14/28 days)) #TBC KN
        :param awesome: Calculate Awesome Oscillator (uses 5/34 EMA) #TBC KN
        :param max_return: List of window sizes (in days) to compute the maximum return over that rolling period #TBC KN
        :param cmo: List of window sizes for Chande Momentum Oscillator (CMO) #TBC KN
        :param trix: List of window sizes for TRIX (Triple Exponential Average) indicator #TBC KN
        :param atr: Calculate Average True Range (ATR, default window = 14) #TBC KN
        :param plus_di: Calculate Positive Directional Indicator (Plus_DI) using ATR and directional movement #TBC KN
        :param minus_di: Calculate Negative Directional Indicator (Minus_DI) using ATR and directional movement #TBC KN
        :param bb: Calculate Bollinger Bands (UB, LB, BB position) using typical price over n=20 days #TBC KN
        :param ulcer: Calculate Ulcer Index (measures downside volatility) #TBC KN
        :param force_index: Calculate Force Index (Price change × Volume)
        :param mean_price_volatility: Calculate Mean Price Volatility (EMA of TP standard deviation over 21d and 252d)
        :param mfi: Calculate Money Flow Index (uses typical price, volume, and positive/negative money flow)
        :param mass_index: Calculate Mass Index (uses ratio of 9-period EMA of high-low difference and its EMA)
        :param cci: List of window sizes for Commodity Channel Index (CCI), requires High, Low, Close
        :param stc: Calculate Schaff Trend Cycle (STC) using double EMA smoothing and cycle logic
        :param amih_l: Calculate Amihoud Illiquidity (abs(returns) / dollar volume, smoothed with EMA)
        :param kyle_l: Calculate Kyle’s Lambda (rolling regression of return on signed dollar volume)
        :param corwin_schultz: Calculate Corwin-Schultz Bid-Ask Spread Estimator. Requires 'High' and 'Low'
        :param autocorrelation: List of lags for calculating simple autocorrelation (e.g., [1, 5, 10])
        :param agg_autocorrelation: List of lags for calculating aggregate autocorrelations using aggregation functions (mean, std, median)
        :param approximate_entropy: Calculate Approximate Entropy (ApEn) using default m=2, r=0.2
        :param ar_coefficient: Calculate AR(k) coefficients using maximum likelihood estimation. Provide lag k (e.g., 10)
        :param adfuller: Calculate Augmented Dickey-Fuller test statistic on 'PX_LAST' column.
        :param binned_entropy: Calculate Binned Entropy (BE) on the 'PX_LAST' column.
        :param cid_ce: Calculate time series complexity (CID) using the 'PX_LAST' column.
        :param count_above_mean: Calculate number of values above the mean (Count_above_mean)
        :param count_below_mean: Calculate number of values below the mean (Count_below_mean)
        :param energy_ratio_chunks: Calculate energy ratio by chunks (Energy_ratio_by_chunks)
        :param fft_aggregated: Whether to calculate spectral centroid (mean), variance, skew, and kurtosis of FFT absolute spectrum.
        :param first_location_maximum: Relative position of first maximum in PX_LAST window
        :param first_location_minimum: Relative position of first minimum in PX_LAST window
        :param fourier_entropy: Compute Welch power spectral density entropy (Fourier entropy)
        :param index_mass_quantile: Float between 0 and 1, quantile of mass center to compute (e.g., 0.5 = center of mass)
        :param kurtosis: Calculate kurtosis (G2) of the time series
        :param last_location_of_maximum: Calculate the last relative index of the maximum value in the time series
        :param lempel_ziv_complexity: Calculate the Lempel-Ziv complexity of the time series
        :param linear_trend_timewise: Calculate linear trend components (p-value, correlation, intercept, slope, stderr)
        :param longest_strike_above_mean: Calculate the length of the longest consecutive subsequence above the mean
        :param longest_strike_below_mean: Calculate the length of the longest consecutive subsequence below the mean.
        :param mean_change: Calculate the average of first differences over the rolling window.
        :param mean_abs_change: Calculate the average of absolute first differences over the rolling window.
        :param mean_second_derivative_central: Calculate the average central approximation of the second derivative.
        :param number_cwt_peaks: Compute number of peaks in CWT-smoothed signal using ricker wavelet.
        :param partial_autocorrelation: List of lags at which to compute the partial autocorrelation
        :param permutation_entropy: Calculate the permutation entropy as a measure of time series complexity.
        :param sample_entropy: Calculate the sample entropy to assess time series complexity.
        :param skewness: Calculate the sample skewness (G1) of the time series.
        :param spkt_welch_density: Calculate spectral Welch density over rolling windows using tsfresh.
        :param time_reversal_asymmetry_statistic: Whether to calculate the time reversal asymmetry statistic (TRAS) from the time series.
        :param variation_coefficient: Whether to calculate the variation coefficient (standard deviation / mean) from the time series.
        :param adfuller_window: Integer rolling window used to compute the ADF statistic.
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
        self.momentum_change = momentum_change
        self.ultimate = ultimate
        self.awesome = awesome
        self.max_return = max_return
        self.cmo = cmo
        self.trix = trix
        self.atr = atr
        self.plus_di = plus_di
        self.minus_di = minus_di
        self.bb = bb
        self.ulcer = ulcer
        self.mean_price_volatility = mean_price_volatility
        self.force_index = force_index
        self.mfi = mfi
        self.mass_index = mass_index
        self.cci = cci
        self.stc = stc
        self.amih_l = amih_l
        self.kyle_l = kyle_l
        self.corwin_schultz = corwin_schultz
        self.adfuller_window = adfuller_window
        self.approximate_entropy = approximate_entropy
        self.adfuller = adfuller
        self.binned_entropy = binned_entropy
        self.cid_ce = cid_ce
        self.count_above_mean = count_above_mean
        self.count_below_mean = count_below_mean
        self.energy_ratio_chunks = energy_ratio_chunks
        self.fft_aggregated = fft_aggregated
        self.first_location_maximum = first_location_maximum
        self.first_location_minimum = first_location_minimum
        self.fourier_entropy = fourier_entropy
        self.index_mass_quantile = index_mass_quantile
        self.kurtosis = kurtosis
        self.last_location_of_maximum = last_location_of_maximum
        self.lempel_ziv_complexity = lempel_ziv_complexity
        self.linear_trend_timewise = linear_trend_timewise
        self.longest_strike_above_mean = longest_strike_above_mean
        self.longest_strike_below_mean = longest_strike_below_mean
        self.mean_change = mean_change
        self.mean_abs_change = mean_abs_change
        self.mean_second_derivative_central = mean_second_derivative_central
        self.number_cwt_peaks = number_cwt_peaks
        self.permutation_entropy = permutation_entropy
        self.sample_entropy = sample_entropy
        self.skewness = skewness
        self.spkt_welch_density = spkt_welch_density
        self.time_reversal_asymmetry_statistic = time_reversal_asymmetry_statistic
        self.variation_coefficient = variation_coefficient
        self.price_ratio = price_ratio
        self.required_base = set()

        self.required_base.update(['PX_LAST'])
        if any([self.obv, self.volume_sma, self.volume_std, self.vroc]):
            self.required_base.add('PX_VOLUME')


        missing_base = [col for col in self.required_base if col not in self.df.columns]
        print('THIS IS COLUMNS', self.df.columns)
        if missing_base:
            raise ValueError(f"Missing required base columns in DataFrame: {missing_base}")
        self.turnover = turnover
        self.beta = beta

    def group_transform(self, group_df: pd.DataFrame) -> pd.DataFrame:
        ticker = group_df.name
        print(f"\nSTART group_transform for {ticker}")

        transform_methods = [
            self._add_sma,
            self._add_ema,
            self._add_rsi,
            self._add_macd,
            self._add_obv,
            self._add_roc,
            self._add_cmo,
            self._add_trix,
            self._add_atr,
            self._add_mfi,
            self._add_force_index,
            self._add_stc,
            self._add_bb,
            self._add_ultimate,
            self._add_awesome,
            self._add_plus_di,
            self._add_minus_di,
            self._add_max_return,
            self._add_price_gap,
            self._add_price_vs_sma,
            self._add_momentum_change,
            self._add_ulcer,
            self._add_mean_price_volatility,
            self._add_approximate_entropy,
            self._add_adfuller,
            self._add_binned_entropy,
            self._add_cid_ce,
            self._add_count_above_mean,
            self._add_count_below_mean,
            self._add_energy_ratio_chunks,
            self._add_fft_aggregated,
            self._add_first_location_maximum,
            self._add_first_location_minimum,
            self._add_fourier_entropy,
            self._add_index_mass_quantile,
            self._add_kurtosis,
            self._add_last_location_of_maximum,
            self._add_lempel_ziv_complexity,
            self._add_linear_trend_timewise,
            self._add_longest_strike_above_mean,
            self._add_longest_strike_below_mean,
            self._add_mean_change,
            self._add_mean_abs_change,
            self._add_mean_second_derivative_central,
            self._add_number_cwt_peaks,
            self._add_permutation_entropy,
            self._add_sample_entropy,
            self._add_skewness,
            self._add_spkt_welch_density,
            self._add_time_reversal_asymmetry_statistic,
            self._add_variation_coefficient,
            self._add_volatility,
            self._add_volume_sma,
            self._add_volume_std,
            self._add_vroc,
            self._add_turnover,
            self._add_beta,
            self._add_mass_index,
            self._add_cci,
            self._add_amih_l,
            self._add_kyle_l,
            self._add_corwin_schultz,
        ]

        try:
            group_df = group_df.sort_index()

            for method in transform_methods:
                print(f"Applying {method.__name__}")
                group_df = method(group_df)

            print(f"END group_transform for {ticker}")
            return group_df

        except Exception as e:
            print(f"[ERROR] group_transform failed for ticker: {ticker} in {method.__name__} with error: {e}")
            raise

    def transform(self) -> pd.DataFrame:
        transformed_df = self.df.groupby('ticker', group_keys=True, observed=False).apply(self.group_transform)
        return transformed_df.sort_index()

    def _add_sma(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.sma is not None:
            for window in self.sma:
                group_df[f'sma_{window}'] = group_df['PX_LAST'].rolling(window=window, min_periods=window).mean()
        return group_df
    
    def _add_vol_accrelaration(self, group_df: pd.DataFrame) -> pd.DataFrame:
        self.sma_vol = [10, 20, 40, 60]
        self._add_sma_vol(group_df)
        group_df['vol_acc_10_20'] = group_df['sma_vol_10'] / group_df['sma_vol_20']
        group_df['vol_acc_10_20'] = group_df['sma_vol_10'] / group_df['sma_vol_20']
        group_df['vol_acc_10_40'] = group_df['sma_vol_10'] / group_df['sma_vol_40']
        group_df['vol_acc_10_60'] = group_df['sma_vol_10'] / group_df['sma_vol_60']
        group_df['vol_acc_20_40'] = group_df['sma_vol_20'] / group_df['sma_vol_40']
        group_df['vol_acc_20_60'] = group_df['sma_vol_20'] / group_df['sma_vol_60']
        group_df['vol_acc_40_60'] = group_df['sma_vol_40'] / group_df['sma_vol_60']
        return group_df
    

    def _add_ema(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.ema is not None:
            for window in self.ema:
                group_df[f'ema_{window}'] = group_df['PX_LAST'].ewm(span=window, adjust=False,
                                                                    min_periods=window).mean()
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

    price_ratio = [9]

    def _add_price_ratio(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.price_ratio is not None:
            for n in self.price_ratio:
                close_0 = group_df['PX_LAST'].shift(n)
                close_n = group_df['PX_LAST']
                group_df[f'momentum_{n}'] = close_0 / close_n
                group_df[f'reversal_{n}'] = close_n / close_0
        return group_df

    def _add_obv(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.obv:
            if 'PX_VOLUME' not in group_df.columns:
                warnings.warn(f"Skipping OBV for group: 'PX_VOLUME' column not found.", UserWarning)
                return group_df
            price_diff = group_df['PX_LAST'].diff()
            volume = group_df['PX_VOLUME']
            signed_volume = pd.Series(np.sign(price_diff) * volume).fillna(0)
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
                group_df[f'roc_{window}'] = ((group_df['PX_LAST'] / shifted_price) - 1).replace([np.inf, -np.inf],
                                                                                                np.nan) * 100
        return group_df

    def _add_momentum_change(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.momentum_change:
            if 'roc_126' not in group_df.columns:
                shifted_price = group_df['PX_LAST'].shift(126)
                group_df[f'roc_126'] = ((group_df['PX_LAST'] / shifted_price) - 1).replace([np.inf, -np.inf],
                                                                                           np.nan) * 100
            group_df['momentum_change'] = group_df['roc_126'].diff(126)
        return group_df

    def _add_volatility(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.volatility is not None:
            daily_return = group_df['PX_LAST'].pct_change(fill_method=None)
            for window in self.volatility:
                group_df[f'volatility_{window}'] = daily_return.rolling(window=window, min_periods=window).std()
        return group_df

    def _add_volume_sma(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.volume_sma:
            if 'PX_VOLUME' not in group_df.columns:
                warnings.warn(f"Skipping Volume SMA for group: 'PX_VOLUME' column not found.", UserWarning)
                return group_df
            for window in self.volume_sma:
                group_df[f'volume_sma_{window}'] = group_df['PX_VOLUME'].rolling(window=window,
                                                                                 min_periods=window).mean()
        return group_df

    def _add_volume_std(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.volume_std is not None:
            if 'PX_VOLUME' not in group_df.columns:
                warnings.warn(f"Skipping Volume StDev for group: 'PX_VOLUME' column not found.", UserWarning)
                return group_df
            for window in self.volume_std:
                group_df[f'volume_std_{window}'] = group_df['PX_VOLUME'].rolling(window=window,
                                                                                 min_periods=window).std()
        return group_df

    def _add_vroc(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.vroc is not None:
            if 'PX_VOLUME' not in group_df.columns:
                warnings.warn(f"Skipping VROC for group: 'PX_VOLUME' column not found.", UserWarning)
                return group_df
            for window in self.vroc:
                shifted_volume = group_df['PX_VOLUME'].shift(window)
                group_df[f'vroc_{window}'] = ((group_df['PX_VOLUME'] / shifted_volume) - 1).replace([np.inf, -np.inf],
                                                                                                    np.nan) * 100
        return group_df

    def _add_turnover(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.turnover is not None:
            for window in self.turnover:
                group_df[f'turnover_{window}'] = group_df['TURNOVER'].rolling(window=window).mean()
        return group_df

    def _add_beta(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.beta:
            raise NotImplementedError
        return group_df

    def _add_ultimate(self, group_df: pd.DataFrame) -> pd.DataFrame:  # TBC KN
        if self.ultimate:
            required_cols = ['PX_LOW', 'PX_HIGH', 'PX_LAST']
            if not all(col in group_df.columns for col in required_cols):
                warnings.warn("Skipping Ultimate Oscillator: required columns missing (High, Low, PX_LAST)",
                              UserWarning)
                return group_df

            close = group_df['PX_LAST']
            high = group_df['PX_HIGH']
            low = group_df['PX_LOW']
            prior_close = close.shift(1)

            bp = pd.Series(close - np.minimum(low, prior_close), index=group_df.index)
            tr = pd.Series(np.maximum(high, prior_close) - np.minimum(low, prior_close), index=group_df.index)

            a1 = bp.rolling(7).sum() / tr.rolling(7).sum()
            a2 = bp.rolling(14).sum() / tr.rolling(14).sum()
            a3 = bp.rolling(28).sum() / tr.rolling(28).sum()

            group_df['ultimate'] = ((4 * a1) + (2 * a2) + a3) / 7

        return group_df

    def _add_awesome(self, group_df: pd.DataFrame) -> pd.DataFrame:  # TBC KN
        if self.awesome:
            if 'PX_HIGH' not in group_df.columns or 'PX_LOW' not in group_df.columns:
                warnings.warn("Skipping Awesome Oscillator: 'High' or 'Low' column not found.", UserWarning)
                return group_df

            median_price = (group_df['PX_HIGH'] + group_df['PX_LOW']) / 2
            short_ma = median_price.rolling(window=5, min_periods=5).mean()
            long_ma = median_price.rolling(window=34, min_periods=34).mean()

            group_df['awesome_oscillator'] = short_ma - long_ma
        return group_df

    def _add_max_return(self, group_df: pd.DataFrame) -> pd.DataFrame:  # TBC KN
        if self.max_return is not None:
            for window in self.max_return:
                max_ret = (
                        group_df['PX_LAST'] / group_df['PX_LAST'].shift(1).rolling(window=window).min() - 1
                ).replace([np.inf, -np.inf], np.nan)
                group_df[f'max_return_{window}'] = max_ret
        return group_df

    def _add_cmo(self, group_df: pd.DataFrame) -> pd.DataFrame:  # TBC KN
        if self.cmo is not None:
            close = group_df['PX_LAST']
            delta = close.diff()

            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)
            for window in self.cmo:
                sum_gains = gains.rolling(window).sum()
                sum_losses = losses.rolling(window).sum()

                cmo = ((sum_gains - sum_losses) / (sum_gains + sum_losses)).replace([np.inf, -np.inf], np.nan) * 100
                group_df[f'cmo_{window}'] = cmo

        return group_df

    def _add_trix(self, group_df: pd.DataFrame) -> pd.DataFrame:  # TBC KN
        if self.trix:
            close = group_df['PX_LAST']
            n = 21

            ema1 = close.ewm(span=n, adjust=False, min_periods=n).mean()
            ema2 = ema1.ewm(span=n, adjust=False, min_periods=n).mean()
            ema3 = ema2.ewm(span=n, adjust=False, min_periods=n).mean()
            ema3_prev = ema3.shift(1)

            trix = ((ema3 - ema3_prev) / ema3_prev).replace([np.inf, -np.inf], np.nan)
            group_df['trix'] = trix

        return group_df

    def _add_atr(self, group_df: pd.DataFrame) -> pd.DataFrame:  # TBC KN
        if self.atr:
            high = group_df['PX_HIGH']
            low = group_df['PX_LOW']
            close = group_df['PX_LAST']
            prior_close = close.shift(1)
            n = 14
            tr = pd.concat([
                (high - low),
                (high - prior_close).abs(),
                (low - prior_close).abs()
            ], axis=1).max(axis=1)
            atr = pd.Series(index=group_df.index, dtype='float64')
            atr.iloc[n - 1] = tr.iloc[:n].mean()
            for i in range(n, len(tr)):
                atr.iloc[i] = ((atr.iloc[i - 1] * (n - 1)) + tr.iloc[i]) / n
            group_df['atr'] = atr

        return group_df

    def _add_plus_di(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.plus_di:
            required_cols = ['PX_HIGH', 'PX_LOW', 'PX_LAST']
            if not all(col in group_df.columns for col in required_cols):
                warnings.warn("Skipping Plus_DI: required columns missing (PX_HIGH, PX_LOW, PX_LAST)", UserWarning)
                return group_df

            n = 21
            high = group_df['PX_HIGH']
            low = group_df['PX_LOW']
            close = group_df['PX_LAST']

            prev_high = high.shift(1)
            prev_low = low.shift(1)
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)

            atr = pd.Series(index=group_df.index, dtype='float64')
            atr.iloc[n - 1] = tr.iloc[:n].mean()
            for i in range(n, len(tr)):
                atr.iloc[i] = ((atr.iloc[i - 1] * (n - 1)) + tr.iloc[i]) / n
            up_move = high - prev_high
            down_move = prev_low - low
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)

            dm_series = pd.Series(plus_dm, index=group_df.index)
            dm_sum = dm_series.rolling(n).sum()
            dm_mean = dm_series.rolling(n).mean()
            dm_smoothed = dm_sum - dm_mean + dm_series

            group_df['plus_di'] = (dm_smoothed / atr) * 100

        return group_df

    def _add_minus_di(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.minus_di:
            required_cols = ['PX_HIGH', 'PX_LOW', 'PX_LAST']
            if not all(col in group_df.columns for col in required_cols):
                warnings.warn("Skipping Minus_DI: required columns missing (High, Low, PX_LAST)", UserWarning)
                return group_df

            n = 21
            high = group_df['PX_HIGH']
            low = group_df['PX_LOW']
            close = group_df['PX_LAST']

            prev_high = high.shift(1)
            prev_low = low.shift(1)
            prev_close = close.shift(1)

            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)

            atr = pd.Series(index=group_df.index, dtype='float64')
            atr.iloc[n - 1] = tr.iloc[:n].mean()
            for i in range(n, len(tr)):
                atr.iloc[i] = ((atr.iloc[i - 1] * (n - 1)) + tr.iloc[i]) / n

            down_move = prev_low - low
            up_move = high - prev_high
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            dm_series = pd.Series(minus_dm, index=group_df.index)

            dm_sum = dm_series.rolling(n).sum()
            dm_mean = dm_series.rolling(n).mean()
            dm_smoothed = dm_sum - dm_mean + dm_series
            group_df['minus_di'] = (dm_smoothed / atr) * 100

        return group_df

    def _add_bb(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.bb:
            required_cols = ['PX_HIGH', 'PX_LOW', 'PX_LAST']
            if not all(col in group_df.columns for col in required_cols):
                warnings.warn("Skipping Bollinger Bands: required columns missing (PX_HIGH, PX_LOW, PX_LAST)",
                              UserWarning)
                return group_df

            n = 20
            high = group_df['PX_HIGH']
            low = group_df['PX_LOW']
            close = group_df['PX_LAST']

            tp = (high + low + close) / 3
            tp_mean = tp.rolling(window=n).mean()
            tp_std = tp.rolling(window=n).std()

            upper_band = tp_mean + 2 * tp_std
            lower_band = tp_mean - 2 * tp_std

            group_df['bb_upper'] = upper_band
            group_df['bb_lower'] = lower_band
            group_df['bb_position'] = (close - lower_band) / (upper_band - lower_band).replace(0, np.nan)

        return group_df

    def _add_ulcer(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.ulcer:
            return group_df

        required_cols = ['PX_HIGH', 'PX_LAST']
        if not all(col in group_df.columns for col in required_cols):
            warnings.warn("Skipping Ulcer Index: required columns missing ('PX_HIGH', PX_LAST)", UserWarning)
            return group_df

        n = 21
        close = group_df['PX_LAST']
        max_high = group_df['PX_HIGH'].rolling(window=n, min_periods=n).max()
        pd_val = ((close - max_high) / max_high) * 100

        pd_squared = pd_val.pow(2)
        ulcer_index = pd_squared.rolling(window=n, min_periods=n).mean().pow(0.5)

        group_df['ulcer_index'] = ulcer_index
        return group_df

    def _add_mean_price_volatility(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_price_volatility is not None:
            if not all(col in group_df.columns for col in ['PX_LAST', 'PX_HIGH', 'PX_LOW']):
                warnings.warn("Skipping Mean Price Volatility: Missing required columns (PX_LAST, PX_HIGH, PX_LOW)",
                              UserWarning)
                return group_df

            tp = (group_df['PX_HIGH'] + group_df['PX_LOW'] + group_df['PX_LAST']) / 3

            for window in self.mean_price_volatility:
                std = tp.rolling(window=window, min_periods=window).std()
                ema_std = std.ewm(span=window, adjust=False, min_periods=window).mean()
                group_df[f'mean_std_{window}'] = ema_std

        return group_df

    def _add_force_index(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.force_index:
            if 'PX_VOLUME' not in group_df.columns:
                warnings.warn("Skipping Force Index: 'PX_VOLUME' column not found.", UserWarning)
                return group_df

            close = group_df['PX_LAST']
            prior_close = close.shift(1)
            volume = group_df['PX_VOLUME']

            group_df['force_index'] = (close - prior_close) * volume
        return group_df

    def _add_mfi(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.mfi:
            if not all(col in group_df.columns for col in ['PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']):
                warnings.warn("Skipping MFI: required columns missing (High, Low, PX_LAST, PX_VOLUME)", UserWarning)
                return group_df

            typical_price = (group_df['PX_HIGH'] + group_df['PX_LOW'] + group_df['PX_LAST']) / 3
            raw_money_flow = typical_price * group_df['PX_VOLUME']
            tp_diff = tp_diff = pd.to_numeric(typical_price.diff(), errors="coerce")

            positive_flow = raw_money_flow.where(tp_diff > 0, 0)
            negative_flow = raw_money_flow.where(tp_diff < 0, 0)

            pos_flow_sum = positive_flow.rolling(window=21).sum()
            neg_flow_sum = negative_flow.rolling(window=21).sum()

            mfr = pos_flow_sum / neg_flow_sum.replace(0, np.nan)
            group_df['mfi_21'] = 100 - (100 / (1 + mfr))
            group_df['mfi_21'] = group_df['mfi_21'].replace([np.inf, -np.inf], np.nan)

        return group_df

    def _add_mass_index(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.mass_index:
            if 'PX_HIGH' not in group_df.columns or 'PX_LOW' not in group_df.columns:
                warnings.warn("Skipping Mass Index: 'PX_HIGH' or 'PX_LOW' column not found.", UserWarning)
                return group_df

            high = group_df['PX_HIGH']
            low = group_df['PX_LOW']
            diff = high - low

            ema1 = diff.ewm(span=9, adjust=False, min_periods=9).mean()
            ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()

            mass = ema1 / ema2
            group_df['mass_index'] = mass.rolling(window=25, min_periods=25).sum()

        return group_df

    def _add_cci(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.cci is not None:
            if not all(col in group_df.columns for col in ['PX_LAST', 'PX_HIGH', 'PX_LOW']):
                warnings.warn("Skipping CCI: required columns missing (High, Low, PX_LAST)", UserWarning)
                return group_df

            typical_price = (group_df['PX_HIGH'] + group_df['PX_LOW'] + group_df['PX_LAST']) / 3

            for window in self.cci:
                ma = typical_price.rolling(window=window, min_periods=window).mean()
                md = (typical_price - ma).abs().rolling(window=window, min_periods=window).mean()
                cci = (typical_price - ma) / (0.015 * md)
                group_df[f'cci_{window}'] = cci.replace([np.inf, -np.inf], np.nan)

        return group_df

    def _add_stc(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.stc:
            c = group_df['PX_LAST']
            n1, n2, n3, n4 = 23, 50, 3, 10

            ema1 = c.ewm(span=n1, adjust=False).mean()
            ema2 = ema1.ewm(span=n2, adjust=False).mean()
            ema_diff = ema1 - ema2

            ema_diff_min = ema_diff.rolling(window=n4).min()
            ema_diff_max = ema_diff.rolling(window=n4).max()

            s = 100 * (ema_diff - ema_diff_min) / (ema_diff_max - ema_diff_min)
            s = s.replace([np.inf, -np.inf], np.nan)

            ema_s = s.ewm(span=n3, adjust=False).mean()
            ema_s_min = ema_s.rolling(window=n4).min()
            ema_s_max = ema_s.rolling(window=n4).max()

            d = 100 * (ema_s - ema_s_min) / (ema_s_max - ema_s_min)
            d = d.replace([np.inf, -np.inf], np.nan)

            stc = d.ewm(span=n3, adjust=False).mean()
            group_df['stc'] = stc

        return group_df

    def _add_amih_l(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.amih_l:
            required_cols = ['PX_LAST', 'PX_VOLUME']
            if not all(col in group_df.columns for col in required_cols):
                warnings.warn("Skipping Amihoud Illiquidity: required columns missing (PX_LAST, PX_VOLUME)",
                              UserWarning)
                return group_df

            close = group_df['PX_LAST']
            prior_close = close.shift(1)
            returns = ((close / prior_close) - 1).replace([np.inf, -np.inf], np.nan)

            dollar_volume = close * group_df['PX_VOLUME']

            ema1 = dollar_volume.ewm(span=21, adjust=False, min_periods=21).mean()
            ema2 = returns.ewm(span=21, adjust=False, min_periods=21).mean()

            group_df['amih_l'] = (np.abs(ema2) / ema1) * 1_000_000

        return group_df

    def _rolling_beta(self, y: pd.Series, x: pd.Series, window: int) -> pd.Series:
        betas = []
        for i in range(len(y)):
            if i < window - 1:
                betas.append(np.nan)
            else:
                yi = y[i - window + 1:i + 1]
                xi = x[i - window + 1:i + 1]
                if xi.std() == 0 or yi.std() == 0:
                    betas.append(np.nan)
                else:
                    beta = np.cov(yi, xi)[0, 1] / np.var(xi)
                    betas.append(beta)
        return pd.Series(betas, index=y.index)

    def _add_kyle_l(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.kyle_l:
            required_cols = ['PX_LAST', 'PX_VOLUME']
            if not all(col in group_df.columns for col in required_cols):
                warnings.warn("Skipping Kyle's Lambda: required columns missing (PX_LAST, PX_VOLUME)", UserWarning)
                return group_df

            close = group_df['PX_LAST']
            volume = group_df['PX_VOLUME']
            prior_close = close.shift(1)
            returns = ((close / prior_close) - 1).replace([np.inf, -np.inf], np.nan)

            sign_r = returns.apply(lambda r: 1 if r > 0 else (-1 if r < 0 else 0))
            dollar_vol = close * volume
            vd = sign_r * np.log(dollar_vol.replace(0, np.nan))

            group_df['kyle_l'] = self._rolling_beta(returns, vd, window=21)

        return group_df

    def _add_corwin_schultz(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.corwin_schultz:
            return group_df

        required_cols = ['PX_HIGH', 'PX_LOW']
        if not all(col in group_df.columns for col in required_cols):
            warnings.warn("Skipping Corwin-Schultz: required columns missing (PX_HIGH, PX_LOW)", UserWarning)
            return group_df

        n1 = 5
        n2 = 21
        high: pd.Series = group_df['PX_HIGH']
        low: pd.Series = group_df['PX_LOW']

        hl_sq = np.log(high / low) ** 2

        h_max = high.rolling(window=n1).max()
        l_min = low.rolling(window=n1).min()
        hl_sq = pd.Series(np.log(high / low) ** 2, index=group_df.index)
        hl_sq_sum = hl_sq.rolling(window=n1).sum()
        b = hl_sq_sum.rolling(window=n2).mean()

        g = np.log(h_max / l_min) ** 2
        c = (np.sqrt(2) - 1) / (3 - 2 * np.sqrt(2))

        A_tilde = ((np.sqrt(2) - 1) * b - (g / 2)) / c
        A = A_tilde.where(A_tilde >= 0, 0)

        cs = 2 * (np.exp(A) - 1) / (1 + np.exp(A))
        group_df['corwin_schultz'] = cs

        return group_df

    def _add_approximate_entropy(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.approximate_entropy:
            return group_df

        group_df['approx_entropy'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: approximate_entropy(x, 2, 0.2), raw=False
        )

        return group_df

    def _add_adfuller(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.adfuller:
            return group_df

        def adf_stat(series):
            try:
                if series.nunique() <= 1 or series.isna().all():
                    return np.nan
                result = adfuller(series.dropna(), autolag='AIC')
                return result[1]
            except Exception as e:
                print(f"[ADF ERROR] {series.name}: {e}")
                return np.nan

        group_df["adfuller"] = (
            group_df["PX_LAST"]
            .rolling(self.adfuller_window, min_periods=self.adfuller_window)
            .apply(adf_stat, raw=False)
        )
        return group_df

    def _add_binned_entropy(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.binned_entropy:
            return group_df

        def safe_binned_entropy(x):
            try:
                hist, _ = np.histogram(x, bins=10, density=True)
                hist = hist[hist > 0]  # Avoid log(0)
                return entropy(hist)
            except Exception:
                return np.nan

        group_df['binned_entropy'] = group_df['PX_LAST'].rolling(window=10, min_periods=10).apply(
            safe_binned_entropy, raw=False
        )
        return group_df

    def _add_cid_ce(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.cid_ce:
            return group_df

        group_df['cid_ce'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: np.sqrt(np.sum(np.diff(pd.Series(x).dropna().to_numpy()) ** 2))
            if len(pd.Series(x).dropna()) >= 2 else np.nan,
            raw=False
        )
        return group_df

    def _add_count_above_mean(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.count_above_mean:
            return group_df

        group_df['count_above_mean'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: np.sum(x > np.mean(x)) if len(x.dropna()) > 0 else np.nan,
            raw=False
        )
        return group_df

    def _add_count_below_mean(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.count_below_mean:
            return group_df

        group_df['count_below_mean'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: np.sum(x < np.mean(x)) if len(x.dropna()) > 0 else np.nan,
            raw=False
        )
        return group_df

    def _add_energy_ratio_chunks(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.energy_ratio_chunks:
            return group_df

        group_df['energy_ratio_by_chunks'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                np.sum(x.dropna().to_numpy()[:len(x.dropna()) // 2] ** 2) /
                np.sum(x.dropna().to_numpy() ** 2)
                if np.sum(x.dropna().to_numpy() ** 2) != 0 else np.nan
            ) if len(x.dropna()) > 0 else np.nan,
            raw=False
        )
        return group_df

    def _add_fft_aggregated(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fft_aggregated:
            return group_df

        stats = {
            'fft_mean': lambda x: np.mean(x),
            'fft_var': lambda x: np.var(x),
            'fft_skew': lambda x: pd.Series(x).skew(),
            'fft_kurt': lambda x: pd.Series(x).kurt()
        }

        for name, func in stats.items():
            group_df[name] = group_df['PX_LAST'].rolling(window=50).apply(
                lambda x: func(np.abs(np.fft.fft(x.dropna().to_numpy())))
                if len(x.dropna()) > 0 else np.nan,
                raw=False
            )

        return group_df

    def _add_first_location_maximum(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.first_location_maximum:
            return group_df

        group_df['first_location_of_maximum'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: np.argmax(x.to_numpy()) / len(x.dropna()) if len(x.dropna()) > 0 else np.nan,
            raw=False
        )
        return group_df

    def _add_first_location_minimum(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.first_location_minimum:
            return group_df

        group_df['first_location_of_minimum'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: np.argmin(x.to_numpy()) / len(x.dropna()) if len(x.dropna()) > 0 else np.nan,
            raw=False
        )
        return group_df

    def _add_fourier_entropy(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fourier_entropy:
            return group_df

        group_df['fourier_entropy'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                entropy(welch(x.dropna(), nperseg=len(x.dropna()))[1]) if len(x.dropna()) > 0 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_index_mass_quantile(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if self.index_mass_quantile is None:
            return group_df

        q = self.index_mass_quantile

        group_df['index_mass_quantile'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                np.searchsorted(np.cumsum(np.abs(x.dropna())),
                                q * np.sum(np.abs(x.dropna()))) / len(x.dropna())
                if len(x.dropna()) > 0 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_kurtosis(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.kurtosis:
            return group_df

        group_df['kurtosis'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                (
                        ((len(x) - 1) / ((len(x) - 2) * (len(x) - 3))) *
                        ((len(x) + 1) * ((x - x.mean()) ** 4).mean() / ((x - x.mean()) ** 2).mean() ** 2 - 3 * (
                                    len(x) - 1))
                ) if len(x.dropna()) >= 4 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_last_location_of_maximum(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.last_location_of_maximum:
            return group_df

        group_df['last_location_of_maximum'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                np.where(x == x.max())[0][-1] / len(x)
                if len(x.dropna()) > 0 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_lempel_ziv_complexity(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.lempel_ziv_complexity:
            return group_df

        def lempel_ziv_complexity_bin(seq: str) -> int:
            i, l, k, complexity = 0, 1, 1, 1
            n = len(seq)
            while True:
                if seq[i:i + k] not in seq[0:i + k - 1]:
                    complexity += 1
                    i += k
                    k = 1
                    if i + k > n:
                        break
                else:
                    k += 1
                    if i + k > n:
                        break
            return complexity

        def complexity_rolling(x):
            s = x.dropna().to_numpy()
            if len(s) == 0:
                return np.nan
            median = np.median(s)
            bin_seq = ''.join(['1' if v > median else '0' for v in s])
            return lempel_ziv_complexity_bin(bin_seq)

        group_df['lempel_ziv_complexity'] = group_df['PX_LAST'].rolling(window=50).apply(
            complexity_rolling, raw=False
        )

        return group_df

    def _add_linear_trend_timewise(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.linear_trend_timewise:
            return group_df

        p_vals, corrs, intercepts, slopes, stderrs = [], [], [], [], []

        for i in range(len(group_df)):
            if i < 49:
                p_vals.append(np.nan)
                corrs.append(np.nan)
                intercepts.append(np.nan)
                slopes.append(np.nan)
                stderrs.append(np.nan)
                continue

            raw_window = group_df['PX_LAST'].iloc[i - 49:i + 1]
            window = raw_window.dropna().to_numpy()

            if len(window) < 2:
                p_vals.append(np.nan)
                corrs.append(np.nan)
                intercepts.append(np.nan)
                slopes.append(np.nan)
                stderrs.append(np.nan)
                continue

            t = np.arange(len(window))

            slope, intercept = np.polyfit(t, window, 1)
            y_pred = intercept + slope * t
            residuals = window - y_pred
            stderr = np.std(residuals)
            corr = np.corrcoef(t, window)[0, 1] if len(window) > 1 else np.nan
            p_val = 2 * (1 - abs(corr)) if not np.isnan(corr) else np.nan

            p_vals.append(p_val)
            corrs.append(corr)
            intercepts.append(intercept)
            slopes.append(slope)
            stderrs.append(stderr)

        result_df = pd.DataFrame({
            'lintrend_p': p_vals,
            'lintrend_corr': corrs,
            'lintrend_intercept': intercepts,
            'lintrend_slope': slopes,
            'lintrend_stderr': stderrs
        }, index=group_df.index)

        group_df = pd.concat([group_df, result_df], axis=1)

        return group_df

    def _add_longest_strike_above_mean(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.longest_strike_above_mean:
            return group_df

        group_df['longest_strike_above_mean'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                max(
                    (sum(1 for _ in g) for k, g in itertools.groupby((x > x.mean()).values) if k),
                    default=np.nan
                ) if len(x.dropna()) > 0 else np.nan
            ),
            raw=False
        )

        return group_df

    def _add_longest_strike_below_mean(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.longest_strike_below_mean:
            return group_df

        group_df['longest_strike_below_mean'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                max(
                    (sum(1 for _ in g) for k, g in itertools.groupby((x < x.mean()).values) if k),
                    default=np.nan
                ) if len(x.dropna()) > 0 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_mean_change(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.mean_change:
            return group_df

        group_df['mean_change'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                np.mean(np.diff(x.dropna())) if len(x.dropna()) > 1 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_mean_abs_change(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.mean_abs_change:
            return group_df

        group_df['mean_abs_change'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                np.mean(np.abs(np.diff(x.dropna()))) if len(x.dropna()) > 1 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_mean_second_derivative_central(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.mean_second_derivative_central:
            return group_df

        group_df['mean_second_derivative_central'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                np.mean(0.5 * (x.dropna()[2:] - 2 * x.dropna()[1:-1] + x.dropna()[:-2]))
                if len(x.dropna()) >= 3 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_number_cwt_peaks(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.number_cwt_peaks:
            return group_df

        group_df['number_cwt_peaks'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                len(find_peaks_cwt(x.dropna().to_numpy(), widths=np.arange(1, 10)))
                if len(x.dropna()) > 0 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_permutation_entropy(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.permutation_entropy:
            return group_df

        group_df['permutation_entropy'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: perm_entropy(x.dropna().to_numpy()) if len(x.dropna()) > 0 else np.nan,
            raw=False
        )

        return group_df

    def _add_sample_entropy(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.sample_entropy:
            return group_df

        group_df['sample_entropy'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: sample_entropy(x.dropna().to_numpy()) if len(x.dropna()) > 0 else np.nan,
            raw=False
        )

        return group_df

    def _add_skewness(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.skewness:
            return group_df

        group_df['skewness'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (
                ((np.sqrt(len(x)) * np.mean((x - x.mean()) ** 3)) /
                 (np.std(x, ddof=0) ** 3)) if len(x.dropna()) > 2 else np.nan
            ),
            raw=False
        )
        return group_df

    def _add_spkt_welch_density(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.spkt_welch_density:
            return group_df

        def safe_spkt(x):
            try:
                result = spkt_welch_density(x.dropna())
                return result[1] if len(result) > 1 else np.nan
            except Exception:
                return np.nan

        group_df['spkt_welch_density'] = group_df['PX_LAST'].rolling(window=50, min_periods=50).apply(
            safe_spkt, raw=False
        )

        return group_df

    def _add_time_reversal_asymmetry_statistic(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.time_reversal_asymmetry_statistic:
            return group_df

        def tras(x, lag=1):
            x = x.dropna().to_numpy()
            n = len(x)
            if n <= 2 * lag:
                return np.nan
            return np.mean((x[2 * lag:] ** 2 * x[:-2 * lag]) - (x[lag:-lag] * x[lag:-lag]))

        group_df['time_reversal_asymmetry_statistic'] = group_df['PX_LAST'].rolling(window=50).apply(
            tras, raw=False
        )

        return group_df

    def _add_variation_coefficient(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if not self.variation_coefficient:
            return group_df

        group_df['variation_coefficient'] = group_df['PX_LAST'].rolling(window=50).apply(
            lambda x: (np.std(x) / np.mean(x)) if np.mean(x) != 0 else np.nan,
            raw=False
        )
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
        transformed_df = self.df.groupby(level='ticker', group_keys=False, observed=False).apply(self.group_transform)
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
                        group_df[f'pe_band_{window}_{q_percent}'] = group_df[pe_col_name].rolling(window=window, min_periods=1).quantile(q_decimal, interpolation='linear')
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