import pandas as pd
import numpy as np
import yfinance as yf  # Added import for yfinance
from abc import ABC, abstractmethod  # Added imports for ABC
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


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
    def __init__(self, df: pd.DataFrame, root_path,  label_duration: int,
                 index_csv: str = "ihsg_may2025", price_column: str = 'PX_LAST',
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
        self.root_path = root_path
        self.label_duration = label_duration
        self.index_csv = index_csv
        self.price_column = price_column
        self.ticker_level_name = ticker_level_name
        self.date_level_name = date_level_name

        self.stock_return_col_name = f'{self.label_duration}d_stock_return'
        self.index_return_col_name = 'INDEX_RETURN'
        self.excess_return_col_name = f'excess_returns_{self.label_duration}'

    def _calculate_returns(self, series: pd.Series) -> pd.Series:
        """Calculates forward looking percentage returns over a future period."""
        future_price = series.shift(-self.label_duration)
        returns = ((future_price - series) / series) * 100
        return returns

    def _get_index_returns(self) -> pd.Series:
        self.ihsg = pd.read_csv(self.root_path / 'data' / f"{self.index_csv}.csv", index_col=0)
        self.ihsg.index = pd.to_datetime(self.ihsg.index)
        return self.ihsg

    def transform(self) -> pd.DataFrame:
        df_copy = self.df.copy()
        ihsg = self._get_index_returns()

        index_returns = (
            self._calculate_returns(ihsg['PX_LAST'])
            .rename(f'index_returns_{self.label_duration}')
        )

        eq_returns = (
            df_copy
            .groupby(level='ticker', group_keys=False, observed=False)['PX_LAST']
            .apply(self._calculate_returns)
            .rename(f'eq_returns_{self.label_duration}')
        )

        aligned_index = (
            eq_returns
            .index
            .get_level_values('Dates')
            .map(index_returns)
        )

        df_copy[eq_returns.name] = eq_returns
        df_copy[index_returns.name] = aligned_index.values
        df_copy[f'excess_returns_{self.label_duration}'] = (
            df_copy[eq_returns.name] - df_copy[index_returns.name]
        )

        return df_copy

class BinaryLabelTransform(ExcessReturnTransform):
    def __init__(self, df: pd.DataFrame, root_path,  label_duration: int, quantile: float,
                 index_csv: str = "ihsg_may2025", price_column: str = 'PX_LAST',
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
        super().__init__(df, root_path, label_duration, index_csv, price_column,
                         ticker_level_name, date_level_name)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("Quantile must be between 0.0 and 1.0.")
        self.quantile = quantile
        self.label_col_name = 'label'

    def _assign_label(self, group: pd.DataFrame) -> pd.Series:
        """
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
    
class KMRFLabelTransform:
    def __init__(self, price_column="PX_LAST", kama_n=10, gamma=0.5):
        self.price_column = price_column
        self.kama_n = kama_n
        self.gamma = gamma

    def calculate_kama(self, price, n=None, fast=2, slow=30):
        n = n or self.kama_n
        delta = price.diff()
        signal = abs(price - price.shift(n))
        noise = delta.abs().rolling(n).sum()
        er = signal / noise.replace(0, np.nan)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = [price.iloc[0]]
        for i in range(1, len(price)):
            k = sc.iloc[i] if not np.isnan(sc.iloc[i]) else slow_sc**2
            kama.append(kama[-1] + k * (price.iloc[i] - kama[-1]))
        return pd.Series(kama, index=price.index)

    def calculate_filter(self, kama, n):
        kama_diff = kama.diff()
        std = kama_diff.rolling(n).std()
        return self.gamma * std

    def compute_msr_state(self, price):
        returns = np.log(price / price.shift(1)).dropna()
        model = MarkovRegression(returns, k_regimes=2, trend='c', switching_variance=True)
        res = model.fit(disp=False)
        smoothed_probs = res.smoothed_marginal_probabilities[1]  # High-variance state
        state = (smoothed_probs > 0.5).astype(int)
        state.index = returns.index
        return state

    def assign_regimes(self, price: pd.Series) -> pd.Series:
        kama = self.calculate_kama(price)
        filt = self.calculate_filter(kama, self.kama_n)
        msr_state = self.compute_msr_state(price)

        kama_diff = kama - kama.rolling(self.kama_n).min()
        drop = kama.rolling(self.kama_n).min() - kama
        condition_up = kama_diff > filt
        condition_down = drop > filt

        condition_up = condition_up.reindex(price.index).fillna(False)
        condition_down = condition_down.reindex(price.index).fillna(False)
        msr_state = msr_state.reindex(price.index).fillna(0)

        regime = pd.Series(index=price.index, dtype='float')
        regime[(msr_state == 0) & condition_down] = 0  # LV bearish
        regime[(msr_state == 0) & condition_up] = 1    # LV bullish
        regime[(msr_state == 1) & condition_down] = 2  # HV bearish
        regime[(msr_state == 1) & condition_up] = 3    # HV bullish
        return regime
    
    def extend_labels(self, regime: pd.Series) -> pd.Series:
        label = pd.Series(index=regime.index, dtype='float').fillna(0)

        # Iterate over full regime series and extend bullish and bearish
        i = 0
        while i < len(regime):
            if regime.iloc[i] == 1:  # LV Bullish
                start = i
                while i < len(regime) and regime.iloc[i] != 3:
                    i += 1
                while i < len(regime) and regime.iloc[i] == 3:
                    i += 1
                label.iloc[start:i] = 1  # Bullish
            elif regime.iloc[i] == 2:  # HV Bearish
                start = i
                while i < len(regime) and regime.iloc[i] != 0:
                    i += 1
                while i < len(regime) and regime.iloc[i] == 0:
                    i += 1
                label.iloc[start:i] = -1  # Bearish
            else:
                i += 1
        return label.fillna(0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.index.nlevels == 2:
            result = df.groupby(level=0).apply(
                lambda g: self.assign_regimes(g[self.price_column])
            )
        else:
            result = self.assign_regimes(df[self.price_column])
        
        df["regime_label_raw"] = result  # 4-class
        df["regime_label"] = self.extend_labels(result)  # 3-class
        return df



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
