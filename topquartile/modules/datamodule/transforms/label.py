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
        self.label_col_name = f'label'

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