import pandas as pd
import numpy as np

class EvaluationPartitioner:
    def __init__(self, df: pd.DataFrame,  n_train: int = 252, n_valid: int = 1):
        """
        :param df: first fold on the dataloader
        :param n_train: number of training days
        :param n_valid: number of prediction days (i know i named it valid lol)
        """
        self.n_train = n_train
        self.n_valid = n_valid
        self.df = df
        self.results = []

        self.df.sort_index(inplace=True)

    def partition_data(self) -> pd.DataFrame:
        results = []

        all_unique_dates = self.df.index.get_level_values('DateIndex').unique().sort_values()

        if len(all_unique_dates) < self.n_train + self.n_valid:
            print(
                f"Not enough unique dates ({len(all_unique_dates)}) to form even one train/test split with train_length={self.n_train} and test_length={self.n_valid}.")
        else:
            for i in range(self.n_train, len(all_unique_dates) - self.n_valid + 1):

                train_period_end_date = all_unique_dates[i - 1]
                train_period_start_date = all_unique_dates[i - self.n_train]

                test_period_start_date = all_unique_dates[i]
                test_period_end_date = all_unique_dates[i + self.n_valid - 1]


                current_train_candidate = self.df.loc[pd.IndexSlice[:, train_period_start_date:train_period_end_date], :]
                current_test_candidate = self.df.loc[pd.IndexSlice[:, test_period_start_date:test_period_end_date], :]


                if current_train_candidate.empty or current_test_candidate.empty:
                    continue

                train_counts = current_train_candidate.groupby(level='TickerIndex', observed=False).size()
                valid_train_tickers = train_counts[train_counts == self.n_train].index

                test_counts = current_test_candidate.groupby(level='TickerIndex', observed=False).size()
                valid_test_tickers = test_counts[test_counts == self.n_valid].index

                common_valid_tickers = valid_train_tickers.intersection(valid_test_tickers)

                if not common_valid_tickers.empty:
                    final_train_df = current_train_candidate.loc[pd.IndexSlice[common_valid_tickers, :], :]
                    final_test_df = current_test_candidate.loc[pd.IndexSlice[common_valid_tickers, :], :]

                    # Double CHECK
                    if not final_train_df.empty and not final_test_df.empty:
                        results.append((final_train_df, final_test_df))

        return results