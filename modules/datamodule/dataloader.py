import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple
import re
from collections import defaultdict
import yfinance as yf


class DataLoader:
    """
    Loads Bloomberg-formatted data
    """
    def __init__(self, covariates_id: str, labels_id: str, label_duration: int,  pred_length: int = 20, n_train: int = 252,
                 n_test: int = 30, n_embargo: int = 20, save: bool = True, save_directory: str = ''):
        self.covariates_id = covariates_id
        self.labels_id = labels_id
        self.label_duration = label_duration
        self.pred_length = pred_length
        self.n_train = n_train
        self.n_test = n_test
        self.n_embargo = n_embargo
        self.save = save
        self.save_directory = save_directory
        self.remove_last_n = self.label_duration


        self.covariates = None
        self.labels = None
        self.covlist = None
        self.pred = None

        cwd = Path.cwd()
        self.covariates_path = cwd / 'data' / self.covariates_id
        self.labels_path = cwd / 'data' / self.labels_id

    def transform_data(self):
        raise NotImplementedError

    def process_data(self):
        self._load_covariates()
        self._load_labels()
        self._impute_columns()


    def _load_data(self) -> pd.DataFrame:
        ticker_df = pd.read_csv(self.covariates_path,
                                skiprows=3, low_memory=False)

        tickernames = ticker_df.columns.tolist()
        tickernames = [ticker for ticker in tickernames if not ticker.startswith('Unnamed')]

        covariates = pd.read_csv(self.covariates_path, skiprows=5, index_col=0)
        covariates.dropna(inplace=True, axis=0, how='all')
        covariates.dropna(inplace=True, axis=1, how='all')

        covariates.index = pd.to_datetime(covariates.index, format='mixed')

        col_dict = defaultdict(list)
        for col in covariates.columns:
            number = self._get_number(col)
            col_dict[number].append(col)

        max_number = max(col_dict.keys())
        self.covlist = [None] * (max_number + 1)

        for number in range(max_number + 1):
            cols = col_dict.get(number, [])
            if cols:
                self.covlist[number] = covariates[cols]
            else:
                self.covlist[number] = pd.DataFrame()

        tickernames = [ticker[:4] for ticker in tickernames] # Becos duplicates show as such "IMJS IJ EQUITY:1"

        first_occurrence_index = {}
        duplicate_indices = []

        for index, ticker in enumerate(tickernames):
            if ticker in first_occurrence_index:
                duplicate_indices.append(index)
            else:
                first_occurrence_index[ticker] = index

        unique_tickernames = []
        unique_covlist = []
        for index, ticker in enumerate(tickernames):
            if index not in duplicate_indices:
                unique_tickernames.append(ticker)
                unique_covlist.append(self.covlist[index])

        print(len(unique_tickernames), len(unique_covlist))
        self.covlist = unique_covlist
        self.tickernames = unique_tickernames

        for idx, cov in enumerate(self.covlist):
            cov_copy = cov.copy()
            cov_copy.loc[:, 'ticker'] = tickernames[idx]
            self.covlist[idx] = cov_copy

        return self.covlist


    def _get_number(self, col_name):
        match = re.match(r'^(.*?)(?:\.(\d+))?$', col_name)
        if match.group(2):
            return int(match.group(2))
        else:
            return 0

    def _load_labels(self):
        index_ticker = "^JKSE"
        self.labels = yf.download(index_ticker, start='2014-01-01')
        self.labels.index = pd.to_datetime(self.labels.index)
        self.labels = self.labels.rename(columns=dict(Close='JKSE_PRICE'))
        self.labels['PCT_CHANGE_JKSE'] = ((self.labels['JKSE_PRICE'].shift(-self.label_duration) - self.labels['JKSE_PRICE']) / self.labels['JKSE_PRICE']) * 100
        self.labels['JKSE_Daily_Return'] = self.labels['JKSE_PRICE'].pct_change()

    def _impute_columns(self):
        """
        TODO: I dont like this, change to explicit call of features to use
        maybe save columns required as datatransform class property
        and then
        """

        required_columns = [
            'PX_LAST', 'RETURN_COM_EQY',
            'CUR_MKT_CAP', 'PX_TO_BOOK_RATIO', 'PX_TO_SALES_RATIO',
            'PROF_MARGIN', 'OPER_MARGIN', 'OPERATING_ROIC', 'SALES_GROWTH', 'PE_RATIO', 'RSI_30D', 'WACC', 'DEBT_TO_MKT_CAP'
        ]

        covlist_merged = pd.concat(self.covlist, axis=0)
        missing_value_threshold = covlist_merged[required_columns].isna().sum().max()

        missing_counts_all = covlist_merged.isna().sum()
        exclude_columns = ['EQY_DVD_YLD_IND', 'NEWS_SENTIMENT_DAILY_AVG']
        columns_to_drop = missing_counts_all[missing_counts_all > missing_value_threshold].index.tolist()

        for idx, cov in enumerate(self.covlist):
            try:
                self.covlist[idx] = self.covlist[idx].drop(axis=1, columns = columns_to_drop + exclude_columns)
            except KeyError:
                print(idx)
                continue
            try:
                self.covlist[idx] = self.covlist[idx].drop(axis=1, columns = 'NEWS_SENTIMENT_DAILY_AVG')
            except KeyError:
                print(idx)
                continue

    def load_preds(self):
        if self.covlist is None:
            #TODO: Define Process data
            self.process_data()

        preds = []
        for cov  in self.covlist:
            preds.append(cov.iloc[:-self.remove_last_n])

        self.covlist = [cov.iloc[:self.remove_last_n] for cov in self.covlist]
        self.pred = pd.concat(preds, axis=0)
        self.pred.drop(['NEWS_SENTIMENT_DAILY_AVG', 'NEWS_HEAT_PUB_DAVG'], axis=1, inplace=True)
        return self.pred

    def split_train_embargo_test(self, df, n_test=30, n_embargo=20, n_train=514):
        df['dataset'] = np.nan
        df['window'] = np.nan

        total_rows = len(df)
        window_number = 1

        current_index = total_rows

        # Apply rolling window starting from the bottom
        while current_index > 0:
            test_start = max(0, current_index - n_test)
            embargo_start = max(0, test_start - n_embargo)
            train_start = max(0, embargo_start - n_train)

            df.iloc[test_start:current_index, df.columns.get_loc('dataset')] = 'test'
            df.iloc[embargo_start:test_start, df.columns.get_loc('dataset')] = 'embargo'
            df.iloc[train_start:embargo_start, df.columns.get_loc('dataset')] = 'train'

            df.iloc[train_start:current_index, df.columns.get_loc('window')] = int(window_number)

            # Update for the next window
            current_index = train_start
            window_number += 1

        return df

    def split_and_get_window(self, df, window_size):
        for idx, cov in enumerate(self.covlist):
            self.covlist[idx] = self.split_train_embargo_test(self.covlist[idx], n_train=self.n_train, n_test=self.n_test, n_embargo=self.n_embargo)

        covlist_merge = []
        for cov in self.covlist:
            covlist_merge.append(cov[cov['window']] == 1.0)

        self.window = pd.concat(covlist_merge, axis=0)
        self.window.drop(['NEWS_SENTIMENT_DAILY_AVG', 'NEWS_HEAT_PUB_DAVG'], axis=1, inplace=True)

    def label_top_n_percentile(self, df, column_name, n_percentile, target_column_name):
        threshold = df[column_name].quantile(n_percentile / 100)
        df[target_column_name] = (df[column_name] > threshold).astype(int)

        return df

    def label_window(self):
        # TODO: This is crap
        cwd = Path.cwd()

        window1_labeled = self.label_top_n_percentile(self.window, 'DELTA', 80, 'TOP_QUANTILE')
        df = window1_labeled.dropna(inplace=False, axis=0, how='any')
        self.train_df =df[df['dataset'] == 'train']
        self.test_df = df[df['dataset'] == 'test']
        self.train_df.drop(['dataset', 'window', 'DELTA', 'PCT_CHANGE'], axis=1, inplace=True)
        self.test_df.drop(['dataset', 'window', 'DELTA'], axis=1, inplace=True)

        self.train_df.to_csv(cwd / 'data' / f'label_{self.label_duration}' / 'train_niuw.csv', index=True)
        self.test_df.to_csv(cwd / 'data' / f'label_{self.label_duration}' / 'test_niuw.csv', index=True)