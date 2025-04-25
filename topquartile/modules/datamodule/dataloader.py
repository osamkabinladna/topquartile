import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Type
import re
from collections import defaultdict
import warnings
import yfinance as yf
from topquartile.modules.datamodule.transforms import (
    CovariateTransform, LabelTransform)


class DataLoader:
    def __init__(self, data_id: str,
                 covariate_transform: Optional[List[Tuple[Type[CovariateTransform], Dict]]] = None,
                 label_transform: Optional[List[Tuple[Type[LabelTransform], Dict]]] = None):
        self.data_id = data_id
        self.covariate_transform = covariate_transform

        self.covariate_transform_config = covariate_transform
        self.label_transform_config = label_transform

        self.data = None
        self.labels = None
        self.pred = None
        self.required_covariates = set()

        root_path = Path(__file__).resolve().parent.parent.parent
        self.covariates_path = root_path / 'data' / f'{self.data_id}.csv'

    def _transform_data(self):
        if self.data is None:
            self._load_data()

        for TransformClass, params in self.covariate_transform_config:
            if not issubclass(TransformClass, CovariateTransform):
                raise ValueError(f"Warning: Invalid transform type in config: {TransformClass}. Must be a subclass of CovariateTransform")
            try:
                transformer_instance = TransformClass(df=self.data, **params)
                self.data = transformer_instance.transform()
                self.required_covariates.update(transformer_instance.required_base)

            except Exception as e:
                raise ValueError (f"Error applying {TransformClass.__name__}: {e}")

        for TransformClass, params in self.label_transform_config:
            if not issubclass(TransformClass, LabelTransform):
                raise ValueError(f"Warning: Invalid transform type in config: {TransformClass}. Must be a subclass of LabelTransform")
            try:
                transformer_instance = TransformClass(df=self.data, **params)
                self.data = transformer_instance.transform()
            except Exception as e:
                raise ValueError (f"Error applying {TransformClass.__name__}: {e}")

        return self.data

    def process_data(self):
        raise NotImplementedError

    def _load_data(self) -> pd.DataFrame:
        ticker_df = pd.read_csv(self.covariates_path,
                                skiprows=3, low_memory=False)

        tickernames = ticker_df.columns.tolist()
        tickernames = [ticker for ticker in tickernames if not ticker.startswith('Unnamed')]

        covariates = pd.read_csv(self.covariates_path, skiprows=5, index_col=0, low_memory=False)
        covariates.dropna(inplace=True, axis=0, how='all')
        covariates.dropna(inplace=True, axis=1, how='all')

        covariates.index = pd.to_datetime(covariates.index, format='mixed')

        col_dict = defaultdict(list)
        for col in covariates.columns:
            number = self._get_number(col)
            col_dict[number].append(col)

        max_number = max(col_dict.keys())
        covlist = [None] * (max_number + 1)

        for number in range(max_number + 1):
            cols = col_dict.get(number, [])
            if cols:
                covlist[number] = covariates[cols]
            else:
                covlist[number] = pd.DataFrame()

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
                unique_covlist.append(covlist[index])

        covlist = unique_covlist
        self.tickernames = unique_tickernames

        for idx, cov in enumerate(covlist):
            cov_copy = cov.copy()
            cov_copy.loc[:, 'ticker'] = tickernames[idx]

            if idx != 0:
                cov_copy.columns = [col.split('.')[0] for col in cov_copy.columns]
            covlist[idx] = cov_copy
        self.data = pd.concat(covlist)

        return self.data

    def _get_number(self, col_name):
        """
        credit to chatgpt
        """
        match = re.match(r'^(.*?)(?:\.(\d+))?$', col_name)
        if match.group(2):
            return int(match.group(2))
        else:
            return 0

    def _impute_columns(self):
        """
        Imputes columns inplace
        """
        missing_value_threshold = self.data[self.required_covariates].isna().sum()
        missing_value_all = self.data.isna().sum()
        columns_to_drop = missing_value_all[missing_value_all > missing_value_threshold]
        if columns_to_drop is not None:
            warnings.warn(f'too much missing value in the following column. {columns_to_drop} will be dropped')

        for idx, cov in enumerate(self.data):
            try:
                self.data[idx] = self.data[idx].drop(axis=1, columns=columns_to_drop)
            except KeyError:
                warnings.warn(f'{columns_to_drop} was not found in the data even though they are suposed to be dropped')
                continue
            try:
                self.data[idx] = self.data[idx].drop(axis=1, columns='NEWS_SENTIMENT_DAILY_AVG')
            except KeyError:
                print(idx)
                continue


    def load_preds(self):
        raise NotImplementedError

    def _partition_data(self):
        raise NotImplementedError