import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Type
import re
from collections import defaultdict
import warnings

from topquartile.modules.datamodule.transforms import (
    CovariateTransform,
    LabelTransform,
)
from topquartile.modules.datamodule.partitions import (
    BasePurgedTimeSeriesPartition,
    PurgedTimeSeriesPartition,
    PurgedGroupTimeSeriesPartition,
)


class DataLoader:
    def _init_(
        self,
        data_id: str,
        *,
        covariate_transform: Optional[
            List[Tuple[Type[CovariateTransform], Dict]]
        ] = None,
        label_transform: Optional[
            List[Tuple[Type[LabelTransform], Dict]]
        ] = None,
        cols2drop: Optional[List[str]] = None,
        prediction_length: int = 20,
        partition_class: Type[BasePurgedTimeSeriesPartition] = PurgedTimeSeriesPartition,
        partition_params: Optional[Dict] = None,
    ):
        self.data_id = data_id
        self.covariate_transform_config = covariate_transform or []
        self.label_transform_config = label_transform or []
        self.cols2drop = cols2drop or ["NEWS_SENTIMENT_DAILY_AVG"]
        self.prediction_length = prediction_length

        if not issubclass(partition_class, BasePurgedTimeSeriesPartition):
            raise ValueError(
                "partition_class must inherit from BasePurgedTimeSeriesPartition"
            )
        self.partitioner: BasePurgedTimeSeriesPartition = partition_class(
            **(partition_params or {})
        )

        self.data: Optional[pd.DataFrame] = None
        self.tickernames: List[str] = []
        self.required_covariates: set[str] = set()
        self.preds: Optional[pd.DataFrame] = None

        root_path = Path(_file_).resolve().parent.parent.parent
        self.covariates_path = root_path / "data" / f"{self.data_id}.csv"

    def load_preds(self) -> pd.DataFrame:
        if self.data is None:
            self._process_data()

        self.preds = (
            self.data.groupby("ticker", group_keys=False).tail(self.prediction_length)
        )
        remaining_index = self.data.index.difference(self.preds.index)
        self.data = self.data.loc[remaining_index]
        return self.preds

    def _process_data(self):
        self._transform_data()
        self._impute_columns()

    def _transform_data(self):
        if self.data is None:
            self._load_data()

        for TransformClass, params in self.covariate_transform_config:
            if not issubclass(TransformClass, CovariateTransform):
                raise ValueError(
                    "Invalid transform in covariate_transform_config: must subclass CovariateTransform"
                )
            transformer = TransformClass(df=self.data, **params)
            self.data = transformer.transform()
            self.required_covariates.update(transformer.required_base)

        for TransformClass, params in self.label_transform_config:
            if not issubclass(TransformClass, LabelTransform):
                raise ValueError(
                    "Invalid transform in label_transform_config: must subclass LabelTransform"
                )
            transformer = TransformClass(df=self.data, **params)
            self.data = transformer.transform()

        return self.data

    def _load_data(self) -> pd.DataFrame:
        ticker_df = pd.read_csv(self.covariates_path, skiprows=3, low_memory=False)

        tickernames = [col for col in ticker_df.columns if not col.startswith("Unnamed")]
        covariates = pd.read_csv(
            self.covariates_path, skiprows=5, index_col=0, low_memory=False
        )
        covariates.dropna(inplace=True, axis=0, how="all")
        covariates.dropna(inplace=True, axis=1, how="all")
        covariates.index = pd.to_datetime(covariates.index, format="mixed")

        col_dict: dict[int, list[str]] = defaultdict(list)
        for col in covariates.columns:
            number = self._get_number(col)
            col_dict[number].append(col)
        max_number = max(col_dict.keys())
        covlist: list[pd.DataFrame] = [pd.DataFrame()] * (max_number + 1)
        for number in range(max_number + 1):
            cols = col_dict.get(number, [])
            covlist[number] = covariates[cols] if cols else pd.DataFrame()

        tickernames = [ticker[:4] for ticker in tickernames]  # "IMJS IJ EQUITY:1" → "IMJS"
        first_occurrence: dict[str, int] = {}
        duplicate_indices: list[int] = []
        for idx, ticker in enumerate(tickernames):
            if ticker in first_occurrence:
                duplicate_indices.append(idx)
            else:
                first_occurrence[ticker] = idx
        unique_covlist: list[pd.DataFrame] = []
        unique_tickernames: list[str] = []
        for idx, ticker in enumerate(tickernames):
            if idx not in duplicate_indices:
                unique_covlist.append(covlist[idx])
                unique_tickernames.append(ticker)
        covlist = unique_covlist
        self.tickernames = unique_tickernames

        for idx, cov in enumerate(covlist):
            cov_copy = cov.copy()
            cov_copy["ticker"] = tickernames[idx]
            if idx != 0:
                cov_copy.columns = [col.split(".")[0] for col in cov_copy.columns]
            covlist[idx] = cov_copy
        self.data = pd.concat(covlist)
        return self.data

    def _get_number(self, col_name: str) -> int:
        match = re.match(r"^(.*?)(?:\.(\d+))?$", col_name)
        return int(match.group(2)) if match and match.group(2) else 0

    def _impute_columns(self):
        missing_value_all = self.data.isna().sum()
        missing_value_threshold = self.data[self.required_covariates].isna().sum()
        columns_to_drop = missing_value_all[missing_value_all > missing_value_threshold]
        if not columns_to_drop.empty:
            warnings.warn(
                f"High missingness – dropping columns: {columns_to_drop.index.tolist()}"
            )
        self.data = self.data.drop(columns=columns_to_drop.index, errors="ignore")
        self.data = self.data.drop(columns=self.cols2drop, errors="ignore")

    def _partition_data(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if self.data is None:
            self._process_data()

        self.data = self.data.sort_index()

        fold_buckets = [
            {"train": [], "test": []} for _ in range(self.partitioner.n_splits)
        ]

        for ticker, df_ticker in self.data.groupby("ticker"):
            df_ticker = df_ticker.sort_index()

            if isinstance(self.partitioner, PurgedGroupTimeSeriesPartition):
                groups = df_ticker.index.normalize()  # one group per date
            else:
                groups = None

            splits = list(self.partitioner.split(df_ticker, groups=groups))
            if len(splits) != self.partitioner.n_splits:
                raise RuntimeError(
                    "Ticker {} produced {} splits but partitioner is configured for {}".format(
                        ticker, len(splits), self.partitioner.n_splits
                    )
                )
            for fold_id, (train_idx, test_idx) in enumerate(splits):
                fold_buckets[fold_id]["train"].append(df_ticker.iloc[train_idx])
                fold_buckets[fold_id]["test"].append(df_ticker.iloc[test_idx])

        cv_folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for bucket in fold_buckets:
            train_df = pd.concat(bucket["train"]).sort_index()
            test_df = pd.concat(bucket["test"]).sort_index()
            cv_folds.append((train_df, test_df))
        return cv_folds
    