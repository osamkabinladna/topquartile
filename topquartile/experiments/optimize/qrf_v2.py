from quantile_forest import RandomForestQuantileRegressor
import joblib
import pandas as pd
import wandb
from sklearn.metrics import classification_report
from pathlib import Path

from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import (TechnicalCovariateTransform, FundamentalCovariateTransform)
from topquartile.modules.datamodule.transforms.label import BinaryLabelTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition


covariate_dict = dict(ema = [10,20,30],
                      sma = [10,20,30],
                      volatility = [10,20,30])


covariate_config = [(TechnicalCovariateTransform, covariate_dict)]
label_dict = dict(label_duration = 30, quantile = 0.75])
label_config = [(BinaryLabelTransform, label_dict)]
partition_dict = dict(n_splits = 5, max_train_size=10,test_size= 60,
                      gap = 20,)

dataloader = DataLoader(data_id='dec2024', covariate_transform = covariate_config,
                        label_transform=label_config, partition_class = PurgedTimeSeriesPartition,
                        partition_kwargs=partition_dict)