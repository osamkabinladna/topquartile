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

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'best_aucpr',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {'min': 0.0005, 'max': 0.1},
        'max_depth': {'min': 5, 'max': 60},
        'min_child_weight': {'min': 1.0, 'max': 30.0},
        'max_bin': {'min': 256, 'max': 1024},
        'gamma': {'min': 0.0, 'max': 0.5},
        'subsample': {'min': 0.6, 'max': 1.0},
        'colsample_bytree': {'min': 0.5, 'max': 1.0},
        'colsample_bylevel': {'min': 0.5, 'max': 1.0},
        'reg_alpha': {'min': 0.0, 'max': 1.0},
        'reg_lambda': {'min': 0.0, 'max': 1.0},
        'grow_policy': {'values': ['depthwise', 'lossguide']},
        'n_estimators': {'values': [800]},
        'scale_pos_weight': {'min': 8.0, 'max': 10.0},
    }
}
