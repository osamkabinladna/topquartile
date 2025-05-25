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

