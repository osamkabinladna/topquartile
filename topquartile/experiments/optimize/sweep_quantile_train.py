import wandb
from quantile_forest import RandomForestQuantileRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import (TechnicalCovariateTransform, FundamentalCovariateTransform)
from topquartile.modules.datamodule.transforms.label import BinaryLabelTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition


def objective_rmse_only(trial):
    config = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0)
    }

    covtrans_config = [((TechnicalCovariateTransform, dict(sma = [20, 30],
                                                       ema = [20, 30],
                                                       momentum_change=True,
                                                       volatility = [20, 30],)))]

    labeltrans_config = [(BinaryLabelTransform, dict(label_duration=20,
                                                quantile=0.75))]

    partition_config = dict(n_splits=5, gap=20, max_train_size=504, test_size=60, verbose=False)

    dataloader = DataLoader(data_id='dec2024', covariate_transform=covtrans_config,
                  label_transform=labeltrans_config, partition_class=PurgedTimeSeriesPartition,
                  partition_kwargs=partition_config)

    wandb.init(project="quantile-sweep-rmse", config=config)
    rmse_scores = []

    for fold_id, (train, valid) in enumerate(folds):
        train = train.dropna()
        valid = valid.dropna()

        to_remove = ['label', 'EXCESS_RETURN', 'INDEX_RETURN', '30d_stock_return']
        X_train = train.drop(columns=to_remove + ['ticker'], errors='ignore')
        y_train = train['EXCESS_RETURN']
        X_valid = valid.drop(columns=to_remove + ['ticker'], errors='ignore')
        y_valid = valid['EXCESS_RETURN']

        model = RandomForestQuantileRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'],
            n_jobs=-1
        )

        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_valid.values, quantiles=[0.5])  
        q50 = y_pred[:, 0]

        rmse = np.sqrt(mean_squared_error(y_valid, q50))
        rmse_scores.append(rmse)

    avg_rmse = np.mean(rmse_scores)
    wandb.log({"avg_rmse": avg_rmse})
    wandb.finish()
    return avg_rmse  

study = optuna.create_study(direction="minimize")
study.optimize(objective_rmse_only, n_trials=30)
