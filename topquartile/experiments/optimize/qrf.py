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
def train_xgb(config=None):
    with wandb.init(config=config):
        config = wandb.config

        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'early_stopping_rounds': 200,
            'callbacks': [WandbCallback(log_model=False, log_feature_importance=True, define_metric=True)],
            'learning_rate': config.learning_rate,
            'max_depth': int(config.max_depth),
            'max_bin': int(config.max_bin),
            'min_child_weight': config.min_child_weight,
            'gamma': config.gamma,
            'subsample': config.subsample,
            'n_estimators': config.n_estimators,
            'colsample_bytree': config.colsample_bytree,
            'colsample_bylevel': config.colsample_bylevel,
            'reg_alpha': config.reg_alpha,
            'grow_policy': config.grow_policy,
            'reg_lambda': config.reg_lambda,
            'scale_pos_weight': config.scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        fit_params = {
            'eval_set': [(x_valid, y_valid)],
        }

        bst = XGBClassifier(**model_params)

        bst.fit(
            x_train, y_train,
            **fit_params
        )

        eval_results = bst.evals_result()

        best_iteration = bst.best_iteration
        best_aucpr = eval_results['validation_0']['aucpr'][best_iteration]
        wandb.log({'best_aucpr': best_aucpr, 'best_iteration': best_iteration})

        y_pred_probs = bst.predict_proba(x_valid)[:, 1]
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred_probs]

        wandb.log({"classification_report": classification_report(y_valid, y_pred_binary, output_dict=True)})


resume = False
if resume:
    sweep_id = "your_sweep_id"
else:
    sweep_id = wandb.sweep(sweep_config, project="xgboost-trainsmol")

wandb.agent(sweep_id, function=train_xgb, count=100)