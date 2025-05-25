# quantile_sweep_train.py
import sys
sys.path.append('/Users/shintarou/coding/topquartile')

import numpy as np
import pandas as pd
import optuna
import wandb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from quantile_forest import RandomForestQuantileRegressor
from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import TechnicalCovariateTransform
from topquartile.modules.datamodule.transforms.label import BinaryLabelTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition

# ===================
# Configuration Setup
# ===================
covariate_dict = dict(ema=[10, 20, 30], sma=[10, 20, 30], volatility=[10, 20, 30])
covariate_config = [(TechnicalCovariateTransform, covariate_dict)]
label_dict = dict(label_duration=30, quantile=0.75)
label_config = [(BinaryLabelTransform, label_dict)]
partition_dict = dict(n_splits=5, max_train_size=504, test_size=60, gap=20)

dataloader = DataLoader(
    data_id='dec2024',
    covariate_transform=covariate_config,
    label_transform=label_config,
    partition_class=PurgedTimeSeriesPartition,
    partition_kwargs=partition_dict
)

folds = dataloader._partition_data()  # returns list of (train_df, valid_df)

# =====================================
# Objective Function for Optuna + W&B
# =====================================
def objective(trial):
    config = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0)
    }

    wandb.init(project="quantile-sweep", config=config)

    rmse_scores = []
    sharpes = []

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
        y_pred = model.predict(X_valid.values, quantiles=[0.1, 0.5, 0.9])
        
        q10, q50, q90 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        # RMSE on q50 (median)
        rmse = np.sqrt(mean_squared_error(y_valid, q50))
        rmse_scores.append(rmse)

        # Sharpe-like score
        risk = q90 - q10
        risk[risk == 0] = np.nan
        sharpe = np.nanmedian(q50 / risk)
        sharpes.append(sharpe)

    avg_rmse = np.mean(rmse_scores)
    avg_sharpe = np.nanmean(sharpes)

    wandb.log({"avg_rmse": avg_rmse, "avg_sharpe": avg_sharpe})
    wandb.finish()

    return avg_rmse  # or -avg_sharpe if you prefer maximizing Sharpe

# =====================
# Run Optuna Optimization
# =====================
study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=30)

# =====================
# Final Training Script
# =====================
print("Best trial parameters:", study.best_trial.params)

# Train all fold and  + Sharpe log
best_params = study.best_trial.params
all_preds = []
all_actuals = []

wandb.init(project="quantile-train", config=best_params)

for fold_id, (train, valid) in enumerate(folds):
    train = train.dropna()
    valid = valid.dropna()

    to_remove = ['label', 'EXCESS_RETURN', 'INDEX_RETURN', '30d_stock_return']
    X_train = train.drop(columns=to_remove + ['ticker'], errors='ignore')
    y_train = train['EXCESS_RETURN']
    X_valid = valid.drop(columns=to_remove + ['ticker'], errors='ignore')
    y_valid = valid['EXCESS_RETURN']

    model = RandomForestQuantileRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        n_jobs=-1
    )

    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_valid.values, quantiles=[0.1, 0.5, 0.9])
    q10, q50, q90 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    df = pd.DataFrame({
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'y_true': y_valid.values,
        'sharpe': q50 / (q90 - q10 + 1e-6)  # avoid 0 division
    })
    wandb.log({f"sharpe_fold_{fold_id}": df['sharpe'].median()})
    all_preds.append(df)

# Integration of all of fold
final_df = pd.concat(all_preds)
wandb.log({"final_sharpe": final_df['sharpe'].median()})
final_df.to_csv("quantile_predictions.csv", index=False)
wandb.save("quantile_predictions.csv")
wandb.finish()
