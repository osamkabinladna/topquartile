import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import mean_squared_error
from quantile_forest import RandomForestQuantileRegressor

from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import TechnicalCovariateTransform
from topquartile.modules.datamodule.transforms.label import BinaryLabelTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition

covariate_dict = dict(
    ema=[10, 20, 30],
    sma=[10, 20, 30],
    volatility=[10, 20, 30]
)
covariate_config = [(TechnicalCovariateTransform, covariate_dict)]

label_dict = dict(
    label_duration=30,
    quantile=0.75
)
label_config = [(BinaryLabelTransform, label_dict)]

partition_dict = dict(
    n_splits=5,
    max_train_size=504,
    test_size=60,
    gap=20
)

dataloader = DataLoader(
    data_id='dec2024',
    covariate_transform=covariate_config,
    label_transform=label_config,
    partition_class=PurgedTimeSeriesPartition,
    partition_kwargs=partition_dict
)

folds = dataloader._partition_data()

# ============================
# Best Parameters from Optuna
# ============================
best_params = {
    'n_estimators': 162,
    'max_depth': 7,
    'min_samples_leaf': 1,
    'max_features': 0.9063233020424546,
    'best_rmse': 3.4228179663815985
}

# ============================
# Utility Functions
# ============================
def compute_sharpe(q10, q50, q90):
    risk = q90 - q10
    risk[risk == 0] = np.nan
    return np.nanmedian(q50 / risk)

def compute_coverage(y_true, q10, q90):
    return ((y_true >= q10) & (y_true <= q90)).mean()

# ============================
# Training & Logging
# ============================
wandb.init(project="quantile-train-final", config=best_params)
all_preds = []

for fold_id, (train, valid) in enumerate(folds):
    train = train.dropna()
    valid = valid.dropna()

    drop_cols = ['label', 'EXCESS_RETURN', 'INDEX_RETURN', '30d_stock_return', 'ticker']
    X_train = train.drop(columns=drop_cols, errors='ignore')
    y_train = train['EXCESS_RETURN']
    X_valid = valid.drop(columns=drop_cols, errors='ignore')
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
        'sharpe': q50 / (q90 - q10 + 1e-6),
        'covered': ((y_valid.values >= q10) & (y_valid.values <= q90)).astype(int)
    })

    wandb.log({
        f"sharpe_fold_{fold_id}": df['sharpe'].median(),
        f"coverage_fold_{fold_id}": df['covered'].mean(),
        f"rmse_fold_{fold_id}": mean_squared_error(y_valid, q50, squared=False),
        f"pred_table_fold_{fold_id}": wandb.Table(dataframe=df.head(100))
    })

    all_preds.append(df)

# ============================
# Final Results Logging
# ============================
final_df = pd.concat(all_preds)
wandb.log({
    "final_sharpe": final_df['sharpe'].median(),
    "final_coverage": final_df['covered'].mean(),
    "final_rmse": mean_squared_error(final_df['y_true'], final_df['q50'], squared=False),
    "prediction_table": wandb.Table(dataframe=final_df.head(200))
})

# Save to CSV
final_df.to_csv("quantile_predictions_final.csv", index=False)
wandb.save("quantile_predictions_final.csv")
wandb.finish()
