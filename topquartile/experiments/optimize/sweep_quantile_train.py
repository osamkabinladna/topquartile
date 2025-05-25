import os
import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from quantile_forest import RandomForestQuantileRegressor

from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import TechnicalCovariateTransform, FundamentalCovariateTransform
from topquartile.modules.datamodule.transforms.label import BinaryLabelTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition


# ----------- Fixed data-prep configuration ----------
covtrans_config = [(
    TechnicalCovariateTransform,
    dict(
        sma=[20, 30],
        ema=[20, 30],
        momentum_change=True,
        volatility=[20, 30],
    ),
)]

labeltrans_config = [(
    BinaryLabelTransform,
    dict(label_duration=20, quantile=0.75),
)]

partition_config = dict(
    n_splits=5,
    gap=20,
    max_train_size=504,
    test_size=60,
    verbose=False,
)

dataloader = DataLoader(
    data_id="dec2024",
    covariate_transform=covtrans_config,
    label_transform=labeltrans_config,
    partition_class=PurgedTimeSeriesPartition,
    partition_kwargs=partition_config,
)

folds = dataloader.get_cv_folds()   # → list of (train_df, test_df) tuples


# ----------- Optuna objective ----------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


    rmse_scores = []
    for fold_id, (train, test) in enumerate(folds):
        # drop rows created by label/covariate lagging
        train, test = train.dropna(), test.dropna()

        target = "EXCESS_RETURN"           # <-- exact spelling the prof asked for
        drop_cols = [target, "label", "INDEX_RETURN", "30d_stock_return", "ticker"]

        X_train = train.drop(columns=drop_cols, errors="ignore")
        y_train = train[target]

        X_test = test.drop(columns=drop_cols, errors="ignore")
        y_test = test[target]

        model = RandomForestQuantileRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            n_jobs=-1,
        )

        model.fit(X_train.values, y_train.values)


        y_pred50 = model.predict(X_test.values, quantiles=[0.5])
        rmse = np.sqrt(mean_squared_error(y_test, y_pred50))
        rmse_scores.append(rmse)

    avg_rmse = float(np.mean(rmse_scores))

    return avg_rmse


# ----------- Bayesian optimisation ----------
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),   # Bayesian / TPE
)
study.optimize(objective, n_trials=30, show_progress_bar=True)


# ----------- Persist best hyper-parameters ----------
best_row = {**study.best_params, "best_rmse": study.best_value}
pd.DataFrame([best_row]).to_csv("best_params.csv", index=False)
print("Saved best_params.csv:")
print(best_row)