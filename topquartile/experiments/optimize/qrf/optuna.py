import numpy as np
import optuna
from types import SimpleNamespace
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_squared_error

from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import TechnicalCovariateTransform
from topquartile.modules.datamodule.transforms.label import ExcessReturnTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition

LABEL_DURATION = 20
covtrans_config = [(TechnicalCovariateTransform, dict(
    sma=[20, 40, 60],
    ema=[20, 40, 60],
    turnover=[20, 40, 60, 120, 240],
    macd=[(12, 26, 9)],
    price_gap=[20, 40, 60],
    price_ratio=[9, 19, 39, 59, 119],
    acceleration_rate=True,
    volatility=[10, 20, 40, 60, 120],
    volume_std=[10, 20, 40, 60, 120],
))]
labeltrans_config = [(ExcessReturnTransform, dict(label_duration=LABEL_DURATION))]
partition_config  = dict(n_splits=5, gap=2, max_train_size=504, test_size=60)

dataloader = DataLoader(
    data_id="covariates_may2025v2",
    covariate_transform=covtrans_config,
    label_transform=labeltrans_config,
    partition_class=PurgedTimeSeriesPartition,
    partition_kwargs=partition_config,
)
folds = dataloader.get_cv_folds()

TARGET    = f"excess_returns_{LABEL_DURATION}"
DROP_COLS = [TARGET, f"index_returns_{LABEL_DURATION}",
             f"eq_returns_{LABEL_DURATION}", "ticker"]
ffill_feats = []

def train_one_fold(fold_id: int, cfg: SimpleNamespace) -> float:
    train_df, test_df = folds[fold_id]
    train_df, test_df = train_df.dropna(), test_df.dropna()

    X_train = train_df.drop(columns=DROP_COLS, errors="ignore")
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=DROP_COLS, errors="ignore")
    y_test  = test_df[TARGET]

    model = RandomForestQuantileRegressor(
        n_estimators            = cfg.n_estimators,
        max_depth               = cfg.max_depth,
        max_leaf_nodes          = cfg.max_leaf_nodes,
        criterion               = cfg.criterion,
        min_samples_split       = cfg.min_samples_split,
        min_samples_leaf        = cfg.min_samples_leaf,
        min_weight_fraction_leaf= cfg.min_weight_fraction_leaf,
        min_impurity_decrease   = cfg.min_impurity_decrease,
        ccp_alpha               = cfg.ccp_alpha,
        max_features            = cfg.max_features,
        bootstrap               = cfg.bootstrap,
        n_jobs                  = -1,
    )
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values, quantiles=[0.5])
    return float(np.sqrt(mean_squared_error(y_test, y_pred)))

def objective(trial):
    cfg = SimpleNamespace(
        n_estimators             = trial.suggest_int("n_estimators", 30, 200),
        max_depth                = trial.suggest_categorical("max_depth", [8, 12, 16, 20, 25, 30]),
        max_leaf_nodes           = trial.suggest_categorical("max_leaf_nodes", [32, 64, 128, 256, 512]),
        criterion                = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
        min_samples_split        = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf         = trial.suggest_int("min_samples_leaf", 1, 15),
        min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.4),
        min_impurity_decrease    = trial.suggest_float("min_impurity_decrease", 1e-7, 1e-2, log=True),
        ccp_alpha                = trial.suggest_float("ccp_alpha", 1e-6, 1e-2, log=True),
        max_features             = trial.suggest_float("max_features", 0.1, 1.0),
        bootstrap                = trial.suggest_categorical("bootstrap", [True, False]),
    )

    fold_rmses = []
    for k in range(len(folds)):
        rmse = train_one_fold(k, cfg)
        print(f'TRAINED FOLD {k} with RMSE: {rmse}')
        fold_rmses.append(rmse)

    avg_rmse = float(np.mean(fold_rmses))
    print(f'AVERAGE RMSE: {avg_rmse}')

    return avg_rmse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="testing123")
    study.optimize(objective, n_trials=50)

    print(f"\nFinished {len(study.trials)} trials.")
    print("Best value (avg_rmse):", study.best_trial.value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")