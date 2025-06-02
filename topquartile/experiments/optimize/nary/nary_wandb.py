import numpy as np
import optuna
from types import SimpleNamespace
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import KBinsDiscretizer
import psutil

from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import TechnicalCovariateTransform
from topquartile.modules.datamodule.transforms.label import ExcessReturnTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition

LABEL_DURATION = 20
TARGET = f"excess_returns_{LABEL_DURATION}"
DROP_COLS = [TARGET, f"index_returns_{LABEL_DURATION}", f"eq_returns_{LABEL_DURATION}", "ticker"]

# === Transforms ===
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
partition_config = dict(n_splits=5, gap=2, max_train_size=504, test_size=60)

dataloader = DataLoader(
    data_id="covariates_may2025v2",
    covariate_transform=covtrans_config,
    label_transform=labeltrans_config,
    partition_class=PurgedTimeSeriesPartition,
    partition_kwargs=partition_config,
)
folds = dataloader.get_cv_folds()

def bin_excess_returns(df, n_bins=5):
    """Discretize excess return into quantile-based classes."""
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df["target_class"] = discretizer.fit_transform(df[[TARGET]]).astype(int)
    return df

def train_one_fold(fold_id: int, cfg: SimpleNamespace) -> float:
    train_df, test_df = folds[fold_id]

    for df in [train_df, test_df]:
        df.drop(columns=["DVD_SH_12M"], errors='ignore', inplace=True)
        df.dropna(inplace=True)

    train_df = bin_excess_returns(train_df)
    test_df = bin_excess_returns(test_df)

    X_train = train_df.drop(columns=DROP_COLS + ["target_class"], errors="ignore")
    y_train = train_df["target_class"]
    X_test = test_df.drop(columns=DROP_COLS + ["target_class"], errors="ignore")
    y_test = test_df["target_class"]

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes,
        criterion=cfg.criterion,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        bootstrap=cfg.bootstrap,
        n_jobs=-1,
    )
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    f1 = f1_score(y_test, y_pred, average="macro")

    del model, X_train, y_train, X_test, y_test, train_df, test_df
    gc.collect()
    return f1

def objective(trial):
    cfg = SimpleNamespace(
        n_estimators=trial.suggest_int("n_estimators", 50, 200),
        max_depth=trial.suggest_categorical("max_depth", [8, 12, 16, 20, 25, 30]),
        max_leaf_nodes=trial.suggest_categorical("max_leaf_nodes", [32, 64, 128, 256, 512]),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
        max_features=trial.suggest_float("max_features", 0.1, 1.0),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
    )

    f1_scores = []
    for k in range(len(folds)):
        f1 = train_one_fold(k, cfg)
        print(f"FOLD {k} F1 Score: {f1}")
        f1_scores.append(f1)

    avg_f1 = float(np.mean(f1_scores))
    print(f"AVERAGE F1 Score: {avg_f1}")
    return -avg_f1  # we minimize in Optuna, so invert F1

if _name_ == "_main_":
    study = optuna.create_study(direction="minimize", study_name="rf_classifier_quantiles")
    study.optimize(objective, n_trials=50)

    print(f"\nFinished {len(study.trials)} trials.")
    print("Best (avg F1):", -study.best_trial.value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
        