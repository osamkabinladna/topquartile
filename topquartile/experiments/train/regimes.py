import os
import gc
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer

from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import TechnicalCovariateTransform
from topquartile.modules.datamodule.transforms.label import ExcessReturnTransform, KMRFLabelTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition
from topquartile.modules.evaluation.partitioner import EvaluationPartitioner

_proc = psutil.Process(os.getpid())
def flush_mem(tag=""):
    gc.collect()

LABEL_DURATION = 20
TARGET = f"regime_label"
DROP_COLS = [
    TARGET,
    'regime_label_raw',
    f"index_returns_{LABEL_DURATION}",
    f"eq_returns_{LABEL_DURATION}",
    "ticker"
]
BEST_PARAMS = dict(
    n_estimators=130,
    max_depth=30,
    max_leaf_nodes=512,
    criterion="gini",
    min_samples_split=14,
    min_samples_leaf=7,
    max_features=0.4674188016607141,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)

OUT_CSV = Path("reg_preds.csv")
OUT_CSV.unlink(missing_ok=True)

covtrans_config = [(
    TechnicalCovariateTransform,
    dict(
        sma=[10, 20, 50, 100],
        ema=[10, 20, 50],
        rsi=[14],
        macd=True,
        macd_signal=True,
        macd_histogram=True,
        roc=[6, 10, 20],
        cmo=[14],
        atr=True,
        trix=[21],
        obv=True,
        mfi=True,
        force_index=True,
        stc=True,
        bb=True,
        ultimate=True,
        awesome=True,
        plus_di=True,
        minus_di=True,
        max_return=[5, 10, 20],
        price_gap=[20],
        price_vs_sma=[20],
        momentum_change=True,
        ulcer=True,
        mean_price_volatility=[21, 252],
        approximate_entropy=True,
        adfuller=True,
        binned_entropy=True,
        cid_ce=True,
        count_above_mean=True,
        count_below_mean=True,
        energy_ratio_chunks=True,
        fft_aggregated=True,
        first_location_maximum=True,
        first_location_minimum=True,
    ),
)]
labeltrans_config = [
    (KMRFLabelTransform, dict(price_column="PX_LAST", kama_n=10, gamma=0.5))]

partition_config = dict(n_splits=5, gap=2, max_train_size=504, test_size=60)

dataloader = DataLoader(
    data_id="covariates_may2025v2",
    covariate_transform=covtrans_config,
    label_transform=labeltrans_config,
    partition_class=PurgedTimeSeriesPartition,
    partition_kwargs=partition_config,
)
folds = dataloader.get_cv_folds()
fold_concat = pd.concat(folds[0], axis=0)
eval = Evaluation(df=fold_concat, n_train=252, n_valid=1)
data_windows = eval.partition_data()
flush_mem("data")

for i in range(252):
    train_df, pred_df = data_windows[i]
    date_val = pred_df.index.get_level_values("DateIndex")[0]
    print(
        f"\n╭─ Window {i+1:03d}/252  {date_val}  "
        f"({len(train_df):,} train rows, {len(pred_df):,} pred rows)"
    )

    missing_ratio = train_df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    if cols_to_drop:
        print(f"    ↳ Dropping {len(cols_to_drop)} columns with >50% NaNs")
        train_df = train_df.drop(columns=cols_to_drop)
        pred_df = pred_df.drop(columns=cols_to_drop, errors="ignore")

    train_df = train_df.dropna()

    if TARGET not in train_df.columns:
        print(f"Skipping window {i+1:03d}: TARGET column '{TARGET}' missing")
        continue
    if train_df.empty:
        print(f"Skipping window {i+1:03d}: train_df is empty after dropna()")
        continue

    discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    try:
        y_train = discretizer.fit_transform(train_df[[TARGET]]).astype(int).ravel()
    except ValueError as e:
        print(f"Skipping window {i+1:03d}: {e}")
        continue

    X_train = train_df.drop(columns=DROP_COLS, errors="ignore")
    pred_df = pred_df.dropna()
    X_pred = pred_df.drop(columns=DROP_COLS, errors="ignore")

    if X_pred.empty:
        print(f"    ↳ Skipping window {i+1:03d}: X_pred is empty after dropna()")
        continue

    model = RandomForestClassifier(**BEST_PARAMS)
    model.fit(X_train.values, y_train)

    pred_class = model.predict(X_pred.values)
    out = pd.DataFrame(
        pred_class,
        index=pred_df.index,
        columns=["pred_class"],
    )

    header = not OUT_CSV.exists()
    out.to_csv(OUT_CSV, mode="a", header=header, index=True)
    print(f"    ↳ Wrote {len(out):,} rows to {OUT_CSV.name}")

    del model, X_train, X_pred, train_df, pred_df, out
    flush_mem(f"end {i+1:03d}")

print("\n✓ All 252 windows processed.")

if OUT_CSV.exists():
    size_mb = OUT_CSV.stat().st_size / 1024**2
    print(f"Final file size: {size_mb:,.2f} MB")
else:
    print(f"No output file found at '{OUT_CSV.name}'.")