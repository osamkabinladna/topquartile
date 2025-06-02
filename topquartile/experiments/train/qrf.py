import os
import gc
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from quantile_forest import RandomForestQuantileRegressor

from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import TechnicalCovariateTransform
from topquartile.modules.datamodule.transforms.label import ExcessReturnTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition
from topquartile.modules.evaluation.partitioner import EvaluationPartitioner

_proc = psutil.Process(os.getpid())

def flush_mem(tag: str = "") -> None:
    gc.collect()
    rss = _proc.memory_info().rss / 1024**2
    print(f"[GC] {tag:>12} • RSS = {rss:8.1f} MB")

gc.set_threshold(300, 5, 5)  # collect early & often

LABEL_DURATION = 20
TARGET = f"excess_returns_{LABEL_DURATION}"
DROP_COLS = [
    TARGET,
    f"index_returns_{LABEL_DURATION}",
    f"eq_returns_{LABEL_DURATION}",
    "ticker",
]

BEST_PARAMS_QRF = dict(
    n_estimators=40,
    max_depth=15,
    max_leaf_nodes=128,
    criterion="absolute_error",
    min_samples_split=12,
    min_samples_leaf=9,
    min_weight_fraction_leaf=0.0036191161135565864,
    min_impurity_decrease=0.00024390708419450344,
    ccp_alpha=0.00013357332971458002,
    max_features=0.377838947193454,
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
)

OUT_CSV = Path("qrf_quantile_predsv2.csv")
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

labeltrans_config = [(
    ExcessReturnTransform,
    dict(
        label_duration=LABEL_DURATION,
        index_csv="ihsg_may2025"
    )
)]

partition_config = dict(
    n_splits=5,
    gap=2,
    max_train_size=504,
    test_size=60,
)

dataloader = DataLoader(
    data_id="covariates_may2025v2",
    covariate_transform=covtrans_config,
    label_transform=labeltrans_config,
    partition_class=PurgedTimeSeriesPartition,
    partition_kwargs=partition_config,
)

folds = dataloader.get_cv_folds()
fold_concat = pd.concat(folds[0], axis=0)

eval_obj = Evaluation(df=fold_concat, n_train=252, n_valid=1)
windows = eval_obj.partition_data()
flush_mem("data ready")

for i, (train_df, pred_df) in enumerate(windows, 1):
    date_str = pred_df.index.get_level_values("DateIndex")[0]
    print(
        f"\n╭─ Window {i:03d}/252  {date_str}"
        f"  ({len(train_df):,} train · {len(pred_df):,} pred rows)"
    )

    missing_ratio = train_df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    if cols_to_drop:
        print(f"    ↳ Dropping {len(cols_to_drop)} columns with >50% NaNs")
        train_df = train_df.drop(columns=cols_to_drop)
        pred_df = pred_df.drop(columns=cols_to_drop, errors="ignore")

    train_df = train_df.dropna()
    pred_df = pred_df.dropna()

    if TARGET not in train_df.columns:
        print(f"Skipping window {i:03d}: TARGET column '{TARGET}' missing")
        continue

    if train_df.empty:
        print(f"Skipping window {i:03d}: train_df is empty after dropna()")
        continue

    X_train = train_df.drop(columns=DROP_COLS, errors="ignore")
    y_train = train_df[TARGET]

    X_pred = pred_df.drop(columns=DROP_COLS, errors="ignore")
    if X_pred.empty:
        print("    ↳ skip: nothing to predict (all rows dropped after NaN filter)")
        flush_mem(f"skip {i:03d}")
        continue

    model = RandomForestQuantileRegressor(**BEST_PARAMS_QRF)
    model.fit(X_train, y_train)

    q10, q50, q90 = model.predict(X_pred, quantiles=[0.1, 0.5, 0.9]).T

    out = pd.DataFrame(
        {"q10": q10, "q50": q50, "q90": q90},
        index=pred_df.index
    )

    header = not OUT_CSV.exists()
    out.to_csv(OUT_CSV, mode="a", header=header, index=True)
    print(f"    ↳ Wrote {len(out):,} rows → {OUT_CSV.name}")

    del model, X_train, X_pred, train_df, pred_df, out
    flush_mem(f"end {i:03d}")

print("\n✓ All 252 windows processed.")

if OUT_CSV.exists():
    size_mb = OUT_CSV.stat().st_size / 1024**2
    print(f"Final file size: {size_mb:,.2f} MB")
else:
    print(f"No output file found at '{OUT_CSV.name}'.")