import numpy as np
import wandb
from sklearn.metrics import mean_squared_error
from quantile_forest import RandomForestQuantileRegressor

from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import (
    TechnicalCovariateTransform, FundamentalCovariateTransform)
from topquartile.modules.datamodule.transforms.label import ExcessReturnTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition


covtrans_config = [((TechnicalCovariateTransform, dict(sma = [20, 30], volume_std = [10, 20, 40, 60, 120],
                                                     ema = [20, 30],
                                                     volatility = [120, 60, 40, 20], 1],
                                                     turnover = [20, 40, 60, 120, 240], volume_std = [10,20,40,60,120] ))), 
                   (FundamentalCovariateTransform, dict(adjusted_roic=True))]

labeltrans_config = [(ExcessReturnTransform, dict(label_duration=20,
                                                quantile=0.75))]
folds = dataloader.get_cv_folds()

TARGET = "EXCESS_RETURN"
DROP_COLS = [TARGET, "label", "INDEX_RETURN", "30d_stock_return", "ticker"]

def train_one_fold(fold_id: int, config) -> float:
    train_df, test_df = folds[fold_id]
    train_df, test_df = train_df.dropna(), test_df.dropna()

    X_train = train_df.drop(columns=DROP_COLS, errors="ignore")
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=DROP_COLS, errors="ignore")
    y_test  = test_df[TARGET]

    model = RandomForestQuantileRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        n_jobs=-1,
    )
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values, quantiles=[0.5])

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return rmse

def main() -> None:
    parent = wandb.init()
    cfg         = parent.config
    parent_id   = parent.id
    parent_proj = parent.project
    wandb.finish()

    rmses = []
    for k in range(len(folds)):
        with wandb.init(
                project=parent_proj,
                group=parent_id,
                job_type=f"fold-{k}",
                config=cfg,
                reinit=True):
            rmse_k = train_one_fold(k, cfg)
            wandb.log({"rmse": rmse_k})
            wandb.summary["rmse"] = rmse_k
            rmses.append(rmse_k)

    parent = wandb.init(
        id=parent_id,
        project=parent_proj,
        resume="allow")
    avg_rmse = float(np.mean(rmses))
    parent.log({"avg_rmse": avg_rmse})
    parent.summary["avg_rmse"] = avg_rmse
    parent.finish()

if _name_ == "_main_":
    main()