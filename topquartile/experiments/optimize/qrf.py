from xgboost import XGBClassifier
import joblib
import pandas as pd
import wandb
from sklearn.metrics import classification_report
from pathlib import Path
from wandb.integration.xgboost import WandbCallback

cwd = Path.cwd()
data_path = cwd.parent.parent / 'data' / 'label_60'

x_train = pd.read_csv(data_path / 'train_niuw.csv', index_col=0)
x_valid = pd.read_csv(data_path / 'test_niuw.csv', index_col=0)


object_cols_train = x_train.select_dtypes(include=['object']).columns.tolist()
object_cols_valid = x_valid.select_dtypes(include=['object']).columns.tolist()

def convert_columns_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

convert_columns_to_numeric(x_train, object_cols_train)
convert_columns_to_numeric(x_valid, object_cols_valid)

# Separate target variable
y_train = x_train['TOP_QUANTILE']
x_train = x_train.drop(['TOP_QUANTILE', 'Ticker'], axis=1)

y_valid = x_valid['TOP_QUANTILE']
x_valid = x_valid.drop(['TOP_QUANTILE', 'Ticker'], axis=1)

rfecv = joblib.load(data_path / 'rfecv.joblib')
print("X_TRAIN", x_train.columns)
print("X_VALID", x_valid.columns)
x_train = rfecv.transform(x_train)
x_valid = rfecv.transform(x_valid.drop(['PCT_CHANGE'], axis=1))

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