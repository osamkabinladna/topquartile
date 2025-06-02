![Cool logo](assets/img.png)

# Folder Structure

```bash
├── README.md
├── __pycache__
├── assets
│   └── img.png
├── data
├── misc
│   ├── features_binary.txt
│   ├── missing_features.txt
│   └── quantile_predictions.csv
├── params
│   └── best_params.csv
├── requirements.txt
└── topquartile
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-312.pyc
    │   └── __init__.cpython-313.pyc
    ├── data
    │   ├── covariates_may2025v2.csv
    │   ├── dec2024v2.csv
    │   ├── ihsg_may2025.csv
    │   └── macro_may2025v2.csv
    ├── experiments
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-312.pyc
    │   │   └── __init__.cpython-313.pyc
    │   ├── notebooks
    │   │   ├── Regime Switching KN.ipynb
    │   │   ├── dataloader_debug.ipynb
    │   │   ├── dataloaderv2.ipynb
    │   │   ├── feature_set.ipynb
    │   │   ├── qrf.ipynb
    │   │   └── tutorial.ipynb
    │   ├── optimize
    │   │   ├── __ init __.py
    │   │   ├── __pycache__
    │   │   ├── nary
    │   │   ├── qrf
    │   │   ├── regimes
    │   │   └── train.py
    │   ├── portfolio
    │   └── train
    │       ├── __init__.py
    │       ├── nary.py
    │       ├── qrf.py
    │       └── regimes.py
    └── modules
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-312.pyc
        │   └── __init__.cpython-313.pyc
        ├── datamodule
        │   ├── __init__.py
        │   ├── __pycache__
        │   ├── dataloader.py
        │   ├── partitions.py
        │   └── transforms
        └── evaluation
            ├── __init__.py
            └── partitioner.py
```

# Example of loading data

```python
from topquartile.modules.datamodule.dataloader import DataLoader
from topquartile.modules.datamodule.transforms.covariate import (TechnicalCovariateTransform, FundamentalCovariateTransform)
from topquartile.modules.datamodule.transforms.label import BinaryLabelTransform, ExcessReturnTransform, NaryLabelTransform
from topquartile.modules.datamodule.partitions import PurgedTimeSeriesPartition
from topquartile.modules.evaluation import Evaluation

covtrans_config = [((TechnicalCovariateTransform, dict(sma = [20, 30],
                                                       ema = [20, 30],
                                                       momentum_change=True,
                                                       volatility = [20, 30],)))]

labeltrans_config = [(BinaryLabelTransform, dict(index_csv='ihsg_may2025',
                                                 label_duration=20,
                                                 quantile=0.20))]

partition_config = dict(n_splits=5,
                        gap=20,
                        max_train_size=504,
                        test_size=60,
                        verbose=False)

dataloader = DataLoader(data_id='covariates_may2025v2',
                        macro_id='ihsg_may2025',
                        covariate_transform=covtrans_config,
                        label_transform=labeltrans_config,
                        partition_class=PurgedTimeSeriesPartition,
                        partition_kwargs=partition_config)

data = dataloader.get_cv_folds() # as cv folds

data = dataloader.transform_data() # as a whole

```

# How to optimize hyperparameters using wandb sweeps

1. cd to where u have the yaml file
2. run this command
```bash
wandb sweep <name of yaml file>
```
3. copy the wandb agent command output
4. cd to top level topquartile folder (the one that does not have __init__ files)
5. run the wandb agent command, for example
```bash
wandb agent repres/topquartile/qkr7gyoq
``` 
