import warnings
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional, Any, Iterable, List
import pandas as pd
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from typing import Optional, Iterable, Iterator, Tuple, Dict, List, Any
import itertools # For chain.from_iterable

class BasePartitioner(ABC):
    @abstractmethod
    def split(self,
              X: Any,
              y: Optional[Any] = None,
              groups: Optional[Iterable[Any]] = None
              ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        pass

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        if hasattr(self, 'n_splits'):
            return self.n_splits
        else:
            raise NotImplementedError



class PurgedGroupTimeSeriesSplit(BasePartitioner):
    """
    Stolen from Kaggle - https://www.kaggle.com/code/marketneutral/purged-time-series-cv-xgboost-optuna

    Time Series cross-validator with non-overlapping groups and purging.

    This splitter provides train/test indices to split time series data based
    on groups provided by a third party. It ensures that the same group does
    not appear in both training and testing sets within a single fold.

    Crucially, it allows for a gap (`group_gap`) of groups between the end of
    the training set and the start of the test set. This purging helps prevent
    information leakage from the test set into the training set, which is
    particularly important when using models with windowed or lagged features.

    In each split `k`, the test set consists of groups from the `(k+1)`-th
    block of groups, and the training set consists of groups from earlier blocks,
    respecting the `group_gap` and `max_train_group_size`. Test indices within
    a split are always chronologically later than training indices.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2. The number of folds will be
        n_splits + 1.
    max_train_group_size : int, default=np.inf
        Maximum number of groups to include in the training set for each split.
        Limits the lookback period.
    max_test_group_size : int, default=np.inf
        Maximum number of groups to include in the test set for each split.
    group_gap : int, default=0
        Number of groups to exclude between the end of the training set and
        the beginning of the test set. This gap helps prevent leakage.

    Raises
    ------
    ValueError
        If `groups` is not provided in the `split` method.
        If `n_splits + 1` is greater than the number of unique groups.
    """

    def __init__(self,
                 n_splits: int = 5,
                 max_train_group_size: int = np.inf,
                 max_test_group_size: int = np.inf,
                 group_gap: int = 0): # gap zero means no purge

        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")
        if not isinstance(max_train_group_size, (int, float)) or max_train_group_size <= 0:
             raise ValueError("max_train_group_size must be a positive number.")
        if not isinstance(max_test_group_size, (int, float)) or max_test_group_size <= 0:
             raise ValueError("max_test_group_size must be a positive number.")
        if not isinstance(group_gap, int) or group_gap < 0:
            raise ValueError("group_gap must be a non-negative integer.")

        self.n_splits = n_splits
        self.max_train_group_size = int(max_train_group_size)
        self.max_test_group_size = int(max_test_group_size)
        self.group_gap = group_gap

    def split(self,
              X: Any,
              y: Optional[Any] = None,
              groups: Optional[Iterable[Any]] = None
              ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or similar indexable
            Training data, where n_samples is the number of samples.
        y : array-like of shape (n_samples,), optional
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. Must be provided. The groups are assumed to be
            sortable and define a temporal order.

        Yields
        ------
        train_indices : ndarray
            The training set indices for that split.
        test_indices : ndarray
            The testing set indices for that split.

        Raises
        ------
        ValueError
            If `groups` is None or if folds cannot be formed.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter cannot be None.")

        X, y, groups_arr = indexable(X, y, groups)
        n_samples = _num_samples(X)
        unique_groups, first_indices = np.unique(groups_arr, return_index=True)
        ordered_unique_groups = unique_groups[np.argsort(first_indices)]
        n_groups = len(ordered_unique_groups)

        group_to_indices: Dict[Any, List[int]] = {}
        for idx, group in enumerate(groups_arr):
            if group not in group_to_indices:
                group_to_indices[group] = []
            group_to_indices[group].append(idx)

        n_folds = self.n_splits + 1
        if n_folds > n_groups:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater than "
                f"the number of unique groups={n_groups}.")

        test_group_size = min(n_groups // n_folds, self.max_test_group_size)
        if test_group_size == 0:
             raise ValueError(f"max_test_group_size={self.max_test_group_size} is too small "
                              f"for the number of groups={n_groups} and n_splits={self.n_splits}. "
                              f"Results in zero groups per test split.")


        test_group_starts = range(n_groups - self.n_splits * test_group_size,
                                  n_groups, test_group_size)

        for test_group_start_idx in test_group_starts:
            train_groups_end_idx = test_group_start_idx - self.group_gap

            train_groups_start_idx = max(0, train_groups_end_idx - self.max_train_group_size)

            train_groups_start_idx = max(0, train_groups_start_idx)
            train_groups_end_idx = max(train_groups_start_idx, train_groups_end_idx) # ensure end >= start


            test_groups_end_idx = min(n_groups, test_group_start_idx + test_group_size)

            train_groups = ordered_unique_groups[train_groups_start_idx:train_groups_end_idx]
            test_groups = ordered_unique_groups[test_group_start_idx:test_groups_end_idx]

            train_indices = list(itertools.chain.from_iterable(
                group_to_indices[group] for group in train_groups
            ))
            test_indices = list(itertools.chain.from_iterable(
                group_to_indices[group] for group in test_groups
            ))

            yield np.array(sorted(train_indices), dtype=int), \
                  np.array(sorted(test_indices), dtype=int)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits