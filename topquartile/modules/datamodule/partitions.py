from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


class _BasePurgedTimeSeriesSplit(_BaseKFold, metaclass=ABCMeta):
    @_deprecate_positional_args
    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
        verbose: bool = False,
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.verbose = verbose

    @abstractmethod
    def split(self, X, y=None, groups=None):
        raise NotImplementedError


class PurgedTimeSeriesSplit(_BasePurgedTimeSeriesSplit):
    def split(self, X, y=None, groups=None):
        if groups is not None:
            raise ValueError(
                "PurgedTimeSeriesSplit does not use groups "
            )

        X, y = indexable(X, y)
        n_samples = _num_samples(X)
        n_folds = self.n_splits + 1

        test_size = (
            self.test_size
            if self.test_size is not None
            else n_samples // n_folds
        )
        if test_size == 0:
            raise ValueError("test_size becomes 0 – PLEAS reduce n_splits")

        indices = np.arange(n_samples)
        test_starts = range(
            n_samples - self.n_splits * test_size, n_samples, test_size
        )

        for fold, test_start in enumerate(test_starts):
            test_end = test_start + test_size
            train_end = test_start - self.gap
            if train_end <= 0:
                raise ValueError(
                    "With these settings the gap eats the entire train set."
                )

            if self.max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_size)

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]

            if self.verbose:
                print(
                    f"[fold {fold}] train:{train_start}:{train_end}  "
                    f"test:{test_start}:{test_end}"
                )

            yield train_idx, test_idx


class PurgedGroupTimeSeriesSplit(_BasePurgedTimeSeriesSplit):
    """
    Stolen from kaggle - https://www.kaggle.com/code/marketneutral/purged-time-series-cv-xgboost-optuna
    """
    @_deprecate_positional_args
    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_group_size: int | None = None,
        max_test_group_size: int | None = None,
        gap: int = 0,
        verbose: bool = False,
    ):
        super().__init__(
            n_splits,
            max_train_size=max_train_group_size,
            test_size=max_test_group_size,
            gap=gap,
            verbose=verbose,
        )

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("You must pass a 'groups' array.")

        X, y, groups = indexable(X, y, groups)
        unique_groups, group_idx = np.unique(groups, return_index=True)
        unique_groups = unique_groups[np.argsort(group_idx)]
        n_groups = unique_groups.size
        n_folds = self.n_splits + 1

        test_group_size = (
            self.test_size
            if self.test_size is not None
            else n_groups // n_folds
        )
        if test_group_size == 0:
            raise ValueError("test_group_size becomes 0 – reduce n_splits")

        group_locs = {g: np.flatnonzero(groups == g) for g in unique_groups}

        test_starts = range(
            n_groups - self.n_splits * test_group_size,
            n_groups,
            test_group_size,
        )

        for fold, g_start in enumerate(test_starts):
            g_stop = g_start + test_group_size
            train_g_stop = g_start - self.gap
            if train_g_stop <= 0:
                raise ValueError(
                    "With these settings the gap eats the entire train set."
                )

            if self.max_train_size is None:
                train_g_start = 0
            else:
                train_g_start = max(0, train_g_stop - self.max_train_size)

            train_groups = unique_groups[train_g_start:train_g_stop]
            test_groups = unique_groups[g_start:g_stop]

            train_idx = np.concatenate([group_locs[g] for g in train_groups])
            test_idx = np.concatenate([group_locs[g] for g in test_groups])

            if self.verbose:
                print(
                    f"[fold {fold}] "
                    f"train groups {train_groups[0]}–{train_groups[-1]}, "
                    f"test groups {test_groups[0]}–{test_groups[-1]}"
                )

            yield np.sort(train_idx), np.sort(test_idx)