import pickle
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

MAX_INT = np.iinfo(np.int32).max


class DataHandler:

    def __init__(
        self,
        dataset_path: Optional[str],
        direct_data: Optional[tuple] = None,
        round_limit: int = 4,
        generate_stats: bool = False,
    ):
        self.__dataset_path = dataset_path
        if dataset_path is None:
            if direct_data is None:
                raise ValueError(
                    "both dataset_path and direct_data cannot be None at the same time"
                )
            X, y, categorical_indicator, attribute_names, dataset_name = direct_data
        else:
            with open(dataset_path, "rb") as f:
                X, y, categorical_indicator, attribute_names, dataset_name = (
                    pickle.load(f)
                )

        self.__feature_names = attribute_names
        self.__dataset_name = dataset_name
        self.__categorical_indicator = categorical_indicator
        if round_limit > 4:
            print(
                "Rounding to more than 4 decimal numbers may lead to inaccuracies in the model."
            )
            print(
                "Checkout https://www.gurobi.com/documentation/10.0/refman/feasibilitytol.html for a potential way to help this"
            )
        self.__round_limit = round_limit

        # the decision variable must not be a part of data, all data is already numerical
        self.__X = np.array(X, dtype=float)
        self.__y = np.array(y).reshape((-1, 1))

        self.__n_features = self.__X.shape[1]
        if generate_stats:
            self.__generate_stats(self.__X, self.__y)

    def get_training_data(
        self,
        split_seed: int = 0,
        test_size: float = 0.2,
        limit: np.int32 = MAX_INT,
        test_limit: np.int32 = MAX_INT,
        reset_stats: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if test_size == 0:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = (
                self.__X,
                np.zeros((0, self.__n_features)),
                self.__y,
                np.zeros((0,)),
            )
        else:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = (
                train_test_split(
                    self.__X, self.__y, test_size=test_size, random_state=split_seed
                )
            )
        X, y = self.__X_train[:limit], self.__y_train[:limit]
        self.__X_used = X
        self.__y_used = y
        if reset_stats:
            self.__generate_stats(X, y)
            self.__split_seed = split_seed
            self.__test_size = test_size
            self.__limit = limit

        self.__n_data = X.shape[0]
        self.__test_limit = test_limit
        return self.normalize(X), self.normalize_y(y)

    def __generate_stats(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__shifts_y = y.min()
        self.__scales_y = (y - self.__shifts_y).max()

        X = X.copy()
        self.__shifts = X.min(axis=0)
        X -= self.__shifts
        self.__scales = X.max(axis=0)
        self.__scales[self.__scales == 0] = 1
        X /= self.__scales
        X = X.round(self.__round_limit)  # round all data for clearer interpretation

        self.__epsilons = np.empty((self.n_features,))
        for i, col_data in enumerate(X.T):
            col_sorted = col_data.copy()
            col_sorted.sort()
            eps = col_sorted[1:] - col_sorted[:-1]
            eps[eps == 0] = np.inf
            self.__epsilons[i] = eps.min()

        # if all values were same (min was infinity), we want eps nonzero to prevent non-deterministic splitting
        self.__epsilons[self.__epsilons == np.inf] = 1
        self.__epsilons = self.__epsilons.round(self.__round_limit)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.__shifts) / self.__scales).round(self.__round_limit)

    def unnormalize(self, X: np.ndarray) -> np.ndarray:
        return X * self.__scales + self.__shifts

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.__shifts_y) / self.__scales_y

    def unnormalize_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.__scales_y + self.__shifts_y

    def get_setup(self):
        return {
            "path": self.__dataset_path,
            "round_limit": self.__round_limit,
            "split_seed": self.__split_seed,
            "test_size": self.__test_size,
            "limit": self.__limit,
        }

    def set_n_data(self, n_data):
        print("Only perform if you know what you are doing")
        self.__n_data = n_data

    @property
    def round_limit(self):
        return self.__round_limit

    @property
    def n_data(self):
        return self.__n_data

    @property
    def n_features(self):
        return self.__n_features

    @property
    def shifts(self):
        return self.__shifts

    @property
    def scales(self):
        return self.__scales

    @property
    def scales_y(self):
        return self.__scales_y

    @property
    def shifts_y(self):
        return self.__shifts_y

    @property
    def epsilons(self):
        return self.__epsilons

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def categorical_indicator(self):
        return self.__categorical_indicator

    @property
    def used_data(self):
        return self.__X_used, self.__y_used

    @property
    def train_data(self):
        return self.__X_train, self.__y_train

    @property
    def test_data(self):
        return self.__X_test[: self.__test_limit], self.__y_test[: self.__test_limit]

    @property
    def all_data(self):
        return self.__X, self.__y
