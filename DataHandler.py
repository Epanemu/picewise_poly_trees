import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, dataset_path, direct_data=None, round_limit=4, generate_stats=False):
        self.__dataset_path = dataset_path
        if dataset_path is None:
            X, y, categorical_indicator, attribute_names, dataset_name = direct_data
        else:
            with open(dataset_path, "rb") as f:
                X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

        self.__feature_names = attribute_names
        self.__dataset_name = dataset_name
        self.__categorical_indicator = categorical_indicator
        if round_limit > 4:
            print("Rounding to more than 4 decimal numbers may lead to inaccuracies in the model.")
            print(f"Checkout https://www.gurobi.com/documentation/10.0/refman/feasibilitytol.html for a potential way to help this")
        self.__round_limit = round_limit

        # the decision variable must not be a part of data, all data is already numerical
        self.__X = np.array(X, dtype=float)
        self.__y = np.array(y).reshape((-1, 1))

        self.__n_features = self.__X.shape[1]
        if generate_stats:
            self.__generate_stats(self.__X, self.__y)

    def get_training_data(self, split_seed=0, test_size=0.2, limit=np.iinfo(np.int32).max, test_limit=np.iinfo(np.int32).max, reset_stats=True):
        if test_size == 0:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = self.__X, np.zeros((0,self.__n_features)), self.__y, np.zeros((0,))
        else:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=test_size, random_state=split_seed)
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

    def __generate_stats(self, X, y):
        self.__shifts_y = y.min()
        self.__scales_y = (y - self.__shifts_y).max()

        X = X.copy()
        self.__shifts = X.min(axis=0)
        X -= self.__shifts
        self.__scales = X.max(axis=0)
        self.__scales[self.__scales == 0] = 1
        X /= self.__scales
        X = X.round(self.__round_limit) # round all data for clearer interpretation

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

    def normalize(self, X):
        return ((X - self.__shifts) / self.__scales).round(self.__round_limit)

    def unnormalize(self, X):
        return X * self.__scales + self.__shifts

    def normalize_y(self, y):
        return ((y - self.__shifts_y) / self.__scales_y)

    def unnormalize_y(self, y):
        return y * self.__scales_y + self.__shifts_y

    def get_setup(self):
        return {
            "path": self.__dataset_path,
            "round_limit": self.__round_limit,
            "split_seed": self.__split_seed,
            "test_size": self.__test_size,
            "limit": self.__limit,
        }

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
        return self.__X_test[:self.__test_limit], self.__y_test[:self.__test_limit]

    @property
    def all_data(self):
        return self.__X, self.__y
