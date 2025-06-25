import numpy as np

from DataHandler import DataHandler


class Tree:
    def __init__(self, context: dict, data_h: DataHandler):
        dec_features = context["a"].argmax(axis=0)
        self.dh = data_h

        self.__axis_aligned = context["axis_aligned"]
        self.__a = context["a"]
        self.__decision_features = dec_features
        self.__thresholds = np.clip(context["b"], 0, 1)
        self.__thresholdsl = np.clip(context["b"], -1, 1)

        self.__n_branch_nodes = dec_features.shape[0]

    def assign(self, x: np.ndarray) -> int:
        i = 0
        while i < self.__n_branch_nodes:
            if (
                self.__axis_aligned
                and x[self.__decision_features[i]] < self.__thresholds[i]
            ) or (
                not self.__axis_aligned and x @ self.__a[:, i] < self.__thresholdsl[i]
            ):
                i = i * 2 + 1
            else:
                i = i * 2 + 2
        return i - self.__n_branch_nodes

    def get_assignment(self, X: np.ndarray, normalized: bool = False) -> np.ndarray:
        i = np.zeros((X.shape[0],), dtype=int)
        if not normalized:
            X = self.dh.normalize(X.copy())
        while np.any(i < self.__n_branch_nodes):
            if self.__axis_aligned:
                right = (
                    X[np.arange(X.shape[0]), self.__decision_features[i]]
                    >= self.__thresholds[i]
                )
            else:
                right = (
                    np.einsum("np,pn->n", X, self.__a[:, i]) >= self.__thresholdsl[i]
                )
            i = i * 2 + 1
            i[right] += 1

        return i - self.__n_branch_nodes
