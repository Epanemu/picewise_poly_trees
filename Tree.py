import numpy as np
try:
    from graphviz import Digraph
except ImportError:
    print("Graphviz not available, will fail if attempted to visualize a tree")

class Tree:
    def __init__(self, context, data_h):

        dec_features = context["a"].argmax(axis=0)
        thresholds = np.clip(context["b"], 0, 1).round(data_h.round_limit) * data_h.scales[dec_features] + data_h.shifts[dec_features]

        self.__model_context = context
        self.__decision_features = dec_features
        self.__thresholds = thresholds

        self.__n_branch_nodes = dec_features.shape[0]

    def assign(self, x):
        i = 0
        while i < self.__n_branch_nodes:
            if x[self.__decision_features[i]] < self.__thresholds[i]:
                i = i*2 + 1
            else:
                i = i*2 + 2
        return i - self.__n_branch_nodes

    def get_assignment(self, X):
        i = np.zeros((X.shape[0],), dtype=int)
        while np.any(i < self.__n_branch_nodes):
            right = X[np.arange(X.shape[0]), self.__decision_features[i]] >= self.__thresholds[i]
            i = i*2 + 1
            i[right] += 1

        return i - self.__n_branch_nodes
