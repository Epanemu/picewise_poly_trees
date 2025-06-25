import pickle
import time

import gurobipy as gb
import numpy as np

from DataHandler import DataHandler
from PWPolyTree_MIP import PieceWisePolyTree_MIP
from Tree import Tree
from utils import breaklines

n_features = 2
val_range = 100  # range of values of features [-val_range, val_range]
n_data = 100  # samples used for the MIP solve
n_sample = 1000  # sample for the ground truth visualization


def norm(sample):
    return max(abs(sample))


def data_cb(model, where):
    if where == gb.GRB.Callback.MIP:
        cur_obj = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gb.GRB.Callback.MIP_OBJBND)

        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd


def fit(depth, time_limit, poly_order=1, axis_aligned=False):
    np.random.seed(0)
    X = np.random.uniform(-val_range, val_range, (n_data, n_features))
    y = np.array([norm(x) for x in X])

    dh = DataHandler(
        None,
        direct_data=(X, y, None, [f"x{i}" for i in range(n_features)], "Direct_data"),
    )
    X_train, y_train = dh.get_training_data(test_size=0)

    pwptree = PieceWisePolyTree_MIP(
        depth, dh, poly_order=poly_order, axis_aligned=axis_aligned
    )

    pwptree.make_model(X_train, y_train)

    pwptree.model._obj = None
    pwptree.model._bd = None
    pwptree.model._data = []
    t = time.time()
    pwptree.model._start = t
    pwptree.optimize(time_limit=time_limit, log_file="full_run.log", callback=data_cb)
    tot_time = time.time() - t
    print(tot_time)
    ctx = pwptree.get_base_context()
    print(ctx["status"])

    tree = Tree(ctx, dh)

    x1, x2 = np.meshgrid(
        np.linspace(-val_range, val_range, n_sample),
        np.linspace(-val_range, val_range, n_sample),
    )
    gt = np.zeros_like(x1)
    for i in range(1000):
        for j in range(1000):
            gt[i, j] = norm(np.array([x1[i, j], x2[i, j]]))

    X_space = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)

    X_leaf = tree.get_assignment(X_space)

    expo = pwptree.get_exponents(n_features)
    leaf_values = dh.unnormalize_y(
        dh.normalize(X_space) @ expo @ ctx["poly_coeffs"] + ctx["intercepts"]
    )

    computed_vals = leaf_values[range(leaf_values.shape[0]), X_leaf].reshape(
        (n_sample, n_sample)
    )
    return (
        computed_vals,
        gt,
        x1,
        x2,
        dh.unnormalize(X_train),
        breaklines(dh, tree, ctx, val_range),
        tot_time,
        pwptree.model._data,
    )


with open("opt_norm.pickle", "wb") as f:
    pickle.dump(fit(2, 60 * 60 * 10, 1, False), f)
