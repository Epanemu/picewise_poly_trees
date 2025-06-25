import pickle

import numpy as np

from DataHandler import DataHandler
from PWPolyTree_MIP import PieceWisePolyTree_MIP
from Tree import Tree
from utils import breaklines

val_range = 100  # range of values of features [-val_range, val_range]
n_data = 250  # samples used for the MIP solve
n_sample = 1000  # sample for the ground truth visualization
n_features = 2  # 2D cone function


def cone(sample: np.ndarray) -> float:
    s = 0.5
    r = 0.5
    res = max(abs(sample))
    # res = 0
    if sample[0] > 0:
        if max(abs(sample[1:])) < r * sample[0]:
            res = -s * sample[0] + ((1 + s) / r) * max(abs(sample)[1:])
    return res


def fit_cone(
    depth: int, time_limit: int, poly_order: int = 1, axis_aligned: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)
    X = np.random.uniform(-val_range, val_range, (n_data, n_features))
    y = np.array([cone(x) for x in X])

    dh = DataHandler(
        None,
        direct_data=(X, y, None, [f"x{i}" for i in range(n_features)], "Direct_data"),
    )
    X_train, y_train = dh.get_training_data(test_size=0)

    pwptree = PieceWisePolyTree_MIP(
        depth, dh, poly_order=poly_order, axis_aligned=axis_aligned
    )

    pwptree.make_model(X_train, y_train)

    pwptree.optimize(time_limit=time_limit)
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
            gt[i, j] = cone(np.array([x1[i, j], x2[i, j]]))

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
    )


results = {}
results["d2,t5,p1"] = fit_cone(2, 60 * 5)  # in the paper
# results["d2,t10,p1"] = fit_cone(2, 60 * 10)
results["d3,t10,p1"] = fit_cone(3, 60 * 10)  # in the paper
# results["d3,t20,p1"] = fit_cone(3, 60 * 20)

with open("cones.pickle", "wb") as f:
    pickle.dump(results, f)
