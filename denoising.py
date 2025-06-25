import pickle

import numpy as np

from DataHandler import DataHandler
from noise_sample_data import generate_data
from PWPolyTree_MIP import PieceWisePolyTree_MIP
from Tree import Tree

val_range = 25
time_limit = 3600
depth = 4
sigma = 0.5  # original paper uses sigma = 1

scenarios = list(range(1, 5))  # paper by Madrid also shows only scenarios 1-4

gt = [None] * len(scenarios)
y = [None] * len(scenarios)
computed_vals = [None] * len(scenarios)
for i, scenario in enumerate(scenarios):
    np.random.seed(scenario)
    gt_i, y_i, patches = generate_data(val_range, scenario, sigma=sigma)

    X = np.indices(y_i.shape).transpose(1, 2, 0).reshape(-1, 2)
    y_i = y_i.reshape(-1, 1)

    dh = DataHandler(
        None,
        direct_data=(
            X,
            y_i,
            None,
            [f"x{i}" for i in range(2)],
            f"Direct_data{scenario}",
        ),
    )
    X_train, y_train = dh.get_training_data(test_size=0)

    pwptree = PieceWisePolyTree_MIP(depth, dh, poly_order=0)

    pwptree.make_model(X_train, y_train)

    pwptree.optimize(time_limit=time_limit)
    ctx = pwptree.get_base_context()
    (ctx["status"], pwptree.model.getObjective().getValue())
    tree = Tree(ctx, dh)

    X_leaf = tree.get_assignment(X)

    expo = pwptree.get_exponents(2)
    leaf_values = dh.unnormalize_y(
        dh.normalize(X) @ expo @ ctx["poly_coeffs"] + ctx["intercepts"]
    )

    computed_vals[i] = leaf_values[range(leaf_values.shape[0]), X_leaf].reshape(
        gt_i.shape
    )
    y[i] = y_i
    gt[i] = gt_i

with open("denoising.pickle", "wb") as f:
    pickle.dump([gt, y, computed_vals], f)
