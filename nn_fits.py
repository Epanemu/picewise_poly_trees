import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from DataHandler import DataHandler
from PWPolyTree_MIP import PieceWisePolyTree_MIP
from Tree import Tree
from utils import breaklines


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx].float()


class NNModel:
    def __init__(self, input_size, hidden_sizes, maxpools, activations, output_size):
        layers = []
        prev_size = input_size
        for curr_size, maxpool, activation in zip(hidden_sizes, maxpools, activations):
            layers.append(nn.Linear(prev_size, curr_size))
            layers.append(self.__parse_activation(activation))
            prev_size = curr_size
            if maxpool > 1:
                layers.append(nn.MaxPool1d(maxpool))
                prev_size //= maxpool
        layers.append(nn.Linear(prev_size, output_size))
        # no ReLU on the output
        self.layers = layers

        self.model = nn.Sequential(*layers)

        self.loss_f = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters())

    def __parse_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        if activation == "sigmoid":
            return nn.Sigmoid()
        if activation == "tanh":
            return nn.Tanh()
        raise ValueError(f"'{activation}' is not a recognized activation function")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            res = self.model(x)
        return res.numpy()

    def train(self, X_train, y_train, epochs=50, batch_size=64):
        dataset = SimpleDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("Training:")
        self.model.train()
        for _ in tqdm(range(epochs)):
            for _, (X, y) in enumerate(dataloader):
                y_pred = self.model(X)

                loss = self.loss_f(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test(self, X_test, y_test):
        dataset = SimpleDataset(X_test, y_test)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        print("Testing:")
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _, (X, y) in enumerate(dataloader):
                y_pred = self.model(X)
                losses.append(self.loss_f(y_pred, y).item())
        print("Average loss:", sum(losses) / y_test.shape[0])


n_features = 2
val_range = 2  # range of values of features [-val_range, val_range]
n_data = 15  # samples used for the NN
n_sample = 1000  # sample for the ground truth visualization


def ground_truth(sample):
    return 2 * np.sin(sample[0]) + 2 * np.cos(sample[1]) + sample[1] / val_range


np.random.seed(0)
X = np.random.uniform(-val_range, val_range, (n_data, n_features))
y = np.array([ground_truth(x) for x in X]) + np.random.normal(0, 0.1, (n_data,))

dh = DataHandler(
    None, direct_data=(X, y, None, [f"x{i}" for i in range(n_features)], "Direct_data")
)
X_train, y_train = dh.get_training_data(test_size=0)

torch.random.manual_seed(42)
nn_model = NNModel(2, [4, 2, 2], [2, 0, 0], ["relu", "sigmoid", "tanh"], 1)
nn_model.train(X_train, y_train, batch_size=n_data, epochs=5000)

n_tree_samples_w = 15
n_tree_samples = n_tree_samples_w * n_tree_samples_w
x1, x2 = np.meshgrid(
    np.linspace(-val_range, val_range, n_tree_samples_w),
    np.linspace(-val_range, val_range, n_tree_samples_w),
)

NNspace = dh.normalize(np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1))
nn_vals = nn_model.predict(NNspace)
dh.set_n_data(n_tree_samples)


def fit_NN(depth, time_limit, poly_order=1, axis_aligned=False):
    pwptree = PieceWisePolyTree_MIP(
        depth, dh, poly_order=poly_order, axis_aligned=axis_aligned
    )

    pwptree.make_model(NNspace, nn_vals)

    pwptree.optimize(time_limit=time_limit)
    ctx = pwptree.get_base_context()
    print(ctx["status"])

    x1, x2 = np.meshgrid(
        np.linspace(-val_range, val_range, n_sample),
        np.linspace(-val_range, val_range, n_sample),
    )
    gt = np.zeros_like(x1)
    for i in range(n_sample):
        for j in range(n_sample):
            gt[i, j] = ground_truth(np.array([x1[i, j], x2[i, j]]))

    X_space = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)

    nn_visualize = dh.unnormalize_y(nn_model.predict(dh.normalize(X_space))).reshape(
        (n_sample, n_sample)
    )

    tree = Tree(ctx, dh)
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
        nn_visualize,
        gt,
        x1,
        x2,
        dh.unnormalize(X_train),
        dh.unnormalize(NNspace),
        breaklines(dh, tree, ctx, val_range),
    )


results = {}
# results["d2,t5,p1"] = fit_NN(2, 60*5)
# results["d2,t10,p1"] = fit_NN(2, 60*10)
# results["d2,t10,p2"] = fit_NN(2, 60*10, 2)
results["d2,t30,p2"] = fit_NN(2, 60 * 30, 2)  # used in the paper
# results["d2,t60,p2"] = fit_NN(2, 60 * 60, 2)
# results["d3,t10,p1"] = fit_NN(3, 60*10)
# results["d3,t10,p2"] = fit_NN(3, 60*10, 2)
# results["d3,t20,p2"] = fit_NN(3, 60*20, 2)
results["d3,t60,p2"] = fit_NN(3, 60 * 60, 2)  # used in the paper

with open("NNfits.pickle", "wb") as f:
    pickle.dump(results, f)
