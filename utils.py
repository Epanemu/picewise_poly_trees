import numpy as np

from DataHandler import DataHandler
from Tree import Tree


def breaklines(
    dh: DataHandler, tree: Tree, ctx: dict[str, object], val_range: float
) -> list[list[list[float]]]:
    a: np.ndarray = ctx["a"]
    b: np.ndarray = ctx["b"]
    sh_x, sh_y = dh.shifts
    sc_x, sc_y = dh.scales
    lims_x = np.array([-val_range, val_range])
    lims_x = (lims_x - sh_x) / sc_x
    lims_y = np.array([-val_range, val_range])
    lims_y = (lims_y - sh_y) / sc_y
    crosspoints: list[list[list[float]]] = [[] for _ in range(b.shape[0])]

    def between(x, y, points):
        biggerx, biggery = points[0]
        smallerx, smallery = points[1]
        if smallerx > biggerx:
            smallerx, biggerx = biggerx, smallerx
        if smallery > biggery:
            smallery, biggery = biggery, smallery
        return smallerx <= x and biggerx >= x and smallery <= y and biggery >= y

    def valid_y(y):
        return y >= lims_y[0] and y <= lims_y[1]

    def valid_x(x):
        return x >= lims_x[0] and x <= lims_x[1]

    def correct_side(i, x, y):
        if i == 0:
            return True
        depth_of_i = np.floor(np.log2(i + 1)).astype(int)
        n_leaves_of_i = 2 ** (ctx["depth"] - depth_of_i)
        first_on_same_depth = 2**depth_of_i - 1
        n_prev_subtrees = i - first_on_same_depth
        allowed_leaves = list(
            range(
                n_prev_subtrees * n_leaves_of_i, (n_prev_subtrees + 1) * n_leaves_of_i
            )
        )
        leaf = tree.get_assignment(np.array([[x, y]]), normalized=True)
        return leaf in allowed_leaves

    for i in range(b.shape[0]):
        parent_i = (i - 1) // 2
        while parent_i >= 0:
            parent_line = np.cross(
                list(crosspoints[parent_i][0]) + [1],
                list(crosspoints[parent_i][1]) + [1],
            )
            intersection = np.cross(parent_line, list(a[:, i]) + [-b[i]])
            if intersection[2] != 0:
                intersection /= intersection[2]
                x, y = intersection[:2]
                # to battle the <= / > constraints in the tree
                diffs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * 1e-4
                correct = any([correct_side(i, x + dx, y + dy) for dx, dy in diffs])
                if valid_x(x) and valid_y(y) and correct:
                    crosspoints[i].append(intersection[:2])
            parent_i = (parent_i - 1) // 2
        y_vals = (b[i] - a[0, i] * lims_x) / a[1, i]
        x_vals = (b[i] - a[1, i] * lims_y) / a[0, i]
        for y, x in zip(y_vals, lims_x):
            if valid_y(y) and correct_side(i, x, y):
                crosspoints[i].append([x, y])
        for y, x in zip(lims_y, x_vals):
            if valid_x(x) and correct_side(i, x, y):
                crosspoints[i].append([x, y])

    for points in crosspoints:
        for i in range(len(points)):
            points[i][1] = points[i][1] * sc_y + sh_y
            points[i][0] = points[i][0] * sc_x + sh_x

    return crosspoints
