from typing import Callable, Optional

import gurobipy as gb
import numpy as np

from DataHandler import DataHandler


class PieceWisePolyTree_MIP:
    def __init__(
        self,
        depth: int,  # D
        data_handler: DataHandler,
        poly_order: int = 1,  # n
        min_in_leaf: int = 1,  # N_min
        use_mse: bool = False,
        axis_aligned: bool = True,
        alpha: float = 0,  # optional coefficient for limiting complexity of the tree itself
        mu: float = 0.0001,  # mu
    ):
        self.depth = depth
        self.data_h = data_handler
        self.min_in_leaf = min_in_leaf
        self.poly_order = poly_order
        self.use_mse = use_mse
        self.axis_aligned = axis_aligned
        self.n_polycombs = self.get_exponents(self.data_h.n_features).shape[1]

        self.__n_leaf_nodes = 2**self.depth
        self.__n_branch_nodes = 2**self.depth - 1
        self.model: Optional[gb.Model] = None
        self.mu = mu
        self.alpha = alpha
        self.optimize_structure = alpha != 0

    def get_exponents(self, dim: int) -> np.ndarray:
        """returns a matrix needed to compute the polynomial combinations of input of dimension dim"""
        if self.poly_order == 0:
            return np.zeros((dim, 0))
        else:
            prev = np.eye(dim)
            res = [prev]
            for _ in range(1, self.poly_order):
                aggregate = []
                for col_i in range(prev.shape[1]):
                    col = prev[:, col_i : col_i + 1]
                    last_nonzero = np.argwhere(col > 0).max()
                    aggregate.append(
                        col
                        + np.concatenate(
                            [
                                np.zeros((last_nonzero, dim - last_nonzero)),
                                np.eye(dim - last_nonzero),
                            ]
                        )
                    )
                prev = np.concatenate(aggregate, axis=1)
                res.append(prev)
        return np.concatenate(res, axis=1)

    def __get_polycombinations(self, X: np.ndarray) -> np.ndarray:
        expo = self.get_exponents(self.data_h.n_features)
        return X @ expo

    def make_model(self, X: np.ndarray, y: np.ndarray):
        """Sets up the model."""
        left_ancestors: list[list[int]] = []  # those where decision went left
        right_ancestors: list[list[int]] = []  # those where decision went right
        for leaf_i in range(self.__n_leaf_nodes):
            left_ancestors.append([])
            right_ancestors.append([])
            prev_i = leaf_i + self.__n_branch_nodes
            for _ in range(self.depth):
                parent_i = (prev_i - 1) // 2
                if (prev_i - 1) % 2:
                    right_ancestors[leaf_i].append(parent_i)
                else:
                    left_ancestors[leaf_i].append(parent_i)
                prev_i = parent_i
        # EXAMPLE
        # node indices for self.depth = 2
        #        0
        #    1       2
        #  3   4   5   6
        #  0   1   2   3 # true indices of leaf nodes
        # print(left_ancestors) # [[1, 0], [0], [2], []]
        # print(right_ancestors) # [[], [1], [0], [2, 0]]

        # MAKE THE MILP MODEL
        self.model = gb.Model("PWPolyTree model")

        d_limit = 1
        if self.optimize_structure:
            d = self.model.addMVar(
                (self.__n_branch_nodes,), vtype=gb.GRB.BINARY, name="d"
            )
            for i in range(1, self.__n_branch_nodes):
                self.model.addConstr(d[i] <= d[(i - 1) // 2])
            d_limit = d

        # branch nodes computation conditions
        if self.axis_aligned:
            a = self.model.addMVar(
                (self.data_h.n_features, self.__n_branch_nodes),
                vtype=gb.GRB.BINARY,
                name="a",
            )
            self.model.addConstr(a.sum(axis=0) == d_limit)
        else:
            a = self.model.addMVar(
                (self.data_h.n_features, self.__n_branch_nodes),
                vtype=gb.GRB.CONTINUOUS,
                lb=-1,
                ub=1,
                name="a",
            )

            apos = self.model.addMVar(
                (self.data_h.n_features, self.__n_branch_nodes),
                vtype=gb.GRB.CONTINUOUS,
                ub=1,
                name="apos",
            )  # a+
            aneg = self.model.addMVar(
                (self.data_h.n_features, self.__n_branch_nodes),
                vtype=gb.GRB.CONTINUOUS,
                ub=1,
                name="aneg",
            )  # a-
            adec = self.model.addMVar(
                (self.data_h.n_features, self.__n_branch_nodes),
                vtype=gb.GRB.BINARY,
                name="adec",
            )  # o
            self.model.addConstr(apos <= adec)
            self.model.addConstr(aneg <= (1 - adec))
            self.model.addConstr(apos.sum(axis=0) + aneg.sum(axis=0) == d_limit)
            self.model.addConstr(a == apos - aneg)

        if self.axis_aligned:
            b = self.model.addMVar(
                (self.__n_branch_nodes,), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="b"
            )
        else:
            b = self.model.addMVar(
                (self.__n_branch_nodes,), lb=-1, ub=1, vtype=gb.GRB.CONTINUOUS, name="b"
            )
        if self.optimize_structure:
            self.model.addConstr(b <= d)
            if not self.axis_aligned:
                self.model.addConstr(b >= -d)

        # leaf nodes assignment conditions
        point_assigned = self.model.addMVar(
            (self.data_h.n_data, self.__n_leaf_nodes),
            vtype=gb.GRB.BINARY,
            name="point_assigned",
        )  # variable z
        any_assigned = self.model.addMVar(
            (self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned"
        )  # variable l
        self.model.addConstr(point_assigned <= any_assigned)
        # if any point is assigned, the node must be assigned at least self.min_in_leaf in total
        self.model.addConstr(
            point_assigned.sum(axis=0) >= any_assigned * self.min_in_leaf
        )
        # points assigned to exactly one leaf
        self.model.addConstr(point_assigned.sum(axis=1) == 1)

        # big-M constants
        if self.axis_aligned:
            M_right = 1
            M_left = 1 + self.data_h.epsilons.max()
        else:
            M_right = 2
            M_left = 2 + self.mu
        # conditions for assignment to node
        for leaf_i in range(self.__n_leaf_nodes):
            if right_ancestors[leaf_i]:  # causes issues if there are no ancestors
                self.model.addConstr(
                    X @ a[:, right_ancestors[leaf_i]]
                    >= b[np.newaxis, right_ancestors[leaf_i]]
                    - M_right * (1 - point_assigned[:, [leaf_i]])
                )
            if left_ancestors[leaf_i]:
                if self.axis_aligned:
                    if self.optimize_structure:
                        mineps = self.data_h.epsilons.min()
                        self.model.addConstr(
                            (X + self.data_h.epsilons - mineps)
                            @ a[:, left_ancestors[leaf_i]]
                            + mineps
                            <= b[np.newaxis, left_ancestors[leaf_i]]
                            + M_left * (1 - point_assigned[:, [leaf_i]])
                        )
                    else:
                        self.model.addConstr(
                            (X + self.data_h.epsilons) @ a[:, left_ancestors[leaf_i]]
                            <= b[np.newaxis, left_ancestors[leaf_i]]
                            + M_left * (1 - point_assigned[:, [leaf_i]])
                        )
                else:
                    self.model.addConstr(
                        X @ a[:, left_ancestors[leaf_i]] + self.mu
                        <= b[np.newaxis, left_ancestors[leaf_i]]
                        + M_left * (1 - point_assigned[:, [leaf_i]])
                    )

        # regression
        X_polycombs = self.__get_polycombinations(X)
        poly_coeffs = self.model.addMVar(
            (self.n_polycombs, self.__n_leaf_nodes),
            lb=float("-inf"),
            vtype=gb.GRB.CONTINUOUS,
            name="poly_coeffs",
        )  # vectors c_t
        intercepts = self.model.addMVar(
            (self.__n_leaf_nodes,),
            lb=float("-inf"),
            vtype=gb.GRB.CONTINUOUS,
            name="intercepts",
        )
        point_error = self.model.addMVar(
            (self.data_h.n_data, self.__n_leaf_nodes),
            lb=float("-inf"),
            vtype=gb.GRB.CONTINUOUS,
            name="point_error",
        )  # variable phi

        self.model.addConstr(
            point_error == y - (X_polycombs @ poly_coeffs + intercepts)
        )

        abs_error = self.model.addMVar(
            (self.data_h.n_data,), vtype=gb.GRB.CONTINUOUS, name="abs_error"
        )  # variable delta

        self.model.addConstrs(
            (point_assigned[i, t].item() == 1)
            >> (abs_error[i].item() >= point_error[i, t].item())
            for i in range(self.data_h.n_data)
            for t in range(self.__n_leaf_nodes)
        )
        self.model.addConstrs(
            (point_assigned[i, t].item() == 1)
            >> (abs_error[i].item() >= -point_error[i, t].item())
            for i in range(self.data_h.n_data)
            for t in range(self.__n_leaf_nodes)
        )

        secondary_terms = 0
        if self.optimize_structure:
            secondary_terms = self.alpha * d.sum()
        if self.use_mse:
            self.model.setObjective(
                (abs_error**2).sum() / self.data_h.n_data + secondary_terms,
                sense=gb.GRB.MINIMIZE,
            )  # MSE objective
        else:
            self.model.setObjective(
                abs_error.sum() / self.data_h.n_data + secondary_terms,
                sense=gb.GRB.MINIMIZE,
            )  # MAE objective

        self.vars = {
            "a": a,
            "b": b,
            "point_assigned": point_assigned,
            "any_assigned": any_assigned,
            "poly_coeffs": poly_coeffs,
            "intercepts": intercepts,
            "point_error": point_error,
            "abs_error": abs_error,
        }

        self.model.update()
        self.model.setParam("Seed", 0)

    def optimize(
        self,
        time_limit: int = 3600,
        mem_limit: Optional[int] = None,
        n_threads: Optional[int] = None,
        mip_focus: int = 0,  # default gurobi
        mip_heuristics: float = 0.05,  # default gurobi
        verbose: bool = False,
        log_file: str = "",
        callback: Optional[Callable] = None,
    ) -> bool:
        if self.model is None:
            raise ValueError("model needs to be set up")

        if verbose:
            self.model.update()
            self.model.printStats()
            self.model.display()
        else:
            if log_file != "":
                self.model.params.LogFile = log_file
                self.model.params.LogToConsole = 0
            else:
                self.model.params.OutputFlag = 0
        self.model.params.TimeLimit = time_limit
        if mem_limit is not None:
            self.model.params.SoftMemLimit = mem_limit
        self.model.params.NodefileStart = 0.5
        self.model.params.NodefileDir = "nodefiles"
        self.model.params.MIPFocus = mip_focus
        self.model.params.Heuristics = mip_heuristics
        if n_threads is not None:
            self.model.params.Threads = n_threads

        if callback is not None:
            self.model.optimize(callback=callback)
        else:
            self.model.optimize()

        return self.model.SolCount > 0  # return whether a solution was found

    def get_humanlike_status(self):
        if self.model.Status == gb.GRB.OPTIMAL:
            return "OPT"
        elif self.model.Status == gb.GRB.INFEASIBLE:
            return "INF"
        elif self.model.Status == gb.GRB.TIME_LIMIT:
            return "TIME"
        elif self.model.Status == gb.GRB.MEM_LIMIT:
            return "MEM"
        elif self.model.Status == gb.GRB.INTERRUPTED:
            return "INT"
        else:
            return f"ST{self.model.status}"

    def get_base_context(self):
        return {
            "depth": self.depth,
            "poly_order": self.poly_order,
            "data_h_setup": self.data_h.get_setup(),
            "use_mse": self.use_mse,
            "axis_aligned": self.axis_aligned,
            "n_polycombs": self.n_polycombs,
            "n_data": self.data_h.n_data,
            "n_features": self.data_h.n_features,
            "min_in_leaf": self.min_in_leaf,
            "a": self.vars["a"].X,
            "b": self.vars["b"].X,
            "poly_coeffs": self.vars["poly_coeffs"].X,
            "intercepts": self.vars["intercepts"].X,
            "objective_bound": self.model.ObjBound if self.model is not None else None,
            "objective_gap": self.model.MIPGap if self.model is not None else None,
            "status": self.get_humanlike_status(),
        }

    def load_sol(self, sol_file: str):
        self.__dummy_model = gb.Model()
        self.__dummy_model.params.OutputFlag = 0

        a = self.__dummy_model.addMVar(
            (self.data_h.n_features, self.__n_branch_nodes),
            vtype=gb.GRB.BINARY,
            name="a",
        )
        b = self.__dummy_model.addMVar(
            (self.__n_branch_nodes,), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="b"
        )
        point_assigned = self.__dummy_model.addMVar(
            (self.data_h.n_data, self.__n_leaf_nodes),
            vtype=gb.GRB.BINARY,
            name="point_assigned",
        )  # variable z
        any_assigned = self.__dummy_model.addMVar(
            (self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned"
        )  # variable l

        poly_coeffs = self.__dummy_model.addMVar(
            (self.n_polycombs, self.__n_leaf_nodes),
            lb=float("-inf"),
            vtype=gb.GRB.CONTINUOUS,
            name="poly_coeffs",
        )  # vectors c_t
        intercepts = self.__dummy_model.addMVar(
            (self.__n_leaf_nodes,),
            lb=float("-inf"),
            vtype=gb.GRB.CONTINUOUS,
            name="intercepts",
        )
        point_error = self.__dummy_model.addMVar(
            (self.data_h.n_data, self.__n_leaf_nodes),
            lb=float("-inf"),
            vtype=gb.GRB.CONTINUOUS,
            name="point_error",
        )  # variable phi

        abs_error = self.__dummy_model.addMVar(
            (self.data_h.n_data,), vtype=gb.GRB.CONTINUOUS, name="abs_error"
        )  # variable delta

        self.vars = {
            "a": a,
            "b": b,
            "point_assigned": point_assigned,
            "any_assigned": any_assigned,
            "poly_coeffs": poly_coeffs,
            "intercepts": intercepts,
            "point_error": point_error,
            "abs_error": abs_error,
        }

        self.__dummy_model.update()
        self.__dummy_model.read(sol_file)
        self.__dummy_model.optimize()

        self.model = None  # should not optimize after this, need to rebuild the model
