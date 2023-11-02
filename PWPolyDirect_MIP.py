import gurobipy as gb
import numpy as np

class PieceWisePolyDirect_MIP:
    def __init__(self, segments, data_handler, poly_order=1, min_in_leaf=1, use_mse=False):
        self.segments = segments
        self.data_h = data_handler

        assert self.data_h.n_features == 1, "Cannot handle multidimensional data"
        assert poly_order == 1, "Cannot handle different order yet"

        self.poly_order = poly_order
        self.use_mse = use_mse
        self.n_polycombs = self.get_exponents(self.data_h.n_features).shape[1]

        self.model = None

    def get_exponents(self, dim):
        if self.poly_order == 0:
            return np.zeros((dim, 0))
        if self.poly_order == 1:
            return np.eye(dim)
        else:
            raise Exception("invalid argument, higher orders not yet implemented")

    def __get_polycombinations(self, X):
        expo = self.get_exponents(self.data_h.n_features)
        return X @ expo

    def make_model(self, X, y):
        

        # MAKE THE MILP MODEL
        self.model = gb.Model("PWPolyDirect model")

        # branch nodes computation conditions
        b = self.model.addMVar((self.segments+1,), ub=1, vtype=gb.GRB.CONTINUOUS, name="breakpoints")
        c = self.model.addMVar((self.segments,), vtype=gb.GRB.CONTINUOUS, name="intercepts")
        beta = self.model.addMVar((self.segments,), vtype=gb.GRB.CONTINUOUS, name="slopes")

        phi = self.model.addMVar((self.data_h.n_data, self.segments), vtype=gb.GRB.BINARY, name="0_if_belongs_to_seg")

        xi = self.model.addMVar((self.data_h.n_data,), vtype=gb.GRB.CONTINUOUS, name="abs_err")
        
        self.model.addConstrs((b[i] <= b[i+1] for i in range(self.segments)), "preserve_order")
        # model.addConstrs(b[0] <= X, "in_interval1")
        # model.addConstrs(X <= b[self.segments], "in_interval2") # TODO simpler if I know x0 and xn
        self.model.addConstr(b[0] == X.min(), "ininterval1")
        self.model.addConstr(b[self.segments] == X.max(), "ininterval2")

        self.model.addConstrs((-phi[:,j] <= b[j+1] - X[:,0] for j in range(self.segments)), "assign_segment1")
        self.model.addConstrs((-phi[:,j] <= X[:,0] - b[j] for j in range(self.segments)), "assign_segment2")

        self.model.addConstr((phi.sum(axis=1) == self.segments-1), "assign_to_one_seg")

        # point_error = self.model.addMVar((self.data_h.n_data, self.segments), lb=float('-inf'), vtype=gb.GRB.CONTINUOUS, name="point_error")
        # self.model.addConstr(point_error == y - beta* + intercepts))


        self.model.addConstrs((phi[i, j].item() == 0) >> (xi[i].item() >= y[i] - c[j] - beta[j]*X[i])
                                    for i in range(self.data_h.n_data) for j in range(self.segments))
        self.model.addConstrs((phi[i, j].item() == 0) >> (xi[i].item() >= -(y[i] - c[j] - beta[j]*X[i]))
                                    for i in range(self.data_h.n_data) for j in range(self.segments))

        # regression
        # X_polycombs = self.__get_polycombinations(X)
        # poly_coeffs = self.model.addMVar((self.n_polycombs, self.__n_leaf_nodes), lb=float('-inf'), vtype=gb.GRB.CONTINUOUS, name="poly_coeffs") # vectors c_t (together with intercept)
        # intercepts = self.model.addMVar((self.__n_leaf_nodes,), lb=float('-inf'), vtype=gb.GRB.CONTINUOUS, name="intercepts")
        # point_error = self.model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), lb=float('-inf'), vtype=gb.GRB.CONTINUOUS, name="point_error") # variable phi
        # self.model.addConstr(point_error == y - (X_polycombs @ poly_coeffs + intercepts))

        # abs_error = self.model.addMVar((self.data_h.n_data,), vtype=gb.GRB.CONTINUOUS, name="abs_error") # variable delta

        # self.model.addConstrs((point_assigned[i, t].item() == 1) >> (abs_error[i].item() >= point_error[i, t].item())
        #                             for i in range(self.data_h.n_data) for t in range(self.__n_leaf_nodes))
        # self.model.addConstrs((point_assigned[i, t].item() == 1) >> (abs_error[i].item() >= -point_error[i, t].item())
        #                             for i in range(self.data_h.n_data) for t in range(self.__n_leaf_nodes))

        if self.use_mse:
            self.model.setObjective((xi**2).sum() / self.data_h.n_data, sense=gb.GRB.MINIMIZE) # MSE objective
        else:
            self.model.setObjective(xi.sum() / self.data_h.n_data, sense=gb.GRB.MINIMIZE) # MAE objective

        self.vars = {
            "slopes": beta,
            "intercepts": c,
            "breakpoints": b,
            "phi": phi,
            "abs_errs": xi,
        }

        self.model.update()


    def optimize(self, time_limit=3600, mem_limit=None, n_threads=None, mip_focus=0, mip_heuristics=0.05, verbose=False, log_file=""):
        assert self.model is not None

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

        self.model.optimize()

        return self.model.SolCount > 0 # return whether a solution was found

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
            "segments": self.segments,
            "poly_order": self.poly_order,
            "data_h_setup": self.data_h.get_setup(),
            "use_mse": self.use_mse,
            "n_polycombs": self.n_polycombs,
            "n_data": self.data_h.n_data,
            "n_features": self.data_h.n_features,
            "breakpoints": self.vars["breakpoints"].X,
            "slopes": self.vars["slopes"].X,
            "intercepts": self.vars["intercepts"].X,
            "objective_bound": self.model.ObjBound if self.model is not None else None,
            "objective_gap": self.model.MIPGap if self.model is not None else None,
            "status": self.get_humanlike_status(),
        }

    # def load_sol(self, sol_file):
    #     self.__dummy_model = gb.Model()
    #     self.__dummy_model.params.OutputFlag = 0

    #     a = self.__dummy_model.addMVar((self.data_h.n_features, self.__n_branch_nodes), vtype=gb.GRB.BINARY, name="a")
    #     b = self.__dummy_model.addMVar((self.__n_branch_nodes,), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="b")
    #     point_assigned = self.__dummy_model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="point_assigned") # variable z
    #     any_assigned = self.__dummy_model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned") # variable l
    #     class_points_in_leaf = self.__dummy_model.addMVar((self.data_h.n_classes, self.__n_leaf_nodes), name="N_class_points_in_leaf") # variable N_kt
    #     points_in_leaf = self.__dummy_model.addMVar((self.__n_leaf_nodes,), name="N_points_in_leaf") # variable N_t

    #     poly_coeffs = self.__dummy_model.addMVar((self.n_polycombs, self.__n_leaf_nodes), lb=float('-inf'), vtype=gb.GRB.CONTINUOUS, name="poly_coeffs") # vectors c_t
    #     intercepts = self.model.addMVar((self.__n_leaf_nodes,), lb=float('-inf'), vtype=gb.GRB.CONTINUOUS, name="intercepts")
    #     point_error = self.__dummy_model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), lb=float('-inf'), vtype=gb.GRB.CONTINUOUS, name="point_error") # variable phi

    #     abs_error = self.__dummy_model.addMVar((self.data_h.n_data,), vtype=gb.GRB.CONTINUOUS, name="abs_error") # variable delta

    #     self.vars = {
    #         "a": a,
    #         "b": b,
    #         "point_assigned": point_assigned,
    #         "any_assigned": any_assigned,
    #         "poly_coeffs": poly_coeffs,
    #         "intercepts": intercepts,
    #         "point_error": point_error,
    #         "abs_error": abs_error,
    #     }

    #     self.__dummy_model.update()
    #     self.__dummy_model.read(sol_file)
    #     self.__dummy_model.optimize()

    #     self.model = None # should not optimize after this, need to rebuild the model
