from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Alternating Proximal gradient
    """
    name = "apg"

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'n_inner_iter': [1, 5],
        'loss': ['euclidean']
    }

    stopping_criterion = SufficientProgressCriterion(
        strategy="callback", key_to_monitor="objective_frobenius"
    )

    def set_objective(self, X, rank, factors_init):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X = X
        self.rank = rank
        self.factors_init = factors_init  # None if not initialized beforehand

    def run(self, callback):
        m, n = self.X.shape
        rank = self.rank
        n_inner_iter = self.n_inner_iter

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(m, rank), np.random.rand(rank, n)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]

        while callback():
            HHt = np.dot(self.H, self.H.T)
            XHt = np.dot(self.X, self.H.T)
            Lw = np.linalg.norm(HHt)  # upper bound of Lw
            # W update
            for _ in range(n_inner_iter):
                self.W = np.maximum(
                    self.W - (np.dot(self.W, HHt) - XHt) / Lw, 0
                )

            # H update
            WtW = np.dot(self.W.T, self.W)
            WtX = np.dot(self.W.T, self.X)
            Lh = np.linalg.norm(WtW)  # upper bound for Lh
            # H update
            for _ in range(n_inner_iter):
                self.H = np.maximum(
                    self.H - (np.dot(WtW, self.H) - WtX) / Lh, 0
                )

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)
