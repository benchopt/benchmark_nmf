import numpy as np


from benchopt import BaseSolver


class Solver(BaseSolver):
    '''
    Alternating Proximal gradient
    '''
    name = "Alternating Proximal Gradient, without acceleration"

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'n_inner_iter': [1, 5]
    }

    def set_objective(self, X):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X = X

    def run(self, n_iter, n_inner_iter):
        m, n = self.X.shape
        rank = self.rank

        # initialization with random values (TODO: improve?)
        W = np.zeros((m,rank))
        H = np.zeros((rank,n))

        for _ in range(n_iter):
            HHt = np.dot(H, H.T)
            XHt = np.dot(self.X, H.T)
            Lw = np.linalg.norm(HHt,2)
            # W update
            for inner in range(n_inner_iter):
                W -= (np.dot(W, HHt) - XHt ) / Lw

            # H update
            WtW = np.dot(W.T, W)
            WtX = np.dot(W.T, self.X)
            Lh = np.linalg.norm(WtW,2)
            # W update
            for inner in range(n_inner_iter):
                H -= (np.dot(WtW, H) - WtX ) / Lh

        self.fac = [W, H]

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.fac
