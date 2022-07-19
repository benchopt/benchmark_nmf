from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Nonnegative Matrix Factorization"

    # All parameters 'p' defined here are available as 'self.p'
    parameters = {
        'fit_intercept': [False],
        'rank': 1 #TODO check right pos?
    }

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        # Solutions of NMF are length two lists with W and H
        m, n = self.X.shape
        rank = self.rank
        return [np.zeros((m, rank)),np.zeros((rank, n))]

    def set_data(self, X, rank):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.X, self.rank = X, rank

    def compute(self):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        # TODO: also allow other losses
        frob = np.linalg.norm(self.X - self.fac[0]@self.fac[1])**2 # TODO: self.fac
        return frob

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(X=self.X)
