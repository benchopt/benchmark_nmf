from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np
    # importing scipy for KL div
    from scipy.special import kl_div


class Objective(BaseObjective):
    name = "Nonnegative Matrix Factorization"
    is_convex = False

    # All parameters 'p' defined here are available as 'self.p'
    parameters = {
        'share_init': [True],
        'loss_type': ['frobenius','kl']
    }

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        n, m = self.X.shape
        return np.ones((n, self.rank)), np.ones((self.rank, m))

    def set_data(self, X, rank):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        # TODO: handle W and H known in only some cases, to track source
        # identification
        self.X = X
        self.rank = rank

    def compute(self, fac):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        # Note: one particular metric should be used to check convergence, thus the logic on outputs. Maybe better way?
        output_dic = {}
        if 'frobenius' in self.loss_type:
            # If frobenius is asked, use it to check convergence
            W, H = fac
            frob = 1/2*np.linalg.norm(self.X - np.dot(W, H))**2
            output_dic.update({'value': frob})
        if 'kl' in self.loss_type:
            # If KL is asked but not frobenius, use it to check convergence, otherwise it is a secondary return
            W, H = fac
            kl_loss = np.sum(kl_div(self.X, np.dot(W, H)))
            if 'frobenius' in self.loss_type:
                output_dic.update({'kl': kl_loss})
            else:
                output_dic.update({'value': kl_loss})
        return output_dic

    def to_dict(self, random_state=27):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        # we compute a random seeded init here to share across methods!
        m, n = self.X.shape
        rank = self.rank
        if self.share_init:
            rng = np.random.RandomState(random_state)
            fac_init = [rng.rand(m, rank), rng.rand(rank, n)]
            for factor in fac_init:
                factor.flags.writeable = False  # Read Only
        else:
            fac_init = None
        return dict(X=self.X, rank=self.rank, fac_init=fac_init)
