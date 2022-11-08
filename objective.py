from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np
    # importing scipy for KL div
    from scipy.special import kl_div
    # Requires Tensorly >=0.7
    #from tensorly.cp_tensor import cp_normalize


class Objective(BaseObjective):
    name = "Nonnegative Matrix Factorization"
    is_convex = False

    # All parameters 'p' defined here are available as 'self.p'
    parameters = {
        'share_init': [True],
        # losses will be computed on different runs
        'loss_type': ['frobenius', 'kl']  # TODO: 'all' for all losses simult
    }

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        n, m = self.X.shape
        return np.ones((n, self.rank)), np.ones((self.rank, m))

    def set_data(self, X, rank, true_fac=None):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        # TODO: handle W and H known in only some cases, to track source
        # identification
        self.X = X
        self.rank = rank
        self.true_fac = true_fac

    def compute(self, fac):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        # Note: one particular metric should be used to check convergence,
        # thus the logic on outputs.
        if 'frobenius' == self.loss_type:
            # If frobenius is asked, use it to check convergence
            W, H = fac
            value = 1/2*np.linalg.norm(self.X - np.dot(W, H))**2
        if 'kl' == self.loss_type:
            # If KL is asked but not frobenius, use it to check convergence,
            #  otherwise it is a secondary return
            W, H = fac
            value = np.sum(kl_div(self.X, np.dot(W, H)))
        if self.true_fac:
            #  compute factor match score
            #  first, solve permutation and scaling ambiguity
            W, H = fac
            W_true, H_true = self.true_fac
            W = W/np.linalg.norm(W, axis=0)
            H = H/np.linalg.norm(H, axis=1)
            W_true = W_true/np.linalg.norm(W_true, axis=0)
            H_true = H_true/np.linalg.norm(H_true, axis=1)
            # TODO: suboptimal permutation for now, want to use tensorly but bugged
            perms = np.argmax((W.T@W_true)*(H@H_true.T), axis=1)
            W = W[:, perms]
            H = H[perms, :]
            fms = np.prod(np.diag(W.T@W_true)*np.diag(H@H_true.T))
            return {'value': value, 'fms': fms}
        return value

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
