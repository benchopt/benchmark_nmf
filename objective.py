from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.special import kl_div
    # Requires Tensorly >=0.8, postpone implementation
    # import tensorly
    from tensorly.cp_tensor import cp_normalize, cp_permute_factors


class Objective(BaseObjective):
    min_benchopt_version = "1.5"
    name = "Nonnegative Matrix Factorization"
    is_convex = False

    # All parameters 'p' defined here are available as 'self.p'
    parameters = {
        'share_init': [True],
    }

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/tensorly/tensorly@main'
    ]

    def get_one_result(self):
        # Return one solution. This should be compatible with 'self.compute'.
        n, m = self.X.shape
        return dict(W=np.ones((n, self.rank)), H=np.ones((self.rank, m)))

    def set_data(self, X, rank, true_factors=None):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.X = X
        self.rank = rank
        self.true_factors = true_factors

    def evaluate_result(self, W, H):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        WH = np.dot(W, H)
        frobenius_loss = 1/2 * np.linalg.norm(self.X - WH, ord="fro")**2
        kl_loss = np.sum(kl_div(self.X, WH))

        output_dict = {
            'value': frobenius_loss,
            'frobenius': frobenius_loss,
            'kullback-leibler': kl_loss,
        }

        if self.true_factors:
            #  compute factor match score
            #  first, solve permutation and scaling ambiguity
            W_true, H_true = self.true_factors
            Ht = H.T
            Ht_true = H_true.T

            # tensorly version, requires Tensorly >= 0.8
            cp_tl = cp_normalize((None, [W, Ht]))
            cp_tl_true = cp_normalize((None, [W_true, Ht_true]))
            cp_tl, _ = cp_permute_factors(cp_tl_true, cp_tl)
            factor_match_score = np.prod(
                np.diag(cp_tl[1][0].T @ cp_tl_true[1][0]) *
                np.diag(cp_tl[1][1].T @ cp_tl_true[1][1])
            )

            output_dict.update({
                'factor_match_score': factor_match_score
            })
        return output_dict

    def get_objective(self, random_state=27):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        # we compute a random seeded init here to share across methods!
        m, n = self.X.shape
        rank = self.rank
        if self.share_init:
            rng = np.random.RandomState(random_state)
            factors_init = [rng.rand(m, rank), rng.rand(rank, n)]
            for factor in factors_init:
                factor.flags.writeable = False  # Read Only
        else:
            factors_init = None
        return dict(X=self.X, rank=self.rank, factors_init=factors_init)
