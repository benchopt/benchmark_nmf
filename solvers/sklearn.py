from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    from sklearn.decomposition import NMF
    import numpy as np


class Solver(BaseSolver):
    '''
    Alternating Proximal gradient
    '''
    name = "sklearn"
    install_cmd = 'conda'
    requirements = ['scikit-learn']

    # any parameter defined here is accessible as a class attribute
    parameters = {
        "strategy": ["cd", "mu"],
        # Other loss choices: float, "kullback-leibler", "itakura-saito"
        "loss": ["frobenius","kullback-leibler"]
    }

    stopping_criterion = SufficientProgressCriterion(strategy="iteration", key_to_monitor="objective_frobenius")

    def skip(self, X, rank, factors_init):
        if self.loss == "kullback-leibler" and self.strategy == "cd":
            return True, (
                f"{self.name} only implements the MU strategy "
                "for the chosen {self.loss} loss"
            )

        return False, None

    def set_objective(self, X, rank, factors_init):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X = X
        self.rank = rank
        self.stopping_criterion.key_to_monitor="objective_"+self.loss
        if factors_init:
            # creating the scikit-learn problem instance
            self.factors_init = factors_init
            self.clf = NMF(
                n_components=rank, init="custom", solver=self.strategy,
                beta_loss=self.loss, tol=0, max_iter=1e32
            )
        else:
            self.clf = NMF(n_components=rank, solver=self.strategy,
                           lbeta_loss=self.loss, tol=0, max_iter=1e32)

    def run(self, n_iter):
        # sklearn doesn't work with max_iter=0
        self.clf.max_iter = max(n_iter,1)# + 1

        if self.clf.init == "custom":
            self.W = self.clf.fit_transform(
                self.X, W=np.copy(self.factors_init[0]),
                H=np.copy(self.factors_init[1])
            )
        else:
            self.W = self.clf.fit_transform(self.X)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return [self.W, self.clf.components_]
