from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import copy
    from tensorly.decomposition._nn_cp import non_negative_parafac
    from tensorly.decomposition._nn_cp import non_negative_parafac_hals

#TODO fix??

class Solver(BaseSolver):
    '''
    HALS and MU implementations in tensorly
    '''
    name = "tensorly"

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'strategy': ['MU', 'HALS']
    }

    stopping_criterion = SufficientProgressCriterion(strategy="iteration", key_to_monitor="objective_frobenius")

    #stopping_strategy = "iteration" TODO: callback ?

    install_cmd = 'conda'
    requirements = ['pip:tensorly']

    def set_objective(self, X, rank, factors_init):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X = X
        self.rank = rank
        if factors_init:
            self.init = (None, [factors_init[0], factors_init[1].T])
        else:
            self.init = 'random'

    def run(self, n_iter):
        if self.strategy == 'MU':
            out = non_negative_parafac(
                self.X, self.rank, n_iter_max=n_iter,
                init=copy.deepcopy(self.init), tol=0
            )
        else:
            out = non_negative_parafac_hals(
                self.X, self.rank, n_iter_max=n_iter,
                init=copy.deepcopy(self.init), tol=0
            )
        factors = out[1]
        self.factors = [factors[0], factors[1].T]

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.factors
