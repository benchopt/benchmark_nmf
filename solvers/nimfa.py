from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import nimfa
    import numpy as np

class Solver(BaseSolver):
    '''
    MU implementations in nimfa
    TODO: change loss depending on user input? cf scikit learn
    '''
    name = "nimfa"

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'strategy': ['MU', 'ALS-PG']
    }

    #stopping_strategy = "callback"

    install_cmd = 'conda'
    requirements = ['pip:nimfa']

    def set_objective(self, X, rank, factors_init):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X = X
        self.rank = rank
        if factors_init:
            self.init = factors_init
        else:
            self.init = 'random'

    def run(self, n_iter):
        if self.init == 'random':
            W = np.random.rand(self.X.shape[0], self.rank)
            H = np.random.rand(self.rank, self.X.shape[1])
        else:
            W = np.copy(self.init[0])
            H = np.copy(self.init[1])            
        
        if self.strategy ==  'MU':
            nmf = nimfa.Nmf(self.X, rank=self.rank, update='euclidean', max_iter=n_iter, 
                            min_residual=0, W=W, H=H, test_conv=0)
        elif self.strategy ==  'ALS-PG':
            # TODO: Add inner_sub_iter to parameters ?
            nmf = nimfa.Lsnmf(self.X, rank=self.rank, max_iter=n_iter, sub_iter=10,
                              inner_sub_iter=10, beta=0.1, W=W, H=H)
        else:
            raise ValueError("Strategy not suported")

        nmf_fit = nmf()
        W = nmf_fit.basis()
        H = nmf_fit.coef()
        self.factors = [W,H]

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        # for nimfa, recast to avoid broadcasting and argmax errors
        W = np.array(self.factors[0])
        H = np.array(self.factors[1])
        return [W,H]
