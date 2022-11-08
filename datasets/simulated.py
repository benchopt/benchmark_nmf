import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'm_dim, n_dim, true_rank, estimated_rank': [
            (50, 50, 5, 5)],
        'snr': [100],
    }

    def __init__(self, m_dim=10, n_dim=50, true_rank=5, estimated_rank=6,
                 snr=100, random_state=27):
        # Store the parameters of the dataset
        self.m_dim = m_dim
        self.n_dim = n_dim
        self.true_rank = true_rank
        self.estimated_rank = estimated_rank
        self.snr = snr
        self.random_state = random_state

    def get_data(self):
        '''
        The generated factors are uniform on [0,1], the data is noised
        with elementwise Gaussian iid noise.
        The Signal to Noise ratio is specified by the user.
        '''

        rng = np.random.RandomState(self.random_state)
        W = rng.rand(self.m_dim, self.true_rank)
        H = rng.rand(self.true_rank, self.m_dim)
        X = np.dot(W, H)
        noise = rng.randn(*X.shape)
        sigma = (10**(-self.snr/20))*np.linalg.norm(X)/np.linalg.norm(noise)
        X = X + sigma*noise

        # `data` (this output) holds the keyword arguments for the `set_data`
        #  method of the objective.
        # They are customizable.
        # TODO: return W,H for tracking sources
        return dict(X=X, rank=self.estimated_rank, true_fac=[W, H])
