import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'm_dim, n_dim, true_rank, estimated_rank': [
            (40, 40, 3, 3)],
        'snr': [100],
    }

    def __init__(self, m_dim=10, n_dim=50, true_rank=3, estimated_rank=3,
                 snr=100, random_state=26):
        # Store the parameters of the dataset
        self.m_dim = m_dim
        self.n_dim = n_dim
        self.true_rank = true_rank
        self.estimated_rank = estimated_rank
        self.snr = snr
        self.random_state = random_state

    def get_data(self):
        """
        The generated factors are uniform on [0, 1], elementwise Gaussian iid
        noise is added to the data matrix.
        The Signal to Noise ratio is specified by the user.
        """

        rng = np.random.RandomState(self.random_state)
        W = rng.rand(self.m_dim, self.true_rank)
        H = rng.rand(self.true_rank, self.n_dim)
        X = np.dot(W, H)
        noise = rng.randn(*X.shape)
        sigma = 10**(-self.snr/20) * (
            np.linalg.norm(X, ord="fro") / np.linalg.norm(noise, ord="fro")
        )
        X += sigma*noise
        return dict(X=X, rank=self.estimated_rank, true_factors=[W, H])
