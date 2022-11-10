import numpy as np

from benchopt import BaseDataset, safe_import_context

#with safe_import_context() as import_ctx:
from urllib.request import urlopen
import scipy.io
from io import BytesIO
import cryptography

class Dataset(BaseDataset):

    name = "Indian Pines HSI"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'm_dim, n_dim, pixel_subsample, true_rank, estimated_rank': [
            (200, 145**2, True, 4, 4)]
    }

    install_cmd = 'conda'
    requirements = ['cryptography=36.0.2']

    def __init__(self, m_dim=10, n_dim=50, true_rank=5, estimated_rank=6,
                 snr=100, random_state=26, pixel_subsample=True):
        # Store the parameters of the dataset
        self.m_dim = m_dim
        self.n_dim = n_dim
        self.true_rank = true_rank
        self.estimated_rank = estimated_rank
        self.random_state = random_state

    def get_data(self):
        '''
        The generated factors are uniform on [0,1], the data is noised
        with elementwise Gaussian iid noise.
        The Signal to Noise ratio is specified by the user.
        '''

        # We fetch the data locally the web
        # Code copied from Tensorly import data function, credits to Caglayan Tuna
        # Requires to downgrade cryptography package
        # conda install cryptography==36.0.2
        url = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
        r = urlopen(url)
        Xtensor = scipy.io.loadmat(BytesIO(r.read()))["indian_pines_corrected"]
        # Subsampling otherwise it is too big
        if self.pixel_subsample:
            self.n_dim = 20**2
            Xtensor = Xtensor[20:40, 20:40,:]

        # Normalization of the data
        Xtensor = Xtensor/np.linalg.norm(Xtensor)
        # Matricizing into spectra x pixels
        Xtensor = np.transpose(Xtensor,[2,1,0])
        X = np.reshape(Xtensor,[self.m_dim, self.n_dim])

        # `data` (this output) holds the keyword arguments for the `set_data`
        #  method of the objective.
        # They are customizable.
        return dict(X=X, rank=self.estimated_rank)
