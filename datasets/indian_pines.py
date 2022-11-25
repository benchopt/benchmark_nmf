from pathlib import Path

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from io import BytesIO
    from urllib.request import urlopen

    import scipy.io
    import numpy as np

    # Downgrading cryptography is necessary when using OpenSSL3
    import ssl
    MAX_CRYPTO_VERSION = None
    if ssl.OPENSSL_VERSION.startswith("OpenSSL 3"):
        MAX_CRYPTO_VERSION = "36.0.2"
        import cryptography
        if cryptography.__version__ > MAX_CRYPTO_VERSION:
            raise ImportError(
                f"Need to downgrade cryptography to {MAX_CRYPTO_VERSION}"
            )

DATA_URL = "http://ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
# Put this in `data` folder of the benchmark folder
DATA_FILE = Path(__file__).parent / "data" / "indian_pines_corrected.npy"


class Dataset(BaseDataset):

    name = "Indian Pines HSI"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'm_dim': [200],
        'n_dim': [145 ** 2],
        'pixel_subsample': [True],
        'true_rank': [4],
        'estimated_rank': [4],
        'random_state': [26],
    }

    install_cmd = 'conda'
    requirements = [f'cryptography{"=36.0.2" if MAX_CRYPTO_VERSION else ""}']

    def get_data(self):
        '''
        The generated factors are uniform on [0,1], the data is noised
        with elementwise Gaussian iid noise.
        The Signal to Noise ratio is specified by the user.
        '''

        # If the data file does not exist, load it from the web
        if not DATA_FILE.exists():
            # We fetch the data locally the web
            # Code copied from Tensorly import data function,
            # credits to Caglayan Tuna
            DATA_FILE.parent.mkdir(exist_ok=True)
            data = BytesIO(urlopen(DATA_URL).read())
            Xtensor = scipy.io.loadmat(data)["indian_pines_corrected"]
            np.save(DATA_FILE, Xtensor)

        Xtensor = np.load(DATA_FILE)

        # Subsampling otherwise it is too big
        if self.pixel_subsample:
            self.n_dim = 20**2
            Xtensor = Xtensor[20:40, 20:40, :]

        # Normalization of the data
        Xtensor = Xtensor/np.linalg.norm(Xtensor)
        # Matricizing into spectra x pixels
        Xtensor = np.transpose(Xtensor, [2, 1, 0])
        X = np.reshape(Xtensor, [self.m_dim, self.n_dim])

        # `data` (this output) holds the keyword arguments for the `set_data`
        #  method of the objective.
        # They are customizable.
        return dict(X=X, rank=self.estimated_rank)
