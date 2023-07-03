from pathlib import Path
from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from tensorly.datasets.data_imports import load_indian_pines 

class Dataset(BaseDataset):

    name = "Indian Pines HSI"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'm_dim': [200],
        'n_dim': [145 ** 2],
        'true_rank': [4],
        'estimated_rank': [4],
        'random_state': [26],
    }

    install_cmd = 'conda'
    requirements = ['pip:tensorly']

    def get_data(self):
        '''
        The generated factors are uniform on [0,1], the data is noised
        with elementwise Gaussian iid noise.
        The Signal to Noise ratio is specified by the user.
        '''

        data = load_indian_pines()
        tensor = data["tensor"]

        # Normalization of the data
        tensor = tensor/np.linalg.norm(tensor)
        # Matricizing into spectra x pixels
        tensor = np.transpose(tensor, [2, 1, 0])
        tensor = np.reshape(tensor, [self.m_dim, self.n_dim])

        # `data` (this output) holds the keyword arguments for the `set_data`
        #  method of the objective.
        # They are customizable.
        return dict(X=tensor, rank=self.estimated_rank)
