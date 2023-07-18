from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from tensorly.datasets.data_imports import load_indian_pines


class Dataset(BaseDataset):

    name = "Indian Pines HSI"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'true_rank': [4],
        'estimated_rank': [4],
        'random_state': [26],
    }

    install_cmd = 'conda'
    requirements = ['pip:tensorly']

    def get_data(self):
        data = load_indian_pines()
        tensor = data["tensor"]
        # shape is 'Spatial dim', 'Spatial dim', 'Hyperspectral bands'
        _, _, n_band = tensor.shape

        # Normalization of the data
        tensor = tensor / np.linalg.norm(tensor)
        # Matricizing into spectra x pixels
        tensor = np.transpose(tensor, [2, 1, 0])
        tensor = np.reshape(tensor, (n_band, -1))

        return dict(X=tensor, rank=self.estimated_rank)
