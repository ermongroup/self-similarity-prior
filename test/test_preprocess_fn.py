import os
import pytest
import ml_collections
from src.data.datasets import get_dataset

DATA_DIR = '/home/datasets/' # insert your path

@pytest.mark.parametrize('dataset', ['mnist', 'ffhq256'])
def test_scale_norm(dataset):
    """Checks whether the `preprocess_fn` (normalizing input and output data) keeps sensible ranges"""

    dummy_config = {
        'dataset': dataset,
        'likelihood_func': 'dmol',
        'n_devices': 1
    }
    dummy_config = ml_collections.ConfigDict(dummy_config)
    datadir = os.path.join(DATA_DIR, dataset)
    trainloader, _, _, preprocess_fn = get_dataset(4, 4, dataset, datadir, 0.0, 1, None)
    batch= next(iter(trainloader))
    x_in, x_out = preprocess_fn(dummy_config, batch)
    # in reality the bounds should be much tighter, but good initial check
    print(f'x_in min: {x_in.min()}, x_in max: {x_in.max()}, x_out min: {x_out.min()}, x_out max: {x_out.max()}')
    assert x_in.max() <= 5. and x_in.min() >= -5 
    assert x_out.max() <= 5. and x_out.min() >= -5