"""Evaluate a generative model loadable from checkpoint.

The checkpoint loading strategy follows `flax.training.checkpoints.latest_checkpoint`,
applied to `checkpoint_dir` as defined in CLI flags. In case of evaluations to be performed not on latest
checkpoints in a folder, delete / or move the unwanted checkpoints before proceeding.
"""

import os
import flax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from absl import app, flags
from flax.training import checkpoints
from jax import lax
from ml_collections.config_flags import config_flags
from PIL import Image

import wandb
from src.data.datasets import get_dataset
from src.frax.fractalvae import *
from src.frax.modules import *
from utils.plot_utils import *
from utils.train_utils import *
from utils.eval_utils import *
from jax import value_and_grad

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", ".", "checkpoint directory to load pretrained model from")
flags.DEFINE_boolean("test", True, "`True` to evaluate on testloader, `False` to evaluate on valloader")

def main(argv):
    assert os.path.isdir(FLAGS.checkpoint_dir), "Provided checkpoint path does not exist"
    config_path = os.path.join(FLAGS.checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f: config = json.load(f)
    config = ml_collections.config_dict.create(**config)
    config.checkpoint_dir, config.restore_from_checkpoint = FLAGS.checkpoint_dir, True

    # Data and model setup
    config, state, _, _ = initialize_model(config, config.model, config.model)

    _, valloader, testloader, preprocess_fn = get_dataset(config.bs, 
        config.bs, config.dataset, config.datadir, 0.0, config.n_devices, config.subset_size,
    )

    val_rng = jax.random.PRNGKey(config.seed)
    loader = testloader if FLAGS.test else valloader
    _, n_dec_blocks = parse_layer_string(config.model, config.dec_blocks)
    print(f'Number of devices in use: {config.n_devices}')

    # Eval and plot samples
    _ = evaluate(config, state, loader, val_rng, preprocess_fn, n_dec_blocks, log_samples_to_wandb=False)
    img_save_dir = os.path.join(FLAGS.checkpoint_dir, 'plots')
    os.makedirs(img_save_dir, exist_ok=True)
    generate_and_plot(state, n_samples=64, temperatures=[0.3, 0.6, 0.9, 1.0, 1.1, 1.2, 1.5], rng=val_rng, save_path=img_save_dir)

if __name__ == '__main__':
    app.run(main)