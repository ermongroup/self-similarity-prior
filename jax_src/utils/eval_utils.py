import json
import typing as T
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from utils.transforms import *
from src.data.datasets import *
from src.frax.fractalvae import *
from src.frax.modules import *

                        
from utils.vae_utils import compute_number_parameters
from utils.plot_utils import log_plots, log_samples, norm_two_to_eight_bit

def evaluate(config, state, valloader, val_rng, preprocess_fn, n_dec_blocks, log_samples_to_wandb=True):
    # For CIFAR10 we will eventually need to do random splits for evaluation (like VDVAE paper), so this is fine.
    val_ds = iter(valloader) 
    val_rate, val_rate_blocks, val_distortion, val_elbo = 0., [0. for _ in range(n_dec_blocks)], 0., 0.
    nan_count = 0
    params = flax.jax_utils.replicate(state.ema_params)

    def val_step(x, x_pre, params, rng):
        def loss_fn(params, x, x_pre, rng):
            metrics, reconstructions, _  = state.apply_fn({'params': params}, x, x_pre, rng, return_sample=True)
            return metrics['elbo'], (metrics, reconstructions)

        elbo, metrics = loss_fn(params, x, x_pre, rng)
        return elbo, metrics

    p_val_step = val_step
    n_eval_batches = config.n_eval_batches 
    if n_eval_batches < 0: n_eval_batches = len(val_ds)

    p_val_step = jax.pmap(val_step, axis_name='batch', in_axes=(0, 0, 0, None)) 

    for k, batch_data in enumerate(val_ds):
        if k >= n_eval_batches:
            break
        x_in, x_pre = preprocess_fn(config, batch_data) 
        val_rng, val_iter_rng = jax.random.split(val_rng)
        elbo, (metrics, reconstructions) = p_val_step(x_in, x_pre, params, val_iter_rng)

        rate, rate_blocks, distortion = metrics['kl'], metrics['kl_blocks'], metrics['recloss']
        elbo = distortion + rate
        rate = jax.tree_map(lambda x: jnp.mean(x, 0), rate) 
        rate_blocks = [jax.tree_map(lambda x: jnp.mean(x, 0), rate_blocks[k]) for k in range(n_dec_blocks)]
        distortion = jax.tree_map(lambda x: jnp.mean(x, 0), distortion) 
        elbo = jax.tree_map(lambda x: jnp.mean(x, 0), elbo)
        val_rate, val_distortion, val_elbo = val_rate + rate , val_distortion + distortion, val_elbo + elbo
        val_rate_blocks = [val_rate_blocks[k] + rate_blocks[k]  for k in range(n_dec_blocks)]
        nan_count = nan_count + metrics['nan_count']

    val_rate, val_distortion, val_elbo  = val_rate / n_eval_batches , val_distortion / n_eval_batches , val_elbo / n_eval_batches 
    val_rate_blocks = [jax.device_get(val_rate_blocks[k]) / n_eval_batches for k in range(n_dec_blocks)]
    val_elbo = jax.device_get(val_elbo)
    val_rate = jax.device_get(val_rate)
    val_distortion = jax.device_get(val_distortion)
    val_elbo_bpd = val_elbo / np.log(2)

    nan_count = jax.device_get(nan_count)
    print("********* Validation metrics ********")
    print(f"\nELBO (nats): {val_elbo:.4f}, ELBO (bpd): {val_elbo_bpd:.4f}, rate (nats): {val_rate:.4f}, distortion (nats): {val_distortion:.4f}")
    print(f"Rate (nats) per decoder block: {val_rate_blocks}")
    print(f"NaN count: {nan_count}")
    
    metrics = {"val elbo (nats)": val_elbo, "val rate": val_rate, "val distortion": val_distortion, "val rate (blocks)": val_rate_blocks}

    if log_samples_to_wandb:
        samples = state.sample_fn({'params': flax.jax_utils.unreplicate(params)}, config.n_samples, val_rng)
        if len(config.superres_eval_factors) > 0:
            for superres_factor in config.superres_eval_factors:
                sh = sw = config.data_res * superres_factor
                superres_samples = state.sample_fn({'params': flax.jax_utils.unreplicate(params)}, config.n_samples, val_rng, superres_factor=superres_factor)
                log_samples(config, jnp.array(superres_samples), is_val=True, name=f'superres:{superres_factor} dec{config.dec_blocks}')

        log_plots(config, norm_two_to_eight_bit(x_pre[0]), jnp.array(reconstructions[0]), samples, is_val=True, name=f'dec{config.dec_blocks}')


    return metrics

def generate_and_plot(state, n_samples, temperatures, rng, save_path):
    """

    Args:
        config (_type_): _description_
        state (_type_): _description_
        valloader (_type_): _description_
        n_samples (_type_): _description_
        temperatures (_type_): _description_
        rng (_type_): _description_
    """
    params = state.ema_params
    for t in temperatures:
        _, rng = jax.random.split(rng)
        samples = state.sample_fn({'params': params}, n_samples, rng, temperature=t)
        log_samples(samples, is_val=True, save_path=save_path, name=f'val_samples_{t}', log_to_wandb=False)    