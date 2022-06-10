"""Train and evaluate a generative model on image datasets
"""
import os
import flax
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax.training import checkpoints
from jax import lax
from ml_collections.config_flags import config_flags

import wandb
from src.data.datasets import get_dataset
from src.frax.fractalvae import *
from src.frax.modules import *
from utils.plot_utils import *
from utils.train_utils import *
from utils.eval_utils import evaluate
from jax import value_and_grad

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("cfg", "config/autoencode/collage/cifar.py", 'training config file', lock_config=True)


def train_epoch(epoch, config, state, optimizer, data, preprocess_fn, opt_state):
    metrics, scalar_metrics = {}, {}
    def train_step(x_pre, params, opt_state, rng):
        def loss_fn(params, x_pre, rng):
            loss, stats = state.apply_fn({'params': params}, x_pre, x_pre, rng)
            return loss, stats

        objective_grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, stats), grad = objective_grad_fn(params, x_pre, rng)
        reconstructions, psnr = stats[0], stats[1]
        grad = lax.pmean(grad, axis_name='batch')
        grad = jax.tree_map(lambda x: jnp.nan_to_num(x, nan=0), grad)
        grad, grad_norm = clip_grad_norm(config, grad)
        updates, opt_state = optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        metrics['grad_norm'] = grad_norm
        metrics['mse'] = loss
        return params, opt_state, metrics, reconstructions, psnr

    p_train_step = train_step
    if config.n_devices > 1:
        p_train_step = jax.pmap(train_step, axis_name='batch', in_axes=(0, 0, 0, None)) 

    rng = jax.random.PRNGKey(config.seed)

    params = flax.jax_utils.replicate(state.train_params)
    opt_state = flax.jax_utils.replicate(state.opt_state)
    step = state.step

    for _, batch_data in enumerate(data):
        x_in, x_pre = preprocess_fn(config, batch_data)
        rng, iter_rng = jax.random.split(rng)
        params, opt_state, metrics, reconstructions, psnr = p_train_step(x_pre, params, opt_state, iter_rng)
        step += 1

        if step % config.log_steps == 0:
            single_replica_x_pre = x_pre[0]

            mse = jnp.mean(metrics['mse'])
            psnr = jnp.mean(psnr)
            grad_norm = np.mean(metrics['grad_norm'])
            print(f"batch step {step}: mse {mse}, psnr: {psnr}, ||grad||_2: {grad_norm}")
            scalar_metrics = {"mse": jax.device_get(mse), "psnr": jax.device_get(psnr), "||grad||_2": jax.device_get(grad_norm)}
            wandb.log(scalar_metrics)


            if step % config.plot_steps == 0:
                log_recs(config, norm_two_to_eight_bit(single_replica_x_pre), norm_two_to_eight_bit(reconstructions[0]))
                
            if step % config.ckpt_steps == 0:
                # Only `params` is replicated for distributed training
                if config.n_devices > 1:
                    save_params  = flax.jax_utils.unreplicate(params)
                
                save_dict = {
                    'opt_state':opt_state,
                    'train_params': save_params,
                    'key': rng,
                    'metrics': scalar_metrics,
                    'step': step}

                state = state.replace(**save_dict)
                checkpoints.save_checkpoint(config.checkpoint_dir, jax.device_get(state), step, overwrite=True)
                print(f'checkpointed at epoch {epoch}: step {step}')

            if step > config.max_steps:
                return

    # last save, then return control 
    save_params  = flax.jax_utils.unreplicate(params)
    opt_state  = flax.jax_utils.unreplicate(opt_state)
    save_dict = {
        'opt_state':opt_state,
        'train_params': save_params,
        'key': rng,
        'metrics': scalar_metrics,
        'step': step}
    state = state.replace(**save_dict)
    return state


def evaluate(config, state, valloader, val_rng, preprocess_fn):
    val_ds = iter(valloader) 
    val_mse, val_psnr, val_bpd = 0., 0., 0.
    params = flax.jax_utils.replicate(state.train_params)

    def val_step(x_pre, params, rng):
        def loss_fn(params, x_pre, rng):
            loss, stats = state.apply_fn({'params': params}, x_pre, x_pre, rng)
            reconstructions, psnr = stats[0], stats[1]
            return loss, reconstructions, psnr

        elbo, reconstructions, psnr = loss_fn(params, x_pre, rng)
        return elbo, reconstructions, psnr

    p_val_step = val_step
    n_eval_batches = config.n_eval_batches 
    if n_eval_batches < 0: n_eval_batches = len(val_ds)

    if config.n_devices > 1:
        p_val_step = jax.pmap(val_step, axis_name='batch', in_axes=(0, 0, None)) 

    for k, batch_data in enumerate(val_ds):
        if k >= n_eval_batches:
            break
        
        x_in, x_pre = preprocess_fn(config, batch_data) 
        
        val_rng, val_iter_rng = jax.random.split(val_rng)
        loss, reconstructions, psnr = p_val_step(x_pre, params, val_iter_rng)

        val_mse = val_mse + jax.tree_map(lambda x: jnp.mean(x, 0), loss) 
        val_psnr = val_psnr + jax.tree_map(lambda x: jnp.mean(x, 0), psnr) 

    val_mse, val_psnr = val_mse / n_eval_batches, val_psnr / n_eval_batches
    print("********* Validation metrics ********")
    print(f"val mse: {val_mse:.4f}")

    metrics = {"val mse": val_mse, "val_psnr": val_psnr}
    log_recs(config, norm_two_to_eight_bit(x_pre[0]), norm_two_to_eight_bit(reconstructions[0]), is_val=True)
    return metrics


def main(argv):
    config = FLAGS.cfg
    full_exp_name = f"{config.dataset}-{config.exp_name}"
    config.exp_name = full_exp_name
    config.checkpoint_dir = os.path.join(config.checkpoint_dir, config.dataset, config.model)  
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    trainloader, valloader, testloader, preprocess_fn = get_dataset(config.bs, 
        config.bs, config.dataset, config.datadir, 0.0, config.n_devices, config.subset_size,
    )
    print(f'trainloader {len(trainloader)} valloader {len(valloader)} testloader {len(testloader)}')
    config, state, optimizer, reloaded = initialize_model(config, config.model, config.model, wandb_dir=config.wandb_dir)
    print(f'experiment {full_exp_name} logging to {config.checkpoint_dir} run {config.run_id} wandb {config.wandb_dir}')

    os.environ['WANDB_SILENT'] = "true"
    resuming = reloaded
    print(f'wandb_dir: {config.wandb_dir}')
    state, config = maybe_restart_wandb(resuming, state, config)

    val_rng = jax.random.PRNGKey(config.seed)
    print('number devices:', config.n_devices, 'bs', config.bs)
    print(f'working directory: {config.checkpoint_dir}')

    for epoch in range(config.epochs):
        print(f'.....................Epoch: {epoch}/{config.epochs}....................')
        train_ds = iter(trainloader)
        if config.n_devices > 1: trainloader.sampler.set_epoch(epoch)
        state = train_epoch(epoch, config, state, optimizer, train_ds, preprocess_fn, state.opt_state)
        
        if epoch % config.eval_epochs == 0:
            if config.n_devices > 1: 
                valloader.sampler.set_epoch(epoch)
                testloader.sampler.set_epoch(epoch)

            metrics = evaluate(config, state, testloader, val_rng, preprocess_fn)
            wandb.log(metrics)
  
    metrics = evaluate(config, state, testloader, val_rng, preprocess_fn)
    wandb.log(metrics)

if __name__ == '__main__':
    app.run(main)
