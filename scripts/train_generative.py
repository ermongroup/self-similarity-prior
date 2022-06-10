"""Train and evaluate a generative model on image datasets
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
config_flags.DEFINE_config_file("cfg", "config/default_cifar.py", 'training config file', lock_config=True)

def train_epoch(epoch, config, state, optimizer, data, preprocess_fn, opt_state):
    metrics = {}
    
    def train_step(x_in, x_out, params, rng, opt_state):
        def loss_fn(params, x_in, x_out, rng):
            metrics, reconstructions, stats = state.apply_fn({'params': params}, x_in, x_out, rng, return_sample=True)
            return metrics['elbo'], (metrics, reconstructions, stats)

        objective_grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (_, (metrics, reconstructions, stats)), grad = objective_grad_fn(params, x_in, x_out, rng)

        grad = lax.pmean(grad, axis_name='batch')
        grad = jax.tree_map(lambda x: jnp.nan_to_num(x, nan=0), grad)
        grad, grad_norm = clip_grad_norm(config, grad)
        updates, opt_state = optimizer.update(grad, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        ema_params = jax.tree_util.tree_multimap(lambda new, old: (1 - config.ema_rate) * new + config.ema_rate * old, new_params, params)
        metrics['grad_norm'] = grad_norm

        return new_params, ema_params, opt_state, metrics, stats, reconstructions

    p_train_step = train_step
    params = state.train_params
    opt_state = state.opt_state

    p_train_step = jax.pmap(train_step, axis_name='batch', in_axes=(0, 0, 0, None, 0)) 
    params = flax.jax_utils.replicate(state.train_params)
    opt_state = flax.jax_utils.replicate(state.opt_state)

    rng = jax.random.PRNGKey(config.seed)


    forward_fn, sample_fn = state.apply_fn, state.sample_fn
    
    step = state.step

    for _, batch_data in enumerate(data):
        x_in, x_out = preprocess_fn(config, batch_data)
        rng, iter_rng = jax.random.split(rng)
        params, ema_params, opt_state, metrics, stats, reconstructions = p_train_step(x_in, x_out, params, iter_rng, opt_state)
        step += 1
        
        if step % config.log_steps == 0:
            single_params = flax.jax_utils.unreplicate(params)
            single_replica_x_out = x_out[0]
            elbo, rate, distortion, forward_time = metrics['elbo'], metrics['kl'], metrics['recloss'], metrics['forward_time']

            grad_norm = np.mean(metrics['grad_norm'])
            forward_time = jnp.mean(metrics['forward_time'])
            elbo, rate, distortion = jax.device_get(jnp.mean(elbo)), jax.device_get(jnp.mean(rate)), jax.device_get(jnp.mean(distortion))

            print(f"batch step {step}: ELBO {elbo}, rate: {rate}, distortion: {distortion}, forward (s): {forward_time}, ||grad||_2: {grad_norm}")
            metrics = {"ELBO": elbo, "rate": rate, "distortion": distortion, "forward (s)": jax.device_get(forward_time), '||grad||_2': jax.device_get(grad_norm)}
            wandb.log(metrics, step=step)

            if step % config.plot_steps == 0:
                rng, sample_rng = random.split(rng)
                samples = sample_fn({'params': single_params}, config.n_samples, sample_rng)
                # norm_cond = False if config.dataset != 'ffhq256' else True
                norm_cond = False # TODO: this doesn't seem to ever be True, double check this 
                log_plots(config, norm_two_to_eight_bit(single_replica_x_out), reconstructions[0], samples, name=f'dec{config.dec_blocks} ({state.run_id})', normalize=norm_cond)
                if config.model == "FractalVAE":

                    mul, add = stats['mul'], stats['add']

                    for c in range(mul.shape[-1]):
                        mul_, add_ = jnp.mean(mul[..., c], axis=(0,1)), jnp.mean(add[..., c], axis=(0,1)) # 16, 7
                        columns = [f"s{i}" for i in range(mul_.shape[-1])]
                        mul_data = jax.device_get(mul_)
                        add_data = jax.device_get(add_)
                        mul_data = [mul_data[k] for k in range(mul_data.shape[0])]
                        add_data = [add_data[k] for k in range(add_data.shape[0])]

                        mul_table = wandb.Table(data=mul_data, columns=columns)
                        add_table = wandb.Table(data=add_data, columns=columns)

                        wandb.log({'mul histograms': mul_table}, step=step)
                        wandb.log({'add histograms': add_table}, step=step)

                        # width = 0.15
                        # y_pos = np.arange(16)
                        # fig = plt.figure()
                        # axs = fig.subplots(nrows=2, ncols=1)
                        # for d in range(mul_.shape[1]):
                        #     axs[0].bar(y_pos + d*width, mul_[:,d], width)
                        #     axs[1].bar(y_pos + d*width, add_[:,d], width)
                        # axs[0].set_title('Mul')
                        # axs[1].set_title('Add')
                        # wandb.log({'Histograms (Interactive)': plt})
                        # wandb.log({'Histograms': wandb.Image(fig)}) # wandb automatically closes these plots
                        

                        # wandb.log({f'mul histograms ({c}/{COL})': wandb.plot.histogram(mul_table, columns[COL], title=f'mul parameter distribution: {c}'), f'add histograms ({c}/{COL})': wandb.plot.histogram(add_table, columns[COL], title=f'add parameter distribution: {c}')})

            if step == 0 or step % config.ckpt_steps == 0:
                # Only `params` is replicated for distributed training
                save_params = params
                save_opt_state  = opt_state
                save_params  = flax.jax_utils.unreplicate(params)
                save_ema_params = flax.jax_utils.unreplicate(ema_params)
                save_dict = {
                    'opt_state': save_opt_state,
                    'train_params': save_params,
                    'ema_params': save_ema_params,
                    'key': rng,
                    'metrics': metrics,
                    'step': step}

                state = state.replace(**save_dict)
                checkpoints.save_checkpoint(
                    config.checkpoint_dir, jax.device_get(state), step, 
                    overwrite=True, keep_every_n_steps=100000
                )
                print(f'run_id {state.run_id} checkpointed at epoch {epoch}: step {step}')

            if step > config.max_steps:
                return state

    # last save, then return control 
    params  = flax.jax_utils.unreplicate(params)
    opt_state  = flax.jax_utils.unreplicate(opt_state)
    ema_params = flax.jax_utils.unreplicate(ema_params)
    save_dict = {
        'opt_state':opt_state,
        'train_params': params,
        'ema_params': ema_params,
        'key': rng,
        'metrics': metrics,
        'step': step}
    state = state.replace(**save_dict)
    return state


def main(argv):
    config = FLAGS.cfg
    full_exp_name = f"{config.dataset}-{config.exp_name}"
    config.exp_name = full_exp_name
    config.checkpoint_dir = os.path.join(config.checkpoint_dir, config.dataset, config.model, str(config.beta))  
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
    _, n_dec_blocks = parse_layer_string(config.model, config.dec_blocks)

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

            metrics = evaluate(config, state, testloader, val_rng, preprocess_fn, n_dec_blocks)
            wandb.log(metrics)
  
    metrics = evaluate(config, state, testloader, val_rng, preprocess_fn, n_dec_blocks)
    wandb.log(metrics)

if __name__ == '__main__':
    app.run(main)
