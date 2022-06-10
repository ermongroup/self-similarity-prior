"""
Train stylizer or compressor.
"""
import os
import shutil

import flax
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import value_and_grad
import numpy as np
import optax
from absl import app, flags
from flax.training import checkpoints
from jax import lax, pmap, vmap
from ml_collections.config_flags import config_flags

import wandb
from utils.transforms import (batched_partition_img,
                                      faster_unpartition_img)
from src.data.datasets import get_dataset
from src.frax.fractalae import *
from src.frax.modules import *

from utils.plot_utils import *
from utils.train_utils import *


DUMMY_STATE = TrainState(train_params=None, ema_params=None, apply_fn=None, sample_fn=None, opt_state=None, metrics={}, key=None, step=None, model_state=None, run_id='')
FLAGS = flags.FLAGS

flags.DEFINE_string("entity", "your_user", "your wandb username")
config_flags.DEFINE_config_file("config", "configs/autoencoding/style_ae.py", 'training config file', lock_config=True)
flags.DEFINE_integer("n_devices", -1, "number of gpu devices.")
flags.DEFINE_string("workdir", "./stylize", "working directory to dump checkpoints and artifacts.")
flags.DEFINE_string("datadir", "/scratch/ssd004/datasets", "parent dataset path")
flags.DEFINE_string("project_name", "style_fractal_ae", "name of wandb project")
flags.DEFINE_string("exp_name", "cifar-fractalstyle-aux", "name of experiment")
flags.DEFINE_bool("f", False, "whether to force a new run and delete checkpoint")


def train_epoch(epoch, config, state, optimizer, data, preprocess_fn, img_partition_fn, opt_state):

    metrics, scalar_metrics = {}, {}
    def train_step(x_pre, params, opt_state, rng):
        def loss_fn(params, x_pre, rng):
            loss, stats = state.apply_fn({'params': params}, x_pre, rng)
            return loss, stats

        objective_grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, stats), grad = objective_grad_fn(params, x_pre, rng)

        reconstructions, psnr, mul, add = stats[0], stats[1], stats[2], stats[3]
        if config.n_devices > 1: grad = lax.pmean(grad, axis_name='batch')
        grad = jax.tree_map(lambda x: jnp.nan_to_num(x, nan=0), grad)
        grad, grad_norm = clip_grad_norm(config, grad)
        updates, opt_state = optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        metrics['grad_norm'] = grad_norm
        metrics['mse'] = loss
        metrics['mul'] = mul
        metrics['add'] = add
        return params, opt_state, metrics, reconstructions, psnr

    p_train_step = train_step
    if config.n_devices > 1:
        p_train_step = jax.pmap(train_step, axis_name='batch', in_axes=(0, 0, 0, None)) 

    rng = jax.random.PRNGKey(config.seed)

    params, opt_state = state.train_params, state.opt_state
    if config.n_devices > 1:
        # unreplicate is not based on `n_devices`, but rather the number of devices visible to JAX
        params = flax.jax_utils.replicate(state.train_params)
        opt_state = flax.jax_utils.replicate(state.opt_state)


    step = state.step

    for _, batch_data in enumerate(data):
        x_in, x_pre = preprocess_fn(config, batch_data) # (n_devices, BS, H, W, C)

        if config.block_dims:
            x_pre = img_partition_fn(x_pre, config.block_dims)
        rng, iter_rng = jax.random.split(rng)
        if len(x_pre.shape) > 4:
            patched_shape = x_pre.shape
            x_pre = x_pre.reshape(config.n_devices, -1, *patched_shape[-3:])

        params, opt_state, metrics, reconstructions, psnr = p_train_step(x_pre, params, opt_state, iter_rng)
        step += 1

        if step % config.log_steps == 0:
            mse = jnp.mean(metrics['mse'])
            psnr = jnp.mean(psnr)
            grad_norm = np.mean(metrics['grad_norm'])
            print(f"batch step {step}: mse {mse}, psnr: {psnr}, ||grad||_2: {grad_norm}")
            scalar_metrics = {"mse": jax.device_get(mse), "psnr": jax.device_get(psnr), "||grad||_2": jax.device_get(grad_norm)}
            wandb.log(scalar_metrics)

            if step == 0 or step % config.ckpt_steps == 0:
                # Only `params` is replicated for distributed training
                save_params = params
                save_opt_state  = opt_state
                if config.n_devices > 1:
                    save_params  = flax.jax_utils.unreplicate(params)
                    save_opt_state  = flax.jax_utils.unreplicate(opt_state)
                save_dict = {
                    'opt_state':save_opt_state,
                    'train_params': save_params,
                    'key': rng,
                    'metrics': scalar_metrics,
                    'step': step}

                state = state.replace(**save_dict)
                checkpoints.save_checkpoint(config.checkpoint_dir, jax.device_get(state), step, overwrite=True)
                print(f'run_id {state.run_id} checkpointed at epoch {epoch}: step {step}')

            if step > config.max_steps:
                return

    # last save, then return control 
    if config.n_devices > 1:
        params  = flax.jax_utils.unreplicate(params)
        opt_state  = flax.jax_utils.unreplicate(opt_state)

    save_dict = {
        'opt_state':opt_state,
        'train_params': params,
        'key': rng,
        'metrics': scalar_metrics,
        'step': step}
    state = state.replace(**save_dict)
    return state

def evaluate(config, state, valloader, val_rng, preprocess_fn, img_partition_fn):
    val_ds = iter(valloader) 
    val_mse, val_psnr, val_bpd = 0., 0., 0.

    params = state.train_params
    if config.n_devices > 1:
        # unreplicate is not based on `n_devices`, but rather the number of devices visible to JAX
        params = flax.jax_utils.replicate(state.train_params)

    def val_step(x_pre, params, rng):
        def loss_fn(params, x_pre, rng):
            loss, stats = state.apply_fn({'params': params}, x_pre, rng, significant_digits=config.significant_digits)
            reconstructions, psnr, mul, add = stats[0], stats[1], stats[2], stats[3]
            return loss, reconstructions, psnr, mul, add

        elbo, reconstructions, psnr, mul, add = loss_fn(params, x_pre, rng)
        return elbo, reconstructions, psnr, mul, add
    p_val_step = val_step

    if config.n_devices > 1:
        p_val_step = jax.pmap(val_step, axis_name='batch', in_axes=(0, 0, None)) 

    n_eval_batches = config.n_eval_batches 
    if n_eval_batches < 0: n_eval_batches = len(val_ds)

    for k, batch_data in enumerate(val_ds):
        if k >= n_eval_batches:
            break

        x_in, x_pre = preprocess_fn(config, batch_data)
        if config.block_dims:
            x_pre = img_partition_fn(x_pre, config.block_dims)
        if len(x_pre.shape) > 4: 
            patched_shape = x_pre.shape
            x_pre = x_pre.reshape(config.n_devices, -1, *patched_shape[-3:])
        
        val_rng, val_iter_rng = jax.random.split(val_rng)

        val_mse = 0.
        val_psnr = 0.
        reconstructions = []
        sub_bs = config.sub_bs if config.sub_bs else x_pre.shape[0] # this prevents the OOM and suboptimal convolutions, but sub_bs must be < the count!
        n_sub_iters, extra_sub_iters = divmod(x_pre.shape[0], sub_bs)

        for k in range(n_sub_iters):
            sub_x_pre = x_pre[sub_bs*k:sub_bs*(k+1)] # WE LOST THE LEADING `1` HERE TOO
            loss, recon, psnr, mul, add = p_val_step(sub_x_pre, params, val_iter_rng)
            reconstructions.append(recon)
            val_mse = val_mse + loss
            val_psnr = val_psnr + psnr

            if extra_sub_iters and k == (n_sub_iters - 1):
                remain_x_pre = x_pre[-extra_sub_iters:]
                loss, recon, psnr, mul, add = p_val_step(remain_x_pre, params, val_iter_rng)
                reconstructions.append(recon)
                val_mse = val_mse + loss
                val_psnr = val_psnr + psnr

        try:
            reconstructions = jnp.concatenate(reconstructions, axis=0)
        except ValueError as e:
            print(f'x_pre: {x_pre.shape} -> n_sub_iters {n_sub_iters} \n\n{e}')

    val_mse, val_psnr = val_mse / n_eval_batches, val_psnr / n_eval_batches

    print("********* Validation metrics ********")
    print(f"val mse: {val_mse:.4f}")

    metrics = {"val mse": val_mse, "val_psnr": val_psnr}
    vmapped_unpartition_fn = vmap(faster_unpartition_img, in_axes=(-1,None,None), out_axes=-1)

    if not config.debug:
        if config.block_dims:
            blocks_h, blocks_w = config.block_dims
            n_blocks_h, n_blocks_w = config.original_data_res // blocks_h, config.original_data_res // blocks_w
            unpartition_reconstruction = vmapped_unpartition_fn(reconstructions, n_blocks_h, n_blocks_w)
            unpartition_x_pre = vmapped_unpartition_fn(x_pre, n_blocks_h, n_blocks_w) 
            log_recs(config, norm_two_to_eight_bit(unpartition_x_pre)[None], norm_two_to_eight_bit(unpartition_reconstruction)[None], is_val=True, name=config.exp_name)
        else: # for when source image is not block-ified when no aux patches are used
            log_recs(config, norm_two_to_eight_bit(x_pre[..., :3])[None], norm_two_to_eight_bit(reconstructions[..., :3])[None], is_val=True, name=f'former c: {config.exp_name}')
            log_recs(config, norm_two_to_eight_bit(x_pre[..., 3:])[None], norm_two_to_eight_bit(reconstructions[..., 3:])[None], is_val=True, name=f'latter c: {config.exp_name}')

    else:
        log_recs(config, norm_two_to_eight_bit(x_pre[:, 0]), norm_two_to_eight_bit(reconstructions[:, 0]), is_val=True, name=config.exp_name)

    return metrics

def partition_fn(img_to_chunk, block_dims):
    "len(img_to_chunk.shape) == 5 pls"
    block_h, block_w = block_dims
    num_blocks_h, num_blocks_w = img_to_chunk.shape[-2] // block_h, img_to_chunk.shape[-3] // block_w

    batched_partition_fn = batched_partition_img
    if len(img_to_chunk.shape) > 4: # expecting `n_devices` leading batch dim
        batched_partition_fn = pmap(batched_partition_fn, axis_name='batch', static_broadcasted_argnums=(1, 2))

    partition_fn = vmap(batched_partition_fn, in_axes=(-1, None, None), out_axes=-1)
    partitioned_source_img, identifiers = partition_fn(img_to_chunk, num_blocks_w, num_blocks_h)

    ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ATTENSHUN !!!!!!!!!!!!!!!!!!!!!!!!!! ####
    partitioned_source_img = partitioned_source_img[0] # <--------- assume to have bs=1 so that the blocks become the new batch
    return partitioned_source_img

def main(argv):
    config = FLAGS.config
    os.environ['WANDB_SILENT'] = "true"
    full_exp_name = f"{config.dataset}-{config.exp_name}"
    FLAGS.exp_name = full_exp_name

    print(f'flags: {FLAGS.n_devices} | pre-override n_devices: {config.n_devices}')
    if FLAGS.n_devices != -1: 
        old_n_devices = config.n_devices
        config.n_devices = FLAGS.n_devices
        config.bs = (config.bs // old_n_devices) * FLAGS.n_devices
    print('number devices:', config.n_devices, 'bs', config.bs)

    config.checkpoint_dir = os.path.join(FLAGS.workdir, f"{config.enc_blocks}", f"{config.n_aux_sources}")

    if FLAGS.f:
        try:
            shutil.rmtree(config.checkpoint_dir)
            print(f'deleted checkpoint_dir {config.checkpoint_dir}')
        except:
            pass

    print(f'working directory: {config.checkpoint_dir}')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if config.eval_on_stitch: 
        assert config.dataset == "dota", "Only dataset 'dota' allowed for evaluation on stitched images."
    if config.dataset == 'dota':
        config.datadir = FLAGS.datadir
    print(f'dataset directory: {config.datadir}')
    
    # TODO caveat: dataset needs to be correct in the config passed since the datadir path is NOT read from the checkpoint -- change in init model / TrainState add attr
    trainloader, valloader, testloader, preprocess_fn = get_dataset(config.bs, 
        config.bs, config.dataset, config.datadir, 0.1, config.n_devices, config.subset_size,
    )
    print(f'trainloader {len(trainloader)} valloader {len(valloader)} testloader {len(testloader)}')

    config, state, optimizer, reloaded = initialize_model(config, FLAGS, config.model, config.sampler, config.wandb_dir)
    val_rng = jax.random.PRNGKey(config.seed)

    os.environ['WANDB_SILENT'] = "true"
    resuming = reloaded
    state, config = maybe_restart_wandb(resuming, state, config)

    if not reloaded:
        # plot single image dataset image
        if config.dataset == "single_source":
            single_img = parse_img_from_path(config.datadir)
            img_train_single = wandb.Image(np.array(single_img), caption=f'single train img ({state.run_id}) {single_img.shape}')
            wandb.log({'Single Image Source': img_train_single})
        # ground truth source
        if config.style_source_img:
            gt_source = parse_img_from_path(config.style_source_img)
            img_gt_source = wandb.Image(np.array(gt_source), caption=f'gt aux ({state.run_id}) {gt_source.shape}')
            wandb.log({'Auxillary Image Source': img_gt_source})
            # pooled down source
            pooled_source = preprocess_style_blocks(config, config.reduce_type)
            img_pooled_source = wandb.Image(np.array(pooled_source), caption=f'aux pooled src ({state.run_id}) {pooled_source.shape}')
            wandb.log({'Auxillary Image Pooled': img_pooled_source})

            print(f'First run: logged single img source, aux img, and pooled aux image')


    for epoch in range(config.epochs):
        print(f'.....................Epoch: {epoch}/{config.epochs}....................')
        train_ds = iter(trainloader)
        if config.n_devices > 1: 
            trainloader.sampler.set_epoch(epoch)
        state = train_epoch(epoch, config, state, optimizer, train_ds, preprocess_fn, partition_fn, state.opt_state)
        
        if epoch % config.eval_epochs == 0:
            if config.n_devices > 1: 
                valloader.sampler.set_epoch(epoch)
                testloader.sampler.set_epoch(epoch)
            metrics = evaluate(config, state, testloader, val_rng, preprocess_fn, partition_fn)
            wandb.log(metrics)
            
    metrics = evaluate(config, state, testloader, val_rng, preprocess_fn, partition_fn)
    wandb.log(metrics)

    print('Finished job successfully - gluck!')

if __name__ == '__main__':
    app.run(main)
