"""Train a fractal autoencoder (EAE) to solve the inverse fractal problem. Stylization and compression of images
Currently supported models: EAE
Currently supported datasets: CIFAR10
"""

import os

import flax
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import value_and_grad
import numpy as np
import optax
import torch
from absl import app, flags
from flax.training import checkpoints
from ml_collections.config_flags import config_flags

import wandb
from src.data.datasets import get_dataset
from src.frax.fractalae import *
from src.frax.modules import *
from utils.plot_utils import *
from utils.train_utils import *


FLAGS = flags.FLAGS
flags.DEFINE_string("entity", "your_user", "your wandb username")
config_flags.DEFINE_config_file("config", "configs/autoencoding/nfe_vanilla.py", 'training config file', lock_config=True)
flags.DEFINE_integer("n_devices", 1, "number of gpu devices.")
flags.DEFINE_string("workdir", "./checkpoints/ae_dota/", "working directory to dump checkpoints and artifacts.")
flags.DEFINE_string("datadir", "../../data", "parent dataset path") 

def train_epoch(epoch, config, state, optimizer, data, preprocess_fn, opt_state):

    metrics, scalar_metrics = {}, {}
    def train_step(x_pre, params, opt_state, rng):
        def loss_fn(params, x_pre, rng):
            loss, stats = state.apply_fn({'params': params}, x_pre, rng)
            return loss, stats

        objective_grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, stats), grad = objective_grad_fn(params, x_pre, rng)

        reconstructions, psnr, mul, add = stats[0], stats[1], stats[2], stats[3]
        grad = lax.pmean(grad, axis_name='batch')
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
            loss, stats = state.apply_fn({'params': params}, x_pre, rng, significant_digits=config.significant_digits)
            reconstructions, psnr, mul, add = stats[0], stats[1], stats[2], stats[3]
            return loss, reconstructions, psnr, mul, add

        elbo, reconstructions, psnr, mul, add = loss_fn(params, x_pre, rng)
        return elbo, reconstructions, psnr, mul, add

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
        loss, reconstructions, psnr, mul, add = p_val_step(x_pre, params, val_iter_rng)

        val_mse = val_mse + jax.tree_map(lambda x: jnp.mean(x, 0), loss) 
        val_psnr = val_psnr + jax.tree_map(lambda x: jnp.mean(x, 0), psnr) 


    val_mse, val_psnr = val_mse / n_eval_batches, val_psnr / n_eval_batches

    print("********* Validation metrics ********")
    print(f"val mse: {val_mse:.4f}")

    metrics = {"val mse": val_mse, "val_psnr": val_psnr}
    log_recs(config, norm_two_to_eight_bit(x_pre[0]), norm_two_to_eight_bit(reconstructions[0]), is_val=True)

    # evaluate on 120 x 120 and 1200 x 1200 aerial images
    if config.eval_on_stitch and config.n_devices > 1 and config.dataset == "dota":
        d120x120_all = torch.load(f"{config.datadir}/dota/val/images/preprocess_val_120x120") / 255.
        d1200x1200_all = torch.load(f"{config.datadir}/dota/val/images/preprocess_val_1200x1200") / 255.

        # n_devices should be 1 or 2, else last stitches might not be eval.
        n_stitch_iters = len(d120x120_all) // config.n_devices 

        vmapped_partition_fn = vmap(vmap(partition_img, in_axes=(-1, None, None), out_axes=-1), in_axes=(0,None,None))
        vmapped_unpartition_fn = vmap(vmap(faster_unpartition_img, in_axes=(0,None,None)), in_axes=(-1,None,None), out_axes=-1)
        for stitch_id in range(n_stitch_iters):
            d120x120 = d120x120_all[stitch_id * config.n_devices : (stitch_id + 1) * config.n_devices]
            d1200x1200 = d1200x1200_all[stitch_id * config.n_devices : (stitch_id + 1) * config.n_devices]

            _, d120x120 = preprocess_fn(config, (d120x120, None)) 
            _, d1200x1200 = preprocess_fn(config, (d1200x1200, None))
          
            d120x120_, _ = vmapped_partition_fn(d120x120[:,0], 3, 3)
            d1200x1200_, _ = vmapped_partition_fn(d1200x1200[:,0], 30, 30)

            loss, rec120x120, psnr120, mul, add = p_val_step(d120x120_, params, val_iter_rng)
            
            # apply to each 40 x 40 subpatch individually, treat as batch size -> 120 x 120 -> bs 9, 1200 x 1200 bs 90
            sub_bs = 90
            all_rec1200x1200 = []
            psnr1200 = 0.
            tot_bpp = 0.
            for k in range(10): # 90 * 10 = 900
                loss, rec1200x1200, psnr, mul, add = p_val_step(d1200x1200_[:, sub_bs*k:sub_bs*(k+1)], params, val_iter_rng)
                all_rec1200x1200.append(rec1200x1200)
                psnr1200 = psnr1200 + psnr

                # Logging bpp computation. Both `add, mul` are constrained to [-1, 1]
                # We use bit-packing and store them as integers, from their clamped float repr. (config.significant_digits)
                factor = 10 ** config.significant_digits

                mul, add = jnp.abs(mul * factor).astype(jnp.int32), jnp.abs(add * factor).astype(jnp.int32)
                mul_bits, add_bits = jnp.ceil(jnp.log2(mul)) + 2, jnp.ceil(jnp.log2(add)) + 2
                mul_bits, add_bits = jnp.nan_to_num(mul_bits, 0), jnp.nan_to_num(add_bits, 0)
                # clip -inf to 0
                
                mul_bits, add_bits = jnp.clip(mul_bits, 0), jnp.clip(add_bits, 0)
                maxbit_mul, maxbit_add = jnp.max(mul_bits), jnp.max(add_bits)
                mul_bits, add_bits = maxbit_mul + 0 * mul_bits, maxbit_add + 0 * add_bits
                
                tot_bits =  jnp.sum(mul_bits + add_bits, axis=(1,2,3,4)) 
                # for each element of mul and add, address
                tot_dim = mul.shape[2] * mul.shape[3]
                blk_address_bits = jnp.log2(mul.shape[1]) + 1
                chnl_address_bits = 3
                tot_bits = tot_bits +  blk_address_bits + mul.shape[-1] * chnl_address_bits * config.data_width
                bpp = tot_bits / (1200 * 1200 * 3)
                tot_bpp = tot_bpp + jax.device_get(bpp)

                if config.separate_maps_per_channel: tot_bpp = tot_bpp / config.data_width

            all_rec1200x1200 = jnp.concatenate(all_rec1200x1200, axis=1)
            collage120x120 = vmapped_unpartition_fn(rec120x120, 3, 3)
            collage1200x1200 = vmapped_unpartition_fn(all_rec1200x1200, 30, 30)

            log_recs(config, norm_two_to_eight_bit(d120x120[:, 0]), norm_two_to_eight_bit(collage120x120[:]), is_val=True, name=f"120x rec {stitch_id}")
            log_recs(config, norm_two_to_eight_bit(d1200x1200[:, 0]), norm_two_to_eight_bit(collage1200x1200[:]), is_val=True, name=f"1200x rec {stitch_id}")

            psnr120, psnr1200 = jax.device_get(psnr120), jax.device_get(psnr1200)
            stitch_eval_metrics = {
                f'psnr120_{stitch_id}': psnr120, 
                f'psnr1200_{stitch_id}_img0': psnr1200[0] / 10,
                f'psnr1200_{stitch_id}_img1': psnr1200[1] / 10,
                f'bpp_{stitch_id}_img0': tot_bpp[0],
                f'bpp_{stitch_id}_img1': tot_bpp[1],
                f'maxbit_mul_{stitch_id}': maxbit_mul.item(),
                f'maxbit_add_{stitch_id}': maxbit_add.item(),
            }
            print(f'psnr1200: {psnr1200 / 10}')
            print(f'bpp: {tot_bpp}')
            wandb.log(stitch_eval_metrics)


    if config.superres_on_eval > 1 and config.n_devices > 1:
        params = state.train_params
        x_in = x_pre[0]
        superres_recs = state.sample_fn({'params': params}, x_in, val_iter_rng, superres_factor=config.superres_on_eval)
        x_upsampled = jax.image.resize(x_in, (x_in.shape[0],
            config.superres_on_eval * config.data_res, 
            config.superres_on_eval * config.data_res, 
            config.data_width), method='nearest'
        )
        log_recs(config, norm_two_to_eight_bit(x_upsampled), norm_two_to_eight_bit(superres_recs), is_val=True, name="superres reconstructions")

    return metrics


def main(argv):
    config = FLAGS.config
    os.environ['WANDB_SILENT'] = "true"
    exp_name = f"{config.dataset}-{config.enc_blocks}-{config.n_aux_sources}"
    wandb.init(project="aerial_neural_fc", name=exp_name, config=config, entity=FLAGS.entity, dir='./wandb')
    if FLAGS.n_devices != 1: config.n_devices = FLAGS.n_devices
    print('number devices:', config.n_devices)

    config.checkpoint_dir = os.path.join(FLAGS.workdir, f"{config.enc_blocks}", f"{config.n_aux_sources}")

    print(f'working directory: {config.checkpoint_dir}')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if config.eval_on_stitch: 
        assert config.dataset == "dota", "Only dataset 'dota' allowed for evaluation on stitched images."
    config.datadir = FLAGS.datadir
    print(f'dataset directory: {config.datadir}')
    
    trainloader, valloader, testloader, preprocess_fn = get_dataset(config.bs, 
        config.bs, config.dataset, config.datadir, 0.1, config.n_devices, config.subset_size,
    )
    # model: NeuralFractalAutoencoder sampler: NeuralFractalSuperresAutoencoder
    config, state, optimizer, reloaded = initialize_model(config, FLAGS, config.model, config.sampler, config.wandb_dir)

    os.environ['WANDB_SILENT'] = "true"
    if not reloaded: 
        run_id = wandb.util.generate_id()
        state.replace(run_id=run_id)
    else: 
        run_id = state.run_id
        
    val_rng = jax.random.PRNGKey(config.seed)

    for epoch in range(config.epochs):
        print(f'.....................Epoch: {epoch}/{config.epochs}....................')
        train_ds = iter(trainloader)
        trainloader.sampler.set_epoch(epoch)
        state = train_epoch(epoch, config, state, optimizer, train_ds, preprocess_fn, state.opt_state)
        
        if epoch % config.eval_epochs == 0:
            valloader.sampler.set_epoch(epoch)
            testloader.sampler.set_epoch(epoch)
            metrics = evaluate(config, state, testloader, val_rng, preprocess_fn)
            wandb.log(metrics)
            
    metrics = evaluate(config, state, testloader, val_rng, preprocess_fn)
    wandb.log(metrics)


if __name__ == '__main__':
    app.run(main)
