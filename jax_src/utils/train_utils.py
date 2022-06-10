import json
import typing as T
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections
import numpy as np
import optax
from absl import flags
from flax import struct
from flax.training import checkpoints
from flax.traverse_util import flatten_dict, unflatten_dict
from PIL import Image

import utils.ptypes as PT
import wandb
from utils.transforms import *
from src.data.datasets import *
from src.frax.fractalae import (NeuralFractalAutoencoder,
                               NeuralFractalStyleAutoencoder,
                               NeuralFractalSuperresAutoencoder,
                               CollageAutoencoder, FCAutoencoder)
from src.frax.fractalvae import *
from src.frax.modules import *
from src.frax.vae import (VDCVAE, VDVAE, LatAuxVDCVAE, LatAuxVDCVAESampler,
                         VDCVAESampler, VDVAESampler)
from utils.vae_utils import compute_number_parameters
import functools


class TrainState(flax.struct.PyTreeNode):
    train_params: PT.Params # trainable
    ema_params: PT.Params
    opt_state: optax.OptState
    metrics: dict
    key: T.Optional[PT.PRNGDict]
    apply_fn: T.Callable = struct.field(pytree_node=False)
    sample_fn: T.Callable = struct.field(pytree_node=False)
    step: int
    model_state: PT.ModelState # for batch norm stats
    run_id: str # for wandb run resuming, added after model is initialized
     
    def merge_nested_dicts(self, *ds):
        merged = {}
        for d in map(flatten_dict, map(flax.core.unfreeze, ds)):
            if any(k in merged.keys() for k in d.keys()):
                raise ValueError('Key conflict!')
            merged.update(d)
        return unflatten_dict(merged)

    @property
    def params(self):
        return self.merge_nested_dicts(self.train_params, self.ema_params)

    @property
    def variables(self):
        return {'params': self.params, **self.model_state}

ACTIVATIONS = {
    'relu': jax.nn.relu,
    'id': lambda x: x,
    'tanh': jnp.tanh,
    'sigmoid': jax.nn.sigmoid,
    'silu': jax.nn.silu,
    'leaky_relu': jax.nn.leaky_relu,
}

lecun_normal = functools.partial(
    jax.nn.initializers.variance_scaling,
    mode='fan_in',
    distribution='truncated_normal')

INITIALIZERS = {
    'delta_orthogonal': jax.nn.initializers.delta_orthogonal,
    'orthogonal': jax.nn.initializers.orthogonal,
    'lecun_normal': lecun_normal,
    'xavier_uniform': jax.nn.initializers.xavier_uniform,
    'zeros': jax.nn.initializers.zeros
}

STR_TO_MODEL = {"VDVAE": VDVAE, "EVAE": EVAE, "FractalVAE": PatchFractalVAE, "VDCVAE": VDCVAE, "LatAuxVDCVAE": LatAuxVDCVAE, "StyleFractalAE": NeuralFractalStyleAutoencoder, 
                "FractalAESuperRes": NeuralFractalSuperresAutoencoder, "FractalAE": NeuralFractalAutoencoder, "CollageAE": CollageAutoencoder, "FCAE": FCAutoencoder}
STR_TO_SAMPLER = {"VDVAE": VDVAESampler, "EVAE": EVAESampler, "FractalVAE": PatchFractalVAESampler, "VDCVAE": VDCVAESampler, "LatAuxVDCVAE": LatAuxVDCVAESampler, 
                "StyleFractalAE": NeuralFractalStyleAutoencoder, "FractalAESuperRes": NeuralFractalSuperresAutoencoder, "FractalAE": NeuralFractalAutoencoder,
                "CollageAE": CollageAutoencoder, "FCAE": FCAutoencoder}
                
DUMMY_STATE = TrainState(train_params=None, ema_params=None, apply_fn=None, sample_fn=None, opt_state=None, metrics={}, key=None, step=None, model_state=None, run_id='')

def initialize_model(config, forward_model=None, sampling_model=None, wandb_dir=f'/checkpoint/xwinxu/wandb'):
    rng = random.PRNGKey(config.seed)
    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optax.adamw(config.lr, b1=0.9, b2=0.9, weight_decay=config.wd)
    
    reloaded = False
    state = None
    forward_model = config.model if forward_model is None else forward_model
    sampling_model = config.model if sampling_model is None else sampling_model
    if config.restore_from_checkpoint:
        try:    
            print(f'restoring from checkpoint {config.checkpoint_dir}')
            # if no checkpoint exists, DUMMY_STATE is returned
            state = checkpoints.restore_checkpoint(config.checkpoint_dir, DUMMY_STATE, step=None)
            with open(f'{config.checkpoint_dir}/config.json', 'r') as f:
                config = json.load(f)
            config = ml_collections.config_dict.create(**config)

            opt_state = optimizer.init(state.train_params)
            forward_fn = STR_TO_MODEL[forward_model](config).apply
            sampling_fn = STR_TO_SAMPLER[sampling_model](config).apply

            # NOTE: opt_state is reset after loading! Issue with optax opt_state loading with flax.checkpoints utility above
            fns = {"sample_fn" : sampling_fn, "apply_fn" : forward_fn, "opt_state": opt_state}
            state = state.replace(**fns)
            if state.step > 0:
                reloaded = True
                state = state.replace(step=state.step)
                print(f'successfully restored checkpoint state at step {state.step} for run: {state.run_id}')
        except:
            print(f'attempted to reload when there were no checkpoints, initializing to default state....')

    if not reloaded:
        step = 0
        init_batch = jnp.zeros((config.bs, config.data_res, config.data_res, config.data_width))
        params_rng, init_rng = random.split(rng)
        params = STR_TO_MODEL[forward_model](config).init({'params': params_rng}, init_batch, init_batch, rng)['params']
               
        forward_fn = STR_TO_MODEL[forward_model](config).apply
        sampling_fn =  STR_TO_SAMPLER[sampling_model](config).apply

        opt_state = optimizer.init(params)
        
        # take care of wandb with new run_id
        new_run = wandb.init(project=config.project_name, name=config.exp_name, entity=config.user, config=config, dir=wandb_dir)

        state = TrainState(
            train_params=params,
            ema_params=params,
            model_state={},
            opt_state=opt_state,
            metrics={},
            key=init_rng,
            step=step,
            apply_fn=forward_fn,
            sample_fn=sampling_fn,
            run_id=(new_run.id),
        )
        
        assert new_run.id == state.run_id, f'state run_id was {state.run_id} and not updated successfully'
        print(f'initialized wandb new run {state.run_id} at {config.user}/{config.project_name}/{config.exp_name}')

        config.run_id = state.run_id
        with open(f'{config.checkpoint_dir}/config.json', 'w') as f:
            json.dump(config.to_dict(), f)

        checkpoints.save_checkpoint(config.checkpoint_dir, jax.device_get(state), step, overwrite=True)

    n_params = compute_number_parameters(state.train_params)
    print(f'n parameters in the model: {n_params}')
    return config, state, optimizer, reloaded


def update_params(config, params, grad, optimizer, opt_state):
    grad = jax.tree_map(lambda x: jnp.mean(x, 0), grad)
    grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grad)
    grad, grad_norm = clip_grad_norm(config, grad)
    
    updates, opt_state = optimizer.update(grad, opt_state)
    params = flax.jax_utils.unreplicate(params)   
    new_params = optax.apply_updates(params, updates)
    ema_params = jax.tree_util.tree_multimap(lambda new, old: 0.01 * new + 0.99 * old, new_params, params)
    params = flax.jax_utils.replicate(params)
    return params, ema_params, grad_norm, opt_state 


def clip_grad_norm(config, grad):
    max_norm = config.max_grad_norm
    norm_tree = jax.tree_map(jnp.linalg.norm, grad)
    g, treedef = jax.tree_flatten(norm_tree)
    total_norm = jnp.sum(jnp.stack(g))
    clip_coeff = jnp.minimum(max_norm / (total_norm + 1e-8), 1)
    clip_grad = jax.tree_map(lambda x: x * clip_coeff, grad)
    return clip_grad, total_norm

def parse_img_from_path(img_path, normalize=True, jaxify=True):
    # pre-process the single image to use as the auxilliary image source
    img = Image.open(f'{img_path}')
    if jaxify:
        img_array = jnp.asarray(img)
    else:
        img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    if normalize:
        img_array = img_array / 255.

    return img_array

def random_crop(img, shapes):
    h, w = shapes
    x = np.random.randint(0, img.shape[1] - w)
    y = np.random.randint(0, img.shape[0] - h)
    img = img[y:y+h, x:x+w]
    return img

def preprocess_style_blocks(config, reduce_type='pool'):
    # pre-process the single image to use as the auxilliary image source
    aux_source_img = parse_img_from_path(config.style_source_img)
    h, w, c, = aux_source_img.shape # (600, 600, 3)

    if reduce_type == 'pool':
        # the range shapes to pool the aux image into --> (20, 20), so image is (600, 600) --> (20, 20)
        rh, rw = config.range_szs
        factors = (h // rh, w // rw) # for (20, 20) --> 600 / 20 = 30
        pooled_aux_img = reduce_all(aux_source_img, factors)
        print(f"aux_source_img {aux_source_img.shape}")
        assert pooled_aux_img.shape[0] == rh and pooled_aux_img.shape[1] == rw, f'(rh, rw): ({rh}, {rw}) but got (h, w, c): {pooled_aux_img.shape}'
        print(f'pooled_aux_img {pooled_aux_img.shape}')
        return pooled_aux_img
    elif reduce_type == 'crop':
        cropped_aux_img = random_crop(aux_source_img, config.range_szs)
        return cropped_aux_img

def maybe_restart_wandb(resuming, state, config, resume='must'):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file. 
        
        Returns the (maybe) updated state and config.
    """
    # if the run_id was previously saved, resume from there
    if resuming:
        assert resuming, f'Something went wrong, entered resuming but nothing to resume!'
        wandb.init(id=state.run_id, project=config.project_name, name=config.exp_name, config=config, entity=config.user, dir=config.wandb_dir, resume=resume)
        print(f'successfully resumed run: {state.run_id}')
    else:
        print( f'Something went wrong, there was no wandb to resume!')

    return state, config


@flax.struct.dataclass
class MaybeSkipGradientUpdateState:
  inner_state: Any


def maybe_skip_gradient_update(
    inner: optax.GradientTransformation,
    gradient_norm_skip_threshold: float,
    ) -> optax.GradientTransformation:
    """A function that wraps an optimiser to skip updates under some condition.

    The purpose of this function is to prevent any optimisation to happen if the
    gradients contain NaNs, Infs, or if its norm is higher than a certain
    threshold. That is, when a NaN of Inf, is detected in the gradients or when
    the norm of the gradient is higher than the threshold, the wrapped optimiser
    ignores that gradient update.

    Args:
        inner: Inner transformation to be wrapped.
        gradient_norm_skip_threshold: float,

    Returns:
        New GradientTransformation.
    """

    def init(params):
        return MaybeSkipGradientUpdateState(inner_state=inner.init(params))

    def update(updates, state, params=None):
        inner_state = state.inner_state
        # Compute gradient norm and clip gradient if necessary
        gradient_norm = optax.global_norm(updates)
        flat_updates = jax.tree_flatten(updates)[0]
        isfinite = jnp.all(
            jnp.array([jnp.all(jnp.isfinite(p)) for p in flat_updates]))
        islowerthan = gradient_norm < gradient_norm_skip_threshold

        def do_update(_):
            return inner.update(updates, inner_state, params)

        def reject_update(_):
            return (jax.tree_map(jnp.zeros_like, updates), inner_state)

        updates, new_inner_state = jax.lax.cond(
            jnp.logical_and(isfinite, islowerthan),
            do_update,
            reject_update,
            operand=None)

        return updates, MaybeSkipGradientUpdateState(inner_state=new_inner_state)

    return optax.GradientTransformation(init=init, update=update)


def preprocess_fn(config, batch_data):
    x, _ = batch_data
    x = x.permute(0, 2, 3, 1)
    x = 2*x - 1 # normalize in [-1, 1]
    x = jnp.asarray(x) 
    x = jnp.reshape(x, (config.n_devices, -1, *x.shape[1:]))
    return x
