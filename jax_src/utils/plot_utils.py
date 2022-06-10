import os
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from PIL import Image


def norm_two_to_eight_bit(ndarr):
    "Rescale ndarr from [0, 1] to [0, 255]"
    # we don't need to assert here, it's fine to be slightly outside the bounds
    #assert ((0 <= ndarr) & (ndarr <= 1)).all(), f'ndarr cannot be rescaled from 2 to 8 bit: [{ndarr.min()}, {ndarr.max()}]'
    
    ndarr = jnp.clip(ndarr, 0., 1.)
    return jnp.clip(ndarr * 255.0 + 0.5, 0, 255).astype(jnp.uint8)


def gridify_images(images, padding=2, pad_value=0.0, format=None, aux_info=None,
                   normalize=False, normalize_aux=True):
    """Turn batch of images into a single (grid) image.

    Pixel values are assumed to be within [0, 255] if normalize=False, [0, 1] if True.

    Args:
      ndarray (array_like): 5D mini-batch of sequence of images (T x B x H x W x C).
    """
    if not (isinstance(images, jnp.ndarray) or
            (isinstance(images, list) and
            all(isinstance(t, jnp.ndarray) for t in images))):
      raise TypeError("array_like of tensors expected, got {}".format(
        type(images)))
    ndarray = jnp.asarray(images)

    # Safety unsqueeze for 4-dim image tensors
    if ndarray.ndim == 4: ndarray = ndarray[None]

    # Add the original source image as an additional step in the sequence of length T
    if aux_info: 
        src_img = aux_info['img']
        if src_img.ndim == 4:
            src_img = aux_info['img'][None]
            if normalize_aux: norm_two_to_eight_bit(src_img)
        ndarray = jnp.vstack([src_img, ndarray])
    
    # Single-channel images are extended to 3 channel
    if ndarray.ndim == 5 and ndarray.shape[-1] == 1:  
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # Make the mini-batch of images into a grid
    n_steps, bsz, h, w, num_channels = ndarray.shape

    ncol = n_steps 
    nrow = bsz

    grid_h, grid_w = int(h + padding), int(w + padding)
    grid = jnp.full(
        (grid_h * nrow + padding, grid_w * ncol + padding, num_channels), pad_value).astype(jnp.float32)

    for y in range(nrow):
        for x in range(ncol):
            grid = jax.ops.index_update(
              grid, jax.ops.index[y * grid_h + padding:(y + 1) * grid_h,
                  x * grid_w + padding:(x + 1) * grid_w], ndarray[x, y])
      
    if normalize: 
        ndarr = norm_two_to_eight_bit(grid)
    else: ndarr = jnp.clip(grid, 0, 255).astype(jnp.uint8)
    return ndarr


def log_plots(config, input_imgs, reconstructions, samples, save_path=None, is_val=False, name='', normalize=False):
    """Who knows what this is supposed to do."""
    sqrt_n_samples = np.sqrt(config.n_samples).astype(int)
    assert sqrt_n_samples * sqrt_n_samples == config.n_samples, f'np.sqrt(config.n_samples).astype(int) * {np.sqrt(config.n_samples).astype(int)} != config.n_samples'
    samples = samples.reshape(sqrt_n_samples, sqrt_n_samples, *samples.shape[1:]) # sqrtn, sqrtn, h, w, c

    aux_info = {'img': input_imgs}
    ndarr_cond = gridify_images(reconstructions, aux_info=aux_info, normalize=normalize)
    ndarr_uncond = gridify_images(samples, normalize=False)
    # TODO: save_path as `log_samples`
    # if save_path:

    #     im_cond, im_uncond = Image.fromarray(ndarr_cond.copy()), Image.fromarray(ndarr_uncond.copy())
    #     save_path_cond, save_path_uncond = save_path + '_cond', save_path + '_uncond'
    #     im_cond.save(im_cond, format=format)
    #     im_uncond.save(save_path_uncond, format=format)

    images_cond, images_uncond = np.array(ndarr_cond), np.array(ndarr_uncond)
    if is_val == False:
      images_cond, images_uncond = wandb.Image(images_cond, caption=f"Reconstructions {name}"), wandb.Image(images_uncond, caption=f"Samples {name}")
      wandb.log({"Reconstructions": images_cond})   
      wandb.log({"Samples": images_uncond})
    else:
      
      images_cond, images_uncond = wandb.Image(images_cond, caption=f"Val Reconstructions {name}"), wandb.Image(images_uncond, caption=f"Val Samples {name}")
      wandb.log({"Val (EMA) Reconstructions": images_cond})   
      wandb.log({"Val (EMA) Samples": images_uncond}) 


def log_samples(samples, save_path=None, is_val=False, name=None, log_to_wandb=True):
    """Produce plots of unconditional samples `bs, h, w, c` and optionally log them to wandb.

    Args:
        samples (tensor): batch of images to plot. Batch size should be square of an integer 
        save_path (_type_, optional): _description_. Defaults to None.
        is_val (bool, optional): _description_. Defaults to False.
        name (_type_, optional): _description_. Defaults to None.
        log_to_wandb (bool, optional): _description_. Defaults to True.
    """
    n_samples = samples.shape[0]
    sqrt_n_samples = np.sqrt(n_samples).astype(int)
    assert sqrt_n_samples * sqrt_n_samples == n_samples, \
        f'n_samples not the square of an integer int(sqrt(n_samples)) * int(sqrt(n_samples)) != n_samples'
    samples = samples.reshape(sqrt_n_samples, sqrt_n_samples, *samples.shape[1:]) # sqrtn, sqrtn, h, w, c
    ndarr = gridify_images(samples, normalize=False)

    name = name if name else "Samples"  
    if save_path:
        pil_images = Image.fromarray(ndarr.copy())
        save_path = os.path.join(save_path, name)
        with open(f"{save_path}.png", "wb") as f:     
          pil_images.save(f, format=None)

    if log_to_wandb:
      nparr = np.array(ndarr)
      if is_val == True: name = f"Val {name}"
      pil_images = wandb.Image(nparr, caption=f"{name}")
      wandb.log({"Samples": pil_images})   


def log_recs(config, input_imgs, reconstructions, save_path=None, is_val=False, name=None):
    """Take ground truth images and reconstructions and plot then side by side in 2 columns."""
    aux_info = {'img': input_imgs}
    ndarr_cond = gridify_images(reconstructions, aux_info=aux_info, normalize=False)
    # TODO: save_path as `log_samples`

    images_cond = np.array(ndarr_cond)
    if is_val == False:
      name_ = name if name else "Reconstructions"
      images_cond = wandb.Image(images_cond, caption=f"{name_}")
      wandb.log({"Reconstructions": images_cond})   
    else:
      name_ = f"val {name}" if name else "Val (EMA) Reconstructions"
      images_cond = wandb.Image(images_cond, caption=f"{name_}")
      # wandb.log({f"{name_}": images_cond})   
      log_panel = "Val (EMA) Reconstructions"
      if 'former' in name:
        log_panel = f'Former {log_panel}'
      if 'latter' in name:
        log_panel = f'Latter {log_panel}'
      wandb.log({log_panel: images_cond})   
 