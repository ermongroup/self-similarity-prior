import jax
import jax.numpy as jnp
import numpy as np
from utils.transforms import (affine_contraction, chunk_from_id,
                                faster_unpartition_img, partition_img,
                                reduce)
from jax import vmap
from tqdm import tqdm

BASE_KEY = jax.random.PRNGKey(0)

def get_noise_schedule(times=[0], key=BASE_KEY, H=32, W=32, noise_type=None, image=None):
    if noise_type is None:
        # Standard decoding in fractal compression starts from a uniform sample 
        print("Using uniform noise schedule at itr 0")
        sched = {t: jax.random.uniform(BASE_KEY, (H, W), minval=0, maxval=1) for t in times}
    else:
        sched = {t: noise_type for t in times}
    if image is not None: # type is some image
        sched = {t: image for t in times}
    return sched


def generate_candidates(chunks, identifiers, use_rotation=True, use_flips=False):
    """Take a batch of chunks and generate candidates with rotation and flips.
       Also updates the list of identifiers for each candidate (coordinates, flip={0,1}, rotated={0,90,180,270})
    """
    candidates = [chunks]
    flip_bit = jnp.zeros((identifiers.shape[0], 1))
    rotate = jnp.zeros((identifiers.shape[0], 1))
    identifiers = jnp.concatenate([identifiers, flip_bit, rotate], 1)

    # try all rotations of the block
    # TODO: assert chunks are square before rotating
    if use_rotation:
        chunks_ = chunks
        for i in range(4):
            chunks_ = jnp.rot90(chunks_, axes=(1,2))
            candidates.append(chunks_)
            new_identifiers = identifiers
            new_identifiers = new_identifiers.at[:,3:4].set(i) # number times it was rotated by 90
            identifiers = jnp.concatenate([identifiers, new_identifiers], 0)
    
    if use_flips:
        # Flip each candidate domain block and consider it as a candidate chunk
        chunks = vmap(jnp.fliplr, in_axes=0)(chunks)
        candidates.append(chunks)
        new_identifiers = identifiers 
        
        # Update bit to indicate whether patch is flipped or not.
        new_identifiers = new_identifiers.at[:,2:3].set(1)
        identifiers = jnp.concatenate([identifiers, new_identifiers], 0)

        if use_rotation:
            chunks_ = chunks
            for i in range(4):
                # try all rotations of the flipped block
                chunks_ =  jnp.rot90(chunks_, axes=(1,2))
                candidates.append(chunks_)
                new_identifiers = identifiers 
                new_identifiers = new_identifiers.at[:,3:4].set(i) # number times it was rotated by 90
                identifiers = jnp.concatenate([identifiers, new_identifiers], 0)
    
    return jnp.concatenate(candidates), jnp.asarray(identifiers)


def augment_sample(img, use_rotation=True, use_flips=False, h_axis=0, w_axis=1):
    """Take single aux image and generate augmentations to append.
    """
    candidates = [img]
    if use_rotation:
        _img = img
        for i in range(4):
            # 0, 1 b/c it's just 1 image for aux patch
            rot_img = jnp.rot90(_img, axes=(h_axis, w_axis))
            # 2, 3 because these are broadcasted tensors
            # rot_img = jnp.rot90(_img, axes=(2, 3))
            candidates.append(rot_img)
    if use_flips:
        # vmap(jnp.fliplr, in_axes=0)(img)
        flip_img = jnp.fliplr(img) # no need to vmap b/c single batch
        candidates.append(flip_img)

        if use_rotation:
            _img = flip_img
            for i in range(4):
                rot_flip_img = jnp.rot90(_img, axes=(h_axis, w_axis))
                candidates.append(rot_flip_img)
    
    # return jnp.concatenate(candidates)
    return candidates


def encode(R_patches, candidate_patches, identifiers, n_sources, cmap_class="affine"):
    # options for cmap_class: {nonlinear, affine, adaptive}
    # # Different selection criteria for each candidate patch
    # # cmap_class == "affine": 1-shot, batched linear solve for coeffs that better reconstruct D->R
    # # cmap_class == "nonlinear": fixed compute budget GD training of parameters that better reconstruct D->R

    # Initialize parameters of all contraction maps as the solutions of `a * D + b = R`
    cmaps = []

    # this loop slows down device runtime quite a lot... 
    print("Starting encoding procedure...")
    if cmap_class == 'affine':
        for k, R_block in tqdm(enumerate(R_patches)):
            # R_block: {B}, [num_R_patches], h, w
            # candidate_patches: {B}, num_candidate_patches, h, w
            # parallelizing across dim 0, R_block needs to be broadcasted in linsolve
            R_block = R_block[None] # 1, h, w
            R_block = jnp.repeat(R_block, candidate_patches.shape[0], axis=0) # 
            A_ = jnp.ones((*candidate_patches.shape, 1)) # augment for bias
            candidate_patches_ = candidate_patches[...,None]
            A = jnp.concatenate([candidate_patches_, A_], axis=-1)
            # Ax = b {B}, num_candidate_patches, h, w, 2  
            # x = [params], b = R_block, A = 

            # 16, H, W, 2 -> vmapped H x W, 2
            A = A.reshape(A.shape[0], -1, 2)
            B = R_block.reshape(A.shape[0], -1)[...,None]

            sol = vmap(jnp.linalg.lstsq, in_axes=(0,0))(A, B)
            opt_params, res = sol[0], sol[1]
            opt_params = opt_params[...,0]

            reconstructed_R = affine_contraction(candidate_patches, opt_params)
            # find optimal parameters (not just argmax)
            residuals = jnp.linalg.norm(reconstructed_R - R_block, ord=2, axis=(1,2))

            sorted_residuals = jnp.sort(residuals)[:n_sources]
            argsorted = jnp.argsort(residuals)
            indices = argsorted[:n_sources]

            # allocate weights
            residual_sum = jnp.sum(sorted_residuals)
            weights = residual_sum - sorted_residuals
            weights = jnp.exp(weights) / jnp.sum(jnp.exp(weights))

            if n_sources == 1: 
                weights = 0*weights + 1
                # avoid linking patches to themselves (which can only happen in d_h == r_h, d_w == r_w)
                indices = jnp.where(indices==k, argsorted[1], indices)
            
            # https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerArrayConversionError
            sources = []
            
            for save_idx in indices:
                idx = save_idx.astype(int)
                idx_D, D, opt_params_ = jnp.asarray(identifiers)[idx], candidate_patches[idx], opt_params[idx]
                sources.append(jnp.concatenate([idx_D, opt_params_, weights[None][:,idx]])) # weights is built in vmap, no bsize
                
            cmaps.append(jnp.stack(sources)) # n_sources, 7

    # # TODO: Adaptive partitioning, finding best contraction maps
    # # Place threshold on `acceptable reconstruction`, if not satisfactory repeat loop above
        
    return jnp.stack(cmaps) # num_R_patches, n_sources, 7 [2d identifier, flip bit, # of rotations) + 2 (params] + weights


def decode(img, configs, key=BASE_KEY):
    """
    noise injections (different types) at step k of decoding.
    controllable start image for decoding.

    Notes:
        cmaps is a list of len(cmap) = R_patches.shape[0] with identifiers
        for partitions links (D -> R) and also contains parameters of contraction maps.
    """
    cmaps = configs['cmaps']
    R_patches = configs['R_patches']
    _noise_schedule = configs['noise_schedule']
    noise_type = configs['noise_type']
    image_noise = configs['image_noise']
    superres_factor = configs['superres_factor']
    H, W = configs['H'], configs['W']
    D_h_chunks, D_w_chunks = configs['D_chunks']
    R_h_chunks, R_w_chunks = configs['R_chunks']
    factors, n_iters = configs['factors'], configs['n_iters']
    R_identifiers = configs['R_identifiers']
    inverse_problem = configs['inverse_problem']
    if superres_factor > 1:
        assert inverse_problem == "None", "Inverse problem incompatible with decoding at higher resolutions"

    noise_schedule = get_noise_schedule(_noise_schedule, noise_type=noise_type,
        H=H*superres_factor, W=W*superres_factor, image=image_noise
    )

    z0 = img
    if inverse_problem == "imputation":
        z0 = z0.at[H//2:].set(0)

    elif inverse_problem == "denoising":
        #z0 = z0 + noise_schedule[0]
        z0 = z0 + 10

    elif inverse_problem == "mixup":
        z0 = configs['mixup_image']

    elif inverse_problem == "None":
        z0 = jnp.zeros(shape=(H*superres_factor, W*superres_factor))
        #D_h_chunks, D_w_chunks = D_h_chunks*superres_factor, D_w_chunks*superres_factor
        #R_h_chunks, R_w_chunks = R_h_chunks*superres_factor, R_w_chunks*superres_factor

    sol_traj = [z0]

    # Inject noise, partion, reduce, apply, reconstruct
    print("Starting decoding fixed-point iterative procedure...")
    for itr in tqdm(range(n_iters)):
        
        cur = sol_traj[-1]
        if itr in noise_schedule:
            
            noise = noise_schedule[itr]
            cur = cur + noise

        # TO JIT
        def decode_step(cur):
            domain_chunks, identifiers = partition_img(cur, D_h_chunks, D_w_chunks)
            candidate_chunks, identifiers = generate_candidates(domain_chunks, identifiers, use_flips=configs['use_flips'], use_rotation=configs['use_rotation'])
            candidate_chunks = reduce(candidate_chunks, factors)

            indices_D, opt_params, weights = jnp.split(cmaps, [4, 6], axis=2)
            # TODO: add weight quantiz. as option (to simulate "realistic" bpp budgets for fractal compression)
            # simulate lossy compression for contraction map parameters
            weights = jnp.asarray(weights, jnp.float16)
            weights = jnp.round(weights, 2)

            opt_params = jnp.asarray(opt_params, jnp.float16)
            opt_params = jnp.round(opt_params, 2)

            batched_chunk_fn = vmap(vmap(chunk_from_id, in_axes=(0, None, None, None, None)), in_axes=(0, None, None, None, None))
            Ds = batched_chunk_fn(indices_D, candidate_chunks, D_h_chunks, D_w_chunks, identifiers)
            
            batched_affine_contraction = vmap(vmap(affine_contraction, in_axes=(0, 0)), in_axes=(0, 0))
            reconstructed_Rs = batched_affine_contraction(Ds, opt_params)
            reconstructed_Rs = jnp.einsum('bk,bkij->bij', weights[...,0], reconstructed_Rs)
             
            # Unchunk the image from reconstructed_Rs
            sol = faster_unpartition_img(reconstructed_Rs, R_h_chunks, R_w_chunks)
            return sol

        sol = decode_step(cur)
        sol_traj.append(sol)

    return sol_traj


def fractal_compress(key, img, configs):
    """Dispatches to an appropriate transform of `_fractal_compress` depending on the type of input image.
    
    Notes:
        Each channel in the input batch of images is treated as a separate independent sample. Total batch size is then
        batch size * number of channels.

        Expected fields in config: 
            n_iters :
            noise_schedule [optional] : the step at which to inject noise (e.g. specific image etc.). Default: white noise.
            init_partition : how to split the data in partitions. Defaults to non-overlapping square tiles ["tiling"]
            cmap_class : class of contraction maps used for decoding. ["affine"]
            
            use_rotation : whether to use rotations [90, 180, 270] to generate candidate source (domain) partitions.
            use_flips : whether to use flips to generate candidate domain (source) partitions.
            
            D_chunks [Tuple] : (D_H_chunks, D_w_chunks). Number of domain (source) partitions in the heigth and width dimensions.
            R_chunks [Tuple] : (R_H_chunks, R_w_chunks). Number of range (target) partitions in the heigth and width dimensions.
    """
    bs, H, W, C = img.shape
    superres_factor = configs['superres_factor']

    if C == 1:
        img = img[...,0]
        sol_traj, cmaps = jax.vmap(_fractal_compress, in_axes=(None, 0, None), out_axes=1)(key, img, configs)
        sol_traj = sol_traj[...,None]
    else: 
        img = jnp.transpose(img, axes=(0, 3, 1, 2))
        img = img.reshape(-1, img.shape[-2], img.shape[-1])
        sol_traj, cmaps = jax.vmap(_fractal_compress, in_axes=(None, 0, None), out_axes=1)(key, img, configs) #  sol_traj: (n_steps, bs*c, h, w)
    sol_traj = sol_traj.reshape(sol_traj.shape[0], bs, C, H*superres_factor, W*superres_factor) # (n_steps, c, b, h, w)
    sol_traj = jnp.transpose(sol_traj, axes=(0, 1, 3, 4, 2)) # (n_steps, b, h, w, c)
    return sol_traj, cmaps


def _fractal_compress(key, img, configs):
    "Driver function for `fractal_compress`"
    H, W = img.shape
    n_sources = configs['n_sources']
    R_h_chunks, R_w_chunks = configs['R_chunks']
    D_h_chunks, D_w_chunks = configs['D_chunks']
    R_h_chunksize, R_w_chunksize = H // R_h_chunks, W // R_w_chunks
    D_h_chunksize, D_w_chunksize = H // D_h_chunks, W // D_w_chunks
    h_factor, w_factor = D_h_chunksize // R_h_chunksize, D_w_chunksize // R_w_chunksize, 
    factors = (h_factor, w_factor)

    D_patches, D_identifiers = partition_img(img, D_w_chunks, D_h_chunks)
    R_patches, R_identifiers = partition_img(img, R_w_chunks, R_h_chunks)

    # Generate candidate patches from src partitions to inject into R via `contraction_maps`
    candidate_patches, identifiers = generate_candidates(D_patches, D_identifiers, use_flips=configs['use_flips'], use_rotation=configs['use_rotation'])
    # Reductions to ensure domain and target patches are of compatible dimensions
    candidate_patches = reduce(candidate_patches, factors=factors)
    
    cmaps = encode(R_patches, candidate_patches, identifiers, n_sources, cmap_class="affine") 

    configs['cmaps'] = cmaps
    configs['R_patches'] = R_patches
    configs['R_identifiers'] = R_identifiers
    configs['factors'] = factors
    configs['H'] = H
    configs['W'] = W
    
    sol_traj = decode(img, configs)

    return jnp.stack(sol_traj, 0), cmaps
