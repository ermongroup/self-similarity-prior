import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax import vmap
from PIL import Image


def rotate(img, angle):
    num_rotations = angle // 90
    return jnp.rot90(img, k=num_rotations, axes=(1,2))


def affine_contraction(blocks, params):
    """Batched computation of reconstructions via affine contraction maps from candidate partitions (reduced)
    """
    contrast, brightness = params[..., :1], params[..., 1:] # keep first dimension to broadcast params
    # unsqueeze for second dim to broadcast since blocks are 2D
    contrast, brightness = contrast[..., None], brightness[..., None]

    return contrast * blocks + brightness 


def nonlinear_contraction(blocks, nets, rng, params):
    "Batched computation of reconstructions via nonlinear contraction maps from candidate partitions (reduced)"
    return nets.apply(rng, blocks, params)


def partition_img(img, num_w_chunks, num_h_chunks, row_major=True):
    
    """Decomposes an image or batch of images into a batches of [h, w] partitions. 
       Returns partitions and their coordinate identifiers. H, W, C -> N, h, w, C.
       Note, pmap(vmap) if batched and distributed.
    """
    n = num_h_chunks * num_w_chunks
    # TODO: this errors out because no channel dim
    rows = jnp.stack(jnp.split(img, num_w_chunks, axis=1), axis=0) # (num_w_chunks, H, w, C) where n_w_chunks * w = W
    patches = jnp.concatenate(jnp.split(rows, num_h_chunks, axis=1), axis=0) # (n, h, w, C)
    
    first_ax, sec_ax = num_h_chunks, num_w_chunks
    if not row_major:
       first_ax, sec_ax = sec_ax, first_ax
    identifiers = jnp.array([[[i,j] for j in range(sec_ax)] for i in range (first_ax)])

    identifiers = identifiers.reshape(n, 2) # (num_total, coordinates)

    assert identifiers.shape[0] == patches.shape[0] == n
    return patches, identifiers


def batched_partition_img(imgs, num_w_chunks, num_h_chunks):
    return vmap(partition_img, in_axes=(0, None, None))(imgs, num_w_chunks, num_h_chunks)


def faster_unpartition_img(patches, num_h_patches, num_w_patches):
    "Reconstruct an image from a batch `patches` of partitioned subdomains of the image"
    # construct `num_h_patches` rows of `num_w_patches` partitions
    rows = []
    for k in range(num_h_patches):
        row = jnp.stack(patches[k*num_w_patches:(k+1)*num_w_patches], axis=-1)
        row = jnp.transpose(row, axes=(0,2,1))
        row = jnp.reshape(row, (row.shape[0], -1))
        rows.append(row)
    return jnp.concatenate(rows, axis=0)

def batched_unpartition_img(x, h, w, H, W, img_channels):
    "(b, w, h, num_patches * img_channels) - > (b, H, W, img_channels)"
    num_h, num_w = H // h, W // w
    x = jnp.reshape(x, (*x.shape[:3], -1, img_channels))
    x = jnp.transpose(x, (0, 3, 1, 2, 4))
    def unpartition_img_(patches):
        rows = []
        for k in range(num_h):
            row = jnp.stack(patches[k*num_w:(k+1)*num_w], axis=-1)
            row = jnp.transpose(row, axes=(0,2,1))
            row = jnp.reshape(row, (row.shape[0], -1))
            rows.append(row)
        return jnp.concatenate(rows, axis=0)[None]

    vmapped_fn = vmap(vmap(unpartition_img_, in_axes=0), in_axes=4, out_axes=4)
    return vmapped_fn(x)[:,0] # remove dummy dim returned by `unpartition_img_`


def unpartition_img(patches, num_w_chunks, num_h_chunks, identifiers, row_major=True):
    """
    This function reconstructs an image from a batch of partitions. Should be sped up by parallelization.
    As it stands, this function cannot be jited, as it is deprecated in favor of `faster_unpartition_img`.
    """
    # follow same order as identifiers to check order of chunking
    # patches: [B], N, h, w, C - > [B], H, W, C

    # [[0,0],[0,1],[0,2],..,[0,n_cols]# ,[1,0],[1,1],..,[1,n_cols],]

    # Adaptive row_major detection turned off to avoid ConcretizationErrors
    #difference_ = (identifiers[1] - identifiers[0])[:2]
    #if difference_[0] == 0: row_major = True
    #else: row_major = False    
    original_img = []
    prev_i, prev_j = 0, 0
    curr_chunk = [patches[0]] # WLOG, rows can also be cols
    for k, identifier in enumerate(identifiers[1:]):
        k = k + 1
        i, j = identifier[:2]

        if row_major and j != prev_j and i == prev_i: 
            curr_chunk.append(patches[k])

        elif row_major and i != prev_i: # new row
            concat_row = jnp.concatenate(curr_chunk, axis=1)
            original_img.append(concat_row)
            curr_chunk = [patches[k]]
            
        elif not row_major and i != prev_i and j == prev_j: 
            curr_chunk.append(patches[k])
            prev_i += 1

        elif not row_major and j != prev_j: # new column
            concat_col = jnp.concatenate(curr_chunk, axis=0)
            original_img.append(concat_col)
            curr_chunk = [patches[k]]
        
        prev_i, prev_j = i, j

    # add last one
    if row_major: 
        concat_row = jnp.concatenate(curr_chunk, axis=1)
        original_img.append(concat_row)
    else:
        concat_col = jnp.concatenate(curr_chunk, axis=1)
        original_img.append(concat_col)

    if row_major:
        unpatched_img = jnp.concatenate(original_img, axis=0)
    else:
        unpatched_img = jnp.concatenate(original_img, axis=1)    
    return unpatched_img
    

def chunk_from_id(identifier, patches, num_w_patches, num_h_patches, all_identifiers):
    "Finds and returns the chunk in `patches` identified by `identifier`"
    # TODO: assume no rotations for now

    num_base_patches = patches.shape[0] # number of candidates
    i, j, f, rot = identifier
    flipped, rotated = all_identifiers[:,2:3], all_identifiers[:,3:4]
    flip_aug = (flipped == 1).any()
    rot_aug = not (rotated == 0).all()

    if flip_aug: num_base_patches = num_base_patches / 2
    if rot_aug: num_base_patches  = num_base_patches / 4

    # row major: finding n_w_patches
    patch_pos = i * (num_base_patches / num_h_patches) + j
    
    # case 0: candidates generated without flips and rotations
    if flip_aug and not rot_aug:
        # https://jax.readthedocs.io/en/latest/errors.html
        patch_pos = jnp.where(f==1, patch_pos + num_base_patches, patch_pos)
    # case 1: rotations only
    elif rot_aug and not flip_aug:
        # if not rotated, we take the previous one in order.
        patch_pos = jnp.where(rot==1, patch_pos + 1*num_base_patches, patch_pos) # 90
        patch_pos = jnp.where(rot==2, patch_pos + 2*num_base_patches, patch_pos) # 180
        patch_pos = jnp.where(rot==3, patch_pos + 3*num_base_patches, patch_pos) # 270

    # case 2: both flips and rotations
    elif rot_aug and flip_aug:

        patch_pos = jnp.where(f==1, patch_pos + num_base_patches, patch_pos)
        patch_pos = jnp.where(rot==1, patch_pos + 1*num_base_patches, patch_pos) # 90
        patch_pos = jnp.where(rot==2, patch_pos + 2*num_base_patches, patch_pos) # 180
        patch_pos = jnp.where(rot==3, patch_pos + 3*num_base_patches, patch_pos) # 270
        
    return patches[patch_pos.astype(int)]
        

def distortion(range_block, domain_block, alpha, t0):
    return jnp.sum((range_block-((alpha * domain_block) + t0))**2)


def reduce_keep_channels(chunks, factors, padding='SAME'):
    """Reduction operation on chunks / images of dimensions [batch, H, W, C]. 
       Returns a pooled version of dimensions [batch, H / factors[0], W / factors[1], C]
       
    Notes:
        We assume (H, W) are divisible by factors.
    """
    h_factor, w_factor = factors
    strides = (h_factor, w_factor)
    output_shape = (chunks.shape[1] // h_factor, chunks.shape[2] // w_factor)
    window_shape = (h_factor, w_factor)

    # silent bug: if chunks does not have a channel dimension, nn.avg_pool will reduce the batch size!
    if len(chunks.shape) < 4: chunks = chunks[..., None]
    assert len(chunks.shape) == 4, "chunks must have 4 dims: [B, H, W, C]"

    reduced = nn.avg_pool(chunks, window_shape, strides=strides, padding=padding)
    redced = reduced[0]

    return reduced

def reduce(chunks, factors, padding='SAME'):
    """Reduction operation on chunks / images of dimensions [batch, H, W, C]. 
       Returns a pooled version of dimensions [batch, H / factors[0], W / factors[1], C]
    """   
    reduced = reduce_keep_channels(chunks, factors, padding=padding)
    # reduced = reduce_keep_channels(chunks, factors) # , padding="SAME")
    reduced = reduced[...,0]
    return reduced

def reduce_all(chunks, factors, padding='SAME'):
    # this is the case where the batch dim is mapped over (i.e. manual aux sources)
    h_factor, w_factor = factors
    strides = (h_factor, w_factor)
    # output_shape = (chunks.shape[1] // h_factor, chunks.shape[2] // w_factor)
    window_shape = (h_factor, w_factor)

    if len(chunks.shape) < 4: chunks = chunks[None, ...]
    assert len(chunks.shape) == 4, "chunks must have 4 dims: [B, H, W, C]"

    reduced = nn.avg_pool(chunks, window_shape, strides=strides, padding=padding)
    reduced = reduced[0]

    return reduced
