"""
Collage decoder variants
"""

from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax_src.nn.modules import ResidualBlock
from utils.fractal_utils import augment_sample, generate_candidates
from utils.transforms import (faster_unpartition_img, partition_img, reduce)
from utils.vae_utils import *
from jax import vmap
from jax.image import resize


class GeneralizedCollageOperator(nn.Module):
    input_channels : int
    input_res : int
    rh : int 
    rw : int 
    quaddepth: int # depth of quadtree used to generate square domains
    dh : int = -1
    dw : int = -1
    decode_steps : int = 10
    hypernet_bottleneck : float = 0.5
    residual : int = True

    def setup(self):
        tiling_info = self.compute_num_patches(input_res=self.input_res)
        self.n_dh, self.n_dw, self.n_rh, self.n_rw, self.h_factors, self.w_factors, self.n_domains, self.n_ranges = tiling_info

        tot_n_domains = sum(self.n_domains)
        out_dims = (tot_n_domains*self.n_ranges + self.n_ranges)*self.input_channels
        self.hypernet = ResidualBlock(
            in_width=self.input_channels,
            middle_width=int(self.hypernet_bottleneck*out_dims),
            out_width=out_dims,
            residual=False) # 1x1 kernel + bias
          
    def __call__(self, x):
        tot_n_domains = sum(self.n_domains)
        collage_params = self.hypernet(x)
        collage_params = nn.avg_pool(collage_params, (collage_params.shape[1], collage_params.shape[2]))[:,0,0,:]
        weight, bias = jnp.split(collage_params, [tot_n_domains*self.n_ranges*self.input_channels], axis=-1) 
        weight = weight.reshape((-1, self.n_ranges, tot_n_domains, 1, 1, self.input_channels))
        bias = bias.reshape((-1, self.n_ranges, self.input_channels))

        # normalization
        bias = bias / jnp.linalg.norm(bias, axis=1)[:,None]
        weight = weight / jnp.linalg.norm(weight, axis=(1,2))[:,None,None]

        def decode_step(z, weight, bias): 
            domains = []
            for depth in range(self.quaddepth):
                new_domains, _ = partition_img(z, num_h_chunks=self.n_dh[depth], num_w_chunks=self.n_dw[depth])
                # pool patches (pre augmentation) for compatibility with range partition shapes
                pooled_domains = reduce(new_domains, (self.h_factors[depth], self.w_factors[depth]))  
                domains += [pooled_domains]
            
            domains = jnp.concatenate(domains, 0)
            domains = jnp.broadcast_to(domains, (self.n_ranges, tot_n_domains, self.rh, self.rw))
            
            out = jax.lax.conv_general_dilated(
                domains,
                rhs=weight,
                dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                padding="SAME",
                window_strides=(1,1),
            )
            
            out = jnp.sum(out + bias[:,None, None, None], axis=0)
            
            out = faster_unpartition_img(out, self.n_rh, self.n_rw)
            
            return out

        z = jnp.zeros_like(x)
        vmapped_decode_step = vmap(vmap(decode_step, in_axes=(0,0,0)), in_axes=(-1,-1,-1), out_axes=-1)
        for _ in range(self.decode_steps):
            z = vmapped_decode_step(z, weight, bias)
        
        zmin, zmax = jnp.min(z, axis=(1,2,3))[:,None,None,None], jnp.max(z, axis=(1,2,3))[:,None,None,None]
        z = (z - zmin) / (zmax - zmin)
        if self.residual: z = z + x
        return z

    def compute_num_patches(self, x=None, input_res=None):
        "Computes auxiliary information for the collage (number of domains, factors)"
        data_res = input_res if input_res else x.shape[1]
        n_dh, n_dw = [data_res // self.dh], [data_res // self.dw]
        n_domains = [n_dh[-1]*n_dw[-1]]
        h_factors, w_factors = [self.dh // self.rh], [self.dw // self.rw]
        for _ in range(self.quaddepth)[1:]: 
            h_factors += [h_factors[-1]//2]
            w_factors += [w_factors[-1]//2]
            n_dh += [n_dh[-1]*2]
            n_dw += [n_dw[-1]*2]
            n_domains += [n_domains[-1]*4]
        n_rh, n_rw = data_res // self.rh, data_res // self.rw
        n_ranges = n_rh * n_rw
        return n_dh, n_dw, n_rh, n_rw, h_factors, w_factors, n_domains, n_ranges


class GeneralizedBiasCollageOperator(GeneralizedCollageOperator):
    input_channels : int
    input_res : int
    rh : int 
    rw : int 
    quaddepth: int # depth of quadtree used to generate square domains
    dh : int = -1
    dw : int = -1
    decode_steps : int = 10
    hypernet_bottleneck : float = 0.5
    residual : int = True

    def setup(self):
        tiling_info = self.compute_num_patches(input_res=self.input_res)
        self.n_dh, self.n_dw, self.n_rh, self.n_rw, self.h_factors, self.w_factors, self.n_domains, self.n_ranges = tiling_info

        tot_n_domains = sum(self.n_domains)
        out_dims = self.n_ranges*self.input_channels
        self.hypernet = ResidualBlock(
            in_width=self.input_channels,
            middle_width=int(self.hypernet_bottleneck*out_dims),
            out_width=out_dims,
            residual=False) # 1x1 kernel + bias

        self.weight = self.param('weight', nn.initializers.zeros, (self.n_ranges, tot_n_domains, 1, 1, self.input_channels))
          
    def __call__(self, x):
        tot_n_domains = sum(self.n_domains)
        collage_params = self.hypernet(x)
        bias = nn.avg_pool(collage_params, (collage_params.shape[1], collage_params.shape[2]))[:,0,0,:]
        bias = bias.reshape((-1, self.n_ranges, self.input_channels))

        # normalization
        bias = bias / jnp.linalg.norm(bias, axis=1)[:,None]
        #key = jax.random.PRNGKey(0)
        #bias = bias + 10*jax.random.normal(key, bias.shape)

        def decode_step(z, weight, bias): 
            domains = []
            for depth in range(self.quaddepth):
                new_domains, _ = partition_img(z, num_h_chunks=self.n_dh[depth], num_w_chunks=self.n_dw[depth])
                # pool patches (pre augmentation) for compatibility with range partition shapes
                pooled_domains = reduce(new_domains, (self.h_factors[depth], self.w_factors[depth]))  
                domains += [pooled_domains]
            
            domains = jnp.concatenate(domains, 0)
            domains = jnp.broadcast_to(domains, (self.n_ranges, tot_n_domains, self.rh, self.rw))
            
            out = jax.lax.conv_general_dilated(
                domains,
                rhs=weight,
                dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                padding="SAME",
                window_strides=(1,1),
            )
            
            out = jnp.sum(out + bias[:,None, None, None], axis=0)
            
            out = faster_unpartition_img(out, self.n_rh, self.n_rw)
            
            return out

        z = jnp.zeros_like(x)
        vmapped_decode_step = vmap(vmap(decode_step, in_axes=(0,None,0)), in_axes=(-1,-1,-1), out_axes=-1)
        for _ in range(self.decode_steps):
            z = vmapped_decode_step(z, self.weight, bias)
        
        zmin, zmax = jnp.min(z, axis=(1,2,3))[:,None,None,None], jnp.max(z, axis=(1,2,3))[:,None,None,None]
        z = (z - zmin) / (zmax - zmin)
        if self.residual: z = z + x
        return z


class CollageDecoder(nn.Module):
    """
    Decoder for collages represented as contraction map parameters `theta`. 
    Both regular (one-to-one) and soft (many-to-one) collages can be specified.
    Domain partitions smaller than the image itself are handled via soft-collages i.e. all ranges are linked to all domains. 

    Args:
        rh (int) : height dimension of range partitions. 
        rw (int) : width dimension of range partitions.
        dh (int) : height dimension of domain partitions.
        dw (int) : width dimension of domain partitions.
    Notes:
        config expects the following fields:
            config.data_res, config.data_width
    """
    config : dict
    rh : int 
    rw : int 
    dh : int = -1
    dw : int = -1
    separate_maps_per_channel : bool = True
    decode_steps : int = 10
    n_aux_sources : int = 0
    n_aux_latents: int = 0
    use_rotation_aug : bool = False
    stylized_img_source : Optional[nn.Module] = None

    def setup(self):
        if self.dh < 0 and self.dw < 0: dh = dw = self.config.data_res # this should be updated (manually?) if source pooled before
        else: dh, dw = self.dh, self.dw
        self.h_factor, self.w_factor = dh // self.rh, dw // self.rw
        self.n_dh_patches, self.n_dw_patches = self.config.data_res // dh, self.config.data_res // dw
        self.n_rh_patches, self.n_rw_patches = self.config.data_res // self.rh, self.config.data_res // self.rw
        self.n_range_patches = self.n_rh_patches * self.n_rw_patches 

        # Calculation of n_domain_patches: 
        # (1) partitioning of the image itself (n_dh_patches * n_dw_patches) + 
        # (2) augmentation-based domain patches (four times if rotations, in clockwise order) +
        # (3) `n_aux_sources` additional learned source patches
        self.n_domain_patches = self.n_dh_patches * self.n_dw_patches 
        if self.use_rotation_aug: self.n_domain_patches = 4 * self.n_domain_patches + 1
        self.n_domain_patches += self.n_aux_sources # ~~connections
        self.n_domain_patches += self.n_aux_latents
        if self.config.augment_aux_sources:
            additional_aux_sources = 10 - self.n_aux_sources # TODO: additional_aux_sources = 10 - self.n_aux_sources? 
            self.n_domain_patches += additional_aux_sources
            print(f'aux sources have been augmented, added {additional_aux_sources} to precompute theta dims')

        # if there is no specification of `collage_width`, assume output collage operator o of width = data channels
        if not self.config.collage_width: 
            self.config.collage_width = self.config.data_width

        # different `mul` and `add` for each image channel (i.e. width), used to reshape `theta`
        self.n_param_copies = 1 if not self.separate_maps_per_channel else self.config.collage_width

        self.aux_sources = self.param('aux_sources', nn.initializers.normal(), (self.n_aux_sources, self.rh, self.rw, self.config.collage_width))

        if self.config.style_source_img:
            # pool down original image to range block dims
            assert self.stylized_img_source is not None, f'no styled_img_source was provided'
            # pooled_img_src = preprocess_style_blocks(config, config.reduce_type)
            self.aux_sources = self.stylized_img_source
        
        self.theta_width = self.n_range_patches * 3 * self.n_domain_patches * self.n_param_copies


        if self.config.depthwise_patch_to_theta:
            self.patch_to_theta =  nn.Conv(self.theta_width, 
                kernel_size=(self.rh, self.rw), 
                strides=(1, 1), 
                padding="SAME",
                precision=None,
                use_bias=False,
                feature_group_count=self.n_range_patches)
        else:
            self.patch_to_theta = get_1x1(self.theta_width)

        # if n_aux_latents > 1, we need a 1x1 conv to bring down to collage width
        if self.n_aux_latents > 0:
            self.aux1x1 = get_1x1(self.config.collage_width * self.n_aux_latents)

    
    def __call__(self, patch_emb, rng_key, bs=1, superres_factor=1, significant_digits=-1, aux_latents=None):
        """Decodes an image as the fixed-point of a collage with contration parameters `theta`.

        Args:
            patch_emb : (bs, rh, rw, n_range_patches * patch_emb_sz). Patch embeddings generated by an encoder.
            bs (int) : batch size of the decoded iterate
            rng_key ([type]): [description]
        """
        h, w, c = self.config.data_res * superres_factor, self.config.data_res * superres_factor, self.config.collage_width
        rh, rw = self.rh * superres_factor, self.rw * superres_factor

        # extract mul, add, mixing_w from theta. pooling kernel is not `superres_factor` dependent.
        theta = self.patch_to_theta(patch_emb)
        if theta.shape[1] > 1: theta = nn.avg_pool(theta, (theta.shape[1], theta.shape[2]))
        assert theta.shape == (bs, 1, 1, self.theta_width)

        theta = jnp.reshape(theta, (theta.shape[0], 3*self.n_range_patches, self.n_domain_patches, self.n_param_copies))
        if not self.separate_maps_per_channel: 
            theta = jnp.repeat(theta, self.config.collage_width, axis=-1)

        mul, add, mixing_w = jnp.split(theta, 3, axis=1)
        mul, add, mixing_w = nn.tanh(mul), nn.tanh(add), nn.softmax(mixing_w, axis=1)

        # premix
        mul, add = mul * mixing_w, add * mixing_w

        # clip to enable bit-packing and further compression
        if significant_digits > 0:
            mul, add = jnp.round(mul, decimals=significant_digits), jnp.round(add, decimals=significant_digits)

        # initial condition for fp iteration
        iterate = jnp.zeros(shape=(bs, h, w, c))

        # prep the latent aux. patches
        if self.n_aux_latents > 0:
            aux_latents = self.aux1x1(aux_latents) # bs, rh, rw, collage_width * n_aux_latents
            aux_latents = jnp.reshape(aux_latents, (bs, self.rh, self.rw, -1, self.config.collage_width))
            aux_latents = jnp.transpose(aux_latents, (0, 3, 1, 2, 4))


        if self.config.model == 'StyleFractalAE':

            def decode_step_style(iterate, mul, add, aux_source_pooled, reweight_d=1., reweight_a=1.):
                domain_p, identifiers = partition_img(iterate, self.n_dh_patches, self.n_dw_patches)
                domain_p, identifiers = generate_candidates(
                    domain_p, 
                    identifiers,
                    use_rotation=self.use_rotation_aug, 
                    use_flips=False
                )

                # pool patches (pre augmentation) for compatibility with range partition shapes
                domain_p = reduce(domain_p, (self.h_factor, self.w_factor)) # (4, 8, 8)

                if self.config.augment_aux_sources:
                    print(f'augmented auxiliary patches!')
                    # augment the aux sources (deterministic only)
                    _aux_sources = augment_sample(aux_source_pooled, use_rotation=True, use_flips=True)
                    aux_sources = []
                    for aux_img in _aux_sources:
                        # n_aux_sources should be 1 in this case
                        _aux_img = jnp.broadcast_to(aux_img, (self.n_range_patches, self.n_aux_sources, self.rh, self.rw))
                        if superres_factor > 1:
                            # TODO: pool patches (pre augmentation) for compatibility with range partition shapes, currently 1 always
                            aux_sources = resize(aux_sources, (self.n_range_patches, self.n_aux_sources, rh, rw), 'nearest')
                        aux_sources.append(_aux_img)


                    # doing this will treat each patch as aux image separately
                    aux_sources = jnp.concatenate(aux_sources, axis=1) # (4, 10, 8, 8) where 10 is the augmented new 'n_aux_sources'
                else:
                    aux_sources = jnp.broadcast_to(aux_source_pooled, (self.n_range_patches, self.n_aux_sources, self.rh, self.rw))


                domain_p = jnp.broadcast_to(domain_p, (self.n_range_patches, domain_p.shape[0], rh, rw)) # (4, 1, 8, 8)
                domain_p = jnp.concatenate([domain_p, aux_sources], axis=1) # (16, 4, 8, 8), (16, 1, 8, 8) | (4, 11, 8, 8)


                reweight_parameters = reweight_d != 1. or reweight_a != 1.
                if reweight_parameters:
                    # mul 'rd' d = d_domains + d_aux, then possibly do re-weighting
                    # shape is (# domain blocks, 8 range blocks)
                    # mul_aux, mul_d = (16, 1) (16, 4) --> concat to get full again (16, 5)
                    mul_aux, mul_d = reweight_a * mul[:, :self.n_aux_sources], reweight_d * mul[:, self.n_aux_sources:]
                    add_aux, add_d = reweight_a * add[:, :self.n_aux_sources], reweight_d * add[:, self.n_aux_sources:]
                    mul, add = jnp.concatenate([mul_d, mul_aux], axis=1), jnp.concatenate([add_d, add_aux], axis=1)
                    print(f'reweighted params aux / d: {reweight_a}/{reweight_d}')

                out = jnp.einsum('rd,rdhw->rhw', mul, domain_p) + jnp.sum(add, axis=1)[:,None,None]

                # unpartition the range patches back into block_dims blocks
                iterate = faster_unpartition_img(out, self.n_rh_patches, self.n_rw_patches)
                return iterate

            vmapped_decode_step = vmap(vmap(decode_step_style, in_axes=(0, 0, 0, None, None, None)), in_axes=(-1, -1, -1, -1, None, None), out_axes=-1)


        elif self.n_aux_latents > 0:

            def decode_step_latents(iterate, mul, add, aux_sources, aux_latents):
                domain_p, identifiers = partition_img(iterate, self.n_dh_patches, self.n_dw_patches)
                domain_p, identifiers = generate_candidates(
                    domain_p, 
                    identifiers, 
                    use_rotation=self.use_rotation_aug, 
                    use_flips=False
                )       
                # pool patches (pre augmentation) for compatibility with range partition shapes
                domain_p = reduce(domain_p, (self.h_factor, self.w_factor))
                aux_sources = jnp.broadcast_to(aux_sources, (self.n_range_patches, self.n_aux_sources, self.rh, self.rw))
                aux_latents = jnp.broadcast_to(aux_latents, (self.n_range_patches, self.n_aux_latents, self.rh, self.rw))

                if superres_factor > 1:
                    aux_sources = resize(aux_sources, (self.n_range_patches, self.n_aux_sources, rh, rw), 'nearest')
                    aux_latents = resize(aux_latents, (self.n_range_patches, self.n_aux_latents, rh, rw), 'nearest')

                domain_p = jnp.broadcast_to(domain_p, (self.n_range_patches, domain_p.shape[0], rh, rw))
                domain_p = jnp.concatenate([domain_p, aux_sources, aux_latents], axis=1) 

                out = jnp.einsum('rd,rdhw->rhw', mul, domain_p) + jnp.sum(add, axis=1)[:,None,None]
                iterate = faster_unpartition_img(out, self.n_rh_patches, self.n_rw_patches)
                return iterate

            vmapped_decode_step = vmap(vmap(decode_step_latents, in_axes=(0,0,0,None,0)), in_axes=(-1,-1,-1,-1,-1), out_axes=-1)
            
           
        else:

            def decode_step(iterate, mul, add, aux_sources):       
                domain_p, identifiers = partition_img(iterate, self.n_dh_patches, self.n_dw_patches)
                domain_p, identifiers = generate_candidates(
                    domain_p, 
                    identifiers, 
                    use_rotation=self.use_rotation_aug, 
                    use_flips=False
                )       
                # pool patches (pre augmentation) for compatibility with range partition shapes
                domain_p = reduce(domain_p, (self.h_factor, self.w_factor))
                aux_sources = jnp.broadcast_to(aux_sources, (self.n_range_patches, self.n_aux_sources, self.rh, self.rw))

                # pool patches (pre augmentation) for compatibility with range partition shapes
                if superres_factor > 1:
                    aux_sources = resize(aux_sources, (self.n_range_patches, self.n_aux_sources, rh, rw), 'nearest')

                domain_p = jnp.broadcast_to(domain_p, (self.n_range_patches, domain_p.shape[0], rh, rw))
                domain_p = jnp.concatenate([domain_p, aux_sources], axis=1) 

                out = jnp.einsum('rd,rdhw->rhw', mul, domain_p) + jnp.sum(add, axis=1)[:,None,None]
                iterate = faster_unpartition_img(out, self.n_rh_patches, self.n_rw_patches)
                return iterate

            vmapped_decode_step = vmap(vmap(decode_step, in_axes=(0,0,0,None)), in_axes=(-1,-1,-1,-1), out_axes=-1)


        for _ in range(self.decode_steps):
            if self.config.model == "StyleFractalAE":
                iterate = vmapped_decode_step(iterate, mul, add, self.aux_sources, self.config.reweight_factor_d, self.config.reweight_factor_a)
            elif self.n_aux_latents > 0:
                iterate = vmapped_decode_step(iterate, mul, add, self.aux_sources, aux_latents)
            else:
                iterate = vmapped_decode_step(iterate, mul, add, self.aux_sources)
                
        return iterate, mul, add