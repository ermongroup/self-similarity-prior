"""
Modules and layer primitives
"""
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
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
        # OR
        # self.theta_width = self.n_range_patches * 3 * self.n_aux_sources * self.n_param_copies
        # (bs, rh, rw, n_range_patches * patch_emb_sz) --> (bs, rh, rw, theta_width)

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

    # theta_width example: (32, 32) image (2, 2) patches, 256 (n_domain = 5, n_copies = 3), 256 * 3 * 5 (~~connections) * 3 = 11520
    # typical "patch_emb" size: patch_emb_sz = 100 (2, 2) patches, 256 patches (n_range_patches), 256 * 100 = 25600
    # self.patch_to_theta 1x1 conv 768 -> 11520
    # (2, 2) patches (rh, rw) 1x1 conv 768 * 11520 = ~9mil   
    #  self.rh * self.rw * 45 (# groups = theta_widths / n_patches) = 4 * 4 * 45 = ~ 600   
    
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
        #iterate = jax.random.normal(key=rng_key, shape=(bs, h, w, c))
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

                # if superres_factor > 1:
                #     aux_sources = resize(aux_sources, (self.n_range_patches, self.n_aux_sources, rh, rw), 'nearest')

                domain_p = jnp.broadcast_to(domain_p, (self.n_range_patches, domain_p.shape[0], rh, rw)) # (4, 1, 8, 8)
                domain_p = jnp.concatenate([domain_p, aux_sources], axis=1) # (16, 4, 8, 8), (16, 1, 8, 8) | (4, 11, 8, 8)

                # mul, add .... `rd` (num_range_patches, num_domain_patches) 
                # num_domain_patches will have a mismatch if you throw away a bunch of domains
                # mul, add = mul[:,:1]

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

                # pool patches (pre augmentation) for compatibility with range partition shapes
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


class FourierBlock(nn.Module):
  """The Fourier input modification used in VDM."""
  n: Tuple[int] = (7, 8)  # or 8

  @nn.compact
  def __call__(self, x):
    x_xfourier = x
    for f in self.n:
        x_proj = x * jnp.pi * 2 ** f
        # concatenate extra channels for each n to input to score network
        x_xfourier = jnp.concatenate([x_xfourier, jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
    return x_xfourier


class ResidualBlock(nn.Module):
    "Residual building block of encoder and decoders of VAEs"
    in_width : int
    middle_width : int
    out_width : int
    res : int = None
    op : str = 'd'
    act : Any = nn.gelu
    down_rate : int = 0
    residual : bool = True
    use_3x3 : bool = True
    zero_last : bool = False
    partial_down_rates : Tuple = ()
    last_rescale_factor : int = 1

    def setup(self):
        self.conv1 = get_1x1(self.middle_width) 
        self.conv2 = get_3x3(self.middle_width) if self.use_3x3 else get_1x1(self.middle_width) 
        self.conv3 = get_3x3(self.middle_width) if self.use_3x3 else get_1x1(self.middle_width)
        if self.zero_last: init_weights_fn = nn.initializers.zeros
        else: init_weights_fn = nn.initializers.variance_scaling(self.last_rescale_factor, 'fan_in', 'truncated_normal')
        self.conv4 = get_1x1(self.out_width, init_weights_fn=init_weights_fn) 

        self.convlayers = [self.conv1, self.conv2, self.conv3]
        
    def __call__(self, inputs):
        transformed_inputs = inputs
        for conv in self.convlayers:
            transformed_inputs = conv(self.act(transformed_inputs))
        transformed_inputs = self.conv4(transformed_inputs)
        outputs = transformed_inputs + inputs if self.residual else transformed_inputs
        if self.down_rate > 1:
            window_shape = 2 * (self.down_rate,)
            outputs = nn.avg_pool(outputs, window_shape, window_shape)
        return outputs


class Encoder(nn.Module):
    config : dict
    
    def setup(self):
        blockstr, n_blocks = parse_layer_string('VDVAE', self.config.enc_blocks)
        widths = get_width_settings(self.config.enc_default_width, self.config.enc_widths)
        rescale_factor_init = np.sqrt(1 / n_blocks**2)
        init_weights_fn = nn.initializers.variance_scaling(rescale_factor_init, 'fan_in', 'truncated_normal')

        self.in_conv = get_3x3(self.config.enc_default_width, init_weights_fn=init_weights_fn)
        self.fourier_block = FourierBlock(n=self.config.n_fourier_enc)

        self.blocks = ()
        for idx, (res, _, down_rate) in enumerate(blockstr):
            block_name = f'EncoderBlock_{idx}_{res}x{res}_d{down_rate}'
            use_3x3 = res > 2
            _in_width = widths[res]
            block = ResidualBlock(
                in_width=_in_width,
                middle_width=int(widths[res] * self.config.enc_bottleneck_factor),
                out_width=widths[res],
                down_rate=down_rate, 
                residual=True,
                use_3x3=use_3x3,
                last_rescale_factor=rescale_factor_init,
                name=block_name
            )
            self.blocks = self.blocks + (block,)


    def __call__(self, inputs):
        # if self.config.use_fourier_enc: # identity if this is ()
        inputs = self.fourier_block(inputs)

        # defaultdict is frozen in setup, need to get it again within __call__
        widths = get_width_settings(self.config.enc_default_width, self.config.enc_widths)

        activations = {}
        inputs = self.in_conv(inputs)
        activations[inputs.shape[1]] = inputs
        for k, block in enumerate(self.blocks):
            inputs = block(inputs)
            res = inputs.shape[1]
            inputs = inputs if inputs.shape[-1] == widths[res] else pad_channels(inputs, widths[res]) # is this even used anywhere?
            activations[res] = inputs
        return activations


class DecoderBlock(nn.Module):
    config : dict
    in_width : int
    aug_width : int 
    middle_width : int
    out_width : int
    res : int
    mixin : int
    last_rescale_factor : int
    latent_dim : int = 0

    def setup(self):
        use_3x3 = self.res > 2
        if self.latent_dim <= 0: self.z_dim = self.config.dec_latent_dim
        else: self.z_dim = self.latent_dim

        self.posterior = ResidualBlock(self.in_width * 2 + self.aug_width, 
                                      self.middle_width, 
                                      self.z_dim * 2, 
                                      residual=False,
                                      zero_last=False,
                                      use_3x3=use_3x3)
        self.prior = ResidualBlock(self.in_width + self.aug_width, 
                                    self.middle_width, 
                                    self.z_dim * 2 + self.in_width,
                                    residual=False,
                                    zero_last=True,
                                    use_3x3=use_3x3)
        self.resnet = ResidualBlock(self.in_width,
                self.middle_width, 
                self.in_width, 
                use_3x3=use_3x3, 
                residual=True,
                last_rescale_factor=self.last_rescale_factor)

        init_weights_fn = nn.initializers.variance_scaling(self.last_rescale_factor, 'fan_in', 'truncated_normal')
        self.z_proj = get_1x1(self.out_width, init_weights_fn=init_weights_fn)


    def __call__(self, input, inputs, enc_feat, block_rng):
        input = jnp.broadcast_to(input, enc_feat.shape)
        if self.mixin > 0: 
            # mixing lower resolution input feature maps by upscaling them to match self.res
            input_to_mix = inputs[self.mixin]
            b, h, w, c = input.shape
            #input_to_mix = jnp.broadcast_to(input_to_mix, (b,) + input_to_mix.shape)
            input = input + resize(input_to_mix, (b, h, w, c), 'nearest')

        latent, input, kl = self.sample(input, enc_feat, block_rng)
        
        input = input + self.z_proj(latent)
        input = self.resnet(input)

        # TODO: saving latents for plotting (in the dictionary below)
        return input, dict(kl=kl)

    def sample(self, input, enc_feat, block_rng):
        concat_features = jnp.concatenate([input, enc_feat], axis=-1)
        prior_feats = self.prior(input)
        pm, pv, xpp = jnp.split(prior_feats, [self.z_dim, 2 * self.z_dim], -1)
        qmqv = self.posterior(concat_features)
        qm, qv = jnp.split(qmqv, 2, axis=-1)
        if self.config.residual_posteriors: qm, qv = qm + pm, qv * pv

        input = input + xpp
        latent = draw_gaussian_diag_samples(block_rng, qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return latent, input, kl

    def forward_uncond(self, input, inputs, block_rng, temp=None):
        if self.mixin > 0: 
            # mixing lower resolution input feature maps by upscaling them to match self.res
            input_to_mix = inputs[self.mixin]
            b, h, w, c = input.shape

            #input_to_mix = jnp.broadcast_to(input_to_mix, (b,) + input_to_mix.shape)
            input = input + resize(input_to_mix, (b, h, w, c), 'nearest')

        z, x = self.sample_uncond(input, block_rng, temp)
        return self.resnet(x + self.z_proj(z))

    def sample_uncond(self, inputs, block_rng, temp=None):
        pm, pv, xpp = jnp.split(self.prior(inputs), [self.z_dim, 2 * self.z_dim], 3)
        samples = draw_gaussian_diag_samples(block_rng, pm, pv * (temp or self.config.base_temperature)) # will eval to temp unless temp=0. or temp=None
        return samples, inputs + xpp


class DecoderHead(nn.Module): 
    config : dict

    def setup(self):
        if self.config.likelihood_func == "dmol": 
            # (bs, h, w, n_mix_coefs + n_mix_params * n_channels)
            out_dims = self.config.dec_num_mixtures * ((self.config.data_width * 3) + 1)
            
        elif self.config.likelihood_func == "bernoulli": 
            out_dims = 1

        self.out_conv = get_1x1(out_dims) 
        self.likelihood_func = LIKELIHOODS[self.config.likelihood_func]
        self.sampling_func = SAMPLERS[self.config.likelihood_func]

    def negative_log_likelihood(self, px_z, x_out):
        if self.config.collage_width: params = self.out_conv(px_z)
        else: params = px_z # collage operator already produces `px_z` of dims `bs, h, w, out_dims`
        negative_log_likelihood = -self.likelihood_func(params, x_out, n_channels=self.config.data_width)
        return negative_log_likelihood

    def sample(self, px_z, rng):
        params = self.out_conv(px_z) 
        samples = self.sampling_func(rng, params, n_channels=self.config.data_width)
        if self.config.likelihood_func == "dmol":
            samples = jnp.round((jnp.clip(samples, -1, 1) + 1) * 127.5).astype(jnp.uint8)
        else:
            samples = jnp.round(samples * 255).astype(jnp.uint8)
        return samples



class Decoder(nn.Module):
    config : dict
    
    def setup(self):     
        widths = get_width_settings(self.config.enc_default_width, self.config.dec_widths)
        self.blockstr, self.n_blocks = parse_layer_string('VDVAE', self.config.dec_blocks)
        resos = set()
        self.blocks = []
   
        # decoder has mixin, encoder has downsample
        
        for idx, (res, _, mixin) in enumerate(self.blockstr):
            block_name = f'DecoderBlock_{idx}_{res}x{res}_m{mixin}'
            block = DecoderBlock(config=self.config,
                                in_width=widths[res],
                                middle_width=int(widths[res] * self.config.dec_bottleneck_factor), 
                                out_width=widths[res],
                                aug_width=0,
                                res=res,
                                mixin=mixin,
                                last_rescale_factor=np.sqrt(1 / self.n_blocks**2),
                                name=block_name)
            self.blocks = self.blocks + (block,)
            resos.add(res)
        
        self.resolutions = sorted(resos)

        aug_width = 2 + 4 * len(self.config.n_fourier_dec) if self.config.use_fourier_dec else 0

        self.input_maps = [
            jnp.zeros((res, res, widths[res] - aug_width))
            for i, res in enumerate(self.resolutions) if res <= self.config.no_bias_above] 

        if self.config.use_fourier_dec:
            self.fourier_block = FourierBlock(n=self.config.n_fourier_dec)
            pos_feats = []
            for i, res in enumerate(self.resolutions):
                if res <= self.config.no_bias_above:
                    x = y = jnp.linspace(0, res - 1, res)
                    xx, yy = jnp.meshgrid(x, y)
                    zz = jnp.stack([xx, yy], axis=-1) / res
                    pos_feats.append(self.fourier_block(zz))

            self.input_maps = [jnp.concatenate([i, p], axis=-1) for (i, p) in zip(self.input_maps, pos_feats)]
        
        self.out_net = DecoderHead(self.config)
        self.final_fn = WidthAffineMap(widths[self.config.data_res])
        self.base_res = self.config.data_res

        # Initialize parametrized, empty feature maps to mix with encoder feats at each decoder block
        # self.input_maps = [
        #     self.param(f'bias_xs_{i}', nn.initializers.zeros,
        #                (res, res, widths[res]))
        #     for i, res in enumerate(self.resolutions) if res <= self.config.no_bias_above] # TODO: add to 
        # self.n_fourier_feats = self.config.n_fourier
        # self.fourier_block = FourierBlock(n=self.n_fourier_feats).apply

    def __call__(self, enc_maps, rng):
        """
        Notes:
            Resolution specific chaining of input/outputs
            Example:
            blockstr: "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
            iteration 0////  xs{1: (1,1,w[1]), 4: (4,4,w[4]), 8: (8,8,w[8]), 16: (16,16,w[16]), 32: (32,32,w[32])}
            iteration 1////   xs{1: x1, 4: (4,4,w[4]), 8: (8,8,w[8]), 16: (16,16,w[16]), 32: (32,32,w[32])}
            iteration 2////   xs{1: x1, 4: x2, 8: (8,8,w[8]), 16: (16,16,w[16]), 32: (32,32,w[32])}    
            iteration 3////   xs{1: x1, 4: x3, 8: (8,8,w[8]), 16: (16,16,w[16]), 32: (32,32,w[32])}   

            x2 = decoderblock(bias(4,4,w[4]), encoder_feat)

            Vanilla "full chaining"
            x0 = zeros(perfect_shape)
            x1, ** = decoderblock(x0, enc_feats)
            x2, ** = decoderblock(x1, enc_feats)
            ...
            x_{N}, ** = decoderblock(x_{N-1}, enc_feats)
        """
        stats = []
        bs = enc_maps[self.blocks[0].res].shape[0]

        input_maps = {im.shape[0] : jnp.broadcast_to(im, (bs,) + im.shape) for im in self.input_maps}
 
        for k, block in enumerate(self.blocks):
            res = block.res

            enc_feat, inputs = enc_maps[res], input_maps[res] 

            rng, block_rng = jax.random.split(rng)
            inputs, block_stats = block(inputs, input_maps, enc_feat, block_rng)
            
            stats.append(block_stats)
            input_maps[res] = inputs

        output = self.final_fn(input_maps[res])
        
        return output, stats

    def forward_uncond(self, n_samples, rng, temperature=None):

        input_maps = {im.shape[0] : jnp.broadcast_to(im, (n_samples,) + im.shape) for im in self.input_maps}
        
        for k, block in enumerate(self.blocks):
            res = block.res
            inputs = input_maps[res]

            t_block = temperature[k] if isinstance(temperature, list) else temperature

            rng, block_rng = jax.random.split(rng)
            inputs = block.forward_uncond(inputs, input_maps, block_rng, temp=t_block)
            #print(inputs.max(), res, k)
            input_maps[res] = inputs
        output = self.final_fn(input_maps[res])
        return output


class LatAuxVDCVAEDecoder(Decoder):
    config : dict
    
    def setup(self):     
        super().setup()
        aux_widths = get_width_settings(self.config.enc_default_width, self.config.dec_widths)
        self.aux_blockstr, n_blocks = parse_layer_string('VDVAE', self.config.aux_dec_blocks)
        resos = set()
        self.aux_blocks = []
        
        for idx, (res, _, mixin) in enumerate(self.aux_blockstr):
                block_name = f'AuxDecoderBlock_{idx}_{res}x{res}_m{mixin}'
                block = DecoderBlock(config=self.config,
                                    in_width=aux_widths[res],
                                    middle_width=int(aux_widths[res] * self.config.dec_bottleneck_factor), 
                                    out_width=aux_widths[res],
                                    aug_width=0,
                                    res=res,
                                    mixin=mixin,
                                    latent_dim=self.config.dec_aux_latent_dim,
                                    last_rescale_factor=np.sqrt(1 / n_blocks**2),
                                    name=block_name)
                resos.add(res)
                self.aux_blocks = self.aux_blocks + (block,)

        self.resolutions = sorted(resos)

        self.aux_input_maps = [
            jnp.zeros((res, res, aux_widths[res]))
            for i, res in enumerate(self.resolutions) if res <= self.config.no_bias_above] 

        self.final_auxfn = WidthAffineMap(aux_widths[self.config.data_res])

    def __call__(self, enc_maps, rng):
        
        output_theta, stats_theta = super().__call__(enc_maps, rng)
        output_aux, stats_aux = self.forward_aux(enc_maps, rng)
        return output_theta, output_aux, stats_theta, stats_aux

    def forward_uncond(self, n_samples, rng, temperature=None):
        output_theta = super().forward_uncond(n_samples, rng, temperature)
        output_aux = self.forward_aux_uncond(n_samples, rng, temperature)
        return output_theta, output_aux

    def forward_aux(self, enc_maps, rng):
        stats_aux = []
        bs = enc_maps[self.aux_blocks[0].res].shape[0]

        input_maps = {im.shape[0] : jnp.broadcast_to(im, (bs,) + im.shape) for im in self.aux_input_maps}

        for k, block in enumerate(self.aux_blocks):
            res = block.res
            enc_feat, inputs = enc_maps[res], input_maps[res] 

            rng, block_rng = jax.random.split(rng)
            inputs, block_stats = block(inputs, input_maps, enc_feat, block_rng)
            
            stats_aux.append(block_stats)
            input_maps[res] = inputs

        output_aux = self.final_auxfn(input_maps[res])
        return output_aux, stats_aux

    def forward_aux_uncond(self, n_samples, rng, temperature):
        input_maps = {im.shape[0] : jnp.broadcast_to(im, (n_samples,) + im.shape) for im in self.aux_input_maps}

        for k, block in enumerate(self.aux_blocks):
            res = block.res
            inputs = input_maps[res]

            t_block = temperature[k] if isinstance(temperature, list) else temperature

            rng, block_rng = jax.random.split(rng)
            inputs = block.forward_uncond(inputs, input_maps, block_rng, temp=t_block)
            
            input_maps[res] = inputs

        output_aux = self.final_auxfn(input_maps[res])
        return output_aux



class WidthAffineMap(nn.Module):
    width : int

    def setup(self):
        self.gain = self.param('gain', nn.initializers.ones, (self.width,))
        self.bias = self.param('bias', nn.initializers.zeros, (self.width,))

    def __call__(self, x):
        return x * self.gain + self.bias

    def manual_forward(self, x, gain, bias):
        return x * gain + bias
        

