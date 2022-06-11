"""
Primitives for VAEs
"""
import numpy as np
from time import time
import jax
import jax.numpy as jnp
import flax.linen as nn

from utils.vae_utils import * 
from functools import partial
from utils.plot_utils import *
from jax_src.frax.modules import CollageDecoder, Encoder, Decoder, LatAuxVDCVAEDecoder


class VDCVAE(nn.Module):
    config : dict

    def setup(self):
        config = self.config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        res = config.data_res
        self.collage_operator = CollageDecoder(
            config=config,
            rh=config.range_dims[0],
            rw=config.range_dims[1],
            dh=res,
            dw=res,
            n_aux_sources=config.n_aux_sources,
            separate_maps_per_channel=config.separate_maps_per_channel,
            decode_steps=config.decode_steps,
            use_rotation_aug=config.use_rotation_aug)
      

    def __call__(self, x_in, x_out, rng, return_sample=True, superres_factor=1):
        ndims = np.prod(x_in.shape[1:])


        start_time = time()
        activations = self.encoder(x_in)
        ptheta_z, stats = self.decoder(activations, rng)

        _, collage_rng = jax.random.split(rng)

        px_theta, mul, add = self.collage_operator(ptheta_z, bs=x_in.shape[0], rng_key=collage_rng, superres_factor=superres_factor)
        end_time = time() - start_time

        distortion = self.decoder.out_net.negative_log_likelihood(px_theta, x_out) 

        rate = jnp.zeros_like(distortion)
        for statdict in stats:
            kl = jnp.sum(statdict['kl'], axis=(1,2,3))
            rate += kl
            
        rate_per_pixel, distortion_per_pixel = rate / ndims, distortion / ndims
        elbo = distortion_per_pixel + self.config.beta * rate_per_pixel
        rate_blocks = [jnp.sum(s['kl'], axis=(1,2,3)) for s in stats]
        
        nan_count = jnp.sum(jnp.isnan(elbo))
        elbo = jnp.mean(jnp.nan_to_num(elbo, nan=0), axis=0)
        distortion_per_pixel = jnp.mean(jnp.nan_to_num(distortion_per_pixel, nan=0), axis=0)
        rate_per_pixel = jnp.mean(jnp.nan_to_num(rate_per_pixel, nan=0), axis=0)

        foward_stats = dict(elbo=elbo,
                            recloss=distortion_per_pixel, 
                            kl=rate_per_pixel,
                            kl_blocks=[jnp.mean(r) / ndims for r in rate_blocks],
                            forward_time=end_time,
                            nan_count=nan_count)

        sample = None
        if return_sample:
            sample = self.decoder.out_net.sample(px_theta, rng)

        return foward_stats, sample, stats


class VDCVAESampler(nn.Module):
    config: dict

    def setup(self):
        config = self.config
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        res = config.data_res

        self.collage_operator = CollageDecoder(
            config=config,
            rh=config.range_dims[0],
            rw=config.range_dims[1],
            dh=res,
            dw=res,
            separate_maps_per_channel=config.separate_maps_per_channel,
            decode_steps=config.decode_steps,
            n_aux_sources=config.n_aux_sources,
            use_rotation_aug=config.use_rotation_aug)
 

    def __call__(self, n_batch, rng, temperature=None, superres_factor=1):
        ptheta_z = self.decoder.forward_uncond(n_batch, rng, temperature=temperature)

        _, collage_rng = jax.random.split(rng)
        px_theta, _, _ = self.collage_operator(ptheta_z, bs=n_batch, rng_key=collage_rng, superres_factor=superres_factor)
        return self.decoder.out_net.sample(px_theta, rng)


class LatAuxVDCVAE(nn.Module):
    config : dict

    def setup(self):
        config = self.config
        self.encoder = Encoder(config)
        self.decoder = LatAuxVDCVAEDecoder(config)

        res = config.data_res
        self.collage_operator = CollageDecoder(
            config=config,
            rh=config.range_dims[0],
            rw=config.range_dims[1],
            dh=res,
            dw=res,
            n_aux_sources=config.n_aux_sources,
            n_aux_latents = config.n_aux_latents,
            separate_maps_per_channel=config.separate_maps_per_channel,
            decode_steps=config.decode_steps,
            use_rotation_aug=config.use_rotation_aug)
      

    def __call__(self, x_in, x_out, rng, return_sample=True, superres_factor=1):
        ndims = np.prod(x_in.shape[1:])

        # if self.config.likelihood_func == "dmol":
        #     x_out = 2*x_out - 1 # rescale from [0, 1] to [-1, 1]
        
        start_time = time()
        activations = self.encoder(x_in)
        ptheta_z, paux_z, stats_theta, stats_aux = self.decoder(activations, rng)
        
        stats = list([*stats_theta, *stats_aux])

        _, collage_rng = jax.random.split(rng)
        
        px_theta, mul, add = self.collage_operator(ptheta_z, bs=x_in.shape[0], rng_key=collage_rng, superres_factor=superres_factor, aux_latents=paux_z)
        
        end_time = time() - start_time

        distortion = self.decoder.out_net.negative_log_likelihood(px_theta, x_out) 

        rate = jnp.zeros_like(distortion)
        for statdict in stats:
            kl = jnp.sum(statdict['kl'], axis=(1,2,3))
            rate += kl
                      
        rate_per_pixel, distortion_per_pixel = rate / ndims, distortion / ndims
        elbo = distortion_per_pixel + self.config.beta * rate_per_pixel
        rate_blocks = [jnp.sum(s['kl'], axis=(1,2,3)) for s in stats]
        
        nan_count = jnp.sum(jnp.isnan(elbo))
        elbo = jnp.mean(jnp.nan_to_num(elbo, nan=0), axis=0)
        distortion_per_pixel = jnp.mean(jnp.nan_to_num(distortion_per_pixel, nan=0), axis=0)
        rate_per_pixel = jnp.mean(jnp.nan_to_num(rate_per_pixel, nan=0), axis=0)

        foward_stats = dict(elbo=elbo,
                            recloss=distortion_per_pixel, 
                            kl=rate_per_pixel,
                            kl_blocks=[jnp.mean(r) / ndims for r in rate_blocks],
                            forward_time=end_time,
                            nan_count=nan_count)

        sample = None
        if return_sample:
            sample = self.decoder.out_net.sample(px_theta, rng)

        return foward_stats, sample, stats


class LatAuxVDCVAESampler(nn.Module):
    config: dict

    def setup(self):
        config = self.config
        self.encoder = Encoder(config)
        self.decoder = LatAuxVDCVAEDecoder(config)

        res = config.data_res
        self.collage_operator = CollageDecoder(
            config=config,
            rh=config.range_dims[0],
            rw=config.range_dims[1],
            dh=res,
            dw=res,
            n_aux_sources=config.n_aux_sources,
            n_aux_latents = config.n_aux_latents,
            separate_maps_per_channel=config.separate_maps_per_channel,
            decode_steps=config.decode_steps,
            use_rotation_aug=config.use_rotation_aug)
 

    def __call__(self, n_batch, rng, temperature=None, superres_factor=1):
        ptheta_z, paux_z = self.decoder.forward_uncond(n_batch, rng, temperature=temperature)

        _, collage_rng = jax.random.split(rng)
        px_theta, _, _ = self.collage_operator(ptheta_z, bs=n_batch, rng_key=collage_rng, superres_factor=superres_factor, aux_latents=paux_z)
        return self.decoder.out_net.sample(px_theta, rng)