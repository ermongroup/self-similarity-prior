"""Primitives for Fractally-constrained autoencoder"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import utils.train_utils as train_utils
from utils.transforms import batched_partition_img
from src.frax.fractalvae import PatchEmbeddingMixer, PixelMixer
from src.frax.modules import CollageDecoder, GeneralizedCollageOperator, GeneralizedBiasCollageOperator
from utils.vae_utils import *
from utils.plot_utils import norm_two_to_eight_bit
from utils.train_utils import *
from jax import jacfwd
from jax import jacrev
from jax import jvp

def compute_psnr(imgs, pred_imgs, max_val=255.):
    "Compute PSNR between two images or batches of 8 bit images"
    error = jnp.power(imgs - pred_imgs, 2).mean()
    psnr = 20. * jnp.log10(max_val) - 10. * jnp.log10(error)
    return psnr

class PatchEmbedder(nn.Module):
    config : dict

    def setup(self):
        blockstr, n_blocks = parse_layer_string('FractalVAE', self.config.enc_blocks)
        rescale_factor_init = np.sqrt(1 / n_blocks)
        init_weights_fn = nn.initializers.variance_scaling(rescale_factor_init, 'fan_in', 'truncated_normal')
        
        self.dw_blocks, self.pw_blocks = (), ()
        for _, partitions in enumerate(blockstr):
            h, w = partitions
            num_h_patches, num_w_patches = self.config.data_res // h, self.config.data_res // w
            n_patches = num_h_patches * num_w_patches
            self.enc_default_width = n_patches * self.config.enc_patch_emb_dim

            block = PixelMixer(
                config=self.config,
                patch_dims=(h, w),
                num_patches=n_patches,
                patch_emb_dim=self.config.enc_patch_emb_dim,
                residual=True,
                last_rescale_factor=rescale_factor_init,
                last_act=False,
            )
            self.pw_blocks = self.pw_blocks + (block,)

            block = PatchEmbeddingMixer(
                config=self.config,
                patch_dims=(h, w),
                num_patches=n_patches,
                patch_emb_dim=self.config.enc_patch_emb_dim,
                residual=True,
                last_rescale_factor=rescale_factor_init,
                last_act=False,
            )
            self.dw_blocks = self.dw_blocks + (block,)

            # we assume equipartitioning in encoding mixing layers
            self.rh, self.rw = h, w

        self.in_conv = get_3x3(self.enc_default_width, init_weights_fn=init_weights_fn)


    def __call__(self, inputs):
        num_w_chunks, num_h_chunks = self.config.data_res // self.rh, self.config.data_res // self.rw
        patches, _ = batched_partition_img(inputs, num_w_chunks, num_h_chunks)
        patches = jnp.reshape(patches, (inputs.shape[0], self.rh, self.rw, -1))

        iterate = self.in_conv(patches)
        context = [iterate]
        for k, _ in enumerate(self.pw_blocks):
            iterate = self.pw_blocks[k](iterate)
            iterate = self.dw_blocks[k](iterate)
            context.append(iterate)
        return iterate


class FCAutoencoder(nn.Module):
    """Simple MLP autoencoder module."""
    config: dict

    def setup(self):
        config = self.config
        self.layer_dims = config.hidden_sizes
        self.activation = config.activation
        self.kernel_inits = [jax.nn.initializers.variance_scaling(scale, 'fan_in', 'truncated_normal') for scale in config.kernel_init]
        assert len(self.layer_dims) == len(self.kernel_inits), \
                f"layer specifications should be same length as kernel scales: {len(self.layer_dims)} != {len(self.kernel_inits)}"
        self.bias_init = INITIALIZERS[config.bias_init]

        self.collage_layer = CollageAutoencoder(self.config)
    
    def flatten(self, x):
        return jnp.reshape(x, (x.shape[0], -1))

    def full_hessian(self, loss_fn, params):
        def hessian(f):
            return jacfwd(jacrev(f))

        hessian_matrix = hessian(loss_fn)(params)
        return hessian_matrix

    def mse(self, logits, x):
        return jnp.mean(jnp.square(self.flatten(x) - jax.nn.sigmoid(logits)), [1])

    def __call__(self, x, x_, rng):
        z = self.flatten(x)
        num_outputs = z.shape[-1]
        for i, (num_hid, k_init) in enumerate((self.layer_dims, self.kernel_inits)):
            z = nn.Dense(num_hid, kernel_init=k_init, bias_init=self.bias_init)(z)
            z = ACTIVATIONS[self.activation](z)
        z = nn.Dense(num_outputs, kernel_init=k_init, bias_init=self.bias_init)(z)
        z = self.collage_layer(z, x_, rng)

        loss = self.mse(z, x)
        psnr = compute_psnr(norm_two_to_eight_bit(x), norm_two_to_eight_bit(z), max_val=255.)

        return jnp.mean(loss), (z, psnr)


class CollageAutoencoder(nn.Module):
    config : dict

    def setup(self):
        config = self.config
        CO = GeneralizedCollageOperator if config.co == 'general' else GeneralizedBiasCollageOperator

        self.coll_layers = ()
        for k in range(3):
            coll_layer = CO(
                input_channels=config.data_width,
                input_res=config.data_res,
                dh=config.dhs[k],
                dw=config.dws[k],
                rh=config.rhs[k],
                rw=config.rws[k],
                quaddepth=config.quaddepths[k],
                hypernet_bottleneck=0.1,
                residual=True
            )
            self.coll_layers = self.coll_layers + (coll_layer,)
        
    def __call__(self, x, x_, rng):
        z = x_ + 2*jax.random.normal(rng, x_.shape)
        
        for coll_layer in self.coll_layers:
            z = coll_layer(z)
        loss = jnp.sum(jnp.mean((z - x)**2, axis=0))
        z = jnp.clip(z, 0, 1)
        psnr = compute_psnr(norm_two_to_eight_bit(x), norm_two_to_eight_bit(z), max_val=255.)
        return loss, (z, psnr)
        

class NeuralFractalAutoencoder(nn.Module):
    "Multiple source (soft) Neural Fractal Autoencoder model for fractal compression or stylization"
    config : dict

    def setup(self):
        config = self.config
        self.encoder = PatchEmbedder(self.config)
        self.decoder = CollageDecoder(
            config=self.config, 
            dh=self.config.domain_szs[0],
            dw=self.config.domain_szs[1],
            rh=self.encoder.rh, 
            rw=self.encoder.rw, 
            separate_maps_per_channel=config.separate_maps_per_channel, 
            decode_steps=config.decode_steps,
            n_aux_sources=config.n_aux_sources, 
            use_rotation_aug=config.use_rotation_aug,
            stylized_img_source=None
        )

    def __call__(self, x, rng, significant_digits=-1):
        patch_embs = self.encoder(x)
        reconstructions, mul, add = self.decoder(patch_embs, rng, bs=x.shape[0], significant_digits=significant_digits)
        loss = jnp.sum(jnp.mean((reconstructions - x)**2, axis=0))
        if self.config.l2_reg:
            loss = loss + 0.1 * jnp.sum(jnp.mean(jnp.power(mul, 2) + jnp.power(add, 2), axis=0))
        reconstructions = jnp.clip(reconstructions, 0, 1)

        psnr = compute_psnr(norm_two_to_eight_bit(x), norm_two_to_eight_bit(reconstructions), max_val=255.)
        return loss, (reconstructions, psnr, mul, add)


class NeuralFractalSuperresAutoencoder(NeuralFractalAutoencoder):
    config : dict

    def setup(self):
        super().setup()

    def __call__(self, x, rng, superres_factor=1, significant_digits=-1):
        patch_embs = self.encoder(x)
        reconstructions, _, _ = self.decoder(patch_embs, rng, superres_factor=superres_factor, bs=x.shape[0], significant_digits=significant_digits)
        return reconstructions


# .png
# preprocessing -> 1, H, W, C
# blockpreproc -> n_blocks, block_h, block_w, C <- (block_h, block_w) is different than (range_h, range_w)
# NeuralFractalStyleAutoencoder(blocks)
# if you want to decode/visualize, compress blocks in batch and then "unchunk"

# image to stylize 1, H, W, C, [optional] source image for style 1, H, W, C, chunked image to stylize n_blocks, block_h, block_w, C
#
class NeuralFractalStyleAutoencoder(NeuralFractalAutoencoder):
    "Multiple source (soft) Neural Fractal Autoencoder model for stylization"
    config : dict

    def setup(self):
        config = self.config
        assert config.style_source_img is not None, f"must specify a style_source_img for Neural fractal style autoencoder"
        self.style_source_blocks = train_utils.preprocess_style_blocks(config, config.reduce_type)
        self.encoder = PatchEmbedder(self.config)
        self.decoder = CollageDecoder(
            config=self.config, 
            dh=self.config.domain_szs[0],
            dw=self.config.domain_szs[1],
            rh=self.encoder.rh, 
            rw=self.encoder.rw, 
            separate_maps_per_channel=config.separate_maps_per_channel, 
            decode_steps=config.decode_steps,
            n_aux_sources=config.n_aux_sources, 
            use_rotation_aug=config.use_rotation_aug,
            stylized_img_source=self.style_source_blocks
        )

    def __call__(self, x, rng, significant_digits=-1):
        # the model operates on a batch of blocks to build an embedding used for the corresponding theta for each batch
        # the check below is activated outside of the initialize_model call, and treats each block as a separate sample.
        print(f'neural fractal style ae inputs: {x.shape}')
        # ensure that the x.shape[0] matches with the encoder block size (enc_blocks)
        patch_embs = self.encoder(x)
        reconstructions, mul, add = self.decoder(patch_embs, rng, bs=x.shape[0], significant_digits=significant_digits)
        # block level
        loss = jnp.sum(jnp.mean((reconstructions - x)**2, axis=0)) # TODO: maybe replace mean
        if self.config.l2_reg:
            loss = loss + jnp.sum(jnp.mean(jnp.power(mul, 2) + jnp.power(add, 2), axis=0))
        reconstructions = jnp.clip(reconstructions, 0, 1)

        psnr = compute_psnr(norm_two_to_eight_bit(x), norm_two_to_eight_bit(reconstructions))
        return loss, (reconstructions, psnr, mul, add)


if __name__ == '__main__':
    from functools import partial
    from pathlib import Path

    TEST_MODEL = "NeuralFractalAutoencoder"

    if TEST_MODEL == "NeuralFractalAutoencoder":
        from config.autoencoding.nfe_vanilla import get_config
        checkpoint_path = Path('checkpoints/test/')

        config = get_config()
        key = jax.random.PRNGKey(0)
        bs = 2
        x = x_target = jnp.ones((bs, config.data_res, config.data_res, config.data_width))

        params = NeuralFractalAutoencoder(config).init({'params': key}, x, key)['params']
        print(compute_number_parameters(params))

        loss, rec = NeuralFractalAutoencoder(config).apply({'params': params}, x, key)

        superrec = NeuralFractalSuperresAutoencoder(config).apply({'params': params}, x, key, superres_factor=2)
        assert superrec.shape[1] == 2 * x.shape[1]
