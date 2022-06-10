from collections import defaultdict
from jax import custom_jvp

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.special import logsumexp
import flax.linen as nn

from flax.linen.initializers import lecun_normal, zeros
default_kernel_init = lecun_normal()

def compute_psnr(imgs, pred_imgs, norm=False):
    "Compute PSNR between two images or batches of images"
    if norm: imgs, pred_imgs = imgs / 255., pred_imgs / 255.
    error = jnp.power(imgs - pred_imgs, 2).mean()
    psnr = 20. * jnp.log10(1.) - 10. * jnp.log10(error)
    return psnr

def compute_number_parameters(params):
    return sum(x.size for x in jax.tree_leaves(params))

def pad_channels(feature_map, width):
    d1, d2, d3, d4 = feature_map.shape
    empty = jnp.zeros(d1, width - d2, d3, d4)
    return jnp.concatenate()
    return empty

def get_width_settings(width, width_str=None):
    """Parses layer width string, defaults value to `width` """
    mapping = defaultdict(lambda: width)
    if width_str:
        width_str = width_str.split(',')
        for ss in width_str:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping

def parse_layer_string(model, s):
    if model in ('VDVAE', 'VDCVAE', 'LatAuxVDCVAE'):
        "Note: default 0 for downsampling rate and mixin"
        layers = []
        n_blocks = 0
        for ss in s.split(','):
            if 'x' in ss:
                res, num = ss.split('x')
                count = int(num)
                layers += [(int(res), None, 0) for _ in range(count)]
                n_blocks += count
            elif 'm' in ss:
                res, mixin = [int(a) for a in ss.split('m')]
                layers.append((res, 'm', mixin))
                n_blocks += 1
            elif 'd' in ss:
                res, down_rate = [int(a) for a in ss.split('d')]
                layers.append((res, 'd', down_rate))
                n_blocks += 1
            else:
                res = int(ss)
                layers.append((res, None, 0))
        return layers, n_blocks
    elif model == 'EVAE':
        layers, n_blocks = [], 0
        for ss in s.split(','):
            if 'x' in ss:
                res, num = ss.split('x')
                count = int(num)
                dh, dw = res.split('dh')[0], ss.split('dw')[0].split('dh')[1]
                rh, rw = res.split('rh')[0].split('dw')[1], ss.split('rw')[0].split('rh')[1]
                layers += [(int(dh), int(dw), int(rh), int(rw)) for _ in range(count)]
                n_blocks += count
        return layers, n_blocks
    elif model == 'FractalVAE':
        # 4h4wx3 for 3 blocks with 4x4 patches
        layers, n_blocks = [], 0
        for ss in s.split(','):
            if 'x' in ss:
                res, num = ss.split('x')
                count = int(num)
                h, w = res.split('h')[0], ss.split('w')[0].split('h')[1]
                layers += [(int(h), int(w)) for _ in range(count)]
                n_blocks += count
        return layers, n_blocks


def get_conv(out_dim,
            kernel_size, 
            stride, 
            init_weights_fn, 
            zero_weights,
            groups=1):
    init_weights_fn = zeros if zero_weights else init_weights_fn 
    conv = nn.Conv(out_dim, 
                   kernel_size=(kernel_size, kernel_size), 
                   strides=(stride, stride), 
                   padding="SAME",
                   kernel_init=init_weights_fn,
                   precision=None,
                   use_bias=False,
                   feature_group_count=groups)
    return conv

def get_3x3(out_dim,  
            init_weights_fn=default_kernel_init, 
            zero_weights=False,
            groups=1):
    return get_conv(out_dim, 3, 1, init_weights_fn, zero_weights, groups)

def get_1x1(out_dim, 
            init_weights_fn=default_kernel_init,
            zero_weights=False, 
            groups=1):
    return get_conv(out_dim, 1, 1, init_weights_fn, zero_weights, groups)

@custom_jvp
def log1mexp(x):
    """Accurate computation of log(1 - exp(-x)) for x > 0."""
    # Method from
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    return jnp.where(
        x > jnp.log(2), jnp.log1p(-jnp.exp(-x)), jnp.log(-jnp.expm1(-x)))
log1mexp.defjvps(lambda t, _, x: t / jnp.expm1(x))



def draw_gaussian_diag_samples(key, mu, logsigma):
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(key, mu.shape)
    return jnp.exp(logsigma) * eps + mu

def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    logsigma1 = jnp.clip(logsigma1, a_min=-7)
    logsigma2 = jnp.clip(logsigma2, a_min=-7)
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (jnp.exp(logsigma1) ** 2 + (mu1 - mu2) ** 2) / (jnp.exp(logsigma2) ** 2)

def log_sum_exp(x):
  """ Numerically stable log_sum_exp implementation that prevents overflow """
  # TF ordering
  axis = - 1
  m = jnp.max(x, axis=axis)
  m2 = jnp.max(x, axis=axis, keepdims=True)
  return m + jnp.log(jnp.sum(jnp.exp(x - m2), axis=axis))

def log_prob_from_logits(x):
  """ numerically stable log_softmax implementation that prevents overflow """
  # TF ordering
  axis = -1
  m = jnp.max(x, axis=axis, keepdims=True)
#   return x - m - jnp.log(jnp.sum(jnp.exp(x - m), axis=axis, keepdims=True))
  return jax.nn.logsumexp(jax.nn.log_softmax(x), axis=axis)


def mean_squared_error(img, output):
    output = output[..., :img.shape[-1]] # e.g. (64, 32, 32, 100) params vs image (64, 32, 32, 3)
    err = output - img
    return jnp.mean(jnp.square(err))

def logistic_preprocess(nn_out, n_channels=3, n_mix=10):
    """Processes output of a VAE decoder according to a dmol parametrization of p(x|z)
    Args:
        nn_out: output of the VAE decoder
        n_mix: number of logistics in the mixture
    Returns:
        means
        scales
        inv_scales
        logitweghts
    """
    *batch, h, w, _ = nn_out.shape

    assert nn_out.shape[-1] % 10 == 0
    k = n_mix
    logit_weights, nn_out = jnp.split(nn_out, [k], -1)
    m, s, t = jnp.moveaxis(
        jnp.reshape(nn_out, tuple(batch) + (h, w, n_channels, k, 3)), (-2, -1), (-4, 0))
    
    assert m.shape == tuple(batch) + (k, h, w, n_channels)
    inv_scales = jnp.maximum(nn.softplus(s), 1e-9)
    return m, jnp.tanh(t), inv_scales, jnp.moveaxis(logit_weights, -1, -3)

def logistic_mix_logpmf(nn_out, img, n_channels=3):
    m, t, inv_scales, logit_weights = logistic_preprocess(nn_out, n_mix=10, n_channels=n_channels)
    img = jnp.expand_dims(img, -4)  # Add mixture dimension
    if n_channels == 3:
        mean_red   = m[..., 0]
        mean_green = m[..., 1] + t[..., 0] * img[..., 0]
        mean_blue  = m[..., 2] + t[..., 1] * img[..., 0] + t[..., 2] * img[..., 1]
        means = jnp.stack((mean_red, mean_green, mean_blue), axis=-1)
    else:
        means = m[...,0]
        means = means[...,None]

    logprobs = jnp.sum(logistic_logpmf(img, means, inv_scales), -1)
    log_mix_coeffs = logit_weights - logsumexp(logit_weights, -3, keepdims=True)
    return jnp.sum(logsumexp(log_mix_coeffs + logprobs, -3), (-2, -1))

def logistic_logpmf(img, means, inv_scales):
    centered = img - means
    top    = -jnp.logaddexp(0,  (centered - 1 / 255) * inv_scales)
    bottom = -jnp.logaddexp(0, -(centered + 1 / 255) * inv_scales)
    mid = log1mexp(inv_scales / 127.5) + top + bottom
    return jnp.select([img == -1, img == 1], [bottom, top], mid)

def logistic_mix_sample(rng, nn_out, n_channels=3):
    m, t, inv_scales, logit_weights = logistic_preprocess(nn_out, n_mix=10, n_channels=n_channels)
    rng_mix, rng_logistic = jax.random.split(rng)
    mix_idx = jax.random.categorical(rng_mix, logit_weights, -3)
    def select_mix(arr):
        return jnp.squeeze(
            jnp.take_along_axis(
                arr, jnp.expand_dims(mix_idx, (-4, -1)), -4), -4)
    m, t, inv_scales = map(lambda x: jnp.moveaxis(select_mix(x), -1, 0),
                           (m, t, inv_scales))
    l = jax.random.logistic(rng_logistic, m.shape) / inv_scales
    if n_channels == 3:
        img_red   = m[0]                                     + l[0]
        img_green = m[1] + t[0] * img_red                    + l[1]
        img_blue  = m[2] + t[1] * img_red + t[2] * img_green + l[2]
        return jnp.stack([img_red, img_green, img_blue], -1)
    else:
        img   = m[0] + l[0]
        return img[...,None]     


def bernoulli_logpmf(nn_out, img, n_channels=None):
    p = nn.sigmoid(nn_out)
    p = jnp.clip(p, 1e-9, 1. - 1e-9)
    return jnp.sum(jnp.log(p) * img + jnp.log(1 - p) * (1 - img), axis=(-3, -2, -1))


def bernoulli_sample(rng, nn_out, n_channels=None):
    p = nn.sigmoid(nn_out)
    p = jnp.clip(p, 1e-9, 1. - 1e-9)
    return jax.random.bernoulli(rng, p)

LIKELIHOODS = {'dmol': logistic_mix_logpmf ,
               'bernoulli': bernoulli_logpmf}
SAMPLERS = {'dmol': logistic_mix_sample ,
               'bernoulli': bernoulli_sample}