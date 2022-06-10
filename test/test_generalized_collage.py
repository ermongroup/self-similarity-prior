from src.frax.modules import GeneralizedCollageOperator, GeneralizedBiasCollageOperator
import jax
import jax.numpy as jnp
from utils.plot_utils import gridify_images, norm_two_to_eight_bit
import os
from PIL import Image

key = jax.random.PRNGKey(0)
bs = 64
x = jax.random.normal(key, (bs, 32, 32, 3))

model = GeneralizedCollageOperator(
    input_channels=3,
    input_reso=32,
    rh=4,
    rw=4,
    quaddepth=2, # l1: 1 l2: 4 l3: 16, l4: 64
    dh=32,
    dw=32,
    hypernet_bottleneck=0.1,
    residual=False
)

params = model.init({'params': key}, x)['params']
params_shapes = jax.tree_map(lambda x: x.shape, params)
params_counts = [x.size for x in jax.tree_leaves(params)]
print(f'num params {params_counts}')
out = model.apply({'params': params}, x)

out = out.reshape(8, 8, *out.shape[1:]) # sqrtn, sqrtn, h, w, c
print(f'max {out.max()}')
out = jnp.clip(out, 0, 1)
out = norm_two_to_eight_bit(out)
ndarr = gridify_images(out, normalize=False)
pil_images = Image.fromarray(ndarr.copy())
with open(f"test_forward.png", "wb") as f:     
    pil_images.save(f, format=None)