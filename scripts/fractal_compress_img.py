import os 
import pathlib

os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from tqdm import tqdm
from absl import app, flags
import ml_collections
from ml_collections.config_flags import config_flags

from jax_src.compress.fractal import fractal_compress

config = ml_collections.ConfigDict()
config.time_factor = 10

FLAGS = flags.FLAGS
config_flags.DEFINE_config_dict('cfg', config)
flags.DEFINE_string("file", "/media/michael_cat100.jpg", "path to image file to plot")
flags.DEFINE_list("domain_patch_szs", [100,100], "")
flags.DEFINE_list("range_patch_szs", [2,2], "")
flags.DEFINE_integer("superres_factor", 1, "")

def main(argv):
    print(f'Running PIFS fractal compression on the target image.')
    jax.config.update('jax_platform_name', 'cpu')

    img_path = os.getcwd() + f'{FLAGS.file}'
    img = Image.open(f'{img_path}')
    img = jnp.asarray(img)[...,:3]

    h, w, c = img.shape
    print(f'Input image shape: {img.shape}')

    key = jax.random.PRNGKey(0)
    r_h, r_w = FLAGS.range_patch_szs
    d_h, d_w = FLAGS.domain_patch_szs

    D_chunks = (h // int(d_h), w // int(d_w))
    R_chunks = (h // int(r_h), w // int(r_w))

    # TODO: standardize here for various image types (8bit, normalized)
    if img.max() > 1: img = img / 255.
    kwargs = {"n_iters": 10, 
            "init_partition": "tiling", 
            "cmap_class": "affine", 
            "n_sources": 1, # n source patches you want for each target patch, 1 is 1 source patch for each range
            "D_chunks": D_chunks, 
            "R_chunks": R_chunks, 
            "use_rotation": True, 
            "use_flips": True,
            "noise_schedule": [i * FLAGS.cfg.time_factor for i in range(20 // 10)],
            "noise_type": None,
            "image_noise": None,
            "superres_factor": FLAGS.superres_factor,
            "inverse_problem": "None",
        }

    sol_traj, _ = fractal_compress(key, img[None], kwargs) 

    save_path = pathlib.Path(f'artifacts')
    save_path.mkdir(exist_ok=True)
    np.save(f'artifacts/decoded_traj_{FLAGS.superres_factor}.npy', sol_traj)


if __name__ == '__main__':
    app.run(main)
