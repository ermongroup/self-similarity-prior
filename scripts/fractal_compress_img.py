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
flags.DEFINE_string("file", "snow232.png", "path to image file to plot")
flags.DEFINE_string("outname", "output_compress", "output artifact name")
flags.DEFINE_list("domain_patch_szs", [230,230], "")
flags.DEFINE_list("range_patch_szs", [2,2], "")
flags.DEFINE_integer("superres_factor", 1, "")

def main(argv):
    jax.config.update('jax_platform_name', 'cpu')
    fpath = f'data/{FLAGS.file}'
    
    img = Image.open(f'{fpath}')
    img = jnp.asarray(img)[...,:3]

    h, w, c = img.shape
    print(img.shape)

    key = jax.random.PRNGKey(0)
    r_h, r_w = FLAGS.range_patch_szs
    d_h, d_w = FLAGS.domain_patch_szs

    D_chunks = (h // int(d_h), w // int(d_w))
    R_chunks = (h // int(r_h), w // int(r_w))

    # TODO: standardize here for various image types (8bit, normalized)
    if img.max() > 1: #
        img = img / 255.

    kwargs = {"n_iters": 10, 
            "init_partition": "tiling", 
            "cmap_class": "affine", 
            "n_sources": 1, # n source patches you want for each target patch, 1 is 1:1
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

    sol_traj, cmaps = fractal_compress(key, img[None], kwargs) # (# decode steps, 28, 28, 1)
    n_sources = kwargs["n_sources"]
    save_path = pathlib.Path(f'artifacts/extra')
    if not save_path.exists(): save_path.mkdir()

    np.save(f'{save_path}/{FLAGS.file}_{FLAGS.superres_factor}.npy', sol_traj)


if __name__ == '__main__':
    app.run(main)
