import pathlib

import jax

import jax.numpy as jnp
import numpy as np
from PIL import Image

from absl import app, flags
from ml_collections.config_flags import config_flags
from jax_src.utils.plot_utils import gridify_images

FLAGS = flags.FLAGS
flags.DEFINE_string("file", "media/michael_cat100.jpg", "path to image file to plot")
flags.DEFINE_integer("last_only", 1, "plot only fixed point or entire trajectory, 1: true, 0: false")
flags.DEFINE_integer("use_source", 1, "plot source image alongside decoded images 1: true, 0: false")
flags.DEFINE_integer("superres_factor", 1, "")

def load_img(fpath):
    img = Image.open(f'{fpath}')
    img = jnp.asarray(img)
    h, w, c = img.shape
    if h % 2 != 0: img = img[1:]
    if w % 2 != 0: img = img[:, 1:]
    if len(img.shape) <= 3: img = img[None]
    if img.max() > 1: img = img / 255.
    return img

def main(argv):
    fpath = f'decoded_traj_{FLAGS.superres_factor}.npy'
    sol_traj = np.load(f'artifacts/{fpath}')

    aux_info = {}
    if FLAGS.use_source == 1:
        fpath = f'{FLAGS.file}'
        img = load_img(fpath)
        aux_info = {'img': img}

    if FLAGS.last_only == 1: sol_traj = sol_traj[-1:]
    
    ndarr = gridify_images(jnp.asarray(sol_traj), padding=2, aux_info=aux_info, normalize=True)
    im = Image.fromarray(ndarr.copy())

    save_path = pathlib.Path('artifacts/')
    if not save_path.exists(): save_path.mkdir()

    with open(f'{save_path}/decoded_img_{FLAGS.superres_factor}.png', 'wb') as f:     
        im.save(f, format=None)


if __name__ == '__main__':
    app.run(main)