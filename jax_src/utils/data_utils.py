import os
import pathlib
import random
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as t
import torchvision.transforms.functional as tfunctional
from PIL import Image
from tqdm import tqdm


class Subset(object):

    def __init__(self, dataset, size):
        assert size <= len(dataset)
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.dataset[index]
        
class DiscreteRandomRotation(object):
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return tfunctional.rotate(x, angle)


class Binarize(object):
    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)


def generate_dota_dataset(dpath, mode='train', dataset_sz=30000, crop_szs=(40, 40)):
    path = pathlib.Path(f"{dpath}/dota/{mode}/images")

    transforms = t.Compose([
        t.RandomCrop(size=crop_szs),
        DiscreteRandomRotation([0, 90, 180, 270]),
        t.RandomVerticalFlip(p=0.5),
        t.RandomHorizontalFlip(p=0.5)
    ])
    all_imgs = []
    count = 0
   
    while count < dataset_sz:
        source_imgs = os.listdir(path)
        n_sources = len(source_imgs)

        print(f'Generating first {n_sources} crops...')
        for img in tqdm(source_imgs):
            if count >= dataset_sz: break
            if img.endswith('png'):
                data = Image.open(f"{path}/{img}")
                data = np.array(data)

                data = torch.tensor(data)[None]
                # Some dota images have a single channel
                if len(data.shape) <= 3: 
                    data = data[...,None].repeat(1,1,1,3)
                # Some images have 4 channels

                data = data[...,:3]
                data = data.permute(0, 3, 1, 2)
                data = transforms(data)
                all_imgs.append(data)
                count = count + 1

    all_imgs = torch.cat(all_imgs)
    torch.save(all_imgs, f'{path}/preprocess_{mode}_{crop_szs[0]}x{crop_szs[1]}')

def parse_img_from_path(img_path, normalize=True):
    # pre-process the single image to use as the auxilliary image source
    img = Image.open(f'{img_path}')
    img_array = np.array(img)
    if normalize:
        img_array = img_array / 255.

    print(f'read image({img_path}): {img_array.shape} --> convert RGB: {img_array[..., :3].shape}')
    return img_array

def save_img(img_array_rgb, save_path):
    from PIL import Image
    im = Image.fromarray(img_array_rgb, 'RGB')
    im.save(save_path)
    print(f'saved to {save_path}')

def stack_images(img1, img2, axis=-1, save_dir='./data', filename='david_romanesco.jpg'):
    # stack two color images via the channel dim and save.
    breakpoint()
    img1_arr = parse_img_from_path(img1, normalize=False)
    img2_arr = parse_img_from_path(img2, normalize=False)
    new_img = np.concatenate([img1_arr, img2_arr], axis=axis)
    assert new_img.shape[-1] == img1_arr.shape[axis] + img2_arr.shape[axis], f"{new_img.shape[-1]} != {img1_arr.shape[axis] + img2_arr.shape[axis]}"

    if False: # no way to save images that are 3+ channels (maybe jpeg, but image libraries won't work)...
        save_path = os.path.join(save_dir, filename)

        six_c_img = parse_img_from_path(save_path)
        print(f'six_c_img {six_c_img.shape}')
    return new_img
