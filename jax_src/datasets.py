import os
import pathlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.datasets.vision import VisionDataset

from utils.data_utils import Binarize, DiscreteRandomRotation, Subset
from utils.train_utils import *

DATA_DIR = Path(__file__).parent.absolute()

class DotaAerialImages(VisionDataset):
    def __init__(self, root='.', train=True, transform=None, target_transform=None, num_crops=20):
        super(DotaAerialImages, self).__init__(root, transform=transform, target_transform=target_transform)
        if train:
            dpath = pathlib.Path(f"{root}/train/images")
            if "preprocess_train_40x40" in os.listdir(dpath):
                all_imgs = torch.load(f"{dpath}/preprocess_train_40x40")
                print(f'all dota images: {all_imgs.shape}')
            else:
                all_imgs = self.generate_crops(dpath, num_crops)

        else:
            dpath = pathlib.Path(f"{root}/val/images")
            if "preprocess_val_40x40" in os.listdir(dpath):
                all_imgs = torch.load(f"{dpath}/preprocess_val_40x40")
            else:
                all_imgs = self.generate_crops(dpath, num_crops)
        self.data = all_imgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.array(img.permute(1,2,0)))
        if self.transform is not None:
            img = self.transform(img)
        return img, img

    def generate_crops(self, dpath, num_crops):
        all_imgs, count = [], 0
        crop_fn = transforms.RandomCrop(size=40)
        for img in os.listdir(dpath):
            if count >= num_crops: break
            if img.endswith('png'):
                data = Image.open(f"{dpath}/{img}")
                data = np.array(data)
                data = torch.tensor(data)[None]
                # Some dota images have 1 or 4 channels
                if len(data.shape) <= 3: data = data[...,None].repeat(1,1,1,3)
                data = data[...,:3]
                data = data.permute(0, 3, 1, 2)
                data = crop_fn(data)
                all_imgs.append(data)
                count = count + 1
        all_imgs = torch.cat(all_imgs)
        return all_imgs


class SingleImageDataset(VisionDataset):
    def __init__(self, root='/', train=True, transform=None, target_transform=None, n_batches=1, n_devices=1, stacked=False):
        super(SingleImageDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        if n_devices == 1:
            # do this so that you don't add an extra dim based on config.bs, else the get_item will error out.
            print(f'n_batches set to {1} from {n_batches} since only {n_devices} device')
            n_batches = 1
        if stacked:
            raise NotImplementedError(f'currently does not support multi channel training -- results were bad, todo / revist!')

        if train:
            D = parse_img_from_path(root, normalize=False) # don't normalize b/c the .toTensor() will do it!
            X, Y = [], []
            for b_no in range(n_batches):
                X.append(D)
                Y.append(1) # 1 is a dummy label
            X = np.stack(X)
            Y = np.stack(Y)
        else:
            val_XY = parse_img_from_path(root, normalize=False)
            X = np.array(val_XY)[None]
            Y = np.array(1)[None]

        self.data = X
        self.targets = Y
        print(f'train: {train} data, targets {self.data.shape} {self.targets.shape}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, 'RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FFHQDataset(VisionDataset):
    def __init__(self, root='/ssd003/projects/ffhq/', resolution=256, train=True, transform=None, target_transform=None, use_npy=False):
        super(FFHQDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.image_info = {
            128: {'path': 'thumbnails128x128'},
            256: {'path': 'ffhq-256.npy', 'shift': -112.8666757481, 'scale': 1. / 69.84780273},
            1024: {'path': 'images1024x1024', 'shift': -0.4387, 'scale': 1.0 / 0.2743, 'shift_loss': -0.5, 'scale_loss': 2.0}
        }
        self.root_dir = os.path.join(self.root, f"{self.image_info[resolution]['path']}")
        if resolution == 256 or use_npy:
            train_ds, val_ds, _ = self.ffhq_load_npy(self.root_dir)
            if train:
                self.data = train_ds
            else:
                self.data = val_ds
        else:
            # note: there are train val splits of the 128 and 1024 versions too
            split_idx = 65000 if resolution == '1024' else 400000
            imgs = os.listdir(self.root_dir)
            all_imgs = []
            if train:
                imgs = imgs[:split_idx]
                curr_images = [os.path.join(self.root_dir, file) for file in imgs if file.endswith('.png')]
                all_imgs += curr_images
            else:
                imgs = imgs[split_idx:]
                curr_images = [os.path.join(self.root_dir, file) for file in imgs if file.endswith('.png')]
                all_imgs += curr_images
            self.data = all_imgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, img

    def ffhq_load_npy(self, npy_path):
        trX = np.load(npy_path, mmap_mode='r')
        print(f'loaded npy into disk {trX.shape}')
        np.random.seed(5)
        tr_va_split_indices = np.random.permutation(trX.shape[0])
        train = trX[tr_va_split_indices[:-7000]]
        valid = trX[tr_va_split_indices[-7000:]]
        return train, valid, valid



def get_binarized_mnist(_script_dir):
    transform = transforms.Compose([transforms.ToTensor(), Binarize()])
    train_set = MNIST(_script_dir, train=True, transform=transform, download=True)
    test_set = MNIST(_script_dir, train=False, transform=transform, download=True)
    norm_fn = lambda x: x
    return train_set, None, test_set, norm_fn


def get_mnist(_script_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST(_script_dir, train=True, transform=transform, download=True)
    test_set = MNIST(_script_dir, train=False, transform=test_transform, download=True)
    norm_fn = lambda x: x
    return train_set, None, test_set, norm_fn


def get_cifar10(_script_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = CIFAR10(_script_dir, train=True, transform=transform, download=True)
    test_set = CIFAR10(_script_dir, train=False, transform=test_transform, download=True)
    norm_fn = lambda x: x
    return train_set, None, test_set, norm_fn

def get_ffhq(_script_dir, resolution='256'):
    image_info = {
            128: {'path': 'thumbnails128x128', 'shift': 0, 'scale': 1.},
            256: {'path': 'ffhq-256.npy', 'shift': -112.8666757481, 'scale': 1. / 69.84780273, 'shift_loss': -127.5, 'scale_loss': 1. / 127.5},
            1024: {'path': 'images1024x1024', 'shift': -0.4387, 'scale': 1.0 / 0.2743, 'shift_loss': -0.5, 'scale_loss': 2.0}
    }
    transform, test_transform  = None, None
    if resolution == 1024:
        transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = FFHQDataset(_script_dir, resolution=resolution, train=True, transform=transform)
    test_set = FFHQDataset(_script_dir, resolution=resolution, train=False, transform=test_transform)

    shift, scale = image_info[resolution]['shift'], image_info[resolution]['scale']
    shift_loss, scale_loss = image_info[resolution]['shift_loss'], image_info[resolution]['scale_loss']
    do_low_bit = resolution == 256

    def norm_fn(x, use_cuda=False):
        nonlocal shift, scale, shift_loss, scale_loss, do_low_bit
        if use_cuda:
            # notice this is slow and ooms sometimes, don't use for now
            shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
            scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
            shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
            scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)
            x_norm = x.cuda(non_blocking=True).float()
            x_lo_bit = x_norm.clone()
            x_norm.add_(shift).mul_(scale)
            if do_low_bit:
                # 5-bit precision is calculated using the following lines, with num_bits=5 and for an x in [0, 1]
                # x = torch.floor(x * 255 / 2 ** (8 - num_bits))
                # x /= (2 ** num_bits - 1)
                x_lo_bit.mul_(1. / 8.).floor_().mul_(8.)
            x_lo_bit.add_(shift_loss).mul_(scale_loss)
            return x_norm.cpu(), x_lo_bit.cpu()
        else:
            x_norm = (x + shift) * scale
            x_lo_bit = x
            if do_low_bit:
                x_lo_bit = (x * (1. / 8.)).floor() * 8.
            return x_norm, (x_lo_bit + shift_loss) * scale_loss

        # return x_norm.cpu() if use_cuda else x_norm

    return train_set, None, test_set, norm_fn

def get_dota(_script_dir, max_count_imgs=1000):

    transform = transforms.Compose(
        [
        transforms.ToTensor(),
        DiscreteRandomRotation([0, 90, 180, 270]),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_set = DotaAerialImages(_script_dir, train=True, num_crops=max_count_imgs, transform=transform)
    test_set = DotaAerialImages(_script_dir, train=False, num_crops=max_count_imgs, transform=test_transform)
    norm_fn = lambda x: x
    return train_set, None, test_set, norm_fn


def get_single_source(img_dir, n_batches, n_devices, stack_sources):
    transform = transforms.Compose(
        [
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(), # Normalize to [0, 1]
        ]
    )
    train_set = SingleImageDataset(img_dir, train=True, transform=transform, n_batches=n_batches, n_devices=n_devices, stacked=stack_sources)
    test_set = SingleImageDataset(img_dir, train=False, transform=test_transform, n_batches=n_batches, n_devices=n_devices,  stacked=stack_sources)

    norm_fn = lambda x: x

    return train_set, None, test_set, norm_fn

def get_dataset(batch_size, 
                test_batch_size, 
                dataset, 
                data_dir=DATA_DIR, 
                val_pcent=0.1,
                n_devices=1,
                subset_size=None,
                stack_sources=''):

    if type(stack_sources) == str and 'ffhq' not in dataset:
        script_dir = os.path.join(data_dir, dataset)
        if script_dir != DATA_DIR and dataset != "single_source": 
            _script_dir = script_dir
            print(f'loading data from {_script_dir}')
    elif 'ffhq' in dataset:
        _script_dir = data_dir
    else:
        assert len(data_dir) == 2, f'expected data_dir to be list of 2 img paths, got {len(data_dir)} paths'
        _script_dir = data_dir
    print(f'loading data from {_script_dir}')

    if dataset == "mnist": train_set, val_set, test_set, norm_fn =  get_mnist(_script_dir)
    elif dataset == "binarized_mnist": train_set, val_set, test_set, norm_fn =  get_binarized_mnist(_script_dir)
    elif dataset == "cifar10": train_set, val_set, test_set, norm_fn =  get_cifar10(_script_dir)
    elif dataset == "dota": train_set, val_set, test_set, norm_fn =  get_dota(_script_dir)
    elif dataset == "single_source": train_set, val_set, test_set, norm_fn = get_single_source(data_dir, 1, n_devices, False)
    elif dataset == "single_source_stacked": train_set, val_set, test_set, norm_fn = get_single_source(_script_dir, 1, n_devices, True)
    elif dataset == 'ffhq128': train_set, val_set, test_set, norm_fn = get_ffhq(_script_dir, 128)
    elif dataset == 'ffhq256': train_set, val_set, test_set, norm_fn = get_ffhq(_script_dir, 256)
    elif dataset == 'ffhq1024': train_set, val_set, test_set, norm_fn = get_ffhq(_script_dir, 1024)

    def preprocess_fn(config, batch_data):
        # only used for pytorch loaders
        x_in, _ = batch_data # no labels
        if 'ffhq' not in config.dataset:
            x_out = norm_fn(x_in)
        else:
            x_in, x_out = norm_fn(x_in)
            # x_in is low 5 bit precision as vdvae: https://github.com/openai/vdvae/blob/ea35b490313bc33e7f8ac63dd8132f3cc1a729b4/data.py#L85
            # normalized version should be trained on
        if config.dataset != 'ffhq256': # TODO: Check 128 too. Once we get 1024 npy loaded, probably no need again
            x_in, x_out = x_in.permute(0, 2, 3, 1), x_out.permute(0, 2, 3, 1)
        if config.likelihood_func == "dmol": # and config.dataset != 'ffhq256':
            # https://github.com/openai/vdvae/blob/ea35b490313bc33e7f8ac63dd8132f3cc1a729b4/vae_helpers.py#L52
            x_out = 2 * x_out - 1 # rescale from [0, 1] to [-1, 1]

        x_in, x_out = jnp.asarray(x_in), jnp.asarray(x_out)
        assert x_in.shape[-1] in (1, 3, 4), f'x_in {x_in.shape}: {x_in.mean()} | x_out {x_out.shape}: {x_out.mean()}'
        x_in = jnp.reshape(x_in, (config.n_devices, -1, *x_in.shape[1:]))
        x_out = jnp.reshape(x_out, (config.n_devices, -1, *x_out.shape[1:]))            

        if dataset == "statically_binarized_mnist":
            x_out = jnp.where(x_out > 0.5, 1, 0)
        return x_in, x_out

    if subset_size: # for debugging
        train_set = Subset(train_set, subset_size)
        val_set = Subset(test_set, subset_size)
    elif val_set is None and dataset != "single_source":
        nval = int(len(train_set) * val_pcent)
        train_set, val_set = torch.utils.data.random_split(train_set, [len(train_set) - nval, nval])
        print(f'Generating validation set dynamically from training data: val data with {len(val_set)} samples')
        
    elif val_set is None and dataset == "single_source": # and len(train_set) > 1
        val_set = train_set
        print(f'set val set to be train set b/c single_source insufficient for {val_pcent}%: batch of {len(train_set)}')

    if n_devices > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=n_devices, shuffle=True, rank=0)
        val_sampler = DistributedSampler(val_set, num_replicas=n_devices, rank=0)
        test_sampler = DistributedSampler(test_set, num_replicas=n_devices, rank=0)

        if dataset != "single_source":
            train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, drop_last=True, sampler=train_sampler)
            val_loader = DataLoader(val_set, batch_size=test_batch_size, sampler=val_sampler, drop_last=True)
            test_loader = DataLoader(test_set, batch_size=test_batch_size, sampler=test_sampler, drop_last=True)
        else:
            train_loader = DataLoader(train_set, batch_size=1, pin_memory=True, drop_last=True, sampler=train_sampler)
            val_loader = DataLoader(val_set, batch_size=1, drop_last=True, sampler=val_sampler)
            test_loader = DataLoader(test_set, batch_size=1, drop_last=True, sampler=test_sampler)

    else:
        if dataset != "single_source":
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=test_batch_size)
            test_loader = DataLoader(test_set, batch_size=test_batch_size)
        else: # if it's single source, just keep bs small since it's just one image.
            train_loader = DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=1)
            test_loader = DataLoader(test_set, batch_size=1)

    return train_loader, val_loader, test_loader, preprocess_fn

