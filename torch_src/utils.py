import torch
import torchvision.transforms.functional as ttf

import functorch 

def img_to_patches(im, patch_h, patch_w):
    "B, C, H, W -> B, C, D, h_patch, w_patch"
    bs, c, h, w = im.shape
    im = im.unfold(-1, patch_h, patch_w).unfold(2, patch_h, patch_w)
    im = im.permute(0, 1, 2, 3, 5, 4)
    im = im.contiguous().view(bs, c, -1, patch_h, patch_w)
    return im

def patches_to_img(patches, num_patch_h, num_patch_w):
    "B, C, D, h_patch, w_patch -> B, C, H, W"
    bs, c, d, h, w = patches.shape
    patches = patches.view(bs, c, num_patch_h, num_patch_w, h, w)
    # fold patches
    patches = torch.cat([patches[..., k, :, :] for k in range(num_patch_w)], dim=-1)
    x = torch.cat([patches[..., k, :, :] for k in range(num_patch_h)], dim=-2)    
    return x

def vmapped_rotate(x, angle):
    "B, C, D, H, W -> B, C, D, H, W"
    rotate_ = functorch.vmap(ttf.rotate, in_dims=2, out_dims=2)
    return rotate_(x, angle=angle)