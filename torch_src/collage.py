import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_src.utils import img_to_patches, patches_to_img, vmapped_rotate

from einops import rearrange, repeat

import torchvision.transforms.functional as ttf
from timm.models.convmixer import ConvMixer


class CollageOperator2d(nn.Module):

    def __init__(self, res, rh, rw, dh=None, dw=None, use_augmentations=False):
        """Collage Operator for two-dimensional data. Given a fractal code, it outputs the corresponding fixed-point.

        Args:
            res (int): Spatial resolutions of input (and output) data.
            rh (int): Height of range (target) square patches.
            rw (int): Width of range (target) square patches.
            dh (int, optional): Height of range domain (source) patches. Defaults to `res`.
            dw (int, optional): Width of range domain (source) patches. Defaults to `res`.
            use_augmentations (bool, optional): Use augmentations of domain square patches at each decoding iteration. Defaults to `False`.
        """
        super().__init__()
        self.dh, self.dw = dh, dw
        if self.dh is None: self.dh = res
        if self.dw is None: self.dw = res

        # 5 refers to the 5 copies of domain patches generated with the current choice of augmentations:
        # 3 rotations (90, 180, 270), horizontal flips and vertical flips.
        self.n_aug_transforms = 5 if use_augmentations else 1

        # precompute useful quantities related to the partitioning scheme into patches, given
        # the desired `dh`, `dw`, `rh`, `rw`. 
        partition_info = self.collage_partition_info(res, self.dh, self.dw, rh, rw)
        self.n_dh, self.n_dw, self.n_rh, self.n_rw, self.h_factors, self.w_factors, self.n_domains, self.n_ranges = partition_info
        
        # At each step of the collage, all (source) domain patches are pooled down to the size of range (target) patches.
        # Notices how the pooling factors do not change if one decodes at higher resolutions, since both domain and range 
        # patch sizes are multiplied by the same integer.
        self.pool = nn.AvgPool3d(kernel_size=(1, self.h_factors, self.w_factors), stride=(1, self.h_factors, self.w_factors))


    def decode_step(self, z, weight, bias, superres_factor):
        """Single Collage Operator step. Performs the steps described in:
        https://arxiv.org/pdf/2204.07673.pdf (Def. 3.1, Figure 2).
        """

        # Given the current iterate `z`, we split it into `n_domains` domain patches.
        domains = img_to_patches(z, patch_h=self.dh * superres_factor, patch_w=self.dw * superres_factor)

        # Pool domains (pre augmentation) for compatibility with range patches.
        pooled_domains = self.pool(domains) 

        # If needed, produce additional candidate domain patches as augmentations of existing domains.
        if self.n_aug_transforms > 1:
            pooled_domains = self.generate_candidates(pooled_domains)

        pooled_domains = repeat(pooled_domains, 'b c d h w -> b c d r h w', r=self.n_ranges)

        # Apply the affine maps to domain patches
        range_domains = torch.einsum('bcdrhw, bcdr -> bcrhw', pooled_domains, weight)
        range_domains = range_domains + bias[:, :, :, None, None]

        # Reconstruct data by "composing" the output patches back together (collage!).
        z = patches_to_img(range_domains, self.n_rh, self.n_rw)
        return z

    def generate_candidates(self, domains):
        rotations = [vmapped_rotate(domains, angle=angle) for angle in (90, 180, 270)]
        hflips = ttf.hflip(domains)
        vflips = ttf.vflip(domains)
        domains = torch.cat([domains, *rotations, hflips, vflips], dim=2)
        return domains

    def forward(self, x, co_w, co_bias, decode_steps=20, superres_factor=1):
        B, C, H, W = x.shape
        # It does not matter which initial condition is chosen, so long as the dimensions match.
        # The fixed-point of a Collage Operator is uniquely determined* by the fractal code
        # *: and auxiliary learned patches, if any.
        z = torch.randn(B, C, H * superres_factor, W * superres_factor).to(x.device)
        for _ in range(decode_steps):
            z = self.decode_step(z, co_w, co_bias, superres_factor)
        return z

    def collage_partition_info(self, input_res, dh, dw, rh, rw):
        """
        Computes auxiliary information for the collage (number of source and target domains, and relative size factors)
        """
        height = width = input_res
        n_dh, n_dw = height // dh, width // dw
        n_domains = n_dh * n_dw

        # Adjust number of domain patches to include augmentations
        n_domains = n_domains + n_domains * self.n_aug_transforms # (3 rotations, hflip, vlip)

        h_factors, w_factors = dh // rh, dw // rw
        n_rh, n_rw = input_res // rh, input_res // rw    
        n_ranges = n_rh * n_rw
        return n_dh, n_dw, n_rh, n_rw, h_factors, w_factors, n_domains, n_ranges


class NeuralCollageOperator2d(nn.Module):
    def __init__(self, out_res, out_channels, rh, rw, dh=None, dw=None, net=None, use_augmentations=False):
        super().__init__()
        self.co = CollageOperator2d(out_res, rh, rw, dh, dw, use_augmentations)
        # In a Collage Operator, the affine map requires a single scalar weight 
        # for each pair of domain and range patches, and a single scalar bias for each range.
        # `net` learns to output these weights based on the objective.
        self.co_w_dim = self.co.n_domains * self.co.n_ranges * out_channels
        self.co_bias_dim = self.co.n_ranges * out_channels
        tot_out_dim = self.co_w_dim + self.co_bias_dim

        # Does not need to be a ConvMixer: for deep generative Neural Collages `net` can be e.g, a VDVAE.
        if net is None:
            net = ConvMixer(dim=32, depth=8, kernel_size=9, patch_size=7, num_classes=tot_out_dim)
        self.net = net

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, x, decode_steps=10, superres_factor=1, return_co_code=False):
        B, C, H, W = x.shape
        co_code = self.net(x) # B, C, co_w_dim + co_mix_dim + co_bias_dim
        co_w, co_bias = torch.split(co_code, [self.co_w_dim, self.co_bias_dim], dim=-1)

        co_w = co_w.view(B, C, self.co.n_domains, self.co.n_ranges)
        co_bias = co_bias.view(B, C, self.co.n_ranges)
        co_bias = self.tanh(co_bias)
        
        z = self.co(x, co_w, co_bias, decode_steps=decode_steps, superres_factor=superres_factor)
        
        if return_co_code: return z, co_w, co_bias
        else: return z
