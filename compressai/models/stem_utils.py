import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.utils import save_image

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.layers import GDN
from compressai.models.priors import CompressionModel

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels)) # 为什么要先ln再求e次方，是为了更高的精度吗？


class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap):
        qmap = f.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma) + beta

        return out


class SFTResblk(nn.Module):
    def __init__(self, x_nc, prior_nc, ks=3):
        super().__init__()
        self.conv_0 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)

        self.norm_0 = SFT(x_nc, prior_nc, ks=ks)
        self.norm_1 = SFT(x_nc, prior_nc, ks=ks)

    def forward(self, x, qmap):
        dx = self.conv_0(self.actvn(self.norm_0(x, qmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, qmap)))
        out = x + dx

        return out

    def actvn(self, x):
        return f.leaky_relu(x, 2e-1)