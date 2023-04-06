import math
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

class EMLoss(nn.Module):
    """
    Entropy Model loss.
    Only need to optimize rate loss because entropy model only influence rate.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, stpm_out, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["y_bpp_loss"] = (torch.log(stpm_out["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels))
        out["z_bpp_loss"] = (torch.log(stpm_out["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels))
        out["loss"] = out["y_bpp_loss"] + out["z_bpp_loss"]

        return out


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class PixelwiseRateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target, lmbdamap):
        # lmbdamap: (B, 1, H, W)
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out['bpp_loss'] = sum(
            (-torch.log2(likelihoods).sum() / num_pixels)
            for likelihoods in output['likelihoods'].values()
        )

        mse = self.mse(output['x_hat'], target)
        lmbdamap = lmbdamap.expand_as(mse)
        out['mse_loss'] = torch.mean(lmbdamap * mse)
        out['loss'] = 255 ** 2 * out['mse_loss'] + out['bpp_loss']

        return out


class MovingAverage(object):
    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.queue = deque()
        self.Max_size = size

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.queue.append(val)
        if len(self.queue) > self.Max_size:
            self.queue.popleft()
        return 1.0 * sum(self.queue) / len(self.queue)


def quality2lambda(qmap):
    # return 0.0044 * torch.exp(1.9852 * qmap) # 自己拟合的曲线,见stem.xlsx stem_roi页拟合曲线
    # return 0.001 * torch.exp(3.4431 * qmap)
    return 0.002 * torch.exp(3.4409 * qmap)
    # return 1e-3 * torch.exp(4.382 * qmap) # iccv paper


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(aux_parameters))),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def save_checkpoint(state, filename="D:/"):
    torch.save(state, filename)