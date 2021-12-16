import math

import torch
import torch.nn as nn
import torch.optim as optim

class EMLoss(nn.Module):
    """
    Entropy Model loss.
    ONLY NEED TO OPTIMIZE RATE BECAUSE Entropy Model ONLY INFLUENCE RATE.
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