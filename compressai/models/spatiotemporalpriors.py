import argparse
import math
import random
import sys
import os
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


from compressai.entropy_models import GaussianConditional
from compressai.models.priors import CompressionModel
from compressai.layers import GDN, MaskedConv2d
from compressai.ans import BufferedRansEncoder, RansDecoder

from .utils import update_registered_buffers

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

#Todo: use inherit to implement ablation models for a better implementation
class SpatioTemporalPriorModelWithoutSPMTPM(CompressionModel):
    """
    One of Ablation Experiments for SpatioTemporalPriorModel that only contains hyperprior but not temporal prior and spatial prior.
    The input of Hyperprior Encoder contains the latent y of the current frame and the latent y of last frame.
    The hyperprior information is used to estimate the gaussian probability model parameters, thus has a GaussianConditional model.
    Model inherits from CompressionModel which contains an entropy_bottleneck used for compress hyperprior latent z.
    """

    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)
        self.HE = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=256, kernel_size=3, padding=3 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2),
        )
        self.HD = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5,
                               padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2,
                               stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels * 2, kernel_size=3, padding=3 // 2, stride=1),
        )
        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=768, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=768, out_channels=576, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=576, out_channels=in_channels * 2, kernel_size=1, padding=1 // 2, stride=1),
        )

        self.gaussian_conditional = GaussianConditional(None)


    def forward(self, y_cur, y_conditioned):
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)

        gaussian_params = self.EPM(torch.cat([hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat)

        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, y_cur, y_conditioned):
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)

        gaussian_params = self.EPM(torch.cat([hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y_cur, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


    def decompress(self, strings, shape, y_conditioned):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        gaussian_params = self.EPM(torch.cat([hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return y_hat


    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


class SpatioTemporalPriorModelWithoutSPM(CompressionModel):
    """
    One of Ablation Experiments for SpatioTemporalPriorModel.
    Temporal Prior Model contains hyperprior and temporal prior but not spatial prior.
    All prior information is used to estimate gaussian probability model parameters, thus has a GaussianConditional model.
    Model inherits from CompressionModel which contains an entropy_bottleneck used for compress hyperprior latent z.
    """

    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)
        self.TPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=320, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=320, out_channels=in_channels * 2, kernel_size=5, padding=5 // 2, stride=1),
        )
        self.HE = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=256, kernel_size=3, padding=3 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2),
        )
        self.HD = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5,
                               padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2,
                               stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels * 2, kernel_size=3, padding=3 // 2, stride=1),
        )
        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2 * 2, out_channels=768, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=768, out_channels=576, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=576, out_channels=in_channels * 2, kernel_size=1, padding=1 // 2, stride=1),
        )

        self.gaussian_conditional = GaussianConditional(None)


    def forward(self, y_cur, y_conditioned):
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)


        temporalprior_params = self.TPM(y_conditioned)


        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat)


        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, y_cur, y_conditioned):
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y_cur, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


    def decompress(self, strings, shape, y_conditioned):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)
        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return y_hat


    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


class SpatioTemporalPriorModelWithoutTPM(CompressionModel):
    """
    One of Ablation Experiments for SpatioTemporalPriorModel that contains hyperprior and spatial prior but not temporal prior.
    All prior information is used to estimate gaussian probability model parameters, thus has a GaussianConditional model.
    Model inherits from CompressionModel which contains an entropy_bottleneck used for compress hyperprior latent z.
    """

    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        '''
        'entropy_bottleneck_channels' means the channels of "entropy_bottleneck". Here we use "entropy_bottleneck" to
            compress Hyperprior latent. Thus it's output channels for "HE" and input channels for "HD".
        'in_channels' means the input image latent channels.
        '''
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels) # todo: add entropy bottleneck channels
        self.HE = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=256, kernel_size=3, padding=3 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=entropy_bottleneck_channels, kernel_size=5, padding=5 // 2, stride=2),
        )
        self.HD = nn.Sequential(
            nn.ConvTranspose2d(in_channels=entropy_bottleneck_channels, out_channels=256, kernel_size=5,
                               padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2,
                               stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels * 2, kernel_size=3, padding=3 // 2, stride=1),
        )
        self.context_prediction = MaskedConv2d(in_channels, in_channels * 2, kernel_size=5, padding=2, stride=1)

        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2 * 2, out_channels=768, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=768, out_channels=576, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=576, out_channels=in_channels * 2, kernel_size=1, padding=1 // 2, stride=1),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.in_channels = in_channels


    def forward(self, y_cur, y_conditioned):
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y_cur, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)

        gaussian_params = self.EPM(torch.cat([hyperprior_params, ctx_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        _, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat)

        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, y_cur, y_conditioned):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        z = self.HE(torch.cat([y_cur, y_conditioned], 1))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2  # 2

        # y_height = z_hat.size(2) * s  # [B C H W]
        # y_width = z_hat.size(3) * s
        y_height = y_cur.size(2)  # H
        y_width = y_cur.size(3)  # W

        y_hat = F.pad(y_cur, (padding, padding, padding, padding))


        y_strings = []
        for i in range(y_cur.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                hyperprior_params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, hyperprior_params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        start = time.time()
        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask  # MaskedConv2d (OUT_C,IN_C,Kernel_H,Kernerl_W)
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]  # y_crop is the ctx of the pixel (h,w) in original picture
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )  # (B,OUT_C,1,1)

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                hp = hyperprior_params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.EPM(torch.cat((hp, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)  # (B,C,H=1,W=1)->(B,C)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)  # (B, M*2)->(B, M), (B, M)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]  # y_crop is the ctx of(h,w), y_crop[:, :, 1, 1] is the pixel (h,w)
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        dec_time = time.time() - start
        print(f"compress: {dec_time:.4f}s")

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string


    def decompress(self, strings, shape, y_conditioned):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.in_channels, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                hyperprior_params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        return y_hat


    def _decompress_ar(
        self, y_string, y_hat, hyperprior_params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        start = time.time()
        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                hp = hyperprior_params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.EPM(torch.cat((hp, ctx_p), dim=1))  # (B, OUT_C+1, 1, 1)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

        dec_time = time.time() - start
        print(f"decompress: {dec_time:.4f}s")



    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


class SpatioTemporalPriorModel(CompressionModel):
    """
    Note that channels is fewer than the original paper.
    SpatioTemporalPriorModel reproduction version that contains hyperprior, spatial prior and temporal prior.
    All prior information is used to estimate gaussian probability model parameters, thus has a GaussianConditional model.
    Model inherits from CompressionModel which contains an entropy_bottleneck used for compress hyperprior latent z.
    """

    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        '''
        'entropy_bottleneck_channels' means the channels of "entropy_bottleneck". Here we use "entropy_bottleneck" to
            compress Hyperprior latent. Thus it's output channels for "HE" and input channels for "HD".
        'in_channels' means the input image latent channels.
        '''
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)
        self.TPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=320, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=320, out_channels=in_channels * 2, kernel_size=5, padding=5 // 2, stride=1),
        )
        self.HE = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=256, kernel_size=3, padding=3 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=entropy_bottleneck_channels, kernel_size=5, padding=5 // 2, stride=2),
        )
        self.HD = nn.Sequential(
            nn.ConvTranspose2d(in_channels=entropy_bottleneck_channels, out_channels=256, kernel_size=5,
                               padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2,
                               stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels * 2, kernel_size=3, padding=3 // 2, stride=1),
        )
        self.context_prediction = MaskedConv2d(in_channels, in_channels * 2, kernel_size=5, padding=2, stride=1)

        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2 * 3, out_channels=768, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=768, out_channels=576, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=576, out_channels=in_channels * 2, kernel_size=1, padding=1 // 2, stride=1),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.in_channels = in_channels


    def forward(self, y_cur, y_conditioned):
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)


        temporalprior_params = self.TPM(y_conditioned)


        y_hat = self.gaussian_conditional.quantize(
            y_cur, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)


        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params, ctx_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        _, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat)


        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, y_cur, y_conditioned):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        z = self.HE(torch.cat([y_cur, y_conditioned], 1))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)



        temporalprior_params = self.TPM(y_conditioned)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2  # 2


        # y_height = z_hat.size(2) * s  # [B C H W]
        # y_width = z_hat.size(3) * s
        y_height = y_cur.size(2)  # H
        y_width = y_cur.size(3)  # W

        y_hat = F.pad(y_cur, (padding, padding, padding, padding))


        y_strings = []
        for i in range(y_cur.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                hyperprior_params[i: i + 1],
                temporalprior_params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, hyperprior_params, temporalprior_params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # start = time.perf_counter()
        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask  # MaskedConv2d (OUT_C,IN_C,Kernel_H,Kernerl_W)
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]  # y_crop is the ctx of(h,w), y_crop[:, :, 1, 1] is the pixel (h,w)
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )  # (B,OUT_C,1,1)

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                hp = hyperprior_params[:, :, h: h + 1, w: w + 1]
                tp = temporalprior_params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.EPM(torch.cat((tp, hp, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)  # (B,C,H=1,W=1)->(B,C)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)  # (B, M*2)->(B, M), (B, M)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]  # y_crop is the ctx of(h,w), y_crop[:, :, 1, 1] is the pixel (h,w)
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()

        return string


    def decompress(self, strings, shape, y_conditioned):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)


        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.in_channels, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                hyperprior_params[i: i + 1],
                temporalprior_params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        return y_hat


    def _decompress_ar(
        self, y_string, y_hat, hyperprior_params, temporalprior_params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                hp = hyperprior_params[:, :, h: h + 1, w: w + 1]
                tp = temporalprior_params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.EPM(torch.cat((tp, hp, ctx_p), dim=1))  # (B, OUT_C+1, 1, 1)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv




    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


class SpatioTemporalPriorModel_Res(CompressionModel):
    """
    Note that channels is fewer than the original paper.
    SpatioTemporalPriorModel reproduction version that contains hyperprior, spatial prior and temporal prior.
    All prior information is used to estimate gaussian probability model parameters for the residual between current y
    and condition y, thus has a GaussianConditional model.
    Model inherits from CompressionModel which contains an entropy_bottleneck used for compress hyperprior latent z.
    """

    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        '''
        'entropy_bottleneck_channels' means the channels of "entropy_bottleneck". Here we use "entropy_bottleneck" to
            compress Hyperprior latent. Thus it's output channels for "HE" and input channels for "HD".
        'in_channels' means the input image latent channels.
        '''
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)
        self.TPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=320, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=320, out_channels=in_channels * 2, kernel_size=5, padding=5 // 2, stride=1),
        )
        self.HE = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=256, kernel_size=3, padding=3 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=entropy_bottleneck_channels, kernel_size=5, padding=5 // 2, stride=2),
        )
        self.HD = nn.Sequential(
            nn.ConvTranspose2d(in_channels=entropy_bottleneck_channels, out_channels=256, kernel_size=5,
                               padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2,
                               stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels * 2, kernel_size=3, padding=3 // 2, stride=1),
        )
        self.context_prediction = MaskedConv2d(in_channels, in_channels * 2, kernel_size=5, padding=2, stride=1)

        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2 * 3, out_channels=768, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=768, out_channels=576, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=576, out_channels=in_channels * 2, kernel_size=1, padding=1 // 2, stride=1),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.in_channels = in_channels


    def forward(self, y_cur, y_conditioned):
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)

        res_cur = y_cur - y_conditioned
        res_hat = self.gaussian_conditional.quantize(
            res_cur, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(res_hat)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params, ctx_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)


        _, y_likelihoods = self.gaussian_conditional(res_cur, scales_hat, means=means_hat)
        y_hat = res_hat + y_conditioned

        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, y_cur, y_conditioned):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        z = self.HE(torch.cat([y_cur, y_conditioned], 1))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)


        temporalprior_params = self.TPM(y_conditioned)

        res_cur = y_cur - y_conditioned

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2  # 2

        # y_height = z_hat.size(2) * s  # [B C H W]
        # y_width = z_hat.size(3) * s
        res_height = res_cur.size(2)  # H
        res_width = res_cur.size(3)  # W

        res_hat = F.pad(res_cur, (padding, padding, padding, padding))


        res_strings = []
        for i in range(res_cur.size(0)):
            string = self._compress_ar(
                res_hat[i: i + 1],
                hyperprior_params[i: i + 1],
                temporalprior_params[i: i + 1],
                res_height,
                res_width,
                kernel_size,
                padding,
            )
            res_strings.append(string)

        return {"strings": [res_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, res_hat, hyperprior_params, temporalprior_params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask  # MaskedConv2d (OUT_C,IN_C,Kernel_H,Kernerl_W)
        for h in range(height):
            for w in range(width):
                res_crop = res_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    res_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )  # (B,OUT_C,1,1)

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                hp = hyperprior_params[:, :, h: h + 1, w: w + 1]
                tp = temporalprior_params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.EPM(torch.cat((tp, hp, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)  # (B,C,H=1,W=1)->(B,C)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)  # (B, M*2)->(B, M), (B, M)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                res_crop = res_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(res_crop, "symbols", means_hat)
                res_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())


        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()

        return string


    def decompress(self, strings, shape, y_conditioned):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder


        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)


        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        res_hat = torch.zeros(
            (z_hat.size(0), self.in_channels, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, res_string in enumerate(strings[0]):
            self._decompress_ar(
                res_string,
                res_hat[i : i + 1],
                hyperprior_params[i: i + 1],
                temporalprior_params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        res_hat = F.pad(res_hat, (-padding, -padding, -padding, -padding))
        y_hat = res_hat + y_conditioned

        return {"y_hat": y_hat}


    def _decompress_ar(
        self, res_string, res_hat, hyperprior_params, temporalprior_params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(res_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                res_crop = res_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    res_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                hp = hyperprior_params[:, :, h: h + 1, w: w + 1]
                tp = temporalprior_params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.EPM(torch.cat((tp, hp, ctx_p), dim=1))  # (B, OUT_C+1, 1, 1)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                res_hat[:, :, hp : hp + 1, wp : wp + 1] = rv



    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated