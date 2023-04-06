import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.models.priors import CompressionModel
from compressai.layers import GDN, MaskedConv2d
from compressai.ans import BufferedRansEncoder, RansDecoder

from .utils import conv, deconv, update_registered_buffers
from .stem_utils import get_scale_table, SFT, SFTResblk



#--------------------------------------------------------------------------------------------------#
# baseline for roi-coding and continuous rate model
class stem_baseline(CompressionModel):
    '''

    '''
    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)

        self.PEncoder = nn.Sequential(
            conv(3, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, in_channels),
        )
        self.PDecoder = nn.Sequential(
            deconv(in_channels, 128),
            GDN(128, inverse=True),
            deconv(128, 128),
            GDN(128, inverse=True),
            deconv(128, 128),
            GDN(128, inverse=True),
            deconv(128, 3),
        )
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


    def forward(self, x_cur, x_conditioned):
        y_cur = self.PEncoder(x_cur)
        y_conditioned = self.PEncoder(x_conditioned)

        z = self.HE(torch.cat([y_cur, y_conditioned], 1))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)


        temporalprior_params = self.TPM(y_conditioned)


        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1)) # 融合时域先验以及超先验
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat) # y_hat是加噪声or四舍五入量化

        x_hat = self.PDecoder(y_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x_cur, x_conditioned):
        y_cur = self.PEncoder(x_cur)
        y_conditioned = self.PEncoder(x_conditioned)
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))


        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y_cur, indexes, means=means_hat)  # y_string 是 list, 且只包含一个元素
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


    def decompress(self, strings, shape, x_conditioned):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        y_conditioned = self.PEncoder(x_conditioned)

        temporalprior_params = self.TPM(y_conditioned)
        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        x_hat = self.PDecoder(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat, "entropy_params": {"scales_hat": scales_hat, "means_hat": means_hat}}


    def getY(self, x, isEval=False):
        if isEval:
            h, w = x.size(2), x.size(3)
            p = 64  # maximum 6 strides of 2
            new_h = (h + p - 1) // p * p  # padding为64的倍数
            new_w = (w + p - 1) // p * p
            padding_left = (new_w - w) // 2
            padding_right = new_w - w - padding_left
            padding_top = (new_h - h) // 2
            padding_bottom = new_h - h - padding_top
            x = F.pad(
                x,
                (padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
        return self.PEncoder(x)


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

# a condition encoder is different from PEncoder
# 和上面的性能差不多
class stem_baselinev2(CompressionModel):
    '''

    '''
    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)

        self.PEncoder = nn.Sequential(
            conv(3, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, in_channels),
        )
        self.ConditionEncoder = nn.Sequential(
            conv(3, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, in_channels),
        )
        self.PDecoder = nn.Sequential(
            deconv(in_channels, 128),
            GDN(128, inverse=True),
            deconv(128, 128),
            GDN(128, inverse=True),
            deconv(128, 128),
            GDN(128, inverse=True),
            deconv(128, 3),
        )
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


    def forward(self, x_cur, x_conditioned):
        y_cur = self.PEncoder(x_cur)
        y_conditioned = self.ConditionEncoder(x_conditioned)

        z = self.HE(torch.cat([y_cur, y_conditioned], 1))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)


        temporalprior_params = self.TPM(y_conditioned)


        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1)) # 融合时域先验以及超先验
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat) # y_hat是加噪声or四舍五入量化

        x_hat = self.PDecoder(y_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x_cur, x_conditioned):
        y_cur = self.PEncoder(x_cur)
        y_conditioned = self.ConditionEncoder(x_conditioned)
        z = self.HE(torch.cat([y_cur, y_conditioned], 1))


        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y_cur, indexes, means=means_hat)  # y_string 是 list, 且只包含一个元素
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


    def decompress(self, strings, shape, x_conditioned):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        y_conditioned = self.ConditionEncoder(x_conditioned)

        temporalprior_params = self.TPM(y_conditioned)
        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        x_hat = self.PDecoder(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat, "entropy_params": {"scales_hat": scales_hat, "means_hat": means_hat}}


    def getY(self, x, isEval=False):
        if isEval:
            h, w = x.size(2), x.size(3)
            p = 64  # maximum 6 strides of 2
            new_h = (h + p - 1) // p * p  # padding为64的倍数
            new_w = (w + p - 1) // p * p
            padding_left = (new_w - w) // 2
            padding_right = new_w - w - padding_left
            padding_top = (new_h - h) // 2
            padding_bottom = new_h - h - padding_top
            x = F.pad(
                x,
                (padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
        return self.PEncoder(x)


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


#--------------------------------------------------------------------------------------------------#
# stem experiments for roi-coding and continuous rate model
class stem_roi(CompressionModel):
    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)

        #--------------------------------------------------------
        # g_a
        self.ga1 = nn.Sequential(
            conv(3, 128),
            GDN(128),
        )
        self.ga1_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.ga2 = nn.Sequential(
            conv(128, 128),
            GDN(128),
        )
        self.ga2_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.ga3 = nn.Sequential(
            conv(128, 128),
            GDN(128),
        )
        self.ga3_SFT = SFT(x_nc=128, prior_nc=128, ks=3)

        self.ga4 = conv(128, in_channels)
        self.ga4_SFTResB1 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        self.ga4_SFTResB2 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)

        self.qmap_feature_ga1 = nn.Sequential(
            conv(4, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 160, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(160, 128, 3, 1)
        )
        self.qmap_feature_ga2 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_ga3 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_ga4 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, in_channels, 1, 1),
        )
        #----------------------------------------------------------------
        # HE
        self.ha1 = conv(in_channels=in_channels * 2, out_channels=256, kernel_size=3, stride=1)
        self.ha1_SFT =SFT(x_nc=256, prior_nc=256, ks=3)
        self.ha1_act = nn.LeakyReLU()
        self.ha2 = conv(in_channels=256, out_channels=256, kernel_size=5, stride=2)
        self.ha2_SFT = SFT(x_nc=256, prior_nc=256, ks=3)
        self.ha2_act = nn.LeakyReLU()
        self.ha3 = conv(in_channels=256, out_channels=256, kernel_size=5, stride=2)
        self.ha3_ResB1 = SFTResblk(256, 256, ks=3)
        self.ha3_ResB2 = SFTResblk(256, 256, ks=3)

        self.qmap_feature_ha1 = nn.Sequential(
            conv(in_channels*2 + 1, 128, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(128, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 256, 3, 1)
        )
        self.qmap_feature_ha2 = nn.Sequential(
            conv(256, 256, 3),
            nn.LeakyReLU(0.1, True),
            conv(256, 256, 1, 1),
        )
        self.qmap_feature_ha3 = nn.Sequential(
            conv(256, 256, 3),
            nn.LeakyReLU(0.1, True),
            conv(256, 256, 1, 1),
        )

        # HD
        self.hs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels * 2, kernel_size=3, padding=3 // 2, stride=1),
        )
        #-------------------------------------------------------------------------------------------------
        # gs
        # generate wmap for decoder side
        self.wmap_generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=192, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=3 // 2, stride=1),
        ) # --> w

        self.gs0_SFTResB1 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        self.gs0_SFTResB2 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        self.gs1 = nn.Sequential(
            deconv(in_channels, 128),
            GDN(128, inverse=True),
        )
        self.gs1_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs2 = nn.Sequential(
            deconv(128, 128),
            GDN(128, inverse=True),
        )
        self.gs2_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs3 = nn.Sequential(
            deconv(128, 128),
            GDN(128, inverse=True),
        )
        self.gs3_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs4 = deconv(128, 3)


        self.qmap_feature_gs0 = nn.Sequential(
            conv(64+in_channels, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 192, 3, 1)
        ) # --> SFTResB
        self.qmap_feature_gs1 = nn.Sequential(
            deconv(192, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_gs2 = nn.Sequential(
            deconv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_gs3 = nn.Sequential(
            deconv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        #---------------------------------------------------------------------------------------------------
        self.ConditionEncoder = nn.Sequential(
            conv(3, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, in_channels),
        )
        #-----------------------------------------Entropy Model---------------------------------------------
        self.TPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=320, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=320, out_channels=in_channels * 2, kernel_size=5, padding=5 // 2, stride=1),
        )
        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2 * 2, out_channels=768, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=768, out_channels=576, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=576, out_channels=in_channels * 2, kernel_size=1, padding=1 // 2, stride=1),
        )

        self.gaussian_conditional = GaussianConditional(None)

    def PEncoder(self, x, Qmap):
        Qmap = self.qmap_feature_ga1(torch.cat([x, Qmap], dim=1))
        x =self.ga1(x)
        x = self.ga1_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga2(Qmap)
        x = self.ga2(x)
        x = self.ga2_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga3(Qmap)
        x = self.ga3(x)
        x = self.ga3_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga4(Qmap)
        x = self.ga4(x)
        x = self.ga4_SFTResB1(x, Qmap)
        x = self.ga4_SFTResB2(x, Qmap)

        return  x

    def PDecoder(self, x, z):
        w = self.wmap_generator(z)
        w = self.qmap_feature_gs0(torch.cat([w, x], dim=1))
        x = self.gs0_SFTResB1(x, w)
        x = self.gs0_SFTResB2(x, w)

        w = self.qmap_feature_gs1(w)
        x = self.gs1(x)
        x = self.gs1_SFT(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.gs2(x)
        x = self.gs2_SFT(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.gs3(x)
        x = self.gs3_SFT(x, w)

        x = self.gs4(x)

        return x

    def HE(self, x, Qmap):
        Qmap = F.adaptive_avg_pool2d(Qmap, x.size()[2:])
        Qmap = self.qmap_feature_ha1(torch.cat([Qmap, x], dim=1))
        x = self.ha1(x)
        x = self.ha1_SFT(x, Qmap)
        x = self.ha1_act(x)

        Qmap = self.qmap_feature_ha2(Qmap)
        x = self.ha2(x)
        x = self.ha2_SFT(x, Qmap)
        x = self.ha2_act(x)

        Qmap = self.qmap_feature_ha3(Qmap)
        x = self.ha3(x)
        x = self.ha3_ResB1(x, Qmap)
        x = self.ha3_ResB2(x, Qmap)

        return x

    def HD(self, x):
        return self.hs(x)


    def forward(self, x_cur, x_conditioned, Qmap):
        y_cur = self.PEncoder(x_cur, Qmap)
        y_conditioned = self.ConditionEncoder(x_conditioned)

        z = self.HE(torch.cat([y_cur, y_conditioned], 1), Qmap)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)


        temporalprior_params = self.TPM(y_conditioned)


        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat)

        x_hat = self.PDecoder(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def forward_compress(self, x_cur, x_conditioned, Qmap):
        y_cur = self.PEncoder(x_cur, Qmap)
        y_conditioned = self.ConditionEncoder(x_conditioned)

        z = self.HE(torch.cat([y_cur, y_conditioned], 1), Qmap)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat)

        x_hat = self.PDecoder(y_hat, z_hat)

        return {
            # "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def forward_decompress(self, z_hat, y_hat, x_conditioned):
        hyperprior_params = self.HD(z_hat)

        y_conditioned = self.ConditionEncoder(x_conditioned)
        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))

        x_hat = self.PDecoder(y_hat, z_hat).clamp_(0, 1)
        # return {"x_hat": x_hat, "y_hat": y_hat, "entropy_params": {"scales_hat": scales_hat, "means_hat": means_hat}}


    def compress(self, x_cur, x_conditioned, Qmap):
        y_cur = self.PEncoder(x_cur, Qmap)
        y_conditioned = self.ConditionEncoder(x_conditioned)
        z = self.HE(torch.cat([y_cur, y_conditioned], 1), Qmap)


        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y_cur, indexes, means=means_hat)  # y_string 是 list, 且只包含一个元素
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


    def decompress(self, strings, shape, x_conditioned):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        y_conditioned = self.ConditionEncoder(x_conditioned)
        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        x_hat = self.PDecoder(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat, "entropy_params": {"scales_hat": scales_hat, "means_hat": means_hat}}


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


# remove condition net in gs
class stem_roi_wo_gsc(CompressionModel):
    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)

        #--------------------------------------------------------
        # g_a
        self.ga1 = nn.Sequential(
            conv(3, 128),
            GDN(128),
        )
        self.ga1_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.ga2 = nn.Sequential(
            conv(128, 128),
            GDN(128),
        )
        self.ga2_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.ga3 = nn.Sequential(
            conv(128, 128),
            GDN(128),
        )
        self.ga3_SFT = SFT(x_nc=128, prior_nc=128, ks=3)

        self.ga4 = conv(128, in_channels)
        self.ga4_SFTResB1 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        self.ga4_SFTResB2 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)

        self.qmap_feature_ga1 = nn.Sequential(
            conv(4, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 160, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(160, 128, 3, 1)
        )
        self.qmap_feature_ga2 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_ga3 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_ga4 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, in_channels, 1, 1),
        )
        #----------------------------------------------------------------
        # HE
        self.ha1 = conv(in_channels=in_channels * 2, out_channels=256, kernel_size=3, stride=1)
        self.ha1_SFT =SFT(x_nc=256, prior_nc=256, ks=3)
        self.ha1_act = nn.LeakyReLU()
        self.ha2 = conv(in_channels=256, out_channels=256, kernel_size=5, stride=2)
        self.ha2_SFT = SFT(x_nc=256, prior_nc=256, ks=3)
        self.ha2_act = nn.LeakyReLU()
        self.ha3 = conv(in_channels=256, out_channels=256, kernel_size=5, stride=2)
        self.ha3_ResB1 = SFTResblk(256, 256, ks=3)
        self.ha3_ResB2 = SFTResblk(256, 256, ks=3)

        self.qmap_feature_ha1 = nn.Sequential(
            conv(in_channels*2 + 1, 128, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(128, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 256, 3, 1)
        )
        self.qmap_feature_ha2 = nn.Sequential(
            conv(256, 256, 3),
            nn.LeakyReLU(0.1, True),
            conv(256, 256, 1, 1),
        )
        self.qmap_feature_ha3 = nn.Sequential(
            conv(256, 256, 3),
            nn.LeakyReLU(0.1, True),
            conv(256, 256, 1, 1),
        )

        # HD
        self.hs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels * 2, kernel_size=3, padding=3 // 2, stride=1),
        )
        #-------------------------------------------------------------------------------------------------
        # gs
        # generate wmap for decoder side
        # self.wmap_generator = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=256, out_channels=192, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=3 // 2, stride=1),
        # ) # --> w
        #
        # self.gs0_SFTResB1 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        # self.gs0_SFTResB2 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        self.gs1 = nn.Sequential(
            deconv(in_channels, 128),
            GDN(128, inverse=True),
        )
        # self.gs1_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs2 = nn.Sequential(
            deconv(128, 128),
            GDN(128, inverse=True),
        )
        # self.gs2_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs3 = nn.Sequential(
            deconv(128, 128),
            GDN(128, inverse=True),
        )
        # self.gs3_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs4 = deconv(128, 3)
        #
        #
        # self.qmap_feature_gs0 = nn.Sequential(
        #     conv(64+in_channels, 192, 3, 1),
        #     nn.LeakyReLU(0.1, True),
        #     conv(192, 192, 3, 1),
        #     nn.LeakyReLU(0.1, True),
        #     conv(192, 192, 3, 1)
        # ) # --> SFTResB
        # self.qmap_feature_gs1 = nn.Sequential(
        #     deconv(192, 128, 3),
        #     nn.LeakyReLU(0.1, True),
        #     conv(128, 128, 1, 1),
        # )
        # self.qmap_feature_gs2 = nn.Sequential(
        #     deconv(128, 128, 3),
        #     nn.LeakyReLU(0.1, True),
        #     conv(128, 128, 1, 1),
        # )
        # self.qmap_feature_gs3 = nn.Sequential(
        #     deconv(128, 128, 3),
        #     nn.LeakyReLU(0.1, True),
        #     conv(128, 128, 1, 1),
        # )
        #---------------------------------------------------------------------------------------------------
        self.ConditionEncoder = nn.Sequential(
            conv(3, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, in_channels),
        )
        #-----------------------------------------Entropy Model---------------------------------------------
        self.TPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=320, kernel_size=5, padding=5 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=320, out_channels=in_channels * 2, kernel_size=5, padding=5 // 2, stride=1),
        )
        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2 * 2, out_channels=768, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=768, out_channels=576, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=576, out_channels=in_channels * 2, kernel_size=1, padding=1 // 2, stride=1),
        )

        self.gaussian_conditional = GaussianConditional(None)

    def PEncoder(self, x, Qmap):
        Qmap = self.qmap_feature_ga1(torch.cat([x, Qmap], dim=1))
        x =self.ga1(x)
        x = self.ga1_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga2(Qmap)
        x = self.ga2(x)
        x = self.ga2_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga3(Qmap)
        x = self.ga3(x)
        x = self.ga3_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga4(Qmap)
        x = self.ga4(x)
        x = self.ga4_SFTResB1(x, Qmap)
        x = self.ga4_SFTResB2(x, Qmap)

        return  x

    def PDecoder(self, x, z):
        # w = self.wmap_generator(z)
        # w = self.qmap_feature_gs0(torch.cat([w, x], dim=1))
        # x = self.gs0_SFTResB1(x, w)
        # x = self.gs0_SFTResB2(x, w)

        # w = self.qmap_feature_gs1(w)
        x = self.gs1(x)
        # x = self.gs1_SFT(x, w)

        # w = self.qmap_feature_gs2(w)
        x = self.gs2(x)
        # x = self.gs2_SFT(x, w)

        # w = self.qmap_feature_gs3(w)
        x = self.gs3(x)
        # x = self.gs3_SFT(x, w)

        x = self.gs4(x)

        return x

    def HE(self, x, Qmap):
        Qmap = F.adaptive_avg_pool2d(Qmap, x.size()[2:])
        Qmap = self.qmap_feature_ha1(torch.cat([Qmap, x], dim=1))
        x = self.ha1(x)
        x = self.ha1_SFT(x, Qmap)
        x = self.ha1_act(x)

        Qmap = self.qmap_feature_ha2(Qmap)
        x = self.ha2(x)
        x = self.ha2_SFT(x, Qmap)
        x = self.ha2_act(x)

        Qmap = self.qmap_feature_ha3(Qmap)
        x = self.ha3(x)
        x = self.ha3_ResB1(x, Qmap)
        x = self.ha3_ResB2(x, Qmap)

        return x

    def HD(self, x):
        return self.hs(x)


    def forward(self, x_cur, x_conditioned, Qmap):
        y_cur = self.PEncoder(x_cur, Qmap)
        y_conditioned = self.ConditionEncoder(x_conditioned)

        z = self.HE(torch.cat([y_cur, y_conditioned], 1), Qmap)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)


        temporalprior_params = self.TPM(y_conditioned)


        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat)

        x_hat = self.PDecoder(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x_cur, x_conditioned, Qmap):
        y_cur = self.PEncoder(x_cur, Qmap)
        y_conditioned = self.ConditionEncoder(x_conditioned)
        z = self.HE(torch.cat([y_cur, y_conditioned], 1), Qmap)


        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)

        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y_cur, indexes, means=means_hat)  # y_string 是 list, 且只包含一个元素
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


    def decompress(self, strings, shape, x_conditioned):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        y_conditioned = self.ConditionEncoder(x_conditioned)
        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        x_hat = self.PDecoder(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat, "entropy_params": {"scales_hat": scales_hat, "means_hat": means_hat}}


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


# i frame model for stem roi-coding and continuous rate model
class stem_roi_i(CompressionModel):
    def __init__(self, entropy_bottleneck_channels=256, in_channels=192):
        super().__init__(entropy_bottleneck_channels = entropy_bottleneck_channels)

        #--------------------------------------------------------
        # g_a
        self.ga1 = nn.Sequential(
            conv(3, 128),
            GDN(128),
        )
        self.ga1_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.ga2 = nn.Sequential(
            conv(128, 128),
            GDN(128),
        )
        self.ga2_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.ga3 = nn.Sequential(
            conv(128, 128),
            GDN(128),
        )
        self.ga3_SFT = SFT(x_nc=128, prior_nc=128, ks=3)

        self.ga4 = conv(128, in_channels)
        self.ga4_SFTResB1 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        self.ga4_SFTResB2 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)

        self.qmap_feature_ga1 = nn.Sequential(
            conv(4, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 160, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(160, 128, 3, 1)
        )
        self.qmap_feature_ga2 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_ga3 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_ga4 = nn.Sequential(
            conv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, in_channels, 1, 1),
        )
        #----------------------------------------------------------------
        # HE
        self.ha1 = conv(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1)
        self.ha1_SFT =SFT(x_nc=256, prior_nc=256, ks=3)
        self.ha1_act = nn.LeakyReLU()
        self.ha2 = conv(in_channels=256, out_channels=256, kernel_size=5, stride=2)
        self.ha2_SFT = SFT(x_nc=256, prior_nc=256, ks=3)
        self.ha2_act = nn.LeakyReLU()
        self.ha3 = conv(in_channels=256, out_channels=256, kernel_size=5, stride=2)
        self.ha3_ResB1 = SFTResblk(256, 256, ks=3)
        self.ha3_ResB2 = SFTResblk(256, 256, ks=3)

        self.qmap_feature_ha1 = nn.Sequential(
            conv(in_channels + 1, 128, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(128, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 256, 3, 1)
        )
        self.qmap_feature_ha2 = nn.Sequential(
            conv(256, 256, 3),
            nn.LeakyReLU(0.1, True),
            conv(256, 256, 1, 1),
        )
        self.qmap_feature_ha3 = nn.Sequential(
            conv(256, 256, 3),
            nn.LeakyReLU(0.1, True),
            conv(256, 256, 1, 1),
        )

        # HD
        self.hs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels * 2, kernel_size=3, padding=3 // 2, stride=1),
        )
        #-------------------------------------------------------------------------------------------------
        # gs
        # generate wmap for decoder side
        self.wmap_generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=192, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=3 // 2, stride=1),
        ) # --> w

        self.gs0_SFTResB1 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        self.gs0_SFTResB2 = SFTResblk(in_channels, prior_nc=in_channels, ks=3)
        self.gs1 = nn.Sequential(
            deconv(in_channels, 128),
            GDN(128, inverse=True),
        )
        self.gs1_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs2 = nn.Sequential(
            deconv(128, 128),
            GDN(128, inverse=True),
        )
        self.gs2_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs3 = nn.Sequential(
            deconv(128, 128),
            GDN(128, inverse=True),
        )
        self.gs3_SFT = SFT(x_nc=128, prior_nc=128, ks=3)
        self.gs4 = deconv(128, 3)


        self.qmap_feature_gs0 = nn.Sequential(
            conv(64+in_channels, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 192, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(192, 192, 3, 1)
        ) # --> SFTResB
        self.qmap_feature_gs1 = nn.Sequential(
            deconv(192, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_gs2 = nn.Sequential(
            deconv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        self.qmap_feature_gs3 = nn.Sequential(
            deconv(128, 128, 3),
            nn.LeakyReLU(0.1, True),
            conv(128, 128, 1, 1),
        )
        #---------------------------------------------------------------------------------------------------
        self.ConditionEncoder = nn.Sequential(
            conv(3, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, 128),
            GDN(128),
            conv(128, in_channels),
        )
        #-----------------------------------------Entropy Model---------------------------------------------
        # self.TPM = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=5, padding=5 // 2, stride=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=320, kernel_size=5, padding=5 // 2, stride=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=320, out_channels=in_channels * 2, kernel_size=5, padding=5 // 2, stride=1),
        # )
        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=768, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=768, out_channels=576, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=576, out_channels=in_channels * 2, kernel_size=1, padding=1 // 2, stride=1),
        )

        self.gaussian_conditional = GaussianConditional(None)

    def PEncoder(self, x, Qmap):
        Qmap = self.qmap_feature_ga1(torch.cat([x, Qmap], dim=1))
        x =self.ga1(x)
        x = self.ga1_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga2(Qmap)
        x = self.ga2(x)
        x = self.ga2_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga3(Qmap)
        x = self.ga3(x)
        x = self.ga3_SFT(x, Qmap)

        Qmap = self.qmap_feature_ga4(Qmap)
        x = self.ga4(x)
        x = self.ga4_SFTResB1(x, Qmap)
        x = self.ga4_SFTResB2(x, Qmap)

        return  x

    def PDecoder(self, x, z):
        w = self.wmap_generator(z)
        w = self.qmap_feature_gs0(torch.cat([w, x], dim=1))
        x = self.gs0_SFTResB1(x, w)
        x = self.gs0_SFTResB2(x, w)

        w = self.qmap_feature_gs1(w)
        x = self.gs1(x)
        x = self.gs1_SFT(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.gs2(x)
        x = self.gs2_SFT(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.gs3(x)
        x = self.gs3_SFT(x, w)

        x = self.gs4(x)

        return x

    def HE(self, x, Qmap):
        Qmap = F.adaptive_avg_pool2d(Qmap, x.size()[2:])
        Qmap = self.qmap_feature_ha1(torch.cat([Qmap, x], dim=1))
        x = self.ha1(x)
        x = self.ha1_SFT(x, Qmap)
        x = self.ha1_act(x)

        Qmap = self.qmap_feature_ha2(Qmap)
        x = self.ha2(x)
        x = self.ha2_SFT(x, Qmap)
        x = self.ha2_act(x)

        Qmap = self.qmap_feature_ha3(Qmap)
        x = self.ha3(x)
        x = self.ha3_ResB1(x, Qmap)
        x = self.ha3_ResB2(x, Qmap)

        return x

    def HD(self, x):
        return self.hs(x)

    def forward(self, x_cur, Qmap):
        y_cur = self.PEncoder(x_cur, Qmap)

        z = self.HE(y_cur, Qmap)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyperprior_params = self.HD(z_hat)


        # temporalprior_params = self.TPM(y_conditioned)


        gaussian_params = self.EPM(hyperprior_params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat)

        x_hat = self.PDecoder(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x_cur, Qmap):
        y_cur = self.PEncoder(x_cur, Qmap)
        z = self.HE(y_cur, Qmap)


        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyperprior_params = self.HD(z_hat)

        # temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(hyperprior_params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y_cur, indexes, means=means_hat)  # y_string 是 list, 且只包含一个元素
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyperprior_params = self.HD(z_hat)

        # y_conditioned = self.ConditionEncoder(x_conditioned)
        # temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(hyperprior_params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        x_hat = self.PDecoder(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat, "entropy_params": {"scales_hat": scales_hat, "means_hat": means_hat}}


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