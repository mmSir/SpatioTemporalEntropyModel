import argparse
import json
import math
import os
import sys
import time
import numpy as np

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import compressai

from compressai.zoo import models
from compressai.models.spatiotemporalpriors import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


@torch.no_grad()
def inferenceI_DVR(model, x):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    out_forward = model(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    y_conditioned = out_dec["y_hat"]

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    estimate_bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                       for likelihoods in out_forward["likelihoods"].values()).item()
    estimate_y_bpp = (torch.log(out_forward["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)).item()
    y_string = len(out_enc["strings"][0][0]) * 8/ num_pixels
    estimate_z_bpp = (torch.log(out_forward["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)).item()
    z_string = len(out_enc["strings"][1][0]) * 8 / num_pixels
    print(f"I frame\nactual_bpp:{bpp:.4f}  estimate_bpp:{estimate_bpp:.4f}")
    print(f"ystring:{y_string:.4f}" + f"  estiamte_y:{estimate_y_bpp:.4f}")
    print(f"zstring:{z_string:.4f}" + f"  estiamte_z:{estimate_z_bpp:.4f}")

    return {
        "y_conditioned": y_conditioned,
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "estimate_bpp": estimate_bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
        "out_forward":out_forward,
    }


@torch.no_grad()
def inferenceP_DVR(IFrameCompressor, stem, x, y_conditioned):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


    start = time.time()
    y_cur, _ = IFrameCompressor.getY(x_padded)
    out_forward = stem(y_cur, y_conditioned)
    out_enc = stem.compress(y_cur, y_conditioned)
    enc_time = time.time() - start

    start = time.time()
    out_dec = stem.decompress(out_enc["strings"], out_enc["shape"], y_conditioned)
    y_hat = out_dec["y_hat"]
    x_hat = IFrameCompressor.getX(y_hat)
    dec_time = time.time() - start

    y_conditioned = y_hat

    # move pad pixel
    x_hat = F.pad(
        x_hat, (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    estimate_bpp = sum( (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                        for likelihoods in out_forward["likelihoods"].values() ).item()
    estimate_y_bpp = (torch.log(out_forward["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)).item()
    estimate_z_bpp = (torch.log(out_forward["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)).item()
    actual_y_bpp = len(out_enc["strings"][0][0]) * 8.0 / num_pixels
    actual_z_bpp = len(out_enc["strings"][1][0]) * 8.0 / num_pixels
    print(f"P frame\nactual_bpp:{bpp:.4f}  estimate_bpp:{estimate_bpp:.4f}")
    print(f"y_bpp:{actual_y_bpp:.4f}  estimate_y_bpp:{estimate_y_bpp:.4f}")
    print(f"z_bpp:{actual_z_bpp:.4f}  estimate_z_bpp:{estimate_z_bpp:.4f}")


    return {
        "y_conditioned": y_conditioned,
        "psnr": psnr(x, x_hat),
        "ms-ssim": ms_ssim(x, x_hat, data_range=1.0).item(),
        "bpp": bpp,
        "estimate_bpp": estimate_bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
        "entropy_params": out_dec["entropy_params"]
    }


def evalDataset(ImageCompressor, stem, dataset, logfile, ALLIntra=False):
    dataset = dataset.upper()
    if dataset == "UVG":
        seq_name = ["Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySteadyGo", "ShakeNDry", "YachtRide"]
        dataset_path = "C:/Users/Administrator/Dataset/UVG/PNG/"
    elif dataset == "HEVCB":
        seq_name = ["Kimono", "ParkScene", "Cactus", "BQTerrace", "BasketballDrive"]
        dataset_path = "D:/MXH/HEVCTestSequenceRawImages/"
    elif dataset == "HEVCC":
        seq_name = ["RaceHorses_832x480", "BQMall", "PartyScene", "BasketballDrill"]
        dataset_path = "D:/MXH/HEVCTestSequenceRawImages/"
    else:
        raise ValueError("Unknown dataset?")
    print(f"Evaluating on {dataset} Dataset! AllIntra: {ALLIntra}")

    global y_conditioned, y_conditioned2, y_conditioned1
    device = next(ImageCompressor.parameters()).device
    PSNR, MSSSIM, BPP, Esti_Bpp = [], [], [], []
    for name in seq_name:
        seq_path = dataset_path + name
        # if dataset == "UVG":
        #     maxIndex = 301 if name == "ShakeNDry" else 601
        # else:
        #     maxIndex = 101
        if dataset == "UVG":
            maxIndex = 37
        else:
            maxIndex = 31
        for index in range(1, maxIndex):
            image = transforms.ToTensor()(Image.open(seq_path + f'/f{index:03d}.png')).to(device)
            print(seq_path + f'/f{index:03d}.png')
            GOP = 12 if dataset == "UVG" else 10
            if ALLIntra:
                ImageCompressor = ImageCompressor.to("cpu")
                image.to("cpu")
                out = inferenceI_DVR(ImageCompressor, image)
            else:
                if index % GOP == 1:
                    # I frame
                    ImageCompressor = ImageCompressor.to("cpu")
                    image.to("cpu")

                    out = inferenceI_DVR(ImageCompressor, image)
                    y_conditioned = out["y_conditioned"]
                else:
                    # P frame
                    ImageCompressor = ImageCompressor.to("cuda")
                    stem = stem.to("cuda")
                    image = image.to("cuda")
                    y_conditioned = y_conditioned.to("cuda")

                    # normal stem
                    out = inferenceP_DVR(ImageCompressor, stem, image, y_conditioned)
                    y_conditioned = out["y_conditioned"]


            PSNR.append(out["psnr"])
            MSSSIM.append(out["ms-ssim"])
            BPP.append(out["bpp"])
            Esti_Bpp.append(out["estimate_bpp"])

    PSNR = np.array(PSNR)
    MSSSIM = np.array(MSSSIM)
    BPP = np.array(BPP)
    Esti_Bpp = np.array(Esti_Bpp)
    logfile.write(
        f'Dataset: {dataset}  PSNR_AVE: {PSNR.mean():.3f}  MSSSIM_AVE: {MSSSIM.mean():.3f}  BPP_AVE: {BPP.mean():.3f}')
    index = 1
    for i, j in zip(PSNR, BPP):
        if (index % (maxIndex - 1) == 1):
            name = seq_name[int(index / maxIndex)]
            logfile.write(f"\nSequence Name:{name}\n")
        frameindex = index % (maxIndex - 1) if index % (maxIndex - 1) != 0 else maxIndex - 1
        logfile.write(f"FrameIndex: {frameindex:03d}  "
                      f"PSNR: {i:.3f}  Total_bpp: {j:.3f}\n")
        index += 1


def evalstem_DVR(IFrameCompressor, stem):
    '''
        eval for discrete variable rate stem model
    '''
    logfile = open('D:\MXH\stem\CompressAI\stem4Video\\train\log.txt', 'a+')
    evalDataset(IFrameCompressor, stem, "uvg", logfile, ALLIntra=False)
    # evalDataset(IFrameCompressor, stem, "HEVCB", logfile, ALLIntra=False)
    # evalDataset(IFrameCompressor, stem, "HEVCC", logfile, ALLIntra=False)
    logfile.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="MXH evaluation script.")
    parser.add_argument(
        "-m",
        "--model",
        # default="mbt2018-mean",
        default="mbt2018",
        choices=models.keys(),
        help="I Frame Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-em",
        "--entropy-model-path",
        type=str,
        help="P Frame Entropy Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="D:\MXH\HEVCTestSequenceRawImages"
        , help="Evaluation dataset")
    parser.add_argument(
        "--tensorboard-runs", type=str, default="D:\MXH\StartDeepLearning\Compression\CompressAI\stem4Video")
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "-gop",
        type=int,
        default=10,
        help="GOP (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    print(f"IFrameCompressor model: {args.model}")
    IFrameCompressor = models[args.model](quality=4)
    IFrameCompressor = IFrameCompressor.to(device)
    stem = SpatioTemporalPriorModel_Res()
    # stem = SpatioTemporalPriorModel()
    # stem = SpatioTemporalPriorModelWithoutSPM()
    # stem = SpatioTemporalPriorModelWithoutTPM()
    # stem = SpatioTemporalPriorModelWithoutSPMTPM()
    stem = stem.to(device)

    # Load Model
    if args.checkpoint:  # load IFrameCompressor
        # args.checkpoint = 'D:\MXH\stem\CompressAI\\trainmeanscale\AutoregressiveMSHyperPrior\lmbda1e-2\ARMSHyperPrior_checkpoint_openimages_epoch50.pth.tar'
        print("Loading IFrameCompressor ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        IFrameCompressor.load_state_dict(checkpoint["state_dict"])
        # IFrameCompressor.load_state_dict(checkpoint)
        IFrameCompressor.update(force=True)
        IFrameCompressor.eval()
    if args.entropy_model_path: # load STEM
        # args.entropy_model_path = 'D:\MXH\stem\CompressAI\stem4Video\models\checkpoint_best_epoch26.pth.tar'
        print("Loading Entropy Model Compressor ", args.entropy_model_path)
        checkpoint = torch.load(args.entropy_model_path, map_location=device)
        stem.load_state_dict(checkpoint["state_dict"])
        # stem.load_state_dict(checkpoint)
        stem.update(force=True)
        stem.eval()

    compressai.set_entropy_coder(args.entropy_coder)

    start = time.time()
    evalstem_DVR(IFrameCompressor, stem)
    eval_time = time.time() - start
    print(f"eval time:{eval_time:.3f}s.")


if __name__ == "__main__":
    main(sys.argv[1:])