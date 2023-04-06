import argparse
import json
import math
import os
import sys
import time
import numpy as np
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf
from torchvision.utils import save_image, make_grid
from pathlib import Path

import compressai
from compressai.zoo import models
from compressai.models.stem_roi import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class STEMTestDataset(Dataset):
    def __init__(self, data_root, GOP):
        '''
        Creates a Video Test Dataset.
        Inputs.
            data_root: root path for the video test dataset. e.g. \\Dataset\\UVG\\PNG\\
            GOP: test gop size. usually set to 12 for uvg/10 for hevc ctc.
            Quality: test quality
        '''
        self.data_root = data_root # .\UVG\PNG
        sequence_names = os.listdir(self.data_root) # ['Beauty', 'Bosphorus', 'HoneyBee', ...]
        print("sequences:", sequence_names)
        self.image_paths = []
        self.len_mark = [0]
        cnt = 0
        for seq_name in sequence_names:
            seq_path = os.path.join(self.data_root, seq_name) # .\UVG\PNG\Beauty
            image_names = os.listdir(seq_path) # ['f001.png', 'f002.png', ...]
            image_names = image_names[0:int(GOP*10)] # only test 10 GOPs
            cnt = cnt + len(image_names)
            for img_name in image_names:
                self.image_paths.append(os.path.join(seq_path, img_name))
            self.len_mark.append(cnt)  # [0, 600, 1200, 1500] seq1:0~599\seq2:600~1200
        self.gop = GOP
        for l in self.len_mark:
            assert l % self.gop == 0, f"The sequence length should be divisible by GOP. Get len:{l} and GOP:{self.gop}"

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        # Load images
        image = tf.to_tensor(Image.open(self.image_paths[index]))

        isIntra = False
        if index % self.gop == 0 or index in self.len_mark:
            isIntra = True

        return image, isIntra



def get_loader(data_root, GOP=12):
    dataset = STEMTestDataset(data_root, GOP=GOP)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


@torch.no_grad()
def inference_i(model_i, x):
    # x = x.unsqueeze(0)  # 增加batch维度
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p # padding为64的倍数
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
    # print(x_padded.size())
    # print(qmap_padded.size())
    out_enc = model_i.compress(x_padded)
    out_forward = model_i(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model_i.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bits = sum(len(s[0]) for s in out_enc["strings"]) * 8.0
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    estimate_bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                       for likelihoods in out_forward["likelihoods"].values()).item()


    return {
        "x_hat": out_dec["x_hat"],
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "bits": bits,
        "estimate_bpp": estimate_bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

@torch.no_grad()
def inference_p(model_p, x, x_condition):
    # x = x.unsqueeze(0)  # 增加batch维度
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p # padding为64的倍数
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
    x_condition_padded = F.pad(
        x_condition,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model_p.compress(x_padded, x_condition_padded)
    out_forward = model_p(x_padded, x_condition_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model_p.decompress(out_enc["strings"], out_enc["shape"], x_condition_padded)
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bits = sum(len(s[0]) for s in out_enc["strings"]) * 8.0
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    estimate_bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                       for likelihoods in out_forward["likelihoods"].values()).item()


    return {
        "x_hat": out_dec["x_hat"],
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "bits": bits,
        "estimate_bpp": estimate_bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


def _eval_stem_roi(model_i, model_p, test_loader, logfile):
    device = next(model_i.parameters()).device
    PSNR, MSSSIM, BPP, Esti_Bpp = [], [], [], []
    for i, (image, isIntra) in enumerate(test_loader):
        image = image.to(device)
        if isIntra:
            out = inference_i(model_i, image)
            x_conditioned = out["x_hat"]
        else:
            assert i != 0, "The first image should be intra frame!"
            out = inference_p(model_p, image, x_conditioned)
            x_conditioned = out['x_hat']

        PSNR.append(out["psnr"])
        MSSSIM.append(out["ms-ssim"])
        BPP.append(out["bpp"])
        Esti_Bpp.append(out["estimate_bpp"])
        if i % 100 == 0 and i != 0:
            print(
                f'{i}/{len(test_loader)} '
                f'PSNR_AVE: {np.array(PSNR).mean():.3f}  '
                f'MSSSIM_AVE: {np.array(MSSSIM).mean():.3f}  '
                f'BPP_AVE: {np.array(BPP).mean():.3f}  '
                f'Est_BPP_AVE: {np.array(Esti_Bpp).mean():.3f}'
            )

    logfile.write(
        f'PSNR_AVE: {np.array(PSNR).mean():.3f}  '
        f'MSSSIM_AVE: {np.array(MSSSIM).mean():.3f}  '
        f'BPP_AVE: {np.array(BPP).mean():.3f}  '
        f'Est_BPP_AVE: {np.array(Esti_Bpp).mean():.3f}\n'
    )
    logfile.flush()


def _eval_stem_roi_seq(model_i, model_p, logfile):
    device = next(model_i.parameters()).device
    PSNR, MSSSIM, BPP, Esti_Bpp = [], [], [], []
    maxIndex = 121
    GOP = 10
    seq_name = ["Kimono", "ParkScene", "Cactus", "BQTerrace", "BasketballDrive"]
    dataset_path = "C:\\Users\Administrator\Dataset\HEVCTestSequenceRawImages\B\\"
    for name in seq_name:
        seq_path = dataset_path + name
        logfile.write('\n' + name + '\n')
        for index in range(1, maxIndex):
            image = transforms.ToTensor()(Image.open(seq_path + f'/f{index:03d}.png')).to(device)
            print(seq_path + f'/f{index:03d}.png')

            if index % GOP == 1:
                out = inference_i(model_i, image)
                x_conditioned = out["x_hat"]
            else:
                out = inference_p(model_p, image, x_conditioned)
                x_conditioned = out['x_hat']

            print(f"bits:{out['bits']} psnr:{out['psnr']:.3f}")
            logfile.write(f"bits:{out['bits']} "
                          f"psnr:{out['psnr']:.3f}\n")



def eval(model_i, model_p, test_loader, args):
    logfile = open(args.logpath, 'w+')
    logfile.write("Eval Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
    logfile.write("Eval model_i:" + args.checkpoint_i + '\n')
    logfile.write("     model_p:" + args.checkpoint_p + '\n')
    _eval_stem_roi_seq(model_i, model_p, logfile)
    # _eval_stem_roi(model_i, model_p, test_loader, logfile)
    logfile.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="MXH evaluation script.")
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018-mean",
        choices=models.keys(),
        help="I Frame Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="C:\\Users\Administrator\Dataset\ImageCompressionDataset\\test"
        , help="Evaluation dataset")
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--checkpoint-i", type=str, help="Path to the test i frame checkpoint", required=True)
    parser.add_argument("--checkpoint-p", type=str, help="Path to the test p frame checkpoint", required=True)
    parser.add_argument("--logpath", type=str, help="Result Output Path", required=True)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    args.model = "mbt2018-mean"
    model_i = models[args.model](quality=4)
    model_i = model_i.to(device)
    model_p = stem_baseline()
    model_p = model_p.to(device)
    # Load Model
    if args.checkpoint_i:  # load ImageCompressor
        print("Loading i model ", args.checkpoint_i)
        checkpoint = torch.load(args.checkpoint_i, map_location=device)
        model_i.load_state_dict(checkpoint["state_dict"])
        model_i.update(force=True)
        model_i.eval()
    if args.checkpoint_p:  # load ImageCompressor
        print("Loading p model ", args.checkpoint_p)
        checkpoint = torch.load(args.checkpoint_p, map_location=device)
        model_p.load_state_dict(checkpoint["state_dict"])
        model_p.update(force=True)
        model_p.eval()

    compressai.set_entropy_coder(args.entropy_coder)

    test_loader = get_loader(args.dataset, GOP=10)

    print("Result Output Path:" + args.logpath)

    start = time.time()
    eval(model_i, model_p, test_loader, args=args)
    eval_time = time.time() - start
    print(f"eval time:{eval_time:.3f}s.")




if __name__ == "__main__":
    main(sys.argv[1:])


