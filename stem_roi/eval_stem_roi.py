import argparse
import json
import math
import os
import sys
import time
import numpy as np
import random
import cv2


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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

QMAP_TESTMODE_MAX = 7


class STEMTestDataset_Qmap(Dataset):
    def __init__(self, data_root, GOP, level_range=(0, 100), quality=0):
        '''
        Creates a Video Test Dataset.
        Inputs.
            data_root: root path for the video test dataset. e.g. \\Dataset\\UVG\\PNG\\
            GOP: test gop size. usually set to 12.
            Quality: test quality
        '''
        self.data_root = data_root # .\UVG\PNG
        sequence_names = os.listdir(self.data_root) # ['Beauty', 'Bosphorus', 'HoneyBee', ...]
        self.image_paths = []
        self.len_mark = [0]
        cnt = 0
        for seq_name in sequence_names:
            seq_path = os.path.join(self.data_root, seq_name) # .\UVG\PNG\Beauty
            image_names = os.listdir(seq_path) # ['f001.png', 'f002.png', ...]
            image_names = image_names[0:int(GOP*10)]
            cnt = cnt + len(image_names)
            for img_name in image_names:
                self.image_paths.append(os.path.join(seq_path, img_name))
            self.len_mark.append(cnt)  # [0, 600, 1200, 1500] seq1:0~599\seq2:600~1200
        self.gop = GOP
        for l in self.len_mark:
            assert l % self.gop == 0, f"The sequence length should be divisible by GOP. Get len:{l} and GOP:{self.gop}"

        self.level_range = level_range # 划分精度，默认为100
        self.level = quality # for generate uniform level map

    def __len__(self):
        return len(self.image_paths)


    def _get_grid(self, size):
        x1 = torch.tensor(range(size[0]))
        x2 = torch.tensor(range(size[1]))
        grid_x1, grid_x2 = torch.meshgrid(x1, x2)

        grid1 = grid_x1.view(size[0], size[1], 1)
        grid2 = grid_x2.view(size[0], size[1], 1)
        grid = torch.cat([grid1, grid2], dim=-1)
        return grid


    def __getitem__(self, index):
        # Load images
        image = tf.to_tensor(Image.open(self.image_paths[index]))
        shape = image.size()[1:]
        if self.level == -1:
            # gradation between two levels, horizontal
            qmap = np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1)).astype(float)
        elif self.level == -2:
            # vertical
            qmap = np.tile(np.linspace(0, 1, shape[0]), (1, shape[1])).astype(float)
        else:
            # uniform
            qmap = np.zeros(shape, dtype=float)
            qmap[:] = self.level

        qmap = torch.FloatTensor(qmap).unsqueeze(dim=0)
        qmap *= 1 / self.level_range[1]  # 0~100 -> 0~1

        isIntra = False
        if index % self.gop == 0 or index in self.len_mark:
            isIntra = True

        return image, qmap, isIntra


def get_loader_roi(data_root, GOP=12, level_range=(0, 100), quality=0):
    dataset = STEMTestDataset_Qmap(data_root, GOP=GOP, level_range=level_range, quality=quality)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


@torch.no_grad()
def inference_i(model_i, x, qmap):
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
    qmap_padded = F.pad(
        qmap,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    # print(x_padded.size())
    # print(qmap_padded.size())
    out_enc = model_i.compress(x_padded, qmap_padded)
    out_forward = model_i(x_padded, qmap_padded)
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
def inference_p(model_p, x, x_condition, qmap):
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
    qmap_padded = F.pad(
        qmap,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model_p.compress(x_padded, x_condition_padded, qmap_padded)
    out_forward = model_p(x_padded, x_condition_padded, qmap_padded)
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


def _eval_stem_roi(model_i, model_p, test_loaders, logfile):
    device = next(model_i.parameters()).device
    for level, test_loader in enumerate(test_loaders):
        PSNR, MSSSIM, BPP, Esti_Bpp = [], [], [], []
        for i, (image, qmap, isIntra) in enumerate(test_loader):
            image = image.to(device)
            qmap = qmap.to(device)
            if isIntra:
                out = inference_i(model_i, image, qmap)
                x_conditioned = out["x_hat"]
            else:
                out = inference_p(model_p, image, x_conditioned, qmap)
                x_conditioned = out['x_hat']

            PSNR.append(out["psnr"])
            MSSSIM.append(out["ms-ssim"])
            BPP.append(out["bpp"])
            Esti_Bpp.append(out["estimate_bpp"])
            if i % 100 == 0 and i != 0:
                print(
                    f'test level={level}  {i}/{len(test_loader)} '
                    f'PSNR_AVE: {np.array(PSNR).mean():.3f}  '
                    f'MSSSIM_AVE: {np.array(MSSSIM).mean():.3f}  '
                    f'BPP_AVE: {np.array(BPP).mean():.3f}  '
                    f'Est_BPP_AVE: {np.array(Esti_Bpp).mean():.3f}'
                )

        logfile.write(
            f'test level={level}  '
            f'PSNR_AVE: {np.array(PSNR).mean():.3f}  '
            f'MSSSIM_AVE: {np.array(MSSSIM).mean():.3f}  '
            f'BPP_AVE: {np.array(BPP).mean():.3f}  '
            f'Est_BPP_AVE: {np.array(Esti_Bpp).mean():.3f}\n'
        )


def _eval_stem_roi_seq(model_i, model_p, logfile):
    device = next(model_i.parameters()).device
    PSNR, MSSSIM, BPP, Esti_Bpp = [], [], [], []
    maxIndex = 37# 121
    GOP = 12
    # seq_name = ["Kimono", "ParkScene", "Cactus", "BQTerrace", "BasketballDrive"]
    # dataset_path = "C:\\Users\Administrator\Dataset\HEVCTestSequenceRawImages\B\\"
    seq_name = ["FourPeople"]
    dataset_path = r"C:\Users\Administrator\Dataset\HEVCTestSequenceRawImages\E\\"
    for name in seq_name:
        seq_path = dataset_path + name
        logfile.write('\n' + name + '\n')
        for index in range(1, maxIndex):
            img = Image.open(seq_path + f'/f{index:03d}.png')
            image = transforms.ToTensor()(img).unsqueeze(dim=0).to(device)
            print(seq_path + f'/f{index:03d}.png')

            # qmap = np.zeros(img.size[::-1], dtype=float)
            # qmap[:] = 0.0
            # roi_img = cv2.imread(r"C:\Users\Administrator\Desktop\roi_visualization\FourPeople_roi_1.png")
            # qmap[:] = roi_img[:, :, 0] / 350
            # qmap = torch.FloatTensor(qmap).unsqueeze(dim=0).unsqueeze(dim=0).to(device)

            qmap = np.zeros(img.size[::-1], dtype=float)
            qmap[:] = 0.31
            qmap = torch.FloatTensor(qmap).unsqueeze(dim=0).unsqueeze(dim=0).to(device)

            if index % GOP == 1:
                out = inference_i(model_i, image, qmap)
                x_conditioned = out["x_hat"]
            else:
                out = inference_p(model_p, image, x_conditioned, qmap)
                x_conditioned = out['x_hat']

            save_image(out["x_hat"], rf"D:\MXH\毕业设计\materials\roi\FourPeople_uniform_{index}.png")
            print(f"index:{index} bpp:{out['bpp']} psnr:{out['psnr']:.3f}")
            logfile.write(f"bpp:{out['bpp']:.4f} "
                          f"psnr:{out['psnr']:.3f}\n")
            logfile.flush()


def _eval_stem_roi_seq_rc(model_i, model_p, level, logfile):
    device = next(model_i.parameters()).device
    maxIndex = 97# 121
    GOP = 12
    # seq_name = ["Kimono", "ParkScene", "Cactus", "BQTerrace", "BasketballDrive"]
    # dataset_path = "C:\\Users\Administrator\Dataset\HEVCTestSequenceRawImages\B\\"
    seq_name = ["BasketballDrill", "BQMall", "PartyScene", "RaceHorses_832x480"]
    dataset_path = r"C:\Users\Administrator\Dataset\HEVCTestSequenceRawImages\C\\"
    # seq_name = ["BasketballPass", "BlowingBubbles", "BQSquare", "RaceHorses_416x240"]
    # dataset_path = r"C:\Users\Administrator\Dataset\HEVCTestSequenceRawImages\D\\"
    # seq_name = ["FourPeople", "Johnny", "KristenAndSara"]
    # dataset_path = r"C:\Users\Administrator\Dataset\HEVCTestSequenceRawImages\E\\"
    for name in seq_name:
        PSNR, BPP, Bits = [], [], []
        seq_path = dataset_path + name
        logfile.write('\n' + name + '\n')
        for index in range(1, maxIndex):
            img = Image.open(seq_path + f'/f{index:03d}.png')
            image = transforms.ToTensor()(img).unsqueeze(dim=0).to(device)
            print(seq_path + f'/f{index:03d}.png')

            qmap = np.zeros(img.size[::-1], dtype=float)
            qmap[:] = level
            qmap = torch.FloatTensor(qmap).unsqueeze(dim=0).unsqueeze(dim=0).to(device)

            if index % GOP == 1:
                out = inference_i(model_i, image, qmap)
                x_conditioned = out["x_hat"]
            else:
                out = inference_p(model_p, image, x_conditioned, qmap)
                x_conditioned = out['x_hat']

            PSNR.append(out["psnr"])
            BPP.append(out["bpp"])
            Bits.append(out["bits"])
            print(f"index:{index} bpp:{out['bpp']} psnr:{out['psnr']:.3f}")
            logfile.write(f"bits:{out['bits']} "
                          f"bpp:{out['bpp']:.6f} "
                          f"psnr:{out['psnr']:.3f}\n")
            logfile.flush()

        logfile.write(
            f'test seq name: {name}   '
            f'PSNR_AVE: {np.array(PSNR).mean():.3f}  '
            f'BITS_AVE: {np.array(Bits).mean():.3f}  '
            f'BPP_AVE: {np.array(BPP).mean():.6f}\n'
        )
        logfile.flush()


def eval(model_i, model_p, test_loaders, args):
    logfile = open(args.logpath, 'w+')
    logfile.write("Eval Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
    logfile.write("Eval model_i:" + args.checkpoint_i + '\n')
    logfile.write("     model_p:" + args.checkpoint_p + '\n')
    # _eval_stem_roi_seq(model_i, model_p, logfile)

    # _eval_stem_roi(model_i, model_p, test_loaders, logfile)
    logfile.close()


def eval_rc(model_i, model_p, args):
    for level in [0.30, 0.45, 0.55, 0.7]:
        log_filename = f"ClassC_{int(level*100):2d}.txt"
        logfile = open(args.rclogpath+log_filename, 'w+')
        logfile.write("Eval Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        logfile.write("Eval model_i:" + args.checkpoint_i + '\n')
        logfile.write("     model_p:" + args.checkpoint_p + '\n')
        _eval_stem_roi_seq_rc(model_i, model_p, level, logfile)
        logfile.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="MXH evaluation script.")
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
    parser.add_argument("--rclogpath", type=str, help="Result Output Path")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model_i = stem_roi_i()
    model_i = model_i.to(device)
    model_p = stem_roi()
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

    L = 4
    # levels = [int(100 * (i / L)) for i in range(L + 1)] # 0, 25, 50, 75, 100
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    test_loaders = []
    for level in levels:
        test_loader = get_loader_roi(args.dataset, GOP=10, level_range=(0, 100), quality=level)
        test_loaders.append(test_loader)

    print("Result Output Path:" + args.logpath)

    start = time.time()
    # eval(model_i, model_p, test_loaders, args=args)
    eval_rc(model_i, model_p, args=args)
    eval_time = time.time() - start
    print(f"eval time:{eval_time:.3f}s.")




if __name__ == "__main__":
    main(sys.argv[1:])


