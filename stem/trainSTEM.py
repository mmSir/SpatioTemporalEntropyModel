import argparse
import math
import random
import sys
import os
import warnings
import time

from torch.utils.tensorboard import SummaryWriter

from compressai.zoo import models
from compressai.models.spatiotemporalpriors import *

from dataset_vidseq import get_loader
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        # default="mbt2018-mean",
        default="mbt2018",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "--tensorboard-runs", type=str, default="D:", help="Tensorboard Path"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a I frame compressor checkpoint")
    parser.add_argument(
        "-em",
        "--entropy-model-path",
        type=str,
        help="P Frame Entropy Model architecture (default: %(default)s)",
    )
    parser.add_argument("--model-save", type=str, help="Path to save a checkpoint")
    args = parser.parse_args(argv)
    return args


def train_singlerate(argv):
    '''
    train single rate stem
    '''
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

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

    optimizer, aux_optimizer = configure_optimizers(stem, args)  # set optimizer for stem
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.2)

    last_epoch = 0
    last_iterations = 0
    if args.i_model_path:
        # args.i_model_path = "D:\MXH\STPM\CompressAI\\trainmeanscale\Cheng2020\Attention\lmbda1e-2\ChengAttn_best_loss.pth.tar"
        # args.i_model_path = 'D:\MXH\STPM\CompressAI\\trainmeanscale\AutoregressiveMSHyperPrior\lmbda1e-2\ARMSHyperPrior_checkpoint_openimages_epoch50.pth.tar'
        print("Loading i_model_path: ", args.i_model_path)
        checkpoint = torch.load(args.i_model_path, map_location=device)
        # IFrameCompressor.load_state_dict(checkpoint)
        IFrameCompressor.load_state_dict(checkpoint["state_dict"])
    if args.entropy_model_path:
        # args.entropy_model_path = 'D:\MXH\STPM\CompressAI\STPM4Video\models\checkpoint_epoch4.pth.tar'
        print("Loading Entropy Model! ", args.entropy_model_path)
        checkpoint = torch.load(args.entropy_model_path, map_location=device)
        stem.load_state_dict(checkpoint["state_dict"])
        last_epoch = checkpoint["epoch"] + 1
        last_iterations = checkpoint["iterations"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # Dataset
    train_loader = get_loader('train', args.dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              cropsize=args.crop_size)
    test_loader = get_loader('test', args.dataset, 1, shuffle=True, num_workers=args.num_workers,
                             cropsize=args.crop_size)

    # Loss
    criterion = EMLoss()

    # Tensorboard
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))

    IFrameCompressor.train()
    stem.train()

    train(args, IFrameCompressor, stem, train_loader, test_loader, optimizer, aux_optimizer, lr_scheduler, criterion,
          device, last_iterations, last_epoch, train_writer)


def train(args, IFrameCompressor, stem, train_loader, test_loader, optimizer, aux_optimizer, lr_scheduler, criterion,
          device, last_iterations, last_epoch, train_writer):
    '''
    condition on last received quantized y
    '''
    global y_condition
    bestEvalLoss = float('inf')
    iterations = last_iterations
    for epoch in range(last_epoch, args.epochs):
        # train loop
        for i, images in enumerate(train_loader):
            rand = random.random()
            if rand <= 0.25:
                images = images[0:7:2]  # 1,3,5,7 # *3
            elif rand <= 0.50:
                images = images[0:7:3]  # 1,4,7 # *2
            elif rand <= 0.75:
                images = images[0:7:6]  # 1,7 # *1
            # else # 1,2,3,4,5,6,7 # *6
            images = [img_.to(device) for img_ in images]
            if (i % 100 == 0):
                print(f"Train epoch {epoch}: ["
                      f"{i * args.batch_size}/{len(train_loader.dataset)}"
                      f" ({100. * i / len(train_loader):.3f}%)]"
                      f" iterations:{iterations}")


            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            for imgidx in range(len(images)):

                if imgidx == 0:
                    # I frame compression.

                    _, y_condition = IFrameCompressor.getY(images[imgidx])

                else:
                    # P frame compression
                    optimizer.zero_grad()
                    aux_optimizer.zero_grad()

                    y_cur, _ = IFrameCompressor.getY(images[imgidx])

                    stem_out = stem(y_cur, y_condition.detach())
                    y_condition = stem_out['y_hat']

                    out_criterion = criterion(stem_out, images[imgidx])
                    out_criterion["loss"].backward()
                    if args.clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(stem.parameters(), args.clip_max_norm)  # mxh add
                    optimizer.step()
                    aux_loss = stem.aux_loss()
                    aux_loss.backward()
                    aux_optimizer.step()

                    if (iterations % 100 == 0):
                        train_writer.add_scalar(f"bpp/Pframe_bpp", out_criterion["loss"], iterations)
                        train_writer.add_scalar(f"bpp/y_bpp", out_criterion["y_bpp_loss"], iterations)
                        train_writer.add_scalar(f"bpp/z_bpp", out_criterion["z_bpp_loss"], iterations)
                        train_writer.add_scalar("aux_loss", aux_loss, iterations)

                    iterations += 1

                    if (iterations % 20000 == 0):
                        # eval loop
                        print("evaluating...")
                        eval_loss = validation(IFrameCompressor, stem, test_loader, criterion, lr_scheduler, device)
                        print(f"epoch:{epoch},  iterations:{iterations},  eval_loss:{eval_loss:.4f},  "
                              f"Learning rate: {optimizer.param_groups[0]['lr']}")
                        train_writer.add_scalar("eval_loss", eval_loss, iterations)
                        if eval_loss < bestEvalLoss:
                            print(f"Saving best eval checkpoint epoch:{epoch}!!")
                            bestEvalLoss = eval_loss
                            save_checkpoint(
                                {
                                    "epoch": epoch,
                                    "iterations": iterations,
                                    "state_dict": stem.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "aux_optimizer": aux_optimizer.state_dict(),
                                    "lr_scheduler": lr_scheduler.state_dict(),
                                },
                                filename=args.model_save + f"checkpoint_best_epoch{epoch}.pth.tar"
                            )

        if epoch % 2 == 0:
            print(f"saving models at epoch{epoch}......")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "iterations": iterations,
                    "state_dict": stem.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=args.model_save + f"checkpoint_epoch{epoch}.pth.tar"
            )


def validation(IFrameCompressor, stem, test_loader, criterion, lr_scheduler, device):
    IFrameCompressor.eval()
    stem.eval()
    eval_loss = 0
    cnt = 0
    for i, images in enumerate(test_loader):
        if str(device) == 'cuda':
            images = [img_.cuda() for img_ in images]
        with torch.no_grad():
            for imgidx in range(len(images)):
                if imgidx == 0:
                    # I frame compression.
                    _, y_condition = IFrameCompressor.getY(images[imgidx])
                else:
                    # P frame compression
                    y_cur, _ = IFrameCompressor.getY(images[imgidx])

                    stem_out = stem(y_cur, y_condition) 
                    y_condition = stem_out['y_hat']

                    out_criterion = criterion(stem_out, images[imgidx])
                    eval_loss += out_criterion["loss"]
                    cnt = cnt + 1

    eval_loss = eval_loss / cnt
    lr_scheduler.step(eval_loss)

    IFrameCompressor.train()
    stem.train()

    return eval_loss




if __name__ == "__main__":
    train_singlerate(sys.argv[1:])