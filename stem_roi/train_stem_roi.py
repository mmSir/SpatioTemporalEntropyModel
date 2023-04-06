import argparse
import random
import sys
import os

from torch.utils.tensorboard import SummaryWriter

from compressai.zoo import models
from compressai.models.stem_roi import *

from stem.dataset_vidseq import get_loader
from stem_roi.stem_roi_dataset import get_loader_roi
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
        "--tensorboard-runs", type=str, default="D:\MXH\STPM\CompressAI\STPM4Video", help="Tensorboard Path"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=300,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        # default=1e-5,
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
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
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
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
    parser.add_argument("--checkpoint", type=str, help="Path to a I frame compressor checkpoint")
    parser.add_argument(
        "-em",
        "--entropy-model-path",
        type=str,
        help="P Frame Entropy Model architecture (default: %(default)s)",
    )
    parser.add_argument("--eval-loss", type=float, default=None, help="last eval loss")
    parser.add_argument("--model-save", type=str, help="Path to save a checkpoint")
    args = parser.parse_args(argv)
    return args



def train_stem_baseline(argv):
    '''
    train single rate model.
    Jointly train I model & P model.
    Transform Network is jointly trained with stem.
    '''
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # args.model = "cheng2020-attn"
    args.model = "mbt2018-mean"
    print(f"ImageCompressor model: {args.model}")
    ImageCompressor = models[args.model](quality=4)  # quality决定了NM的值 N=128 M=192
    ImageCompressor = ImageCompressor.to(device)
    # stem = stem_baseline()
    stem = stem_baselinev2()
    stem = stem.to(device)


    optimizer_p, aux_optimizer_p = configure_optimizers(stem, args)
    optimizer_i, aux_optimizer_i = configure_optimizers(ImageCompressor, args)
    # optimizer, aux_optimizer = configure_optimizers2(stem, args)  # 仅设置stem的优化器, aux_op包含两个
    # optimizer, aux_optimizer = configure_optimizers_plus8(stem, args)  # 仅设置stem的优化器, aux_op包含两个
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.2)  # 连续5次eval没有下降，lr -> lr*0.2

    last_epoch = 0
    last_iterations = 0
    if args.checkpoint:  # load from previous checkpoint
        # args.checkpoint = "D:\MXH\STPM\CompressAI\\trainmeanscale\Cheng2020\Attention\lmbda1e-2\ChengAttn_best_loss.pth.tar"
        # args.checkpoint = 'D:\MXH\STPM\CompressAI\\trainmeanscale\MSHyperPrior\lmbda1e-2\MSHyperPrior_checkpoint_openimages_epoch100.pth.tar'
        # args.checkpoint = 'D:\MXH\STPM\CompressAI\\trainmeanscale\MSHyperPrior\lmbda3e-3\MSHyperPrior_checkpoint_openimages_epoch50.pth.tar'
        args.checkpoint = 'D:\MXH\STPM\CompressAI\STPM4Video\models\iframemodel_checkpoint_epoch37.pth.tar'
        # args.checkpoint = 'D:\MXH\STPM\CompressAI\STPM4Video\stem_roi\\baseline_single_rate_model\lmbda10e-3\iframemodel_checkpoint_best_epoch72.pth.tar'
        # args.checkpoint = 'D:\MXH\STPM\CompressAI\STPM4Video\stem_roi\\baseline_single_rate_model\lmbda31e-3\iframemodel_checkpoint_best_epoch72.pth.tar'
        # args.checkpoint = 'D:\MXH\STPM\CompressAI\\trainmeanscale\AutoregressiveMSHyperPrior\lmbda1e-2\ARMSHyperPrior_checkpoint_openimages_epoch50.pth.tar'
        print("Loading ImageCompressor", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # ImageCompressor.load_state_dict(checkpoint)
        ImageCompressor.load_state_dict(checkpoint["state_dict"])
        optimizer_i.load_state_dict(checkpoint["optimizer"])
        aux_optimizer_i.load_state_dict(checkpoint["aux_optimizer"])
    if args.entropy_model_path:
        # args.entropy_model_path = 'D:\MXH\STPM\CompressAI\STPM4Video\stem_roi\\baseline_single_rate_model\lmbda10e-3\stem_checkpoint_best_epoch72.pth.tar'
        args.entropy_model_path = 'D:\MXH\STPM\CompressAI\STPM4Video\models\stem_checkpoint_epoch37.pth.tar'
        print("Loading Entropy Model! ", args.entropy_model_path)
        checkpoint = torch.load(args.entropy_model_path, map_location=device)
        stem.load_state_dict(checkpoint["state_dict"])
        last_epoch = checkpoint["epoch"] + 1
        last_iterations = checkpoint["iterations"]
        optimizer_p.load_state_dict(checkpoint["optimizer"])
        aux_optimizer_p.load_state_dict(checkpoint["aux_optimizer"])
        args.eval_loss = checkpoint["eval_loss"]
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    args.learning_rate = 1e-4
    print(f"configure optimizer learning rate: {args.learning_rate}")
    optimizer_p.param_groups[0]['lr'] = args.learning_rate
    optimizer_i.param_groups[0]['lr'] = args.learning_rate

    # Dataset
    train_loader = get_loader('train', args.dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              cropsize=args.crop_size) # train
    test_loader = get_loader('test', args.dataset, 1, shuffle=True, num_workers=args.num_workers,
                             cropsize=args.crop_size)

    # Loss
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    print(f"Loss = {args.lmbda} * 255 * 255 * D + R, lmbda = {args.lmbda}")

    # Tensorboard
    train_writer = SummaryWriter(args.model_save)

    ImageCompressor.train()
    stem.train()

    _train_stem_baseline(args, ImageCompressor, stem, train_loader, test_loader, optimizer_p, aux_optimizer_p, optimizer_i, aux_optimizer_i,
                     criterion, device, last_iterations, last_epoch, train_writer)


def _train_stem_baseline(args, ImageCompressor, stem, train_loader, test_loader, optimizer_p, aux_optimizer_p, optimizer_i, aux_optimizer_i,
                     criterion, device, last_iterations, last_epoch, train_writer):
    '''
    condition on last received quantized y
    '''
    global y_condition, x_conditioned
    eval_loss = float('inf') if args.eval_loss is None else args.eval_loss
    bestEvalLoss = float('inf')
    out_criterion = {"mse_loss": 0, "bpp_loss": 0, "loss": 0}
    iterations = last_iterations
    for epoch in range(last_epoch, args.epochs):
        # train loop
        for i, images in enumerate(train_loader):
            # 只取其中一部分，以学习不同运动大小，也是为了加速训练
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
                      f" iterations:{iterations} loss:{out_criterion['loss']:.4f}"
                      f" mse_loss:{out_criterion['mse_loss']:.4f} bpp_loss:{out_criterion['bpp_loss']:.4f}")

            # STPMLoss = torch.tensor(0.0, requires_grad=True)
            optimizer_i.zero_grad()
            aux_optimizer_i.zero_grad()
            optimizer_p.zero_grad()
            aux_optimizer_p.zero_grad()

            for imgidx in range(len(images)):
                # curImg = images[imgidx] #当前要压缩的图像
                if imgidx == 0:
                    # I frame compression.
                    optimizer_i.zero_grad()
                    aux_optimizer_i.zero_grad()

                    iframemodel_out = ImageCompressor(images[imgidx])
                    x_conditioned = iframemodel_out["x_hat"]
                    out_criterion_i = criterion(iframemodel_out, images[imgidx])
                    out_criterion_i["loss"].backward()
                    if args.clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(ImageCompressor.parameters(), args.clip_max_norm)  # mxh add
                    optimizer_i.step()
                    aux_loss_i = ImageCompressor.aux_loss()
                    aux_loss_i.backward()
                    aux_optimizer_i.step()

                    # _, y_condition = ImageCompressor.getY(images[imgidx])  # 这个y_condition在解码端需要保存下来
                    # y_condition 在训练时是noise,其他时候是dequantize

                else:
                    # P frame compression
                    optimizer_p.zero_grad()
                    aux_optimizer_p.zero_grad()

                    stem_out = stem(images[imgidx], x_conditioned.detach())
                    # y_condition = stem_out['y_hat'] # 可以是这个y_hat，也可以是x_recon再经过PEncoder
                    x_conditioned = stem_out['x_hat']


                    out_criterion = criterion(stem_out, images[imgidx])
                    # out_criterion["mse_loss"].backward() #iteration or epoch为判断依据选择是只优化D还是联合RD优化
                    out_criterion["loss"].backward()
                    if args.clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(stem.parameters(), args.clip_max_norm)  # mxh add
                    optimizer_p.step()
                    aux_loss = stem.aux_loss()
                    aux_loss.backward()
                    aux_optimizer_p.step()

                    if (iterations % 100 == 0):
                        # train_writer.add_scalar("bpp/Iframe_bpp", I_bpp, iterations)
                        train_writer.add_scalar(f"TrainLoss/Total_loss_i", out_criterion_i["loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/bpp_loss_i", out_criterion_i["bpp_loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/mse_loss_i", out_criterion_i["mse_loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/aux_loss_i", aux_loss_i, iterations)
                        train_writer.add_scalar(f"TrainLoss/Total_loss_p", out_criterion["loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/bpp_loss_p", out_criterion["bpp_loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/mse_loss_p", out_criterion["mse_loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/aux_loss_p", aux_loss, iterations)
                        # train_writer.add_scalar("mse", out_criterion["mse_loss"], iterations)
                        # train_writer.add_scalar("total_loss", out_criterion["loss"], iterations)

                    iterations += 1

                    if (iterations % 20000 == 0):
                        # eval loop
                        print("evaluating...")
                        eval_loss = _validation_stem_baseline(ImageCompressor, stem, test_loader, criterion, device)
                        eval_totalloss = eval_loss["loss"]
                        eval_bpploss = eval_loss["bpp_loss"]
                        eval_mseloss = eval_loss["mse_loss"]
                        print(
                            f"epoch:{epoch},  iterations:{iterations},  eval_loss:{eval_totalloss:.4f},  eval_bpploss:{eval_bpploss:.4f},  "
                            f"eval_mseloss:{eval_mseloss:.4f} (psnr:{-10 * math.log10(eval_mseloss):.4f}dB)  Learning rate: {optimizer_p.param_groups[0]['lr']}")
                        train_writer.add_scalar("EvalLoss/total_loss", eval_totalloss, iterations)
                        train_writer.add_scalar("EvalLoss/bpploss", eval_bpploss, iterations)
                        train_writer.add_scalar("EvalLoss/mseloss", eval_mseloss, iterations)
                        if eval_totalloss < bestEvalLoss:
                            print(f"Saving best eval checkpoint epoch:{epoch}!!")
                            bestEvalLoss = eval_totalloss
                            save_checkpoint(
                                {
                                    "epoch": epoch,
                                    "iterations": iterations,
                                    "state_dict": stem.state_dict(),
                                    "optimizer": optimizer_p.state_dict(),
                                    "aux_optimizer": aux_optimizer_p.state_dict(),
                                    "eval_loss": eval_loss,
                                },
                                filename=args.model_save + f"stem_checkpoint_best_epoch{epoch}.pth.tar"
                            )
                            save_checkpoint(
                                {
                                    "epoch": epoch,
                                    "iterations": iterations,
                                    "state_dict": ImageCompressor.state_dict(),
                                    "optimizer": optimizer_i.state_dict(),
                                    "aux_optimizer": aux_optimizer_i.state_dict(),
                                },
                                filename=args.model_save + f"iframemodel_checkpoint_best_epoch{epoch}.pth.tar"
                            )

        if epoch % 2 == 0:
            print(f"saving models at epoch{epoch}......")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "iterations": iterations,
                    "state_dict": stem.state_dict(),
                    "optimizer": optimizer_p.state_dict(),
                    "aux_optimizer": aux_optimizer_p.state_dict(),
                    "eval_loss": eval_loss,
                },
                filename=args.model_save + f"stem_checkpoint_epoch{epoch}.pth.tar"
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "iterations": iterations,
                    "state_dict": ImageCompressor.state_dict(),
                    "optimizer": optimizer_i.state_dict(),
                    "aux_optimizer": aux_optimizer_i.state_dict(),
                },
                filename=args.model_save + f"iframemodel_checkpoint_epoch{epoch}.pth.tar"
            )

@torch.no_grad()
def _validation_stem_baseline(ImageCompressor, stem, test_loader, criterion, device):
    ImageCompressor.eval()
    stem.eval()
    eval_loss = {}
    eval_loss["loss"] = 0
    eval_loss["mse_loss"] = 0
    eval_loss["bpp_loss"] = 0
    cnt = 0
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            if str(device) == 'cuda':  # 判断内容是否相同用 == ； 判断是否是同一个对象用 is
                images = [img_.cuda() for img_ in images]
            with torch.no_grad():
                for imgidx in range(len(images)):
                    if imgidx == 0:
                        # I frame compression.
                        iframemodel_out = ImageCompressor(images[imgidx])
                        x_conditioned = iframemodel_out["x_hat"]
                        out_criterion_i = criterion(iframemodel_out, images[imgidx])
                        eval_loss["loss"] += out_criterion_i["loss"]
                        eval_loss["mse_loss"] += out_criterion_i["mse_loss"]
                        eval_loss["bpp_loss"] += out_criterion_i["bpp_loss"]
                        cnt = cnt + 1
                    else:
                        # P frame compression
                        stem_out = stem(images[imgidx], x_conditioned)
                        # y_condition = stem_out['y_hat']  # 可以是这个y_hat，也可以是x_recon再经过PEncoder
                        x_conditioned = stem_out['x_hat']

                        out_criterion = criterion(stem_out, images[imgidx])
                        eval_loss["loss"] += out_criterion["loss"]
                        eval_loss["mse_loss"] += out_criterion["mse_loss"]
                        eval_loss["bpp_loss"] += out_criterion["bpp_loss"]
                        cnt = cnt + 1

    eval_loss["loss"] = eval_loss["loss"] / cnt
    eval_loss["mse_loss"] = eval_loss["mse_loss"] / cnt
    eval_loss["bpp_loss"] = eval_loss["bpp_loss"] / cnt
    # lr_scheduler.step(eval_loss["loss"])

    ImageCompressor.train()
    stem.train()

    return eval_loss

#---------------------------------------------------------------------------------------------#
def train_stem_roi(argv):
    '''
    Jointly train I model(stem_roi_i) & P model(stem_roi).
    '''
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # args.model = "cheng2020-attn"
    # args.model = "mbt2018-mean"
    # print(f"ImageCompressor model: {args.model}")
    # ImageCompressor = models[args.model](quality=4)  # quality决定了NM的值 N=128 M=192
    print(f"ImageCompressor model: stem_roi_i")
    ImageCompressor = stem_roi_i()
    ImageCompressor = ImageCompressor.to(device)
    stem = stem_roi()
    # stem = stem_roi_wo_gsc()
    stem = stem.to(device)


    optimizer_p, aux_optimizer_p = configure_optimizers(stem, args)
    optimizer_i, aux_optimizer_i = configure_optimizers(ImageCompressor, args)

    last_epoch = 0
    last_iterations = 0
    if args.checkpoint:  # load i frame compressor
        # args.checkpoint = 'D:\MXH\STPM\CompressAI\STPM4Video\models\iframemodel_checkpoint_best_epoch41.pth.tar'
        # args.checkpoint = 'D:\MXH\STPM\CompressAI\STPM4Video\stem_roi\\baseline_single_rate_model\lmbda10e-3\iframemodel_checkpoint_best_epoch72.pth.tar'
        print("Loading ImageCompressor!", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # ImageCompressor.load_state_dict(checkpoint)
        ImageCompressor.load_state_dict(checkpoint["state_dict"])
        optimizer_i.load_state_dict(checkpoint["optimizer"])
        aux_optimizer_i.load_state_dict(checkpoint["aux_optimizer"])
    if args.entropy_model_path:
        # args.entropy_model_path = 'D:\MXH\STPM\CompressAI\STPM4Video\stem_roi\\roi_v1\stem_checkpoint_best_epoch76.pth.tar'
        # args.entropy_model_path = 'D:\MXH\STPM\CompressAI\STPM4Video\models\stem_checkpoint_best_epoch41.pth.tar'
        print("Loading Entropy Model! ", args.entropy_model_path)
        checkpoint = torch.load(args.entropy_model_path, map_location=device)
        stem.load_state_dict(checkpoint["state_dict"])
        last_epoch = checkpoint["epoch"] + 1
        last_iterations = checkpoint["iterations"]
        optimizer_p.load_state_dict(checkpoint["optimizer"])
        aux_optimizer_p.load_state_dict(checkpoint["aux_optimizer"])
        args.eval_loss = checkpoint["eval_loss"]
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    args.learning_rate = 1e-4
    print(f"configure optimizer learning rate: {args.learning_rate}")
    optimizer_p.param_groups[0]['lr'] = args.learning_rate
    optimizer_i.param_groups[0]['lr'] = args.learning_rate

    # Dataset
    train_loader = get_loader_roi('train', args.dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, cropsize=args.crop_size) # train
    L = 4
    levels = [int(100 * (i / L)) for i in range(L + 1)]
    test_loaders = []
    for level in levels:
        test_loader = get_loader_roi('test', args.dataset, 1, shuffle=True, num_workers=args.num_workers, cropsize=args.crop_size, level=level)  # test
        test_loaders.append(test_loader)

    # Loss
    criterion = PixelwiseRateDistortionLoss()
    print(f"Loss = PixelwiseRateDistortionLoss(lmbda * D + R), lmbda = 0.001 * torch.exp(3.4431 * qmap)")
    lossAvger_i = MovingAverage(100)
    lossAvger_p = MovingAverage(100)

    # Tensorboard
    train_writer = SummaryWriter(args.model_save + 'tensorboard\\train\\')
    test_writer = []
    for i in range(len(levels)):
        test_writer.append(SummaryWriter(args.model_save + f'tensorboard\\test\\test_dataset_{i}'))

    ImageCompressor.train()
    stem.train()


    _train_stem_roi(args, ImageCompressor, stem, train_loader, test_loaders, optimizer_i, aux_optimizer_i, optimizer_p, aux_optimizer_p,
                    lossAvger_i, lossAvger_p, criterion, device, last_iterations, last_epoch, train_writer, test_writer)


def _train_stem_roi(args, ImageCompressor, stem, train_loader, test_loaders, optimizer_i, aux_optimizer_i, optimizer_p, aux_optimizer_p,
                    lossAvger_i, lossAvger_p, criterion, device, last_iterations, last_epoch, train_writer, test_writer):
    '''
    condition on last reconstruction.
    '''
    global y_condition, x_conditioned
    eval_loss = float('inf')
    bestEvalLoss = float('inf')
    out_criterion = {"mse_loss": 0, "bpp_loss": 0, "loss": 0}
    iterations = last_iterations
    for epoch in range(last_epoch, args.epochs):
        # train loop
        for i, (images, Qmap) in enumerate(train_loader):
            # 只取其中一部分，以学习不同运动大小，也是为了加速训练， todo: move this part of code to dataset
            rand = random.random()
            if rand <= 0.25:
                images = images[0:7:2]  # 1,3,5,7 # *3
            elif rand <= 0.50:
                images = images[0:7:3]  # 1,4,7 # *2
            elif rand <= 0.75:
                images = images[0:7:6]  # 1,7 # *1
            # else # 1,2,3,4,5,6,7 # *6
            images = [img_.to(device) for img_ in images]
            Qmap = Qmap.to(device)
            if (i % 100 == 0):
                print(f"Train epoch {epoch}: ["
                      f"{i * args.batch_size:05d}/{len(train_loader.dataset)}"
                      f" ({100. * i / len(train_loader):.3f}%)]"
                      f" iterations:{iterations} loss:{out_criterion['loss']:.4f}"
                      f" mse_loss:{out_criterion['mse_loss']:.7f} bpp_loss:{out_criterion['bpp_loss']:.4f}")

            optimizer_i.zero_grad()
            aux_optimizer_i.zero_grad()
            optimizer_p.zero_grad()
            aux_optimizer_p.zero_grad()
            lmbdamap = quality2lambda(Qmap)
            for imgidx in range(len(images)):
                if imgidx == 0:
                    # I frame compression.
                    # optimizer_i.zero_grad()
                    # aux_optimizer_i.zero_grad()

                    iframemodel_out = ImageCompressor(images[imgidx], Qmap)
                    x_conditioned = iframemodel_out["x_hat"]

                    out_criterion_i = criterion(iframemodel_out, images[imgidx], lmbdamap)

                    if out_criterion_i['loss'].isnan().any() or out_criterion_i['loss'].isinf().any() or out_criterion[
                        'loss'] > 3: # 一般Loss都在0.3~0.6之间,若Loss超过5倍，就认为是无效loss
                        print("i skip invalid loss!")
                        break  # continue或许会影响当前GoP接下去的帧,所以改成了break

                    # loss_avg = lossAvger_i.next(out_criterion_i["loss"].item())
                    # loss_scale = loss_avg / out_criterion_i["loss"].item()
                    # (loss_scale * out_criterion_i["loss"]).backward()
                    out_criterion_i["loss"].backward(retain_graph=True)
                    if args.clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(ImageCompressor.parameters(), args.clip_max_norm)  # mxh add
                    # optimizer_i.step()
                    aux_loss_i = ImageCompressor.aux_loss()
                    aux_loss_i.backward()
                    # aux_optimizer_i.step()

                else:
                    # P frame compression

                    # optimizer_p.zero_grad()
                    # aux_optimizer_p.zero_grad()

                    stem_out = stem(images[imgidx], x_conditioned, Qmap)
                    # stem_out = stem(images[imgidx], x_conditioned.detach(), Qmap)
                    x_conditioned = stem_out['x_hat']

                    out_criterion = criterion(stem_out, images[imgidx], lmbdamap)
                    # for stability
                    if out_criterion['loss'].isnan().any() or out_criterion['loss'].isinf().any() or out_criterion[
                        'loss'] > 3:
                        print("p skip invalid loss!")
                        break  # continue或许会影响当前GoP接下去的帧,所以改成了break

                    # loss_avg = lossAvger_p.next(out_criterion["loss"].item())
                    # loss_scale = loss_avg / out_criterion["loss"].item()
                    # (loss_scale * out_criterion["loss"]).backward()
                    out_criterion["loss"].backward(retain_graph=True)
                    if args.clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(stem.parameters(), args.clip_max_norm)  # mxh add
                    # optimizer_p.step()
                    aux_loss = stem.aux_loss()
                    aux_loss.backward()
                    # aux_optimizer_p.step()

                    if (iterations % 100 == 0):
                        # train_writer.add_scalar(f"TrainLoss/Total_loss_i", out_criterion_i["loss"], iterations)
                        # train_writer.add_scalar(f"TrainLoss/bpp_loss_i", out_criterion_i["bpp_loss"], iterations)
                        # train_writer.add_scalar(f"TrainLoss/mse_loss_i", out_criterion_i["mse_loss"], iterations)
                        # train_writer.add_scalar(f"TrainLoss/aux_loss_i", aux_loss_i, iterations)
                        train_writer.add_scalar(f"TrainLoss/Total_loss_p", out_criterion["loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/bpp_loss_p", out_criterion["bpp_loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/mse_loss_p", out_criterion["mse_loss"], iterations)
                        train_writer.add_scalar(f"TrainLoss/aux_loss_p", aux_loss, iterations)

                    iterations += 1

                    if (iterations % 10000 == 0):
                        # eval loop
                        print("evaluating...")
                        eval_loss_result = _validation_stem_roi(ImageCompressor, stem, test_loaders, criterion, device)
                        print(f"epoch:{epoch},  iterations:{iterations} evaluating result:")
                        eval_total_result = 0
                        cnt = 1
                        for eval_loss in eval_loss_result:
                            eval_totalloss = eval_loss["loss"]
                            eval_bpploss = eval_loss["bpp_loss"]
                            eval_mseloss = eval_loss["mse_loss"]
                            eval_psnr = -10 * math.log10(eval_loss["mse"])
                            print(
                                f"[{cnt}] eval_loss:{eval_totalloss:.4f},  eval_bpploss:{eval_bpploss:.4f},  "
                                f"eval_mseloss:{eval_mseloss:.6f}   eval_psnr:{eval_psnr:.4f}")
                            test_writer[cnt-1].add_scalar(f"EvalLoss/total_loss", eval_totalloss, iterations)
                            test_writer[cnt-1].add_scalar(f"EvalLoss/bpploss", eval_bpploss, iterations)
                            test_writer[cnt-1].add_scalar(f"EvalLoss/mseloss", eval_mseloss, iterations)
                            test_writer[cnt-1].add_scalar(f"EvalLoss/psnr", eval_psnr, iterations)
                            eval_total_result += eval_totalloss
                            cnt += 1
                        if eval_total_result < bestEvalLoss:
                            print(f"Saving best eval checkpoint epoch:{epoch}!!")
                            bestEvalLoss = eval_total_result
                            save_checkpoint(
                                {
                                    "epoch": epoch,
                                    "iterations": iterations,
                                    "state_dict": stem.state_dict(),
                                    "optimizer": optimizer_p.state_dict(),
                                    "aux_optimizer": aux_optimizer_p.state_dict(),
                                    "eval_loss": eval_loss,
                                },
                                filename=args.model_save + f"stem_checkpoint_best_epoch{epoch}.pth.tar"
                            )
                            save_checkpoint(
                                {
                                    "epoch": epoch,
                                    "iterations": iterations,
                                    "state_dict": ImageCompressor.state_dict(),
                                    "optimizer": optimizer_i.state_dict(),
                                    "aux_optimizer": aux_optimizer_i.state_dict(),
                                },
                                filename=args.model_save + f"iframemodel_checkpoint_best_epoch{epoch}.pth.tar"
                            )

            optimizer_i.step()
            aux_optimizer_i.step()
            optimizer_p.step()
            aux_optimizer_p.step()

        if epoch % 1 == 0:
            print(f"saving models at epoch{epoch}......")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "iterations": iterations,
                    "state_dict": stem.state_dict(),
                    "optimizer": optimizer_p.state_dict(),
                    "aux_optimizer": aux_optimizer_p.state_dict(),
                    "eval_loss": eval_loss,
                },
                filename=args.model_save + f"stem_checkpoint_epoch{epoch}.pth.tar"
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "iterations": iterations,
                    "state_dict": ImageCompressor.state_dict(),
                    "optimizer": optimizer_i.state_dict(),
                    "aux_optimizer": aux_optimizer_i.state_dict(),
                },
                filename=args.model_save + f"iframemodel_checkpoint_epoch{epoch}.pth.tar"
            )


@torch.no_grad()
def _validation_stem_roi(ImageCompressor, stem, test_loaders, criterion, device):
    ImageCompressor.eval()
    stem.eval()
    criterion_rd = RateDistortionLoss()
    MSE = nn.MSELoss(reduction='none')

    eval_loss_result = []
    for test_loader in test_loaders:
        eval_loss = {}
        eval_loss["loss"] = 0
        eval_loss["mse_loss"] = 0
        eval_loss["bpp_loss"] = 0
        eval_loss["mse"] = 0
        cnt = 0
        with torch.no_grad():
            for i, (images, Qmap) in enumerate(test_loader):
                if str(device) == 'cuda':  # 判断内容是否相同用 == ； 判断是否是同一个对象用 is
                    images = [img_.cuda() for img_ in images]
                    Qmap = Qmap.to(device)
                lmbdamap = quality2lambda(Qmap)
                with torch.no_grad():
                    for imgidx in range(len(images)):
                        if imgidx == 0:
                            # I frame compression.
                            iframemodel_out = ImageCompressor(images[imgidx], Qmap)
                            x_conditioned = iframemodel_out["x_hat"]
                            out_criterion_i = criterion(iframemodel_out, images[imgidx], lmbdamap)
                            out_criterion_aux = criterion_rd(iframemodel_out, images[imgidx])
                            eval_loss["loss"] += out_criterion_i["loss"]
                            eval_loss["mse"] += out_criterion_aux["mse_loss"]  # 没有乘lmbdamap的mse
                            eval_loss["mse_loss"] += out_criterion_i["mse_loss"]
                            eval_loss["bpp_loss"] += out_criterion_i["bpp_loss"]
                            cnt = cnt + 1
                        else:
                            # P frame compression
                            stem_out = stem(images[imgidx], x_conditioned, Qmap)
                            x_conditioned = stem_out['x_hat']

                            # lmbdamap = quality2lambda(Qmap)
                            out_criterion = criterion(stem_out, images[imgidx], lmbdamap)
                            out_criterion_aux = criterion_rd(stem_out, images[imgidx])
                            eval_loss["loss"] += out_criterion["loss"]
                            eval_loss["mse_loss"] += out_criterion["mse_loss"]
                            eval_loss["mse"] += out_criterion_aux["mse_loss"] # 没有乘lmbdamap的mse
                            eval_loss["bpp_loss"] += out_criterion["bpp_loss"]
                            cnt = cnt + 1

        eval_loss["loss"] = eval_loss["loss"] / cnt
        eval_loss["mse_loss"] = eval_loss["mse_loss"] / cnt
        eval_loss["bpp_loss"] = eval_loss["bpp_loss"] / cnt
        eval_loss["mse"] = eval_loss["mse"] / cnt
        # lr_scheduler.step(eval_loss["loss"])
        eval_loss_result.append(eval_loss)

    ImageCompressor.train()
    stem.train()

    return eval_loss_result





if __name__ == "__main__":
    # train_stem_baseline(sys.argv[1:])
    train_stem_roi(sys.argv[1:])