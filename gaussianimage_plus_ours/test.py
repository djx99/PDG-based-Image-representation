import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim,ssim
import sys
from PIL import Image, ImageOps
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import wandb
import copy
import json
from models.utils import loss_fn
from models.gaussianimage_PDG import GaussianImage_PDG
from models.gaussianimage_covariance import GaussianImage_Covariance



class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""

    def __init__(
            self,
            image_path: Path,
            log_dir: str,
            num_points: int = 2000,
            iterations: int = 30000,
            args=None,
    ):
        self.device = torch.device("cuda:0")
        gt_image = image_path_to_tensor(image_path)

        self.gt_image = gt_image.to(self.device)

        self.args = args
        self.num_points = num_points
        self.max_num_points = args.max_num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.loss_type = args.loss_type

        self.add_stage = 0
        self.log_dir = Path(os.path.join(log_dir, self.image_name))
        self.print = args.print
        self.resume = False
        self.logwriter = LogWriter(self.log_dir)

        model_path = os.path.join(log_dir, self.image_name, 'gaussian_model.pth.tar')

        self.logwriter.write(f"loading model path:{model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.num_points = checkpoint['num_gs']

        if args.method == 'Gaussian':
            self.gaussian_model = GaussianImage_Covariance(loss_type=self.loss_type, opt_type=args.opt_type,
                                                    num_points=self.num_points, H=self.H, W=self.W,
                                                    BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                                    device=self.device, lr=args.lr, quantize=args.quantize,
                                                    args=args, logwriter=self.logwriter).to(self.device)
            self.gaussian_model.cholesky_bound = checkpoint['slv_bound']
        else:
            self.gaussian_model = GaussianImage_PDG(loss_type=self.loss_type, opt_type=args.opt_type,
                                                       num_points=self.num_points, H=self.H, W=self.W,
                                                       BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                                       device=self.device, lr=args.lr, quantize=args.quantize,
                                                       args=args, logwriter=self.logwriter).to(self.device)
        model_dict = self.gaussian_model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['gs'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.gaussian_model.load_state_dict(model_dict)
        self.resume = True


    def test(self):

        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model(H=self.H, W=self.W)

        transform = transforms.ToPILImage()
        img = transform(out["render"].float().squeeze(0))
        name = self.image_name + "_fitting.png"
        img.save(str(self.log_dir / name))

        transform = transforms.ToPILImage()
        img = transform(self.gt_image.squeeze(0))
        name = self.image_name + "_gt.png"
        img.save(str(self.log_dir / name))

        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        return psnr, ms_ssim_value


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )

    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--prune_iter", type=int, default=100, help="iteration of each pruning  (default: %(default)s)"
    )
    parser.add_argument(
        "--grow_iter", type=int, default=5000, help="iteration of each growing (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Covariance",
        help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )

    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument("--num_points", type=int, default=2500, help="2D GS points (default: %(default)s)")
    parser.add_argument("--max_num_points", type=int, default=5000, help="max 2D GS points (default: %(default)s)", )
    parser.add_argument("--opt_type", type=str, default="adam", help="the type of optimizer")
    parser.add_argument("-opt", "--opt_nums", type=int, default=1, help="the nums of optimizer")
    parser.add_argument("--seed", type=int, default=3047, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--print", type=bool, default=False, help="if need print details")
    parser.add_argument("--lr", type=float, default=0.018,help="Learning rate (default: %(default)s)")
    parser.add_argument("--warmup_iter", type=float, default=15000)
    parser.add_argument('--radius_clip', type=float, default=1.0)
    parser.add_argument("--prune", type=bool, default=True,  help="turn on pruning")
    parser.add_argument("--adaptive_add", type=bool, default=True, help="turn on adaptive add densification")
    parser.add_argument("--wandb-project", type=str, default=None, help='Weights & Biases Project')
    parser.add_argument("--loss_type", type=str, default="L2")
    parser.add_argument("--SLV_init", type=bool, default=True, help="if turn on CAF filter")
    parser.add_argument("--color_norm",action='store_true')
    parser.add_argument("--coords_norm", action='store_true', help="if normalize the coordinates")
    parser.add_argument("--coords_act", type=str, default="tanh", help="tanh")
    parser.add_argument("--save_interval", type=int, default=5, help="save interval")
    parser.add_argument("--clip_coe", type=float, default=3.)
    #  quantization parameters =======================================
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize")
    parser.add_argument("--cov_quant", type=str, default="lsq", help="type of covariance quantization")
    parser.add_argument("--color_quant", type=str, default="lsq")
    parser.add_argument("--xy_quant", type=str, default="lsq")
    parser.add_argument("--xy_bit", type=int, default=12, help="bitdepth of xy attri")
    parser.add_argument("--cov_bit", type=int, default=10, help="bitdepth of cov attri")
    parser.add_argument("--color_bit", type=int, default=6, help="bitdepth of color attri")

    args = parser.parse_args(argv)
    return args


def main(args):

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    log_dir = (
            f"./checkpoint/{args.method}/{args.data_name}/{args.model_name}_M{args.max_num_points}_N{args.num_points}{'_SLV' if args.SLV_init else ''}_R{args.radius_clip}{'_add' if args.adaptive_add else ''}" +
            f"{'_prune' if args.prune else ''}{'_colornorm' if args.color_norm else ''}"
    )
    logwriter = LogWriter(Path(log_dir), train=False)

    script_name = os.path.basename(__file__)
    logwriter.write(script_name)
    logwriter.write(args_text)

    psnrs, ms_ssims, gs_nums, params = [], [], [], []

    dataset_dir = Path(args.dataset)
    image_paths = sorted([p for p in dataset_dir.iterdir() if p.is_file()])
    image_h, image_w = 0, 0
    image_length = len(image_paths)
    for image_path in image_paths:
        print(str(image_path))

        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points,
                                  iterations=args.iterations,  args=args,
                                  log_dir=log_dir)

        #  ===========overfiting training=================

        psnr, ms_ssim_v = trainer.test()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim_v)

        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        finally_gs_nums = trainer.gaussian_model.cur_num_points
        finally_params = sum([p.numel() for p in trainer.gaussian_model.parameters() if p.requires_grad])
        gs_nums.append(finally_gs_nums)
        params.append(finally_params / 1e6)
        logwriter.write(
            "{}\t{}x{}\tPSNR\t{:.4f}\tMS-SSIM\t{:.4f}\tgs_nums\t{:.2e}\tParams(M)\t{:.2f}".format(
                image_name, trainer.H, trainer.W, psnr, ms_ssim_v, finally_gs_nums, finally_params / 1e6))

        # representation recording===========
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_h = image_h // image_length
    avg_w = image_w // image_length
    avg_gs_nums = sum(gs_nums) / image_length
    avg_params = sum(params) / image_length

    logwriter.write(
        "Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, gs_nums:{:.2e}, Params(M):{:.2f}".format(
            avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_gs_nums,
            avg_params))


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = parse_args(argv)

    for method in ['PDG+STE', 'Gaussian', 'PDG', ]:
        args.method = method
        for num in [10000, 50000]:
            args.num_points = int(num / 2)
            args.max_num_points = num
            args.data_name = 'kodak'
            args.dataset = './datasets/kodak/'
            main(args)

        for num in [100000, 500000]:
            args.num_points = int(num / 2)
            args.max_num_points = num
            args.data_name = 'ImageGS_texture'
            args.dataset = './datasets/ImageGS_texture/'
            main(args)

        for num in [100000, 500000]:
            args.num_points = int(num / 2)
            args.max_num_points = num
            args.data_name = 'DIV2K_valid_HR'
            args.dataset = './datasets/DIV2K_valid_HR/'
            main(args)
