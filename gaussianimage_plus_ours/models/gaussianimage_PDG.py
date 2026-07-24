import sys
from pdg_gsplat import project_gaussians_uv, rasterize_gaussians_uv_sigmoid_k_beta
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
from models.utils import *


class GaussianImage_PDG(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.cur_num_points = self.init_num_points
        self.H, self.W = int(kwargs["H"]), int(kwargs["W"])
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )  #

        self.device = kwargs["device"]
        self.SLV = kwargs["args"].SLV_init

        self.iterations = kwargs["args"].iterations
        self.logwriter = kwargs['logwriter']

        w_init = torch.rand(self.init_num_points, 1, device=self.device) * self.W  # 0-w
        h_init = torch.rand(self.init_num_points, 1, device=self.device) * self.H  # 0-h
        self.means = nn.Parameter(torch.cat((w_init, h_init), dim=1))

        self.log_scales = nn.Parameter(torch.zeros(self.init_num_points, 2, device=self.device))
        self.log_scales.data[:] = math.log(2.0)
        self.thetas = nn.Parameter(torch.zeros(self.init_num_points, 1, device=self.device))
        self.sig_ks = nn.Parameter(torch.zeros(self.init_num_points, 2, device=self.device))
        self.betas = nn.Parameter(torch.zeros(self.init_num_points, 1, device=self.device))
        self.rgbs = nn.Parameter(torch.zeros(self.init_num_points, 3, device=self.device))

        if kwargs["opt_type"] == "adam":
            l = [
                {'params': [self.means], 'lr': kwargs["lr"], "name": "means"},
                {'params': [self.log_scales], 'lr': kwargs["lr"], "name": "log_scales"},
                {'params': [self.thetas], 'lr': kwargs["lr"], "name": "thetas"},
                {'params': [self.sig_ks], 'lr': kwargs["lr"], "name": "sig_ks"},
                {'params': [self.betas], 'lr': kwargs["lr"], "name": "betas"},
                {'params': [self.rgbs], 'lr': kwargs["lr"], "name": "rgbs"},
            ]

            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def forward(self, H=None, W=None):

        cov2d = scales_theta_to_cov2d(self.log_scales, self.thetas)
        extents, num_tiles_hit = project_gaussians_uv(cov2d, self.means, H, W, 16)

        betas = 1.0 + torch.exp(self.betas)
        out_img = rasterize_gaussians_uv_sigmoid_k_beta(self.means, self.log_scales, self.thetas, self.sig_ks, betas,
                                                        self.rgbs, extents, num_tiles_hit, H, W, 16)

        out_img_noclamp = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
        if out_img_noclamp.requires_grad:
            out_img_noclamp.retain_grad()
        out_img = clamp_ste(out_img_noclamp, 0, 1)
        # out_img = torch.clamp(out_img_noclamp, 0, 1)
        return {"render": out_img,
                "render_no_clamp": out_img_noclamp,
                }

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def train_iter(self, H, W, gt_image, isprint=False):
        render_pkg = self.forward(H=H, W=W)
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)

        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer_step()
        return loss, psnr, image.detach()


    def train_iter_with_gradaccum(self, H, W, gt_image, grad_accum, isprint=False):
        render_pkg = self.forward(H=H, W=W)
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)

        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
            grad2d = render_pkg["render_no_clamp"].grad.detach().abs().sum(dim=1)
            grad_accum += grad2d

        self.optimizer_step()
        return loss, psnr, image.detach(), grad_accum

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_means, new_rgbs):

        new_num_points = new_means.shape[0]

        new_log_scales = torch.zeros(new_num_points, 2, device=self.device)
        new_log_scales.data[:] = math.log(2.0)
        new_thetas = torch.zeros(new_num_points, 1, device=self.device)
        new_sig_ks = torch.zeros(new_num_points, 2, device=self.device)
        new_betas = torch.zeros(new_num_points, 1, device=self.device)

        d = {"means": new_means,
             "rgbs": new_rgbs,
             "log_scales": new_log_scales,
             "thetas": new_thetas,
             "sig_ks": new_sig_ks,
             "betas": new_betas,
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.means = optimizable_tensors["means"]
        self.rgbs = optimizable_tensors["rgbs"]
        self.log_scales = optimizable_tensors["log_scales"]
        self.thetas = optimizable_tensors["thetas"]
        self.sig_ks = optimizable_tensors["sig_ks"]
        self.betas = optimizable_tensors["betas"]

        new_num_points = self.means.shape[0]
        self.cur_num_points = new_num_points

        return new_num_points, 0


