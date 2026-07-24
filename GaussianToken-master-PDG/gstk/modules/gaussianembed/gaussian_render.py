import torch
from pdg_gsplat import project_gaussians_uv, rasterize_gaussians_uv_sigmoid_k_beta

from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum

from dataclasses import dataclass


def scales_theta_to_cov2d(log_scales, theta, scale=1):
    # sigma > 0
    sigmas = torch.exp(log_scales) * scale  # (N,2) -> (sx, sy)
    sx2 = sigmas[:, 0] ** 2
    sy2 = sigmas[:, 1] ** 2

    th = theta[:, 0]  # (N,)
    c = torch.cos(th)
    s = torch.sin(th)

    # Sigma = R diag(sx^2, sy^2) R^T
    # a = c^2*sx2 + s^2*sy2
    # b = c*s*(sx2 - sy2)
    # c_ = s^2*sx2 + c^2*sy2
    a = c * c * sx2 + s * s * sy2
    b = c * s * (sx2 - sy2)
    c_ = s * s * sx2 + c * c * sy2

    cov2d = torch.stack([a, b, c_], dim=1)  # (N,3)
    return cov2d


@dataclass
class RenderSet:
    img_H: int = 256
    img_W: int = 256
    block_H: int = 16
    block_W: int = 16
    tile_bounds: tuple = None
    background: torch.Tensor = None
    
    def post_set(self, feature_dim):
        self.tile_bounds = (
            (self.img_W + self.block_W - 1) // self.block_W,
            (self.img_H + self.block_H - 1) // self.block_H,
            1,
        )
        self.background = torch.zeros(feature_dim, dtype=torch.float).cuda()
        
        return self


def _gaussian2image(_xy, _scaling, _rotation, _feature, render_set: RenderSet, _opacity=None):
    _opacity = torch.ones(_xy.shape[0], 1, dtype=torch.float, device=_xy.device) if _opacity is None or _opacity.shape[-1]==0 else _opacity

    # xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(_xy, _scaling, _rotation, render_set.img_H, render_set.img_W, render_set.tile_bounds)
    # out_img = rasterize_gaussians_sum(xys, depths, radii, conics, num_tiles_hit, _feature, _opacity, render_set.img_H, render_set.img_W, render_set.block_H, render_set.block_W, background=render_set.background, return_alpha=False)
    # out_img = out_img.permute(2, 0, 1).contiguous()

    means = torch.stack(
        (_xy[:, 0] * render_set.img_W, _xy[:, 1] * render_set.img_H),
        dim=-1
    )
    cov2d = scales_theta_to_cov2d(_scaling, _rotation)
    extents, num_tiles_hit = project_gaussians_uv(cov2d, means, render_set.img_H, render_set.img_W, 16)

    betas = 1.0 + torch.exp(_opacity[:, 0:1])
    sig_ks = _opacity[:, 1:]
    # print(sig_ks.shape)
    # print(betas.shape)
    out_img = rasterize_gaussians_uv_sigmoid_k_beta(means, _scaling, _rotation, sig_ks, betas, _feature, extents, num_tiles_hit, render_set.img_H, render_set.img_W, 16)
    # print(out_img.shape)
    # print(render_set.img_H, render_set.img_W)
    # print(_feature.shape)
    # out_img = out_img[..., :3]
    out_img = out_img.permute(2, 0, 1).contiguous()

    return out_img


def render_gaussians(gaussians, render_set: RenderSet):
    img_batch = []
    for mean, scale, rot, feature, opacity in zip(gaussians.means, gaussians.scales, gaussians.rotations, gaussians.features, gaussians.opacities):
        img = _gaussian2image(mean, scale, rot, feature, render_set, opacity)
        img_batch.append(img)
        
    img_batch = torch.stack(img_batch)
    
    return img_batch
