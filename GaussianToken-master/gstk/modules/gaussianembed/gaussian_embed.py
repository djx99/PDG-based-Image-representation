import torch
import torch.nn as nn
import copy

from .lifter_module import build_anchor_encoder, build_fm_lifter, build_gaussian_lifter
from .deformable_module import build_attn_encoder, build_attn_decoder
from .ffn_module import build_ffn
from .refine_module import build_refine_module
from .spconv2d_module import build_spconv2d
from .misc import safe_sigmoid, safe_tanh


def _get_reference_points(anchor):
    reference_points = safe_sigmoid(anchor[..., :2]) # [0, 1]，最后输出时再缩放到[-1, 1]
    reference_points = reference_points.unsqueeze(2) # [bs, num_gs, 1, 2]
    return reference_points


class GaussianEmbed(nn.Module):
    def __init__(self, fm_lifter_cfg, gaussian_lifter_cfg, anchor_encoder_cfg, attn_encoder_cfg, attn_decoder_cfg, ffn_cfg, refine_cfg, spconv_cfg, operation_order, op_param_share=True):
        super().__init__()
        self.fm_lifter = build_fm_lifter(fm_lifter_cfg)
        self.gaussian_lifter = build_gaussian_lifter(gaussian_lifter_cfg)
        self.anchor_encoder = build_anchor_encoder(anchor_encoder_cfg)
        self.attn_encoder = build_attn_encoder(attn_encoder_cfg)
        
        self.operation_order = operation_order
        self.param_share = op_param_share
        
        self.op_module_map = nn.ModuleDict({
            "cross_attn": build_attn_decoder(attn_decoder_cfg),
            "ffn": build_ffn(ffn_cfg),
            "refine": build_refine_module(refine_cfg),
            "spconv": build_spconv2d(spconv_cfg),
        })
        
        if self.param_share:
            get_op_module = lambda op: self.op_module_map[op]
        else:
            get_op_module = lambda op: copy.deepcopy(self.op_module_map[op])

        self.layers = nn.ModuleList(
            [get_op_module(op) for op in self.operation_order]
        )
        
    def forward(self, feature_map):
        src, pos = self.fm_lifter(feature_map) # (1) 特征图扁平化 + 位置编码
        
        instance_feature, anchor = self.gaussian_lifter(src)  # (2) 初始化 K 个 anchor + K 个 query/instance_feature
        anchor_embed = self.anchor_encoder(anchor) # (3) 把 anchor 参数编码成 embedding
        
        memory = self.attn_encoder(src, pos) # MSDA not support fp16 # (4) 编码 feature_map（Deformable Attn）
        
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, op in enumerate(self.operation_order): # (5) 交替更新 instance_feature 和 anchor
                if op == "ffn":
                    instance_feature = self.layers[i](instance_feature)
                elif op == "cross_attn":
                    instance_feature = self.layers[i](instance_feature, anchor_embed, memory, \
                        _get_reference_points(anchor))
                elif op == "refine":
                    anchor = self.layers[i](instance_feature, anchor_embed, anchor)
                    if i != len(self.operation_order) - 1:
                        anchor_embed = self.anchor_encoder(anchor)
                elif op == "spconv":
                    instance_feature = self.layers[i](instance_feature, anchor)
        
        gaussian = self.layers[-1].get_gaussian(anchor) # (6) 把 anchor（logit 空间）转换成实际高斯参数（坐标/尺度/旋转/…）
        
        return  gaussian
    

def build_gs_embed(cfg):
    gs_embed = GaussianEmbed(**cfg)
    
    return gs_embed