"""Baseline monocular depth-estimation training.

Implements two published depth-estimation architectures for comparison
against the LSTM-augmented model in ``train.py``:

  --model bts       BTS — "From Big to Small: Multi-Scale Local Planar
                    Guidance for Monocular Depth Estimation"
                    Lee, Han, Ko, Suh (2020).
                    https://arxiv.org/abs/1907.10326
                    Reference implementation:
                    https://github.com/cleinc/bts
                    Uses a DenseNet-161 encoder (paper-default) with ASPP
                    and Local Planar Guidance layers at strides 8/4/2.

  --model adabins   AdaBins — "AdaBins: Depth Estimation Using Adaptive Bins"
                    Bhat, Alhashim, Wonka (CVPR 2021).
                    https://arxiv.org/abs/2011.14141
                    Reference implementation:
                    https://github.com/shariqfarooq123/AdaBins
                    Uses an EfficientNet-B5 encoder-decoder (paper-default)
                    followed by a mini-ViT ("mViT") that predicts a set of
                    image-adaptive depth-bin widths plus per-pixel range
                    attention.

Both models are trained single-frame (no LSTM, no TBTT) with large-batch
fp16 autocast so we can saturate a 96 GB GPU.

Outputs are namespaced by model name:
  training_plots_<model>/training_curves.png
  training_snapshots_<model>/epoch_XXX/<scene>.mp4
  best_model_<model>.pt  /  last_model_<model>.pt

Loss: scale-invariant log loss (SILog, eq. 6 of BTS / AdaBins) which is the
canonical monocular-depth training objective.  We add an auxiliary L1 term
solely for human-readable reporting.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from collections import defaultdict
from glob import glob

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    DenseNet161_Weights,
    EfficientNet_B5_Weights,
    densenet161,
    efficientnet_b5,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


# ==========================================================================
# Dataset — frame-level (fully shuffleable)
# ==========================================================================
class FrameDepthDataset(Dataset):
    """Emits (rgb, depth_norm, valid_mask) per RGBD frame."""

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, data_dir, scene_names, img_size=416,
                 depth_scale=65535.0, augment=False):
        self.img_size = img_size
        self.depth_scale = depth_scale
        self.augment = augment
        files = []
        for scene in scene_names:
            files.extend(sorted(glob(os.path.join(data_dir, scene, "*-color.png"))))
        self.files = files
        self.scene_names = scene_names

    def __len__(self):
        return len(self.files)

    def _load(self, color_path):
        depth_path = color_path.replace("-color.png", "-depth.png")
        if not os.path.isfile(depth_path):
            depth_path = color_path.replace("-color.png", "-aligned-depth.png")
        bgr = cv2.imread(color_path)
        if bgr is None:
            return None, None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
        return rgb, depth

    def __getitem__(self, idx):
        path = self.files[idx]
        rgb, depth = self._load(path)
        if rgb is None:
            rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            depth = np.zeros((self.img_size, self.img_size), dtype=np.uint16)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                depth = np.ascontiguousarray(depth[:, ::-1])
            if np.random.rand() < 0.5:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1] *= np.random.uniform(0.85, 1.15)
                hsv[..., 2] *= np.random.uniform(0.85, 1.15)
                rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgb_t = (rgb_t - self.IMAGENET_MEAN) / self.IMAGENET_STD
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / self.depth_scale
        valid_t = (depth_t > 0).float()
        return {"rgb": rgb_t, "depth": depth_t, "valid": valid_t, "path": path}


# ==========================================================================
# BTS — "From Big to Small" (Lee et al., 2020)
#   Paper:  https://arxiv.org/abs/1907.10326
#   Code:   https://github.com/cleinc/bts  (Apache-2.0)
# Architecture (sec. 3 of the paper):
#   - DenseNet-161 encoder; extract features at strides 2,4,8,16,32.
#   - ASPP on the stride-32 feature (rates 3,6,12,18).
#   - Decoder up-projection blocks; at strides 8, 4, 2 a Local Planar
#     Guidance (LPG) head predicts four plane parameters (n1..n4) which
#     are converted to dense depth via a normalised ray–plane
#     intersection (eq. 3–5 of the paper).
#   - LPG outputs + final upsample feed the 1×1 depth head.
# ==========================================================================
class _DenseNet161Encoder(nn.Module):
    """Expose DenseNet-161 features at strides 2, 4, 8, 16, 32."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
        net = densenet161(weights=weights)
        f = net.features
        self.conv0 = nn.Sequential(f.conv0, f.norm0, f.relu0)      # s2,  96
        self.pool0 = f.pool0                                        # s4,  96
        self.block1 = f.denseblock1                                 # s4,  384
        self.trans1 = f.transition1                                 # s8,  192
        self.block2 = f.denseblock2                                 # s8,  768
        self.trans2 = f.transition2                                 # s16, 384
        self.block3 = f.denseblock3                                 # s16, 2112
        self.trans3 = f.transition3                                 # s32, 1056
        self.block4 = f.denseblock4                                 # s32, 2208
        self.norm5 = f.norm5
        self.channels = {"s2": 96, "s4": 384, "s8": 768, "s16": 2112, "s32": 2208}

    def forward(self, x):
        s2 = self.conv0(x)
        s4 = self.block1(self.pool0(s2))
        s8 = self.block2(self.trans1(s4))
        s16 = self.block3(self.trans2(s8))
        s32 = self.norm5(self.block4(self.trans3(s16)))
        return s2, s4, s8, s16, s32


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — rates {3,6,12,18} per BTS sec 3.2."""

    def __init__(self, in_ch: int, out_ch: int = 128):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ELU(inplace=True)),
        ] + [
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ELU(inplace=True))
            for r in (3, 6, 12, 18)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        x = torch.cat([b(x) for b in self.branches], dim=1)
        return self.project(x)


class _LocalPlanarGuidance(nn.Module):
    """LPG layer (eq. 3-5 of BTS).

    Input:  (B, 4, h, w) — per-pixel plane parameters.
    Output: (B, 1, h*up, w*up) — dense depth reconstructed by treating each
    h x w cell as a local plane and sampling ``up x up`` sub-pixel positions:
        depth(u, v) = n4 / (n1*u + n2*v + n3)
    with (u, v) normalised to [-0.5, 0.5] within the cell, matching the
    reference implementation.  The plane normal is enforced to unit length
    via a (sin theta cos phi, sin theta sin phi, cos theta) parameterisation.
    """

    def __init__(self, upratio: int):
        super().__init__()
        self.upratio = upratio
        u = (torch.arange(upratio).float() + 0.5) / upratio - 0.5
        uu, vv = torch.meshgrid(u, u, indexing="ij")
        self.register_buffer("u", uu.reshape(1, 1, upratio, upratio))
        self.register_buffer("v", vv.reshape(1, 1, upratio, upratio))

    def forward(self, plane: torch.Tensor) -> torch.Tensor:
        theta = torch.sigmoid(plane[:, 0:1]) * math.pi / 3
        phi = torch.sigmoid(plane[:, 1:2]) * math.pi * 2
        dist = torch.sigmoid(plane[:, 2:3])
        n1 = torch.sin(theta) * torch.cos(phi)
        n2 = torch.sin(theta) * torch.sin(phi)
        n3 = torch.cos(theta)
        n4 = dist
        B, _, h, w = plane.shape
        up = self.upratio
        n1 = n1.repeat_interleave(up, 2).repeat_interleave(up, 3)
        n2 = n2.repeat_interleave(up, 2).repeat_interleave(up, 3)
        n3 = n3.repeat_interleave(up, 2).repeat_interleave(up, 3)
        n4 = n4.repeat_interleave(up, 2).repeat_interleave(up, 3)
        u = self.u.repeat(1, 1, h, w)
        v = self.v.repeat(1, 1, h, w)
        denom = n1 * u + n2 * v + n3
        depth = n4 / denom.clamp(min=1e-4)
        return depth


class _UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.body(x)


class BTS(nn.Module):
    """BTS with DenseNet-161 encoder (paper default)."""

    def __init__(self, pretrained: bool = True, feat_ch: int = 256):
        super().__init__()
        self.enc = _DenseNet161Encoder(pretrained=pretrained)
        ch = self.enc.channels

        self.aspp = _ASPP(ch["s32"], out_ch=feat_ch)

        self.up16 = _UpConv(feat_ch, feat_ch)
        self.reduce16 = nn.Sequential(
            nn.Conv2d(ch["s16"] + feat_ch, feat_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch), nn.ELU(inplace=True))

        self.up8 = _UpConv(feat_ch, feat_ch // 2)
        self.reduce8 = nn.Sequential(
            nn.Conv2d(ch["s8"] + feat_ch // 2, feat_ch // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch // 2), nn.ELU(inplace=True))
        self.lpg8_head = nn.Conv2d(feat_ch // 2, 3, 1)  # 3 channels: theta, phi, dist
        self.lpg8 = _LocalPlanarGuidance(upratio=8)

        self.up4 = _UpConv(feat_ch // 2, feat_ch // 4)
        self.reduce4 = nn.Sequential(
            nn.Conv2d(ch["s4"] + feat_ch // 4 + 1, feat_ch // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch // 4), nn.ELU(inplace=True))
        self.lpg4_head = nn.Conv2d(feat_ch // 4, 3, 1)  # 3 channels: theta, phi, dist
        self.lpg4 = _LocalPlanarGuidance(upratio=4)

        self.up2 = _UpConv(feat_ch // 4, feat_ch // 8)
        self.reduce2 = nn.Sequential(
            nn.Conv2d(ch["s2"] + feat_ch // 8 + 2, feat_ch // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch // 8), nn.ELU(inplace=True))
        self.lpg2_head = nn.Conv2d(feat_ch // 8, 3, 1)  # 3 channels: theta, phi, dist
        self.lpg2 = _LocalPlanarGuidance(upratio=2)

        self.up1 = _UpConv(feat_ch // 8, feat_ch // 16)
        self.final = nn.Sequential(
            nn.Conv2d(feat_ch // 16 + 3, feat_ch // 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch // 16),
            nn.ELU(inplace=True),
            nn.Conv2d(feat_ch // 16, 1, 3, 1, 1),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        s2, s4, s8, s16, s32 = self.enc(x)

        y = self.aspp(s32)
        y = self.up16(y)
        y = self.reduce16(torch.cat([y, s16], dim=1))

        y = self.up8(y)
        y = self.reduce8(torch.cat([y, s8], dim=1))
        lpg8 = self.lpg8(self.lpg8_head(y))
        lpg8_at_4 = F.interpolate(lpg8, scale_factor=0.25, mode="bilinear", align_corners=False)
        lpg8_at_2 = F.interpolate(lpg8, scale_factor=0.5, mode="bilinear", align_corners=False)

        y = self.up4(y)
        y = self.reduce4(torch.cat([y, s4, lpg8_at_4], dim=1))
        lpg4 = self.lpg4(self.lpg4_head(y))
        lpg4_at_2 = F.interpolate(lpg4, scale_factor=0.5, mode="bilinear", align_corners=False)

        y = self.up2(y)
        y = self.reduce2(torch.cat([y, s2, lpg8_at_2, lpg4_at_2], dim=1))
        lpg2 = self.lpg2(self.lpg2_head(y))

        y = self.up1(y)
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        depth = self.final(torch.cat([y, lpg8, lpg4, lpg2], dim=1))
        return torch.sigmoid(depth)


# ==========================================================================
# AdaBins — Bhat, Alhashim, Wonka (CVPR 2021)
#   Paper:  https://arxiv.org/abs/2011.14141
#   Code:   https://github.com/shariqfarooq123/AdaBins  (GPLv3)
# Architecture (sec. 3):
#   - EfficientNet-B5 encoder-decoder ("ENB5").
#   - mini-ViT (mViT) takes the decoded feature map, outputs:
#       * N=256 bin-width logits → normalised bin centres in [0, 1].
#       * per-pixel range-attention maps R (one per bin) via dot-product
#         between pixel features and transformer tokens.
#   - Final depth = sum_i softmax(R)_i * bin_center_i  (eq. 3).
# ==========================================================================
class _EffB5EncoderDecoder(nn.Module):
    """EfficientNet-B5 feature extractor with a lightweight fusion decoder.

    Mirrors the AdaBins "ENB5" block: features at stages {1,2,3,5,7} of the
    torchvision EfficientNet-B5 are fused top-down into one 1/2-res map.
    """

    TAP_INDICES = (1, 2, 3, 5, 7)
    TAP_CH = (24, 40, 64, 176, 512)

    def __init__(self, pretrained: bool = True, decoder_ch: int = 128):
        super().__init__()
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b5(weights=weights)
        self.stem = net.features[0]
        self.blocks = nn.ModuleList([net.features[i] for i in range(1, 8)])

        c = self.TAP_CH
        self.reduce32 = nn.Conv2d(c[4], decoder_ch, 1)
        self.up32_16 = nn.Sequential(
            nn.Conv2d(decoder_ch + c[3], decoder_ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.up16_8 = nn.Sequential(
            nn.Conv2d(decoder_ch + c[2], decoder_ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.up8_4 = nn.Sequential(
            nn.Conv2d(decoder_ch + c[1], decoder_ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.up4_2 = nn.Sequential(
            nn.Conv2d(decoder_ch + c[0], decoder_ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.out_channels = decoder_ch

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.stem(x)
        taps = []
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if i in self.TAP_INDICES:
                taps.append(x)
        f_s2, f_s4, f_s8, f_s16, f_s32 = taps

        y = self.reduce32(f_s32)
        y = self.up32_16(torch.cat([self._up(y, f_s16), f_s16], dim=1))
        y = self.up16_8(torch.cat([self._up(y, f_s8), f_s8], dim=1))
        y = self.up8_4(torch.cat([self._up(y, f_s4), f_s4], dim=1))
        y = self.up4_2(torch.cat([self._up(y, f_s2), f_s2], dim=1))
        return y  # stride 2, decoder_ch channels


class _PatchTransformer(nn.Module):
    """mViT body (sec. 3.2 of AdaBins)."""

    def __init__(self, in_ch: int, patch_size: int = 16, embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 4, num_queries: int = 128,
                 max_tokens: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, num_queries + max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, E)
        N = tok.shape[1]
        queries = self.pos[:, :self.num_queries].expand(B, -1, -1)
        pos = self.pos[:, self.num_queries:self.num_queries + N]
        tok = torch.cat([queries, tok + pos], dim=1)
        tok = self.encoder(tok)
        return tok[:, :self.num_queries], tok[:, self.num_queries:]


class AdaBins(nn.Module):
    def __init__(self, pretrained: bool = True, n_bins: int = 256,
                 min_depth: float = 1e-3, max_depth: float = 1.0):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.backbone = _EffB5EncoderDecoder(pretrained=pretrained, decoder_ch=128)
        C = self.backbone.out_channels  # 128

        embed_dim = 128
        self.mvit = _PatchTransformer(
            in_ch=C, patch_size=16, embed_dim=embed_dim,
            num_heads=4, num_layers=4, num_queries=n_bins,
        )
        self.pixel_proj = nn.Conv2d(C, embed_dim, 1)
        self.bin_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        feat = self.backbone(x)                          # (B, C, H/2, W/2)
        query_tok, _ = self.mvit(feat)                   # (B, n_bins, E)

        widths = self.bin_mlp(query_tok).squeeze(-1)     # (B, n_bins)
        widths = F.softmax(widths, dim=1) * (self.max_depth - self.min_depth)
        widths = F.pad(widths, (1, 0), value=self.min_depth)
        edges = torch.cumsum(widths, dim=1)              # (B, n_bins+1)
        centres = (edges[:, :-1] + edges[:, 1:]) * 0.5   # (B, n_bins)

        pix = self.pixel_proj(feat)                      # (B, E, H/2, W/2)
        B, E, Hh, Ww = pix.shape
        pix = pix.view(B, E, Hh * Ww)
        logits = torch.bmm(query_tok, pix)               # (B, n_bins, HW)
        attn = F.softmax(logits, dim=1)

        depth = (attn * centres.unsqueeze(-1)).sum(dim=1, keepdim=True)
        depth = depth.view(B, 1, Hh, Ww)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        return depth


# ==========================================================================
# Loss — SILog (BTS eq. 6 / AdaBins eq. 4) + L1 for logging
# ==========================================================================
def silog_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               variance_focus: float = 0.85) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale-invariant log loss used by both BTS and AdaBins."""
    pred = pred.float()
    target = target.float()
    mask = valid > 0.5
    if not mask.any():
        zero = pred.sum() * 0.0
        return zero, zero.detach()

    p = pred[mask].clamp(min=1e-4)
    t = target[mask].clamp(min=1e-4)
    g = torch.log(p) - torch.log(t)
    Dg = (g ** 2).mean() - variance_focus * (g.mean() ** 2)
    loss = 10.0 * torch.sqrt(Dg.clamp(min=1e-8))
    l1 = (p - t).abs().mean().detach()
    return loss, l1


# ==========================================================================
# Snapshot video + plots (match train.py style)
# ==========================================================================
def save_scene_video(snaps, epoch, out_dir):
    if not snaps:
        return
    os.makedirs(out_dir, exist_ok=True)
    scene = snaps[0]["scene"]
    writer = None
    out_path = os.path.join(out_dir, f"{scene}.mp4")
    for snap in snaps:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(snap["rgb"]); axes[0].set_title("RGB")
        vmax = max(float(snap["gt"].max()), 1e-3)
        pm = np.ma.masked_where(snap["gt"] == 0, snap["pred"])
        gm = np.ma.masked_where(snap["gt"] == 0, snap["gt"])
        axes[1].imshow(pm, cmap="inferno", vmin=0, vmax=vmax); axes[1].set_title("Predicted Depth")
        im_gt = axes[2].imshow(gm, cmap="inferno", vmin=0, vmax=vmax); axes[2].set_title("GT Depth")
        err = np.ma.masked_where(snap["gt"] == 0, np.abs(snap["pred"] - snap["gt"]))
        axes[3].imshow(err, cmap="hot", vmin=0, vmax=vmax); axes[3].set_title("|err|")
        for ax in axes: ax.axis("off")
        fig.colorbar(im_gt, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        fig.suptitle(f"{scene}  frame {snap['idx']}  |  MAE={snap['mae']:.4f}  "
                     f"RMSE={snap['rmse']:.4f}  δ<1.25={snap['delta1']:.2%}  |  epoch {epoch}",
                     fontsize=12)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        if writer is None:
            h, w = buf.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.release()
    gc.collect()


def save_training_plots(history, plot_dir, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss", markersize=3)
    ax.plot(epochs, history["val_loss"], "o-", label="Val Loss", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("SILog"); ax.set_title("Train / Val Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if history.get("depth_loss"):
        ep = range(1, len(history["depth_loss"]) + 1)
        ax.plot(ep, history["depth_loss"], "o-", color="tab:red", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("L1"); ax.set_title("Train Depth L1")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if history["mae"]:
        ep = range(1, len(history["mae"]) + 1)
        ax.plot(ep, history["mae"], "o-", label="MAE", markersize=3)
        ax.plot(ep, history["rmse"], "o-", label="RMSE", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error"); ax.set_title("Val MAE / RMSE")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if history["delta1"]:
        ep = range(1, len(history["delta1"]) + 1)
        ax.plot(ep, [d * 100 for d in history["delta1"]], "o-", color="green", markersize=3)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Val δ < 1.25")
    ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[2, 0]
    ax.plot(epochs, history["lr"], "o-", color="orange", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.3); ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    axes[2, 1].axis("off")
    fig.suptitle(f"[{model_name}] Training Progress — Epoch {len(history['train_loss'])}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plot_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


# ==========================================================================
# Train / val loop
# ==========================================================================
def build_model(name: str) -> nn.Module:
    if name == "bts":
        return BTS(pretrained=True)
    if name == "adabins":
        return AdaBins(pretrained=True)
    raise ValueError(f"Unknown model '{name}'. Choose: bts, adabins")


def run_epoch(loader, model, optimizer, device, *, is_train, epoch,
              snap_dir, max_snap_scenes=4):
    model.train() if is_train else model.eval()
    loss_sum = 0.0; l1_sum = 0.0; n = 0
    sum_mae = sum_sq = sum_d1 = 0.0; total_px = 0
    snapshots_by_scene: dict[str, list[dict]] = defaultdict(list)
    scenes_kept: set[str] = set()
    SNAP_STRIDE = 5
    t0 = time.time()

    for step, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            pred = model(rgb)
            loss, l1 = silog_loss(pred, depth, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(rgb)
                loss, l1 = silog_loss(pred, depth, valid)
            p = pred.float(); g = depth.float()
            m = valid > 0.5
            if m.any():
                err = (p - g)[m]
                sum_mae += err.abs().sum().item()
                sum_sq += (err ** 2).sum().item()
                ratio = torch.max(p[m] / g[m].clamp(min=1e-6), g[m] / p[m].clamp(min=1e-6))
                sum_d1 += (ratio < 1.25).sum().item()
                total_px += int(m.sum().item())

            B = rgb.shape[0]
            for b in range(B):
                path = batch["path"][b]
                scene = os.path.basename(os.path.dirname(path))
                if scene not in scenes_kept and len(scenes_kept) >= max_snap_scenes:
                    continue
                frame_name = os.path.splitext(os.path.basename(path))[0]
                try:
                    idx_int = int(frame_name.split("-")[0])
                except ValueError:
                    idx_int = len(snapshots_by_scene[scene])
                if idx_int % SNAP_STRIDE != 0:
                    continue
                scenes_kept.add(scene)
                gt_np = g[b, 0].cpu().numpy()
                pd_np = p[b, 0].cpu().numpy()
                vm = gt_np > 0
                if vm.any():
                    sp = pd_np[vm]; sg = gt_np[vm]
                    smae = float(np.abs(sp - sg).mean())
                    srmse = float(np.sqrt(((sp - sg) ** 2).mean()))
                    ratio = np.maximum(sp / np.clip(sg, 1e-6, None),
                                       sg / np.clip(sp, 1e-6, None))
                    sd1 = float((ratio < 1.25).mean())
                else:
                    smae = srmse = sd1 = 0.0
                rgb_np = rgb[b].cpu().numpy().transpose(1, 2, 0)
                rgb_np = rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                rgb_np = np.clip(rgb_np, 0, 1)
                snapshots_by_scene[scene].append({
                    "scene": scene, "idx": idx_int, "rgb": rgb_np,
                    "pred": pd_np, "gt": gt_np,
                    "mae": smae, "rmse": srmse, "delta1": sd1,
                })

        loss_sum += float(loss.detach())
        l1_sum += float(l1)
        n += 1
        if (step + 1) % 20 == 0:
            mode = "Train" if is_train else "Val"
            print(f"    [{mode}] step {step+1}/{len(loader)}  "
                  f"loss={loss_sum/n:.5f}  l1={l1_sum/n:.5f}", end="\r")

    dt = time.time() - t0
    mode = "Train" if is_train else "Val"
    print(f"  {mode} done — {n} steps in {dt:.1f}s | loss={loss_sum/max(n,1):.5f} l1={l1_sum/max(n,1):.5f}")

    metrics = None
    if not is_train and total_px > 0:
        metrics = {"mae": sum_mae / total_px,
                   "rmse": (sum_sq / total_px) ** 0.5,
                   "delta1": sum_d1 / total_px}

    snap_names: list[str] = []
    if not is_train:
        epoch_dir = os.path.join(snap_dir, f"epoch_{epoch:03d}")
        for scene, snaps in snapshots_by_scene.items():
            snaps.sort(key=lambda s: s["idx"])
            save_scene_video(snaps, epoch, epoch_dir)
            snap_names.append(scene)

    return loss_sum / max(n, 1), l1_sum / max(n, 1), metrics, snap_names


# ==========================================================================
# Main
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bts", "adabins"], required=True)
    # Training hyperparameters mirror train.py (LSTM model) so the three
    # runs are directly comparable.  See train.py L50-59.
    p.add_argument("--img-size", type=int, default=480)       # == IMG_SIZE
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=300)         # == EPOCHS
    p.add_argument("--lr", type=float, default=1e-4)           # BTS paper: 1e-4
    p.add_argument("--weight-decay", type=float, default=1e-4) # == AdamW wd
    p.add_argument("--warmup-epochs", type=int, default=5)     # == warmup_epochs
    p.add_argument("--val-split", type=float, default=0.20)    # == VAL_SPLIT
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--resume", default="")
    p.add_argument("--depth-scale", type=float, default=65535.0)  # == DEPTH_SCALE
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    all_files = sorted(glob(os.path.join(DATA_DIR, "*/*-color.png")))
    scenes_map: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        scenes_map[os.path.basename(os.path.dirname(f))].append(f)
    all_scenes = sorted(scenes_map.keys())
    # Match train.py's legacy RNG so val/train scene splits are identical.
    np.random.seed(42)
    np.random.shuffle(all_scenes)
    n_val = max(1, int(len(all_scenes) * args.val_split))
    val_scenes = all_scenes[:n_val]
    train_scenes = all_scenes[n_val:]

    train_ds = FrameDepthDataset(DATA_DIR, train_scenes, img_size=args.img_size,
                                 depth_scale=args.depth_scale, augment=True)
    val_ds = FrameDepthDataset(DATA_DIR, val_scenes, img_size=args.img_size,
                               depth_scale=args.depth_scale, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=max(2, args.num_workers // 2), pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    print(f"Model: {model_name} | img_size={args.img_size} | "
          f"train_frames={len(train_ds)} val_frames={len(val_ds)}")
    print(f"  Train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"  Val scenes   ({len(val_scenes)}): {val_scenes}")

    model = build_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Optimizer + schedule: use a lower LR for the pretrained encoder to preserve
    # ImageNet features. Both BTS (DenseNet-161) and AdaBins (EfficientNet-B5) have a
    # pretrained backbone; the randomly-initialised decoder can tolerate a higher LR.
    # BTS paper: encoder 1e-5, decoder 1e-4.  AdaBins paper: same 10x ratio.
    _pretrained = getattr(model, 'enc', None) or getattr(model, 'backbone', None)
    if _pretrained is not None:
        encoder_params = list(_pretrained.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
        param_groups = [
            {"params": encoder_params, "lr": args.lr * 0.1},
            {"params": decoder_params, "lr": args.lr},
        ]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(0, args.warmup_epochs)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup_epochs))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs))

    plot_dir = os.path.join(_PROJECT_ROOT, f"training_plots_{model_name}")
    snap_dir = os.path.join(_PROJECT_ROOT, f"training_snapshots_{model_name}")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)
    best_path = os.path.join(_PROJECT_ROOT, f"best_model_{model_name}.pt")
    last_path = os.path.join(_PROJECT_ROOT, f"last_model_{model_name}.pt")

    start_epoch = 1
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "mae": [], "rmse": [],
               "delta1": [], "lr": [], "depth_loss": []}

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        try: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception: pass
        try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception: pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        history = ckpt.get("history", history)
        print(f"  Resumed from {args.resume} (epoch {start_epoch - 1})")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}\n[{model_name}] Epoch {epoch}/{args.epochs}\n{'='*60}")

        train_loss, train_l1, _, _ = run_epoch(
            train_loader, model, optimizer, device,
            is_train=True, epoch=epoch, snap_dir=snap_dir)
        val_loss, _, val_metrics, val_snaps = run_epoch(
            val_loader, model, optimizer, device,
            is_train=False, epoch=epoch, snap_dir=snap_dir)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)
        history["depth_loss"].append(train_l1)
        if val_metrics:
            history["mae"].append(val_metrics["mae"])
            history["rmse"].append(val_metrics["rmse"])
            history["delta1"].append(val_metrics["delta1"])

        tag = ""
        ckpt = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_val_loss": best_val,
            "args": vars(args),
        }
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, best_path)
            tag = " * saved"
        torch.save(ckpt, last_path)

        print(f"[{model_name}] Epoch {epoch:3d}/{args.epochs} | "
              f"Train {train_loss:.5f} | Val {val_loss:.5f} | LR {cur_lr:.2e}{tag}")
        if val_metrics:
            print(f"  Val MAE={val_metrics['mae']:.5f} RMSE={val_metrics['rmse']:.5f} "
                  f"δ<1.25={val_metrics['delta1']:.2%}")

        save_training_plots(history, plot_dir, model_name)
        print(f"  Plots → {plot_dir}/training_curves.png")
        if val_snaps:
            print(f"  Snapshots → {os.path.join(snap_dir, f'epoch_{epoch:03d}')}/ ({len(val_snaps)} scenes)")


if __name__ == "__main__":
    main()
