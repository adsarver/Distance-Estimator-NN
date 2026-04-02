"""
Visual analysis: load best_model.pt, run inference on validation scenes,
and save side-by-side comparison images (RGB | Predicted Depth | Ground Truth Depth).
"""

import matplotlib
matplotlib.use("Agg")

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict

from data import RGBDDataset, scene_collate_fn
from model import DistanceNN
from torch.utils.data import DataLoader

# ── Config (mirrors train defaults) ───────────────────────────────────────────
IMG_SIZE = 256
HIDDEN_SIZE = 128
LSTM_LAYERS = 1
MEMORY_LENGTH = 30
DEPTH_SCALE = 29842.0
MEMORY_STRIDE = 1
VAL_SPLIT = 0.25

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visual depth-prediction analysis")
parser.add_argument("--checkpoint", default="best_model.pt", help="Model checkpoint path")
parser.add_argument("--data-dir", default="rgbd-scenes-v2/imgs", help="Dataset root")
parser.add_argument("--out-dir", default="analysis_output", help="Where to save images")
parser.add_argument("--max-frames", type=int, default=10, help="Max frames to visualize per scene")
parser.add_argument("--frame-stride", type=int, default=None,
                    help="Sample every N-th frame (default: auto to get --max-frames)")
parser.add_argument("--scenes", nargs="*", default=None,
                    help="Specific scene names to analyze (default: validation scenes)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Reproduce the same val split used in training ─────────────────────────────
DATA_DIR = args.data_dir
all_files = sorted(glob(os.path.join(DATA_DIR, "*/*-color.png")))
scenes_map = defaultdict(list)
for f in all_files:
    scenes_map[os.path.basename(os.path.dirname(f))].append(f)

all_scene_names = sorted(scenes_map.keys())
np.random.seed(42)
np.random.shuffle(all_scene_names)

n_val = max(1, int(len(all_scene_names) * VAL_SPLIT))
val_scene_names = all_scene_names[:n_val]

if args.scenes is not None:
    scene_names = args.scenes
else:
    scene_names = val_scene_names

print(f"Analyzing scenes: {scene_names}")

# ── Dataset & model ───────────────────────────────────────────────────────────
ds = RGBDDataset(DATA_DIR, scene_names=scene_names, random_seed=123)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=scene_collate_fn, num_workers=0)

model = DistanceNN(
    hidden_size=HIDDEN_SIZE,
    lstm_num_layers=LSTM_LAYERS,
    memory_length=MEMORY_LENGTH,
    memory_stride=MEMORY_STRIDE,
    img_size=IMG_SIZE,
).to(device)

state = torch.load(args.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint: {args.checkpoint}")

# ── Run inference & save figures ──────────────────────────────────────────────
os.makedirs(args.out_dir, exist_ok=True)

for si, scene_meta in enumerate(loader):
    scene_name = scene_meta["scene"]
    T = scene_meta["T"]
    if T < 2:
        continue

    # Determine which frames to visualize
    stride = args.frame_stride or max(1, T // args.max_frames)
    frame_indices = list(range(0, T, stride))[: args.max_frames]

    model.reset_lstm()
    saved = 0

    for t in range(T):
        frame = ds.load_frame(scene_meta, t)
        crop_h, crop_w = frame["crop_dim"]
        if crop_h < 2 or crop_w < 2:
            continue

        img = F.interpolate(frame["rgb"].unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False).to(device)
        bbox_t = frame["bbox"].unsqueeze(0).to(device)
        obj_img = frame["rgb_crop"].unsqueeze(0).to(device)
        gt_depth = frame["depth_crop"].squeeze(0)
        gt_norm = (gt_depth / DEPTH_SCALE).clamp(0, 1).cpu().numpy()

        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred, log_unc = model(img, bbox_t, crop_h, crop_w, obj_img=obj_img)
            pred = pred.squeeze(0)
            log_unc = log_unc.squeeze(0)
        pred_np = pred.float().cpu().numpy()
        unc_np = np.exp(log_unc.float().cpu().numpy())

        if t not in frame_indices:
            continue

        # ── Build the figure ──────────────────────────────────────────────
        rgb_crop_np = frame["rgb_crop"].permute(1, 2, 0).numpy()  # (H, W, 3)
        rgb_full_np = frame["rgb"].permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(rgb_full_np)
        axes[0].set_title("Full RGB")

        axes[1].imshow(rgb_crop_np)
        axes[1].set_title("Cropped RGB")

        vmin = min(gt_norm[gt_norm > 0].min() if (gt_norm > 0).any() else 0, pred_np.min())
        vmax = max(gt_norm.max(), pred_np.max())

        im_pred = axes[2].imshow(pred_np, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[2].set_title("Predicted Depth")

        im_gt = axes[3].imshow(gt_norm, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[3].set_title("Ground Truth Depth")

        for ax in axes:
            ax.axis("off")

        fig.colorbar(im_gt, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)

        # ── Per-frame metrics ─────────────────────────────────────────────
        valid = gt_norm > 0
        if valid.any():
            p, g = pred_np[valid], gt_norm[valid]
            mae = np.abs(p - g).mean()
            rmse = np.sqrt(((p - g) ** 2).mean())
            ratio = np.maximum(p / np.clip(g, 1e-6, None), g / np.clip(p, 1e-6, None))
            delta1 = (ratio < 1.25).mean()
            fig.suptitle(
                f"{scene_name}  frame {t}/{T}  |  MAE={mae:.4f}  RMSE={rmse:.4f}  δ<1.25={delta1:.2%}",
                fontsize=13,
            )
        else:
            fig.suptitle(f"{scene_name}  frame {t}/{T}  |  no valid GT pixels", fontsize=13)

        out_path = os.path.join(args.out_dir, f"{scene_name}_frame{t:04d}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        saved += 1

    print(f"  [{scene_name}] saved {saved} images (out of {T} frames)")

print(f"\nDone — results in {args.out_dir}/")
