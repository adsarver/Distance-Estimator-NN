"""compare_models_visual.py — Compare depth models with sample image outputs.

For each sampled frame the script produces a row of five panels:
  1. Cropped RGB image
  2. Cropped Depth Truth
  3. LSTM (DistanceNN) output
  4. BTS output
  5. AdaBins output

Outputs are written to a single grid PNG (compare_models_visual.png) and,
optionally, per-sample PNGs under compare_models_visual/.

Usage:
    python compare_models_visual.py \
        --lstm best_model.pt \
        --bts  best_model_bts.pt \
        --adabins best_model_adabins.pt \
        --n-samples 8
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from glob import glob

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

from baseline_train import AdaBins, BTS  # noqa: E402
from data import LOG_DEPTH_SCALE  # noqa: E402
from model import DistanceNN  # noqa: E402

_DEPTH_SCALE = 65535.0

# ImageNet stats used by BTS / AdaBins FrameDepthDataset
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Display size for each panel (pixels)
_DISPLAY_SIZE = 256


# ---------------------------------------------------------------------------
# Dataset / scene helpers
# ---------------------------------------------------------------------------

def _val_scene_names(val_split: float = 0.20) -> list[str]:
    """Reproduce the train.py validation split (seed=42 shuffle)."""
    all_files = sorted(glob(os.path.join(DATA_DIR, "*/*-color.png")))
    scenes: dict[str, list] = defaultdict(list)
    for f in all_files:
        scenes[os.path.basename(os.path.dirname(f))].append(f)
    all_scenes = sorted(scenes.keys())
    np.random.seed(42)
    np.random.shuffle(all_scenes)
    n_val = max(1, int(len(all_scenes) * val_split))
    return all_scenes[:n_val]


def _load_state_dict(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------

def _load_frame_tensors(color_path: str, img_size: int,
                        cx: float, cy: float, bw: float, bh: float
                        ) -> dict | None:
    """Load one RGBD frame and return tensors for the given bbox.

    Returns None if the file is unreadable or the depth map is missing.
    """
    depth_path = color_path.replace("-color.png", "-depth.png")
    if not os.path.isfile(depth_path):
        depth_path = color_path.replace("-color.png", "-aligned-depth.png")

    bgr = cv2.imread(color_path)
    if bgr is None:
        return None
    rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None

    H_raw, W_raw = rgb_raw.shape[:2]
    top    = max(0, int((cy - bh / 2) * H_raw))
    left   = max(0, int((cx - bw / 2) * W_raw))
    bottom = min(H_raw, top  + int(bh * H_raw))
    right  = min(W_raw, left + int(bw * W_raw))
    crop_h, crop_w = bottom - top, right - left
    if crop_h < 2 or crop_w < 2:
        return None

    rgb_t   = torch.from_numpy(rgb_raw).permute(2, 0, 1).float() / 255.0
    depth_t = torch.from_numpy(depth_raw.astype(np.float32)).unsqueeze(0)

    img = F.interpolate(rgb_t.unsqueeze(0), size=(img_size, img_size),
                        mode="bilinear", align_corners=False).squeeze(0)

    rgb_crop   = rgb_t[:, top:bottom, left:right]
    depth_crop = depth_t[:, top:bottom, left:right]

    gt_f  = depth_crop.float()
    valid = (gt_f > 0).float()

    gt_log   = (torch.log(gt_f + 1) / LOG_DEPTH_SCALE).clamp(0, 1).squeeze(0)
    depth_gt = (torch.exp(gt_log * LOG_DEPTH_SCALE) - 1.0) / _DEPTH_SCALE

    sq = max(crop_h, crop_w)
    obj_img = F.interpolate(rgb_crop.unsqueeze(0), size=(sq, sq),
                            mode="bilinear", align_corners=False).squeeze(0)

    return {
        "img":      img.unsqueeze(0),
        "bbox":     torch.tensor([cx, cy, bw, bh], dtype=torch.float32).unsqueeze(0),
        "crop_h":   crop_h,
        "crop_w":   crop_w,
        "obj_img":  obj_img.unsqueeze(0),
        "gt_log":   gt_log.unsqueeze(0),
        "rgb_crop": rgb_crop,
        "depth_gt": depth_gt.unsqueeze(0),
        "valid":    valid,
    }


def collect_sequences(val_scenes: list[str], img_size: int, n_samples: int,
                      seq_len: int = 30,
                      seed: int | None = None,
                      min_valid_frac: float = 0.25) -> list[dict]:
    """Pick *n_samples* sequences of *seq_len* consecutive frames.

    For each output row:
      - A random validation scene is chosen.
      - A random contiguous run of *seq_len* frames is selected from that scene.
      - The LSTM will be warmed up on all *seq_len* frames; only the last
        frame is displayed and evaluated.
      - BTS / AdaBins also run on the last frame only.

    Memory-safe: only *seq_len* frames are held in memory per sample (and
    only the last frame is retained after LSTM warm-up).

    Each entry in the returned list contains:
      warmup_paths  list[str]   – first seq_len-1 color file paths
      last_frame    dict        – tensors for the final frame (for display)
      bbox          (cx,cy,bw,bh) tuple shared across the whole sequence
    """
    rng = np.random.default_rng(seed)

    # Build per-scene sorted frame lists (paths only, no I/O yet)
    all_paths = sorted(glob(os.path.join(DATA_DIR, "*/*-color.png")))
    scene_files: dict[str, list[str]] = defaultdict(list)
    scene_set = set(val_scenes)
    for p in all_paths:
        scene = os.path.basename(os.path.dirname(p))
        if scene in scene_set:
            scene_files[scene].append(p)

    eligible = [s for s, fs in scene_files.items() if len(fs) >= seq_len]
    if not eligible:
        raise RuntimeError(
            f"No validation scene has >= {seq_len} frames. "
            "Lower --seq-len or add more scenes.")

    sequences: list[dict] = []
    attempts = 0
    max_attempts = n_samples * 50

    while len(sequences) < n_samples and attempts < max_attempts:
        attempts += 1
        scene = eligible[int(rng.integers(len(eligible)))]
        files = scene_files[scene]
        start = int(rng.integers(0, len(files) - seq_len + 1))
        run   = files[start : start + seq_len]

        # Random bbox — fixed for the whole sequence
        bw = 0.65 * rng.random() + 0.05
        bh = 0.65 * rng.random() + 0.05
        cx = (1 - bw) * rng.random() + 0.5 * bw
        cy = (1 - bh) * rng.random() + 0.5 * bh

        # Validate the last frame only (cheap — we check validity before accepting)
        last = _load_frame_tensors(run[-1], img_size, cx, cy, bw, bh)
        if last is None:
            continue
        if float(last["valid"].mean()) < min_valid_frac:
            continue

        sequences.append({
            "scene":        scene,
            "start":        start,
            "warmup_paths": run[:-1],   # first seq_len-1 paths
            "last_path":    run[-1],
            "last_frame":   last,
            "bbox":         (cx, cy, bw, bh),
        })

    if not sequences:
        print("[warn] Could not collect any valid sequences.")
    return sequences


# ---------------------------------------------------------------------------
# Per-model inference on collected samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_lstm(model: DistanceNN, sequences: list[dict],
            img_size: int, device: torch.device) -> list[np.ndarray]:
    """Warm-up LSTM on the full sequence; return depth for the last frame only.

    For each sequence:
      1. Reset LSTM hidden state.
      2. Feed all warm-up frames (seq_len-1) through the model, discarding output.
      3. Feed the last frame and return its depth prediction.
    """
    model.eval()
    outputs = []
    for seq in sequences:
        cx, cy, bw, bh = seq["bbox"]
        model.reset_lstm()

        # ---- warm-up pass: load and feed each preceding frame ----
        for path in seq["warmup_paths"]:
            f = _load_frame_tensors(path, img_size, cx, cy, bw, bh)
            if f is None:
                continue  # skip unreadable frames; LSTM state carries over
            model(f["img"].to(device),
                  f["bbox"].to(device),
                  f["crop_h"], f["crop_w"],
                  obj_img=f["obj_img"].to(device))

        # ---- final frame ----
        lf = seq["last_frame"]
        pred_log, _ = model(
            lf["img"].to(device),
            lf["bbox"].to(device),
            lf["crop_h"], lf["crop_w"],
            obj_img=lf["obj_img"].to(device),
        )
        pred_lin = (torch.exp(pred_log.clamp(0, 1) * LOG_DEPTH_SCALE) - 1.0) / _DEPTH_SCALE
        outputs.append(pred_lin.squeeze().cpu().numpy())
    return outputs


@torch.no_grad()
def run_baseline(model: torch.nn.Module, sequences: list[dict],
                 img_size: int, device: torch.device) -> list[np.ndarray]:
    """Runs BTS or AdaBins on the last frame of each sequence.

    Returns list of (crop_h, crop_w) linear-depth arrays in [0,1].
    """
    model.eval()
    outputs = []
    for s in [seq["last_frame"] for seq in sequences]:
        # Apply ImageNet normalisation expected by BTS/AdaBins
        img_norm = (s["img"].squeeze(0) - _IMAGENET_MEAN) / _IMAGENET_STD  # (3,H,W)
        rgb_in = img_norm.unsqueeze(0).to(device)                           # (1,3,H,W)

        pred_full = model(rgb_in)  # (1,1,img_size,img_size) or (1,img_size,img_size)
        if pred_full.dim() == 3:
            pred_full = pred_full.unsqueeze(1)
        pred_full = F.interpolate(pred_full, size=(img_size, img_size),
                                  mode="bilinear", align_corners=False)
        pred_np = pred_full.squeeze().cpu().numpy()  # (img_size, img_size)

        # Crop to the same bbox as the LSTM output
        cx, cy, bw, bh = s["bbox"].squeeze().tolist()
        top    = max(0, int((cy - bh / 2) * img_size))
        left   = max(0, int((cx - bw / 2) * img_size))
        bottom = min(img_size, top  + int(bh * img_size))
        right  = min(img_size, left + int(bw * img_size))

        crop = pred_np[top:bottom, left:right]
        if crop.size == 0:
            crop = pred_np

        # Resize to match the LSTM crop dimensions
        crop_h, crop_w = s["crop_h"], s["crop_w"]
        crop_resized = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        outputs.append(crop_resized)
    return outputs


# ---------------------------------------------------------------------------
# Colourmap helpers
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray, vmin: float | None = None,
              vmax: float | None = None) -> np.ndarray:
    """Normalise *arr* to [0,255] uint8, masking NaN/inf."""
    arr = arr.copy().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    _vmin = vmin if vmin is not None else float(arr.min())
    _vmax = vmax if vmax is not None else float(arr.max())
    if _vmax - _vmin < 1e-8:
        _vmax = _vmin + 1e-8
    norm = np.clip((arr - _vmin) / (_vmax - _vmin), 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def _depth_colormap(arr: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Convert a 2-D depth array → (H,W,3) uint8 RGB using the 'plasma' LUT."""
    u8 = _to_uint8(arr, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("plasma")
    rgb = (cmap(u8 / 255.0)[..., :3] * 255).astype(np.uint8)
    return rgb


def _resample(img_hw3: np.ndarray, size: int) -> np.ndarray:
    """Resize (H,W,3) to (size,size,3)."""
    return cv2.resize(img_hw3, (size, size), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def build_figure(sequences: list[dict],
                 lstm_preds:    list[np.ndarray],
                 bts_preds:     list[np.ndarray],
                 adabins_preds: list[np.ndarray],
                 display_size:  int = _DISPLAY_SIZE) -> plt.Figure:
    """Return a matplotlib Figure with one row per sequence (last frame) and 5 columns."""
    N    = len(sequences)
    cols = ["Cropped RGB", "Cropped Depth GT", "LSTM Output", "BTS Output", "AdaBins Output"]
    ncols = len(cols)

    fig_w = ncols * (display_size / 72) + 2.0
    fig_h = N    * (display_size / 72) + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=72)
    gs  = gridspec.GridSpec(N, ncols, figure=fig,
                            hspace=0.04, wspace=0.04,
                            left=0.01, right=0.99,
                            top=1.0 - 0.9 / fig_h,
                            bottom=0.01)

    # Column headers
    for c, title in enumerate(cols):
        ax = fig.add_subplot(gs[0, c])
        ax.set_title(title, fontsize=9, pad=3)

    for r, (seq, l_pred, b_pred, a_pred) in enumerate(
            zip(sequences, lstm_preds, bts_preds, adabins_preds)):
        s = seq["last_frame"]
        # --- shared depth range (use GT for consistent scale) ---------------
        gt_np  = s["depth_gt"].squeeze().numpy()
        valid  = s["valid"].squeeze().numpy() > 0.5
        vmin = float(gt_np[valid].min()) if valid.any() else 0.0
        vmax = float(gt_np[valid].max()) if valid.any() else 1.0

        # 1) Cropped RGB
        rgb_np = s["rgb_crop"].permute(1, 2, 0).numpy()           # (H,W,3) [0,1]
        rgb_u8 = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
        panels = [
            _resample(rgb_u8, display_size),
            _resample(_depth_colormap(gt_np,  vmin=vmin, vmax=vmax), display_size),
            _resample(_depth_colormap(l_pred, vmin=vmin, vmax=vmax), display_size),
            _resample(_depth_colormap(b_pred, vmin=vmin, vmax=vmax), display_size),
            _resample(_depth_colormap(a_pred, vmin=vmin, vmax=vmax), display_size),
        ]

        for c, panel in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(panel)
            ax.axis("off")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visual comparison of LSTM / BTS / AdaBins outputs.")
    p.add_argument("--lstm",     default="best_model.pt")
    p.add_argument("--bts",      default="best_model_bts.pt")
    p.add_argument("--adabins",  default="best_model_adabins.pt")
    p.add_argument("--lstm-img-size",    type=int, default=480)
    p.add_argument("--bts-img-size",     type=int, default=480)
    p.add_argument("--adabins-img-size", type=int, default=480)
    p.add_argument("--val-split",   type=float, default=0.20)
    p.add_argument("--n-samples",   type=int,   default=8,
                   help="Number of sequences (output rows) to visualise")
    p.add_argument("--seq-len",     type=int,   default=30,
                   help="Frames fed to LSTM per sequence (last frame is displayed)")
    p.add_argument("--seed",        type=int,   default=-1,
                   help="Random seed for sequence selection (-1 = random each run)")
    p.add_argument("--out",         default="compare_models_visual.png",
                   help="Output PNG path")
    p.add_argument("--display-size", type=int, default=_DISPLAY_SIZE,
                   help="Panel size in pixels (square)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_scenes = _val_scene_names(args.val_split)
    print(f"Validation scenes ({len(val_scenes)}): {val_scenes}")

    # Use the LSTM img_size for sample collection (all models share the same
    # source frames; baseline inputs are derived from the same full image).
    img_size = args.lstm_img_size
    seed = None if args.seed < 0 else args.seed
    print(f"\nCollecting {args.n_samples} sequences (seq_len={args.seq_len}, seed={seed}) …")
    sequences = collect_sequences(val_scenes, img_size, args.n_samples,
                                  seq_len=args.seq_len, seed=seed)
    if not sequences:
        print("No valid sequences found — check DATA_DIR and val split.")
        return
    for seq in sequences:
        print(f"  Scene={seq['scene']}  start={seq['start']}")
    print()

    # -----------------------------------------------------------------------
    # Build and load all models
    # -----------------------------------------------------------------------
    configs = [
        ("LSTM (DistanceNN)", args.lstm,    "lstm",    args.lstm_img_size),
        ("BTS",               args.bts,     "bts",     args.bts_img_size),
        ("AdaBins",           args.adabins, "adabins", args.adabins_img_size),
    ]

    models: dict[str, torch.nn.Module | None] = {}
    for label, ckpt_path, kind, mimg_size in configs:
        print(f"Loading {label} from {ckpt_path} …")
        if not os.path.isfile(ckpt_path):
            print(f"  [missing] — will be skipped.\n")
            models[kind] = None
            continue

        if kind == "lstm":
            m = DistanceNN(hidden_size=128, lstm_num_layers=1, img_size=mimg_size)
        elif kind == "bts":
            m = BTS(pretrained=False)
        else:
            m = AdaBins(pretrained=False)

        state = _load_state_dict(ckpt_path, device)
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing:
            print(f"  [warn] {len(missing)} missing keys")
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected keys")
        m.to(device).eval()
        models[kind] = m
        print(f"  Loaded.\n")

    # -----------------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------------
    _dummy = np.zeros((_DISPLAY_SIZE, _DISPLAY_SIZE), dtype=np.float32)

    print("Running LSTM inference (warm-up + last frame) …")
    lstm_m = models.get("lstm")
    lstm_preds = (run_lstm(lstm_m, sequences, img_size, device) if lstm_m is not None
                  else [_dummy] * len(sequences))

    print("Running BTS inference …")
    bts_m = models.get("bts")
    bts_preds = (run_baseline(bts_m, sequences, args.bts_img_size, device) if bts_m is not None
                 else [_dummy] * len(sequences))

    print("Running AdaBins inference …")
    ada_m = models.get("adabins")
    ada_preds = (run_baseline(ada_m, sequences, args.adabins_img_size, device) if ada_m is not None
                 else [_dummy] * len(sequences))

    # -----------------------------------------------------------------------
    # Build and save figure
    # -----------------------------------------------------------------------
    print("\nRendering figure …")
    fig = build_figure(sequences, lstm_preds, bts_preds, ada_preds,
                       display_size=args.display_size)

    out_path = os.path.join(_PROJECT_ROOT, args.out)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
