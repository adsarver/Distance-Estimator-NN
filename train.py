import os, sys, subprocess, importlib, shutil, time, gc, ctypes, math

# Prevent threading-related segfaults
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

# Force glibc to return freed pages to OS (Linux only)
try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim(): _libc.malloc_trim(0)
except (OSError, AttributeError):
    def _malloc_trim(): pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = None
for _candidate in [os.getcwd(), "/content/Distance-Estimator-NN", "/home/adsarver/Distance-Estimator-NN"]:
    if os.path.isfile(os.path.join(_candidate, "data.py")):
        _PROJECT_ROOT = _candidate
        break
if _PROJECT_ROOT is None:
    raise FileNotFoundError("Cannot find project root containing data.py")

os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# Force-reload local modules in case they were cached
for _mod in ["data", "model"]:
    if _mod in sys.modules:
        importlib.reload(sys.modules[_mod])

print(f"Working directory: {os.getcwd()}")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from data import RGBDDataset, scene_collate_fn
from model import DistanceNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
IMG_SIZE = 480
HIDDEN_SIZE = 128
LSTM_LAYERS = 1
LR = 3e-4
EPOCHS = 300
TBTT_LEN = 15           # truncation window: backprop every N frames
VAL_SPLIT = 0.20        # fraction of scenes held out for validation
PRINT_EVERY = 50       # print frame progress every N frames
DEPTH_SCALE = 65535.0  # global max depth in dataset (mm); uint16 max, covers all scenes

RESUME_FROM = "last_model.pt"  # path to checkpoint, or "" to start fresh
MIN_VALID_FRAC = 0.25  # skip frames with fewer valid depth pixels than this

# --- Scene train/val split ---
from glob import glob
from collections import defaultdict

all_files = sorted(glob(os.path.join(DATA_DIR, "*/*-color.png")))
_scenes_map = defaultdict(list)
for f in all_files:
    _scenes_map[os.path.basename(os.path.dirname(f))].append(f)

all_scene_names = sorted(_scenes_map.keys())
np.random.seed(42)
np.random.shuffle(all_scene_names)

# Quick experiment: 1 smallest scene each for train/val
train_scene_names = ['scene_13', 'scene_07']  # 462 frames
val_scene_names   = ['scene_11']  # 640 frames
n_val = max(1, int(len(all_scene_names) * VAL_SPLIT))
val_scene_names = all_scene_names[:n_val]
train_scene_names = all_scene_names[n_val:]

# --- Train-time augmentations ---
# Each transform is (rgb_fn, depth_fn_or_None) applied to (C,H,W) float tensors.
import torchvision.transforms.v2 as T

train_transforms = [
    (T.Identity(), False),
    (T.ColorJitter(0.1, 0.1, 0.1, 0.1), False),
    (T.RandomHorizontalFlip(1.0), True),
    (T.RandomVerticalFlip(1.0), True),
    (T.RandomRotation((180, 180)), True),
    (T.JPEG(10), False),
]

# --- DataLoaders (one scene per batch) ---
def _worker_init_fn(worker_id):
    """Reseed RNG and disable OpenCV threading in each DataLoader worker."""
    import cv2 as _cv2
    _cv2.setNumThreads(0)
    _cv2.ocl.setUseOpenCL(False)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(info.dataset.rng.integers(2**31) + worker_id)

train_ds = RGBDDataset(DATA_DIR, transforms=train_transforms, scene_names=train_scene_names, img_size=IMG_SIZE)
val_ds   = RGBDDataset(DATA_DIR, scene_names=val_scene_names, random_seed=123, img_size=IMG_SIZE, keep_raw=True)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=scene_collate_fn, num_workers=2, worker_init_fn=_worker_init_fn, prefetch_factor=1)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=scene_collate_fn, num_workers=1, worker_init_fn=_worker_init_fn, prefetch_factor=1)

print(f"Scenes — train: {len(train_scene_names)} ({len(train_ds)} with augments), val: {len(val_ds)}")
print(f"  Train scenes: {train_scene_names}")
print(f"  Val scenes:   {val_scene_names}")

# --- Model ---
model = DistanceNN(
    hidden_size=HIDDEN_SIZE,
    lstm_num_layers=LSTM_LAYERS,
    img_size=IMG_SIZE,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
warmup_epochs = 5
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
scaler = GradScaler("cuda")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# --- Resume from checkpoint ---
def partial_load_state_dict(model, saved_dict):
    """Load parameters from saved_dict into model, handling shape mismatches
    by copying the overlapping region (e.g. 1-ch conv → 2-ch conv)."""
    model_dict = model.state_dict()
    loaded, partial, skipped = [], [], []
    for key, saved_param in saved_dict.items():
        if key not in model_dict:
            skipped.append(key)
            continue
        model_param = model_dict[key]
        if saved_param.shape == model_param.shape:
            model_dict[key] = saved_param
            loaded.append(key)
        else:
            # Copy overlapping slice (works for any number of dims)
            slices = tuple(slice(0, min(s, m)) for s, m in
                           zip(saved_param.shape, model_param.shape))
            model_dict[key][slices] = saved_param[slices]
            partial.append(f"{key}: {list(saved_param.shape)}→{list(model_param.shape)}")
    missing = [k for k in model_dict if k not in saved_dict]
    model.load_state_dict(model_dict)
    print(f"  Loaded {len(loaded)} params, {len(partial)} partial, "
          f"{len(skipped)} skipped, {len(missing)} new")
    for p in partial:
        print(f"    partial: {p}")
    for m in missing:
        print(f"    new (random init): {m}")

start_epoch = 1
best_val_loss = float("inf")
history = {"train_loss": [], "val_loss": [], "mae": [], "rmse": [], "delta1": [], "lr": [], "depth_loss": []}
if RESUME_FROM and os.path.isfile(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=device, weights_only=True)
    # Handle both plain state_dict and full checkpoint dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        partial_load_state_dict(model, ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError):
                print("  Optimizer state incompatible, starting fresh optimizer")
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "history" in ckpt:
            history = ckpt["history"]
        if "best_val_loss" in ckpt:
            best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed full checkpoint from {RESUME_FROM} (epoch {start_epoch - 1})")
    else:
        # Plain state_dict (e.g. old best_model.pt)
        partial_load_state_dict(model, ckpt)
        print(f"Loaded model weights from {RESUME_FROM} (no optimizer/epoch state)")
else:
    print(f"Starting fresh (no checkpoint at '{RESUME_FROM}')")

def _ssim_map(pred, target, window_size=7, C1=0.01**2, C2=0.03**2):
    """Compute per-pixel SSIM between two (1,1,H,W) tensors. Returns (1,1,H,W).

    Always runs in float32 regardless of autocast context.
    """
    pred = pred.float()
    target = target.float()
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device, dtype=torch.float32)
    kernel = kernel / (window_size * window_size)

    # Explicit float32 — disable autocast so F.conv2d stays fp32
    with torch.amp.autocast("cuda", enabled=False):
        mu_p  = F.conv2d(pred,   kernel, padding=pad)
        mu_t  = F.conv2d(target, kernel, padding=pad)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sig_pp = F.conv2d(pred * pred,     kernel, padding=pad) - mu_pp
        sig_tt = F.conv2d(target * target, kernel, padding=pad) - mu_tt
        sig_pt = F.conv2d(pred * target,   kernel, padding=pad) - mu_pt

        # Clamp variances to zero (E[X^2]-E[X]^2 can go slightly negative from fp rounding)
        sig_pp = sig_pp.clamp(min=0)
        sig_tt = sig_tt.clamp(min=0)

        numer = (2 * mu_pt + C1) * (2 * sig_pt + C2)
        denom = (mu_pp + mu_tt + C1) * (sig_pp + sig_tt + C2)
        return numer / denom.clamp(min=1e-8)  # per-pixel SSIM in [-1, 1]


def masked_ssim_loss(pred, target, window_size=7):
    valid = target > 0
    if not valid.any():
        return (pred * 0).sum()

    # SSIM needs spatial layout → (1,1,H,W)
    p4 = pred.unsqueeze(0).unsqueeze(0).float()
    t4 = target.unsqueeze(0).unsqueeze(0).float()

    ssim_map = _ssim_map(p4, t4, window_size=window_size)  # (1,1,H,W)
    ssim_map = ssim_map.squeeze()                            # (H,W)
    ssim_loss = (1.0 - ssim_map[valid]).mean() / 2.0

    return ssim_loss


def masked_combined_loss(pred, target, ssim_weight=0.0, window_size=7):
    """L1 + SSIM combined loss.  L1 prevents constant-prediction collapse."""
    valid = target > 0
    if not valid.any():
        return (pred * 0).sum()

    l1 = (pred[valid].float() - target[valid].float()).abs().mean()
    ssim = masked_ssim_loss(pred, target, window_size=window_size)
    return (1.0 - ssim_weight) * l1 + ssim_weight * ssim


def _gradient_loss(pred, target):
    """Edge-aware gradient loss: penalise differences in spatial derivatives.

    Computes L1 between horizontal/vertical gradients of pred and target,
    masked to valid (>0) depth pixels.  Returns scalar.
    """
    pred = pred.float()
    target = target.float()

    # Sobel-style finite differences (H and W)
    pred_dy = pred[1:, :] - pred[:-1, :]
    pred_dx = pred[:, 1:] - pred[:, :-1]
    tgt_dy  = target[1:, :] - target[:-1, :]
    tgt_dx  = target[:, 1:] - target[:, :-1]

    # Masks: both neighbours must be valid
    valid_y = (target[1:, :] > 0) & (target[:-1, :] > 0)
    valid_x = (target[:, 1:] > 0) & (target[:, :-1] > 0)

    if not valid_y.any() or not valid_x.any():
        return (pred * 0).sum()

    loss_y = (pred_dy[valid_y] - tgt_dy[valid_y]).abs().mean()
    loss_x = (pred_dx[valid_x] - tgt_dx[valid_x]).abs().mean()
    return (loss_y + loss_x) / 2.0


def uncertainty_aware_loss(pred, log_var, target, ssim_weight=0.35, grad_weight=0.15, l1_weight=0.25):
    """L1 + Laplacian NLL + SSIM + gradient combined loss.

    A direct L1 term (not modulated by uncertainty) prevents the network from
    collapsing depth loss to zero by inflating uncertainty.

    L1   = |pred - target|                  (direct, not attenuated by unc.)
    NLL  = |pred - target| * exp(-s) + s   (pixelwise, over valid pixels)
    SSIM = structural similarity loss       (spatial structure)
    Grad = L1 between depth gradients       (edge sharpness)
    """
    valid = target > 0
    if not valid.any():
        return (pred * 0).sum()

    p = pred[valid].float()
    s = log_var[valid].float()
    t = target[valid].float()

    l1 = torch.abs(p - t).mean()

    # Clamp log-variance to a sane range:
    #   lower bound 0 => exp(-s) <= 1, so NLL >= |err|*exp(-s) + 0 >= 0
    #   upper bound 4 => exp(-s) ~ 0.018, prevents ignoring large errors
    s_clamped = s.clamp(min=0.0, max=4.0)
    nll = (torch.abs(p - t) * torch.exp(-s_clamped) + s_clamped).mean()

    ssim = masked_ssim_loss(pred, target)
    grad = _gradient_loss(pred, target)

    nll_weight = 1.0 - ssim_weight - grad_weight - l1_weight
    return l1_weight * l1 + nll_weight * nll + ssim_weight * ssim + grad_weight * grad

def safe_detach_head(head):
    head.hx = tuple(h.detach() for h in head.hx)

def _move_scene_to_gpu(frames, device):
    """Bulk-transfer all frame tensors to GPU and free CPU copies."""
    for frame in frames:
        for key in ("img", "bbox", "obj_img", "gt_norm"):
            if key in frame:
                frame[key] = frame[key].to(device, non_blocking=True)
    torch.cuda.synchronize()
    return frames

def run_epoch(loader, model, optimizer, scaler, device, is_train=True, epoch=0):
    """TBTT epoch: one scene at a time, frames pre-loaded by DataLoader workers."""
    mode = "Train" if is_train else "Val"
    model.train() if is_train else model.eval()

    total_frames = 0

    # Validation metric accumulators
    sum_mae = 0.0
    sum_sq_err = 0.0
    sum_delta1 = 0
    total_val_pixels = 0

    # Snapshot collector (val only)
    SNAP_STRIDE = 5
    snapshots = []

    # Loss reporting accumulators
    report_loss_acc = torch.tensor(0.0, device=device)
    report_depth_loss_acc = torch.tensor(0.0, device=device)
    report_count = 0

    epoch_t0 = time.time()

    if not is_train:
        loader.dataset.rng = np.random.default_rng(123)

    for scene_idx, scene_data in enumerate(loader):
        scene_name = scene_data["scene"]
        T = scene_data["T"]
        frames = scene_data["frames"]

        if T < 2:
            continue

        model.reset_lstm()
        chunk_loss = torch.tensor(0.0, device=device)
        chunk_steps = 0
        scene_t0 = time.time()
        scene_frames = []  # for val snapshot video

        # Bulk-transfer entire scene to GPU, free CPU copies
        frames = _move_scene_to_gpu(frames, device)

        for t, frame in enumerate(frames):
            img     = frame["img"].unsqueeze(0)
            bbox_t  = frame["bbox"].unsqueeze(0)
            obj_img = frame["obj_img"].unsqueeze(0)
            gt_norm = frame["gt_norm"]
            crop_h  = frame["crop_h"]
            crop_w  = frame["crop_w"]

            valid_frac = (gt_norm > 0).float().mean().item()
            needs_loss = valid_frac >= MIN_VALID_FRAC

            if is_train:
                if needs_loss:
                    with autocast("cuda"):
                        pred, log_unc = model(img, bbox_t, crop_h, crop_w, obj_img=obj_img)
                        pred = pred.squeeze(0)
                        log_unc = log_unc.squeeze(0)
                    depth_loss = uncertainty_aware_loss(pred.float(), log_unc.float(), gt_norm.float()) * valid_frac
                    chunk_loss = chunk_loss + depth_loss

                    with torch.no_grad():
                        report_depth_loss_acc = report_depth_loss_acc + depth_loss.detach()
                        vm = gt_norm > 0
                        if vm.any():
                            report_loss_acc = report_loss_acc + (pred[vm].float() - gt_norm[vm].float()).abs().mean()
                            report_count += 1
                else:
                    with torch.no_grad(), autocast("cuda"):
                        _ = model(img, bbox_t, crop_h, crop_w, obj_img=obj_img)
            else:
                with torch.no_grad(), autocast("cuda"):
                    pred, log_unc = model(img, bbox_t, crop_h, crop_w, obj_img=obj_img)
                    pred = pred.squeeze(0)
                    log_unc = log_unc.squeeze(0)

                if needs_loss:
                    frame_loss = masked_combined_loss(pred.float(), gt_norm.float())
                    chunk_loss = chunk_loss + frame_loss
                    report_loss_acc = report_loss_acc + frame_loss.detach()
                    report_count += 1

                    valid = gt_norm > 0
                    if valid.any():
                        p = pred[valid].float()
                        g = gt_norm[valid].float()
                        n_px = p.numel()
                        sum_mae += (p - g).abs().sum().item()
                        sum_sq_err += ((p - g) ** 2).sum().item()
                        ratio = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
                        sum_delta1 += (ratio < 1.05).sum().item()
                        total_val_pixels += n_px

                    if t % SNAP_STRIDE == 0 and "rgb" in frame:
                        gt_np = gt_norm.float().cpu().numpy()
                        pred_np = pred.float().cpu().numpy()
                        snap_valid = gt_np > 0
                        snap_valid_pct = float(snap_valid.mean()) * 100.0
                        if snap_valid.any():
                            sp, sg = pred_np[snap_valid], gt_np[snap_valid]
                            snap_mae = float(np.abs(sp - sg).mean())
                            snap_rmse = float(np.sqrt(((sp - sg) ** 2).mean()))
                            snap_ratio = np.maximum(sp / np.clip(sg, 1e-6, None),
                                                    sg / np.clip(sp, 1e-6, None))
                            snap_d1 = float((snap_ratio < 1.05).mean())
                        else:
                            snap_mae = snap_rmse = snap_d1 = 0.0
                        scene_frames.append({
                            "scene": scene_name,
                            "frame_idx": frame["frame_idx"],
                            "T": T,
                            "rgb_full": frame["rgb"].permute(1, 2, 0).numpy(),
                            "rgb_crop": frame["rgb_crop"].permute(1, 2, 0).numpy(),
                            "pred": pred_np,
                            "gt": gt_np,
                            "uncertainty": np.exp(log_unc.float().cpu().numpy()),
                            "mae": snap_mae,
                            "rmse": snap_rmse,
                            "delta1": snap_d1,
                            "valid_pct": snap_valid_pct,
                        })

            total_frames += 1
            chunk_steps += 1

            # --- TBTT boundary ---
            if is_train and chunk_steps % TBTT_LEN == 0 and chunk_loss.requires_grad:
                avg_loss = chunk_loss / chunk_steps
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(avg_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                chunk_loss = torch.tensor(0.0, device=device)
                chunk_steps = 0

                safe_detach_head(model.ctx_head)
                safe_detach_head(model.shape_head)
                safe_detach_head(model.obj_head)

            # Periodic frame progress
            if (t + 1) % PRINT_EVERY == 0:
                running = report_loss_acc.item() / max(report_count, 1)
                print(f"    [{((t+1) / T) * 100:.1f}%] Frame {t+1}/{T} ({scene_name})  avg_mae={running:.5f}", end='\r')

        # Flush remaining chunk
        if is_train and chunk_steps > 0 and chunk_loss.requires_grad:
            avg_loss = chunk_loss / chunk_steps
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(avg_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        del chunk_loss
        torch.cuda.empty_cache()

        scene_dt = time.time() - scene_t0

        # Save scene video (val only)
        if scene_frames and not is_train:
            epoch_dir = os.path.join(SNAP_DIR, f"epoch_{epoch:03d}")
            os.makedirs(epoch_dir, exist_ok=True)
            _save_scene_video(scene_frames, epoch, epoch_dir)
            snapshots.append(scene_name)
        del scene_frames, frames

        gc.collect()
        _malloc_trim()

        report_loss_sum = report_loss_acc.item()
        scene_avg = report_loss_sum / max(report_count, 1)
        print(f"  [{mode}] Scene {scene_idx+1} ({scene_name}) "
              f"| {T} frames in {scene_dt:.1f}s "
              f"| avg_mae={scene_avg:.5f}")

    epoch_dt = time.time() - epoch_t0
    report_loss_sum = report_loss_acc.item()
    avg_mae = report_loss_sum / max(report_count, 1)
    print(f"  {mode} done — {total_frames} frames ({report_count} scored) in {epoch_dt:.1f}s | avg_mae={avg_mae:.6f}")

    if not is_train and total_val_pixels > 0:
        metrics = {
            "mae":    sum_mae / total_val_pixels,
            "rmse":   (sum_sq_err / total_val_pixels) ** 0.5,
            "delta1": sum_delta1 / total_val_pixels,
        }
        return avg_mae, metrics, snapshots

    if is_train and report_count > 0:
        train_metrics = {
            "depth_loss": report_depth_loss_acc.item() / report_count,
        }
        return avg_mae, train_metrics, None

    return avg_mae, None, None

# --- Training Loop (TBTT, scene-level streaming via DataLoader) ---
if not hasattr(history, '__len__') or not isinstance(history, dict):
    history = {"train_loss": [], "val_loss": [], "mae": [], "rmse": [], "delta1": [], "lr": [], "depth_loss": []}
if best_val_loss is None:
    best_val_loss = float("inf")

PLOT_DIR = os.path.join(_PROJECT_ROOT, "training_plots")
SNAP_DIR = os.path.join(_PROJECT_ROOT, "training_snapshots")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

def save_training_plots(history, plot_dir):
    """Save live training curves to disk after each epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # ── (0,0) Train & Val Loss ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss", markersize=3)
    ax.plot(epochs, history["val_loss"],   "o-", label="Val Loss",   markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (L1)")
    ax.set_title("Train / Val Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # ── (0,1) Train Depth Loss ─────────────────────────────────────────────
    ax = axes[0, 1]
    if history.get("depth_loss"):
        dl_epochs = range(1, len(history["depth_loss"]) + 1)
        ax.plot(dl_epochs, history["depth_loss"], "o-", label="Depth Loss (NLL+SSIM)", markersize=3, color="tab:red")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Train Depth Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # ── (1,0) MAE & RMSE ─────────────────────────────────────────────────
    ax = axes[1, 0]
    if history["mae"]:
        m_epochs = range(1, len(history["mae"]) + 1)
        ax.plot(m_epochs, history["mae"],  "o-", label="MAE",  markersize=3)
        ax.plot(m_epochs, history["rmse"], "o-", label="RMSE", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error")
    ax.set_title("Val MAE / RMSE"); ax.legend(); ax.grid(True, alpha=0.3)

    # ── (1,1) δ < 1.25 ───────────────────────────────────────────────────
    ax = axes[1, 1]
    if history["delta1"]:
        d_epochs = range(1, len(history["delta1"]) + 1)
        ax.plot(d_epochs, [d * 100 for d in history["delta1"]], "o-",
                label="δ < 1.05", color="green", markersize=3)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Val δ < 1.05"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── (2,0) Learning Rate ───────────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(epochs, history["lr"], "o-", color="orange", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule"); ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # ── (2,1) Empty ───────────────────────────────────────────────────────
    axes[2, 1].axis("off")

    fig.suptitle(f"Training Progress — Epoch {len(history['train_loss'])}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plot_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def _save_scene_video(scene_frames, epoch, epoch_dir):
    """Render one scene's snapshot frames to an MP4 and free memory."""
    if not scene_frames:
        return
    try:
        import cv2 as _cv2
        scene_name = scene_frames[0]["scene"]

        # Write frames directly to video to avoid accumulating in memory
        h, w = None, None
        writer = None
        out_path = os.path.join(epoch_dir, f"{scene_name}.mp4")

        for snap in scene_frames:
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            axes[0].imshow(snap["rgb_full"])
            axes[0].set_title("Full RGB")

            axes[1].imshow(snap["rgb_crop"])
            axes[1].set_title("Cropped RGB")

            gt = snap["gt"]
            pred = snap["pred"]
            
            # Fixed depth scale [0, 1] for log-normalized depth
            # Makes frames directly comparable without needing to check colorbar
            # Reference: 0.56 ≈ 0.5m, 0.62 ≈ 1m, 0.77 ≈ 5m, 0.93 ≈ 30m
            vmin, vmax = 0.0, 1.0

            # Mask invalid (zero) GT pixels to transparent so they don't flatten the colormap
            pred_masked = np.ma.masked_where(gt == 0, pred)
            gt_masked = np.ma.masked_where(gt == 0, gt)

            axes[2].imshow(pred_masked, cmap="inferno", vmin=vmin, vmax=vmax)
            axes[2].set_title("Predicted Depth")

            im_gt = axes[3].imshow(gt_masked, cmap="inferno", vmin=vmin, vmax=vmax)
            axes[3].set_title("Ground Truth Depth")

            unc = snap.get("uncertainty")
            if unc is not None:
                im_unc = axes[4].imshow(unc, cmap="hot")
                axes[4].set_title("Uncertainty (σ)")
                fig.colorbar(im_unc, ax=axes[4], fraction=0.046, pad=0.04)
            else:
                axes[4].axis("off")
                axes[4].set_title("Uncertainty (N/A)")

            for ax in axes[:4]:
                ax.axis("off")

            fig.colorbar(im_gt, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
            fig.suptitle(
                f"{snap['scene']}  frame {snap['frame_idx']}/{snap['T']}  |  "
                f"MAE={snap['mae']:.4f}  RMSE={snap['rmse']:.4f}  "
                f"δ<1.05={snap['delta1']:.2%}  valid={snap.get('valid_pct', 100):.0f}%  |  epoch {epoch}",
                fontsize=13,
            )

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            del fig, axes

            # Initialize writer on first frame
            if writer is None:
                h, w = buf.shape[:2]
                fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
                writer = _cv2.VideoWriter(out_path, fourcc, 10, (w, h))

            # Write immediately and free
            writer.write(_cv2.cvtColor(buf, _cv2.COLOR_RGB2BGR))
            del buf

        if writer is not None:
            writer.release()
            del writer

        # Force cleanup
        gc.collect()

    except Exception as e:
        print(f"\n  [video] Error saving {scene_frames[0]['scene']}: {e}")
        gc.collect()

for epoch in range(start_epoch, EPOCHS + 1):
    print(f"\n{'='*50}\nEpoch {epoch}/{EPOCHS}\n{'='*50}")
    
    train_loss, train_metrics, _ = run_epoch(train_loader, model, optimizer, scaler, device, is_train=True, epoch=epoch)
    print()
    val_loss, val_metrics, val_snapshots = run_epoch(val_loader, model, optimizer, scaler, device, is_train=False, epoch=epoch)
    print()

    scheduler.step()
    cur_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["lr"].append(cur_lr)
    if train_metrics:
        history["depth_loss"].append(train_metrics["depth_loss"])
    if val_metrics:
        history["mae"].append(val_metrics["mae"])
        history["rmse"].append(val_metrics["rmse"])
        history["delta1"].append(val_metrics["delta1"])

    tag = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
        }, "best_model.pt")
        tag = " * saved"
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "best_val_loss": val_loss,
    }, "last_model.pt")

    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {cur_lr:.2e}{tag}")
    if val_metrics:
        print(f"  Val MAE: {val_metrics['mae']:.5f} | "
              f"RMSE: {val_metrics['rmse']:.5f} | "
              f"δ<1.05: {val_metrics['delta1']:.2%}")

    save_training_plots(history, PLOT_DIR)
    print(f"  Plots saved to {PLOT_DIR}/training_curves.png")

    if val_snapshots:
        snap_epoch_dir = os.path.join(SNAP_DIR, f"epoch_{epoch:03d}")
        print(f"  Snapshots saved to {snap_epoch_dir}/ ({len(val_snapshots)} scenes)")