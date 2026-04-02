import os, sys, subprocess, importlib, shutil, time
import numpy as np
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

DATA_DIR = os.path.join(_PROJECT_ROOT, "rgbd-scenes-v2", "imgs")

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
IMG_SIZE = 256
HIDDEN_SIZE = 128
LSTM_LAYERS = 1
MEMORY_LENGTH = 10
MEMORY_STRIDE = 1
LR = 1e-3
EPOCHS = 100
TBTT_LEN = 15           # truncation window: backprop every N frames
VAL_SPLIT = 0.25        # fraction of scenes held out for validation
PRINT_EVERY = 5       # print frame progress every N frames
DEPTH_SCALE = 29842.0  # global max depth in dataset (mm); fixed scale for consistent targets
RESUME_FROM = "best_model.pt"  # path to checkpoint, or "" to start fresh
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

# --- DataLoaders (one scene per batch) ---
train_ds = RGBDDataset(DATA_DIR, scene_names=train_scene_names)
val_ds   = RGBDDataset(DATA_DIR, scene_names=val_scene_names, random_seed=123)

def _worker_init_fn():
    """Re-seed each worker's numpy RNG so every worker (and epoch) gets unique random bboxes."""
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(torch.initial_seed() % (2**32))

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=scene_collate_fn, num_workers=0, worker_init_fn=_worker_init_fn)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=scene_collate_fn, num_workers=0, worker_init_fn=_worker_init_fn)

print(f"Scenes — train: {len(train_ds)}, val: {len(val_ds)}")
print(f"  Train scenes: {train_scene_names}")
print(f"  Val scenes:   {val_scene_names}")

# --- Model ---
model = DistanceNN(
    hidden_size=HIDDEN_SIZE,
    lstm_num_layers=LSTM_LAYERS,
    img_size=IMG_SIZE,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
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


def uncertainty_aware_loss(pred, log_var, target, ssim_weight=0.5):
    """Laplacian NLL + SSIM combined loss with learned per-pixel uncertainty.

    NLL  = |pred - target| * exp(-s) + s   (pixelwise, over valid pixels)
    SSIM = structural similarity loss       (spatial structure)

    Combined: (1 - ssim_weight) * NLL + ssim_weight * SSIM
    """
    valid = target > 0
    if not valid.any():
        return (pred * 0).sum()

    p = pred[valid].float()
    s = log_var[valid].float()
    t = target[valid].float()
    nll = (torch.abs(p - t) * torch.exp(-s) + s).mean()

    ssim = masked_ssim_loss(pred, target)

    return (1.0 - ssim_weight) * nll + ssim_weight * ssim

def safe_detach_head(head):
    head.hx = tuple(h.detach() for h in head.hx)

def run_epoch(loader, model, optimizer, scaler, device, is_train=True):
    """
    TBTT epoch with mixed-precision, print-based progress, and validation metrics.
    Frames are loaded lazily one at a time to avoid OOM.
    Returns (avg_loss, metrics_dict | None, snapshots | None).
    During validation, captures one snapshot per scene (at ~50% through) for
    visual comparison.
    """
    mode = "Train" if is_train else "Val"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_frames = 0
    dataset = loader.dataset
    n_scenes = len(dataset)

    # Validation metric accumulators
    sum_mae = 0.0
    sum_sq_err = 0.0
    sum_delta1 = 0
    total_val_pixels = 0

    # Snapshot collector (val only): every SNAP_STRIDE-th frame per scene
    SNAP_STRIDE = 5
    snapshots = []  # list of per-scene lists

    # Separate counters for accurate loss reporting
    report_loss_sum = 0.0
    report_count = 0

    epoch_t0 = time.time()

    # Re-seed val RNG so the same bbox crop is used every epoch
    if not is_train:
        loader.dataset.rng = np.random.default_rng(123)

    for si, scene_meta in enumerate(loader):
        T = scene_meta["T"]

        if T < 2:
            print(f"  Scene {si+1}/{n_scenes} ({scene_meta['scene']}): too few frames, skipping.")
            continue

        model.reset_lstm()
        chunk_loss = torch.tensor(0.0, device=device)
        chunk_frames = 0
        scene_t0 = time.time()
        scene_frames = []  # collect frames for video

        for t in range(T):
            frame = dataset.load_frame(scene_meta, t)
            crop_h, crop_w = frame["crop_dim"]
            if crop_h < 2 or crop_w < 2:
                continue

            img = F.interpolate(frame["rgb"].unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False).to(device)
            bbox_t = frame["bbox"].unsqueeze(0).to(device)
            obj_img = frame["rgb_crop"].unsqueeze(0).to(device)
            gt_depth = frame["depth_crop"].squeeze(0).to(device)
            gt_norm = (gt_depth / DEPTH_SCALE).clamp(0, 1)

            # Skip frames with too few valid depth pixels — sensor failures
            # produce misleading loss that rewards flat predictions
            valid_frac = (gt_norm > 0).float().mean().item()
            if valid_frac < MIN_VALID_FRAC:
                # Still run the forward pass to keep LSTM state consistent
                with torch.no_grad(), autocast("cuda"):
                    _ = model(img, bbox_t, crop_h, crop_w, obj_img=obj_img)
                del img, bbox_t, obj_img, gt_depth, gt_norm, frame
                total_frames += 1
                chunk_frames += 1
                continue

            if is_train:
                with autocast("cuda"):
                    pred, log_unc = model(img, bbox_t, crop_h, crop_w, obj_img=obj_img)
                    pred = pred.squeeze(0)
                    log_unc = log_unc.squeeze(0)
                # Uncertainty-aware loss for gradient; weight by valid_frac
                loss = uncertainty_aware_loss(pred.float(), log_unc.float(), gt_norm.float()) * valid_frac
                chunk_loss = chunk_loss + loss
                
                # Track unweighted MAE for reporting
                with torch.no_grad():
                    valid_mask = gt_norm > 0
                    if valid_mask.any():
                        frame_mae = (pred[valid_mask].float() - gt_norm[valid_mask].float()).abs().mean().item()
                        report_loss_sum += frame_mae
                        report_count += 1
            else:
                with torch.no_grad(), autocast("cuda"):
                    pred, log_unc = model(img, bbox_t, crop_h, crop_w, obj_img=obj_img)
                    pred = pred.squeeze(0)
                    log_unc = log_unc.squeeze(0)
                frame_loss = masked_combined_loss(pred.float(), gt_norm.float())
                chunk_loss = chunk_loss + frame_loss
                report_loss_sum += frame_loss.item()
                report_count += 1

                # --- Per-pixel validation metrics ---
                valid = gt_norm > 0
                if valid.any():
                    p = pred[valid].float()
                    g = gt_norm[valid].float()
                    n_px = p.numel()

                    sum_mae += (p - g).abs().sum().item()
                    sum_sq_err += ((p - g) ** 2).sum().item()
                    ratio = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
                    sum_delta1 += (ratio < 1.25).sum().item()
                    total_val_pixels += n_px

                # --- Capture frames for video ---
                if t % SNAP_STRIDE == 0:
                    gt_norm_np = gt_norm.float().cpu().numpy()
                    pred_np = pred.float().cpu().numpy()
                    snap_valid = gt_norm_np > 0
                    snap_valid_pct = float(snap_valid.mean()) * 100.0
                    if snap_valid.any():
                        sp, sg = pred_np[snap_valid], gt_norm_np[snap_valid]
                        snap_mae = float(np.abs(sp - sg).mean())
                        snap_rmse = float(np.sqrt(((sp - sg) ** 2).mean()))
                        snap_ratio = np.maximum(sp / np.clip(sg, 1e-6, None),
                                                sg / np.clip(sp, 1e-6, None))
                        snap_d1 = float((snap_ratio < 1.25).mean())
                    else:
                        snap_mae = snap_rmse = snap_d1 = 0.0
                    scene_frames.append({
                        "scene": scene_meta["scene"],
                        "frame_idx": t,
                        "T": T,
                        "rgb_full": frame["rgb"].permute(1, 2, 0).numpy(),
                        "rgb_crop": frame["rgb_crop"].permute(1, 2, 0).numpy(),
                        "pred": pred_np,
                        "gt": gt_norm_np,
                        "uncertainty": np.exp(log_unc.float().cpu().numpy()),
                        "mae": snap_mae,
                        "rmse": snap_rmse,
                        "delta1": snap_d1,
                        "valid_pct": snap_valid_pct,
                    })

            del img, bbox_t, obj_img, gt_depth, gt_norm, pred, frame
            chunk_frames += 1
            total_frames += 1

            # --- TBTT boundary ---
            if is_train and chunk_frames % TBTT_LEN == 0:
                avg_loss = chunk_loss / chunk_frames
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(avg_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += chunk_loss.item()
                chunk_loss = torch.tensor(0.0, device=device)
                chunk_frames = 0

                safe_detach_head(model.ctx_head)
                safe_detach_head(model.shape_head)
                safe_detach_head(model.obj_head)
                torch.cuda.empty_cache()

            # Periodic frame progress
            if (t + 1) % PRINT_EVERY == 0:
                running = report_loss_sum / max(report_count, 1)
                print(f"    [{((t+1) / T) * 100:.1f}%] Frame {t+1}/{T}  avg_mae={running:.5f}", end='\r')

        # Flush remaining chunk (train: backprop leftover; val: always accumulate)
        leftover = chunk_frames % TBTT_LEN
        if is_train and leftover > 0 and chunk_loss.requires_grad:
            avg_loss = chunk_loss / leftover
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(avg_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += chunk_loss.item()

        del chunk_loss
        torch.cuda.empty_cache()

        scene_dt = time.time() - scene_t0
        if scene_frames:
            snapshots.append(scene_frames)
        scene_avg = report_loss_sum / max(report_count, 1)
        print(f"  [{mode}] Scene {si+1}/{n_scenes} ({scene_meta['scene']}) "
              f"| {T} frames in {scene_dt:.1f}s "
              f"| avg_mae={scene_avg:.5f}")

    epoch_dt = time.time() - epoch_t0
    avg_mae = report_loss_sum / max(report_count, 1)
    print(f"  {mode} done — {total_frames} frames ({report_count} scored) in {epoch_dt:.1f}s | avg_mae={avg_mae:.6f}")

    if not is_train and total_val_pixels > 0:
        metrics = {
            "mae":    sum_mae / total_val_pixels,
            "rmse":   (sum_sq_err / total_val_pixels) ** 0.5,
            "delta1": sum_delta1 / total_val_pixels,
        }
        return avg_mae, metrics, snapshots

    return avg_mae, None, None

# --- Training Loop (TBTT, scene-level streaming via DataLoader) ---
best_val_loss = float("inf")
history = {"train_loss": [], "val_loss": [], "mae": [], "rmse": [], "delta1": [], "lr": []}

PLOT_DIR = os.path.join(_PROJECT_ROOT, "training_plots")
SNAP_DIR = os.path.join(_PROJECT_ROOT, "training_snapshots")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

def save_training_plots(history, plot_dir):
    """Save live training curves to disk after each epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (0,0) Train & Val Loss ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss", markersize=3)
    ax.plot(epochs, history["val_loss"],   "o-", label="Val Loss",   markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (L1)")
    ax.set_title("Train / Val Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # ── (0,1) MAE & RMSE ─────────────────────────────────────────────────
    ax = axes[0, 1]
    if history["mae"]:
        m_epochs = range(1, len(history["mae"]) + 1)
        ax.plot(m_epochs, history["mae"],  "o-", label="MAE",  markersize=3)
        ax.plot(m_epochs, history["rmse"], "o-", label="RMSE", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error")
    ax.set_title("Val MAE / RMSE"); ax.legend(); ax.grid(True, alpha=0.3)

    # ── (1,0) δ < 1.25 ───────────────────────────────────────────────────
    ax = axes[1, 0]
    if history["delta1"]:
        d_epochs = range(1, len(history["delta1"]) + 1)
        ax.plot(d_epochs, [d * 100 for d in history["delta1"]], "o-",
                label="δ < 1.25", color="green", markersize=3)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Val δ < 1.25"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── (1,1) Learning Rate ───────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], "o-", color="orange", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule"); ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    fig.suptitle(f"Training Progress — Epoch {len(history['train_loss'])}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plot_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def save_val_snapshots(snapshots, epoch, snap_dir):
    """Save one MP4 video per val scene for this epoch.

    Each frame: Full RGB | Cropped RGB | Predicted Depth | GT Depth
    Saved under  snap_dir/epoch_XX/<scene>.mp4
    snapshots is a list of per-scene lists of frame dicts.
    """
    epoch_dir = os.path.join(snap_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    for scene_frames in snapshots:
        if not scene_frames:
            continue
        scene_name = scene_frames[0]["scene"]

        # Render all frames to numpy arrays first to get consistent size
        rendered = []
        for snap in scene_frames:
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            axes[0].imshow(snap["rgb_full"])
            axes[0].set_title("Full RGB")

            axes[1].imshow(snap["rgb_crop"])
            axes[1].set_title("Cropped RGB")

            gt = snap["gt"]
            pred = snap["pred"]
            gt_valid = gt[gt > 0]
            vmin = min(gt_valid.min() if gt_valid.size > 0 else 0, pred.min())
            vmax = max(gt.max(), pred.max())

            axes[2].imshow(pred, cmap="inferno", vmin=vmin, vmax=vmax)
            axes[2].set_title("Predicted Depth")

            im_gt = axes[3].imshow(gt, cmap="inferno", vmin=vmin, vmax=vmax)
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
                f"δ<1.25={snap['delta1']:.2%}  valid={snap.get('valid_pct', 100):.0f}%  |  epoch {epoch}",
                fontsize=13,
            )

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            rendered.append(buf)
            plt.close(fig)

        # Write video with cv2
        import cv2
        h, w = rendered[0].shape[:2]
        out_path = os.path.join(epoch_dir, f"{scene_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 10, (w, h))
        for frame_rgb in rendered:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        writer.release()

    return epoch_dir

for epoch in range(start_epoch, EPOCHS + 1):
    print(f"\n{'='*50}\nEpoch {epoch}/{EPOCHS}\n{'='*50}")
    
    train_loss, _, _ = run_epoch(train_loader, model, optimizer, scaler, device, is_train=True)
    print()
    val_loss, val_metrics, val_snapshots = run_epoch(val_loader, model, optimizer, scaler, device, is_train=False)
    print()

    scheduler.step()
    cur_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["lr"].append(cur_lr)
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
              f"δ<1.25: {val_metrics['delta1']:.2%}")

    save_training_plots(history, PLOT_DIR)
    print(f"  Plots saved to {PLOT_DIR}/training_curves.png")

    if val_snapshots:
        snap_epoch_dir = save_val_snapshots(val_snapshots, epoch, SNAP_DIR)
        print(f"  Snapshots saved to {snap_epoch_dir}/ ({len(val_snapshots)} images)")