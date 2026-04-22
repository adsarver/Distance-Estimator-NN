"""Compare the three trained depth estimators on the validation split.

Loads the three checkpoints (LSTM model + two baselines) and reports:

  * δ < 1.05   — ratio accuracy (higher is better)
  * MSE        — mean squared error on normalised depth (lower is better)
  * Params     — trainable parameter count
  * Inference  — mean forward-pass latency (ms / sample, batch=1, fp16, GPU warm)

Usage:
    python compare_models.py \
        --lstm best_model.pt \
        --bts  best_model_bts.pt \
        --adabins best_model_adabins.pt

The LSTM model uses the same cropped-object + full-image + bbox pipeline as
``train.py``; BTS/AdaBins take the full RGB frame resized to their training
resolution.  All models are evaluated on the same validation scenes so the
metrics are comparable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

from baseline_train import AdaBins, BTS, FrameDepthDataset  # noqa: E402
from data import RGBDDataset, scene_collate_fn  # noqa: E402
from model import DistanceNN  # noqa: E402


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _val_scene_names(val_split: float = 0.20) -> list[str]:
    """Reproduce the train.py / baseline_train.py val-split.

    Uses the legacy ``np.random.seed(42) + np.random.shuffle`` sequence so the
    held-out scenes are identical to the ones seen by ``train.py`` during
    validation.
    """
    all_files = sorted(glob(os.path.join(DATA_DIR, "*/*-color.png")))
    scenes = defaultdict(list)
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


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _measure_latency(model: torch.nn.Module, sample_fn, *, device: torch.device,
                     warmup: int = 10, iters: int = 50) -> float:
    """Return mean per-call latency in milliseconds (CUDA-synchronised)."""
    model.eval()
    # Warm up (compile, cache kernels, cuDNN auto-tune).
    for _ in range(warmup):
        args = sample_fn()
        with autocast("cuda", dtype=torch.float16):
            _ = model(*args)
    torch.cuda.synchronize()

    starters = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    enders = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        args = sample_fn()
        starters[i].record()
        with autocast("cuda", dtype=torch.float16):
            _ = model(*args)
        enders[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starters, enders)]
    return float(np.mean(times_ms))


# ----------------------------------------------------------------------
# Metric accumulators (match train.py / baseline_train.py definitions)
# ----------------------------------------------------------------------
class MetricAccumulator:
    """Accumulates δ<1.05 and MSE over an arbitrary number of pixels."""

    DELTA_THRESH = 1.05

    def __init__(self):
        self.sq_sum = 0.0
        self.d1 = 0
        self.px = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> None:
        # Matches train.py L409-416 and baseline_train.py L614-617 exactly:
        #   - raw (p - g) for MSE, no clamp
        #   - ratio = max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
        mask = valid > 0.5
        if not mask.any():
            return
        p = pred[mask].float()
        g = gt[mask].float()
        self.sq_sum += float(((p - g) ** 2).sum().item())
        ratio = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
        self.d1 += int((ratio < self.DELTA_THRESH).sum().item())
        self.px += int(mask.sum().item())

    def summary(self) -> dict:
        if self.px == 0:
            return {"mse": float("nan"), "delta1": float("nan"), "pixels": 0}
        return {"mse": self.sq_sum / self.px,
                "delta1": self.d1 / self.px,
                "pixels": self.px}


# ----------------------------------------------------------------------
# Per-model evaluators
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_baseline(model: torch.nn.Module, val_scenes: list[str],
                  img_size: int, device: torch.device) -> dict:
    ds = FrameDepthDataset(DATA_DIR, val_scenes, img_size=img_size, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False,
                                         num_workers=4, pin_memory=True)
    acc = MetricAccumulator()
    model.eval()
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.float16):
            pred = model(rgb)
        acc.update(pred, depth, valid)
    return acc.summary()


@torch.no_grad()
def eval_lstm(model: DistanceNN, val_scenes: list[str], img_size: int,
              device: torch.device, min_valid_frac: float = 0.25) -> dict:
    """Walk validation scenes in temporal order (as train.py does for val)."""
    ds = RGBDDataset(DATA_DIR, scene_names=val_scenes, random_seed=123)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         collate_fn=scene_collate_fn, num_workers=1)
    acc = MetricAccumulator()
    model.eval()
    for scene_meta in loader:
        model.reset_lstm()
        T = scene_meta["T"]
        for t in range(T):
            frame = ds.load_frame(scene_meta, t)
            rgb_full = F.interpolate(frame["rgb"].unsqueeze(0), size=(img_size, img_size),
                                     mode="bilinear", align_corners=False).to(device)
            crop = frame["rgb_crop"].unsqueeze(0)
            crop_h, crop_w = frame["crop_dim"]
            if crop_h == 0 or crop_w == 0:
                continue
            obj = F.interpolate(crop, size=(img_size, img_size),
                                mode="bilinear", align_corners=False).to(device)
            bbox_t = frame["bbox"].unsqueeze(0).to(device)

            depth_full = frame["depth"]  # (1,H,W) uint16 as float
            depth_crop = frame["depth_crop"]
            gt_norm = (depth_crop.float() / 65535.0).to(device)  # (1, ch, cw)
            valid = (gt_norm > 0).float()
            if float(valid.mean().item()) < min_valid_frac:
                # Still run forward to update LSTM state, but skip scoring.
                with autocast("cuda", dtype=torch.float16):
                    _ = model(rgb_full, bbox_t, int(crop_h), int(crop_w), obj_img=obj)
                continue

            with autocast("cuda", dtype=torch.float16):
                pred, _ = model(rgb_full, bbox_t, int(crop_h), int(crop_w), obj_img=obj)
            # model returns (1, ch, cw); gt_norm is (1, ch, cw) — keep aligned.
            acc.update(pred, gt_norm, valid)

    return acc.summary()


# ----------------------------------------------------------------------
# Latency samplers
# ----------------------------------------------------------------------
def _baseline_sampler(img_size: int, device):
    def _s():
        return (torch.randn(1, 3, img_size, img_size, device=device),)
    return _s


def _lstm_sampler(img_size: int, device, model: DistanceNN):
    crop_h = crop_w = img_size
    bbox = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device, dtype=torch.float32)

    def _s():
        # Reset LSTM hidden state so timing reflects steady-state cost.
        model.reset_lstm()
        img = torch.randn(1, 3, img_size, img_size, device=device)
        obj = torch.randn(1, 3, img_size, img_size, device=device)
        return (img, bbox, crop_h, crop_w, obj)
    return _s


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lstm", default="best_model.pt", help="LSTM checkpoint (train.py)")
    p.add_argument("--bts", default="best_model_bts.pt", help="BTS checkpoint")
    p.add_argument("--adabins", default="best_model_adabins.pt", help="AdaBins checkpoint")
    p.add_argument("--lstm-img-size", type=int, default=480,
                   help="Image size used when training the LSTM model")
    p.add_argument("--bts-img-size", type=int, default=480)
    p.add_argument("--adabins-img-size", type=int, default=480)
    p.add_argument("--val-split", type=float, default=0.20)
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip validation-set evaluation (only report params + latency)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    val_scenes = _val_scene_names(args.val_split)
    print(f"Validation scenes ({len(val_scenes)}): {val_scenes}\n")

    results: list[dict] = []

    configs = [
        ("LSTM (DistanceNN)", args.lstm, "lstm", args.lstm_img_size),
        ("BTS", args.bts, "bts", args.bts_img_size),
        ("AdaBins", args.adabins, "adabins", args.adabins_img_size),
    ]

    for label, ckpt_path, kind, img_size in configs:
        print(f"=== {label} ===")
        if not os.path.isfile(ckpt_path):
            print(f"  [missing] {ckpt_path} — skipping\n")
            results.append({"name": label, "missing": True, "ckpt": ckpt_path})
            continue

        # --- Build model
        if kind == "lstm":
            model = DistanceNN(hidden_size=128, lstm_num_layers=1, img_size=img_size).to(device)
        elif kind == "bts":
            model = BTS(pretrained=False).to(device)
        elif kind == "adabins":
            model = AdaBins(pretrained=False).to(device)
        else:
            raise ValueError(kind)

        state = _load_state_dict(ckpt_path, device)
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  [warn] missing keys: {len(missing)}")
            if unexpected:
                print(f"  [warn] unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"  [error] failed to load state dict: {e}")
            results.append({"name": label, "missing": True, "ckpt": ckpt_path})
            continue

        n_params = _count_params(model)
        print(f"  Params: {n_params:,}")

        # --- Metrics on val split
        if args.skip_eval:
            metrics = {"mse": float("nan"), "delta1": float("nan"), "pixels": 0}
        else:
            t0 = time.time()
            if kind == "lstm":
                metrics = eval_lstm(model, val_scenes, img_size, device)
            else:
                metrics = eval_baseline(model, val_scenes, img_size, device)
            print(f"  Eval: δ<1.05={metrics['delta1']:.4%}  MSE={metrics['mse']:.6e}  "
                  f"({metrics['pixels']:,} px, {time.time()-t0:.1f}s)")

        # --- Latency (batch=1, fp16)
        if args.skip_latency:
            latency_ms = float("nan")
        else:
            if kind == "lstm":
                sampler = _lstm_sampler(img_size, device, model)
            else:
                sampler = _baseline_sampler(img_size, device)
            latency_ms = _measure_latency(model, sampler, device=device)
            print(f"  Latency: {latency_ms:.2f} ms/frame (batch=1, fp16)")

        results.append({
            "name": label,
            "ckpt": ckpt_path,
            "img_size": img_size,
            "params": n_params,
            "delta1": metrics["delta1"],
            "mse": metrics["mse"],
            "latency_ms": latency_ms,
            "missing": False,
        })
        print()

        del model
        torch.cuda.empty_cache()

    # --- Summary table
    print("=" * 86)
    print(f"{'Model':<24} {'Params':>14} {'δ<1.05':>10} {'MSE':>14} {'Latency(ms)':>13} {'FPS':>7}")
    print("-" * 86)
    for r in results:
        if r.get("missing"):
            print(f"{r['name']:<24} {'(checkpoint missing: ' + r['ckpt'] + ')'}")
            continue
        fps = 1000.0 / r["latency_ms"] if r["latency_ms"] and not np.isnan(r["latency_ms"]) else float("nan")
        d1 = f"{r['delta1']*100:.2f}%" if not np.isnan(r["delta1"]) else "n/a"
        mse = f"{r['mse']:.4e}" if not np.isnan(r["mse"]) else "n/a"
        lat = f"{r['latency_ms']:.2f}" if not np.isnan(r["latency_ms"]) else "n/a"
        fps_s = f"{fps:.1f}" if not np.isnan(fps) else "n/a"
        print(f"{r['name']:<24} {r['params']:>14,} {d1:>10} {mse:>14} {lat:>13} {fps_s:>7}")
    print("=" * 86)

    # --- CSV output for external plotting
    csv_path = os.path.join(_PROJECT_ROOT, "model_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("model,params,delta1,mse,latency_ms,fps,ckpt\n")
        for r in results:
            if r.get("missing"):
                continue
            fps = 1000.0 / r["latency_ms"] if not np.isnan(r["latency_ms"]) else float("nan")
            f.write(f"{r['name']},{r['params']},{r['delta1']:.6f},{r['mse']:.6e},"
                    f"{r['latency_ms']:.4f},{fps:.4f},{r['ckpt']}\n")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
