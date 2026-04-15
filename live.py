"""
Live webcam depth estimation using a TorchScript model.

Usage:
    python live.py [--model best_model_scripted.pt] [--camera 0]

Streams side-by-side (RGB | Predicted Depth | Uncertainty) to a window.
Press 'q' to quit.
"""

import argparse
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Live webcam depth estimation")
parser.add_argument("--model", default="best_model_scripted.pt",
                    help="Path to TorchScript model (default: best_model_scripted.pt)")
parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
parser.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("CX", "CY", "W", "H"),
                    help="Fixed crop bbox in normalized coords [0-1] (default: center 0.6x0.6)")
parser.add_argument("--img-size", type=int, default=480, help="Model input resolution")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = args.img_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = torch.jit.load(args.model, map_location=device)
model.eval()
model.reset_lstm()
print(f"Loaded TorchScript model from {args.model}")

# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"Error: cannot open camera {args.camera}")
    sys.exit(1)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera: {frame_w}x{frame_h}")

# ── Bbox setup ────────────────────────────────────────────────────────────────
if args.bbox is not None:
    bbox_np = np.array(args.bbox, dtype=np.float32)
else:
    bbox_np = np.array([0.5, 0.5, 0.6, 0.6], dtype=np.float32)

cx, cy, bw, bh = bbox_np
top = max(0, int((cy - bh / 2) * frame_h))
left = max(0, int((cx - bw / 2) * frame_w))
bottom = min(frame_h, top + int(bh * frame_h))
right = min(frame_w, left + int(bw * frame_w))
crop_h = bottom - top
crop_w = right - left
print(f"Crop region: ({left},{top})-({right},{bottom}) = {crop_w}x{crop_h}")

# ── Colormap helper ───────────────────────────────────────────────────────────
def apply_colormap(gray, cmap=cv2.COLORMAP_INFERNO):
    gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cmap)

# ── Sliding depth scale (EMA-smoothed percentile normalization) ───────────────
EMA_ALPHA = 0.15          # smoothing factor (higher = faster adaptation)
PCT_LO, PCT_HI = 2, 98   # percentiles to trim outliers
ema_lo, ema_hi = None, None

def normalize_depth(depth_raw):
    """Rescale depth to [0,1] using EMA-smoothed percentile bounds."""
    global ema_lo, ema_hi
    lo = np.percentile(depth_raw, PCT_LO)
    hi = np.percentile(depth_raw, PCT_HI)
    if ema_lo is None:
        ema_lo, ema_hi = lo, hi
    else:
        ema_lo = ema_lo + EMA_ALPHA * (lo - ema_lo)
        ema_hi = ema_hi + EMA_ALPHA * (hi - ema_hi)
    span = max(ema_hi - ema_lo, 1e-6)
    return (depth_raw - ema_lo) / span

# ── Main loop ─────────────────────────────────────────────────────────────────
print("Press 'q' to quit\n")
bbox_t = torch.tensor(bbox_np, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        # Prepare model inputs
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img = F.interpolate(rgb_tensor.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False).to(device)

        crop_rgb = rgb[top:bottom, left:right]
        crop_bgr = bgr[top:bottom, left:right]
        crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
        obj_img = crop_tensor.unsqueeze(0).to(device)

        # Forward pass
        with torch.amp.autocast("cuda"):
            depth, log_unc = model(img, bbox_t, crop_h, crop_w, obj_img)

        depth_np = depth.squeeze(0).float().cpu().numpy()
        unc_np = torch.exp(log_unc.squeeze(0)).float().cpu().numpy()

        # Build panels
        panel_rgb = crop_bgr.copy()
        depth_norm = normalize_depth(depth_np)
        panel_depth = apply_colormap(depth_norm)
        unc_max = max(unc_np.max(), 1e-6)
        panel_unc = apply_colormap(unc_np / unc_max, cv2.COLORMAP_MAGMA)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel_rgb, "RGB", (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel_depth, f"Depth [{ema_lo:.2f}-{ema_hi:.2f}]", (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel_unc, f"Unc (max={unc_max:.3f})", (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        canvas = np.concatenate([panel_rgb, panel_depth, panel_unc], axis=1)

        cv2.imshow("Live Depth Estimation", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit by user")
            break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Done.")
