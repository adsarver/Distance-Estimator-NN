"""
Real-time depth estimation from a video file.

Usage:
    python test.py --video path/to/video.mp4 [--checkpoint best_model.pt] [--output output.mp4]

Streams side-by-side (RGB | Predicted Depth | Uncertainty) to a window
and saves the result to a video file. Press 'q' to quit early.
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import DistanceNN

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run depth estimation on a video")
parser.add_argument("--video", required=True, help="Path to input video file")
parser.add_argument("--checkpoint", default="best_model.pt", help="Model checkpoint path")
parser.add_argument("--output", default=None, help="Output video path (default: <input>_depth.mp4)")
parser.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("CX", "CY", "W", "H"),
                    help="Fixed crop bbox in normalized coords [0-1] (default: center 0.6x0.6)")
parser.add_argument("--img-size", type=int, default=256, help="Model input resolution")
parser.add_argument("--no-display", action="store_true", help="Skip live window display")
args = parser.parse_args()

if args.output is None:
    base, ext = os.path.splitext(args.video)
    args.output = f"{base}_depth.mp4"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = args.img_size
HIDDEN_SIZE = 128
LSTM_LAYERS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = DistanceNN(
    hidden_size=HIDDEN_SIZE,
    lstm_num_layers=LSTM_LAYERS,
    img_size=IMG_SIZE,
).to(device)

ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded model from checkpoint (epoch {ckpt.get('epoch', '?')})")
else:
    model.load_state_dict(ckpt, strict=False)
    print(f"Loaded model weights from {args.checkpoint}")

model.eval()
model.reset_lstm()

# ── Video input ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Error: cannot open video '{args.video}'")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Input: {frame_w}x{frame_h} @ {fps:.1f} fps, {total_frames} frames")

# ── Bbox setup ────────────────────────────────────────────────────────────────
if args.bbox is not None:
    bbox_np = np.array(args.bbox, dtype=np.float32)
else:
    bbox_np = np.array([0.5, 0.5, 0.6, 0.6], dtype=np.float32)  # center crop

cx, cy, bw, bh = bbox_np
top = max(0, int((cy - bh / 2) * frame_h))
left = max(0, int((cx - bw / 2) * frame_w))
bottom = min(frame_h, top + int(bh * frame_h))
right = min(frame_w, left + int(bw * frame_w))
crop_h = bottom - top
crop_w = right - left
print(f"Crop region: ({left},{top})-({right},{bottom}) = {crop_w}x{crop_h}")

# ── Video output ──────────────────────────────────────────────────────────────
# Output layout: [RGB with bbox | Depth | Uncertainty], each panel = crop size
panel_h = crop_h
panel_w = crop_w
out_w = panel_w * 3
out_h = panel_h

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))
if not writer.isOpened():
    print(f"Error: cannot create output video '{args.output}'")
    sys.exit(1)

print(f"Output: {args.output} ({out_w}x{out_h} @ {fps:.1f} fps)")
print(f"Press 'q' to quit\n")

# ── Colormap helper ───────────────────────────────────────────────────────────
def apply_colormap(gray, cmap=cv2.COLORMAP_INFERNO):
    """Convert single-channel float [0,1] to BGR colormap uint8."""
    gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cmap)


# ── Main loop ─────────────────────────────────────────────────────────────────
frame_idx = 0
bbox_t = torch.tensor(bbox_np, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        # Prepare model inputs
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        img = F.interpolate(rgb_tensor.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False).to(device)

        crop_rgb = rgb[top:bottom, left:right]
        crop_bgr = bgr[top:bottom, left:right]
        crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0  # (3, cH, cW)
        obj_img = crop_tensor.unsqueeze(0).to(device)

        # Forward pass
        with torch.amp.autocast("cuda"):
            depth, log_unc, pred_scale = model(img, bbox_t, crop_h, crop_w, obj_img=obj_img)

        depth_np = depth.squeeze(0).float().cpu().numpy()   # (cH, cW) in [0,1]
        unc_np = torch.exp(log_unc.squeeze(0)).float().cpu().numpy()

        # Build panels
        # Panel 1: RGB crop with bbox label
        panel_rgb = crop_bgr.copy()

        # Panel 2: Depth prediction
        panel_depth = apply_colormap(depth_np)

        # Panel 3: Uncertainty (normalize to [0,1] for display)
        unc_max = max(unc_np.max(), 1e-6)
        panel_unc = apply_colormap(unc_np / unc_max, cv2.COLORMAP_MAGMA)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel_rgb, "RGB", (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel_depth, "Depth", (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel_unc, f"Unc (max={unc_max:.3f})", (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Concatenate
        canvas = np.concatenate([panel_rgb, panel_depth, panel_unc], axis=1)

        # Write output
        writer.write(canvas)

        # Display
        if not args.no_display:
            cv2.imshow("Depth Estimation", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit by user")
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
writer.release()
if not args.no_display:
    cv2.destroyAllWindows()
print(f"\nDone. Processed {frame_idx} frames → {args.output}")
