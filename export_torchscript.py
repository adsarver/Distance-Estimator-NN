"""Export a DistanceNN checkpoint to TorchScript format.

Usage:
    python export_torchscript.py --checkpoint best_model.pt [--output model_scripted.pt]
"""

import argparse
import os
import torch

from model import DistanceNN

parser = argparse.ArgumentParser(description="Export DistanceNN to TorchScript")
parser.add_argument("--checkpoint", required=True, help="Path to .pt weight file")
parser.add_argument("--output", default=None,
                    help="Output TorchScript file (default: <checkpoint>_scripted.pt)")
parser.add_argument("--img-size", type=int, default=480, help="Model input resolution")
parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size")
parser.add_argument("--lstm-layers", type=int, default=1, help="Number of LSTM layers")
args = parser.parse_args()

if args.output is None:
    base, _ = os.path.splitext(args.checkpoint)
    args.output = f"{base}_scripted.pt"

device = torch.device("cpu")

model = DistanceNN(
    hidden_size=args.hidden_size,
    lstm_num_layers=args.lstm_layers,
    img_size=args.img_size,
    # use_obj_head=False
).to(device)

# Load weights
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
else:
    model.load_state_dict(ckpt, strict=False)
    print(f"Loaded weights from {args.checkpoint}")

model.eval()

# Script the model
scripted = torch.jit.script(model)
scripted.save(args.output)
print(f"Saved TorchScript model to {args.output}")
