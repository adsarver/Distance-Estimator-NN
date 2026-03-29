import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from glob import glob

# Designed to work with the RGBD dataset from, will load one scene at a time

# https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/

class RGBDDataset(Dataset):
    def __init__(self, img_dir, transform=None, random_seed=42, max_frames=30, frame_interval=10):
        self.img_dir = img_dir
        self.transform = transform
        self.max_frames = max_frames
        self.frame_interval = frame_interval

        self.rng = np.random.default_rng(random_seed)

        # Get sorted color frames
        self.color_files = sorted(glob(os.path.join(img_dir, '*-color.png')))

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        # Random start frame
        start_idx = self.rng.integers(0, len(self.color_files))
        end_idx = min(start_idx + self.max_frames * self.frame_interval, len(self.color_files))

        frame_paths = self.color_files[start_idx:end_idx:self.frame_interval]
        T = len(frame_paths)

        # Generate start & end bbox
        bbox_start = self.random_bbox()
        bbox_end   = self.random_bbox()

        rgb_seq = []
        depth_crops = []
        bbox_seq = []

        for t, path in enumerate(frame_paths):
            rgb, depth = self.load_rgbd(path)

            _, H, W = rgb.shape

            # Linear interpolation
            alpha = t / max(T - 1, 1)
            bbox = (1 - alpha) * bbox_start + alpha * bbox_end

            cx, cy, bw, bh = bbox

            # Convert to pixel coords
            top = int((cy - bh / 2) * H)
            left = int((cx - bw / 2) * W)
            height = int(bh * H)
            width = int(bw * W)

            # Clamp
            top = max(0, top)
            left = max(0, left)
            bottom = min(H, top + height)
            right = min(W, left + width)

            crop = depth[:, top:bottom, left:right]

            rgb_seq.append(rgb)
            depth_crops.append(crop)
            bbox_seq.append(torch.tensor(bbox, dtype=torch.float32))

        rgb_seq = torch.stack(rgb_seq)              # (T, 3, H, W)
        bbox_seq = torch.stack(bbox_seq)            # (T, 4)

        return rgb_seq, depth_crops, bbox_seq
    
    def load_rgbd(self, color_path):
        depth_path = color_path.replace('-color.png', '-depth.png')

        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float()

        return rgb, depth

    def random_bbox(self):
        w = 0.65 * self.rng.random() + 0.05
        h = 0.65 * self.rng.random() + 0.05

        cx = (1 - w) * self.rng.random() + (0.5 * w)
        cy = (1 - h) * self.rng.random() + (0.5 * h)

        return np.array([cx, cy, w, h])