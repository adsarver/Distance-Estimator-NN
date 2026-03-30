import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from glob import glob
from collections import defaultdict

# Designed to work with the RGBD dataset from, will load one scene at a time

# https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/

class RGBDDataset(Dataset):
    def __init__(self, img_dir, transform=None, scene_names=None, random_seed=42):
        self.img_dir = img_dir
        self.transform = transform
        self.rng = np.random.default_rng(random_seed)

        # Group frames by scene
        all_files = sorted(glob(os.path.join(img_dir, '*/*-color.png')))
        scenes = defaultdict(list)
        for f in all_files:
            scene = os.path.basename(os.path.dirname(f))
            scenes[scene].append(f)

        # Optionally filter to a subset of scenes
        if scene_names is not None:
            self.scene_names = [s for s in scene_names if s in scenes]
        else:
            self.scene_names = sorted(scenes.keys())

        self.scenes = {s: scenes[s] for s in self.scene_names}

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]
        frame_paths = self.scenes[scene_name]
        T = len(frame_paths)

        # Smooth random bbox trajectory across the scene
        bbox_start = self._random_bbox()
        bbox_end = self._random_bbox()

        rgb_frames = []      # list of (3, H, W) tensors
        depth_frames = []    # list of (1, H, W) tensors
        rgb_crops = []       # list of (3, crop_h, crop_w)
        depth_crops = []     # list of (1, crop_h, crop_w)
        bbox_seq = []        # list of (4,) tensors
        crop_dims = []       # list of (crop_h, crop_w) tuples

        for t, path in enumerate(frame_paths):
            rgb, depth = self._load_rgbd(path)
            _, H, W = rgb.shape

            alpha = t / max(T - 1, 1)
            bbox = (1 - alpha) * bbox_start + alpha * bbox_end
            cx, cy, bw, bh = bbox

            top = max(0, int((cy - bh / 2) * H))
            left = max(0, int((cx - bw / 2) * W))
            bottom = min(H, top + int(bh * H))
            right = min(W, left + int(bw * W))

            rgb_frames.append(rgb)
            depth_frames.append(depth)
            rgb_crops.append(rgb[:, top:bottom, left:right])
            depth_crops.append(depth[:, top:bottom, left:right])
            bbox_seq.append(torch.tensor(bbox, dtype=torch.float32))
            crop_dims.append((bottom - top, right - left))

        return {
            "scene": scene_name,
            "rgb": rgb_frames,
            "depth": depth_frames,
            "rgb_crops": rgb_crops,
            "depth_crops": depth_crops,
            "bboxes": bbox_seq,
            "crop_dims": crop_dims,
        }

    # Helper fns

    def _load_rgbd(self, color_path):
        depth_path = color_path.replace('-color.png', '-depth.png')
        rgb = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        return rgb, depth

    def _random_bbox(self):
        w = 0.65 * self.rng.random() + 0.05
        h = 0.65 * self.rng.random() + 0.05
        cx = (1 - w) * self.rng.random() + (0.5 * w)
        cy = (1 - h) * self.rng.random() + (0.5 * h)

        return np.array([cx, cy, w, h])


def scene_collate_fn(batch):
    return batch[0]