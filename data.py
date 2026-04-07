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
    def __init__(self, img_dir, transforms=None, scene_names=None, random_seed=42):
        self.img_dir = img_dir
        self.transforms = transforms
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

        # Tuple (scene_name, frame_paths, pkgd_tf)
        if self.transforms != None:
            self.scene_list = [(name, pkgd_tf) for name in self.scene_names for pkgd_tf in self.transforms]
        else:
            self.scene_list = [(name, None) for name in self.scene_names]

    def _get_scene_info(self, idx):

        scene_name, pkgd_tf = self.scene_list[idx]
        frame_paths = self.scenes[scene_name]

        return scene_name, frame_paths, pkgd_tf

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        """Return lightweight scene metadata — no images loaded yet."""

        scene_name, frame_paths, pkgd_tf = self._get_scene_info(idx)
        T = len(frame_paths)

        bbox_start = self._random_bbox()
        bbox_end = self._random_bbox()

        transform = None
        # Choose a random transform to apply to the data
        if self.transforms != None:
            transform = self.transforms[self.rng.integers(0, len(self.transforms))]

        return {
            "scene": scene_name,
            "frame_paths": frame_paths,
            "T": T,
            "bbox_start": bbox_start,
            "bbox_end": bbox_end,
            "pkgd_tf": pkgd_tf
        }

    def load_frame(self, scene_meta, t):
        """Load a single frame on demand. Returns dict with tensors for one timestep."""
        path = scene_meta["frame_paths"][t]
        T = scene_meta["T"]
        bbox_start = scene_meta["bbox_start"]
        bbox_end = scene_meta["bbox_end"]
        pkgd_tf = scene_meta["pkgd_tf"]

        rgb, depth = self._load_rgbd(path, pkgd_tf)
        _, H, W = rgb.shape

        alpha = t / max(T - 1, 1)
        bbox = (1 - alpha) * bbox_start + alpha * bbox_end
        cx, cy, bw, bh = bbox

        top = max(0, int((cy - bh / 2) * H))
        left = max(0, int((cx - bw / 2) * W))
        bottom = min(H, top + int(bh * H))
        right = min(W, left + int(bw * W))

        return {
            "rgb": rgb,
            "depth": depth,
            "rgb_crop": rgb[:, top:bottom, left:right],
            "depth_crop": depth[:, top:bottom, left:right],
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "crop_dim": (bottom - top, right - left),
        }

    # Helper fns

    def _load_rgbd(self, color_path, pkgd_transform=None):
        depth_path = color_path.replace('-color.png', '-depth.png')
        if not os.path.isfile(depth_path):
            depth_path = color_path.replace('-color.png', '-aligned-depth.png')
        rgb_raw = cv2.imread(color_path)
        if rgb_raw is None:
            # Corrupt/unreadable color image — return dummy tensors
            return torch.zeros(3, 480, 640), torch.zeros(1, 480, 640)
        rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # uint8 (C,H,W)
        if depth is not None:
            depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        else:
            # No depth available — return zeros (will be skipped by valid_frac check)
            depth = torch.zeros(1, rgb.shape[1], rgb.shape[2], dtype=torch.float32)

        # Apply the transforms on uint8 rgb (required by v2 transforms like JPEG/ColorJitter)
        if not pkgd_transform == None:
            rgb = pkgd_transform[0](rgb)

            if pkgd_transform[1]:
                depth = pkgd_transform[0](depth)

        rgb = rgb.float() / 255.0

        return rgb, depth

    def _random_bbox(self):
        w = 0.65 * self.rng.random() + 0.05
        h = 0.65 * self.rng.random() + 0.05
        cx = (1 - w) * self.rng.random() + (0.5 * w)
        cy = (1 - h) * self.rng.random() + (0.5 * h)

        return np.array([cx, cy, w, h])


def scene_collate_fn(batch):
    return batch[0]

def batch_scene_collate_fn(batch):
    return batch