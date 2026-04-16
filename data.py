import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import math
from glob import glob
from collections import defaultdict

# Designed to work with the RGBD dataset from, will load one scene at a time
# https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/

DEPTH_SCALE = 65535.0
LOG_DEPTH_SCALE = math.log(DEPTH_SCALE + 1)


class RGBDDataset(Dataset):
    """Each __getitem__ returns a full scene with all frames preloaded.

    Returns dict with:
        scene:  str           - scene name
        T:      int           - number of valid frames
        frames: list[dict]    - per-frame tensors ready for the model
    """

    def __init__(self, img_dir, transforms=None, scene_names=None, random_seed=42,
                 img_size=480, keep_raw=False):
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_size = img_size
        self.keep_raw = keep_raw
        self.rng = np.random.default_rng(random_seed)

        # Group frames by scene
        all_files = sorted(glob(os.path.join(img_dir, '*/*-color.png')))
        scenes = defaultdict(list)
        for f in all_files:
            scene = os.path.basename(os.path.dirname(f))
            scenes[scene].append(f)

        if scene_names is not None:
            self.scene_names = [s for s in scene_names if s in scenes]
        else:
            self.scene_names = sorted(scenes.keys())

        self.scenes = {s: scenes[s] for s in self.scene_names}

        if self.transforms is not None:
            self.scene_list = [(name, pkgd_tf) for name in self.scene_names for pkgd_tf in self.transforms]
        else:
            self.scene_list = [(name, None) for name in self.scene_names]

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_name, pkgd_tf = self.scene_list[idx]
        frame_paths = self.scenes[scene_name]
        T = len(frame_paths)

        bbox_start = self._random_bbox()
        bbox_end = self._random_bbox()

        frames = []
        for t in range(T):
            frame = self._load_and_preprocess(frame_paths[t], t, T, bbox_start, bbox_end, pkgd_tf)
            if frame is not None:
                frames.append(frame)

        return {
            "scene": scene_name,
            "T": len(frames),
            "frames": frames,
        }

    def _load_and_preprocess(self, path, t, T, bbox_start, bbox_end, pkgd_tf):
        rgb, depth = self._load_rgbd(path, pkgd_tf)
        _, H, W = rgb.shape

        alpha = t / max(T - 1, 1)
        bbox = (1 - alpha) * bbox_start + alpha * bbox_end
        cx, cy, bw, bh = bbox

        top = max(0, int((cy - bh / 2) * H))
        left = max(0, int((cx - bw / 2) * W))
        bottom = min(H, top + int(bh * H))
        right = min(W, left + int(bw * W))

        crop_h, crop_w = bottom - top, right - left
        if crop_h < 2 or crop_w < 2:
            return None

        rgb_crop = rgb[:, top:bottom, left:right]
        depth_crop = depth[:, top:bottom, left:right]
        bbox_t = torch.tensor(bbox, dtype=torch.float32)

        img = F.interpolate(rgb.unsqueeze(0), size=(self.img_size, self.img_size),
                            mode="bilinear", align_corners=False).squeeze(0)

        sq = max(crop_h, crop_w)
        obj_img = F.interpolate(rgb_crop.unsqueeze(0), size=(sq, sq),
                                mode="bilinear", align_corners=False).squeeze(0)

        gt_f = depth_crop.float()
        valid_mask = gt_f > 0
        if valid_mask.any():
            gt_norm = (torch.log(gt_f + 1) / LOG_DEPTH_SCALE).clamp(0, 1).squeeze(0)
        else:
            gt_norm = gt_f.squeeze(0)

        result = {
            "img": img,
            "bbox": bbox_t,
            "obj_img": obj_img,
            "gt_norm": gt_norm,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "frame_idx": t,
        }
        if self.keep_raw:
            result["rgb"] = rgb
            result["rgb_crop"] = rgb_crop
        return result

    def _load_rgbd(self, color_path, pkgd_transform=None):
        depth_path = color_path.replace('-color.png', '-depth.png')
        if not os.path.isfile(depth_path):
            depth_path = color_path.replace('-color.png', '-aligned-depth.png')
        rgb_raw = cv2.imread(color_path)
        if rgb_raw is None:
            return torch.zeros(3, 480, 640), torch.zeros(1, 480, 640)
        rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        if depth is not None:
            depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        else:
            depth = torch.zeros(1, rgb.shape[1], rgb.shape[2], dtype=torch.float32)

        if pkgd_transform is not None:
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