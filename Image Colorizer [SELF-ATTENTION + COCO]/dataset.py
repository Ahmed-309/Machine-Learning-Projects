"""COCO colorization dataset.

Colorization is self-supervised: we only need RGB photos. Each item converts an
RGB image to CIE Lab, returns the L channel as input and the ab channels as the
target. Normalisation keeps both roughly in [-1, 1] so the tanh-headed model can
match the target range:

    L_norm  = L / 50 - 1        (L   in [0, 100]  -> ~[-1, 1])
    ab_norm = ab / 110          (a,b in ~[-110, 110] -> ~[-1, 1])
"""
import os
import glob
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from skimage import color

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

L_CENTER, L_SCALE, AB_SCALE = 50.0, 50.0, 110.0


def normalize_lab(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split an H x W x 3 Lab array into normalized (L[1,H,W], ab[2,H,W])."""
    L = lab[:, :, :1] / L_SCALE - 1.0
    ab = lab[:, :, 1:] / AB_SCALE
    L = np.transpose(L, (2, 0, 1)).astype(np.float32)
    ab = np.transpose(ab, (2, 0, 1)).astype(np.float32)
    return L, ab


def denormalize_to_rgb(L_norm: np.ndarray, ab_norm: np.ndarray) -> np.ndarray:
    """Inverse of normalize_lab + Lab->RGB. Inputs are [1,H,W] and [2,H,W] arrays.

    Returns an H x W x 3 uint8 RGB image.
    """
    L = (L_norm[0] + 1.0) * L_SCALE
    a = ab_norm[0] * AB_SCALE
    b = ab_norm[1] * AB_SCALE
    lab = np.stack([L, a, b], axis=-1).astype(np.float64)
    rgb = color.lab2rgb(lab)  # 0..1
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def list_images(root: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(files)


class ColorizationDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 256,
        val_fraction: float = 0.10,
        split_seed: int = 42,
        limit_samples: int = 0,
    ):
        assert split in ("train", "val")
        all_files = list_images(root)
        if not all_files:
            raise FileNotFoundError(
                f"No images found under '{root}'. Run download_data.py first."
            )
        rng = random.Random(split_seed)
        rng.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_fraction))
        self.files = all_files[n_val:] if split == "train" else all_files[:n_val]
        if limit_samples > 0:
            self.files = self.files[:limit_samples]
        self.split = split
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def _load_rgb(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        s = self.image_size
        if self.split == "train":
            # random resized crop + horizontal flip
            img = img.resize((int(s * 1.15), int(s * 1.15)), Image.BILINEAR)
            max_x = img.width - s
            max_y = img.height - s
            left = random.randint(0, max(0, max_x))
            top = random.randint(0, max(0, max_y))
            img = img.crop((left, top, left + s, top + s))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.resize((s, s), Image.BILINEAR)
        return np.asarray(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = self._load_rgb(self.files[idx])
        lab = color.rgb2lab(rgb).astype(np.float32)  # L in [0,100], ab real
        L, ab = normalize_lab(lab)
        return torch.from_numpy(L), torch.from_numpy(ab)
