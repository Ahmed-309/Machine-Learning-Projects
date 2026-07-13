"""Colorize grayscale (or colour) images with a trained checkpoint.

Keeps the original full-resolution L channel and only upsamples the *predicted*
ab channels back to the original size (the same trick as the existing OpenCV
script `Colorize Black & white images [OPEN CV]/image_colarization.py`), so the
output stays as sharp as the input.
"""
import argparse
import os
import glob

import numpy as np
from PIL import Image
import torch

from config import get_config
from dataset import IMG_EXTS, L_SCALE, AB_SCALE
from model import build_model
from skimage import color


def _list_inputs(path):
    if os.path.isdir(path):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        return sorted(files)
    return [path]


@torch.no_grad()
def colorize_image(model, path, device, model_size=256):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    lab = color.rgb2lab(rgb).astype(np.float32)
    L_full = lab[:, :, 0]                       # original resolution L
    h, w = L_full.shape

    # Model input: L resized to the training resolution, normalized.
    L_small = np.asarray(
        Image.fromarray(L_full).resize((model_size, model_size), Image.BILINEAR)
    )
    L_in = torch.from_numpy((L_small / L_SCALE - 1.0)[None, None]).float().to(device)
    ab_small = model(L_in)[0].cpu().numpy() * AB_SCALE   # 2 x model_size x model_size

    # Upsample predicted ab back to the original resolution.
    ab_up = np.stack([
        np.asarray(Image.fromarray(ab_small[c]).resize((w, h), Image.BILINEAR))
        for c in range(2)
    ], axis=-1)

    lab_out = np.concatenate([L_full[:, :, None], ab_up], axis=-1).astype(np.float64)
    out = (np.clip(color.lab2rgb(lab_out), 0, 1) * 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description="Colorize image(s) from a checkpoint")
    ap.add_argument("input", help="image file or directory")
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded {args.ckpt} (step {ck.get('global_step', '?')}) on {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    for path in _list_inputs(args.input):
        out = colorize_image(model, path, device, cfg.image_size)
        name = os.path.splitext(os.path.basename(path))[0] + "_color.png"
        dest = os.path.join(args.out_dir, name)
        Image.fromarray(out).save(dest)
        print(f"  {path} -> {dest}")


if __name__ == "__main__":
    main()
