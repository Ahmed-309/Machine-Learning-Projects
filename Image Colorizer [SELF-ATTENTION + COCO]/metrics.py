"""PSNR / SSIM evaluation.

Reconstructs RGB from (input L, predicted ab) and compares against the
ground-truth RGB, using the same skimage functions and settings already used in
the existing `Colorize_Black_and_White_Image.ipynb` notebook.
"""
from typing import Dict

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dataset import denormalize_to_rgb


def image_psnr_ssim(gt_rgb: np.ndarray, pred_rgb: np.ndarray):
    p = psnr(gt_rgb, pred_rgb, data_range=255)
    s = ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=255)
    return p, s


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 0) -> Dict[str, float]:
    model.eval()
    psnr_vals, ssim_vals = [], []
    for i, (L, ab_gt) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        L = L.to(device)
        ab_pred = model(L).cpu().numpy()
        L_np = L.cpu().numpy()
        ab_gt_np = ab_gt.numpy()
        for b in range(L_np.shape[0]):
            gt = denormalize_to_rgb(L_np[b], ab_gt_np[b])
            pred = denormalize_to_rgb(L_np[b], ab_pred[b])
            p, s = image_psnr_ssim(gt, pred)
            psnr_vals.append(p)
            ssim_vals.append(s)
    return {
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        "n": len(psnr_vals),
    }


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from config import get_config
    from dataset import ColorizationDataset
    from model import build_model

    ap = argparse.ArgumentParser(description="Evaluate PSNR/SSIM on the val split")
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()

    cfg = get_config()
    if args.data_root:
        cfg.data_root = args.data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = ColorizationDataset(
        cfg.data_root, split="val", image_size=cfg.image_size,
        val_fraction=cfg.val_fraction, split_seed=cfg.split_seed,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {args.ckpt} (step {ckpt.get('global_step', '?')})")

    res = evaluate(model, val_loader, device, max_batches=args.max_batches)
    print(f"PSNR: {res['psnr']:.4f} dB   SSIM: {res['ssim']:.4f}   (n={res['n']})")
