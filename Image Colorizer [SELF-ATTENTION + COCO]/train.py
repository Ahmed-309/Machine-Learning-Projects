"""Training loop for the self-attention colorizer.

Plain PyTorch (no Lightning) so the checkpoint/resume logic is explicit and
robust to Colab/Kaggle timeouts:
  * `checkpoints/latest.pt` is rewritten every `ckpt_every_steps` and at each
    epoch end, storing model + optimizer + scaler + step/epoch.
  * on start-up, if `latest.pt` exists it is loaded and training resumes from the
    saved step, so re-running the script just continues.
Mixed precision (AMP) is used on CUDA for speed/memory.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset import ColorizationDataset
from model import build_model
from losses import ColorizationLoss
from metrics import evaluate


def parse_args(cfg):
    ap = argparse.ArgumentParser(description="Train the colorizer")
    ap.add_argument("--data-root", default=cfg.data_root)
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--image-size", type=int, default=cfg.image_size)
    ap.add_argument("--lambda-perceptual", type=float, default=cfg.lambda_perceptual)
    ap.add_argument("--attention", nargs="*", default=cfg.attention_levels,
                    help="which stages get self-attention: bottleneck mid")
    ap.add_argument("--limit-samples", type=int, default=cfg.limit_samples,
                    help=">0 uses only N images (fast smoke test)")
    ap.add_argument("--ckpt-dir", default=cfg.ckpt_dir)
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args()


def save_ckpt(path, model, opt, scaler, epoch, global_step):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, tmp)
    os.replace(tmp, path)  # atomic: never leave a half-written checkpoint


def main():
    cfg = get_config()
    args = parse_args(cfg)
    cfg.data_root = args.data_root
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.image_size = args.image_size
    cfg.lambda_perceptual = args.lambda_perceptual
    cfg.attention_levels = list(args.attention)
    cfg.limit_samples = args.limit_samples
    cfg.ckpt_dir = args.ckpt_dir

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.amp and not args.no_amp and device == "cuda"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    print(f"Device: {device} | AMP: {use_amp} | attention: {cfg.attention_levels}")

    train_ds = ColorizationDataset(
        cfg.data_root, "train", cfg.image_size, cfg.val_fraction,
        cfg.split_seed, cfg.limit_samples,
    )
    val_ds = ColorizationDataset(
        cfg.data_root, "val", cfg.image_size, cfg.val_fraction,
        cfg.split_seed, cfg.limit_samples,
    )
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    model = build_model(cfg).to(device)
    criterion = ColorizationLoss(cfg.lambda_perceptual).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- resume ----
    start_epoch, global_step = 0, 0
    latest = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"]
        global_step = ck["global_step"]
        print(f"Resumed from {latest}: epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running = 0.0
        for L, ab_gt in train_loader:
            L, ab_gt = L.to(device), ab_gt.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                ab_pred = model(L)
                loss = criterion(ab_pred, ab_gt, L)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running += loss.item()
            if global_step % cfg.log_every_steps == 0:
                avg = running / cfg.log_every_steps
                running = 0.0
                print(f"epoch {epoch} step {global_step} loss {avg:.4f}")
            if global_step % cfg.ckpt_every_steps == 0:
                save_ckpt(latest, model, optimizer, scaler, epoch, global_step)

        # ---- end of epoch: validate + checkpoint ----
        metrics = evaluate(model, val_loader, device, max_batches=cfg.val_max_batches)
        val_loss = -metrics["psnr"]  # higher PSNR is better -> minimise its negative
        scheduler.step(val_loss)
        print(
            f"[epoch {epoch}] val PSNR {metrics['psnr']:.4f} dB "
            f"SSIM {metrics['ssim']:.4f} (n={metrics['n']})"
        )
        save_ckpt(latest, model, optimizer, scaler, epoch + 1, global_step)
        save_ckpt(
            os.path.join(cfg.ckpt_dir, f"epoch_{epoch:03d}.pt"),
            model, optimizer, scaler, epoch + 1, global_step,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
