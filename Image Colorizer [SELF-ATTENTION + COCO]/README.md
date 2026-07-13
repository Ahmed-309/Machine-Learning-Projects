# Image Colorizer — Pretrained Backbone + Self-Attention (COCO)

A **trainable** deep image colorizer built in PyTorch. Unlike the sibling
`Colorize Black & white images [OPEN CV]` project (which only runs a frozen
pretrained Caffe model for inference), this project trains its own network on
COCO images and can be fine-tuned and improved.

## Idea

Colorization is treated as a self-supervised regression in **CIE Lab** space:

- **Input**  = the `L` (lightness) channel — i.e. the grayscale image.
- **Target** = the `a` and `b` chrominance channels.
- Reconstruct the colour image by combining the input `L` with the predicted `ab`.

Only RGB photos are needed — the COCO-Stuff segmentation masks are **not** used.

## Architecture

```
L (1×H×W)
   │
   ▼
ResNet34 encoder (ImageNet-pretrained; conv1 adapted to 1 channel)
   │   skips ─────────────────────────────────┐
   ▼                                           │
bottleneck 512×8×8  ──►  Self-Attention (SAGAN)│
   │                                           │
   ▼   U-Net decoder (upsample + concat skip)  │
   ├───────────────────────────────────────────┘
   ▼   (optional Self-Attention at 32×32)
1×1 conv → tanh → ab (2×H×W)
```

- **Pretrained backbone**: `torchvision` ResNet34; `conv1` is replaced by a
  1-channel conv whose weights are the channel-mean of the pretrained RGB conv,
  preserving the learned low-level filters.
- **Self-attention** (`model.SelfAttention2d`): SAGAN-style spatial attention at
  the bottleneck (and optionally at a mid-decoder stage) so the model reasons
  about long-range colour consistency, not just local texture.
- **Loss**: L1 on the `ab` channels, plus an optional VGG16 perceptual term
  (off by default; enable once the L1 baseline is stable).

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyper-parameters in one dataclass |
| `download_data.py` | Fetch COCO images (val2017 zip, or train subset via FiftyOne) |
| `dataset.py` | `ColorizationDataset` + Lab normalize/denormalize helpers |
| `model.py` | `SelfAttention2d`, `AttentionUNet` |
| `losses.py` | L1 + optional VGG perceptual loss (+ differentiable Lab→RGB) |
| `train.py` | AMP training loop with checkpoint/resume |
| `metrics.py` | PSNR / SSIM over the val split |
| `infer.py` | Colorize an image or folder from a checkpoint |
| `demo.ipynb` | End-to-end demo: load checkpoint → colorize → metrics |

## Quick start

```bash
pip install -r requirements.txt

# 1. Get data (small subset is fine to start)
python download_data.py --data-root data/coco --max-samples 2000

# 2. Train (Colab/Kaggle friendly: AMP + resume)
python train.py --data-root data/coco --epochs 20 --batch-size 16

# 3. Evaluate
python metrics.py --ckpt checkpoints/latest.pt --data-root data/coco

# 4. Colorize new images
python infer.py path/to/bw_image.jpg --ckpt checkpoints/latest.pt --out-dir outputs
```

Enable a second attention block and the perceptual loss for a stronger model:

```bash
python train.py --attention bottleneck mid --lambda-perceptual 0.1
```

## Colab / Kaggle notes

- Training uses **mixed precision** on CUDA automatically.
- `checkpoints/latest.pt` is rewritten every `ckpt_every_steps` and at each epoch
  end. If a session times out, just **re-run `train.py`** — it resumes from the
  last checkpoint (step, optimizer and AMP scaler state included).
- Start with the default `val2017` subset; scale up to a `train2017` subset via
  `--source fiftyone` when you want more data.

## Metrics

`metrics.py` reports **PSNR** and **SSIM** of the reconstructed RGB vs. ground
truth, using the same `scikit-image` settings as the sibling notebook. Note these
reward matching the *original* colours; a plausible but different colorization
will still score lower even if it looks good.
