"""Central configuration for the self-attention image colorizer.

A single dataclass keeps every knob in one place so `train.py`, `infer.py` and
`metrics.py` all read the same settings. Override fields from the CLI in the
scripts, or edit the defaults here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---- data ----
    data_root: str = "data/coco"          # folder that holds the images (flat glob)
    image_size: int = 256                 # train/eval square resolution
    val_fraction: float = 0.10            # deterministic train/val split
    split_seed: int = 42
    num_workers: int = 2

    # ---- model ----
    # Which decoder stages get a self-attention block.
    #   "bottleneck" -> 8x8 (512 ch), cheap and high-impact
    #   "mid"        -> 32x32 (128 ch), more global reasoning, costlier
    attention_levels: List[str] = field(default_factory=lambda: ["bottleneck"])
    pretrained_backbone: bool = True

    # ---- optimisation ----
    batch_size: int = 16
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 0.0
    lambda_perceptual: float = 0.0        # 0 -> pure L1 baseline; raise once stable
    amp: bool = True                      # mixed precision (CUDA only)

    # ---- checkpointing / logging ----
    ckpt_dir: str = "checkpoints"
    ckpt_every_steps: int = 500           # periodic save so Colab timeouts are safe
    log_every_steps: int = 50
    val_max_batches: int = 20             # cap val metric cost per epoch

    # ---- misc ----
    seed: int = 0
    limit_samples: int = 0                # >0 -> only use N images (fast smoke tests)


def get_config() -> Config:
    return Config()
