"""Colorization losses.

Primary term is L1 on the normalized ab channels. An optional VGG16 perceptual
term compares the reconstructed RGB images; it is off by default
(`lambda_perceptual = 0`) and should be enabled only once the L1 baseline trains
stably.

The perceptual term needs a *differentiable* Lab -> RGB conversion (skimage's is
numpy-only), so a torch implementation of the D65 Lab->sRGB transform lives here.
Inputs to `colorization_loss` are the normalized tensors used everywhere else:
    L_norm  = L/50 - 1 ,  ab_norm = ab/110
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

L_CENTER, L_SCALE, AB_SCALE = 50.0, 50.0, 110.0


def _f_inv(t: torch.Tensor) -> torch.Tensor:
    t3 = t ** 3
    return torch.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Differentiable Lab -> sRGB. `lab` is B x 3 x H x W with real Lab ranges.

    Returns B x 3 x H x W in [0, 1].
    """
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    x = _f_inv(fx) * 0.95047
    y = _f_inv(fy) * 1.00000
    z = _f_inv(fz) * 1.08883

    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    def gamma(c: torch.Tensor) -> torch.Tensor:
        c = torch.clamp(c, 0.0, 1.0)
        return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * torch.pow(c, 1 / 2.4) - 0.055)

    return torch.stack([gamma(r), gamma(g), gamma(bl)], dim=1)


def reconstruct_rgb(L_norm: torch.Tensor, ab_norm: torch.Tensor) -> torch.Tensor:
    """Combine normalized L + ab into an RGB image in [0, 1] (differentiable)."""
    L = (L_norm + 1.0) * L_SCALE
    ab = ab_norm * AB_SCALE
    lab = torch.cat([L, ab], dim=1)
    return lab_to_rgb(lab)


class VGGPerceptual(nn.Module):
    """L2 distance between VGG16 relu2_2 features of two RGB images."""

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).features[:9]
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg.eval()
        self.register_buffer("mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
        p = (pred_rgb - self.mean) / self.std
        g = (gt_rgb - self.mean) / self.std
        return F.mse_loss(self.vgg(p), self.vgg(g))


class ColorizationLoss(nn.Module):
    def __init__(self, lambda_perceptual: float = 0.0):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        self.l1 = nn.L1Loss()
        self.perceptual = VGGPerceptual() if lambda_perceptual > 0 else None

    def forward(self, ab_pred, ab_gt, L_norm):
        loss = self.l1(ab_pred, ab_gt)
        if self.perceptual is not None:
            pred_rgb = reconstruct_rgb(L_norm, ab_pred)
            gt_rgb = reconstruct_rgb(L_norm, ab_gt)
            loss = loss + self.lambda_perceptual * self.perceptual(pred_rgb, gt_rgb)
        return loss
