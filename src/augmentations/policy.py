from __future__ import annotations

from typing import Any

from torchvision import transforms

from src.models.factory import RandomGamma


def build_train_augment(cfg: dict[str, Any]) -> transforms.Compose:
    """
    PIL -> PIL augmentation chain.
    Apply AFTER forced resize (Step 6) and BEFORE tensor/normalize.
    """
    augment = cfg.get("augment", {})
    if not augment.get("enabled", True):
        return transforms.Compose([])

    ops: list[transforms.Transform] = []

    # Horizontal flip (allowed; orientation not meaningful)
    hflip_p = float(augment.get("hflip_p", 0.5))
    if hflip_p > 0:
        ops.append(transforms.RandomHorizontalFlip(p=hflip_p))

    # Small rotation (conservative)
    rotate_enabled = bool(augment.get("rotate_enabled", True))
    rotate_p = float(augment.get("rotate_p", 0.3))
    rotate_degrees = float(augment.get("rotate_degrees", 5.0))
    if rotate_enabled and rotate_p > 0 and rotate_degrees > 0:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=rotate_degrees)],
                p=rotate_p,
            )
        )

    # Mild brightness/contrast jitter
    brightness = float(augment.get("brightness", 0.05))
    contrast = float(augment.get("contrast", 0.05))
    if brightness > 0 or contrast > 0:
        ops.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    # Mild gamma jitter
    gamma_enabled = bool(augment.get("gamma_enabled", True))
    gamma_p = float(augment.get("gamma_p", 0.3))
    gamma_min = float(augment.get("gamma_min", 0.95))
    gamma_max = float(augment.get("gamma_max", 1.05))
    if gamma_enabled and gamma_p > 0 and gamma_max >= gamma_min:
        ops.append(transforms.RandomApply([RandomGamma(gamma_min, gamma_max)], p=gamma_p))

    return transforms.Compose(ops)


def build_eval_augment() -> transforms.Compose:
    """No augmentation for val/test/infer."""
    return transforms.Compose([])
