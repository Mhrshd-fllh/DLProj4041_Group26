from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.augmentations.policy import build_train_augment
from src.datasets.mammo_dataset import DatasetConfig, MammoGradesDataset


@dataclass
class LoaderConfig:
    batch_size: int = 16
    num_workers: int = 2
    pin_memory: bool = True


def build_datasets(
    raw_dir: str,
    images_dir: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    image_mode: str,
    image_size: int,
    normalize_type: str,
    mean: float,
    std: float,
    augment_cfg_path: str = "configs/augment.yaml",
) -> tuple[MammoGradesDataset, MammoGradesDataset, MammoGradesDataset]:
    base = dict(
        raw_dir=Path(raw_dir),
        images_dir=Path(images_dir),
        image_mode=image_mode,
        image_size=image_size,
        normalize_type=normalize_type,
        mean=mean,
        std=std,
    )

    cfg_path = Path(augment_cfg_path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            aug_cfg = yaml.safe_load(f) or {}
    else:
        aug_cfg = {"augment": {"enabled": False}}

    train_aug = build_train_augment(aug_cfg)
    train_ds = MammoGradesDataset(
        DatasetConfig(csv_path=Path(train_csv), split="train", augment=train_aug, **base)
    )
    val_ds = MammoGradesDataset(
        DatasetConfig(csv_path=Path(val_csv), split="val", augment=None, **base)
    )
    test_ds = MammoGradesDataset(
        DatasetConfig(csv_path=Path(test_csv), split="test", augment=None, **base)
    )
    return train_ds, val_ds, test_ds


def build_loaders(
    train_ds: MammoGradesDataset,
    val_ds: MammoGradesDataset,
    test_ds: MammoGradesDataset,
    cfg: LoaderConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, val_loader, test_loader
