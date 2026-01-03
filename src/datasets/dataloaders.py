from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

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

    train_ds = MammoGradesDataset(DatasetConfig(csv_path=Path(train_csv), **base))
    val_ds = MammoGradesDataset(DatasetConfig(csv_path=Path(val_csv), **base))
    test_ds = MammoGradesDataset(DatasetConfig(csv_path=Path(test_csv), **base))
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
