from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    raw_dir: Path
    images_dir: Path
    csv_path: Path
    image_mode: str = "L"  # grayscale
    image_size: int = 512
    normalize_type: str = "fixed"  # fixed | per_image
    mean: float = 0.0
    std: float = 1.0


class MammoGradesDataset(Dataset):
    """
    Reads a split CSV with columns: filename, label (grades 1..5).
    Loads images from images_dir/filename, converts to grayscale,
    returns tensor [1,H,W] float32 normalized, and label in [0..4].
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv_path)

        if "filename" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(
                f"CSV must contain columns filename and label. Found: {list(self.df.columns)}"
            )

        self.df["label"] = self.df["label"].astype(int)
        if not self.df["label"].isin([1, 2, 3, 4, 5]).all():
            bad = self.df[~self.df["label"].isin([1, 2, 3, 4, 5])].head()
            raise ValueError(f"Found labels outside 1..5. Example:\n{bad}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, filename: str) -> Image.Image:
        path = (self.cfg.images_dir / str(filename)).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        img = img.convert(self.cfg.image_mode)

        if img.size != (self.cfg.image_size, self.cfg.image_size):
            img = img.resize((self.cfg.image_size, self.cfg.image_size))
        return img

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0  # [H,W] in [0,1]
        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return x

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.normalize_type == "per_image":
            mean = x.mean()
            std = x.std().clamp_min(1e-6)
            return (x - mean) / std

        mean = torch.tensor(self.cfg.mean, dtype=x.dtype)
        std = torch.tensor(self.cfg.std, dtype=x.dtype).clamp_min(1e-6)
        return (x - mean) / std

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        grade = int(row["label"])  # 1..5

        img = self._load_image(filename)
        x = self._to_tensor(img)
        x = self._normalize(x)

        y = grade - 1

        meta = {"filename": filename}
        return x, y, meta
