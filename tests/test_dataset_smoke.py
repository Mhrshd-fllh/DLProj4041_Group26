from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.datasets.mammo_dataset import DatasetConfig, MammoGradesDataset


def test_dataset_smoke(tmp_path: Path):
    # Create dummy image
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    img = (np.random.rand(512, 512) * 255).astype(np.uint8)
    img_path = images_dir / "x.png"
    Image.fromarray(img).save(img_path)

    # Create dummy CSV
    csv_path = tmp_path / "split.csv"
    pd.DataFrame({"filename": ["x.png"], "label": [3]}).to_csv(csv_path, index=False)

    ds = MammoGradesDataset(
        DatasetConfig(
            raw_dir=tmp_path,
            images_dir=images_dir,
            csv_path=csv_path,
            image_mode="L",
            image_size=512,
            normalize_type="fixed",
            mean=0.0,
            std=1.0,
        )
    )

    x, y, meta = ds[0]
    assert x.shape == (1, 512, 512)
    assert y == 2  # grade 3 -> class index 2
    assert meta["filename"] == "x.png"
