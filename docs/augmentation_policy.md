# Step 07 — Augmentation Policy (Mammography-safe)

## Context
Splits are stored as separate CSV files under `data/splits/`:
- train/val/test = 80/10/10

Augmentation is applied **only to the training split**.

## Assumptions
- Grayscale images (non-DICOM).
- Split CSV columns: `filename`, `label` where label ∈ {1..5}.
- Dataset maps labels to {0..4} via `y = grade - 1`.
- Orientation has no semantic meaning → horizontal flip allowed.

## Policy
Conservative augmentations to avoid damaging anatomy/geometry:
- Horizontal flip (p=0.5)
- Small rotation (±5°) with p=0.3
- Mild brightness/contrast jitter (±0.05)
- Mild gamma jitter in [0.95, 1.05] with p=0.3

No augmentation is applied for val/test/inference.

## Configuration
- `configs/augment.yaml` controls parameters and enables/disables the whole policy via `augment.enabled`.

## Implementation
- `src/augmentations/policy.py`: builds torchvision PIL→PIL augmentation chain.
- `src/datasets/mammo_dataset.py`: applies augmentation only when `split == "train"` and `augment` is provided.
- Datasets are created from split CSVs (train/val/test) located under `data/splits/`.
