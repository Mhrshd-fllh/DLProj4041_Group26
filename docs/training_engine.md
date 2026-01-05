# Step 08 — Training Engine + Runner (PyTorch)

## Goal
Implement a robust training runner:
- train/val loop
- checkpointing (last + best)
- run artifacts written under `outputs/<run_id>/`
- config-driven via `configs/train.yaml`

## Inputs
- `configs/train.yaml` provides:
  - data paths (raw/images + split CSVs)
  - dataloader params
  - model params (resnet18, pretrained, num_classes=5)
  - train params (epochs, lr)
- Step 7 provides `configs/augment.yaml` used only for training.

## Outputs (per run)
`outputs/<run_id>/` contains:
- copied config, pip-freeze lock, run meta
- `metrics.json` (history + best selection)
- `checkpoints/last.pt`, `checkpoints/best.pt`

## Best checkpoint rule (Step 08)
- Best model is selected by minimum `val_loss`.
- This will be updated in Step 09+ to use task metrics (macro-F1, balanced accuracy).

## AMP
- If CUDA is available, mixed precision (AMP) is enabled by default for efficiency.

## Notes
- Labels are mapped to 0..4 in the dataset.
- Loss: CrossEntropyLoss.
