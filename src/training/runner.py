from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.factory import build_model, count_params
from src.training.checkpoint import BestTracker, save_checkpoint
from src.training.engine import train_one_epoch, validate_one_epoch


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float


def run_training(
    cfg: dict[str, Any],
    train_loader,
    val_loader,
    run_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg["model"]
    model = build_model(
        name=str(model_cfg["name"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        num_classes=int(model_cfg["num_classes"]),
        in_channels=1,  # grayscale
    ).to(device)

    train_cfg = TrainConfig(
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
    )

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = run_dir / "checkpoints"
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"

    tracker = BestTracker()
    history: list[dict[str, Any]] = []
    params = count_params(model)

    for epoch in range(train_cfg.epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        va = validate_one_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": tr.loss,
            "train_acc": tr.acc,
            "val_loss": va.loss,
            "val_acc": va.acc,
            "lr": train_cfg.lr,
        }
        history.append(row)

        # always save last
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extra={"metrics": row, "model_params": params},
        )

        # save best by val_loss (Step 8 choice)
        if tracker.is_best(va.loss):
            tracker.best_val_loss = va.loss
            tracker.best_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"metrics": row, "model_params": params},
            )

    result = {
        "device": str(device),
        "amp": bool(use_amp),
        "model_params": params,
        "best_epoch": tracker.best_epoch,
        "best_val_loss": tracker.best_val_loss,
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    return result
