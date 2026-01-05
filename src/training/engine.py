from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochStats:
    loss: float
    acc: float


@torch.no_grad()
def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += _accuracy(logits.detach(), y) * bs
        n += bs

    return EpochStats(loss=total_loss / max(n, 1), acc=total_acc / max(n, 1))


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += _accuracy(logits, y) * bs
        n += bs

    return EpochStats(loss=total_loss / max(n, 1), acc=total_acc / max(n, 1))
