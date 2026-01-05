from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.datasets.dataloaders import LoaderConfig, build_datasets, build_loaders
from src.training.runner import run_training
from src.utils.run_logger import (
    copy_config,
    init_run,
    make_run_id,
    snapshot_pip_freeze,
    write_decision_note,
    write_run_meta,
)

CONFIG_PATH = "configs/train.yaml"


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = _load_yaml(CONFIG_PATH)

    run_id = make_run_id(prefix=cfg["run"].get("run_name_prefix", "run"))
    paths = init_run(output_root=str(cfg["run"].get("output_dir", "outputs")), run_id=run_id)

    copy_config(CONFIG_PATH, paths.config_copy)
    snapshot_pip_freeze(paths.requirements_lock)
    write_run_meta(paths.run_meta, extra={"purpose": "train_step8"})

    # Build datasets/loaders
    data = cfg["data"]
    img = data["image"]
    norm = img["normalize"]

    train_ds, val_ds, test_ds = build_datasets(
        raw_dir=str(data["raw_dir"]),
        images_dir=str(data["images_dir"]),
        train_csv=str(data["splits"]["train_csv"]),
        val_csv=str(data["splits"]["val_csv"]),
        test_csv=str(data["splits"]["test_csv"]),
        image_mode=str(img["mode"]),
        image_size=int(img["size"]),
        normalize_type=str(norm["type"]),
        mean=float(norm["mean"]),
        std=float(norm["std"]),
        augment_cfg_path="configs/augment.yaml",
    )

    dl = cfg["dataloader"]
    loaders = build_loaders(
        train_ds,
        val_ds,
        test_ds,
        cfg=LoaderConfig(
            batch_size=int(dl["batch_size"]),
            num_workers=int(dl["num_workers"]),
            pin_memory=bool(dl["pin_memory"]),
        ),
    )
    train_loader, val_loader, _test_loader = loaders

    result = run_training(
        cfg=cfg, train_loader=train_loader, val_loader=val_loader, run_dir=paths.run_dir
    )

    import json

    with open(paths.metrics, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    write_decision_note(paths.decision_note)

    print(f"[OK] Step 8 finished. Run dir: {paths.run_dir}")


if __name__ == "__main__":
    main()
