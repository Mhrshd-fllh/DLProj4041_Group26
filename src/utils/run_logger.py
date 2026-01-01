import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunPaths:
    run_dir: Path
    config_copy: Path
    requirements_lock: Path
    run_meta: Path
    metrics: Path
    decision_note: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "run") -> str:
    # Sortable ID
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def snapshot_pip_freeze(out_path: Path) -> None:
    cmd = [os.sys.executable, "-m", "pip", "freeze"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out_path.write_text(result.stdout, encoding="utf-8")


def write_run_meta(out_path: Path, extra: dict[str, Any] | None = None) -> None:
    meta: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        meta.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": getattr(torch.version, "cuda", None),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0
                    else None
                ),
            }
        )
    except Exception:
        meta.update({"torch_version": None})

    if extra:
        meta.update(extra)

    out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def init_run(output_root: str, run_id: str) -> RunPaths:
    root = Path(output_root)
    ensure_dir(root)
    run_dir = root / run_id
    ensure_dir(run_dir)

    return RunPaths(
        run_dir=run_dir,
        config_copy=run_dir / "config.yaml",
        requirements_lock=run_dir / "requirements.lock.txt",
        run_meta=run_dir / "run_meta.json",
        metrics=run_dir / "metrics.json",
        decision_note=run_dir / "decision_note.md",
    )


def copy_config(src_config_path: str, dst_config_path: Path) -> None:
    shutil.copyfile(src_config_path, dst_config_path)


def write_decision_note(path: Path, text: str = "") -> None:
    if not text:
        text = (
            "# Decision Note\n\n"
            "- Placeholder for trial rationale.\n"
            "- Later, the LLM orchestrator will write a justified note here.\n"
        )
    path.write_text(text, encoding="utf-8")


def write_metrics_dummy(path: Path, run_id: str, seed: int = 42, split: str = "smoke") -> None:
    metrics = {
        "run_id": run_id,
        "seed": seed,
        "split": split,
        "primary_metric_name": "smoke_ok",
        "primary_metric_value": 1,
        "timestamp_utc": utc_now_iso(),
    }
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
