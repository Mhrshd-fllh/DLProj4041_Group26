from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Defaults
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_IMAGES_DIR = DEFAULT_RAW_DIR / "images"
DEFAULT_CSV = None


# Helpers
def utc_now_id(prefix: str = "eda") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def safe_read_image(path: Path) -> tuple[bool, Image.Image | None, str | None]:
    """Try to open an image safely. Returns (ok, image, error_message)."""

    try:
        img = Image.open(path)
        img.load()
        return True, img, None
    except Exception as e:
        return False, None, str(e)


def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute file MD5 hash (byte-level duplicates)"""

    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.udpate(chunk)
    return md5.hexdigest()


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Try to detect the image path column and grade label column.
    """

    img_col = "filename"
    lbl_col = "label"

    return img_col, lbl_col


def map_grade_to_group(grade: int) -> str:
    # Default Mapping
    # Benign = 1, 2 | Borderline = 3 | Malignant = 4, 5

    if grade in (1, 2):
        return "benign"
    if grade == 3:
        return "borderline"
    if grade in (4, 5):
        return "malignant"
    return "unknown"


@dataclass
class ScanConfig:
    raw_dir: Path
    images_dir: Path
    csv_path: Path
    out_dir: Path
    max_images: int | None
    compute_hashes: bool


# Main Scan Logic


def run_scan(cfg: ScanConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.csv_path)

    img_col, lbl_col = detect_columns(df)

    df[lbl_col] = df[lbl_col].astype(int)

    bad_labels = df[~df[lbl_col].isin([1, 2, 3, 4, 5])]
    if len(bad_labels) > 0:
        (cfg.out_dir / "bad_labels.csv").write_text(
            bad_labels.to_csv(index=False), encoding="utf-8"
        )
        raise ValueError(
            f"Found Labels outside 1..5. Wrote bad rows to {cfg.out_dir / 'bad_labels.csv'}"
        )

    def resolve_path(p: str) -> Path:
        p = str(p).strip()
        path = Path(p)

        if path.is_absolute():
            return path

        if path.parts and path.parts[0].lower() == "images":
            return (cfg.raw_dir / path).resolve()

        return (cfg.images_dir / path).resolve()

    df["abs_image_path"] = df[img_col].apply(resolve_path)
    df["group"] = df[lbl_col].apply(map_grade_to_group)

    df_scan = df.copy()
    missing_rows: list[dict] = []
    corrupt_rows: list[dict] = []
    size_rows: list[dict] = []
    intensity_rows: list[dict] = []
    hash_rows: list[dict] = []

    for idx, row in df_scan.iterrows():
        p: Path = row["abs_image_path"]
        grade: int = int(row[lbl_col])
        group: str = str(row["group"])

        if not p.exists():
            missing_rows.append(
                {
                    "row_index": int(idx),
                    "image_ref": row[img_col],
                    "abs_path": str(p),
                    "grade": grade,
                    "group": group,
                }
            )
            continue

        ok, img, err = safe_read_image(p)
        if not ok or img is None:
            corrupt_rows.append(
                {
                    "row_index": int(idx),
                    "image_ref": row[img_col],
                    "abs_path": str(p),
                    "grade": grade,
                    "group": group,
                    "error": err,
                }
            )
            continue

        w, h = img.size
        mode = img.mode
        size_rows.append(
            {
                "row_index": int(idx),
                "abs_path": str(p),
                "width": int(w),
                "height": int(h),
                "mode": mode,
                "grade": grade,
                "group": group,
            }
        )

        try:
            arr = np.array(img.convert("L"), dtype=np.float32)  # [0..255] usually
            arr_norm = arr / 255.0
            intensity_rows.append(
                {
                    "row_index": int(idx),
                    "abs_path": str(p),
                    "mean": float(arr_norm.mean()),
                    "std": float(arr_norm.std()),
                    "min": float(arr_norm.min()),
                    "max": float(arr_norm.max()),
                    "grade": grade,
                    "group": group,
                }
            )
        except Exception as e:
            corrupt_rows.append(
                {
                    "row_index": int(idx),
                    "image_ref": row[img_col],
                    "abs_path": str(p),
                    "grade": grade,
                    "group": group,
                    "error": f"intensity_compute_failed: {e}",
                }
            )

        if cfg.compute_hashes:
            try:
                md5 = compute_md5(p)
                hash_rows.append({"abs_path": str(p), "md5": md5, "grade": grade, "group": group})
            except Exception as e:
                hash_rows.append(
                    {
                        "abs_path": str(p),
                        "md5": None,
                        "grade": grade,
                        "group": group,
                        "error": str(e),
                    }
                )

    grade_dist = (
        df[lbl_col].value_counts().sort_index().rename_axis("grade").reset_index(name="count")
    )
    group_dist = (
        df["group"]
        .value_counts()
        .sort_index()
        .reindex(["benign", "borderline", "malignant"], fill_value=0)
        .rename_axis("group")
        .reset_index(name="count")
    )

    grade_dist.to_csv(cfg.out_dir / "grade_distribution.csv", index=False)
    group_dist.to_csv(cfg.out_dir / "group_distribution.csv", index=False)

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(cfg.out_dir / "missing_files.csv", index=False)
    if corrupt_rows:
        pd.DataFrame(corrupt_rows).to_csv(cfg.out_dir / "corrupt_files.csv", index=False)

    if size_rows:
        size_df = pd.DataFrame(size_rows)
        size_df.to_csv(cfg.out_dir / "image_sizes.csv", index=False)
        size_summary = {
            "n_scanned": int(len(size_df)),
            "width_min": int(size_df["width"].min()),
            "width_max": int(size_df["width"].max()),
            "height_min": int(size_df["height"].min()),
            "height_max": int(size_df["height"].max()),
            "mode_counts": size_df["mode"].value_counts().to_dict(),
        }
    else:
        size_summary = {"n_scanned": 0}

    if intensity_rows:
        inten_df = pd.DataFrame(intensity_rows)
        inten_df.to_csv(cfg.out_dir / "intensity_stats.csv", index=False)
        inten_summary = {
            "n_scanned": int(len(inten_df)),
            "mean_mean": float(inten_df["mean"].mean()),
            "mean_std": float(inten_df["mean"].std()),
            "std_mean": float(inten_df["std"].mean()),
            "min_global": float(inten_df["min"].min()),
            "max_global": float(inten_df["max"].max()),
        }
    else:
        inten_summary = {"n_scanned": 0}

    dup_summary = {"computed": bool(cfg.compute_hashes), "n_hashed": 0, "n_duplicates_groups": 0}
    if cfg.compute_hashes and hash_rows:
        hash_df = pd.DataFrame(hash_rows)
        hash_df.to_csv(cfg.out_dir / "file_hashes.csv", index=False)

        valid_hash = hash_df.dropna(subset=["md5"]).copy()
        dup_groups = (
            valid_hash.groupby("md5")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        dup_groups = dup_groups[dup_groups["count"] > 1]

        dup_summary["n_hashed"] = int(len(valid_hash))
        dup_summary["n_duplicates_groups"] = int(len(dup_groups))

        if len(dup_groups) > 0:
            dup_groups.to_csv(cfg.out_dir / "duplicates_by_md5.csv", index=False)

    summary = {
        "run_id": cfg.out_dir.name,
        "raw_dir": str(cfg.raw_dir),
        "images_dir": str(cfg.images_dir),
        "csv_path": str(cfg.csv_path),
        "total_rows_in_csv": int(len(df)),
        "scanned_rows": int(len(df_scan)),
        "label_column": lbl_col,
        "image_column": img_col,
        "missing_files": int(len(missing_rows)),
        "corrupt_files": int(len(corrupt_rows)),
        "grade_distribution": grade_dist.set_index("grade")["count"].to_dict(),
        "group_distribution": group_dist.set_index("group")["count"].to_dict(),
        "size_summary": size_summary,
        "intensity_summary": inten_summary,
        "duplicate_summary": dup_summary,
        "notes": [
            "Distributions are computed on full CSV.",
            "Sizes/intensity are computed on scanned subset if max_images is set.",
        ],
    }

    (cfg.out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[OK] EDA scan complete. Outputs saved to: {cfg.out_dir}")


def auto_detect_csv(raw_dir: Path) -> Path:
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {raw_dir}. Place your labels CSV in data/raw/")
    return csvs[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset scanner + EDA artifact generator")
    p.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW_DIR))
    p.add_argument("--images_dir", type=str, default=str(DEFAULT_IMAGES_DIR))
    p.add_argument("--csv_path", type=str, default="")
    p.add_argument("--out_root", type=str, default="outputs/eda")
    p.add_argument("--max_images", type=int, default=0, help="0 means scan all rows (can be slow)")
    p.add_argument(
        "--hashes", action="store_true", help="Compute MD5 hashes to detect byte-level duplicates"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    images_dir = Path(args.images_dir)

    csv_path = Path(args.csv_path) if args.csv_path else auto_detect_csv(raw_dir)
    out_root = Path(args.out_root)
    out_dir = out_root / utc_now_id(prefix="eda")

    max_images = None if args.max_images <= 0 else args.max_images

    cfg = ScanConfig(
        raw_dir=raw_dir,
        images_dir=images_dir,
        csv_path=csv_path,
        out_dir=out_dir,
        max_images=max_images,
        compute_hashes=bool(args.hashes),
    )

    run_scan(cfg)


if __name__ == "__main__":
    main()
