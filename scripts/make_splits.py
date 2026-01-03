from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def utc_now_id(prefix: str = "splits") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def validate_df(df: pd.DataFrame, img_col: str, lbl_col: str) -> None:
    required = {img_col, lbl_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df[lbl_col] = df[lbl_col].astype(int)
    bad = df[~df[lbl_col].isin([1, 2, 3, 4, 5])]
    if len(bad) > 0:
        raise ValueError(f"Found labels outside 1..5. Example rows:\n{bad.head()}")

    dup = df[df[img_col].duplicated(keep=False)].sort_values(img_col)
    if len(dup) > 0:
        print(f"[WARN] Found {len(dup)} rows with duplicated {img_col}.")


def dist_table(df: pd.DataFrame, lbl_col: str) -> pd.DataFrame:
    counts = df[lbl_col].value_counts().sort_index()
    total = counts.sum()
    out = pd.DataFrame({"count": counts, "pct": (counts / total * 100.0).round(2)})
    out.index.name = "grade"
    return out.reset_index()


def split_stratified(
    df: pd.DataFrame,
    img_col: str,
    lbl_col: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("Ratios must sum to 1.0")

    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df[lbl_col],
    )

    val_within_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_within_temp),
        random_state=seed,
        stratify=temp_df[lbl_col],
    )

    return train_df, val_df, test_df


def main() -> None:
    p = argparse.ArgumentParser(description="Create stratified train/val/test splits (by grade)")
    p.add_argument("--in_csv", type=str, default="data/raw/train_labels.csv")
    p.add_argument("--img_col", type=str, default="filename")
    p.add_argument("--lbl_col", type=str, default="label")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default="data/splits")
    p.add_argument("--report_root", type=str, default="outputs/splits")
    args = p.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    report_root = Path(args.report_root)
    run_id = utc_now_id("splits")
    report_dir = report_root / run_id

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    validate_df(df, args.img_col, args.lbl_col)

    dup = df[df[args.img_col].duplicated(keep=False)].sort_values(args.img_col)
    if len(dup) > 0:
        dup.to_csv(report_dir / "duplicate_filenames.csv", index=False)

    train_df, val_df, test_df = split_stratified(
        df=df,
        img_col=args.img_col,
        lbl_col=args.lbl_col,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    report = {
        "run_id": run_id,
        "input_csv": str(in_csv),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "counts": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }
    (report_dir / "split_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    dist = {
        "full": dist_table(df, args.lbl_col),
        "train": dist_table(train_df, args.lbl_col),
        "val": dist_table(val_df, args.lbl_col),
        "test": dist_table(test_df, args.lbl_col),
    }
    for k, table in dist.items():
        table.to_csv(report_dir / f"{k}_grade_distribution.csv", index=False)

    print("[OK] Wrote splits to:", out_dir)
    print("[OK] Wrote report to:", report_dir)


if __name__ == "__main__":
    main()
