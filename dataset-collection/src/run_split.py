"""
Driver for Lab 1 Task 6: Train / Validation / Test Split.

Usage:
    uv run python run_split.py --data path/to/data.csv --outdir data/splits/
    uv run python run_split.py --data path/to/data.csv --label category
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from split import print_split_report, save_splits, split_dataset, RANDOM_SEED


def run(data_path: Path, outdir: Path, label_col: str) -> None:
    sep = "═" * 62

    print(sep)
    print("LOADING DATA".center(62))
    print(sep)

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"  Loaded {len(df):,} rows from {data_path}")
    print(f"  Columns : {list(df.columns)}")

    print(sep)
    print("SPLITTING".center(62))
    print(sep)
    print(f"  Seed     : {RANDOM_SEED}")
    print(f"  Strategy : stratified on '{label_col}'")
    print("  Ratios   : train=80%  val=10%  test=10%")
    print()

    train_df, val_df, test_df = split_dataset(df, label_col=label_col)

    print_split_report(train_df, val_df, test_df, label_col=label_col)

    print(sep)
    print("SAVING".center(62))
    print(sep)
    save_splits(train_df, val_df, test_df, outdir=outdir)

    print(sep)
    print("Done.")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 1 Task 6 — dataset split")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to full dataset CSV or Parquet",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for split CSVs (default: data/splits/)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="label",
        help="Column name to stratify on (default: label)",
    )
    args = parser.parse_args()
    run(args.data, args.outdir, args.label)
