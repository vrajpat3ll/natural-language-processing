"""
split.py
========
Lab 1 - Task 6: Train / Validation / Test Split.

Splits the dataset into reproducible 80/10/10 partitions using a fixed
random seed. Stratified by label to preserve class distribution across
all three splits.

Public API:
    split_dataset(df, label_col, seed)  -> tuple[DataFrame, DataFrame, DataFrame]
    save_splits(train, val, test, outdir)
    print_split_report(train, val, test, label_col)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED: int = 42

# Split ratios -> sum to 1
TRAIN_RATIO: float = 0.80
VAL_RATIO:   float = 0.10
TEST_RATIO:  float = 0.10  # remainder after train+val


def split_dataset(
    df: pd.DataFrame,
    label_col: str = "category",
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train / val / test sets.

    Ratios  : 80% / 10% / 10%
    Strategy: Stratified on label_col to preserve class balance.

    Args:
        df:        Full dataset DataFrame. Must contain label_col.
        label_col: Column name to stratify on (default 'category').
        seed:      Random seed for reproducibility (default RANDOM_SEED=42).

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If label_col is not present in df.
    """
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Step 1: split off 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - TRAIN_RATIO,       # 0.20
        random_state=seed,
        stratify=df[label_col],
    )

    # Step 2: split temp 50/50 -> val (10%) and test (10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[label_col],
    )

    # Reset indices for clean downstream indexing
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    outdir:   str | Path = "data/splits",
) -> dict[str, Path]:
    """
    Save train / val / test DataFrames as CSV files.

    Output files:
        <outdir>/train.csv
        <outdir>/val.csv
        <outdir>/test.csv

    Args:
        train_df: Training split DataFrame.
        val_df:   Validation split DataFrame.
        test_df:  Test split DataFrame.
        outdir:   Directory to write CSVs into (created if absent).

    Returns:
        Dict mapping split name -> saved Path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = outdir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
        print(f"  [{name:>5}] saved -> {p}  ({len(df):,} rows)")

    return paths


def print_split_report(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    label_col: str = "category",
) -> None:
    """
    Print a report showing split sizes and per-class distribution.

    Args:
        train_df:  Training split.
        val_df:    Validation split.
        test_df:   Test split.
        label_col: Column to report class distribution for.
    """
    total = len(train_df) + len(val_df) + len(test_df)
    sep   = "─" * 62

    print(sep)
    print("DATASET SPLIT REPORT".center(62))
    print(sep)
    print(f"  {'Split':<10} {'Count':>8}  {'% of total':>12}")
    print(f"  {'─'*8}   {'─'*7}   {'─'*11}")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pct = len(df) / total * 100
        print(f"  {name:<10} {len(df):>8,}  {pct:>11.1f}%")
    print(f"  {'─'*8}   {'─'*7}   {'─'*11}")
    print(f"  {'TOTAL':<10} {total:>8,}  {'100.0%':>12}")

    if label_col in train_df.columns:
        print()
        print(f"  Class distribution (stratified on '{label_col}'):")
        print()

        # Collect all class labels across splits
        all_labels = sorted(
            set(train_df[label_col].unique())
            | set(val_df[label_col].unique())
            | set(test_df[label_col].unique())
        )

        # Header
        print(f"  {'Class':<22} {'Train':>8} {'Val':>8} {'Test':>8}  {'Train%':>8}")
        print(f"  {'─'*22}   {'─'*6}   {'─'*6}   {'─'*6}   {'─'*7}")

        for label in all_labels:
            tr = (train_df[label_col] == label).sum()
            va = (val_df[label_col]   == label).sum()
            te = (test_df[label_col]  == label).sum()
            tr_pct = tr / len(train_df) * 100
            print(f"  {str(label):<22} {tr:>8,} {va:>8,} {te:>8,}  {tr_pct:>7.1f}%")

    print(sep)
    print(f"  Random seed : {RANDOM_SEED}")
    print(f"  Strategy    : Stratified split on '{label_col}'")
    print(sep)
