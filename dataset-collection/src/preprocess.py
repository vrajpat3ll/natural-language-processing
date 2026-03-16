"""
preprocess.py
=============
Lab 1 - Text Cleaning & Normalization pipeline.

Loads the raw dataset, cleans each abstract, and saves a CSV with
two columns:
  - abstract          original raw text
  - abstract_cleaned  fully cleaned, stopwords removed

Usage:
    uv run python preprocess.py --data path/to/data.parquet --output data/cleaned.csv
    uv run python preprocess.py --data path/to/data.csv --output data/cleaned.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from processing.cleaner import clean
from utils import get_logger

log = get_logger(__name__)


def load(data_path: Path) -> pd.DataFrame:
    """Load CSV or Parquet, keeping only the 'abstract' column."""
    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported format: {suffix!r}. Use .csv or .parquet")

    if "abstract" not in df.columns:
        raise KeyError(f"'abstract' column not found. Available: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=["abstract"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d rows with null abstracts.", dropped)

    log.info("Loaded %d abstracts from %s", len(df), data_path)
    return df


def save(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to CSV, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved cleaned dataset -> %s  (%d rows)", output_path, len(df))


def clean_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'abstract_cleaned' column: fully cleaned, stopwords removed.

    The original 'abstract' column is preserved unchanged.

    Args:
        df: DataFrame with an 'abstract' column.

    Returns:
        Same DataFrame with an additional 'abstract_cleaned' column.
    """
    log.info("Cleaning %d abstracts ...", len(df))

    cleaned: list[str] = []
    for text in tqdm(df["abstract"], desc="Cleaning", unit="doc"):
        cleaned.append(clean(str(text), remove_stops=True))

    df = df.copy()
    df["abstract_cleaned"] = cleaned

    # Quick sanity stats
    empty = sum(1 for t in cleaned if not t.strip())
    avg_len = sum(len(t.split()) for t in cleaned) / max(len(cleaned), 1)
    log.info(
        "Done. Avg cleaned length: %.1f tokens | Empty after cleaning: %d",
        avg_len,
        empty,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and normalise the arXiv abstracts corpus."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to input dataset (.csv or .parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output CSV  (e.g. data/cleaned.csv)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Path for the output CSV  (e.g. data/cleaned.csv)",
    )
    args = parser.parse_args()

    df = load(args.data)
    df = clean_corpus(df)

    ordered = [
        "arxiv_id",
        "title",
        "abstract",
        "abstract_cleaned",
        "primary_cat",
        "all_cats",
        "label",
    ]
    df = df[ordered]

    save(df, args.output)


if __name__ == "__main__":
    main()
