"""
data/loader.py

Loads AG News (benchmark) and arXiv CS abstracts
into pandas DataFrames with consistent column names and split structure.

Pipeline per dataset:
    1. Load from local pre-split CSVs (data/splits/ for arXiv, data/processed/ for AG News)
    2. Map labels to integer indices (0-3)
    3. Return a DataSplit

Returned DataFrames always have exactly three columns:
    raw_text : str  — original document text
    text     : str  — cleaned text
    label    : int  — integer class index (0-based)

Split structure:
    arXiv — Loaded from data/splits/{train,val,test}.csv
    AG News — Loaded from data/processed/ag_news.csv and split as before (90/10/test)

Usage:
    from loader import load_ag_news, load_arxiv

    ag    = load_ag_news()
    arxiv = load_arxiv()

    ag.train    # pd.DataFrame  columns: [raw_text, text, label]
    ag.val
    ag.test
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_logger

log = get_logger(__name__)

# Constants
RANDOM_SEED: int = 42

# Fraction of AG News *train* split reserved for validation
AG_VAL_FRACTION: float = 0.10

# Processed CSV paths  (data/processed/)
_DATA_DIR: Path = Path(__file__).parent / "data"
_PROCESSED_DIR: Path = _DATA_DIR / "processed"
_AG_CSV: Path = _PROCESSED_DIR / "ag_news.csv"

# Pre-split CSV paths (data/splits/)
_SPLITS_DIR: Path = _DATA_DIR / "splits"
_TRAIN_CSV: Path = _SPLITS_DIR / "train.csv"
_VAL_CSV: Path = _SPLITS_DIR / "val.csv"
_TEST_CSV: Path = _SPLITS_DIR / "test.csv"

# Label mapping for arXiv
_ARXIV_LABEL_MAP = {
    "Algorithms": 0,
    "NLP/AI": 1,
    "Systems/Infra": 2,
    "Vision": 3,
}


@dataclass(frozen=True)
class DataSplit:
    """
    Immutable container for train / val / test DataFrames.

    Each DataFrame has columns:  raw_text (str), text (str), label (int)
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"DataSplit("
            f"train={len(self.train)}, "
            f"val={len(self.val)}, "
            f"test={len(self.test)})"
        )

    def summary(self) -> pd.DataFrame:
        """
        Returns a small DataFrame showing per-split size and label
        distribution — handy for quick sanity checks.
        """
        rows = []
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            counts = df["label"].value_counts().sort_index().to_dict()
            rows.append({"split": name, "total": len(df), **counts})
        return pd.DataFrame(rows).set_index("split")


def _split_two(
    df: pd.DataFrame,
    test_size: float,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified two-way split. Returns (main, held_out)."""
    main, held_out = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    return main.reset_index(drop=True), held_out.reset_index(drop=True)


def load_ag_news() -> DataSplit:
    """
    Load and return AG News as a DataSplit.

    Loads from data/processed/ag_news.csv.

    Splitting:
        test  — the original HF test rows (standard AG News test set)
        val   — 10 % of the original HF train, stratified
        train — remaining 90 % of the original HF train

    Label mapping:
        0 — World  |  1 — Sports  |  2 — Business  |  3 — Sci/Tech
    """
    if not _AG_CSV.exists():
        log.error(f"AG News processed CSV not found at '{_AG_CSV}'.")
        raise FileNotFoundError(f"Missing AG News data: {_AG_CSV}")

    log.info(f"Loading AG News from '{_AG_CSV}'.")
    df = pd.read_csv(
        _AG_CSV,
        dtype={"raw_text": str, "text": str, "label": int, "_origin": str},
        keep_default_na=False,
    )

    # Recover the original HF train / test partition
    original_train = (
        df[df["_origin"] == "train"].drop(columns="_origin").reset_index(drop=True)
    )
    test_df = df[df["_origin"] == "test"].drop(columns="_origin").reset_index(drop=True)

    # Carve validation out of original train
    train_df, val_df = _split_two(original_train, test_size=AG_VAL_FRACTION)

    return DataSplit(train=train_df, val=val_df, test=test_df)


def load_arxiv() -> DataSplit:
    """
    Load and return the arXiv CS abstracts dataset as a DataSplit.

    Loads from pre-split CSVs in data/splits/.

    Label mapping (Lab 1 scheme):
        0 — Algorithms  |  1 — NLP/AI  |  2 — Systems/Infra  |  3 — Vision
    """
    if not (_TRAIN_CSV.exists() and _VAL_CSV.exists() and _TEST_CSV.exists()):
        log.error(f"arXiv pre-split CSVs not found in '{_SPLITS_DIR}'.")
        raise FileNotFoundError(f"Missing arXiv splits in: {_SPLITS_DIR}")

    log.info("Loading arXiv from pre-split CSVs in '%s'.", _SPLITS_DIR)

    def _load_and_process(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Rename columns to match expected schema
        df = df.rename(columns={"abstract": "raw_text", "abstract_cleaned": "text"})
        # Map string labels to integers
        df["label"] = df["label"].map(_ARXIV_LABEL_MAP)
        # Fill any NaNs (especially in text/raw_text) with empty strings to prevent sklearn failures
        df["raw_text"] = df["raw_text"].fillna("")
        df["text"] = df["text"].fillna("")
        # Keep only required columns
        return df[["raw_text", "text", "label"]]

    train_df = _load_and_process(_TRAIN_CSV)
    val_df = _load_and_process(_VAL_CSV)
    test_df = _load_and_process(_TEST_CSV)

    return DataSplit(train=train_df, val=val_df, test=test_df)
