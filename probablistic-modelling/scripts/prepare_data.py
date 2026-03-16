"""
scripts/prepare_data.py

Data preparation pipeline. Loads local data and fits vectorizers.

What it does:
    1. Loads AG News from data/processed/ag_news.csv
    2. Loads arXiv CS abstracts from data/splits/{train,val,test}.csv
    3. Fits Binary BoW / TF / TF-IDF vectorizers on each dataset's training
       split and saves them to representations/saved/

Usage:
    python scripts/prepare_data.py

    # Force re-fitting vectorizers:
    python scripts/prepare_data.py --force

    # Prepare only one dataset:
    python scripts/prepare_data.py --dataset ag_news
    python scripts/prepare_data.py --dataset arxiv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from loader import load_ag_news, load_arxiv
from representations.vectorizers import build_vectorizers
from utils import get_logger

log = get_logger(__name__)


def prepare(dataset: str, *, force: bool = False) -> None:
    """
    Run the preparation pipeline for one dataset.

    Args:
        dataset: "ag_news" or "arxiv"
        force:   If True, re-fit vectorizers even if cached.
    """
    log.info("=" * 60)
    log.info("Preparing dataset: %s", dataset)
    log.info("=" * 60)

    t0 = time.perf_counter()

    if force:
        _delete_cache(dataset)

    log.info("Step 1/2 — Loading local data ...")
    if dataset == "ag_news":
        split = load_ag_news()
    elif dataset == "arxiv":
        split = load_arxiv()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'ag_news' or 'arxiv'.")

    log.info("DataSplit ready: %s", split)
    log.info("Label distribution:\n%s", split.summary().to_string())

    # Step 2: fit and save vectorizers
    log.info(
        "Step 2/2 — Building vectorizers on training split (%d docs) ...",
        len(split.train),
    )
    build_vectorizers(
        train_texts=split.train["text"],
        dataset=dataset,
        force=force,
    )

    elapsed = time.perf_counter() - t0
    log.info("Done — %s prepared in %.1fs.", dataset, elapsed)


def _delete_cache(dataset: str) -> None:
    """Remove saved vectorizers for a dataset."""
    from representations.vectorizers import _SAVED_DIR, Representation

    for rep in Representation:
        p = _SAVED_DIR / f"{dataset}_{rep.file_stem()}.joblib"
        if p.exists():
            p.unlink()
            log.info("Deleted saved vectorizer: '%s'.", p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare datasets and vectorizers for experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["ag_news", "arxiv", "all"],
        default="all",
        help="Which dataset to prepare (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fitting vectorizers even if cached files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets = ["ag_news", "arxiv"] if args.dataset == "all" else [args.dataset]

    total_t0 = time.perf_counter()
    for dataset in datasets:
        prepare(dataset, force=args.force)

    total_elapsed = time.perf_counter() - total_t0
    log.info("All datasets prepared in %.1fs.", total_elapsed)


if __name__ == "__main__":
    main()
