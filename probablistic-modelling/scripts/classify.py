"""
scripts/classify.py

Runs Naive Bayes classification experiments for the final year project.

Trains MultinomialNB with Binary BoW, TF, and TF-IDF representations on
both AG News and arXiv, evaluates on test sets, and saves results tables
to results/naive_bayes/.

Prerequisites:
    Run prepare_data.py first to ensure processed CSVs and fitted
    vectorizers exist on disk.

Usage:
    python scripts/classify.py

    # Only one dataset:
    python scripts/classify.py --dataset ag_news
    python scripts/classify.py --dataset arxiv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loader import load_ag_news, load_arxiv
from representations.vectorizers import load_vectorizers, vectorizers_exist
from naive_bayes.classifier import run_naive_bayes
from utils import get_logger

log = get_logger(__name__)

# Results are saved here
_RESULTS_DIR: Path = Path(__file__).parent.parent / "results" / "naive_bayes"


def run(dataset: str) -> None:
    """
    Load data + vectorizers, run experiments, save results CSV.

    Args:
        dataset: "ag_news" or "arxiv"
    """
    log.info("=" * 60)
    log.info("Running Naive Bayes — dataset: %s", dataset)
    log.info("=" * 60)

    # Guard: ensure prepare_data.py has been run first
    if not vectorizers_exist(dataset):
        log.error(
            "Vectorizers not found for '%s'. "
            "Run `uv run prepare_data_cli.py --dataset %s` first.",
            dataset,
            dataset,
        )
        sys.exit(1)

    t0 = time.perf_counter()

    # Load data (CSV cache — fast)
    log.info("Loading %s split ...", dataset)
    split = load_ag_news() if dataset == "ag_news" else load_arxiv()
    log.info("  %s", split)

    # Load fitted vectorizers from disk
    log.info("Loading vectorizers for %s ...", dataset)
    vectorizers = load_vectorizers(dataset)

    # Run experiments
    results_df = run_naive_bayes(split, vectorizers, dataset=dataset)

    # Save results
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{dataset}.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Results saved to '%s'.", out_path)

    # Print table to stdout for quick inspection
    print(f"\n{'=' * 50}")
    print(f"Results — {dataset}")
    print("=" * 50)
    print(results_df.to_string(index=False))
    print()

    elapsed = time.perf_counter() - t0
    log.info("Done — %s finished in %.1fs.", dataset, elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Naive Bayes experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["ag_news", "arxiv", "all"],
        default="all",
        help="Which dataset to run experiments on (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["ag_news", "arxiv"] if args.dataset == "all" else [args.dataset]

    total_t0 = time.perf_counter()
    for dataset in datasets:
        run(dataset)

    total_elapsed = time.perf_counter() - total_t0
    log.info("All experiments completed in %.1fs.", total_elapsed)


if __name__ == "__main__":
    main()
