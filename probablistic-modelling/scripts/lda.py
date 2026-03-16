"""
scripts/lda.py

Runs LDA topic modeling experiments for the project.

Trains LDA with k in {5, 10, 20} topics on AG News and arXiv, 
evaluates perplexity and CV coherence, and saves results to
results/lda/.

Prerequisites:
    Run prepare_data.py first to ensure processed CSVs exist on disk.
    (Vectorizers are NOT needed for LDA — it builds its own Gensim corpus.)

Usage:
    python scripts/lda.py

    # Only one dataset:
    python scripts/lda.py --dataset ag_news
    python scripts/lda.py --dataset arxiv

    # Force retrain all models:
    python scripts/lda.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loader import load_ag_news, load_arxiv
from topic_modeling.lda import run_lda, TOP_N_WORDS
from utils import get_logger

log = get_logger(__name__)


def run(dataset: str, *, force: bool = False) -> None:
    """
    Load data, run LDA experiments, print results.

    Args:
        dataset: "ag_news" or "arxiv"
        force:   Retrain models even if saved files exist.
    """
    log.info("Loading %s ...", dataset)
    split = load_ag_news() if dataset == "ag_news" else load_arxiv()
    log.info("  %s", split)

    t0 = time.perf_counter()
    results_df, topic_dfs = run_lda(split, dataset=dataset, force=force)
    elapsed = time.perf_counter() - t0

    # Print metrics table
    print(f"\n{'=' * 50}")
    print(f"LDA Results — {dataset}")
    print("=" * 50)
    print(results_df.to_string(index=False))

    # Print top words per topic count
    for n_topics, df in topic_dfs.items():
        print(f"\n--- Top {TOP_N_WORDS} words per topic (k={n_topics}) ---")
        for topic_id in range(n_topics):
            words = df[df["topic_id"] == topic_id]["word"].tolist()
            print(f"  Topic {topic_id:>2}: {', '.join(words)}")

    print()
    log.info("Done — %s finished in %.1fs.", dataset, elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LDA topic modeling experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["ag_news", "arxiv", "all"],
        default="all",
        help="Which dataset to run LDA on (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain LDA models even if saved files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["ag_news", "arxiv"] if args.dataset == "all" else [args.dataset]

    total_t0 = time.perf_counter()
    for dataset in datasets:
        run(dataset, force=args.force)

    total_elapsed = time.perf_counter() - total_t0
    log.info("All LDA experiments completed in %.1fs.", total_elapsed)


if __name__ == "__main__":
    main()
