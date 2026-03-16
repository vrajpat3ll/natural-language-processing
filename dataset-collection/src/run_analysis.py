"""
Driver for Lab 1 Tasks 4 & 5.

Usage:
    uv run python run_analysis.py --data path/to/data.parquet --plots plots/

Reads the 'abstract' column, cleans text, then runs vocabulary analysis
(Task 4) and n-gram analysis (Task 5). Saves plots to --plots directory.
"""

import argparse
from pathlib import Path

import pandas as pd

# from processing.cleaner import clean, tokenize
from processing.clean_text import preprocess
from analysis.vocab import (
    build_vocab,
    flatten_tokens,
    freq_histogram,
    print_stats,
    zipf_plot,
)
from analysis.ngrams import ngram_report


RANDOM_SEED = 42


def load_abstracts(data_path: Path) -> list[str]:
    """Load raw abstract strings from CSV or Parquet."""
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if "abstract" not in df.columns:
        raise ValueError(
            f"Expected column 'abstract' in {data_path}. Found: {list(df.columns)}"
        )

    abstracts = df["abstract"].dropna().tolist()
    print(f"  Loaded {len(abstracts):,} abstracts from {data_path}")
    return abstracts


def build_token_corpus(
    abstracts: list[str],
) -> tuple[list[str], list[list[str]]]:
    """
    Clean all abstracts and return:
      - flat_tokens      list[str]           for vocab analysis
      - sentence_tokens  list[list[str]]     for n-gram analysis
    """
    sentence_tokens: list[list[str]] = []

    for abstract in abstracts:
        # preprocess returns list[list[str]] — one inner list per sentence
        sentences = preprocess(abstract, remove_stops=True)
        sentence_tokens.extend(sentences)

    flat_tokens = flatten_tokens(sentence_tokens)
    print(f"  Total tokens      : {len(flat_tokens):,}")
    print(f"  Total sentences   : {len(sentence_tokens):,}")
    return flat_tokens, sentence_tokens


def run(data_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sep = "═" * 60

    # Load
    print(sep)
    print("LOADING DATA".center(60))
    print(sep)
    abstracts = load_abstracts(data_path)

    # Build token corpus
    print(sep)
    print("PREPROCESSING".center(60))
    print(sep)
    flat_tokens, sentence_tokens = build_token_corpus(abstracts)

    # Task 4: Vocabulary Analysis
    print(sep)
    print("TASK 4 — VOCABULARY ANALYSIS".center(60))
    print(sep)

    counter = build_vocab(flat_tokens)
    print_stats(counter)

    zipf_path = plots_dir / "zipf_plot.png"
    hist_path = plots_dir / "freq_histogram.png"
    zipf_plot(counter, zipf_path)
    freq_histogram(counter, hist_path, top_n=50)

    # Task 5: N-gram Analysis
    print(sep)
    print("TASK 5 — N-GRAM ANALYSIS".center(60))
    print(sep)
    # Pass sentence_tokens so bigrams/trigrams respect sentence boundaries
    ngram_report(sentence_tokens, k=20)

    print(sep)
    print("Done. Plots saved to:", plots_dir)
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 1 Tasks 4 & 5 analysis")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset CSV or Parquet file",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        default=Path("plots"),
        help="Directory to save PNG plots (default: ./plots/)",
    )
    args = parser.parse_args()
    run(args.data, args.plots)
