"""
Taken from Lab 1.

Accepts a flat list[str] of already-cleaned tokens.
Call flatten_tokens() first if you have list[list[str]] from preprocess().

Public API:
    flatten_tokens(sentences)       list[list[str]] -> list[str]
    build_vocab(tokens)             -> Counter
    vocab_stats(counter)            -> dict  (report-ready numbers)
    rare_words(counter, threshold)  -> list[str]
    zipf_plot(counter, path)        -> saves PNG
    freq_histogram(counter, path)   -> saves PNG
    print_stats(counter)            -> console report
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive — save only, no display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def flatten_tokens(sentences: list[list[str]]) -> list[str]:
    """
    Flatten sentence-segmented token structure into a single token list.

    Args:
        sentences: Output of text_cleaning.preprocess().

    Returns:
        Single flat list of all tokens.
    """
    return [tok for sent in sentences for tok in sent]


def build_vocab(tokens: list[str]) -> Counter:
    """
    Build a frequency Counter from a flat token list.

    Args:
        tokens: Flat list of cleaned, lowercased tokens.

    Returns:
        Counter mapping token -> frequency.
    """
    return Counter(tokens)


def vocab_stats(counter: Counter) -> dict:
    """
    Compute summary statistics for the report's Task 4 table.

    Returns a dict with:
        total_tokens  int   — total tokens (with repetition)
        vocab_size    int   — unique token types
        hapax_count   int   — tokens appearing exactly once
        hapax_pct     float — hapax legomena as % of vocab
        top_50        list  — [(token, freq), ...] top 50
        avg_freq      float — mean frequency across all types
        max_freq      int   — frequency of most common token
        min_freq      int   — frequency of least common token

    Args:
        counter: Output of build_vocab().
    """
    total = sum(counter.values())
    vocab = len(counter)
    hapax = sum(1 for f in counter.values() if f == 1)

    return {
        "total_tokens": total,
        "vocab_size": vocab,
        "hapax_count": hapax,
        "hapax_pct": round(hapax / vocab * 100, 2) if vocab else 0.0,
        "top_50": counter.most_common(50),
        "avg_freq": round(total / vocab, 4) if vocab else 0.0,
        "max_freq": counter.most_common(1)[0][1] if counter else 0,
        "min_freq": min(counter.values()) if counter else 0,
    }


def rare_words(counter: Counter, threshold: int = 2) -> list[str]:
    """
    Return tokens whose frequency is <= threshold.

    Args:
        counter:   Output of build_vocab().
        threshold: Max frequency to be considered rare (default 2).

    Returns:
        Sorted list of rare token strings.
    """
    return sorted(tok for tok, freq in counter.items() if freq <= threshold)


def print_stats(counter: Counter) -> None:
    """Pretty-print vocab statistics for the report."""
    stats = vocab_stats(counter)
    sep = "─" * 52
    print(sep)
    print("VOCABULARY STATISTICS".center(52))
    print(sep)
    print(f"  Total tokens (with repetition) : {stats['total_tokens']:,}")
    print(f"  Vocabulary size (unique types)  : {stats['vocab_size']:,}")
    print(
        f"  Hapax legomena                  : {stats['hapax_count']:,}  ({stats['hapax_pct']}% of vocab)"
    )
    print(f"  Average token frequency         : {stats['avg_freq']}")
    print(f"  Most frequent token count       : {stats['max_freq']:,}")
    print()
    print("  Top 50 tokens:")
    print(f"  {'Rank':<6} {'Token':<25} {'Frequency':>10}")
    print(f"  {'─' * 4}   {'─' * 23}   {'─' * 9}")
    for rank, (tok, freq) in enumerate(stats["top_50"], 1):
        print(f"  {rank:<6} {tok:<25} {freq:>10,}")
    print(sep)


def zipf_plot(counter: Counter, path: str | Path) -> Path:
    """
    Save a Zipf's Law log-log plot (rank vs frequency) as PNG.

    Also overlays the ideal Zipf curve (f ∝ 1/rank) for comparison.

    Args:
        counter: Output of build_vocab().
        path:    Output file path (.png).

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    freqs = sorted(counter.values(), reverse=True)
    ranks = list(range(1, len(freqs) + 1))
    ideal = [freqs[0] / r for r in ranks]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(ranks, freqs, linewidth=1.8, color="#2563EB", label="Observed frequency")
    ax.loglog(
        ranks,
        ideal,
        linewidth=1.2,
        color="#DC2626",
        linestyle="--",
        alpha=0.7,
        label="Ideal Zipf  (f ∝ 1/rank)",
    )

    ax.set_xlabel("Rank  (log scale)", fontsize=11)
    ax.set_ylabel("Frequency  (log scale)", fontsize=11)
    ax.set_title(
        "Zipf's Law — Token Frequency vs Rank\n(arXiv CS Abstracts)", fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [zipf_plot] saved -> {path}")
    return path


def freq_histogram(
    counter: Counter,
    path: str | Path,
    *,
    top_n: int = 50,
) -> Path:
    """
    Save a bar chart of the top-N most frequent tokens as PNG.

    Args:
        counter: Output of build_vocab().
        path:    Output file path (.png).
        top_n:   Number of top tokens to display (default 50).

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    top = counter.most_common(top_n)
    words = [w for w, _ in top]
    freqs = [f for _, f in top]

    fig, ax = plt.subplots(figsize=(14, 5))

    bars = ax.bar(range(len(words)), freqs, color="#2563EB", alpha=0.85, width=0.7)

    for bar, freq in zip(bars, freqs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(freqs) * 0.005,
            str(freq),
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#374151",
        )

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Top {top_n} Most Frequent Tokens — arXiv CS Abstracts", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [freq_histogram] saved -> {path}")
    return path
