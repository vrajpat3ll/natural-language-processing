"""
Lab 1 - Task 5: N-gram Analysis.

Accepts either:
  - list[str]        flat token list  (unigrams only — no sentence boundary issue)
  - list[list[str]]  sentence-segmented token lists  (preferred for bi/trigrams)

Building bigrams/trigrams from a flat list would produce spurious cross-sentence
n-grams (e.g. last word of abstract 1 paired with first word of abstract 2).
Always prefer the nested form for n >= 2.

Public API:
    build_ngrams(token_source, n)   -> Counter of n-gram tuples
    top_ngrams(counter, k)          -> list of (ngram_tuple, freq)
    ngram_report(token_source)      -> prints uni/bi/trigram top-20 tables
"""

from __future__ import annotations

from collections import Counter
from typing import Union

TokenSource = Union[list[str], list[list[str]]]


def _iter_ngrams(tokens: list[str], n: int):
    """Yield all n-gram tuples from a single token list."""
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def _normalise_source(token_source: TokenSource) -> list[list[str]]:
    """
    Always return list[list[str]] regardless of input shape.

    A flat list[str] is treated as a single sentence.
    """
    if not token_source:
        return []
    if isinstance(token_source[0], str):
        # flat list — wrap in a single sentence
        return [token_source]  # type: ignore[list-item]
    return token_source  # type: ignore[return-value]


def build_ngrams(token_source: TokenSource, n: int) -> Counter:
    """
    Build a frequency Counter of n-gram tuples.

    N-grams are computed per sentence so no spurious cross-sentence
    pairs are generated.

    Args:
        token_source: Either a flat list[str] or list[list[str]] from
                      text_cleaning.preprocess().
        n:            N-gram order (1=unigram, 2=bigram, 3=trigram).

    Returns:
        Counter mapping tuple[str, ...] -> frequency.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    sentences = _normalise_source(token_source)
    counter: Counter = Counter()
    for sent in sentences:
        counter.update(_iter_ngrams(sent, n))
    return counter


def top_ngrams(counter: Counter, k: int = 20) -> list[tuple[tuple[str, ...], int]]:
    """
    Return the top-k most frequent n-grams.

    Args:
        counter: Output of build_ngrams().
        k:       Number of top n-grams to return (default 20).

    Returns:
        List of (ngram_tuple, frequency) pairs, descending by frequency.
    """
    return counter.most_common(k)


def ngram_report(token_source: TokenSource, *, k: int = 20) -> None:
    """
    Print a formatted uni/bi/trigram frequency table for the report.

    Args:
        token_source: flat list[str] or list[list[str]] from preprocess().
        k:            Number of top n-grams to display per order (default 20).
    """
    sep = "─" * 60

    for n, label in [(1, "UNIGRAMS"), (2, "BIGRAMS"), (3, "TRIGRAMS")]:
        counter = build_ngrams(token_source, n)
        top = top_ngrams(counter, k)

        print(sep)
        print(f"Top {k} {label}".center(60))
        print(f"Total unique {label.lower()}: {len(counter):,}".center(60))
        print(sep)
        print(f"  {'Rank':<6} {'N-gram':<38} {'Freq':>8}")
        print(f"  {'─' * 4}   {'─' * 36}   {'─' * 7}")

        for rank, (ngram, freq) in enumerate(top, 1):
            gram_str = " ".join(ngram)
            # Truncate long n-gram strings for display alignment
            if len(gram_str) > 36:
                gram_str = gram_str[:33] + "..."
            print(f"  {rank:<6} {gram_str:<38} {freq:>8,}")

        print()

    print(sep)
    _print_observations(token_source)


def _print_observations(token_source: TokenSource) -> None:
    """Print quick quantitative observations for the report discussion."""
    uni_counter = build_ngrams(token_source, 1)
    bi_counter = build_ngrams(token_source, 2)
    tri_counter = build_ngrams(token_source, 3)

    total_uni = sum(uni_counter.values())
    total_bi = sum(bi_counter.values())
    total_tri = sum(tri_counter.values())

    top1_bi = bi_counter.most_common(1)
    top1_tri = tri_counter.most_common(1)

    print("OBSERVATIONS".center(60))
    print("─" * 60)
    print(f"  Unique unigrams  : {len(uni_counter):,}")
    print(
        f"  Unique bigrams   : {len(bi_counter):,}  ({len(bi_counter) / max(total_bi, 1) * 100:.1f}% of all bigram tokens are unique)"
    )
    print(
        f"  Unique trigrams  : {len(tri_counter):,}  ({len(tri_counter) / max(total_tri, 1) * 100:.1f}% of all trigram tokens are unique)"
    )

    if top1_bi:
        print(
            f"\n  Most common bigram  : '{' '.join(top1_bi[0][0])}'  (freq={top1_bi[0][1]:,})"
        )
    if top1_tri:
        print(
            f"  Most common trigram : '{' '.join(top1_tri[0][0])}'  (freq={top1_tri[0][1]:,})"
        )

    # Data sparsity indicator — useful for Naive Bayes discussion in Lab 3
    hapax_bi = sum(1 for f in bi_counter.values() if f == 1)
    hapax_tri = sum(1 for f in tri_counter.values() if f == 1)
    print(
        f"\n  Hapax bigrams    : {hapax_bi:,}  ({hapax_bi / max(len(bi_counter), 1) * 100:.1f}% of unique bigrams)"
    )
    print(
        f"  Hapax trigrams   : {hapax_tri:,}  ({hapax_tri / max(len(tri_counter), 1) * 100:.1f}% of unique trigrams)"
    )
    print("  -> High hapax % indicates data sparsity.")
    print("─" * 60)
