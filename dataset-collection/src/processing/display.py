"""
Lab 1 - Task 3: Report utility.

Add to text_cleaning.py, or import standalone.
Shows a side-by-side comparison of all three normalization methods
for the report's required comparison table.
"""

from __future__ import annotations

import textwrap


def show_normalization_comparison(
    raw_tokens: list[str],
    *,
    width: int = 88,
) -> None:
    """
    Print a side-by-side comparison of stopword removal, stemming,
    and lemmatization applied to the same token list.

    Intended for the report's Task 3 comparison requirement.

    Args:
        raw_tokens: Token list produced by tokenize() in cleaner.py.
                    Should NOT have stopwords removed yet (pass remove_stops=False
                    to clean() so all three methods start from the same baseline).
        width:      Display width for wrapping.
    """
    from cleaner import _STOPWORDS
    from normalizer import stem, lemmatize

    sep = "─" * width

    def _fmt(tokens: list[str]) -> str:
        joined = " ".join(tokens)
        return textwrap.fill(joined, width=width - 4, subsequent_indent="    ")

    # ── Compute each variant ─────────────────────────────────────────────────
    after_stopwords = [t for t in raw_tokens if t not in _STOPWORDS]
    after_stem = stem(after_stopwords)
    after_lemma = lemmatize(after_stopwords)

    # ── Display ──────────────────────────────────────────────────────────────
    print(sep)
    print("NORMALIZATION COMPARISON".center(width))
    print(sep)

    print(f"\n{'[ Raw tokens ]':─<{width}}")
    print(f"  Count : {len(raw_tokens)}")
    print(f"  Tokens: {_fmt(raw_tokens)}")

    print(f"\n{'[ After stopword removal ]':─<{width}}")
    removed = len(raw_tokens) - len(after_stopwords)
    print(f"  Count : {len(after_stopwords)}  ({removed} stopwords removed)")
    print(f"  Tokens: {_fmt(after_stopwords)}")

    print(f"\n{'[ After Porter stemming ]':─<{width}}")
    vocab_raw = len(set(after_stopwords))
    vocab_stem = len(set(after_stem))
    print(
        f"  Count : {len(after_stem)}  (vocab {vocab_raw} -> {vocab_stem} unique stems)"
    )
    print(f"  Tokens: {_fmt(after_stem)}")

    print(f"\n{'[ After WordNet lemmatization ]':─<{width}}")
    vocab_lemma = len(set(after_lemma))
    print(
        f"  Count : {len(after_lemma)}  (vocab {vocab_raw} -> {vocab_lemma} unique lemmas)"
    )
    print(f"  Tokens: {_fmt(after_lemma)}")

    # ── Token-level diff table ───────────────────────────────────────────────
    print(f"\n{'[ Token-level diff (first 20 tokens) ]':─<{width}}")
    header = f"  {'Original':<20} {'Stemmed':<20} {'Lemmatized':<20}"
    print(header)
    print(f"  {'─' * 18}  {'─' * 18}  {'─' * 18}")
    for orig, st, lm in zip(after_stopwords[:20], after_stem[:20], after_lemma[:20]):
        changed_stem = " *" if st != orig else ""
        changed_lemma = " *" if lm != orig else ""
        print(f"  {orig:<20} {st + changed_stem:<20} {lm + changed_lemma:<20}")
    print("\n  * = changed from original")

    print(f"\n{sep}\n")


def show_method_discussion() -> None:
    """
    Print a brief discussion of when each normalization method is appropriate.
    Use this text as a starting point for the report's discussion section.
    """
    discussion = """
WHEN TO USE EACH METHOD
────────────────────────────────────────────────────────────────────────────────

Stopword Removal
  Best for:   Bag-of-words tasks (TF-IDF, Naive Bayes, LDA topic modelling)
              where function words ('the', 'a', 'is') add noise without meaning.
  Avoid for:  POS tagging, sentiment analysis, or any task where grammatical
              structure matters — removing 'not' flips sentence polarity.

Porter Stemming
  Best for:   High-recall retrieval, vocabulary compression, situations where
              speed matters and linguistic precision does not.
              e.g. 'training', 'trained', 'trains' -> 'train' (good collapse).
  Drawbacks:  Over-stemming artifacts ('studies' -> 'studi', 'universe' -> 'univers').
              Output is not a real word, so human-readable reports suffer.
  Avoid for:  Tasks requiring interpretable vocabulary (topic modelling word lists,
              LDA top-words display).

WordNet Lemmatization
  Best for:   Tasks requiring real dictionary words in output (topic modelling,
              named entity analysis, report-friendly vocabulary).
              More conservative — only reduces to attested forms.
  Drawbacks:  Without POS context defaults to noun, so verb forms are under-reduced
              ('running' stays 'running'). Slower than stemming.
  Avoid for:  Situations where speed is critical and over-stemming is acceptable.

Recommendation for this lab:
  Lab 1 vocab analysis  ->  lemmatization  (interpretable top-word lists)
  Lab 2 TF-IDF / BoW   ->  stemming       (vocabulary compression, speed)
  Lab 3 Naive Bayes     ->  stemming       (consistent with Lab 2 representations)
  Lab 3 LDA             ->  lemmatization  (readable topic word lists)
  Lab 3 POS Tagging     ->  NO normalization (keep original word forms)
"""
    print(discussion)
