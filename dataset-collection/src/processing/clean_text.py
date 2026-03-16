"""
Lab 1 - Task 2: Text Cleaning additions.

Extends cleaner.py with:
  - Sentence segmentation   (sent_segment)
  - Full pipeline           (preprocess)
  - Stage-by-stage display  (show_pipeline)

Import pattern:
    from cleaner import clean, tokenize
    from text_cleaning import sent_segment, preprocess, show_pipeline
"""

from __future__ import annotations

import textwrap
from typing import Callable

import nltk
from nltk.tokenize import sent_tokenize

from processing.cleaner import clean, tokenize, _ensure_nltk_resources

_ensure_nltk_resources()

# punkt sentence segmenter needs to know these abbreviations common in CS/arXiv
# abstracts so it doesn't split on them (e.g. "e.g. our model" -> one sentence)
_CS_ABBREVIATIONS = [
    "e.g",
    "i.e",
    "et al",
    "vs",
    "fig",
    "eq",
    "sec",
    "approx",
    "ref",
    "resp",
    "cf",
    "dept",
    "univ",
    "govt",
    "corp",
    "inc",
    "prob",
    "avg",
    "std",
    "acc",
    "tab",
    "app",
    "proc",
]

try:
    _tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
    _tokenizer._params.abbrev_types.update(_CS_ABBREVIATIONS)
except Exception:
    _tokenizer = None


def sent_segment(text: str) -> list[str]:
    """
    Split a raw abstract into sentences.

    Uses NLTK's Punkt model with CS-domain abbreviations pre-registered
    to avoid spurious splits on 'e.g.', 'et al.', 'Fig.', etc.

    Args:
        text: Raw or lightly cleaned abstract string.

    Returns:
        List of sentence strings.
    """
    if _tokenizer is not None:
        sentences = _tokenizer.tokenize(text.strip())
    else:
        sentences = sent_tokenize(text.strip())
    return [s for s in sentences if s.strip()]


def preprocess(
    text: str,
    *,
    remove_stops: bool = True,
) -> list[list[str]]:
    """
    Full preprocessing pipeline: raw text -> list of token lists.

    Each inner list is one sentence's tokens after cleaning.

    Pipeline:
        raw text
          └─ sent_segment()          # sentence boundaries first
               └─ clean()            # latex / url / punct / digits / stops
                    └─ tokenize()    # word tokens, drops len-1 stragglers

    Args:
        text:         Raw abstract string.
        remove_stops: Forward to clean(); set False for POS-tagging tasks.

    Returns:
        List[List[str]] — one token list per sentence.
    """
    sentences = sent_segment(text)
    result: list[list[str]] = []
    for sentence in sentences:
        cleaned = clean(sentence, remove_stops=remove_stops)
        tokens = tokenize(cleaned)
        if tokens:  # drop sentences that cleaned to nothing
            result.append(tokens)
    return result


def show_pipeline(raw: str, *, width: int = 88) -> None:
    """
    Print a stage-by-stage before/after view of the cleaning pipeline.
    Intended for the report's 'raw vs processed' requirement.

    Args:
        raw:   Raw abstract string to trace through.
        width: Terminal wrap width for display.
    """
    from cleaner import (
        strip_latex,
        strip_urls,
        strip_citations,
        to_lowercase,
        strip_punctuation,
        strip_digits,
        normalize_whitespace,
        remove_stopwords,
    )

    sep = "─" * width

    def _show(label: str, text: str) -> None:
        wrapped = textwrap.fill(text, width=width - 2, subsequent_indent="  ")
        print(f"\n{'[ ' + label + ' ]':─<{width}}")
        print(f"  {wrapped}")

    print(sep)
    print("PIPELINE TRACE".center(width))
    print(sep)

    stages: list[tuple[str, Callable[[str], str]]] = [
        ("0 · Raw input", lambda t: t),
        ("1 · strip_latex", strip_latex),
        ("2 · strip_urls", strip_urls),
        ("3 · strip_citations", strip_citations),
        ("4 · to_lowercase", to_lowercase),
        ("5 · strip_punctuation", strip_punctuation),
        ("6 · strip_digits", strip_digits),
        ("7 · normalize_whitespace", normalize_whitespace),
        ("8 · remove_stopwords", remove_stopwords),
    ]

    current = raw
    for label, fn in stages:
        current = fn(current)
        _show(label, current)

    final_tokens = tokenize(current)
    print(f"\n{'[ 9 · tokenize -> token list ]':─<{width}}")
    print(f"  {final_tokens}")

    sentences = sent_segment(raw)
    print(f"\n{'[ sentence segmentation (on raw) ]':─<{width}}")
    for i, s in enumerate(sentences, 1):
        print(f"  [{i}] {s}")

    print(f"\n{sep}\n")
