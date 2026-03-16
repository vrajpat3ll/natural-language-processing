"""
Lab 1 - Task 3: Normalization.

Provides:
  - stem(tokens)               Porter stemming
  - lemmatize(tokens)          WordNet lemmatization
  - normalize(tokens, method)  unified dispatcher

Stopword removal is intentionally kept in cleaner.py.
Import it from there:
    from cleaner import remove_stopwords
"""

from __future__ import annotations

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_resources() -> None:
    resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_resources()

# Module-level singletons (instantiation is cheap but no reason to repeat it)
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def stem(tokens: list[str]) -> list[str]:
    """
    Apply Porter stemming to a list of tokens.

    Porter is an aggressive rule-based stemmer that strips suffixes
    without consulting a dictionary.  Fast and deterministic.

    Examples:
        training   -> train
        performance -> perform
        networks   -> network
        running    -> run
        studies    -> studi      # known over-stemming artifact

    Args:
        tokens: List of lowercase word tokens.

    Returns:
        List of stemmed tokens (same length as input).
    """
    return [_stemmer.stem(tok) for tok in tokens]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Apply WordNet lemmatization to a list of tokens.

    Uses NLTK's WordNetLemmatizer with the default POS tag (noun).
    This means verb forms are not fully reduced without explicit POS context
    (e.g. 'running' stays 'running' instead of becoming 'run').
    Appropriate for bag-of-words tasks; for POS-aware lemmatization use spaCy.

    Examples:
        studies    -> study      (noun form correctly reduced)
        running    -> running    (verb; stays unchanged without POS hint)
        networks   -> network
        models     -> model
        better     -> better     (adjective; unchanged without POS hint)

    Args:
        tokens: List of lowercase word tokens.

    Returns:
        List of lemmatized tokens (same length as input).
    """
    return [_lemmatizer.lemmatize(tok) for tok in tokens]


_METHODS = {"stem", "lemmatize"}


def normalize(
    tokens: list[str],
    method: str,
) -> list[str]:
    """
    Unified normalization dispatcher.

    Args:
        tokens: List of word tokens (should already be lowercased).
        method: One of 'stem' or 'lemmatize'.

    Returns:
        Normalized token list.

    Raises:
        ValueError: If method is not recognized.
    """
    if method == "stem":
        return stem(tokens)
    if method == "lemmatize":
        return lemmatize(tokens)
    raise ValueError(
        f"Unknown normalization method '{method}'. Choose from: {sorted(_METHODS)}"
    )
