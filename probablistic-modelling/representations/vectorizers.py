"""
representations/vectorizers.py

Wraps scikit-learn vectorizers to produce the three text representations
required by the project:

    - Binary Bag-of-Words  (CountVectorizer, binary=True)
    - Term Frequency       (CountVectorizer, binary=False)
    - TF-IDF               (TfidfVectorizer)

Design principles:
    - Vectorizers are always fit on training data only.
    - Transform is a separate step (no fit_transform on val/test).
    - All three share identical vocabulary settings for fair comparison.
    - Returns scipy sparse matrices — compatible with MultinomialNB directly.
    - Fitted vectorizers are persisted to disk via joblib for reuse across
      experiments without re-fitting.

Saved file layout:
    representations/saved/{dataset}_{representation}.joblib
    e.g.  ag_news_tfidf.joblib
          arxiv_binary_bow.joblib

Usage:
    from representations.vectorizers import (
        build_vectorizers,
        load_vectorizers,
        Representation,
    )

    # First run — fits and saves
    reps = build_vectorizers(train_texts, dataset="ag_news")

    # Subsequent runs — loads from disk (skip re-fitting)
    reps = load_vectorizers(dataset="ag_news")

    X_train = reps[Representation.TFIDF].transform(train_texts)
    X_test  = reps[Representation.TFIDF].transform(test_texts)
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Iterable

import joblib
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils import get_logger

log = get_logger(__name__)

# Constants — shared vocabulary settings

# Ignore terms appearing in more than 95 % of documents (near-constant, uninformative)
MAX_DF: float = 0.95

# Ignore terms appearing in fewer than 2 documents (hapax legomena add noise)
MIN_DF: int = 2

# Cap vocabulary size — keeps memory tractable for 100k+ AG News documents
MAX_FEATURES: int = 50_000

# Directory where fitted vectorizers are saved
_SAVED_DIR: Path = Path(__file__).parent / "saved"


class Representation(Enum):
    BINARY_BOW = auto()
    TF = auto()
    TFIDF = auto()

    def file_stem(self) -> str:
        """Short lowercase string used in saved filenames."""
        return {
            Representation.BINARY_BOW: "binary_bow",
            Representation.TF: "tf",
            Representation.TFIDF: "tfidf",
        }[self]


class FittedVectorizer:
    """
    Thin wrapper around a fitted sklearn vectorizer.

    Exposes:
        .transform(texts)  -> scipy sparse matrix
        .vocabulary_size   -> int
        .name              -> str  (human-readable label for reports)
    """

    def __init__(
        self, vectorizer: CountVectorizer | TfidfVectorizer, name: str
    ) -> None:
        self._vectorizer = vectorizer
        self.name = name

    def transform(self, texts: Iterable[str]) -> spmatrix:
        """
        Transform a list of cleaned text strings into a sparse matrix.

        Args:
            texts: Iterable of cleaned document strings.
                   Must NOT be fit again — only transform is called.

        Returns:
            scipy sparse matrix of shape (n_docs, vocab_size).
        """
        return self._vectorizer.transform(texts)

    @property
    def vocabulary_size(self) -> int:
        return len(self._vectorizer.vocabulary_)

    def __repr__(self) -> str:
        return (
            f"FittedVectorizer(name={self.name!r}, vocab_size={self.vocabulary_size})"
        )


# Save / load helpers
def _saved_path(dataset: str, rep: Representation) -> Path:
    """Return the joblib path for a given dataset + representation."""
    return _SAVED_DIR / f"{dataset}_{rep.file_stem()}.joblib"


def save_vectorizers(
    vectorizers: dict[Representation, FittedVectorizer],
    dataset: str,
) -> None:
    """
    Persist all fitted vectorizers for a dataset to disk.

    Args:
        vectorizers: Dict returned by build_vectorizers().
        dataset:     Short dataset name used in filenames,
                     e.g. "ag_news" or "arxiv".
    """
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    for rep, fitted in vectorizers.items():
        path = _saved_path(dataset, rep)
        joblib.dump(fitted, path)
        log.info("Saved %s vectorizer to '%s'.", fitted.name, path)


def load_vectorizers(dataset: str) -> dict[Representation, FittedVectorizer]:
    """
    Load all three fitted vectorizers for a dataset from disk.

    Args:
        dataset: Short dataset name, e.g. "ag_news" or "arxiv".

    Returns:
        Dict mapping Representation -> FittedVectorizer.

    Raises:
        FileNotFoundError: if any of the three expected files are missing.
                           Call build_vectorizers() first to create them.
    """
    result: dict[Representation, FittedVectorizer] = {}
    for rep in Representation:
        path = _saved_path(dataset, rep)
        if not path.exists():
            raise FileNotFoundError(
                f"Saved vectorizer not found: '{path}'. "
                f"Run build_vectorizers(train_texts, dataset={dataset!r}) first."
            )
        result[rep] = joblib.load(path)
        log.info("Loaded %s vectorizer from '%s'.", result[rep].name, path)
    return result


def vectorizers_exist(dataset: str) -> bool:
    """Return True if all three saved vectorizer files exist for a dataset."""
    return all(_saved_path(dataset, rep).exists() for rep in Representation)


# Public API


def build_vectorizers(
    train_texts: Iterable[str],
    dataset: str,
    *,
    max_df: float | int = MAX_DF,
    min_df: int = MIN_DF,
    max_features: int = MAX_FEATURES,
    force: bool = False,
) -> dict[Representation, FittedVectorizer]:
    """
    Fit all three vectorizers on training texts, save them, and return them.

    If saved files already exist for this dataset and force=False, loads
    from disk instead of re-fitting (cache-friendly).

    All three vectorizers share the same max_df / min_df / max_features
    settings so vocabulary differences don't confound representation
    comparisons.

    Args:
        train_texts:  Iterable of cleaned training document strings.
                      Must be the training split only — never val or test.
        dataset:      Short name used in saved filenames,
                      e.g. "ag_news" or "arxiv".
        max_df:       Ignore terms in more than this fraction of documents.
        min_df:       Ignore terms in fewer than this many documents.
        max_features: Maximum vocabulary size.
        force:        If True, re-fit even if saved files already exist.

    Returns:
        Dict mapping Representation -> FittedVectorizer.
    """
    if not force and vectorizers_exist(dataset):
        log.info(
            "Saved vectorizers found for dataset '%s'. Loading from disk.", dataset
        )
        return load_vectorizers(dataset)

    shared_kwargs = dict(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        token_pattern=r"(?u)\b\w\w+\b",  # 2+ char tokens, consistent with cleaner.py
    )

    specs: list[tuple[Representation, str, CountVectorizer | TfidfVectorizer]] = [
        (
            Representation.BINARY_BOW,
            "Binary BoW",
            CountVectorizer(binary=True, **shared_kwargs),
        ),
        (
            Representation.TF,
            "Term Frequency",
            CountVectorizer(binary=False, **shared_kwargs),
        ),
        (
            Representation.TFIDF,
            "TF-IDF",
            TfidfVectorizer(**shared_kwargs),
        ),
    ]

    # Materialise once — avoids exhausting a generator across three .fit() calls
    train_list = list(train_texts)
    
    result: dict[Representation, FittedVectorizer] = {}
    for rep, name, vectorizer in specs:
        log.info("Fitting %s vectorizer on %d documents ...", name, len(train_list))
        vectorizer.fit(train_list)
        fitted = FittedVectorizer(vectorizer, name=name)
        log.info("  -> vocabulary size: %d", fitted.vocabulary_size)
        result[rep] = fitted

    save_vectorizers(result, dataset=dataset)

    return result
