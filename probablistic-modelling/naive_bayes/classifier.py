"""
naive_bayes/classifier.py

Naive Bayes classification experiments.

Trains a MultinomialNB classifier for each of the three representations
(Binary BoW, TF, TF-IDF) on both AG News and arXiv, evaluates on the
test set, and returns results as a tidy DataFrame suitable for the
report table.

Models are saved to naive_bayes/saved/{dataset}_{representation}.joblib
after training and can be reloaded via load_model().

Report table schema (per dataset):
    Representation | Accuracy | Macro F1 | Training Time (s)

Usage:
    from naive_bayes.classifier import run_naive_bayes, load_model
    from representations.vectorizers import Representation

    results = run_naive_bayes(split, vectorizers, dataset="ag_news")

    model = load_model("ag_news", Representation.TFIDF)
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

from representations.vectorizers import FittedVectorizer, Representation
from loader import DataSplit
from utils import get_logger

log = get_logger(__name__)

_SAVED_DIR: Path = Path(__file__).parent / "saved"


def _model_path(dataset: str, rep: Representation) -> Path:
    return _SAVED_DIR / f"{dataset}_{rep.file_stem()}.joblib"


def _save_model(model: MultinomialNB, dataset: str, rep: Representation) -> None:
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(dataset, rep)
    joblib.dump(model, path)
    log.info("Saved model to '%s'.", path)


def load_model(dataset: str, rep: Representation) -> MultinomialNB:
    """
    Load a previously saved MultinomialNB model from disk.

    Args:
        dataset: "ag_news" or "arxiv"
        rep:     Representation enum value

    Raises:
        FileNotFoundError: if the model file does not exist yet.
                           Run run_naive_bayes() for this dataset first.
    """
    path = _model_path(dataset, rep)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at '{path}'. "
            f"Run run_naive_bayes() for dataset='{dataset}' first."
        )
    model = joblib.load(path)
    log.info("Loaded model from '%s'.", path)
    return model


def _run_single(
    split: DataSplit,
    fitted: FittedVectorizer,
    rep: Representation,
    dataset: str,
) -> dict:
    """
    Train MultinomialNB on one representation, save to disk, evaluate on test.

    Args:
        split:   DataSplit with .train / .val / .test DataFrames.
        fitted:  FittedVectorizer for this representation.
        rep:     Representation enum value, used for save path and logging.
        dataset: Dataset name used for save path, e.g. "ag_news".

    Returns:
        Dict with keys: Representation, Accuracy, Macro F1, Training Time (s)
    """
    log.info("Running Naive Bayes with %s ...", fitted.name)

    # Transform only — vectorizers were fit in prepare_data.py
    X_train = fitted.transform(split.train["text"])
    X_test = fitted.transform(split.test["text"])
    y_train = split.train["label"].to_numpy()
    y_test = split.test["label"].to_numpy()

    # Train
    model = MultinomialNB()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Persist model
    _save_model(model, dataset, rep)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    log.info(
        "  %s -> Accuracy: %.4f | Macro F1: %.4f | Train time: %.3fs",
        fitted.name,
        accuracy,
        macro_f1,
        train_time,
    )

    return {
        "Representation": fitted.name,
        "Accuracy": round(accuracy, 4),
        "Macro F1": round(macro_f1, 4),
        "Training Time (s)": round(train_time, 3),
    }


def run_naive_bayes(
    split: DataSplit,
    vectorizers: dict[Representation, FittedVectorizer],
    dataset: str,
) -> pd.DataFrame:
    """
    Run Naive Bayes for all three representations on one dataset.

    Iterates over Binary BoW -> TF -> TF-IDF, trains a fresh MultinomialNB
    for each, saves it to disk, evaluates on test, and collects results.

    Args:
        split:       DataSplit for this dataset.
        vectorizers: Dict from build_vectorizers() or load_vectorizers().
        dataset:     Short name e.g. "ag_news" or "arxiv".

    Returns:
        DataFrame with one row per representation, columns:
            Representation | Accuracy | Macro F1 | Training Time (s)
        Rows are in BoW -> TF -> TF-IDF order matching the report table.

    Note on TF-IDF + MultinomialNB:
        TF-IDF scores are non-negative floats so MultinomialNB accepts them.
        Strictly, MultinomialNB assumes integer counts — ComplementNB would
        be more principled for TF-IDF, but MultinomialNB is used here for
        consistency across all three representations.
    """
    log.info("=" * 60)
    log.info("Naive Bayes experiments -- dataset: %s", dataset)
    log.info(
        "  Train: %d | Val: %d | Test: %d",
        len(split.train),
        len(split.val),
        len(split.test),
    )
    log.info("=" * 60)

    rows = []
    for rep in [Representation.BINARY_BOW, Representation.TF, Representation.TFIDF]:
        row = _run_single(split, vectorizers[rep], rep, dataset)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    log.info("Results for %s:\n%s", dataset, results_df.to_string(index=False))
    return results_df
