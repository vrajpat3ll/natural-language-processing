"""
topic_modeling/lda.py

LDA topic modeling experiments.

Trains Gensim LDA models with varying topic counts (5, 10, 20) on a
dataset, evaluates perplexity and CV coherence, extracts top words per
topic, and persists both the fitted models and results to disk.

Saved file layout:
    topic_modeling/saved/{dataset}_lda_{n_topics}.joblib  — fitted LDA
    results/lda/{dataset}.csv                              — metrics table
    results/lda/{dataset}_topics_{n_topics}.csv            — top words

Usage:
    from topic_modeling.lda import run_lda

    results_df, topic_dfs = run_lda(split, dataset="ag_news")
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from loader import DataSplit
from utils import get_logger

log = get_logger(__name__)

# Constants
TOPIC_COUNTS: list[int] = [5, 10, 20]
TOP_N_WORDS: int = 10  # words per topic to extract and save
PASSES: int = 10  # LDA training passes (convergence)
RANDOM_SEED: int = 42

_SAVED_DIR: Path = Path(__file__).parent / "saved"
_RESULTS_DIR: Path = Path(__file__).parent.parent / "results" / "lda"


def _build_corpus(
    texts: list[str],
) -> tuple[corpora.Dictionary, list[list[tuple[int, int]]]]:
    """
    Build a Gensim Dictionary and BoW corpus from a list of cleaned strings.

    Applies mild filtering on top of what the cleaner already did:
        - Drop tokens appearing in fewer than 5 documents
        - Drop tokens appearing in more than 50 % of documents
        - Keep at most 50 000 token types

    Args:
        texts: List of cleaned, whitespace-tokenised document strings.

    Returns:
        (dictionary, bow_corpus)
    """
    tokenised = [text.split() for text in texts]

    dictionary = corpora.Dictionary(tokenised)
    dictionary.filter_extremes(no_below=5, no_above=0.50, keep_n=50_000)

    bow_corpus = [dictionary.doc2bow(tokens) for tokens in tokenised]
    log.info(
        "Gensim corpus built — vocab: %d tokens, docs: %d",
        len(dictionary),
        len(bow_corpus),
    )
    return dictionary, bow_corpus


def _train_lda(
    bow_corpus: list[list[tuple[int, int]]],
    dictionary: corpora.Dictionary,
    n_topics: int,
    dataset: str,
    *,
    force: bool = False,
) -> LdaModel:
    """
    Train (or load from cache) an LDA model for a given topic count.

    Args:
        bow_corpus:  Gensim BoW corpus.
        dictionary:  Gensim Dictionary.
        n_topics:    Number of latent topics.
        dataset:     Short dataset name used in the saved filename.
        force:       Retrain even if a saved model exists.

    Returns:
        Fitted LdaModel.
    """
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _SAVED_DIR / f"{dataset}_lda_{n_topics}.joblib"

    if not force and save_path.exists():
        log.info("Cache hit — loading LDA (k=%d) for %s.", n_topics, dataset)
        return joblib.load(save_path)

    log.info(
        "Training LDA — dataset: %s | topics: %d | passes: %d ...",
        dataset,
        n_topics,
        PASSES,
    )
    t0 = time.perf_counter()

    model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=PASSES,
        alpha="auto",  # learned asymmetric prior — better topic separation
        eta="auto",  # learned asymmetric prior on word distribution
        random_state=RANDOM_SEED,
        eval_every=None,  # disable per-chunk perplexity logging (speed)
    )

    elapsed = time.perf_counter() - t0
    log.info("  Trained in %.1fs.", elapsed)

    joblib.dump(model, save_path)
    log.info("  Saved to '%s'.", save_path)

    return model


def _compute_metrics(
    model: LdaModel,
    bow_corpus: list[list[tuple[int, int]]],
    dictionary: corpora.Dictionary,
    texts: list[str],
    n_topics: int,
) -> tuple[float, float]:
    """
    Compute perplexity and CV coherence for a fitted LDA model.

    Args:
        model:      Fitted LdaModel.
        bow_corpus: Gensim BoW corpus (same used for training).
        dictionary: Gensim Dictionary.
        texts:      Tokenised texts as list-of-lists (for coherence).
        n_topics:   Number of topics (for logging).

    Returns:
        (perplexity, coherence_cv)
    """
    # Perplexity — lower is better
    perplexity = model.log_perplexity(bow_corpus)

    # CV coherence — higher is better, correlates well with human judgement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # gensim deprecation noise
        cm = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        )
        coherence = cm.get_coherence()

    log.info(
        "  k=%d -> perplexity: %.4f | coherence (CV): %.4f",
        n_topics,
        perplexity,
        coherence,
    )
    return round(perplexity, 4), round(coherence, 4)


def _extract_top_words(model: LdaModel, n_topics: int) -> pd.DataFrame:
    """
    Extract the top-N words for every topic as a DataFrame.

    Returns:
        DataFrame with columns: topic_id, rank, word, weight
    """
    rows = []
    for topic_id in range(n_topics):
        top_words = model.show_topic(topic_id, topn=TOP_N_WORDS)
        for rank, (word, weight) in enumerate(top_words, start=1):
            rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "word": word,
                    "weight": round(weight, 6),
                }
            )
    return pd.DataFrame(rows)


def run_lda(
    split: DataSplit,
    dataset: str,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Run LDA experiments for all topic counts on one dataset.

    Trains models for k ∈ {5, 10, 20}, computes perplexity + CV coherence,
    extracts top words per topic, and saves everything to disk.

    Args:
        split:   DataSplit for the dataset (uses .train["text"] for training).
        dataset: Short dataset name, e.g. "ag_news" or "arxiv".
        force:   If True, retrain models even if saved files exist.

    Returns:
        results_df  — DataFrame with columns:
                          Topics | Perplexity | Coherence (CV)
                      One row per topic count, matching the assignment table.

        topic_dfs   — Dict mapping n_topics -> DataFrame of top words
                          columns: topic_id | rank | word | weight
    """
    log.info("=" * 60)
    log.info("LDA experiments — dataset: %s", dataset)
    log.info("=" * 60)

    # Build corpus once — shared across all topic counts
    train_texts = split.train["text"].tolist()
    tokenised = [text.split() for text in train_texts]
    dictionary, bow_corpus = _build_corpus(train_texts)

    results_rows: list[dict] = []
    topic_dfs: dict[int, pd.DataFrame] = {}

    for n_topics in TOPIC_COUNTS:
        model = _train_lda(bow_corpus, dictionary, n_topics, dataset, force=force)
        perplexity, coherence = _compute_metrics(
            model, bow_corpus, dictionary, tokenised, n_topics
        )
        top_words_df = _extract_top_words(model, n_topics)

        results_rows.append(
            {
                "Topics": n_topics,
                "Perplexity": perplexity,
                "Coherence (CV)": coherence,
            }
        )
        topic_dfs[n_topics] = top_words_df

    results_df = pd.DataFrame(results_rows)

    # Persist results
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = _RESULTS_DIR / f"{dataset}.csv"
    results_df.to_csv(metrics_path, index=False)
    log.info("Metrics saved to '%s'.", metrics_path)

    for n_topics, df in topic_dfs.items():
        words_path = _RESULTS_DIR / f"{dataset}_topics_{n_topics}.csv"
        df.to_csv(words_path, index=False)
        log.info("Top words (k=%d) saved to '%s'.", n_topics, words_path)

    log.info("LDA results for %s:\n%s", dataset, results_df.to_string(index=False))
    return results_df, topic_dfs
