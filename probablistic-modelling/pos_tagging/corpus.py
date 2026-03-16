"""
pos_tagging/corpus.py

Loads the NLTK Penn Treebank sample into a TaggedCorpus — the POS-tagging
analogue of DataSplit from loader.py.

Pipeline:
    1. Download nltk treebank corpus if not already present
    2. Load all tagged sentences as list[list[tuple[str, str]]]
    3. Shuffle with RANDOM_SEED (42) for reproducibility
    4. Split 80 / 10 / 10  (plain shuffle — no stratification needed)
    5. Return a TaggedCorpus

TaggedCorpus contract:
    Each split is a list of sentences.
    Each sentence is a list of (word, tag) tuples.
    Tags follow the Penn Treebank tagset (~36 tags).

Usage:
    from pos_tagging.corpus import load_treebank

    corpus = load_treebank()
    corpus.train   # list[list[tuple[str, str]]]
    corpus.val
    corpus.test
    print(corpus.summary())
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd
import nltk

from utils import get_logger

log = get_logger(__name__)

# Reuse the project-wide seed — mirrors loader.py
RANDOM_SEED: int = 42

# Split fractions — mirror the arXiv split in loader.py
_TRAIN_FRAC: float = 0.80
_VAL_FRAC: float = 0.10
# test is the remainder: 0.10

# Sentence = list of (word, tag) pairs
Sentence = list[tuple[str, str]]


@dataclass(frozen=True)
class TaggedCorpus:
    """
    Immutable container for train / val / test POS-tagged sentence splits.

    Each split is a list of sentences.
    Each sentence is a list of (word, tag) tuples using the Penn Treebank tagset.

    Mirrors the structure and API of DataSplit in loader.py.
    """

    train: list[Sentence]
    val: list[Sentence]
    test: list[Sentence]

    def __repr__(self) -> str:
        return (
            f"TaggedCorpus("
            f"train={len(self.train)} sents, "
            f"val={len(self.val)} sents, "
            f"test={len(self.test)} sents)"
        )

    def summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame with per-split statistics:
            sentences  — number of sentences
            tokens     — total token count
            tagset     — number of unique POS tags seen
        Handy for quick sanity checks, mirrors DataSplit.summary().
        """
        rows = []
        for name, split in [
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ]:
            tokens = [tok for sent in split for tok in sent]
            tagset = {tag for _, tag in tokens}
            rows.append(
                {
                    "split": name,
                    "sentences": len(split),
                    "tokens": len(tokens),
                    "tagset_size": len(tagset),
                }
            )
        return pd.DataFrame(rows).set_index("split")


def _ensure_nltk_treebank() -> None:
    """Download the NLTK treebank corpus if not already present."""
    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        log.info("Downloading NLTK treebank corpus ...")
        nltk.download("treebank", quiet=True)


def load_treebank(
    train_frac: float = _TRAIN_FRAC,
    val_frac: float = _VAL_FRAC,
    seed: int = RANDOM_SEED,
) -> TaggedCorpus:
    """
    Load the NLTK Penn Treebank sample and return a TaggedCorpus.

    The full corpus (~3 900 sentences) is shuffled once with `seed` then
    divided into train / val / test at the given fractions.

    Plain shuffle split — no stratification — because POS sentence
    distributions are already uniform across the NLTK treebank sample,
    and there is no class label to stratify on.

    Args:
        train_frac: Fraction of sentences for training (default 0.80).
        val_frac:   Fraction of sentences for validation (default 0.10).
        seed:       Random seed for shuffle reproducibility (default 42).

    Returns:
        TaggedCorpus with .train / .val / .test lists of tagged sentences.
    """
    _ensure_nltk_treebank()

    log.info("Loading Penn Treebank tagged sentences ...")
    sentences: list[Sentence] = list(nltk.corpus.treebank.tagged_sents())
    n = len(sentences)
    log.info("Loaded %d sentences from Penn Treebank sample.", n)

    # Reproducible shuffle — mirrors fixed seed convention in loader.py
    rng = random.Random(seed)
    sentences = sentences.copy()
    rng.shuffle(sentences)

    # Compute split boundaries
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = sentences[:train_end]
    val = sentences[train_end:val_end]
    test = sentences[val_end:]

    log.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))

    return TaggedCorpus(train=train, val=val, test=test)
