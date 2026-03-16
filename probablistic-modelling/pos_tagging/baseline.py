"""
pos_tagging/baseline.py

Most-Frequent-Tag baseline POS tagger.

For every word seen during training, always predict the tag it appeared
with most often.  For OOV words, fall back to the single most frequent
tag in the entire training corpus (empirically "NN" on Penn Treebank).

This baseline is surprisingly strong (~90 %+ token accuracy on PTB)
because most English words have one dominant part of speech.  It serves
as the lower-bound comparison for the HMMTagger in evaluate.py.

Usage:
    from pos_tagging.baseline import MostFrequentTagBaseline

    baseline = MostFrequentTagBaseline()
    baseline.fit(corpus.train)
    pred = baseline.predict(["The", "cat", "sat"])
    # ["DT", "NN", "VBD"]
"""

from __future__ import annotations

from collections import defaultdict, Counter

from utils import get_logger

log = get_logger(__name__)

# Type alias — mirrors corpus.py
Sentence = list[tuple[str, str]]


class MostFrequentTagBaseline:
    """
    Most-Frequent-Tag POS tagger.

    Attributes:
        _word_to_tag : dict[str, str]
            Maps every training word to its most frequent observed tag.
        _default_tag : str
            Fallback tag for OOV words — the globally most frequent tag.
        _fitted : bool
            Whether fit() has been called.
    """

    def __init__(self) -> None:
        self._word_to_tag: dict[str, str] = {}
        self._default_tag: str = "NN"
        self._fitted: bool = False

    # Training
    def fit(self, sentences: list[Sentence]) -> "MostFrequentTagBaseline":
        """
        Build the word -> most-frequent-tag lookup from tagged sentences.

        Args:
            sentences: Training sentences, each a list of (word, tag) tuples.

        Returns:
            self  (allows method chaining)
        """
        # word -> Counter of tags
        word_tag_counts: dict[str, Counter] = defaultdict(Counter)
        global_tag_counts: Counter = Counter()

        for sent in sentences:
            for word, tag in sent:
                word_tag_counts[word][tag] += 1
                global_tag_counts[tag] += 1

        # For each word, keep the single most common tag
        self._word_to_tag = {
            word: tag_counts.most_common(1)[0][0]
            for word, tag_counts in word_tag_counts.items()
        }

        # Global fallback for OOV words
        self._default_tag = global_tag_counts.most_common(1)[0][0]

        self._fitted = True
        log.info(
            "Baseline fitted: vocab=%d words, default_tag='%s'",
            len(self._word_to_tag),
            self._default_tag,
        )
        return self

    # Inference
    def predict(self, words: list[str]) -> list[str]:
        """
        Predict POS tags for a list of words.

        Args:
            words: Tokenised sentence as a plain list of strings.

        Returns:
            List of predicted POS tag strings, one per word.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        return [self._word_to_tag.get(word, self._default_tag) for word in words]

    def predict_batch(self, sentences: list[list[str]]) -> list[list[str]]:
        """
        Predict POS tags for a batch of tokenised sentences.

        Args:
            sentences: List of tokenised sentences.

        Returns:
            List of predicted tag sequences.
        """
        return [self.predict(words) for words in sentences]
