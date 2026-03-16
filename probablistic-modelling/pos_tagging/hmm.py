"""
pos_tagging/hmm.py

First-order Hidden Markov Model POS tagger.

Model:
    Hidden states  : POS tags  (t_i)
    Observations   : words     (w_i)

    P(tags | words) proportional to P(t_1) * ∏ P(t_i | t_{i-1}) * ∏ P(w_i | t_i)
                                    initial   transitions              emissions

All probabilities are stored and computed in **log space** to avoid
numerical underflow on long sentences.

OOV handling:
    Words appearing fewer than `unk_threshold` times in training are
    replaced by the special token "<UNK>" during fit().  At prediction
    time, any word absent from the training vocabulary is also mapped to
    "<UNK>".  This gives OOV words a non-zero emission probability.

Sentence boundaries:
    A special "<START>" pseudo-tag is prepended to every training sentence
    so that initial-state probabilities are captured naturally inside the
    transition matrix — no separate π vector is needed.  "<END>" is
    appended to handle terminal transitions cleanly.

Usage:
    from pos_tagging.hmm import HMMTagger

    tagger = HMMTagger(unk_threshold=5)
    tagger.fit(corpus.train)
    tags = tagger.predict(["The", "cat", "sat"])
    # ["DT", "NN", "VBD"]
"""

from __future__ import annotations

import math
from collections import defaultdict, Counter

from utils import get_logger

log = get_logger(__name__)

# Sentinel tokens / tags
_UNK = "<UNK>"
_START = "<START>"
_END = "<END>"

_NEG_INF = float("-inf")

# Type alias
Sentence = list[tuple[str, str]]


class HMMTagger:
    """
    First-order HMM POS tagger with log-space Viterbi decoding.

    Args:
        unk_threshold:
            Words with training frequency strictly below this value are
            replaced by "<UNK>".  Default 5 (standard PTB practice).

    Attributes:
        tags_          : sorted list of real POS tags (excludes <START>/<END>)
        vocab_         : set of known words (includes <UNK>)
        log_trans_     : dict[prev_tag][tag]  -> log P(tag | prev_tag)
        log_emit_      : dict[tag][word]      -> log P(word | tag)
        _fitted        : bool
    """

    def __init__(self, unk_threshold: int = 5) -> None:
        self.unk_threshold = unk_threshold

        # Populated by fit()
        self.tags_: list[str] = []
        self.vocab_: set[str] = set()
        self.log_trans_: dict[str, dict[str, float]] = {}
        self.log_emit_: dict[str, dict[str, float]] = {}
        self._fitted: bool = False

    # Training
    def fit(self, sentences: list[Sentence]) -> "HMMTagger":
        """
        Estimate transition and emission probabilities from tagged sentences.

        Steps:
            1. Count word frequencies; map rare words to <UNK>.
            2. Count tag bigrams (with <START> / <END> sentinels).
            3. Count (tag, word) emission pairs.
            4. Normalise counts to log-probabilities with Laplace smoothing.

        Args:
            sentences: Training sentences, each a list of (word, tag) tuples.

        Returns:
            self  (allows method chaining)
        """
        log.info("HMMTagger.fit() — counting corpus statistics ...")

        # ── Step 1: word frequency -> OOV map ──────────────────────────
        word_freq: Counter = Counter()
        for sent in sentences:
            for word, _ in sent:
                word_freq[word] += 1

        known_words: set[str] = {
            w for w, c in word_freq.items() if c >= self.unk_threshold
        }
        known_words.add(_UNK)
        self.vocab_ = known_words

        def _unk(word: str) -> str:
            return word if word in known_words else _UNK

        # ── Step 2: transition counts ──────────────────────────────────
        # trans_counts[prev_tag][tag] = count
        trans_counts: dict[str, Counter] = defaultdict(Counter)

        # ── Step 3: emission counts ────────────────────────────────────
        # emit_counts[tag][word] = count
        emit_counts: dict[str, Counter] = defaultdict(Counter)

        # Collect all real tags (exclude sentinels)
        all_tags: set[str] = set()

        for sent in sentences:
            # Prepend START sentinel
            prev_tag = _START
            for word, tag in sent:
                all_tags.add(tag)
                trans_counts[prev_tag][tag] += 1
                emit_counts[tag][_unk(word)] += 1
                prev_tag = tag
            # Append END sentinel
            trans_counts[prev_tag][_END] += 1

        self.tags_ = sorted(all_tags)
        all_tags_with_end = self.tags_ + [_END]

        log.info(
            "Corpus: %d sentences, %d real tags, vocab size=%d (unk_threshold=%d)",
            len(sentences),
            len(self.tags_),
            len(self.vocab_),
            self.unk_threshold,
        )

        # ── Step 4a: log transition probabilities (Laplace smoothed) ──
        # Denominator = count(prev_tag) + |tags_with_end|  (add-1)
        all_prev_tags = [_START] + self.tags_  # valid previous tags
        self.log_trans_ = {}

        for prev_tag in all_prev_tags:
            counts = trans_counts[prev_tag]
            denom = sum(counts.values()) + len(all_tags_with_end)  # Laplace
            self.log_trans_[prev_tag] = {
                tag: math.log((counts[tag] + 1) / denom) for tag in all_tags_with_end
            }

        # ── Step 4b: log emission probabilities (Laplace smoothed) ────
        # Denominator = count(tag) + |vocab|  (add-1 over vocab)
        vocab_size = len(self.vocab_)
        self.log_emit_ = {}

        for tag in self.tags_:
            counts = emit_counts[tag]
            denom = sum(counts.values()) + vocab_size  # Laplace
            self.log_emit_[tag] = {
                word: math.log((counts[word] + 1) / denom) for word in self.vocab_
            }

        self._fitted = True
        log.info("HMMTagger fitted successfully.")
        return self

    # Inference
    def predict(self, words: list[str]) -> list[str]:
        """
        Predict the most likely POS tag sequence via the Viterbi algorithm.

        Args:
            words: Tokenised sentence as a plain list of strings.

        Returns:
            List of POS tag strings, one per word.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        if not words:
            return []

        return self._viterbi([self._unk(w) for w in words])

    def predict_batch(self, sentences: list[list[str]]) -> list[list[str]]:
        """
        Predict POS tags for a batch of tokenised sentences.

        Args:
            sentences: List of tokenised sentences (list of word lists).

        Returns:
            List of predicted tag sequences.
        """
        return [self.predict(words) for words in sentences]

    # Viterbi (log-space DP)
    def _unk(self, word: str) -> str:
        """Map an unseen word to <UNK>."""
        return word if word in self.vocab_ else _UNK

    def _viterbi(self, words: list[str]) -> list[str]:
        """
        Log-space Viterbi decoder.

        Trellis layout:
            viterbi[i][tag] = log P of the best path ending at `tag` at position i
            backptr[i][tag] = the best previous tag for that path

        Args:
            words: OOV-mapped word sequence (output of _unk()).

        Returns:
            Best POS tag sequence.
        """
        tags = self.tags_
        n = len(words)

        # viterbi[i] : dict[tag -> best log-prob reaching tag at position i]
        # backptr[i] : dict[tag -> best previous tag]
        viterbi: list[dict[str, float]] = [{} for _ in range(n)]
        backptr: list[dict[str, str]] = [{} for _ in range(n)]

        # ── Initialisation (position 0) ────────────────────────────────
        word0 = words[0]
        for tag in tags:
            log_trans = self.log_trans_[_START].get(tag, _NEG_INF)
            log_emit = self.log_emit_[tag].get(word0, _NEG_INF)
            viterbi[0][tag] = log_trans + log_emit
            backptr[0][tag] = _START

        # ── Recursion ──────────────────────────────────────────────────
        for i in range(1, n):
            word_i = words[i]
            for tag in tags:
                log_emit = self.log_emit_[tag].get(word_i, _NEG_INF)
                if log_emit == _NEG_INF:
                    # Emission is impossible — skip this tag entirely
                    viterbi[i][tag] = _NEG_INF
                    backptr[i][tag] = tags[0]
                    continue

                best_score = _NEG_INF
                best_prev = tags[0]
                for prev_tag in tags:
                    prev_score = viterbi[i - 1][prev_tag]
                    if prev_score == _NEG_INF:
                        continue
                    log_trans = self.log_trans_[prev_tag].get(tag, _NEG_INF)
                    score = prev_score + log_trans + log_emit
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag

                viterbi[i][tag] = best_score
                backptr[i][tag] = best_prev

        # ── Termination ────────────────────────────────────────────────
        # Pick the tag at the last position with the highest score,
        # accounting for the transition to <END>
        best_last_score = _NEG_INF
        best_last_tag = tags[0]
        for tag in tags:
            score = viterbi[n - 1][tag] + self.log_trans_[tag].get(_END, _NEG_INF)
            if score > best_last_score:
                best_last_score = score
                best_last_tag = tag

        # ── Backtracking ───────────────────────────────────────────────
        path: list[str] = [best_last_tag]
        for i in range(n - 1, 0, -1):
            path.append(backptr[i][path[-1]])
        path.reverse()

        return path
