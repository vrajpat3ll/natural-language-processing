"""
Dataset summary utilities for Lab 1 - Task 1.

Computes corpus-level and per-label statistics required by the assignment
summary table:
    - Number of Documents
    - Number of Sentences
    - Average Sentence Length  (tokens / sentences)
    - Total Tokens
    - Source

Usage
-----
    from analyze import DatasetAnalyzer

    analyzer = DatasetAnalyzer(df)
    analyzer.print()
    analyzer.print_per_label()
    stats = analyzer.global_stats()   # dict, for programmatic use
"""

from __future__ import annotations

import re
from functools import cached_property

import nltk
import pandas as pd


def _ensure_nltk_resources() -> None:
    """Download required NLTK data if not already present."""
    for resource, path in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(path, quiet=True)


_ensure_nltk_resources()


class DatasetAnalyzer:
    """
    Computes and formats corpus statistics for a scraped arXiv DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `ArXivScraper.collect()`.  Must contain at minimum the
        columns `abstract` and `label`.
    text_col : str
        Column to treat as the document text (default: `"abstract"`).
    source : str
        Human-readable dataset source string shown in the summary table.
    """

    SOURCE_DEFAULT = "arXiv API via `arxiv` Python client"

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str = "abstract",
        source: str | None = None,
    ) -> None:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")
        self.df = df.copy()
        self.text_col = text_col
        self.source = source or self.SOURCE_DEFAULT

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Word-tokenize a single string using NLTK (lowercased)."""
        return nltk.word_tokenize(text.lower())

    @staticmethod
    def _sentencize(text: str) -> list[str]:
        """Split a single string into sentences using NLTK Punkt."""
        return nltk.sent_tokenize(text)

    @staticmethod
    def _clean_for_count(text: str) -> str:
        """Minimal cleaning: collapse whitespace, strip leading/trailing."""
        return re.sub(r"\s+", " ", text).strip()

    # ? Cached per-row derived columns (computed once, reused everywhere)
    @cached_property
    def _sentences_series(self) -> pd.Series:
        """Series of sentence lists, one entry per document row."""
        return self.df[self.text_col].apply(
            lambda t: self._sentencize(self._clean_for_count(t))
        )

    @cached_property
    def _tokens_series(self) -> pd.Series:
        """Series of token lists, one entry per document row."""
        return self.df[self.text_col].apply(
            lambda t: self._tokenize(self._clean_for_count(t))
        )

    @cached_property
    def _sentence_counts(self) -> pd.Series:
        return self._sentences_series.apply(len)

    @cached_property
    def _token_counts(self) -> pd.Series:
        return self._tokens_series.apply(len)

    def global_stats(self) -> dict:
        """
        Return corpus-level statistics as a plain dict.

        Keys match the assignment's summary table properties.
        """
        n_docs = len(self.df)
        n_sentences = int(self._sentence_counts.sum())
        total_tokens = int(self._token_counts.sum())
        avg_sent_len = round(total_tokens / n_sentences, 2) if n_sentences else 0.0

        return {
            "Number of Documents": n_docs,
            "Number of Sentences": n_sentences,
            "Average Sentence Length": avg_sent_len,
            "Total Tokens": total_tokens,
            "Source": self.source,
        }

    def per_label_stats(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per label group containing the same
        statistics as `global_stats()`.
        """
        # Attach derived counts temporarily for groupby
        tmp = self.df[["label"]].copy()
        tmp["_n_sentences"] = self._sentence_counts.values
        tmp["_n_tokens"] = self._token_counts.values

        rows = []
        for label, grp in tmp.groupby("label"):
            n_docs = len(grp)
            n_sents = int(grp["_n_sentences"].sum())
            tot_tokens = int(grp["_n_tokens"].sum())
            avg_sl = round(tot_tokens / n_sents, 2) if n_sents else 0.0

            rows.append(
                {
                    "Label": label,
                    "Number of Documents": n_docs,
                    "Number of Sentences": n_sents,
                    "Average Sentence Length": avg_sl,
                    "Total Tokens": tot_tokens,
                    "Source": self.source,
                }
            )

        return pd.DataFrame(rows).set_index("Label")

    def print(self) -> str:
        """Print the assignment-style summary table to stdout."""
        stats = self.global_stats()
        
        # Calculate dynamic column widths based on content
        col_w = max(len(prop) for prop in stats.keys()) + 2
        val_w = max(len(f"{val:,}" if isinstance(val, int) else str(val)) for val in stats.values()) + 2
        sep = "+" + "-" * col_w + "+" + "-" * val_w + "+"

        output = []
        output += [sep]
        output += [f"| {'Property':<{col_w - 1}}| {'Value':<{val_w - 1}}|"]
        output += [sep]
        for prop, val in stats.items():
            val_str = f"{val:,}" if isinstance(val, int) else str(val)
            output += [f"| {prop:<{col_w - 1}}| {val_str:<{val_w - 1}}|"]
        output += [sep]

        output = "\n".join(output)
        print(output)
        return output

    def print_per_label(self) -> str:
        """Print per-label breakdown to stdout using pandas formatting."""
        df = self.per_label_stats().drop(columns=["Source"])
        # Format integer columns with commas
        int_cols = ["Number of Documents", "Number of Sentences", "Total Tokens"]
        for c in int_cols:
            df[c] = df[c].apply(lambda x: f"{x:,}")
        output = []
        output += ["\nPer-Label Summary"]
        output += ["=" * 70]
        output += [df.to_string()]
        output += []
        output = "\n".join(output)

        print(output)
        return output
