"""
pos_tagging/evaluate.py

Evaluation utilities for POS taggers.

Functions:
    token_accuracy           — flat token-level accuracy across all sentences
    sentence_accuracy        — fraction of fully correct sentences
    confusion_pairs          — top-N most confused (gold, pred) tag pairs
    classification_report_df — per-tag precision / recall / F1 as DataFrame

Usage:
    from pos_tagging.evaluate import token_accuracy, confusion_pairs

    acc  = token_accuracy(pred_seqs, gold_seqs)
    conf = confusion_pairs(pred_seqs, gold_seqs, top_n=10)
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


def token_accuracy(
    predicted: list[list[str]],
    gold: list[list[str]],
) -> float:
    """
    Compute flat token-level accuracy across all sentences.

    Counts every (predicted_tag, gold_tag) pair and returns the fraction
    that match — the standard metric reported for POS taggers.

    Args:
        predicted: Predicted tag sequences, one list per sentence.
        gold:      Gold tag sequences, one list per sentence.

    Returns:
        Float in [0, 1].

    Raises:
        ValueError: If the number of sentences or any sentence length differs.
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"Sentence count mismatch: predicted={len(predicted)}, gold={len(gold)}"
        )

    correct = total = 0
    for pred_seq, gold_seq in zip(predicted, gold):
        if len(pred_seq) != len(gold_seq):
            raise ValueError(
                f"Sentence length mismatch: "
                f"predicted={len(pred_seq)}, gold={len(gold_seq)}"
            )
        for p, g in zip(pred_seq, gold_seq):
            total += 1
            if p == g:
                correct += 1

    return correct / total if total > 0 else 0.0


def sentence_accuracy(
    predicted: list[list[str]],
    gold: list[list[str]],
) -> float:
    """
    Compute sentence-level accuracy — fraction of sentences where every
    token is tagged correctly.

    Args:
        predicted: Predicted tag sequences.
        gold:      Gold tag sequences.

    Returns:
        Float in [0, 1].
    """
    if not predicted:
        return 0.0

    correct = sum(
        1 for pred_seq, gold_seq in zip(predicted, gold) if pred_seq == gold_seq
    )
    return correct / len(predicted)


def confusion_pairs(
    predicted: list[list[str]],
    gold: list[list[str]],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-N most confused (gold_tag, predicted_tag) pairs.

    Only mismatches are counted — correct predictions are excluded.
    This is the most informative slice of the confusion matrix for
    error analysis.

    Args:
        predicted: Predicted tag sequences.
        gold:      Gold tag sequences.
        top_n:     Number of top confused pairs to return (default 10).

    Returns:
        DataFrame with columns [gold, predicted, count] sorted descending
        by count.
    """
    counts: Counter = Counter()
    for pred_seq, gold_seq in zip(predicted, gold):
        for p, g in zip(pred_seq, gold_seq):
            if p != g:
                counts[(g, p)] += 1

    rows = [
        {"gold": g, "predicted": p, "count": c}
        for (g, p), c in counts.most_common(top_n)
    ]
    return pd.DataFrame(rows)


def classification_report_df(
    predicted: list[list[str]],
    gold: list[list[str]],
) -> pd.DataFrame:
    """
    Per-tag precision, recall, and F1 score as a tidy DataFrame.

    Macro-averages are appended as the final row.

    Args:
        predicted: Predicted tag sequences.
        gold:      Gold tag sequences.

    Returns:
        DataFrame indexed by tag with columns [precision, recall, f1, support].
    """
    # Flatten
    flat_pred = [t for seq in predicted for t in seq]
    flat_gold = [t for seq in gold for t in seq]

    tags = sorted(set(flat_gold))

    # Per-tag TP, FP, FN
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for p, g in zip(flat_pred, flat_gold):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    rows = []
    for tag in tags:
        prec = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) > 0 else 0.0
        rec = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp[tag] + fn[tag]
        rows.append(
            {"tag": tag, "precision": prec, "recall": rec, "f1": f1, "support": support}
        )

    df = pd.DataFrame(rows).set_index("tag")

    # Macro averages
    macro = df[["precision", "recall", "f1"]].mean()
    macro["support"] = df["support"].sum()
    macro.name = "macro-avg"
    df = pd.concat([df, macro.to_frame().T])

    return df.round(4)
