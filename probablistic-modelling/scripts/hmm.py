"""
scripts/hmm.py

Command-line entry point for the POS Tagging experiments.

Runs the full experiment pipeline:
    1. Load Penn Treebank via load_treebank()
    2. Fit MostFrequentTagBaseline and HMMTagger on training split
    3. Evaluate both on the test split  -> accuracy table
    4. Print top-10 confusion pairs for error analysis
    5. Ablation: retrain HMM at 5 training-size fractions -> accuracy-vs-size plot
    6. Save results to results/pos_tagging/

Usage:
    python scripts/hmm.py [--unk-threshold 5] [--results-dir results/pos_tagging]

Mirrors the structure of classifier_cli.py and lda_cli.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import pandas as pd

from pos_tagging.corpus import load_treebank, TaggedCorpus
from pos_tagging.baseline import MostFrequentTagBaseline
from pos_tagging.hmm import HMMTagger
from pos_tagging import evaluate as ev
from utils import get_logger

log = get_logger(__name__)

# Ablation fractions of the training set to evaluate
_ABLATION_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]

# Fixed seed — mirrors project-wide convention
_SEED = 42


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="POS Tagging — HMM vs Baseline")
    p.add_argument(
        "--unk-threshold",
        type=int,
        default=5,
        help="Words with freq < this are mapped to <UNK> (default: 5)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/pos_tagging"),
        help="Directory to save tables and plots (default: results/pos_tagging)",
    )
    return p.parse_args()


def _extract_words_and_tags(
    sentences: list[list[tuple[str, str]]],
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Separate a list of tagged sentences into parallel word and tag sequences.

    Returns:
        (word_seqs, tag_seqs) — two aligned lists of lists.
    """
    word_seqs = [[w for w, _ in sent] for sent in sentences]
    tag_seqs = [[t for _, t in sent] for sent in sentences]
    return word_seqs, tag_seqs


def _subsample_train(
    sentences: list[list[tuple[str, str]]],
    frac: float,
    seed: int = _SEED,
) -> list[list[tuple[str, str]]]:
    """Return a reproducible random subsample of `frac` of `sentences`."""
    n = max(1, int(len(sentences) * frac))
    rng = random.Random(seed)
    return rng.sample(sentences, n)


def run(args: argparse.Namespace) -> None:
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load corpus ─────────────────────────────────────────────────
    log.info("Loading Penn Treebank ...")
    corpus: TaggedCorpus = load_treebank()
    log.info("%s", corpus)
    print("\nCorpus summary:")
    print(corpus.summary().to_string())
    print()

    test_words, test_tags = _extract_words_and_tags(corpus.test)

    # ── 2. Fit models on full training split ───────────────────────────
    log.info("Fitting MostFrequentTagBaseline ...")
    baseline = MostFrequentTagBaseline().fit(corpus.train)

    log.info("Fitting HMMTagger (unk_threshold=%d) ...", args.unk_threshold)
    hmm = HMMTagger(unk_threshold=args.unk_threshold).fit(corpus.train)

    # ── 3. Evaluate on test split ──────────────────────────────────────
    log.info("Evaluating on test split ...")

    baseline_preds = baseline.predict_batch(test_words)
    hmm_preds = hmm.predict_batch(test_words)

    baseline_acc = ev.token_accuracy(baseline_preds, test_tags)
    hmm_acc = ev.token_accuracy(hmm_preds, test_tags)

    baseline_sent_acc = ev.sentence_accuracy(baseline_preds, test_tags)
    hmm_sent_acc = ev.sentence_accuracy(hmm_preds, test_tags)

    results_table = pd.DataFrame(
        [
            {
                "model": "Baseline (MFT)",
                "token_accuracy": round(baseline_acc, 4),
                "sentence_accuracy": round(baseline_sent_acc, 4),
            },
            {
                "model": "HMM",
                "token_accuracy": round(hmm_acc, 4),
                "sentence_accuracy": round(hmm_sent_acc, 4),
            },
        ]
    ).set_index("model")

    print("\n── Accuracy Results ──────────────────────────────────")
    print(results_table.to_string())
    results_table.to_csv(results_dir / "accuracy_results.csv")
    log.info("Saved accuracy table to '%s'.", results_dir / "accuracy_results.csv")

    # ── 4. Error analysis — top confusion pairs ────────────────────────
    print("\n── Top-10 Confusion Pairs (HMM) ──────────────────────")
    conf = ev.confusion_pairs(hmm_preds, test_tags, top_n=10)
    print(conf.to_string(index=False))
    conf.to_csv(results_dir / "hmm_confusion_pairs.csv", index=False)

    print("\n── Per-tag Classification Report (HMM) ───────────────")
    report = ev.classification_report_df(hmm_preds, test_tags)
    print(report.to_string())
    report.to_csv(results_dir / "hmm_classification_report.csv")

    # ── 5. Ablation: accuracy vs training size ─────────────────────────
    log.info("Running training-size ablation ...")
    ablation_rows = []

    for frac in _ABLATION_FRACS:
        sub_train = _subsample_train(corpus.train, frac)
        n_sents = len(sub_train)
        n_tokens = sum(len(s) for s in sub_train)

        # Baseline
        b = MostFrequentTagBaseline().fit(sub_train)
        b_preds = b.predict_batch(test_words)
        b_acc = ev.token_accuracy(b_preds, test_tags)

        # HMM
        h = HMMTagger(unk_threshold=args.unk_threshold).fit(sub_train)
        h_preds = h.predict_batch(test_words)
        h_acc = ev.token_accuracy(h_preds, test_tags)

        ablation_rows.append(
            {
                "frac": frac,
                "train_sentences": n_sents,
                "train_tokens": n_tokens,
                "baseline_acc": round(b_acc, 4),
                "hmm_acc": round(h_acc, 4),
            }
        )
        log.info(
            "frac=%.2f  sents=%d  baseline=%.4f  hmm=%.4f",
            frac,
            n_sents,
            b_acc,
            h_acc,
        )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(results_dir / "ablation_results.csv", index=False)

    print("\n── Training-Size Ablation ────────────────────────────")
    print(ablation_df.to_string(index=False))

    # ── 6. Plot: accuracy vs training size ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        ablation_df["train_sentences"],
        ablation_df["baseline_acc"],
        marker="o",
        linestyle="--",
        label="Baseline (MFT)",
        color="steelblue",
    )
    ax.plot(
        ablation_df["train_sentences"],
        ablation_df["hmm_acc"],
        marker="s",
        linestyle="-",
        label="HMM",
        color="darkorange",
    )
    ax.set_xlabel("Training Sentences")
    ax.set_ylabel("Token Accuracy")
    ax.set_title("POS Tagging Accuracy vs. Training Size\n(Penn Treebank, test split)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plot_path = results_dir / "accuracy_vs_training_size.png"
    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    log.info("Saved ablation plot to '%s'.", plot_path)

    print(f"\nAll results saved to '{results_dir}/'")


def main():
    run(_parse_args())


if __name__ == "__main__":
    main()
