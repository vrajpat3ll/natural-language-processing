# NLP Lab - Assignment 5: Distributed Word Representations

This project implements and evaluates various distributed word representations (embeddings) on two datasets: the **arXiv CS corpus** (domain-specific abstracts) and **AG News** (general news).

## Project Structure

- `baseline_onehot.py`: Implements Part 1 (One-hot encoding baseline and discussion).
- `train_embeddings.py`: Implements Part 2 (Training CBOW, Skip-gram, FastText, and custom GloVe).
- `evaluate_embeddings.py`: Implements Part 3 (Similarity, Nearest Neighbors, and Analogies).
- `visualize_embeddings.py`: Implements Part 4 (PCA and t-SNE visualizations).
- `downstream_task.py`: Implements Part 5 (Text classification comparing BoW vs. Embeddings).
- `loader.py`: Data loading utilities for arXiv and AG News.
- `clean_text.py` & `cleaner.py`: Text preprocessing pipeline.
- `Lab_5_report.tex`: Final report in LaTeX format.

## Setup

This project uses `uv` for dependency management.

1. **Install uv** (if not already installed):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Sync dependencies**:
   ```powershell
   uv sync
   ```

## How to Run

Follow these steps in order to reproduce the full analysis:

### 1. Baseline Representation

Discusses dimensionality, sparsity, and lack of semantic similarity.

```powershell
uv run baseline_onehot.py
```

### 2. Training Word Embeddings

Trains models for dimensions 80, 100, and 200.
_Note: FastText for dim 200 is skipped by default to manage disk space._

```powershell
uv run train_embeddings.py
```

### 3. Embedding Evaluation

Calculates cosine similarities and performs nearest neighbor/analogy searches.

```powershell
uv run evaluate_embeddings.py
```

### 4. Visualization

Generates PCA and t-SNE plots in the `plots/` directory.

```powershell
uv run visualize_embeddings.py
```

### 5. Downstream Task

Compares classification accuracy between Bag-of-Words and document-averaged embeddings.

```powershell
uv run downstream_task.py
```

## Results Summary

| Representation  | arXiv Accuracy | AG News Accuracy |
| --------------- | -------------- | ---------------- |
| BoW (Baseline)  | 92.05%         | **89.37%**       |
| CBOW (avg)      | 92.20%         | 88.18%           |
| Skip-gram (avg) | **92.65%**     | 89.29%           |
| FastText (avg)  | 91.90%         | 87.28%           |
| GloVe (avg)     | 75.65%         | 76.28%           |

Detailed observations and analysis can be found in the `Lab_5_report.tex` (or the generated PDF).
