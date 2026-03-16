# Lab 3: Probabilistic Models

Please go through the `results/` directory for looking at the results, and `logs/` as well for better clarity on LDA topics.

## Prerequisites

- Python 3.12+

We use uv for package and project management.

Install uv:

```bash
pip install uv
```

or follow official instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

## Setup

Run these commands in terminal:

```bash
git clone https://github.com/vrajpat3ll/natural-language-processing.git

cd natural-language-processing/probabilistic-modelling

uv sync

# activate virtual environment
./.venv/Scripts/activate   # Windows
source .venv/bin/activate  # macOS/Linux
```

## Commands to Run

```sh
# To prepare data
uv run prepare_data_cli.py
# To get results of the Naive Bayes classifier
uv run classifier_cli.py
# To run LDA for topic modelling
uv run lda_cli.py
# To run the HMM benchmarking
uv run hmm_cli.py

# use -h flag for help in corresponding part
```
