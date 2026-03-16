# Data Collection

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

cd natural-language-processing/dataset-collection

uv sync

# activate virtual environment
./.venv/Scripts/activate   # Windows
source .venv/bin/activate  # macOS/Linux
```

## Commands to Run

- To collect the data and display metadata

```bash
uv run .\src\collect.py [opts]

usage: collect.py [-h] [--limit LIMIT] [--seed SEED] [--no-cache] [--format {parquet,csv,both}] [--out-dir OUT_DIR]

arXiv Dataset Collection Tool

options:
  -h, --help                    show this help message and exit
  --limit LIMIT                 Papers per class
  --seed SEED                   Random seed
  --no-cache                    Disable using cached contents
  --format {parquet,csv,both}   Storage format for local data
  --out-dir OUT_DIR             Directory to save the files
```

- To normalize the text

```bash
uv run .\src\preprocess.py --data .\data\raw\arxiv_5000_per_class.csv --output .\data\processed\arxiv_5000_cleaned.csv --normalize
```

- To run vocab/n-gram analysis

```bash
uv run .\src\run_analysis.py --data .\data\processed\arxiv_5000_cleaned.csv
```

Plots can be found in `plots/` directory.

- To split the data, run

```bash
uv run src/run_split.py --data data/processed/arxiv_5000_cleaned.csv --outdir data/splits/
```
