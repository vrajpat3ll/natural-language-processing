# Natural Language Processing

These are my solutions for the lab assignments of CS-5-45(MO), i.e., Natural Language Processing taught in semester 02, 2025-26.

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

cd natural-language-processing

uv venv

# activate virtual environment
./.venv/Scripts/activate   # Windows
source .venv/bin/activate  # macOS/Linux
```

Usually each step requires completely different dependencies, so each of them separately has a `uv.lock` file containing all the deps. Run `uv sync` when in a particular directory to install the relevant ones.

Go through the corresponding `README.md` files in each step to get a better understanding.
