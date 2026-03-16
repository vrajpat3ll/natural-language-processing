"""
Uploader for HuggingFace Hub with .env token management and dataset card generation.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from utils import get_logger

log = get_logger(__name__)


class DatasetUploader:
    """
    Handles dataset conversion and pushing to HuggingFace Hub with dynamic metadata.
    """

    def __init__(self, token: str | None = None, repo_id: str | None = None) -> None:
        load_dotenv()
        self.token = token or os.getenv("HF_TOKEN")
        self.repo_id = repo_id or os.getenv("HF_REPO_ID")

        self.card_template_path = Path(__file__).parents[1] / "DATASET_CARD.md"

        if not self.token:
            log.warning("HF_TOKEN not found in .env or environment.")
        if not self.repo_id:
            log.info("HF_REPO_ID not found in .env; will require --repo-id via CLI.")

    def generate_card(self, seed: int) -> str:
        """Read and format the dataset card template."""
        if not self.card_template_path.exists():
            log.warning(f"Dataset card template not found at {self.card_template_path}")
            return "# arXiv CS Abstracts\nDataset card template missing."

        template = self.card_template_path.read_text(encoding="utf-8")
        return template.format(
            seed=seed, collection_date=datetime.today().strftime("%Y-%m-%d")
        )

    def push(
        self,
        df: pd.DataFrame,
        repo_id: str | None = None,
        seed: int = 42,
        private: bool = False,
        commit_message: str = "Upload dataset via DatasetUploader",
    ) -> str:
        """
        Converts DataFrame to HF Dataset, generates card, and pushes to Hub.
        """
        target_repo = repo_id or self.repo_id
        if not target_repo:
            raise ValueError(
                "Repository ID is required. Provide it via CLI or HF_REPO_ID in .env"
            )

        if not self.token:
            raise ValueError("HF_TOKEN is required. Check your .env file.")

        try:
            from datasets import ClassLabel, Dataset, Features, Value
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "Run 'uv add datasets huggingface-hub' to use the uploader."
            )

        log.info(f"Preparing dataset for Hub: {target_repo}")

        # Use fixed label order for consistent indexing across all runs
        from src.scraping.scraper import DEFAULT_LABEL_GROUPS
        label_names = sorted(DEFAULT_LABEL_GROUPS.keys())
        label2id = {name: i for i, name in enumerate(label_names)}

        features = Features(
            {
                "arxiv_id": Value("string"),
                "title": Value("string"),
                "abstract": Value("string"),
                "primary_cat": Value("string"),
                "all_cats": Value("string"),
                "label": ClassLabel(names=label_names),
            }
        )

        cols = list(features.keys())
        df_hf = df[cols].copy()

        # Map string labels to integers for ClassLabel
        df_hf["label"] = df_hf["label"].map(label2id)
        dataset = Dataset.from_pandas(df_hf, features=features, preserve_index=False)

        # Push Dataset
        dataset.push_to_hub(
            repo_id=target_repo,
            token=self.token,
            private=private,
            commit_message=commit_message,
        )

        # Push Dataset Card (README.md in repo)
        card_content = self.generate_card(seed=seed)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=target_repo,
            repo_type="dataset",
            token=self.token,
            commit_message="Update README.md with dynamic metadata",
        )

        url = f"https://huggingface.co/datasets/{target_repo}"
        log.info(f"Dataset and card successfully pushed to: {url}")
        return url
