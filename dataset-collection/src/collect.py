"""
CLI for Task 1: Dataset Collection from arXiv.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from scraping.scraper import ArXivScraper
from scraping.uploader import DatasetUploader
from utils import get_logger

log = get_logger(__name__)


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments and environment before starting computation."""

    if args.limit <= 0:
        raise ValueError(f"Limit must be a positive integer, got {args.limit}")

    if args.push:
        load_dotenv()
        repo_id = args.repo_id or os.getenv("HF_REPO_ID")
        if not repo_id:
            raise ValueError(
                "--repo-id is required when --push is enabled, "
                "or set HF_REPO_ID in your .env file."
            )

        if not os.getenv("HF_TOKEN"):
            raise ValueError(
                "HF_TOKEN not found in .env or environment. "
                "Pushing to HuggingFace Hub will fail."
            )

    if args.out_dir.exists() and not args.out_dir.is_dir():
        raise ValueError(f"Output path {args.out_dir} exists but is not a directory.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="arXiv Dataset Collection Tool")

    # Collection args
    parser.add_argument("--limit", type=int, default=1000, help="Papers per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable process-level caching",
    )
    parser.set_defaults(use_cache=True)

    # Storage args
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv", "both"],
        default="csv",
        help="Storage format for local data",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save the files",
    )

    # HF Hub args
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, help="HuggingFace repository ID")
    parser.add_argument("--private", action="store_true", help="Make Hub repo private")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        log.error(f"Validation Error: {e}")
        return

    scraper = ArXivScraper(
        papers_per_class=args.limit,
        seed=args.seed,
        cache_dir=args.out_dir,
    )

    df = scraper.collect(use_cache=args.use_cache)

    base_name = f"arxiv_{args.limit}_per_class"

    if args.format in ["parquet", "both"]:
        scraper.save_parquet(args.out_dir / f"{base_name}.parquet")

    if args.format in ["csv", "both"]:
        scraper.save_csv(args.out_dir / f"{base_name}.csv")

    scraper.analyze()

    if args.push:
        uploader = DatasetUploader()
        try:
            uploader.push(
                df,
                repo_id=args.repo_id,
                seed=args.seed,
                private=args.private,
            )
        except Exception as e:
            log.error(f"Failed to push to Hub: {e}")


if __name__ == "__main__":
    main()
