import hashlib
import json
from pathlib import Path
import re

import arxiv
import pandas as pd
from tqdm import tqdm

from analyze import DatasetAnalyzer
from utils import get_logger

log = get_logger(__name__)

# Default groups for Task 1
DEFAULT_LABEL_GROUPS = {
    "NLP/AI": ["cs.CL", "cs.AI", "cs.LG", "cs.IR"],
    "Systems/Infra": ["cs.NI", "cs.DC", "cs.OS", "cs.SY"],
    "Vision": ["cs.CV", "cs.GR", "cs.MM"],
    "Algorithms": ["cs.DS", "cs.CC", "cs.GT", "cs.CG"],
}


class ArXivScraper:
    """
    Scraper for arXiv that focuses on process-level caching and easy-to-use API.

    Parameters
    ----------
    label_groups : dict[str, list[str]] | None
        Mapping of label name → list of arXiv category codes.
    papers_per_class : int
        Target number of papers per label.
    seed : int
        Random seed for class balancing.
    cache_dir : Path
        Directory for storing cached data.
    """

    def __init__(
        self,
        label_groups: dict[str, list[str]] | None = None,
        papers_per_class: int = 1000,
        seed: int = 42,
        cache_dir: Path = Path("data/raw"),
    ) -> None:
        self.label_groups = label_groups or DEFAULT_LABEL_GROUPS
        self.papers_per_class = papers_per_class
        self.seed = seed
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = arxiv.Client(
            page_size=500,
            delay_seconds=3.0,
            num_retries=3,
        )

        self._config_hash = self._generate_config_hash()
        self._df: pd.DataFrame | None = None

    def _generate_config_hash(self) -> str:
        """Generate a unique hash for the current configuration."""
        config = {
            "label_groups": self.label_groups,
            "papers_per_class": self.papers_per_class,
            "seed": self.seed,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, extension: str) -> Path:
        """Returns the cache path for a given extension based on config hash."""
        return self.cache_dir / f"arxiv_cache_{self._config_hash}.{extension}"

    def load_from_cache(self) -> bool:
        """Attempts to load data from existing cache files."""
        cache_path = self._get_cache_path("csv")
        if cache_path.exists():
            self._df = pd.read_csv(cache_path)
            log.info(f"Loaded from cache: {cache_path}")
            return True
        return False

    def collect(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Main entrypoint for collection. Uses cache if available and requested.
        """
        if use_cache and self.load_from_cache():
            return self._df  # type: ignore

        log.info("Starting fresh collection from arXiv API...")
        all_records = []

        for label, categories in self.label_groups.items():
            log.info(f"Fetching group: {label}")
            group_records = self._fetch_group(label, categories)
            log.info(f"Fetched group: {label}")
            all_records.extend(group_records)

        df = pd.DataFrame(all_records)
        df = self._post_process(df)
        self._df = df

        # Auto-save to cache after fresh collection
        self.save_csv(self._get_cache_path("csv"))
        return df

    def _fetch_group(self, label: str, categories: list[str]) -> list[dict]:
        """Fetch papers for a single group and its categories."""
        records = []
        seen_ids = set()

        # Fetch more than needed to account for:
        # 1. Duplicates within the group (across categories)
        # 2. Global deduplication (papers in multiple groups)
        target_with_buffer = int(self.papers_per_class * 1.2)

        for cat in categories:
            # We search for the full buffered target in each category.
            # This ensures that even if some categories have fewer papers or many
            # duplicates, we can fulfill the quota from other categories in the group.
            search = arxiv.Search(
                query=f"cat:{cat}",
                max_results=target_with_buffer,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            with tqdm(
                total=target_with_buffer, desc=f"  {cat}", leave=False, unit="paper"
            ) as pbar:
                for result in self.client.results(search):
                    if result.entry_id in seen_ids:
                        continue

                    match = re.match(
                        r"https?://arxiv.org/abs/([0-9\.]+)v?[0-9]*",
                        result.entry_id,
                    )
                    id = match.group(1) if match else result.entry_id.split("/")[-1]
                    records.append(
                        {
                            "arxiv_id": id,
                            "title": result.title.replace("\n", " ").strip(),
                            "abstract": result.summary.replace("\n", " ").strip(),
                            "primary_cat": result.primary_category,
                            "all_cats": " | ".join(result.categories),
                            "label": label,
                        }
                    )
                    seen_ids.add(id)
                    pbar.update(1)

                    if len(records) >= target_with_buffer:
                        break

            if len(records) >= target_with_buffer:
                break

        return records[:target_with_buffer]

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate and balance class sizes."""
        df = df.drop_duplicates(subset="arxiv_id")

        # Ensure we have exactly papers_per_class or less for each label.
        # We sample indices to avoid losing columns like 'label' due to groupby behaviors.
        sampled_indices = (
            df.groupby("label")["arxiv_id"]
            .apply(
                lambda x: x.sample(
                    n=min(len(x), self.papers_per_class), random_state=self.seed
                )
            )
            .index.get_level_values(1)
        )

        return df.loc[sampled_indices].reset_index(drop=True)

    def save_parquet(self, path: Path) -> None:
        """Save to Parquet (primary format)."""
        if self._df is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._df.to_parquet(path, index=False)
            log.info(f"Saved Parquet to {path}")

    def save_csv(self, path: Path) -> None:
        """Save to CSV (secondary format)."""
        if self._df is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._df.to_csv(path, index=False)
            log.info(f"Saved CSV to {path}")

    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("No data collected. Call .collect() first.")
        return self._df

    def analyze(
        self,
        print_global: bool = True,
        print_per_label: bool = True,
    ) -> DatasetAnalyzer:
        """
        Compute and (optionally) print corpus statistics for Lab 1 - Task 1.

        Returns the `DatasetAnalyzer` instance so callers can access
        `.global_stats()` and `.per_label_stats()` programmatically.

        Parameters
        ----------
        print_global : bool
            Print the assignment-style global summary table.
        print_per_label : bool
            Print the per-label breakdown table.

        Example
        -------
        >>> scraper = ArXivScraper()
        >>> df = scraper.collect()
        >>> s = scraper.analyze()
        >>> stats = s.global_stats()   # dict with all table values
        >>> df_label = s.per_label_stats()
        """
        analyzer = DatasetAnalyzer(self.data)

        if print_global:
            print("\nGlobal Dataset Summary (Lab 1 - Task 1)")
            print("=" * 53)
            log.info(analyzer.print())

        if print_per_label:
            log.info(analyzer.print_per_label())

        return analyzer
