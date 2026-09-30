"""
src/stages/prepare.py
---------------------
DVC Stage 1: Data Preparation & Ingestion.
Loads raw telemetry, errors, failures, maintenance, and machine CSVs,
merges them into an interim dataset, and applies cleaning.

Outputs:
  - data/interim/merged_dataset.csv
  - data/interim/cleaned_dataset.csv
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_datasets
from src.data.data_merger import merge_datasets
from src.data.data_cleaner import clean_data
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_data(force: bool = False):
    """Execute Stage 1: Ingest, Merge, Clean."""
    config = load_config()
    interim_dir = Path(config["paths"]["interim_data_dir"])
    merged_path = interim_dir / "merged_dataset.csv"
    cleaned_path = interim_dir / "cleaned_dataset.csv"

    if not force and merged_path.exists() and cleaned_path.exists():
        logger.info(f"[prepare] Target files already exist: {merged_path.name}, {cleaned_path.name}")
        logger.info("[prepare] Stage up to date. Use --force to reprocess.")
        return

    logger.info("[prepare] Loading raw datasets...")
    telemetry, errors, failures, maint, machines = load_datasets(config)

    logger.info("[prepare] Merging datasets...")
    merged_df = merge_datasets(telemetry, errors, failures, maint, machines, config=config)

    logger.info("[prepare] Cleaning datasets...")
    clean_data(merged_df, config=config)
    logger.info("[prepare] Stage 1 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Prepare PdM Data")
    parser.add_argument("--force", action="store_true", help="Force rerun even if outputs exist")
    args = parser.parse_args()
    prepare_data(force=args.force)
