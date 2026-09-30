"""
src/stages/featurize.py
-----------------------
DVC Stage 2: Feature Engineering & Preprocessing.
Consumes cleaned dataset and produces engineered time-series features,
train/test splits.

Outputs:
  - data/processed/features_engineered.csv
  - data/processed/train_data.csv
  - data/processed/test_data.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import engineer_features
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def featurize(force: bool = False):
    """Execute Stage 2: Feature Engineering."""
    config = load_config()
    processed_dir = Path(config["paths"]["processed_data_dir"])
    features_path = processed_dir / "features_engineered.csv"
    train_path = processed_dir / "train_data.csv"
    test_path = processed_dir / "test_data.csv"

    if not force and features_path.exists() and train_path.exists() and test_path.exists():
        logger.info(f"[featurize] Processed features already exist: {features_path.name}")
        logger.info("[featurize] Stage up to date. Use --force to recompute.")
        return

    interim_dir = Path(config["paths"]["interim_data_dir"])
    cleaned_path = interim_dir / "cleaned_dataset.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}. Run prepare stage first.")

    logger.info("[featurize] Reading cleaned interim dataset...")
    cleaned_df = pd.read_csv(cleaned_path)

    logger.info("[featurize] Running feature engineering pipeline...")
    engineer_features(cleaned_df, config=config)
    logger.info("[featurize] Stage 2 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Featurize PdM Data")
    parser.add_argument("--force", action="store_true", help="Force rerun even if outputs exist")
    args = parser.parse_args()
    featurize(force=args.force)
