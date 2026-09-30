"""
artifact_manager.py
-------------------
Task 7 – Phase 2: Artifact Management

Consolidates and validates all Phase 2 output artifacts.
Generates a final manifest listing every produced file.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ArtifactManager:
    """
    Validates that all expected Phase 2 artifacts exist and logs a manifest.

    Parameters
    ----------
    config : dict, optional
        Project config. Auto-loaded if not provided.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_config()
        self.reports_dir = Path(self.config["paths"]["reports_dir"])
        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.plots_dir = self.reports_dir / "plots"

    # ── Expected artifacts ─────────────────────────────────────────────────────

    def _expected_artifacts(self):
        return {
            "Models": [
                self.models_dir / "best_model.pkl",
                self.models_dir / "preprocessor.pkl",
                self.models_dir / "feature_names.pkl",
                self.models_dir / "label_encoder.pkl",
                self.models_dir / "best_model.joblib",
            ],
            "Reports": [
                self.reports_dir / "training_report.html",
                self.reports_dir / "hyperparameter_results.csv",
                self.reports_dir / "best_parameters.json",
            ],
            "Metrics": [
                self.metrics_dir / "evaluation_metrics.csv",
                self.metrics_dir / "classification_report.csv",
                self.metrics_dir / "feature_importance.csv",
                self.metrics_dir / "model_comparison.csv",
                self.metrics_dir / "best_model_meta.json",
            ],
            "Plots": [
                self.plots_dir / "roc_curves.png",
                self.plots_dir / "pr_curves.png",
                self.plots_dir / "confusion_matrices.png",
                self.plots_dir / "calibration_curves.png",
                self.plots_dir / "model_comparison_bar.png",
            ],
        }

    # ── Manifest ───────────────────────────────────────────────────────────────

    def validate_and_log(self) -> dict:
        """
        Check existence of all expected artifacts, log status, and return manifest.

        Returns
        -------
        dict
            Manifest with 'present' and 'missing' lists.
        """
        logger.info("=" * 60)
        logger.info("STEP 12: Artifact Validation")
        logger.info("=" * 60)

        present, missing = [], []
        expected = self._expected_artifacts()

        for category, paths in expected.items():
            logger.info("  [%s]", category)
            for path in paths:
                if path.exists():
                    size_kb = path.stat().st_size / 1024
                    logger.info("    ✔ %-55s (%.1f KB)", str(path.name), size_kb)
                    present.append(str(path))
                else:
                    logger.warning("    ✗ MISSING: %s", str(path))
                    missing.append(str(path))

        # Count extra plots generated
        extra_plots = list(self.plots_dir.glob("*.png")) if self.plots_dir.exists() else []
        logger.info(
            "  [Plots] %d total PNG files in %s", len(extra_plots), self.plots_dir
        )

        manifest = {
            "generated_at": datetime.now().isoformat(),
            "present": present,
            "missing": missing,
            "total_expected": len(present) + len(missing),
            "total_present": len(present),
            "total_plots": len(extra_plots),
        }

        manifest_path = self.reports_dir / "artifact_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info("  Artifact manifest → %s", manifest_path)

        if missing:
            logger.warning(
                "  ⚠ %d artifact(s) missing — check logs for errors in prior stages.",
                len(missing),
            )
        else:
            logger.info("  ✔ All expected artifacts are present.")

        return manifest
