"""
hyperparameter_tuner.py
-----------------------
Task 1 – Phase 2: Hyperparameter Optimization

Performs RandomizedSearchCV with TimeSeriesSplit for:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

TimeSeriesSplit is used instead of StratifiedKFold to respect the
temporal ordering of the PdM dataset.  Each fold's validation set is
always in the future relative to its training set.

Outputs
-------
  reports/hyperparameter_results.csv  — per-fold CV results
  reports/best_parameters.json        — best params for each model

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Default search spaces ──────────────────────────────────────────────────────
_DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__max_iter": [500, 1000, 2000],
        "clf__solver": ["lbfgs", "saga"],
    },
    "Decision Tree": {
        "clf__max_depth": [5, 8, 10, 15, 20, None],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "clf__n_estimators": [50, 100, 200, 300],
        # max_depth: None removed — unconstrained depth causes 61MB+ models
        "clf__max_depth": [5, 10, 15, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    },
    "XGBoost": {
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__max_depth": [3, 5, 7, 9],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__gamma": [0, 0.1, 0.2],
    },
}

# Mapping from config metric name → sklearn scorer string
_SCORER_MAP: Dict[str, str] = {
    "f1_score": "f1",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


class HyperparameterTuner:
    """
    Runs RandomizedSearchCV with StratifiedKFold for each model pipeline.

    Parameters
    ----------
    config : dict, optional
        Project config dict. Auto-loaded from YAML if not provided.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_config()
        tuning_cfg = self.config.get("tuning", {})
        eval_cfg = self.config.get("evaluation", {})

        self.enabled: bool = tuning_cfg.get("enabled", True)
        self.n_iter: int = int(tuning_cfg.get("n_iter", 10))
        self.cv: int = int(tuning_cfg.get("cv", 5))    # default raised to 5
        self.random_state: int = int(tuning_cfg.get("random_state", 42))
        self.use_timeseries_split: bool = tuning_cfg.get("use_timeseries_split", True)

        raw_metric = eval_cfg.get("primary_metric", "f1_score")
        self.scoring: str = _SCORER_MAP.get(raw_metric, "f1")

        self.reports_dir = Path(self.config["paths"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Results accumulation
        self._cv_results: list = []
        self._best_params: Dict[str, dict] = {}
        self._training_times: Dict[str, float] = {}
        self._best_scores: Dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def tune(
        self,
        model_name: str,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Pipeline, dict, float]:
        """
        Tune a single pipeline using RandomizedSearchCV.

        Parameters
        ----------
        model_name : str
            Human-readable model name (used for logging and output keys).
        pipeline : Pipeline
            Unfitted sklearn Pipeline (preprocessor + estimator).
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        search_space : dict, optional
            Override the default search space for this model.

        Returns
        -------
        tuple
            (best_pipeline, best_params_dict, training_time_seconds)
        """
        if not self.enabled:
            logger.info("  [Tuner] Disabled — fitting %s with baseline config.", model_name)
            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0
            self._training_times[model_name] = elapsed
            self._best_params[model_name] = {}
            self._best_scores[model_name] = 0.0
            return pipeline, {}, elapsed

        params = search_space or _DEFAULT_SEARCH_SPACES.get(model_name)
        if not params:
            logger.info("  [Tuner] No search space for %s — fitting baseline.", model_name)
            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0
            self._training_times[model_name] = elapsed
            self._best_params[model_name] = {}
            self._best_scores[model_name] = 0.0
            return pipeline, {}, elapsed

        # ── Subsample large datasets to avoid memory issues ──────────
        _MAX_TUNING_ROWS = 50_000
        if len(X_train) > _MAX_TUNING_ROWS:
            logger.info(
                "  [Tuner] Dataset has %d rows — subsampling to %d for CV search.",
                len(X_train), _MAX_TUNING_ROWS,
            )
            X_tune, y_tune = resample(
                X_train, y_train,
                n_samples=_MAX_TUNING_ROWS,
                stratify=y_train,
                random_state=self.random_state,
            )
        else:
            X_tune, y_tune = X_train, y_train

        logger.info(
            "  [Tuner] RandomizedSearchCV: %s | n_iter=%d | cv=%d | scoring=%s | rows=%d",
            model_name, self.n_iter, self.cv, self.scoring, len(X_tune),
        )

        # TimeSeriesSplit preserves temporal order across folds.
        # Each fold's validation window is strictly AFTER its training window.
        if self.use_timeseries_split:
            cv_strategy = TimeSeriesSplit(n_splits=self.cv)
            logger.info(
                "  [Tuner] Using TimeSeriesSplit (n_splits=%d) — temporal CV",
                self.cv,
            )
        else:
            from sklearn.model_selection import StratifiedKFold as SKF
            cv_strategy = SKF(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            logger.info(
                "  [Tuner] Using StratifiedKFold (n_splits=%d)", self.cv
            )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=self.n_iter,
            cv=cv_strategy,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=1,
            refit=False,
            error_score=0.0,
            return_train_score=True,
        )

        t0 = time.perf_counter()
        search.fit(X_tune, y_tune)

        best_params = search.best_params_
        best_score = search.best_score_
        cv_elapsed = time.perf_counter() - t0

        logger.info(
            "  [Tuner] ✔ %s CV done in %.1fs — best %s=%.4f",
            model_name, cv_elapsed, self.scoring, best_score,
        )
        logger.info("  [Tuner]   Best params: %s", best_params)

        # Refit with best params on full training data
        logger.info("  [Tuner] Refitting %s on full training data (%d rows)…", model_name, len(X_train))
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        # Accumulate CV rows
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.insert(0, "model", model_name)
        self._cv_results.append(cv_df)

        self._best_params[model_name] = best_params
        self._training_times[model_name] = elapsed
        self._best_scores[model_name] = best_score

        return pipeline, best_params, elapsed

    def save_results(self) -> None:
        """Persist hyperparameter_results.csv and best_parameters.json."""
        if self._cv_results:
            combined = pd.concat(self._cv_results, ignore_index=True)
            results_path = self.reports_dir / "hyperparameter_results.csv"
            combined.to_csv(results_path, index=False)
            logger.info("  [Tuner] CV results saved → %s", results_path)

        params_path = self.reports_dir / "best_parameters.json"
        with open(params_path, "w", encoding="utf-8") as fh:
            json.dump(self._best_params, fh, indent=2, default=str)
        logger.info("  [Tuner] Best parameters saved → %s", params_path)

    @property
    def training_times(self) -> Dict[str, float]:
        """Return per-model training time in seconds."""
        return dict(self._training_times)

    @property
    def best_params(self) -> Dict[str, dict]:
        """Return best hyperparameters per model."""
        return dict(self._best_params)

    @property
    def best_scores(self) -> Dict[str, float]:
        """Return best CV scores per model."""
        return dict(self._best_scores)
