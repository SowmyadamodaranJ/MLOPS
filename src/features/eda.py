"""
eda.py
------
Performs Exploratory Data Analysis (EDA) on the cleaned and
feature-engineered PdM DataFrames.

Charts generated
----------------
1.  Class distribution (failure_label)
2.  Sensor distributions (KDE + histogram)
3.  Sensor correlation heatmap
4.  Sensor time-series for a sample machine
5.  Error frequency bar chart
6.  Maintenance component frequency
7.  Failure rate by machine model
8.  Failure rate by machine age
9.  Rolling average sensors around failure events
10. Feature importance preview (variance-based)

All charts are saved as high-resolution PNG files to ``reports/figures/``.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config_loader import load_config
from src.utils.constants import CHART_PALETTE as PALETTE, FIG_DPI, SENSOR_COLS
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

# ── Styling ────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="husl", font="DejaVu Sans")


class EDAAnalyzer:
    """
    Runs a complete EDA suite and saves all charts to the figures directory.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Loaded automatically if not provided.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or load_config()
        self.figures_dir = Path(self.config["paths"]["figures_dir"])
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, filename: str) -> None:
        """Save a matplotlib figure to the figures directory."""
        path = self.figures_dir / filename
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Chart saved → %s", path)

    # ── Chart methods ─────────────────────────────────────────────────────────

    def plot_class_distribution(self, df: pd.DataFrame) -> None:
        """Bar chart of failure label distribution with percentage labels."""
        logger.info("  Plotting class distribution …")
        counts = df["failure_label"].value_counts().sort_index()
        labels = ["No Failure (0)", "Failure (1)"]
        colors = [PALETTE["no_failure"], PALETTE["failure"]]
        total = counts.sum()

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.45)
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{count:,}\n({count / total * 100:.2f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax.set_title("Target Class Distribution (failure_label)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        self._save(fig, "01_class_distribution.png")

    def plot_sensor_distributions(self, df: pd.DataFrame) -> None:
        """KDE + histogram for each sensor, coloured by failure label."""
        logger.info("  Plotting sensor distributions …")
        present = [c for c in SENSOR_COLS if c in df.columns]
        n = len(present)
        if n == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, sensor in enumerate(present):
            ax = axes[i]
            for label_val, color, name in zip(
                [0, 1],
                [PALETTE["no_failure"], PALETTE["failure"]],
                ["No Failure", "Failure"],
            ):
                subset = df.loc[df["failure_label"] == label_val, sensor].dropna()
                ax.hist(
                    subset, bins=80, density=True, alpha=0.45,
                    color=color, label=name, edgecolor="none"
                )
                
                # Sample subset for KDE estimation to keep it extremely fast
                sample_size = 10000
                if len(subset) > sample_size:
                    subset_sample = subset.sample(sample_size, random_state=42)
                else:
                    subset_sample = subset
                subset_sample.plot.kde(ax=ax, color=color, linewidth=2)
                
            ax.set_title(f"Distribution of '{sensor}'", fontsize=12, fontweight="bold")
            ax.set_xlabel(sensor)
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)

        # Hide extra subplots if < 4 sensors
        for j in range(n, 4):
            axes[j].set_visible(False)

        fig.suptitle("Sensor Value Distributions by Failure Label", fontsize=15, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "02_sensor_distributions.png")

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Correlation heatmap for numeric features."""
        logger.info("  Plotting correlation heatmap …")
        numeric_df = df.select_dtypes(include=[np.number]).drop(
            columns=["machineID"], errors="ignore"
        )
        # Keep top 30 columns by variance to keep heatmap readable
        top_cols = (
            numeric_df.var().nlargest(30).index.tolist()
        )
        corr = numeric_df[top_cols].corr()

        fig, ax = plt.subplots(figsize=(18, 14))
        sns.heatmap(
            corr,
            ax=ax,
            annot=False,
            cmap="RdYlGn",
            center=0,
            linewidths=0.3,
            square=False,
        )
        ax.set_title("Feature Correlation Heatmap (Top 30 by Variance)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "03_correlation_heatmap.png")

    def plot_sensor_timeseries(self, df: pd.DataFrame, machine_id: int = 1) -> None:
        """Time-series of sensor readings for a single machine with failure markers."""
        logger.info("  Plotting sensor time-series for machineID=%d …", machine_id)
        mdf = df[df["machineID"] == machine_id].copy()
        if mdf.empty:
            logger.warning("  machineID=%d not found; skipping time-series plot.", machine_id)
            return

        mdf = mdf.sort_values("datetime")
        present = [c for c in SENSOR_COLS if c in mdf.columns]

        fig, axes = plt.subplots(len(present), 1, figsize=(16, 3 * len(present)), sharex=True)
        if len(present) == 1:
            axes = [axes]

        failure_times = mdf.loc[mdf["failure_label"] == 1, "datetime"]

        for ax, sensor in zip(axes, present):
            ax.plot(mdf["datetime"], mdf[sensor], linewidth=0.8, color="#1976D2")
            for ft in failure_times:
                ax.axvline(ft, color=PALETTE["failure"], alpha=0.4, linewidth=0.8)
            ax.set_ylabel(sensor, fontsize=10)
            ax.set_title(f"Machine {machine_id} – {sensor}", fontsize=10)

        axes[-1].set_xlabel("Datetime")
        fig.suptitle(
            f"Sensor Time-Series for Machine ID {machine_id}\n(Red = Failure Window)",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        self._save(fig, f"04_sensor_timeseries_machine{machine_id}.png")

    def plot_error_frequency(self, df: pd.DataFrame) -> None:
        """Bar chart of total errors per error type."""
        logger.info("  Plotting error frequency …")
        error_cols = [c for c in df.columns if c.startswith("error") and "_rolling" not in c
                      and "_lag" not in c and c != "total_errors"]
        if not error_cols:
            logger.warning("  No error columns found; skipping error frequency plot.")
            return

        error_totals = df[error_cols].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            error_totals.index, error_totals.values,
            color=sns.color_palette("husl", len(error_totals)), edgecolor="white"
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + error_totals.max() * 0.01,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_title("Total Error Counts by Error Type", fontsize=14, fontweight="bold")
        ax.set_xlabel("Error Type")
        ax.set_ylabel("Total Count")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        self._save(fig, "05_error_frequency.png")

    def plot_maintenance_frequency(self, df: pd.DataFrame) -> None:
        """Bar chart of total maintenance events per component."""
        logger.info("  Plotting maintenance frequency …")
        maint_cols = [c for c in df.columns if c.startswith("maint_")]
        if not maint_cols:
            logger.warning("  No maintenance columns found; skipping maint frequency plot.")
            return

        maint_totals = df[maint_cols].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            maint_totals.index, maint_totals.values,
            color=sns.color_palette("coolwarm", len(maint_totals)), edgecolor="white"
        )
        ax.set_title("Maintenance Events by Component", fontsize=14, fontweight="bold")
        ax.set_xlabel("Component")
        ax.set_ylabel("Total Events")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        self._save(fig, "06_maintenance_frequency.png")

    def plot_failure_rate_by_model(self, df: pd.DataFrame) -> None:
        """Failure rate by machine model."""
        logger.info("  Plotting failure rate by machine model …")
        model_cols = [c for c in df.columns if c.startswith("model_")]
        if not model_cols:
            logger.warning("  No model columns found; skipping model failure-rate plot.")
            return

        # Reconstruct model from one-hot dummies (simplified)
        fig, ax = plt.subplots(figsize=(10, 5))
        rates = []
        for col in model_cols:
            subset = df[df[col] == 1]
            rate = subset["failure_label"].mean() if len(subset) > 0 else 0.0
            rates.append((col.replace("model_", ""), rate))

        # Add baseline (all rows where no model_ col is 1 → model0)
        base = df[df[model_cols].sum(axis=1) == 0]
        if len(base) > 0:
            rates.insert(0, ("model_base", base["failure_label"].mean()))

        labels, values = zip(*rates) if rates else ([], [])
        bars = ax.bar(labels, values, color=sns.color_palette("husl", len(labels)), edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{bar.get_height():.3%}",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_title("Failure Rate by Machine Model", fontsize=14, fontweight="bold")
        ax.set_xlabel("Machine Model")
        ax.set_ylabel("Failure Rate")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2%}"))
        fig.tight_layout()
        self._save(fig, "07_failure_rate_by_model.png")

    def plot_failure_rate_by_age(self, df: pd.DataFrame) -> None:
        """Failure rate binned by machine age."""
        logger.info("  Plotting failure rate by machine age …")
        if "age" not in df.columns:
            logger.warning("  'age' column not found; skipping age plot.")
            return

        df_age = df.copy()
        df_age["age_bin"] = pd.cut(df_age["age"], bins=6)
        agg = df_age.groupby("age_bin")["failure_label"].mean().reset_index()
        agg.columns = ["age_bin", "failure_rate"]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            agg["age_bin"].astype(str), agg["failure_rate"],
            color=sns.color_palette("Reds_r", len(agg)), edgecolor="white"
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{bar.get_height():.3%}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_title("Failure Rate by Machine Age Group", fontsize=14, fontweight="bold")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Failure Rate")
        plt.xticks(rotation=30)
        fig.tight_layout()
        self._save(fig, "08_failure_rate_by_age.png")

    def plot_sensor_boxplot(self, df: pd.DataFrame) -> None:
        """Boxplot of sensor readings for failure vs no-failure."""
        logger.info("  Plotting sensor boxplots by label …")
        present = [c for c in SENSOR_COLS if c in df.columns]
        if not present:
            return

        # Stratified sampling to make boxplot rendering extremely fast
        failures = df[df["failure_label"] == 1]
        non_failures = df[df["failure_label"] == 0]
        max_non_failures = 100000
        if len(non_failures) > max_non_failures:
            non_failures = non_failures.sample(max_non_failures, random_state=42)
        sample_df = pd.concat([failures, non_failures], axis=0)

        melt_df = sample_df[present + ["failure_label"]].melt(
            id_vars="failure_label", var_name="Sensor", value_name="Value"
        )
        melt_df["Class"] = melt_df["failure_label"].map(
            {0: "No Failure", 1: "Failure"}
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=melt_df, x="Sensor", y="Value", hue="Class",
            palette={"No Failure": PALETTE["no_failure"], "Failure": PALETTE["failure"]},
            ax=ax, flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        ax.set_title("Sensor Readings: Failure vs No-Failure", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sensor")
        ax.set_ylabel("Reading Value")
        fig.tight_layout()
        self._save(fig, "09_sensor_boxplot_by_label.png")

    def plot_monthly_failure_trend(self, df: pd.DataFrame) -> None:
        """Line chart showing monthly failure count over time."""
        logger.info("  Plotting monthly failure trend …")
        if "datetime" not in df.columns:
            logger.warning("  'datetime' column not found; skipping trend plot.")
            return

        df_trend = df.copy()
        df_trend["datetime"] = pd.to_datetime(df_trend["datetime"])
        df_trend["month"] = df_trend["datetime"].dt.to_period("M")
        monthly = df_trend.groupby("month")["failure_label"].sum().reset_index()
        monthly["month_str"] = monthly["month"].astype(str)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(
            monthly["month_str"], monthly["failure_label"],
            alpha=0.3, color=PALETTE["failure"]
        )
        ax.plot(
            monthly["month_str"], monthly["failure_label"],
            color=PALETTE["failure"], linewidth=2, marker="o", markersize=5
        )
        ax.set_title("Monthly Failure Count Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Failure Counts")
        plt.xticks(rotation=45)
        fig.tight_layout()
        self._save(fig, "10_monthly_failure_trend.png")

    # ── Main runner ───────────────────────────────────────────────────────────

    def run(
        self,
        cleaned_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> None:
        """
        Execute all EDA charts.

        Parameters
        ----------
        cleaned_df : pd.DataFrame
            Output of DataCleaner (used for time-series and raw sensor charts).
        features_df : pd.DataFrame
            Output of FeatureEngineer (used for correlation heatmap).
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Exploratory Data Analysis (EDA)")
        logger.info("=" * 60)

        self.plot_class_distribution(features_df)
        self.plot_sensor_distributions(cleaned_df)
        self.plot_correlation_heatmap(features_df)
        self.plot_sensor_timeseries(cleaned_df, machine_id=1)
        self.plot_error_frequency(features_df)
        self.plot_maintenance_frequency(features_df)
        self.plot_failure_rate_by_model(features_df)
        self.plot_failure_rate_by_age(features_df)
        self.plot_sensor_boxplot(cleaned_df)
        self.plot_monthly_failure_trend(cleaned_df)

        logger.info("EDA complete. All charts saved to '%s'.\n", self.figures_dir)


def run_eda(
    cleaned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    config: dict = None,
) -> None:
    """
    Convenience wrapper around EDAAnalyzer.run().

    Parameters
    ----------
    cleaned_df : pd.DataFrame
    features_df : pd.DataFrame
    config : dict, optional
    """
    analyzer = EDAAnalyzer(config=config)
    analyzer.run(cleaned_df, features_df)
