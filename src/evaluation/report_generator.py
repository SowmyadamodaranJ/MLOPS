"""
report_generator.py
-------------------
Task 6 – Phase 2: Training Report Generator

Generates a professional, self-contained HTML report embedding:
  - Dataset Summary
  - Feature Engineering Summary
  - Hyperparameter Configuration & Best Params
  - Model Comparison Table
  - Evaluation Metrics Table
  - All PNG plots (base64 embedded)
  - SHAP Summary (if available)
  - Best Model details
  - Local SHAP prediction explanation
  - Training recommendations

Saved to reports/training_report.html

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PLOTS_DIR = "plots"


def _b64_image(path: Path) -> str:
    """Read a PNG file and return a base64 data URI."""
    if not path.exists():
        return ""
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _df_to_html(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a styled HTML table."""
    def _fmt(val: Any) -> str:
        if isinstance(val, float):
            return float_fmt.format(val)
        return str(val)

    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_fmt(v)}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"


class ReportGenerator:
    """
    Generates a professional HTML training report.

    Parameters
    ----------
    config : dict, optional
        Project config. Auto-loaded if not provided.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_config()
        self.reports_dir = Path(self.config["paths"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.reports_dir / _PLOTS_DIR
        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])

    # ── CSS & HTML shell ───────────────────────────────────────────────────────

    @staticmethod
    def _css() -> str:
        return """
        <style>
          :root {
            --primary: #4361EE; --secondary: #7209B7; --accent: #F72585;
            --bg: #0f172a; --card: #1e293b; --text: #e2e8f0;
            --muted: #94a3b8; --border: #334155; --success: #22c55e;
            --warning: #f59e0b; --danger: #ef4444;
          }
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg); color: var(--text); line-height: 1.6;
          }
          .header {
            background: linear-gradient(135deg, #0f172a, #1a0533, #0f172a);
            border-bottom: 1px solid var(--border);
            padding: 3rem 4rem; text-align: center;
          }
          .header h1 {
            font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
            background: linear-gradient(90deg, #ffffff, #a3b8ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          }
          .header .subtitle {
            color: var(--muted); font-size: 1.05rem; margin-top: 0.5rem;
          }
          .header .badge {
            display: inline-block; margin-top: 1rem;
            background: var(--primary); color: white;
            padding: 0.3rem 1rem; border-radius: 999px; font-size: 0.85rem;
            font-weight: 600;
          }
          .container { max-width: 1400px; margin: 0 auto; padding: 2rem 3rem; }
          .section {
            background: var(--card); border: 1px solid var(--border);
            border-radius: 16px; padding: 2rem; margin-bottom: 2rem;
          }
          .section h2 {
            font-size: 1.4rem; font-weight: 700; color: #f1f5f9;
            margin-bottom: 1.2rem; padding-bottom: 0.6rem;
            border-bottom: 2px dashed var(--border);
          }
          .kpi-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 1rem; margin-bottom: 1.5rem;
          }
          .kpi-card {
            background: #0f172a; border: 1px solid var(--border);
            border-radius: 12px; padding: 1.2rem; text-align: center;
          }
          .kpi-card .value {
            font-size: 2rem; font-weight: 800; color: var(--primary);
          }
          .kpi-card .label { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }
          table {
            width: 100%; border-collapse: collapse;
            font-size: 0.88rem; margin-top: 0.5rem;
          }
          th {
            background: #0f172a; color: var(--primary);
            padding: 0.75rem 1rem; text-align: left;
            border-bottom: 2px solid var(--border); font-weight: 700;
          }
          td {
            padding: 0.65rem 1rem; border-bottom: 1px solid var(--border);
            color: var(--text);
          }
          tr:hover td { background: rgba(67,97,238,0.06); }
          .plot-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
            gap: 1.5rem; margin-top: 1rem;
          }
          .plot-card {
            background: #0f172a; border: 1px solid var(--border);
            border-radius: 12px; overflow: hidden;
          }
          .plot-card .plot-title {
            padding: 0.75rem 1rem; font-size: 0.9rem;
            font-weight: 600; color: var(--muted);
            border-bottom: 1px solid var(--border);
          }
          .plot-card img { width: 100%; display: block; }
          .best-badge {
            display: inline-block; background: var(--success); color: #fff;
            padding: 0.2rem 0.8rem; border-radius: 999px;
            font-size: 0.8rem; font-weight: 700; margin-left: 0.5rem;
          }
          .explanation-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
          }
          .explanation-item {
            background: #0f172a; border: 1px solid var(--border);
            border-radius: 8px; padding: 0.8rem 1rem;
          }
          .explanation-item .key { color: var(--muted); font-size: 0.8rem; }
          .explanation-item .val {
            font-size: 1.1rem; font-weight: 700; color: var(--text);
            margin-top: 0.2rem;
          }
          .rec-list { list-style: none; }
          .rec-list li {
            padding: 0.6rem 0; border-bottom: 1px solid var(--border);
            display: flex; align-items: flex-start; gap: 0.6rem; color: var(--muted);
          }
          .rec-list li::before { content: "→"; color: var(--primary); font-weight: 700; }
          .footer {
            text-align: center; padding: 2rem;
            color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--border);
          }
          code {
            background: #0f172a; padding: 0.15rem 0.4rem;
            border-radius: 4px; font-size: 0.85rem; color: var(--accent);
          }
          .tag {
            display: inline-block; background: rgba(67,97,238,0.15);
            color: var(--primary); padding: 0.2rem 0.6rem; border-radius: 6px;
            font-size: 0.78rem; font-weight: 600; margin: 0.2rem;
          }
        </style>
        """

    # ── Section builders ───────────────────────────────────────────────────────

    def _section_header(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        proj = self.config.get("project", {})
        return f"""
        <div class="header">
          <h1>🏭 Smart Factory Predictive Maintenance</h1>
          <p class="subtitle">Phase 2 — ML Pipeline Training Report</p>
          <span class="badge">Generated: {ts}</span>
          <span class="badge">v{proj.get('version', '2.0.0')}</span>
        </div>
        """

    def _section_dataset(self, dataset_info: Dict) -> str:
        kpis = ""
        for k, v in dataset_info.items():
            kpis += f"""
            <div class="kpi-card">
              <div class="value">{v}</div>
              <div class="label">{k}</div>
            </div>"""
        return f"""
        <div class="section">
          <h2>📊 Dataset Summary</h2>
          <div class="kpi-grid">{kpis}</div>
        </div>"""

    def _section_features(self, feature_names: List[str]) -> str:
        tags = "".join(f'<span class="tag">{f}</span>' for f in feature_names[:50])
        overflow = f"<p style='margin-top:0.8rem;color:var(--muted)'>…and {len(feature_names)-50} more</p>" if len(feature_names) > 50 else ""
        return f"""
        <div class="section">
          <h2>⚙️ Feature Engineering Summary</h2>
          <p style="color:var(--muted);margin-bottom:1rem">
            <strong style="color:var(--text)">{len(feature_names)}</strong> features engineered
            from telemetry, errors, maintenance, and machine metadata.
          </p>
          {tags}{overflow}
        </div>"""

    def _section_hyperparams(self, best_params: Dict, tuning_cfg: Dict) -> str:
        rows = ""
        for model, params in best_params.items():
            if not params:
                rows += f"<tr><td><strong>{model}</strong></td><td colspan='2' style='color:var(--muted)'>Baseline config (tuning disabled or no search space)</td></tr>"
            else:
                for k, v in params.items():
                    rows += f"<tr><td><strong>{model}</strong></td><td><code>{k}</code></td><td>{v}</td></tr>"

        cfg_str = (
            f"n_iter={tuning_cfg.get('n_iter', 10)} | "
            f"cv={tuning_cfg.get('cv', 3)} | "
            f"scoring=f1 | "
            f"StratifiedKFold | n_jobs=-1"
        )
        return f"""
        <div class="section">
          <h2>🔧 Hyperparameter Optimization</h2>
          <p style="color:var(--muted);margin-bottom:1rem">
            RandomizedSearchCV — <code>{cfg_str}</code>
          </p>
          <table>
            <thead><tr><th>Model</th><th>Parameter</th><th>Best Value</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    def _section_comparison(self, comparison_df: pd.DataFrame, best_name: str) -> str:
        def _row_html(row: pd.Series) -> str:
            is_best = str(row.get("model", "")) == best_name
            badge = '<span class="best-badge">BEST</span>' if is_best else ""
            cells = ""
            for col in comparison_df.columns:
                val = row[col]
                formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
                if col == "model":
                    cells += f"<td><strong>{formatted}</strong>{badge}</td>"
                else:
                    cells += f"<td>{formatted}</td>"
            return f"<tr>{cells}</tr>"

        header = "".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in comparison_df.columns)
        rows = "".join(_row_html(row) for _, row in comparison_df.iterrows())
        return f"""
        <div class="section">
          <h2>🏆 Model Comparison</h2>
          <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    def _section_plots(self, plots_dir: Path, plot_files: List[str], title: str) -> str:
        cards = ""
        for fname in plot_files:
            path = plots_dir / fname
            img_uri = _b64_image(path)
            if not img_uri:
                continue
            label = fname.replace(".png", "").replace("_", " ").title()
            cards += f"""
            <div class="plot-card">
              <div class="plot-title">{label}</div>
              <img src="{img_uri}" alt="{label}" loading="lazy"/>
            </div>"""
        return f"""
        <div class="section">
          <h2>{title}</h2>
          <div class="plot-grid">{cards}</div>
        </div>""" if cards else ""

    def _section_best_model(
        self,
        best_name: str,
        metrics_df: pd.DataFrame,
        best_params: Dict,
    ) -> str:
        row = metrics_df[metrics_df["model"] == best_name].iloc[0] if not metrics_df.empty else pd.Series()
        items = ""
        for col in metrics_df.columns:
            if col == "model":
                continue
            val = row.get(col, "N/A")
            formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
            items += f"""
            <div class="explanation-item">
              <div class="key">{col.replace('_', ' ').upper()}</div>
              <div class="val">{formatted}</div>
            </div>"""

        params_html = ""
        for k, v in best_params.get(best_name, {}).items():
            params_html += f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"

        return f"""
        <div class="section">
          <h2>⭐ Best Model — {best_name}</h2>
          <div class="explanation-grid">{items}</div>
          {f'<h3 style="margin-top:1.5rem;margin-bottom:0.8rem;color:var(--muted)">Best Hyperparameters</h3><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>{params_html}</tbody></table>' if params_html else ""}
        </div>"""

    def _section_local_explanation(self, explanations: Dict) -> str:
        if not explanations:
            return ""
        # Use the first available explanation
        model_name = next(iter(explanations))
        exp = explanations[model_name]
        if not exp:
            return ""

        pred_color = "#ef4444" if exp.get("prediction") == 1 else "#22c55e"
        pred_label = exp.get("prediction_label", "Unknown")

        top5_rows = ""
        for feat_info in exp.get("top_5_features", []):
            color = "#ef4444" if feat_info["shap_value"] > 0 else "#22c55e"
            top5_rows += (
                f"<tr><td>{feat_info['feature']}</td>"
                f"<td style='color:{color};font-weight:700'>{feat_info['shap_value']:+.6f}</td></tr>"
            )

        return f"""
        <div class="section">
          <h2>🔍 Local Prediction Explanation ({model_name})</h2>
          <div class="explanation-grid">
            <div class="explanation-item">
              <div class="key">MACHINE ID</div>
              <div class="val">{exp.get('machine_id', 'N/A')}</div>
            </div>
            <div class="explanation-item">
              <div class="key">PREDICTION</div>
              <div class="val" style="color:{pred_color}">{pred_label}</div>
            </div>
            <div class="explanation-item">
              <div class="key">FAILURE PROBABILITY</div>
              <div class="val">{exp.get('failure_probability', 0):.4f}</div>
            </div>
            <div class="explanation-item">
              <div class="key">CONFIDENCE SCORE</div>
              <div class="val">{exp.get('confidence_score', 0):.4f}</div>
            </div>
          </div>
          <h3 style="margin-top:1.5rem;margin-bottom:0.8rem;color:var(--muted)">Top 5 Contributing Features</h3>
          <table>
            <thead><tr><th>Feature</th><th>SHAP Contribution</th></tr></thead>
            <tbody>{top5_rows}</tbody>
          </table>
        </div>"""

    def _section_recommendations(self, best_name: str, metrics_df: pd.DataFrame) -> str:
        recs = [
            "Monitor data drift on sensor features (volt, rotate, pressure, vibration) in production.",
            f"Deploy <strong>{best_name}</strong> as the champion model via the model registry.",
            "Retrain the pipeline periodically as new telemetry data accumulates.",
            "Review SHAP dependence plots to identify sensor threshold thresholds for maintenance alerts.",
            "Enable StratifiedKFold hyperparameter tuning on scheduled retraining runs.",
            "Integrate prediction logging to track model performance over time.",
            "Consider calibrating probabilities with CalibratedClassifierCV for better uncertainty estimates.",
        ]
        if not metrics_df.empty and "balanced_accuracy" in metrics_df.columns:
            ba = metrics_df[metrics_df["model"] == best_name]["balanced_accuracy"].values
            if len(ba) > 0 and float(ba[0]) < 0.85:
                recs.append("Balanced accuracy is below 0.85 — consider SMOTE or class-weight tuning.")
        items = "".join(f"<li>{r}</li>" for r in recs)
        return f"""
        <div class="section">
          <h2>💡 Recommendations</h2>
          <ul class="rec-list">{items}</ul>
        </div>"""

    # ── Main entry point ───────────────────────────────────────────────────────

    def generate(
        self,
        dataset_info: Dict,
        feature_names: List[str],
        best_params: Dict,
        comparison_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        best_name: str,
        local_explanations: Optional[Dict] = None,
    ) -> Path:
        """
        Compose and write the full HTML training report.

        Returns
        -------
        Path
            Absolute path to the generated HTML file.
        """
        logger.info("=" * 60)
        logger.info("STEP 11: Generating HTML Training Report")
        logger.info("=" * 60)

        tuning_cfg = self.config.get("tuning", {})

        # Discover all plot files
        eval_plots = [
            "roc_curves.png", "pr_curves.png", "confusion_matrices.png",
            "calibration_curves.png", "model_comparison_bar.png",
        ]
        fi_plots = sorted(
            [p.name for p in self.plots_dir.glob("feature_importance_*.png")]
            if self.plots_dir.exists() else []
        )
        shap_plots = sorted(
            [p.name for p in self.plots_dir.glob("shap_*.png")]
            if self.plots_dir.exists() else []
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Factory PdM — Training Report</title>
  {self._css()}
</head>
<body>
  {self._section_header()}
  <div class="container">
    {self._section_dataset(dataset_info)}
    {self._section_features(feature_names)}
    {self._section_hyperparams(best_params, tuning_cfg)}
    {self._section_comparison(comparison_df, best_name)}
    {self._section_plots(self.plots_dir, eval_plots, "📈 Evaluation Plots")}
    {self._section_plots(self.plots_dir, fi_plots, "📊 Feature Importance Plots")}
    {self._section_plots(self.plots_dir, shap_plots, "🔮 SHAP Explainability Plots") if shap_plots else ""}
    {self._section_best_model(best_name, metrics_df, best_params)}
    {self._section_local_explanation(local_explanations or {})}
    {self._section_recommendations(best_name, metrics_df)}
  </div>
  <div class="footer">
    Smart Factory Predictive Maintenance — Phase 2 Training Report
    &nbsp;•&nbsp; Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    &nbsp;•&nbsp; Smart Factory PdM Team
  </div>
</body>
</html>"""

        report_path = self.reports_dir / "training_report.html"
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("  HTML Training Report saved → %s", report_path)
        return report_path
