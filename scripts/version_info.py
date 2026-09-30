"""
version_info.py
---------------
Version information utility for the Smart Factory Predictive
Maintenance platform.

Prints a version matrix covering all components:
  backend, frontend, API, ML model, and MLflow registry.

Usage:
  python scripts/version_info.py           # human-readable table
  python scripts/version_info.py --json    # machine-readable JSON
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_version_file() -> str:
    """Read the canonical version from the VERSION file."""
    version_file = PROJECT_ROOT / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()
    return "unknown"


def _detect_model_version() -> tuple[str, str]:
    """Detect the active model version from best_model_meta.json."""
    meta_path = PROJECT_ROOT / "reports" / "metrics" / "best_model_meta.json"
    if not meta_path.exists():
        return "unknown", "unknown"
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        algorithm = data.get("algorithm", data.get("model_name", "unknown"))
        version   = data.get("version", _read_version_file())
        return algorithm, version
    except (json.JSONDecodeError, OSError):
        return "unknown", "unknown"


def _detect_mlflow_model() -> str:
    """Detect the registered MLflow model version from local DB if available."""
    try:
        import mlflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("PdM_BestModel")
        if versions:
            v = versions[0]
            return f"v{v.version} ({v.current_stage})"
        return "no versions registered"
    except Exception:
        return "mlflow unavailable"


def _read_frontend_version() -> str:
    """Read the version from frontend/package.json."""
    pkg_path = PROJECT_ROOT / "frontend" / "package.json"
    if not pkg_path.exists():
        return "unknown"
    try:
        pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        return pkg.get("version", "unknown")
    except (json.JSONDecodeError, OSError):
        return "unknown"


def collect_version_info() -> dict:
    """Collect version information for all platform components."""
    project_version  = _read_version_file()
    algo, model_ver  = _detect_model_version()
    frontend_version = _read_frontend_version()

    return {
        "project":  project_version,
        "api":      f"v{project_version}",
        "backend":  f"v{project_version}",
        "frontend": f"v{frontend_version}",
        "ml_model": {
            "version":   f"v{model_ver}",
            "algorithm": algo,
            "artifact":  str(PROJECT_ROOT / "models" / "saved_models" / "best_model.joblib"),
        },
        "mlflow_registry": _detect_mlflow_model(),
        "python": sys.version.split()[0],
    }


def print_table(info: dict) -> None:
    """Print a human-readable version table."""
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║      Smart Factory PDM — Component Version Matrix           ║")
    print("  ╠══════════════════════════════════╦═══════════════════════════╣")
    print(f"  ║  {'Component':<32} ║ {'Version / Detail':<25} ║")
    print("  ╠══════════════════════════════════╬═══════════════════════════╣")

    rows = [
        ("Platform Version",      info["project"]),
        ("API Version",           info["api"]),
        ("Backend Service",       info["backend"]),
        ("Frontend Dashboard",    info["frontend"]),
        ("ML Model (Algorithm)",  f"{info['ml_model']['version']} ({info['ml_model']['algorithm']})"),
        ("MLflow Registry",       info["mlflow_registry"]),
        ("Python Runtime",        info["python"]),
    ]
    for label, value in rows:
        print(f"  ║  {label:<32} ║ {value:<25} ║")

    print("  ╚══════════════════════════════════╩═══════════════════════════╝")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Factory PDM — Version Information"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output version info as machine-readable JSON.",
    )
    args = parser.parse_args()

    info = collect_version_info()

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print_table(info)


if __name__ == "__main__":
    main()
