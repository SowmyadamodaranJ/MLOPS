"""
run_pipeline.py
---------------
Entry-point script to launch the Phase 1 ML pipeline.

Usage (from the project root smart_factory_pdm/):
    python run_pipeline.py

Ensure your virtual environment is activated and the five PdM CSV files
are placed under  data/raw/  before running.
"""

from src.pipeline import run_phase1_pipeline

if __name__ == "__main__":
    run_phase1_pipeline()
