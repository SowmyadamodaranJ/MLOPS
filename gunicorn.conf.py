# ============================================================
# gunicorn.conf.py — Production Gunicorn Configuration
# Smart Factory Predictive Maintenance API
# ============================================================
#
# This file is loaded automatically when Gunicorn is started
# with: gunicorn -c gunicorn.conf.py backend.app:create_app()
#
# Environment variable overrides (set in .env or docker-compose):
#   WEB_CONCURRENCY   — number of worker processes (default: 2×CPU+1)
#   GUNICORN_TIMEOUT  — worker silent timeout in seconds (default: 120)
#   GUNICORN_THREADS  — threads per worker (default: 2)
#   FLASK_PORT        — port to bind on (default: 5000)
# ============================================================

import multiprocessing
import os

# ── Binding ──────────────────────────────────────────────────
_port = os.environ.get("FLASK_PORT", "5000")
bind = f"0.0.0.0:{_port}"

# ── Worker Processes ──────────────────────────────────────────
# WEB_CONCURRENCY env var lets ops teams tune without rebuilding.
# Default formula: (2 × vCPU cores) + 1  — a well-known Gunicorn guideline.
# Keep worker class as 'sync': joblib/numpy/sklearn are not async-safe.
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
threads = int(os.environ.get("GUNICORN_THREADS", "2"))

# ── Timeouts ─────────────────────────────────────────────────
# 120s to accommodate SHAP explainability computations which can be slow
# on large datasets.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30          # How long to wait for existing requests during shutdown
keepalive = 5                  # Seconds to hold idle HTTP keep-alive connections

# ── Request Limits ────────────────────────────────────────────
# Recycle workers after N requests to prevent memory bloat from ML models.
# Jitter adds randomness so all workers don't restart simultaneously.
max_requests = 1000
max_requests_jitter = 100

# ── Memory Optimisation ───────────────────────────────────────
# preload_app loads the Flask application in the master process before
# forking workers, enabling copy-on-write memory sharing for the large
# ML model objects loaded at startup.
preload_app = True

# ── Logging ──────────────────────────────────────────────────
# Log to stdout/stderr so Docker captures everything via the logging driver.
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
accesslog = "-"      # "-" = stdout
errorlog  = "-"      # "-" = stderr
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s '
    '"%(f)s" "%(a)s" %(L)ss'
)

# ── Process Naming ────────────────────────────────────────────
proc_name = "smart-factory-pdm-backend"

# ── Server Hooks ─────────────────────────────────────────────

def on_starting(server):
    """Called just before the master process is initialised."""
    server.log.info("=" * 60)
    server.log.info("Smart Factory PDM — Gunicorn Production Server Starting")
    server.log.info(f"  Workers  : {workers}")
    server.log.info(f"  Threads  : {threads}")
    server.log.info(f"  Timeout  : {timeout}s")
    server.log.info(f"  Binding  : {bind}")
    server.log.info(f"  Log Level: {loglevel}")
    server.log.info("=" * 60)


def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    worker.log.info(f"Worker {worker.pid} received interrupt signal.")


def worker_exit(server, worker):
    """Called when a worker exits after handling its final request."""
    server.log.info(f"Worker {worker.pid} exited gracefully.")


def on_exit(server):
    """Called just before the master process exits."""
    server.log.info("Smart Factory PDM — Gunicorn Server shut down cleanly.")
