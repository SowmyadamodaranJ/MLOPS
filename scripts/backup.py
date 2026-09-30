"""
backup.py
---------
Production-grade automated backup utility for the Smart Factory
Predictive Maintenance system.

Features:
  - Timestamped tar.gz archives of all critical data
  - Configurable retention policy (delete old backups)
  - MD5 checksum verification of archives
  - Dry-run mode for safe testing
  - Structured logging to logs/backup.log
  - Exit codes for CI/cron integration

Usage:
  python scripts/backup.py                # run backup
  python scripts/backup.py --dry-run      # simulate without writing
  python scripts/backup.py --retention 14 # keep last 14 backups
"""

import argparse
import hashlib
import logging
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR   = PROJECT_ROOT / os.environ.get("BACKUP_DIR", "backups")
LOG_DIR      = PROJECT_ROOT / "logs"

# Items included in every backup archive
ITEMS_TO_BACKUP = [
    PROJECT_ROOT / "data" / "predictions.db",
    PROJECT_ROOT / "mlflow.db",
    PROJECT_ROOT / "models" / "saved_models",
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "mlruns",
]


# ── Logging setup ─────────────────────────────────────────────
def _configure_logging() -> logging.Logger:
    """Configure logger writing to both stdout and logs/backup.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "backup.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("pdm.backup")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    logger.addHandler(fh)

    return logger


logger = _configure_logging()


# ── MD5 Checksum ──────────────────────────────────────────────
def _md5_of_file(path: Path) -> str:
    """Compute MD5 hex digest of a file for integrity verification."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Retention Policy ──────────────────────────────────────────
def _apply_retention(retention_days: int, dry_run: bool) -> None:
    """Delete backup archives older than `retention_days` backups.

    retention_days is interpreted as 'keep the N most recent archives'.
    Sorted by creation time; oldest beyond the limit are removed.
    """
    archives = sorted(
        BACKUP_DIR.glob("backup_pdm_*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,      # Newest first
    )

    to_delete = archives[retention_days:]   # Everything beyond the limit
    if not to_delete:
        logger.info("Retention: no old archives to remove (limit=%d).", retention_days)
        return

    for old_archive in to_delete:
        age_days = (datetime.now().timestamp() - old_archive.stat().st_mtime) / 86400
        if dry_run:
            logger.info("[DRY-RUN] Would delete: %s (%.1f days old)", old_archive.name, age_days)
        else:
            old_archive.unlink()
            logger.info("Deleted old archive: %s (%.1f days old)", old_archive.name, age_days)


# ── Core Backup Logic ─────────────────────────────────────────
def run_backup(dry_run: bool = False, retention_days: int = 7) -> int:
    """Create a timestamped tar.gz backup archive.

    Args:
        dry_run: If True, simulate without creating any files.
        retention_days: Keep this many most-recent archives; delete older ones.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"backup_pdm_{timestamp}.tar.gz"
    archive_path = BACKUP_DIR / archive_name

    logger.info("=" * 60)
    logger.info("SMART FACTORY PDM — AUTOMATED BACKUP")
    logger.info("  Mode      : %s", "DRY-RUN" if dry_run else "LIVE")
    logger.info("  Destination: %s", archive_path)
    logger.info("  Retention : keep last %d archives", retention_days)
    logger.info("=" * 60)

    archived_count  = 0
    skipped_count   = 0
    total_size_bytes = 0

    if dry_run:
        # In dry-run mode, just report what would be included
        for item in ITEMS_TO_BACKUP:
            if item.exists():
                arcname = item.relative_to(PROJECT_ROOT)
                logger.info("[DRY-RUN] Would archive: %s", arcname)
                archived_count += 1
            else:
                logger.warning("[DRY-RUN] Not found (would skip): %s", item)
                skipped_count += 1
        logger.info("DRY-RUN complete — no files written.")
        return 0

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            for item in ITEMS_TO_BACKUP:
                if item.exists():
                    arcname = item.relative_to(PROJECT_ROOT)
                    tar.add(item, arcname=str(arcname))
                    logger.info("[OK] Added: %s", arcname)
                    archived_count += 1
                else:
                    logger.warning("[SKIP] Not found: %s", item)
                    skipped_count += 1

        # ── Integrity verification ────────────────────────────
        checksum = _md5_of_file(archive_path)
        size_mb  = round(archive_path.stat().st_size / (1024 * 1024), 2)

        # Write checksum sidecar file
        checksum_path = archive_path.with_suffix(".tar.gz.md5")
        checksum_path.write_text(f"{checksum}  {archive_name}\n", encoding="utf-8")

        logger.info("=" * 60)
        logger.info("BACKUP COMPLETE")
        logger.info("  Archive   : %s", archive_name)
        logger.info("  Size      : %.2f MB", size_mb)
        logger.info("  MD5       : %s", checksum)
        logger.info("  Items OK  : %d", archived_count)
        logger.info("  Skipped   : %d", skipped_count)
        logger.info("=" * 60)

        # ── Apply retention policy ────────────────────────────
        _apply_retention(retention_days, dry_run=False)

        return 0

    except Exception as exc:
        logger.error("BACKUP FAILED: %s", exc, exc_info=True)
        # Remove partial archive if it exists
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        return 1


# ── CLI Entry Point ───────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Factory PDM — Automated Backup Utility"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate backup without writing any files.",
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=int(os.environ.get("BACKUP_RETENTION_DAYS", "7")),
        help="Number of most-recent backup archives to retain (default: 7).",
    )
    args = parser.parse_args()
    sys.exit(run_backup(dry_run=args.dry_run, retention_days=args.retention))


if __name__ == "__main__":
    main()
