"""
restore.py
----------
Backup restore utility for the Smart Factory Predictive Maintenance system.

Features:
  - List available backup archives with timestamps and sizes
  - Restore from a named or most-recent backup
  - Dry-run mode to preview what will be restored
  - MD5 integrity check before restore
  - Prompts for confirmation before overwriting live data

Usage:
  python scripts/restore.py --list                           # list available backups
  python scripts/restore.py --latest                         # restore most recent
  python scripts/restore.py --file backup_pdm_20240101_120000.tar.gz
  python scripts/restore.py --latest --dry-run               # preview without restoring
"""

import argparse
import hashlib
import logging
import sys
import tarfile
from datetime import datetime
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR   = PROJECT_ROOT / "backups"
LOG_DIR      = PROJECT_ROOT / "logs"


# ── Logging ───────────────────────────────────────────────────
def _configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt    = "%(asctime)s | %(levelname)-8s | %(message)s"
    logger = logging.getLogger("pdm.restore")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    fh = logging.FileHandler(LOG_DIR / "restore.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


logger = _configure_logging()


# ── Helpers ───────────────────────────────────────────────────
def _list_backups() -> list[Path]:
    """Return all backup archives sorted newest-first."""
    if not BACKUP_DIR.exists():
        return []
    return sorted(
        BACKUP_DIR.glob("backup_pdm_*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _verify_checksum(archive_path: Path) -> bool:
    """Verify archive against its .md5 sidecar file (if present)."""
    checksum_file = archive_path.with_suffix(".tar.gz.md5")
    if not checksum_file.exists():
        logger.warning("No checksum file found for %s — skipping integrity check.", archive_path.name)
        return True     # Non-fatal: older backups may lack checksums

    expected_line  = checksum_file.read_text(encoding="utf-8").strip()
    expected_hash  = expected_line.split()[0]

    h = hashlib.md5()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual_hash = h.hexdigest()

    if actual_hash != expected_hash:
        logger.error(
            "CHECKSUM MISMATCH!\n  Expected : %s\n  Actual   : %s",
            expected_hash, actual_hash,
        )
        return False

    logger.info("Checksum verified OK: %s", actual_hash)
    return True


def _print_backup_list(backups: list[Path]) -> None:
    """Print a formatted table of available backups."""
    if not backups:
        print("\n  No backup archives found in:", BACKUP_DIR)
        print("  Run `python scripts/backup.py` to create the first backup.\n")
        return

    print()
    print("  ┌─────┬──────────────────────────────────────────────┬───────────┬─────────────────────┐")
    print("  │  #  │ Archive Name                                 │ Size (MB) │ Created             │")
    print("  ├─────┼──────────────────────────────────────────────┼───────────┼─────────────────────┤")
    for i, path in enumerate(backups, start=1):
        size_mb = round(path.stat().st_size / (1024 * 1024), 2)
        mtime   = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        marker  = "* " if i == 1 else "  "
        print(f"  │ {marker}{i:<2} │ {path.name:<44} │ {size_mb:>9.2f} │ {mtime} │")
    print("  └─────┴──────────────────────────────────────────────┴───────────┴─────────────────────┘")
    print("  (* = most recent)\n")


# ── Core Restore Logic ────────────────────────────────────────
def run_restore(archive_path: Path, dry_run: bool = False) -> int:
    """Restore project data from a backup archive.

    Args:
        archive_path: Path to the .tar.gz backup archive.
        dry_run: Preview contents without extracting.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    logger.info("=" * 60)
    logger.info("SMART FACTORY PDM — RESTORE UTILITY")
    logger.info("  Archive   : %s", archive_path.name)
    logger.info("  Target    : %s", PROJECT_ROOT)
    logger.info("  Mode      : %s", "DRY-RUN" if dry_run else "LIVE RESTORE")
    logger.info("=" * 60)

    if not archive_path.exists():
        logger.error("Archive not found: %s", archive_path)
        return 1

    # ── Integrity check ───────────────────────────────────────
    if not _verify_checksum(archive_path):
        logger.error("Aborting restore due to checksum failure.")
        return 1

    # ── Inspect archive contents ──────────────────────────────
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            logger.info("Archive contains %d items:", len(members))
            for m in members[:20]:    # Show first 20 entries
                logger.info("  %s", m.name)
            if len(members) > 20:
                logger.info("  ... and %d more.", len(members) - 20)

            if dry_run:
                logger.info("DRY-RUN complete — no files extracted.")
                return 0

            # ── Confirm before overwriting ────────────────────
            print()
            print("  ⚠️  WARNING: This will overwrite existing data in:")
            print(f"     {PROJECT_ROOT}")
            print()
            confirm = input("  Type 'yes' to confirm restore: ").strip().lower()
            if confirm != "yes":
                logger.info("Restore cancelled by user.")
                return 0

            # ── Extract ───────────────────────────────────────
            tar.extractall(path=PROJECT_ROOT)    # noqa: S202 — trusted local backups only

        logger.info("=" * 60)
        logger.info("RESTORE COMPLETE — %d items restored.", len(members))
        logger.info("  Restart all services for changes to take effect.")
        logger.info("    docker-compose restart")
        logger.info("=" * 60)
        return 0

    except (tarfile.TarError, OSError) as exc:
        logger.error("RESTORE FAILED: %s", exc, exc_info=True)
        return 1


# ── CLI ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Factory PDM — Backup Restore Utility"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="List all available backup archives.",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Restore from the most recent backup.",
    )
    group.add_argument(
        "--file",
        type=str,
        metavar="ARCHIVE_NAME",
        help="Restore from a specific archive filename (in backups/ directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview restore contents without extracting any files.",
    )
    args = parser.parse_args()

    backups = _list_backups()

    if args.list:
        _print_backup_list(backups)
        sys.exit(0)

    if args.latest:
        if not backups:
            logger.error("No backup archives found in %s", BACKUP_DIR)
            sys.exit(1)
        archive_path = backups[0]

    elif args.file:
        archive_path = BACKUP_DIR / args.file
        if not archive_path.exists():
            logger.error("Archive not found: %s", archive_path)
            sys.exit(1)

    sys.exit(run_restore(archive_path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
