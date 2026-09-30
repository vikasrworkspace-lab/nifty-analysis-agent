"""Wrap a job with Cloud Storage cache sync, for the Cloud Run migration.

A Cloud Run job has an ephemeral filesystem, so this restores the cache before
the work and saves it afterwards::

    python scripts/cloud_job.py -- python scripts/update_data.py
    python scripts/cloud_job.py --only-down -- python scripts/update_data.py
    python scripts/cloud_job.py --roots historical_dir -- python scripts/update_data.py

The wrapped command is passed through untouched, so this composes with the
existing scripts rather than reimplementing them. With ``GCS_CACHE_BUCKET``
unset the wrapper is a transparent pass-through, which keeps local runs and the
GitHub Actions workflow behaving exactly as before.

Safety default: if the wrapped command fails, the cache is **not** uploaded. A
partially written cache must never replace a good one in the bucket. Pass
``--up-on-failure`` to override deliberately.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.settings as settings_mod
from core import gcs_cache


def _select_roots(keys: list[str] | None, settings: dict) -> list[Path]:
    if not keys:
        return gcs_cache.cache_roots(settings)
    roots = []
    for key in keys:
        try:
            roots.append(settings_mod.rel(settings, key))
        except KeyError:
            print(f"[cloud] unknown path key '{key}'; ignoring", file=sys.stderr)
    return roots


def _report(label: str, summary: dict) -> None:
    if not summary.get("enabled"):
        print(f"[cloud] {label}: cache sync disabled (GCS_CACHE_BUCKET unset); skipping")
        return
    for err in summary.get("errors", []):
        print(f"[cloud] {label} ERROR: {err}", file=sys.stderr)
    counts = {k: v for k, v in summary.items() if k not in ("enabled", "errors")}
    detail = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"[cloud] {label}: {detail}" + (f", {len(summary['errors'])} error(s)" if summary["errors"] else ""))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command with Cloud Storage cache sync around it.",
    )
    parser.add_argument("--roots", nargs="*", default=None,
                        help="Settings path keys to sync (default: all cache roots).")
    parser.add_argument("--only-down", action="store_true",
                        help="Restore the cache but do not save it afterwards.")
    parser.add_argument("--only-up", action="store_true",
                        help="Save the cache but do not restore it first.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace local files that already exist on download.")
    parser.add_argument("--up-on-failure", action="store_true",
                        help="Upload the cache even if the command fails "
                             "(off by default so a broken run cannot publish "
                             "a partial cache).")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command to run, after --")
    args = parser.parse_args()

    command = [c for c in args.command if c != "--"]
    if not command:
        parser.error("no command given; pass it after --")

    settings = settings_mod.load_settings()
    roots = _select_roots(args.roots, settings)

    if gcs_cache.is_enabled():
        print(f"[cloud] bucket={gcs_cache.bucket_name()} prefix={gcs_cache.prefix()}")
        print(f"[cloud] roots={[str(r) for r in roots]}")
    else:
        print("[cloud] GCS_CACHE_BUCKET not set; running without cache sync")

    if not args.only_up:
        _report("sync down", gcs_cache.sync_down(roots=roots, settings=settings,
                                                 overwrite=args.overwrite))

    result = subprocess.run(command)

    if not args.only_down:
        if result.returncode != 0 and not args.up_on_failure:
            print(f"[cloud] command exited {result.returncode}; skipping cache upload "
                  f"to avoid publishing a partial cache (override with --up-on-failure)")
        else:
            _report("sync up", gcs_cache.sync_up(roots=roots, settings=settings))

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
