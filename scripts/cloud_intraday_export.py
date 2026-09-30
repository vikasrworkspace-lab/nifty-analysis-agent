"""Cloud-side intraday publish: freshness gate -> export -> upload.

This is the body of the ``nifty-intraday-export`` Cloud Run job. Ordering here
is the whole point of the file, so read it before changing anything:

1. ``scripts/update_data.py --fyers`` has already run, in the separate
   ``nifty-fyers-update`` job, and its result is what this job downloaded from
   the GCS cache. So the archives inspected below are the newest bars the
   collector actually managed to write -- not necessarily the current session.

2. Gate on that freshness *before* doing any work. The Fyers collector swallows
   network errors and its exit code has been observed to be 0 on a failed
   fetch, so neither the exit code nor a "success" line proves the data is
   current. The only trustworthy signal is the timestamp on the newest bar
   itself, which is what the gate reads.

   On a stale feed this returns non-zero *before* exporting and before
   uploading anything. That is deliberate: the last good JSON already in the
   bucket stays published and continues to serve, rather than being overwritten
   with bars that masquerade as live. A failed job execution is the intended
   visible signal, matching the local loop's ALERT behaviour. The cache sync
   wrapper must not be run with ``--up-on-failure`` for the same reason.

3. Only past the gate do we regenerate the JSON and publish it.

The gate itself is imported from ``scripts/auto_intraday_loop.py`` rather than
reimplemented, so the cloud path and the local path can never drift apart on
what counts as fresh.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# scripts/ is not a package (no __init__.py), and auto_intraday_loop is only
# ever imported here, so put the directory on the path rather than adding a
# package marker. Importing the module is side-effect free: its polling loop is
# guarded by __main__.
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from auto_intraday_loop import MAX_BAR_AGE_MIN, data_is_fresh  # noqa: E402


def run(cmd: list[str], desc: str) -> int:
    print(f"[intraday-export] {desc}...", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(
            f"[intraday-export] {desc} failed with exit code {result.returncode}",
            file=sys.stderr,
            flush=True,
        )
    return result.returncode


def main() -> int:
    fresh, reason = data_is_fresh()
    if not fresh:
        print(
            f"[intraday-export] ALERT: refusing to publish. Feed is stale: {reason}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[intraday-export] ALERT: last good intraday JSON remains published in "
            f"gs://{os.environ.get('GCS_CACHE_BUCKET', '?')}/dashboard/. "
            "No upload was attempted.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        f"[intraday-export] freshness gate passed ({reason}); "
        f"tolerance is {MAX_BAR_AGE_MIN} min",
        flush=True,
    )

    code = run(
        [sys.executable, str(ROOT / "scripts" / "export_intraday.py")],
        "Running intraday WFO export",
    )
    if code != 0:
        return code

    # --skip-export because export_intraday.py above already wrote the files;
    # re-running the daily exporter here would be wasted work and would race
    # the daily job for the same bucket prefix.
    return run(
        [
            sys.executable,
            str(ROOT / "scripts" / "cloud_export.py"),
            "--only",
            "intraday",
            "--skip-export",
        ],
        "Publishing intraday JSON to GCS",
    )


if __name__ == "__main__":
    sys.exit(main())
