"""Publish dashboard JSON artifacts to the GCS bucket the data service serves.

Two output groups exist, because they have different producers and very
different cadences:

``dashboard``
    ``dashboard_data*.json``, written by ``scripts/export_dashboard.py``. One
    row per session close, and it carries the BTST/WFO ``_meta`` contract.
``intraday``
    ``intraday_<tf>_<symbol>.json``, written by ``scripts/export_intraday.py``.
    Re-derived every few minutes during the session.

The default invocation is deliberately unchanged -- run the dashboard exporter,
then publish the dashboard group -- so the existing ``nifty-dashboard-export``
job keeps working exactly as it did before this file grew a second group.

``--only intraday --skip-export`` is what the high-frequency intraday job uses.
It skips the daily re-export (which is the daily job's job, and which would
otherwise race it for the same objects) and publishes only the intraday JSON
that ``scripts/export_intraday.py`` has already written to disk.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from google.cloud import storage

BUCKET = os.environ["GCS_CACHE_BUCKET"]
PREFIX = "dashboard"

# The dashboard objects under this prefix are served straight to browsers, so the
# Cache-Control has to be set on the object itself rather than left to the reader.
# Without it the default `no-cache` lets the browser/GCS CDN revalidate a stale
# market-data response. Every re-upload overwrites the metadata, so this has to be
# applied on each upload rather than patched onto the objects once.
CACHE_CONTROL = "no-store"

# These objects are fetched directly by browsers, so each one is published with a
# public-read ACL. The bucket cannot scope a public grant to the dashboard/ prefix
# (GCS rejects IAM conditions on public resources), so access is granted per object
# instead. The ACL has to be re-applied on every upload: uploading replaces the
# object and resets its ACL, so granting it once would silently revoke public
# access within one export cycle.
PUBLIC_READ = True

DASHBOARD_OUTPUTS = [
    "dashboard_data.json",
    "dashboard_data_banknifty.json",
    "dashboard_data_reliance.json",
]

# Timeframes and symbols mirror the hardcoded maps in
# scripts/export_intraday.py. Reliance is deliberately absent: it is attempted
# there but has no Fyers 5m archive, so no file is ever written for it. Listing
# it here would make every intraday publish fail its verification step.
INTRADAY_TFS = ("5", "15", "30", "60")
INTRADAY_SYMBOLS = ("nifty", "banknifty")
INTRADAY_OUTPUTS = [
    f"intraday_{tf}_{symbol}.json"
    for symbol in INTRADAY_SYMBOLS
    for tf in INTRADAY_TFS
]

OUTPUT_GROUPS = {
    "dashboard": DASHBOARD_OUTPUTS,
    "intraday": INTRADAY_OUTPUTS,
}
OUTPUTS = DASHBOARD_OUTPUTS + INTRADAY_OUTPUTS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and/or upload dashboard JSON artifacts to GCS."
    )
    parser.add_argument(
        "--only",
        choices=("dashboard", "intraday", "all"),
        default="dashboard",
        help="Which output group to publish (default: dashboard, which is the "
             "pre-existing behaviour).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Publish the artifacts already on disk instead of regenerating "
             "them first. Used by the intraday job, which runs "
             "scripts/export_intraday.py itself under its own freshness gate.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    outputs = (
        OUTPUTS if args.only == "all" else OUTPUT_GROUPS[args.only]
    )

    if not args.skip_export:
        print("[export] Running dashboard exporter...")
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "export_dashboard.py")],
            cwd=root,
        )

        if result.returncode != 0:
            print(
                f"[export] export_dashboard.py failed with exit code {result.returncode}",
                file=sys.stderr,
            )
            return result.returncode

    missing = [
        filename
        for filename in outputs
        if not (root / filename).is_file() or (root / filename).stat().st_size == 0
    ]

    if missing:
        print(
            f"[export] Missing/empty output files: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    print(f"[export] All {len(outputs)} '{args.only}' output(s) verified.")

    client = storage.Client()
    bucket = client.bucket(BUCKET)

    for filename in outputs:
        source = root / filename
        destination = f"{PREFIX}/{filename}"

        print(f"[export] Uploading {filename} -> gs://{BUCKET}/{destination}")
        blob = bucket.blob(destination)
        # Cache-Control is object metadata, not an upload_from_filename argument:
        # passing it as a kwarg raises TypeError and fails the whole export.
        blob.cache_control = CACHE_CONTROL
        blob.upload_from_filename(
            str(source),
            content_type="application/json",
        )

        if PUBLIC_READ:
            blob.acl.all_users().grant_read()
            blob.acl.save()

    print("[export] Dashboard export and upload completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
