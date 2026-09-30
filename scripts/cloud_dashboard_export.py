"""Cloud-side daily dashboard publish.

This is the body of the ``nifty-dashboard-export`` Cloud Run job, and it is the
reason that job no longer runs ``cloud_export.py`` on its own.

Two modes, chosen from the IST clock:

* **Market hours** (Mon-Fri 09:15-15:35): export from the cache WITHOUT
  refreshing. The day's settled close does not exist yet, so the daily/BTST
  view should keep showing the last settled session. This is the every-5-minute
  tick.
* **Post-close**: refresh first, then export. The refresh settles the day's
  close and pulls the final FII/DII. If it fails, STOP: publishing the old
  cache is precisely the behaviour being removed, so a failed refresh is a
  failed run instead of a quiet fallback.

The cache restore is done by ``scripts/cloud_job.py``, which wraps this script,
so the post-close refresh is incremental rather than a cold download. Ordering
matters more than the individual commands: a crash between the refresh and the
export leaves the previously published JSON in the bucket, which is the correct
outcome. The service serves the last good artifact until a run completes.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]

_IST = ZoneInfo("Asia/Kolkata")

# Indian cash-market session. While it is open the 5-minute dashboard job must
# NOT re-download daily history: the settled close for the day does not exist
# yet, so a refresh would only churn the cache (and, without the in-session
# guard in core.data, could freeze a partial bar). The cache stays on the last
# settled session, which is exactly what the BTST/daily view should show during
# the day. The authoritative settle refresh happens on the post-close run.
_MARKET_OPEN = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 35)


def _in_market_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now(_IST)
    if now.weekday() >= 5:
        return False
    return _MARKET_OPEN <= now.time() < _MARKET_CLOSE


def run(cmd: list[str], desc: str) -> int:
    print(f"[dashboard-export] {desc}...", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(
            f"[dashboard-export] {desc} failed with exit code {result.returncode}",
            file=sys.stderr,
            flush=True,
        )
    return result.returncode


def main() -> int:
    if _in_market_hours():
        print(
            "[dashboard-export] market hours: skipping daily refresh. The cache "
            "holds the last settled session, which is what the daily/BTST view "
            "should show during the session; the post-close run settles it.",
            flush=True,
        )
    else:
        # No --only-down here: the refreshed history is the useful output of
        # this run and must be written back to the cache, otherwise the next
        # tick would re-download the same stale series and redo the same work.
        code = run(
            [sys.executable, str(ROOT / "scripts" / "update_data.py")],
            "Refreshing daily market data",
        )
        if code != 0:
            print(
                "[dashboard-export] refresh failed; NOT exporting. The previously "
                "published dashboard JSON stays live rather than being regenerated "
                "from an unrefreshed cache.",
                file=sys.stderr,
                flush=True,
            )
            return code

    # --only dashboard is the default, but stated explicitly: this job must
    # never start publishing the intraday artifacts, because that is the
    # nifty-intraday-export job's responsibility and the two would race for
    # the same bucket prefix.
    return run(
        [
            sys.executable,
            str(ROOT / "scripts" / "cloud_export.py"),
            "--only",
            "dashboard",
        ],
        "Publishing dashboard JSON to GCS",
    )


if __name__ == "__main__":
    sys.exit(main())
