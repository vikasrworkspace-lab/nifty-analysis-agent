"""Generate the Firebase Hosting copy of the dashboard UI.

The repository's root ``index.html`` is the hand-maintained, git-tracked source
of truth (see AGENTS.md). It is served by GitHub Pages straight from the
repository root, and Firebase Hosting serves a generated copy from ``public/``.

Having two hand-edited copies is what let the two sites drift: the Firebase copy
gained the GCS data-origin change while the root copy kept fetching a bundled
``dashboard_data.json`` with a query-string cache-buster, so the same dashboard
shipped different behaviour depending on which URL you loaded. This script makes
``public/`` a build output instead -- copy, never edit.

Usage::

    python scripts/sync_public.py            # write public/
    python scripts/sync_public.py --check    # fail if public/ is stale

``--check`` exists so CI or a pre-deploy step can prove ``public/`` matches the
root without mutating anything; it exits 1 and lists the differing files
otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "public"

# Root-relative paths copied into public/ verbatim. index.html references
# data-config.js relatively, so both files must land side by side in public/.
SYNCED_FILES = ("index.html", "data-config.js")

# Generated previously and served from public/, now removed (see AGENTS.md): the
# dashboard JSON is fetched from GCS at runtime, and these copies kept going
# stale against it. Any of them showing up in public/ again means something
# re-created them and the SPA rewrite would serve stale data again.
LEGACY_STATIC_ARTIFACTS = ("dashboard_data.json", "dashboard_data_banknifty.json",
                           "dashboard_data_reliance.json")


def _sync() -> list[Path]:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    changed: list[Path] = []
    for name in SYNCED_FILES:
        src = ROOT / name
        if not src.exists():
            raise SystemExit(f"missing source file: {src}")
        dst = PUBLIC_DIR / name
        payload = src.read_bytes()
        if dst.exists() and dst.read_bytes() == payload:
            continue
        dst.write_bytes(payload)
        changed.append(dst)
    for name in LEGACY_STATIC_ARTIFACTS:
        stale = PUBLIC_DIR / name
        if stale.exists():
            stale.unlink()
            changed.append(stale)
    return changed


def _check() -> list[Path]:
    stale: list[Path] = []
    for name in SYNCED_FILES:
        src = ROOT / name
        dst = PUBLIC_DIR / name
        if not src.exists():
            stale.append(src)
        elif not dst.exists() or dst.read_bytes() != src.read_bytes():
            stale.append(dst)
    for name in LEGACY_STATIC_ARTIFACTS:
        if (PUBLIC_DIR / name).exists():
            stale.append(PUBLIC_DIR / name)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify public/ matches the root UI without writing")
    args = parser.parse_args()

    if args.check:
        stale = _check()
        if stale:
            print("public/ is out of date; run: python scripts/sync_public.py")
            for path in stale:
                print(f"  {path.relative_to(ROOT)}")
            return 1
        print(f"public/ is up to date ({len(SYNCED_FILES)} files)")
        return 0

    changed = _sync()
    if not changed:
        print(f"public/ already up to date ({len(SYNCED_FILES)} files)")
    for path in changed:
        rel = path.relative_to(ROOT)
        print(f"{'removed' if not path.exists() else 'wrote'} {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
