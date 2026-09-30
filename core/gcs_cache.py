"""Explicit Cloud Storage cache sync for the Cloud Run migration.

Cloud Run jobs have an ephemeral filesystem, so without this every invocation
would start with an empty cache and re-download twenty years of history. This
module mirrors the on-disk caches to a GCS bucket around each job.

Design constraints:

* **Opt-in.** Inert unless ``GCS_CACHE_BUCKET`` is set, so local development
  and the existing GitHub Actions workflow are completely unaffected and never
  import a Google credential.
* **No FUSE.** There is no mounted filesystem; this is an explicit up/down pair
  called at job start and job end.
* **Config-first.** Every synced root is derived from ``settings["paths"]``
  (``historical_dir``, ``intraday_archive_dir``, ``options_dir``,
  ``fyers_db_dir``). There are no hardcoded cache paths here.
* **Opaque blobs.** Files are synced as bytes and never parsed, so this works
  for the raw-I/O sites in ``scripts/`` as well as the ``core/`` seams.
* **A cloud job must never publish a *shorter* cache than it received.** Uploads
  therefore replace a remote object only when the local copy is newer
  (see ``_is_newer``), which keeps a stale or partially-written local file from
  clobbering good remote history.
"""

from __future__ import annotations

import os
from pathlib import Path

import core.settings as settings_mod

BUCKET_ENV_VAR = "GCS_CACHE_BUCKET"
PREFIX_ENV_VAR = "GCS_CACHE_PREFIX"
DEFAULT_PREFIX = "nifty-cache"

# Settings keys that hold the cache roots. All are resolved via settings.rel().
CACHE_PATH_KEYS = (
    "historical_dir",
    "intraday_archive_dir",
    "fyers_db_dir",
    "options_dir",
)

# Extensions worth syncing. Keeps stray logs/reports out of the bucket.
SYNCED_SUFFIXES = (".csv",)

# Skip obvious noise while walking the cache roots.
SKIP_DIR_NAMES = {"__pycache__"}


def bucket_name() -> str | None:
    """Configured bucket, or None when the cache layer is disabled."""
    return (os.getenv(BUCKET_ENV_VAR) or "").strip() or None


def is_enabled() -> bool:
    return bucket_name() is not None


def prefix() -> str:
    return (os.getenv(PREFIX_ENV_VAR) or DEFAULT_PREFIX).strip("/")


def cache_roots(settings: dict = None) -> list[Path]:
    """Absolute cache directories, all derived from settings.

    Roots that do not exist on disk are still returned; callers skip them.
    """
    if settings is None:
        settings = settings_mod.load_settings()
    roots: list[Path] = []
    for key in CACHE_PATH_KEYS:
        try:
            root = settings_mod.rel(settings, key)
        except KeyError:
            # A config without this key simply does not contribute a root.
            continue
        if root not in roots:
            roots.append(root)
    return roots


def _client():
    """Build a storage client. Imported lazily so the dependency is optional
    at import time and never touched while the layer is disabled."""
    from google.cloud import storage

    return storage.Client()


def _object_name(relative: Path) -> str:
    return f"{prefix()}/{relative.as_posix()}"


def _is_syncable(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SYNCED_SUFFIXES


def _is_newer(local: Path, remote) -> bool:
    """True when the local file is newer than the remote object.

    Falls back to uploading when the remote has no update time (or the bucket
    has been deleted), so a first sync always populates the bucket.
    """
    if remote is None:
        return True
    updated = getattr(remote, "updated", None)
    if updated is None:
        return True
    try:
        import datetime as _dt
        local_mtime = _dt.datetime.fromtimestamp(
            local.stat().st_mtime, tz=_dt.timezone.utc
        )
    except OSError:
        return False
    return local_mtime > updated


def _iter_files(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if _is_syncable(path):
            yield path


def _relativise(path: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        try:
            return path.relative_to(root.parent)
        except ValueError:
            continue
    return None


def sync_up(roots: list[Path] = None, settings: dict = None) -> dict:
    """Upload local cache files to the bucket.

    Returns a summary dict; a no-op summary when disabled.
    """
    if not is_enabled():
        return {"enabled": False, "uploaded": 0, "skipped": 0, "errors": []}

    if roots is None:
        roots = cache_roots(settings)
    bucket = _client().bucket(bucket_name())
    uploaded = skipped = 0
    errors: list[str] = []

    for root in roots:
        for local in _iter_files(root):
            relative = _relativise(local, roots)
            if relative is None:
                continue
            name = _object_name(relative)
            try:
                blob = bucket.get_blob(name)
                if not _is_newer(local, blob):
                    skipped += 1
                    continue
                blob = bucket.blob(name)
                blob.upload_from_filename(str(local))
                uploaded += 1
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort the job
                errors.append(f"upload {name}: {exc}")

    return {"enabled": True, "uploaded": uploaded, "skipped": skipped, "errors": errors}


def sync_down(roots: list[Path] = None, settings: dict = None, overwrite: bool = False) -> dict:
    """Download cache files from the bucket, creating parent directories.

    Local files that already exist are kept unless ``overwrite`` is set, so a
    job never destroys work it just did unless asked to.
    """
    if not is_enabled():
        return {"enabled": False, "downloaded": 0, "kept": 0, "errors": []}

    if roots is None:
        roots = cache_roots(settings)
    bucket = _client().bucket(bucket_name())
    downloaded = kept = 0
    errors: list[str] = []

    for blob in _client().list_blobs(bucket, prefix=f"{prefix()}/"):
        relative = Path(blob.name[len(prefix()) + 1:])
        target = None
        for root in roots:
            # blob.name is "<prefix>/<root name>/<...>"; map it back onto a root.
            try:
                tail = relative.relative_to(root.name)
            except ValueError:
                continue
            target = root / tail
            break
        if target is None:
            continue
        if target.exists() and not overwrite:
            kept += 1
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            downloaded += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"download {blob.name}: {exc}")

    return {"enabled": True, "downloaded": downloaded, "kept": kept, "errors": errors}
