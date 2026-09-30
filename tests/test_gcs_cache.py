"""Tests for the Cloud Storage cache layer.

A fake in-memory bucket stands in for GCS, so these verify path derivation,
round-trip fidelity and the safety gates without credentials or network.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import gcs_cache


class FakeBlob:
    def __init__(self, name, bucket):
        self.name = name
        self._bucket = bucket
        self.updated = None

    def upload_from_filename(self, path):
        self._bucket.objects[self.name] = Path(path).read_bytes()
        import datetime as _dt
        self.updated = _dt.datetime.now(_dt.timezone.utc)

    def download_to_filename(self, path):
        Path(path).write_bytes(self._bucket.objects[self.name])


class FakeBucket:
    def __init__(self, name):
        self.name = name
        self.objects = {}

    def get_blob(self, name):
        data = self.objects.get(name)
        return FakeBlob(name, self) if data is not None else None

    def blob(self, name):
        return FakeBlob(name, self)


class FakeClient:
    def __init__(self):
        self.buckets = {}

    def bucket(self, name):
        return self.buckets.setdefault(name, FakeBucket(name))

    def list_blobs(self, bucket, prefix=""):
        return [FakeBlob(n, bucket) for n in sorted(bucket.objects) if n.startswith(prefix)]


@pytest.fixture
def fake_gcs(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(gcs_cache, "_client", lambda: client)
    monkeypatch.setenv(gcs_cache.BUCKET_ENV_VAR, "test-bucket")
    return client


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Settings whose cache roots resolve into a temp dir, never the real data/."""
    s = {
        "paths": {
            "historical_dir": "hist",
            "intraday_archive_dir": "intraday",
            "fyers_db_dir": "fyers_db",
            "options_dir": "options",
        }
    }
    # settings.rel() normally anchors at the project root; re-anchor it so these
    # tests never touch the real data/ directory.
    monkeypatch.setattr(
        gcs_cache.settings_mod, "rel",
        lambda st, key: tmp_path / st["paths"][key],
    )
    return s, tmp_path


def _make_roots(settings_bundle):
    s, base = settings_bundle
    roots = {}
    for key in gcs_cache.CACHE_PATH_KEYS:
        root = gcs_cache.settings_mod.rel(s, key)
        root.mkdir(parents=True, exist_ok=True)
        roots[key] = root
    return s, roots


def test_disabled_without_bucket(monkeypatch):
    monkeypatch.delenv(gcs_cache.BUCKET_ENV_VAR, raising=False)
    assert gcs_cache.is_enabled() is False
    # Both directions are no-ops and must not touch a client.
    monkeypatch.setattr(gcs_cache, "_client", lambda: (_ for _ in ()).throw(AssertionError("client used")))
    assert gcs_cache.sync_up()["enabled"] is False
    assert gcs_cache.sync_down()["enabled"] is False


def test_blank_bucket_is_treated_as_disabled(monkeypatch):
    monkeypatch.setenv(gcs_cache.BUCKET_ENV_VAR, "   ")
    assert gcs_cache.is_enabled() is False


def test_roots_derive_from_settings(settings):
    s, roots = _make_roots(settings)
    derived = gcs_cache.cache_roots(s)
    assert set(derived) == set(roots.values())


def test_roots_skip_missing_config_keys(monkeypatch, tmp_path):
    s = {"paths": {"historical_dir": "hist"}}
    monkeypatch.setattr(gcs_cache.settings_mod, "rel",
                        lambda st, key: tmp_path / st["paths"][key])
    # historical_dir only; the other three keys are absent and must be skipped.
    assert len(gcs_cache.cache_roots(s)) == 1


def test_round_trip_preserves_bytes(fake_gcs, settings):
    s, roots = _make_roots(settings)
    payload = b"date,open,high,low,close,volume\n2026-01-01,1,2,3,4,5\n"
    (roots["historical_dir"] / "nifty.csv").write_bytes(payload)
    (roots["intraday_archive_dir"] / "5m").mkdir(parents=True, exist_ok=True)
    (roots["intraday_archive_dir"] / "5m" / "nifty.csv").write_bytes(payload)

    up = gcs_cache.sync_up(roots=list(roots.values()), settings=s)
    assert up["enabled"] and up["uploaded"] == 2 and not up["errors"]

    # Wipe local state, exactly like a fresh Cloud Run container.
    for r in roots.values():
        for f in r.rglob("*"):
            if f.is_file():
                f.unlink()

    down = gcs_cache.sync_down(roots=list(roots.values()), settings=s)
    assert down["downloaded"] == 2 and not down["errors"]
    assert (roots["historical_dir"] / "nifty.csv").read_bytes() == payload
    assert (roots["intraday_archive_dir"] / "5m" / "nifty.csv").read_bytes() == payload


def test_only_csv_files_are_synced(fake_gcs, settings):
    s, roots = _make_roots(settings)
    (roots["historical_dir"] / "nifty.csv").write_bytes(b"x")
    (roots["historical_dir"] / "notes.txt").write_bytes(b"x")
    (roots["historical_dir"] / "log.log").write_bytes(b"x")
    up = gcs_cache.sync_up(roots=list(roots.values()), settings=s)
    assert up["uploaded"] == 1


def test_empty_remote_bucket_downloads_nothing(fake_gcs, settings):
    s, roots = _make_roots(settings)
    down = gcs_cache.sync_down(roots=list(roots.values()), settings=s)
    assert down["downloaded"] == 0


def test_existing_local_file_is_kept_by_default(fake_gcs, settings):
    s, roots = _make_roots(settings)
    (roots["historical_dir"] / "nifty.csv").write_bytes(b"fresh")
    fake_gcs.bucket("test-bucket").objects[f"{gcs_cache.prefix()}/hist/nifty.csv"] = b"stale"
    down = gcs_cache.sync_down(roots=list(roots.values()), settings=s)
    assert down["kept"] == 1 and down["downloaded"] == 0
    assert (roots["historical_dir"] / "nifty.csv").read_bytes() == b"fresh"


def test_overwrite_replaces_existing_local_file(fake_gcs, settings):
    s, roots = _make_roots(settings)
    (roots["historical_dir"] / "nifty.csv").write_bytes(b"fresh")
    fake_gcs.bucket("test-bucket").objects[f"{gcs_cache.prefix()}/hist/nifty.csv"] = b"stale"
    down = gcs_cache.sync_down(roots=list(roots.values()), settings=s, overwrite=True)
    assert down["downloaded"] == 1
    assert (roots["historical_dir"] / "nifty.csv").read_bytes() == b"stale"


def test_stale_local_file_does_not_clobber_newer_remote(fake_gcs, settings):
    s, roots = _make_roots(settings)
    local = roots["historical_dir"] / "nifty.csv"
    local.write_bytes(b"stale")
    import time
    old = time.time() - 10_000
    os.utime(local, (old, old))

    remote = FakeBlob(f"{gcs_cache.prefix()}/hist/nifty.csv", fake_gcs.bucket("test-bucket"))
    remote.uploaded_payload = b"newer"
    # Seed the object so get_blob returns it, and mark it newer than the local file.
    fake_gcs.bucket("test-bucket").objects[remote.name] = b"newer"
    blob = fake_gcs.bucket("test-bucket").get_blob(remote.name)
    import datetime as _dt
    blob.updated = _dt.datetime.now(_dt.timezone.utc)
    monkeypatched = gcs_cache.bucket_name()
    assert gcs_cache._is_newer(local, blob) is False


def test_newer_local_file_is_uploaded_over_remote(fake_gcs, settings, tmp_path):
    s, roots = _make_roots(settings)
    local = roots["historical_dir"] / "nifty.csv"
    local.write_bytes(b"newer")
    import datetime as _dt
    old_blob = FakeBlob("hist/nifty.csv", fake_gcs.bucket("test-bucket"))
    old_blob.updated = _dt.datetime.fromtimestamp(0, tz=_dt.timezone.utc)
    assert gcs_cache._is_newer(local, old_blob) is True


def test_missing_remote_always_uploads(settings, tmp_path):
    assert gcs_cache._is_newer(tmp_path / "nope.csv", None) is True


def test_prefix_is_configurable(monkeypatch, fake_gcs, settings):
    monkeypatch.setenv(gcs_cache.PREFIX_ENV_VAR, "/custom/prefix/")
    assert gcs_cache.prefix() == "custom/prefix"
    s, roots = _make_roots(settings)
    (roots["historical_dir"] / "nifty.csv").write_bytes(b"x")
    gcs_cache.sync_up(roots=list(roots.values()), settings=s)
    assert "custom/prefix/hist/nifty.csv" in fake_gcs.bucket("test-bucket").objects
