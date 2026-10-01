"""Guards the GCS upload contract in scripts/cloud_export.py.

`Blob.upload_from_filename()` does not accept a `cache_control` keyword. Passing
it raised TypeError inside the container, which meant every dashboard export
failed after the exporter had already done all its work, and the bucket kept
serving the last good JSON. That is invisible except in Cloud Logging, so it is
pinned here against the installed google-cloud-storage signature.
"""

import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_cloud_export():
    os.environ.setdefault("GCS_CACHE_BUCKET", "test-bucket")
    spec = importlib.util.spec_from_file_location(
        "cloud_export_under_test", ROOT / "scripts" / "cloud_export.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cloud_export = _load_cloud_export()


def test_upload_from_filename_rejects_cache_control():
    """Documents why cache_control must be set on the blob, not the call."""
    from google.cloud.storage.blob import Blob

    params = inspect.signature(Blob.upload_from_filename).parameters
    assert "cache_control" not in params


def test_blob_exposes_cache_control_attribute():
    from google.cloud.storage.blob import Blob

    assert hasattr(Blob, "cache_control")


class _FakeAcl:
    def all_users(self):
        return self

    def grant_read(self):
        return self

    def save(self):
        return None


def test_object_acl_has_no_all_users():
    """Documents why public read is granted at upload, not via blob.acl.

    google-cloud-storage 3.x removed ObjectACL.all_users(), so the old
    `blob.acl.all_users().grant_read()` raised AttributeError *after* the object
    had already been replaced, leaving it private and returning 403 to browsers.
    """
    from google.cloud.storage.acl import ObjectACL

    assert not hasattr(ObjectACL, "all_users")


class _FakeBlob:
    def __init__(self, destination):
        self.destination = destination
        self.cache_control = None
        self.content_type = None

    def upload_from_filename(self, filename, content_type=None, **kwargs):
        if "cache_control" in kwargs:
            raise TypeError(
                "upload_from_filename() got an unexpected keyword argument "
                "'cache_control'"
            )
        self.content_type = content_type
        self.uploaded_from = filename
        self.predefined_acl = kwargs.get("predefined_acl")


class _FakeBucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, destination):
        blob = _FakeBlob(destination)
        self.blobs[destination] = blob
        return blob


class _FakeClient:
    def __init__(self):
        self._bucket = _FakeBucket()

    def bucket(self, name):
        return self._bucket


def test_main_publishes_with_no_store_cache_control(tmp_path, monkeypatch):
    """End-to-end over the real main(): must set no-store without raising."""
    (tmp_path / "dashboard_data.json").write_text("{}", encoding="utf-8")

    client = _FakeClient()

    class _RootPath:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return self

        @property
        def parents(self):
            return [self, self]

        def __truediv__(self, other):
            return tmp_path / str(other)

    monkeypatch.setattr(cloud_export, "BUCKET", "test-bucket")
    monkeypatch.setattr(cloud_export, "OUTPUTS", ["dashboard_data.json"])
    monkeypatch.setattr(cloud_export, "Path", _RootPath)
    monkeypatch.setattr(cloud_export.storage, "Client", lambda *a, **k: client)
    monkeypatch.setattr(
        cloud_export,
        "parse_args",
        lambda argv=None: argparse.Namespace(only="all", skip_export=True),
    )

    rc = cloud_export.main()

    assert rc == 0
    blob = client._bucket.blobs["dashboard/dashboard_data.json"]
    assert blob.cache_control == cloud_export.CACHE_CONTROL == "no-store"
    assert blob.content_type == "application/json"
    # Public read must be granted by the upload, or the object ends up 403.
    assert blob.predefined_acl == "publicRead"


def test_source_does_not_pass_cache_control_to_upload():
    """Belt-and-braces: reject the regression at the source level."""
    source = (ROOT / "scripts" / "cloud_export.py").read_text(encoding="utf-8")
    assert "cache_control=CACHE_CONTROL" not in source
    assert "blob.cache_control = CACHE_CONTROL" in source


def test_source_grants_public_read_via_predefined_acl():
    """Reject the removed all_users() ACL call regressing back in."""
    source = (ROOT / "scripts" / "cloud_export.py").read_text(encoding="utf-8")
    assert "acl.all_users()" not in source
    assert 'predefined_acl="publicRead"' in source
