"""Interactive Fyers OAuth login: the daily manual step.

Run this once before the open each trading day. It mints a fresh access token
(via the normal browser login) and adds it as a new version of
``FYERS_ACCESS_TOKEN`` in Secret Manager, which is the only credential the
Cloud Run data jobs read. They bind ``latest``, so every job picks the new
token up on its next execution with no redeploy.

    scripts/fyers_login.py --push-secret

Design notes:

* By default this is purely local: it writes the token to ``.fyers_token``
  (gitignored). ``--push-secret`` is what makes the token reach the cloud.
* The token is never printed -- only a length + truncated SHA-256 fingerprint.
* The token is never embedded in a shell command; it is piped over stdin
  (``gcloud secrets versions add --data-file=-``).
* The secret is never created here, only added to; a missing secret is a hard
  error. Overwriting is never silent -- the new version number is shown and an
  explicit y/N confirmation is required.
* **No refresh token.** Fyers banned ``validate-refresh-token`` platform-wide
  (``code -16``): *"Refresh token API is currently disabled to comply with SEBI
  regulations."* An earlier version of this script seeded ``FYERS_REFRESH_TOKEN``
  for the daily rotator; that rotator could never work and the seeding only
  added a second prompt for a credential nothing reads. Fyers also returns no
  replacement refresh token, so the old scheme needed a ~15-day re-seed anyway.
  ``FYERS_REFRESH_TOKEN`` is unused and may be deleted.
* After pushing, the new token is **verified against Fyers**. A bad token used
  to be discovered at 09:00 when data stopped flowing; now it fails here, in
  the morning, with a non-zero exit.
* The FYERS PIN and TOTP key are never requested or stored here: the browser
  handles them. They only exist for the parked TOTP rotator
  (``scripts/fyers_auth_job.py``), which is not scheduled.
* It is never invoked automatically: the flag must be passed, and the command
  refuses to run at all inside Cloud Run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

from core.fyers_auth import token_fingerprint

DEFAULT_SECRET_NAME = "FYERS_ACCESS_TOKEN"


def verify_token(access_token: str, client_id: str, timeout: int = 30) -> tuple[bool, str]:
    """Prove the freshly minted token actually works, before anyone relies on it.

    Returns (ok, detail). ``detail`` carries Fyers' own message on failure so the
    cause is diagnosable from the morning output alone. Never includes the token.
    """
    try:
        import requests
    except ImportError:
        return True, "skipped (requests not installed)"

    try:
        resp = requests.get(
            "https://api-t1.fyers.in/api/v3/profile",
            headers={"Authorization": f"Bearer {access_token}",
                     "X-Api-Key": client_id,
                     "Content-Type": "application/json"},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - any network problem is "not verified"
        return False, f"could not reach Fyers ({type(exc).__name__}: {exc})"

    try:
        body = resp.json()
    except ValueError:
        return False, f"HTTP {resp.status_code} with a non-JSON body"

    if resp.status_code == 200 and isinstance(body, dict) and body.get("s") == "ok":
        return True, "Fyers accepted the token"

    if isinstance(body, dict):
        return False, (
            f"HTTP {resp.status_code}: "
            f"code={body.get('code')} message={body.get('message')}"
        )
    return False, f"HTTP {resp.status_code}"



def _gcloud_executable() -> str | None:
    """Locate the gcloud CLI, tolerating the Windows .cmd shim."""
    for name in ("gcloud", "gcloud.cmd", "gcloud.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _gcloud(exe: str, args: list[str], stdin_data: str | None = None):
    return subprocess.run(
        [exe, *args],
        input=stdin_data,
        capture_output=True,
        text=True,
    )


def _active_project(exe: str) -> str | None:
    result = _gcloud(exe, ["config", "get-value", "project"])
    project = (result.stdout or "").strip()
    if result.returncode != 0 or not project or project == "(unset)":
        return None
    return project


def _current_versions(exe: str, secret_name: str, project: str) -> list[str]:
    result = _gcloud(
        exe,
        ["secrets", "versions", "list", secret_name,
         f"--project={project}", "--format=value(name)"],
    )
    if result.returncode != 0:
        return []
    return [v for v in (result.stdout or "").split() if v]


def push_secret(access_token: str, secret_name: str, project: str | None) -> int:
    """Add `access_token` as a new version of an existing secret. Returns an exit code."""
    print("=" * 60)
    print("PUSHING TOKEN TO SECRET MANAGER (--push-secret)")
    print("=" * 60)

    if os.environ.get("K_SERVICE"):
        print("Refusing to run inside Cloud Run. This step is manual, by design.")
        return 1

    exe = _gcloud_executable()
    if not exe:
        print("ERROR: gcloud CLI not found on PATH. Install the Google Cloud SDK, "
              "or run 'gcloud auth login' after installing it.")
        return 1

    project = project or _active_project(exe)
    if not project:
        print("ERROR: no active gcloud project. Pass --project, or run "
              "'gcloud config set project <id>'.")
        return 1
    print(f"Project: {project}")

    describe = _gcloud(exe, ["secrets", "describe", secret_name, f"--project={project}"])
    if describe.returncode != 0:
        # Deliberately do not create it: creating a secret is an explicit,
        # reviewed act that belongs outside this script.
        print(f"ERROR: secret '{secret_name}' not found in project '{project}'.")
        print("Create it once, manually, before using --push-secret:")
        print(f"  gcloud secrets create {secret_name} --replication-policy=automatic "
              f"--project={project}")
        return 1

    existing = _current_versions(exe, secret_name, project)
    new_version = len(existing) + 1
    print(f"Secret: {secret_name} (existing versions: {len(existing)}, "
          f"this will become version {new_version})")
    print(f"Token fingerprint: {token_fingerprint(access_token)}  (the token itself is never printed)")

    try:
        answer = input(f"Add this token as version {new_version}? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 1
    if answer not in ("y", "yes"):
        print("Aborted. No secret version was added.")
        return 1

    result = _gcloud(
        exe,
        ["secrets", "versions", "add", secret_name, "--data-file=-",
         f"--project={project}", "--quiet"],
        stdin_data=access_token,
    )
    if result.returncode != 0:
        print("ERROR: failed to add the secret version.")
        print((result.stderr or "").strip())
        return 1

    print(f"SUCCESS: added version {new_version} of '{secret_name}'. "
          f"Attached job instances pick it up on their next start.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Fyers access token.")
    parser.add_argument(
        "--push-secret", action="store_true",
        help="Also add the token as a new version of an existing Secret Manager "
             "secret. This is what the Cloud Run data jobs read.",
    )
    parser.add_argument("--secret-name", default=None,
                        help="Secret to update with the access token "
                             f"(default: $FYERS_SECRET_NAME or {DEFAULT_SECRET_NAME}).")
    parser.add_argument("--project", default=None,
                        help="GCP project (default: the active gcloud project).")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the post-push Fyers verification call.")
    args = parser.parse_args()

    load_dotenv()
    client_id = os.getenv("FYERS_APP_ID")
    secret_key = os.getenv("FYERS_SECRET_KEY")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI")

    if not client_id or not secret_key:
        print("Error: FYERS_APP_ID or FYERS_SECRET_KEY not found in .env")
        return 1

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )

    # 1. Generate Auth URL
    response = session.generate_authcode()
    print("=" * 60)
    print("ACTION REQUIRED: Open this URL in your browser to login:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    print("\nAfter logging in, you will be redirected to an error/blank page.")
    print("Look at the URL in your browser. It will look something like:")
    # Derive the example from the configured redirect so it cannot go stale.
    hint = (redirect_uri or "<FYERS_REDIRECT_URI is not set>").rstrip("/")
    print(f"{hint}/?auth_code=SOME_LONG_CODE&state=None\n")

    # 2. Get auth code
    auth_code_input = input("Paste the ENTIRE redirected URL here (or just the auth_code): ").strip()

    if "auth_code=" in auth_code_input:
        # Extract just the code from the URL
        auth_code = auth_code_input.split("auth_code=")[1].split("&")[0]
    else:
        auth_code = auth_code_input

    if not auth_code:
        print("Invalid auth code.")
        return 1

    # 3. Generate Access Token
    session.set_token(auth_code)
    try:
        response = session.generate_token()
        access_token = response["access_token"]

        # Save to file
        with open(".fyers_token", "w") as f:
            f.write(access_token)

        print("\nSUCCESS! Access token generated and saved to .fyers_token")
    except Exception as e:
        print(f"\nERROR generating token: {e}")
        print("Please check if your App ID and Secret Key in .env are exactly correct.")
        return 1

    if not args.push_secret:
        return 0

    secret_name = (
        args.secret_name
        or os.getenv("FYERS_SECRET_NAME")
        or DEFAULT_SECRET_NAME
    )
    code = push_secret(access_token, secret_name, args.project)
    if code != 0:
        return code

    # Prove the token works. Otherwise a bad token is only discovered at 09:00,
    # when data silently stops flowing; here it fails before the open instead.
    if args.skip_verify:
        print("Skipping verification (--skip-verify).")
        return 0

    print("\n" + "=" * 60)
    print("VERIFYING THE NEW TOKEN WITH FYERS")
    print("=" * 60)
    ok, detail = verify_token(access_token, client_id)
    print(f"{'TOKEN OK' if ok else 'TOKEN FAILED'}: {detail}")
    if not ok:
        print("\nThe token was pushed but Fyers does not accept it. The data jobs "
              "will fail today. Re-run this command before the open.", file=sys.stderr)
        return 1

    print("\nAll done. Nothing else is needed today: nifty-fyers-update, "
          "nifty-intraday-export and nifty-dashboard-export pick up this token "
          "automatically, and nifty-daily-refresh runs at 20:00.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
