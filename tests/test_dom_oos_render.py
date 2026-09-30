"""Live render check for the per-date OOS panel.

``tests/test_trade_gate.py`` asserts the *structure* of ``renderPerDateOOS()``.
This executes it against the committed ``dashboard_data*.json`` with a minimal
DOM stub, so the actual rendered output is verified too -- which is the gap
noted in the test_trade_gate docstring.

Skips cleanly when node is unavailable.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "dom_oos_render.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)


def test_per_date_oos_panel_renders_without_error():
    result = subprocess.run(
        ["node", str(HARNESS)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "renderPerDateOOS harness failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "ALL DOM RENDER CASES PASSED" in result.stdout
    # A vacuous pass is the failure mode this harness is guarding against:
    # if every case silently took the fallback branch, it would still "pass".
    assert "!! EXPECTED A REAL RESULT BUT GOT THE FALLBACK" not in result.stdout
