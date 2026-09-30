"""Live render check for the Layer A / Layer B trade engine.

``tests/test_option_model.py`` asserts the Python option model. This executes the
dashboard's own JavaScript -- the Black-Scholes price function, the option
R:R construction, and the Layer A provenance / gap helpers -- against a minimal
DOM stub, verifying the numbers that actually reach the screen.

It also pins CROSS-LANGUAGE PARITY: the browser's Black-Scholes must agree with
``core/option_pricing.py``. The two implementations are unavoidable (the
exporter persists the chain, the browser prices the scenario legs), so without
this check they could drift apart silently and the rendered ratio would stop
matching the persisted one.

Skips cleanly when node is unavailable.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "dom_option_layer.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)


def _run():
    return subprocess.run(
        ["node", str(HARNESS)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_option_layer_renders_without_error():
    result = _run()
    assert result.returncode == 0, (
        "option layer harness failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "ALL CHECKS PASSED" in result.stdout


def test_browser_black_scholes_matches_python():
    """Guards the JS/Python split from drifting apart.

    A silent divergence here is invisible to every Python test and would show
    the user an option ratio inconsistent with the persisted chain.
    """
    result = _run()
    assert "ATM call price matches Python" in result.stdout
    assert "OTM call price matches Python" in result.stdout
    assert "put-call parity at r=0 holds in JS" in result.stdout


def test_sub_one_ratio_is_rendered_not_hidden():
    """A UI that suppresses an unfavourable ratio passes every Python test.

    The whole point of Layer B is that an unfavourable trade is visible, so the
    rendered DOM is asserted to contain the number AND the warning badge.
    """
    result = _run()
    assert "sub-1.0 ratio is rendered numerically" in result.stdout
    assert "unfavourable warning badge is shown" in result.stdout
    assert "UNFAVOURABLE ratio still displays a number" in result.stdout


def test_missing_and_stale_data_render_a_refusal():
    """No chain, no IV and an expired chain must all refuse, not fabricate."""
    result = _run()
    assert "absent chain -> NO CHAIN status" in result.stdout
    assert "stale chain -> STALE status" in result.stdout
    assert "missing IV -> rr null" in result.stdout
    assert "expired expiry -> rr null" in result.stdout


def test_underlying_forecast_is_tested_directly():
    """buildUnderlyingForecast() is the extracted pure Layer A.

    Asserting on the section headings means the harness cannot quietly stop
    covering the function that the whole two-layer design hangs off.
    """
    result = _run()
    for marker in (
        "buildUnderlyingForecast(): pure Layer A, tested directly",
        "Qualification gates are individually load-bearing",
        "Structured BTST gap state",
        "Risk-free rate labelling",
    ):
        assert marker in result.stdout, "missing section: " + marker


def test_gap_state_is_structured_not_inferred_from_prose():
    """The old code derived BTST direction by searching provenance text.

    Removing that inference must not be silently reversible, so the structured
    fields and their UNKNOWN default are asserted directly.
    """
    result = _run()
    for marker in (
        "BTST gap is UNKNOWN with no overnight data",
        "BTST gap is UNKNOWN with NaN, not coerced to FLAT",
        "no-trade BTST forecast has an UNKNOWN gap, not a fabricated one",
        "LONG + gap UP is ALIGNED",
        "SHORT + gap DOWN is ALIGNED",
    ):
        assert marker in result.stdout, "missing assertion: " + marker


def test_production_rate_is_labelled_as_a_proxy():
    """7.17% must never be presented as a maturity-matched curve."""
    result = _run()
    for marker in (
        "production rate is rendered as 7.17%",
        "production rate is named as the 10Y G-Sec source",
        "production rate is labelled a PROXY, not a curve",
        "an unconfigured rate is disclosed, never invented",
    ):
        assert marker in result.stdout, "missing assertion: " + marker


def test_both_layers_describe_the_same_trade():
    """Layer B must consume the same percentiles as the Layer A ratio.

    If these decouple, the two R:R figures on screen describe different trades
    and the comparison between them is meaningless.
    """
    result = _run()
    assert "option expected move equals the T2 move" in result.stdout
    assert "option invalidation equals the stop move" in result.stdout
    assert "displayed stop IS the stop used in the R:R" in result.stdout
