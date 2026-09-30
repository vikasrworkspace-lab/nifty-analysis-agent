"""Next Day Targets card: Max Pain must never be left on "Loading...".

Regression cover for the intraday Max Pain bug.

The card renders three fields -- ``res-r1``, ``res-s1`` and ``res-pain``.
Resistance/support were driven by an if/else-if/else chain that picked between
options levels, standard pivots, and a "Data Syncing..." placeholder. Max Pain
was rendered *inside* the first arm of that chain, so it was only reached when
``options_resistance && options_support`` were both present.

The intraday exporter emits ``options_max_pain`` but never emits
``options_resistance``/``options_support``, so every intraday bar took the
standard-pivot arm, skipped the Max Pain render entirely, and left the field
on its initial "Loading..." text for the whole session. R1/S1 looked fine,
which is what made it hard to spot.

Max Pain is now keyed off ``options_max_pain`` alone, on a path every payload
reaches, with an explicit "N/A" fallback.

These are static assertions on the HTML, matching the approach already used in
tests/test_trade_gate.py: there is no jsdom in this repo, so the tests pin the
structure rather than execute the DOM. The tracked ``index.html`` is always
checked; the untracked ``public/index.html`` copy is checked when present, since
the two had already drifted once.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_INDEX_REL = "index.html"
_PUBLIC_REL = "public/index.html"

# Only index.html is git-tracked; public/ is a local untracked copy. Include the
# public copy when it exists so a clean clone (where it does not) still passes.
TARGET_FILES = [r for r in (_INDEX_REL, _PUBLIC_REL) if (ROOT / r).exists()]

# The options arm of the chain, and the arms that follow it.
OPTIONS_ARM = "if (data.signals.options_resistance && data.signals.options_support) {"
PIVOT_ARM = "} else if (data.signals.high && data.signals.low) {"
SYNCH_ARM = "} else {"


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _block_at(text, brace_index):
    """Return the ``{...}`` block whose opening brace is at ``brace_index``."""
    depth, i = 0, brace_index
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_index:i + 1]
        i += 1
    raise AssertionError("unbalanced braces starting at offset %d" % brace_index)


@pytest.mark.parametrize("rel", TARGET_FILES)
class TestMaxPainNotNested:
    def test_options_arm_exists(self, rel):
        assert _read(rel).count(OPTIONS_ARM) == 1, "%s: options arm not found" % rel

    def test_max_pain_absent_from_options_arm(self, rel):
        """The core regression: Max Pain must not live inside the options arm."""
        text = _read(rel)
        start = text.index(OPTIONS_ARM)
        arm = _block_at(text, text.index("{", start))
        assert "res-pain" not in arm, (
            "%s: res-pain is rendered inside the options_resistance/options_support "
            "arm, so any payload lacking those two fields (all intraday bars) "
            "leaves Max Pain on its initial Loading... text" % rel
        )

    def test_max_pain_hoisted_past_the_chain(self, rel):
        """The Max Pain render must sit after the whole chain, not inside it."""
        text = _read(rel)
        options_at = text.index(OPTIONS_ARM)
        pivot_at = text.index(PIVOT_ARM, options_at)
        render_at = text.index("data.signals.options_max_pain")
        # The guard is the last mention, well clear of the chain's first arm.
        assert render_at > pivot_at, (
            "%s: the options_max_pain guard is still inside the options arm" % rel
        )

    def test_max_pain_gated_on_max_pain_alone(self, rel):
        """The guard must key off options_max_pain, not the resistance/support pair."""
        text = _read(rel)
        assert "if (data.signals.options_max_pain) {" in text, (
            "%s: Max Pain must be keyed off options_max_pain alone" % rel
        )

    def test_max_pain_has_na_fallback(self, rel):
        """A payload with no options data must read N/A, not Loading.

        The fallback lives in the ``else`` arm, so match that arm directly
        rather than hoping the ``if`` block happens to contain it.
        """
        text = _read(rel)
        guard = text.index("if (data.signals.options_max_pain) {")
        block = _block_at(text, text.index("{", guard))
        rest = text[text.index(block) + len(block):]
        m = re.match(r"\s*else\s*", rest)
        assert m, "%s: the options_max_pain guard has no else arm" % rel
        else_arm = _block_at(rest, rest.index("{", m.end()))
        assert 'getElementById(\'res-pain\').textContent = "N/A"' in else_arm, (
            "%s: the Max Pain else arm must set res-pain to N/A" % rel
        )

    def test_loading_text_is_only_the_initial_value(self, rel):
        """No code path should write 'Loading...' over a resolved field."""
        text = _read(rel)
        for el in ("res-r1", "res-s1", "res-pain"):
            assigns = re.findall(
                r"getElementById\('%s'\)\.textContent\s*=\s*([^;]+);" % el, text
            )
            assert assigns, "%s: %s is never assigned" % (rel, el)
            for value in assigns:
                assert "Loading" not in value, (
                    "%s: %s can be assigned 'Loading...' (%s)" % (rel, el, value.strip())
                )


@pytest.mark.parametrize("rel", TARGET_FILES)
class TestResistanceSupportIntact:
    def test_options_arm_still_sets_levels(self, rel):
        text = _read(rel)
        start = text.index(OPTIONS_ARM)
        arm = _block_at(text, text.index("{", start))
        assert "data.signals.options_resistance" in arm
        assert "data.signals.options_support" in arm
        assert "Max call writers (Sellers)" in arm
        assert "Max put writers (Buyers)" in arm

    def test_pivot_arm_still_sets_levels(self, rel):
        text = _read(rel)
        start = text.index(PIVOT_ARM)
        arm = _block_at(text, text.index("{", start))
        assert "Standard Pivots" in arm
        assert "res-r1" in arm and "res-s1" in arm

    def test_synch_arm_covers_all_three(self, rel):
        """Even the last-resort arm must resolve all three fields, not leave Loading."""
        text = _read(rel)
        start = text.index(SYNCH_ARM, text.index(PIVOT_ARM))
        arm = _block_at(text, text.index("{", start))
        assert "res-r1" in arm and "res-s1" in arm


class TestCopiesInSync:
    """Guard the drift between the tracked page and the local public/ copy."""

    def test_target_chain_identical_across_copies(self):
        if _PUBLIC_REL not in TARGET_FILES:
            pytest.skip("public/index.html not present in this checkout")
        root = _read(_INDEX_REL)
        pub = _read(_PUBLIC_REL)

        def chain(text):
            start = text.index(OPTIONS_ARM)
            return text[start:text.index(PIVOT_ARM, start) + len(PIVOT_ARM)]

        assert chain(root) == chain(pub), (
            "index.html and public/index.html have diverged in the "
            "Next Day Targets chain"
        )

    def test_same_number_of_max_pain_renders(self):
        if _PUBLIC_REL not in TARGET_FILES:
            pytest.skip("public/index.html not present in this checkout")
        assert _read(_INDEX_REL).count("getElementById('res-pain').textContent") == (
            _read(_PUBLIC_REL).count("getElementById('res-pain').textContent")
        )


def _latest_with_signals(rel):
    data = json.loads(_read(rel))
    keys = sorted(k for k in data if k != "_meta" and "signals" in data[k])
    return data[keys[-1]]["signals"] if keys else None


class TestPayloadShapes:
    """Pin the payload shape that triggered the bug, when data is present.

    Generated dashboard data is gitignored, so these skip rather than fail on a
    clean checkout.
    """

    def test_intraday_has_max_pain_but_no_options_levels(self):
        path = ROOT / "intraday_5_nifty.json"
        if not path.exists():
            pytest.skip("intraday_5_nifty.json not generated in this checkout")
        signals = _latest_with_signals("intraday_5_nifty.json")
        assert signals is not None
        assert signals.get("options_max_pain"), "payload shape changed: no max pain"
        assert not signals.get("options_resistance"), (
            "payload shape changed: intraday now emits options_resistance, so the "
            "options arm is reachable again -- re-check the Max Pain wiring"
        )
        assert not signals.get("options_support")

    def test_daily_carries_all_three_options_fields(self):
        path = ROOT / "dashboard_data.json"
        if not path.exists():
            pytest.skip("dashboard_data.json not present in this checkout")
        signals = _latest_with_signals("dashboard_data.json")
        assert signals is not None
        for field in ("options_resistance", "options_support", "options_max_pain"):
            assert signals.get(field), "daily payload missing %s" % field
