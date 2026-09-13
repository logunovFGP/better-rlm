"""Tests for ``src.describe`` -- the data-only mode catalogue.

Mirrors cline-2's ``describeMode()``-style tests: every value comes back
in the expected shape, every unknown input is rendered as a help hint
rather than raising, and ``all_modes`` is stable.
"""

from __future__ import annotations

import pytest

from src.describe import (
    AUTO_DESCRIPTION,
    HOST_MODE_LABEL,
    MODE_API,
    MODE_AUTO,
    MODE_CLI,
    PROXY_MODE_LABEL,
    VALID_MODES,
    all_modes,
    describe_mode,
)


def test_all_modes_returns_three_in_documented_order() -> None:
    """The picker shows modes in this exact order: auto first, then the two
    concrete ones. Same for VALID_MODES -- it is the source of truth for
    what is acceptable in ``mode:``."""
    assert all_modes() == (MODE_AUTO, MODE_CLI, MODE_API)
    assert VALID_MODES == (MODE_AUTO, MODE_CLI, MODE_API)


@pytest.mark.parametrize("mode", [MODE_AUTO, MODE_CLI, MODE_API])
def test_describe_mode_known_returns_full_description(mode: str) -> None:
    """Every known mode has a label, a non-empty summary, and pros/cons.

    Mirrors cline-2's contract: ``describeMode(mode)`` always returns a
    ModeDescription with at least one pro and at least one con so the
    picker's two-column compare never renders an empty column.
    """
    d = describe_mode(mode)
    assert d.label
    assert d.summary
    assert d.pros, f"mode {mode} has no pros"
    assert d.cons, f"mode {mode} has no cons"


def test_describe_mode_auto_uses_dedicated_entry() -> None:
    """auto is described separately from the cli/api pair -- the prose
    differs enough that conflating them would lie about fallback behaviour."""
    d = describe_mode(MODE_AUTO)
    assert d is AUTO_DESCRIPTION


def test_describe_mode_cli_is_proxy_label() -> None:
    """The picker's column header for ``claude-cli`` uses the proxy-mode
    label, matching cline-2's terminology."""
    d = describe_mode(MODE_CLI)
    assert d.label == PROXY_MODE_LABEL


def test_describe_mode_api_is_host_label() -> None:
    """The picker's column header for ``api`` uses the host-mode label."""
    d = describe_mode(MODE_API)
    assert d.label == HOST_MODE_LABEL


def test_describe_mode_unknown_renders_help_in_cons() -> None:
    """Unknown modes do NOT raise -- they return a description whose
    ``cons`` field carries the remediation hint, so the picker can render
    it without a try/except."""
    d = describe_mode("garbage")
    assert "Unrecognised mode" in d.summary
    assert "garbage" in d.summary
    assert d.pros == ()
    assert d.cons  # at least one remediation hint


@pytest.mark.parametrize("mode", [None, "", "  "])
def test_describe_mode_empty_or_none_safe(mode: str | None) -> None:
    """None / empty / whitespace must never raise -- the picker's /status
    line calls ``describe_mode`` on whatever is in config.yaml."""
    d = describe_mode(mode or "")
    assert d.summary
    assert d.cons


def test_no_provider_catalogue_is_exported() -> None:
    """describe.py must not grow a provider picker back.

    config.PROVIDER_KEY_ENV was deleted in f855b9c and auth.require_anthropic raises on
    every provider but anthropic at every door that leads to a model call -- only its
    transport passes through the session-window ledger, so a Gemini/OpenAI run would
    record no spend and pass no gate. A catalogue here is how a picker offering those
    four gets re-added, so the absence is the thing worth pinning.
    """
    import src.describe as d

    for gone in ("PROVIDERS", "describe_provider", "all_providers"):
        assert not hasattr(d, gone), f"{gone} is back; see auth.require_anthropic"


def test_mode_prose_never_promises_another_provider() -> None:
    """The api-mode row used to read "Works with any provider (Anthropic, Gemini,
    OpenAI, Azure, Portkey)". The server refuses all but the first."""
    blob = " ".join(
        part
        for m in all_modes()
        for d in [describe_mode(m)]
        for part in (d.label, d.summary, *d.pros, *d.cons)
    ).lower()
    for vendor in ("gemini", "openai", "azure", "portkey"):
        assert vendor not in blob, f"mode prose still advertises {vendor}"
