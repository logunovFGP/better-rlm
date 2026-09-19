"""Tests for ``better_rlm.describe`` -- the data-only mode catalogue.

Mirrors cline-2's ``describeMode()``-style tests: every value comes back
in the expected shape, every unknown input is rendered as a help hint
rather than raising, and ``all_modes`` is stable.
"""

from __future__ import annotations

import pytest

from better_rlm.describe import (
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


def test_every_offered_provider_is_protocol_compatible():
    """The invariant the old `no catalogue at all` test was really protecting.

    That test asserted describe.py must export no provider catalogue, because the
    picker it came with offered gemini/openai/azure/portkey -- vendors whose selection
    wrote a config.yaml auth.require_anthropic then rejected at the first model call.
    The ban was aimed at the wrong thing. What must hold is that the picker never
    offers a provider the server would refuse, and every entry here is
    provider=anthropic: they differ by endpoint and by how you authenticate, never by
    client, so the ledger and the budget floor survive any selection.
    """
    from better_rlm.describe import AUTH_API_KEY, AUTH_CLI, PROVIDERS, all_providers

    assert all_providers(), "the catalogue is empty; the picker has nothing to offer"
    for pid in all_providers():
        d = PROVIDERS[pid]
        assert d.auth in (AUTH_CLI, AUTH_API_KEY), pid
        if d.auth == AUTH_API_KEY:
            assert d.key_env, f"{pid} has no key variable of its own"

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


# --- the model catalogue -------------------------------------------------------


def test_the_minimax_base_url_does_not_end_in_v1():
    """cline stores MiniMax as .../anthropic/v1. Copying that verbatim 404s everything.

    The Anthropic SDK appends /v1/messages to base_url itself. Measured against the
    installed SDK:

        https://api.minimax.io/anthropic     -> .../anthropic/v1/messages      OK
        https://api.minimax.io/anthropic/v1  -> .../anthropic/v1/v1/messages   404

    So the string that is right in cline's TS catalogue is wrong here, and the
    difference is invisible until the first model call. Pinned because "port the
    catalogue" is exactly the change that would paste the /v1 in.
    """
    from better_rlm.describe import PROVIDER_MINIMAX, describe_provider

    url = describe_provider(PROVIDER_MINIMAX).base_url
    assert url, "MiniMax must carry an explicit endpoint"
    assert not url.rstrip("/").endswith("/v1"), (
        f"{url} would make the SDK request /v1/v1/messages"
    )


def test_every_vendor_has_a_model_table_and_role_defaults():
    """A vendor offered on the first screen with no models behind it is a dead end."""
    from better_rlm.describe import MODELS, ROLE_DEFAULTS, VENDORS

    for vid in VENDORS:
        rows = MODELS.get(vid, ())
        assert rows, f"{vid} is offered but has no models"
        ids = {m.id for m in rows}
        assert vid in ROLE_DEFAULTS, f"{vid} has no role defaults"
        for role, pick in zip(("root", "override", "sub"), ROLE_DEFAULTS[vid]):
            assert pick in ids, f"{vid} defaults its {role} to {pick}, which it does not serve"


def test_model_ids_are_unique_across_vendors():
    """describe_model() looks an id up across every vendor, so a collision hands one
    vendor's row to the other's model -- the wrong window and output ceiling,
    silently."""
    from better_rlm.describe import MODELS

    seen: dict[str, str] = {}
    for vid, rows in MODELS.items():
        for m in rows:
            assert m.id not in seen, f"{m.id} is in both {seen[m.id]} and {vid}"
            seen[m.id] = vid


def test_models_for_a_custom_endpoint_is_empty():
    """A custom endpoint gets free-text entry, never another vendor's list.

    vendor_of returns "" for it, and offering it the Claude catalogue would be the
    same defect that had MiniMax installs running claude-sonnet-5.
    """
    from better_rlm.describe import PROVIDER_CUSTOM, models_for, role_defaults

    assert models_for(PROVIDER_CUSTOM) == ()
    assert models_for("nonsense") == ()
    # Defaults still resolve, so the picker opens on something rather than crashing.
    assert len(role_defaults(PROVIDER_CUSTOM)) == 3


def test_the_cli_and_api_paths_to_claude_share_one_model_list():
    """claude-cli and anthropic are one vendor reached two ways. Two lists drift."""
    from better_rlm.describe import PROVIDER_ANTHROPIC, PROVIDER_CLAUDE_CLI, models_for

    assert models_for(PROVIDER_CLAUDE_CLI) == models_for(PROVIDER_ANTHROPIC)


def test_no_module_quotes_a_price():
    """There is no rate table, and there must not be one again.

    One hand-maintained table priced all seven MiniMax models at $0.00 -- cost_usd
    returned 0.0 for an id it did not know -- while the picker offered them. Deriving
    one table from the other fixed the inconsistency and still left the tool quoting
    numbers it could not check against an invoice. Token counts come back FROM the
    API and stay; money does not.
    """
    import ast
    from pathlib import Path

    banned = {"COST_PER_MTOK", "cost_usd", "price_in", "price_out", "report_cost"}
    offenders: list[str] = []
    for path in sorted(Path(__file__).resolve().parent.parent.glob("better_rlm/*.py")):
        # Identifiers, not raw text: several modules explain in prose WHY there is no
        # price table, and a substring search would fail on that documentation.
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            for name in (getattr(node, "id", None), getattr(node, "attr", None),
                         getattr(node, "arg", None)):
                if name in banned:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders


def test_the_engine_knows_every_offered_model_context_window():
    """_sub_ctx asks the engine for the sub-model window and falls back to 128k.

    An id the engine has never heard of is read 8x smaller than it is, so the server
    chunks far earlier than it needs to -- conservative, but wrong, and silent.
    """
    from rlm.utils.token_utils import DEFAULT_CONTEXT_LIMIT, get_context_limit

    from better_rlm.describe import MODELS

    for vid, rows in MODELS.items():
        for m in rows:
            got = get_context_limit(m.id)
            assert got != DEFAULT_CONTEXT_LIMIT or m.context == DEFAULT_CONTEXT_LIMIT, (
                f"{vid}/{m.id} has a {m.context} window but the engine falls back to "
                f"{DEFAULT_CONTEXT_LIMIT}"
            )
