"""`allow`: which tool GROUPS this server advertises to the agent.

The distinction these tests protect is between two things that look similar and are
not. `engine.sandbox_guidance` handles a sandbox that was supposed to be there and is
not — a fact about the machine, reported at call time. `allow` handles a deployment
that never offered one: the tool is never registered, so the agent cannot call it at
all. Only the second one saves the round trip, because an agent that can see rlm_exec
will call it before it can learn that Docker is down.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

import better_rlm.config as config_mod
import better_rlm.server as srv

#: Group -> the tools it carries. The ONLY place the mapping is written down twice
#: (the other is the decorators in server.py), which is the point: a decorator retagged
#: by hand diverges from this and fails below.
GROUP_TOOLS = {"sandbox": {"rlm_exec", "rlm_query"}}


def _advertised() -> set[str]:
    """Tool names FastMCP would send in a tools/list response — the agent's actual view.

    Goes through the public `list_tools`, not `mcp._tool_manager._tools`, so the test
    breaks if the advertised set stops matching the registry for some other reason.
    """
    return {t.name for t in asyncio.run(srv.mcp.list_tools())}


def test_the_default_config_withholds_the_sandbox_group():
    """Reads the BAKED-IN default, not the checkout's config.yaml, so a developer who
    opts into the sandbox locally does not turn this into a red test — the intent being
    pinned is what a fresh install does."""
    assert config_mod._allow(config_mod._DEFAULTS["allow"]) == ("no-sandbox",)


def test_only_allowed_groups_are_advertised():
    """The live module, against whatever config it actually loaded.

    Both directions asserted: a withheld group must advertise NOTHING, and an allowed
    group must advertise EVERYTHING. One-sided, this passes on a server that registers
    no tools at all.
    """
    names = _advertised()
    assert names, "no tools advertised at all"
    for group, tools in GROUP_TOOLS.items():
        if group in srv.CFG.allow:
            assert tools <= names, f"{group} allowed but missing {sorted(tools - names)}"
        else:
            assert not (tools & names), \
                f"{group} withheld but advertised {sorted(tools & names)}"


def test_a_withheld_group_still_leaves_the_function_callable():
    """`mcp.tool()` returns fn unchanged, so withholding must only move the ADVERTISED
    set. If _tool ever started returning None or a wrapper for the withheld branch,
    every direct-call test in the suite would break at once instead of here."""
    assert callable(srv.rlm_exec) and callable(srv.rlm_query)


def test_allowing_the_sandbox_group_registers_it(monkeypatch):
    """The other half of the branch in `_tool`.

    Decorated into a THROWAWAY FastMCP: registering into the module's real one would
    leave rlm_exec advertised for every test that runs after this file.
    """
    from mcp.server.fastmcp import FastMCP

    monkeypatch.setattr(srv, "mcp", FastMCP("test"))
    monkeypatch.setattr(srv, "CFG",
                        dataclasses.replace(srv.CFG, allow=("no-sandbox", "sandbox")))

    @srv._tool("sandbox")
    def rlm_exec(code: str) -> str:
        return code

    assert "rlm_exec" in {t.name for t in asyncio.run(srv.mcp.list_tools())}


def test_rlm_status_names_the_tools_it_is_withholding():
    """A withheld tool is INVISIBLE, so the only symptom of a wrong `allow` is an agent
    that never mentions rlm_query. The status line has to name the cause."""
    line = srv._exposed_line()
    if srv.CFG.allows_sandbox:
        assert "WITHHELD" not in line
    else:
        assert "rlm_exec" in line and "rlm_query" in line and "RLM_ALLOW" in line


@pytest.mark.parametrize("bad", [
    [], "", ["no-sanbdox"], ["sandbox", "docker"], None,
    # Scalars from YAML (`allow: 5`, `allow: true`). These used to escape as TypeError
    # "'int' object is not iterable" -- a config error reported in words that name
    # neither the setting nor its valid values.
    5, True, 3.5,
    # A mapping iterates its keys, so `{no-sandbox: 1}` used to be ACCEPTED as
    # ("no-sandbox",). Silently reading a malformed shape as valid is the failure this
    # validator exists to prevent.
    {"no-sandbox": 1},
])
def test_an_unusable_allow_is_rejected_not_degraded(bad):
    """Same reasoning as `_sandbox`: a silently-ignored typo exposes a group the
    operator meant to withhold, with nothing in the logs saying so. An empty set is
    rejected too — a server with no tools is a mistake, never an intent."""
    with pytest.raises(ValueError):
        config_mod._allow(bad)


@pytest.mark.parametrize("bad", [5, True, 3.5, {"no-sandbox": 1}, None])
def test_a_rejected_allow_names_the_setting_and_its_valid_values(bad):
    """The load-bearing half of the test above. `pytest.raises(ValueError)` alone passed
    on messages that named neither, which is how the scalar case read as a Python bug
    rather than a config one."""
    with pytest.raises(ValueError, match=r"allow must be a non-empty subset"):
        config_mod._allow(bad)


def test_the_env_var_wins_over_config_and_takes_a_comma_list(monkeypatch):
    """RLM_ALLOW is how `claude mcp add -e ...` sets the exposed set at registration,
    which is the only place some deployments can set it at all."""
    monkeypatch.setenv("RLM_ALLOW", "no-sandbox, sandbox")
    assert config_mod.load_config().allow == ("no-sandbox", "sandbox")
    assert config_mod.load_config().allows_sandbox


def test_allow_is_order_preserving_and_deduped():
    assert config_mod._allow(["sandbox", "no-sandbox", "sandbox"]) == ("sandbox", "no-sandbox")
