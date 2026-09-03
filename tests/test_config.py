import pytest

from src.config import (
    MODEL_HAIKU,
    MODEL_SONNET,
    MODEL_SONNET_5,
    cost_usd,
    estimate_tokens,
    load_config,
)


def test_defaults():
    c = load_config()
    assert c.root_model == MODEL_SONNET_5
    assert c.sub_model == MODEL_HAIKU
    assert c.use_docker is True
    assert c.output_cap_bytes == 4096
    # synthesis answers are bound far more generously than raw-content tool output
    assert c.answer_cap_bytes > c.output_cap_bytes


def test_rlm_sandbox_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("RLM_SANDBOX", "LOCAL")  # case/space tolerant like mode/provider
    c = load_config()
    assert c.sandbox == "local"
    assert c.use_docker is False


def test_rlm_sandbox_rejects_unknown_value(monkeypatch):
    # A typo must NOT silently degrade to host execution of model-written Python.
    monkeypatch.setenv("RLM_SANDBOX", "dcoker")
    with pytest.raises(ValueError, match="docker.*local"):
        load_config()


def test_cost_usd_sonnet():
    # 1M in + 1M out at $3/$15
    assert cost_usd(MODEL_SONNET, 1_000_000, 1_000_000) == 18.0


def test_estimate_tokens():
    assert estimate_tokens("abcd" * 25) == 25  # 100 chars / 4



def test_every_config_path_is_redirected_by_the_cfg_fixture(cfg, tmp_path):
    """The `cfg` fixture must redirect EVERY Path-typed field, discovered from the
    dataclass -- not a hand-kept list.

    Four times this session a path was added to Config and the fixture was not updated:
    the log dir, the context store, the spend ledger, and finally the answer cache. Each
    time tests wrote into the operator's real ~/.rlm, and the last one made five tests
    pass or fail on each other's cached entries. A hand-enumerated redirect loses to the
    next field added; enumerating the dataclass does not.
    """
    import dataclasses
    from pathlib import Path

    strays = []
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, Path) and f.name != "sources_file":   # read-only registry, never written
            try:
                v.resolve().relative_to(tmp_path.resolve())
            except ValueError:
                strays.append(f"{f.name}={v}")
    assert not strays, f"Config path(s) not redirected under tmp_path by the cfg fixture: {strays}"
