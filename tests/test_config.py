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
