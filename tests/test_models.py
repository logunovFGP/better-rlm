import dataclasses

from better_rlm.config import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET_5, load_config
from better_rlm.models import (OAUTH_SIBLING, Role, configured, map_for_mode,
                               policy_name)


def test_apikey_mode_uses_configured_models_verbatim():
    cfg = load_config()
    for role, expected in ((Role.ROOT, MODEL_SONNET_5),
                           (Role.OVERRIDE, MODEL_OPUS),
                           (Role.SUB, MODEL_HAIKU)):
        assert map_for_mode("apikey", configured(cfg, role)) == expected
    assert map_for_mode("apikey", "claude-fable-5") == "claude-fable-5"  # no remap


def test_oauth_mode_maps_unavailable_to_closest_sibling():
    cfg = load_config()
    # current models pass through unchanged
    assert map_for_mode("oauth", configured(cfg, Role.ROOT)) == MODEL_SONNET_5
    assert map_for_mode("oauth", configured(cfg, Role.SUB)) == MODEL_HAIKU
    # an unavailable model is mapped to its closest sibling
    assert map_for_mode("oauth", "claude-fable-5") == "claude-opus-4-8"
    # and via role config too
    cfg2 = dataclasses.replace(cfg, root_model="claude-fable-5")
    assert map_for_mode("oauth", configured(cfg2, Role.ROOT)) == "claude-opus-4-8"


def test_policy_name_reports_the_active_mode():
    assert policy_name("oauth") == "OAuthSibling"
    assert policy_name("apikey") == "Direct"


# --- the configured model is the one that runs -------------------------------
def test_a_third_party_endpoint_is_never_handed_a_claude_id():
    """OAUTH_SIBLING describes what ANTHROPIC's subscription serves. Against any other
    endpoint that knowledge is not merely useless, it is wrong: it would swap the
    operator's configured model for a Claude id the host has never heard of."""
    for model_id in OAUTH_SIBLING:
        assert map_for_mode("oauth", model_id, "https://api.minimax.io/anthropic") == model_id
    assert map_for_mode("oauth", "MiniMax-M2.7",
                        "https://api.minimax.io/anthropic") == "MiniMax-M2.7"


def test_select_passes_the_configured_endpoint_to_the_policy(monkeypatch):
    """Not map_for_mode in isolation -- select() is what every call site uses, so the
    guard is worthless if select forgets to hand the base_url over."""
    import better_rlm.models as mod

    monkeypatch.setattr(mod, "_mode", lambda cfg: "oauth")
    cfg = dataclasses.replace(load_config(), base_url="https://api.minimax.io/anthropic",
                              root_model="claude-fable-5", sub_model="MiniMax-M2.7")
    assert mod.select(cfg, Role.ROOT) == "claude-fable-5", "remapped despite the endpoint"
    assert mod.select(cfg, Role.SUB) == "MiniMax-M2.7"
    assert mod.map_model(cfg, "claude-fable-5") == "claude-fable-5"


def test_anthropics_own_endpoint_still_gets_the_sibling_map():
    """The guard must not disable the remap for the case it was written for."""
    cfg = dataclasses.replace(load_config(), base_url="", root_model="claude-fable-5")
    assert map_for_mode("oauth", configured(cfg, Role.ROOT), cfg.base_url) == "claude-opus-4-8"
