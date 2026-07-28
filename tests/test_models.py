import dataclasses

from src.config import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET_5, load_config
from src.models import Role, configured, map_for_mode, policy_name


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
