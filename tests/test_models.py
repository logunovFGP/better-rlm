import dataclasses

from src.config import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET, load_config
from src.models import (
    DirectStrategy,
    OAuthSiblingStrategy,
    Role,
    get_strategy,
)


def test_direct_strategy_uses_configured_models():
    cfg = load_config()
    s = DirectStrategy(cfg)
    assert s.model_for(Role.ROOT) == MODEL_SONNET
    assert s.model_for(Role.OVERRIDE) == MODEL_OPUS
    assert s.model_for(Role.SUB) == MODEL_HAIKU
    assert s.map_explicit("claude-fable-5") == "claude-fable-5"  # no remap on api key


def test_oauth_strategy_maps_unavailable_to_closest_sibling():
    cfg = load_config()
    s = OAuthSiblingStrategy(cfg)
    # current models pass through unchanged
    assert s.model_for(Role.ROOT) == MODEL_SONNET
    assert s.model_for(Role.SUB) == MODEL_HAIKU
    # an unavailable model is mapped to its closest sibling
    assert s.map_explicit("claude-fable-5") == "claude-opus-4-8"
    # and via role config too
    cfg2 = dataclasses.replace(cfg, root_model="claude-fable-5")
    assert OAuthSiblingStrategy(cfg2).model_for(Role.ROOT) == "claude-opus-4-8"


def test_get_strategy_picks_by_auth_mode():
    cfg = load_config()
    assert isinstance(get_strategy(cfg, "oauth"), OAuthSiblingStrategy)
    assert isinstance(get_strategy(cfg, "apikey"), DirectStrategy)
