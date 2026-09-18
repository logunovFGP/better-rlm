"""
Token counting and model context limits for compaction and context sizing.

Uses tiktoken for OpenAI-style models when available; otherwise estimates
with ~4 characters per token.
"""

from typing import Any

# Default context limit when model is unknown (tokens)
DEFAULT_CONTEXT_LIMIT = 128_000

# Characters per token when tokenizer is unavailable (conservative estimate)
CHARS_PER_TOKEN_ESTIMATE = 4

# Model context limits (max input context in tokens).
# Match: key contained in model_name (e.g. "gpt-4o" matches "@openai/gpt-4o").
# Longest matching key wins.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI (GPT-5: 272k input, 128k reasoning+output)
    "gpt-5-nano": 272_000,
    "gpt-5": 272_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-2024": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-32k": 32_768,
    "gpt-4": 8_192,
    "gpt-3.5-turbo-16k": 16_385,
    "gpt-3.5-turbo": 16_385,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o1": 200_000,
    # Anthropic. Claude 4/5 ids added by better-rlm: without them every current
    # model fell through to DEFAULT_CONTEXT_LIMIT (128k) while a 2024 id resolved
    # correctly, so _should_compact fired at 8x less context than the model has.
    # Where a limit is not documented in this repo, the CONSERVATIVE 200k is used:
    # under-reading only compacts earlier, over-reading overruns the window.
    # better_rlm/describe.py::MODELS now documents real windows, and five ids here
    # (opus-4-7/-4-6, sonnet-4-6, sonnet-4-5 and its dated build) still read 200k
    # against a documented 1M. Left as-is deliberately: raising them makes the engine
    # compact LATER, which is a behaviour change to argue on its own evidence, not a
    # side effect of adding a model picker.
    "claude-sonnet-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-fable-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-2.1": 200_000,
    "claude-2": 100_000,
    # Gemini
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.0-pro": 30_720,
    # Qwen (Alibaba)
    "qwen3-max": 256_000,
    "qwen3-72b": 128_000,
    "qwen3-32b": 128_000,
    "qwen3-8b": 32_768,
    "qwen3": 128_000,
    # Kimi (Moonshot)
    "kimi-k2.5": 262_000,
    "kimi-k2-0905": 256_000,
    "kimi-k2-thinking": 256_000,
    "kimi-k2": 128_000,
    "kimi": 128_000,
    # GLM (Zhipu)
    "glm-4.6": 200_000,
    "glm-4-9b": 1_000_000,
    "glm-4": 128_000,
    "glm": 128_000,
    # MiniMax, added by better-rlm alongside its model catalogue. Reached over the
    # Anthropic messages protocol at api.minimax.io/anthropic, so these ids arrive
    # here like any other. Two entries cover all seven: longest-key-wins substring
    # matching means "MiniMax-M2" also answers for M2.1/M2.5/M2.7 and the
    # -highspeed variants, which all share the 204_800 window.
    "MiniMax-M3": 1_048_576,
    "MiniMax-M2": 204_800,
}


def get_context_limit(model_name: str) -> int:
    """
    Return max context size in tokens for a model.

    Matches when the dict key is contained in model_name (e.g. "gpt-4o" matches
    "@openai/gpt-4o"). Longest matching key wins. Falls back to DEFAULT_CONTEXT_LIMIT
    for unknown models.
    """
    if not model_name or model_name == "unknown":
        return DEFAULT_CONTEXT_LIMIT
    exact = MODEL_CONTEXT_LIMITS.get(model_name)
    if exact is not None:
        return exact
    best = 0
    best_limit = DEFAULT_CONTEXT_LIMIT
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in model_name and len(key) > best:
            best = len(key)
            best_limit = limit
    return best_limit


def _count_tokens_tiktoken(messages: list[dict[str, Any]], model_name: str) -> int | None:
    """Count tokens with tiktoken if available. Returns None on failure."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    total = 0
    # Approximate OpenAI message format overhead per message
    tokens_per_message = 3
    tokens_per_name = 1
    for m in messages:
        total += tokens_per_message
        content = m.get("content")
        if isinstance(content, str):
            total += len(enc.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(enc.encode(part.get("text", "") or ""))
        elif content is not None and content != "":
            total += len(enc.encode(str(content)))
        if m.get("name"):
            total += tokens_per_name
    return total


def count_tokens(messages: list[dict[str, Any]], model_name: str) -> int:
    """
    Count tokens in a list of message dicts (role, content).

    Uses tiktoken for OpenAI-style models when the package is available;
    otherwise estimates with character length / CHARS_PER_TOKEN_ESTIMATE.
    """
    if not messages:
        return 0
    if model_name and model_name != "unknown":
        n = _count_tokens_tiktoken(messages, model_name)
        if n is not None:
            return n
    # Fallback: count chars (stringify in case content is not str, e.g. list)
    total_chars = 0
    for m in messages:
        raw = m.get("content", "") or ""
        total_chars += len(raw) if isinstance(raw, str) else len(str(raw))
    return (total_chars + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE
