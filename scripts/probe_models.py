"""Probe which Claude models the active auth (Claude Code OAuth or API key) can call.

Makes one minimal call per candidate and reports success + the model the API
resolved to, or the error. Use this to pick the right model per role under OAuth.
Usage:  python scripts/probe_models.py
"""
import re
import src.auth as auth

CANDIDATES = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-5",
    "claude-fable-5",
    "claude-3-5-haiku-20241022",
]


def main() -> None:
    print("auth mode:", auth.auth_status())
    c = auth.make_client()
    for m in CANDIDATES:
        try:
            r = c.messages.create(model=m, max_tokens=4,
                                  messages=[{"role": "user", "content": "Reply OK"}])
            print(f"OK    {m:30s} -> resolved={r.model}")
        except Exception as e:
            em = re.sub(r"sk-ant-[A-Za-z0-9_-]+", "sk-ant-…", str(e)).replace("\n", " ")[:150]
            print(f"FAIL  {m:30s} -> {type(e).__name__}: {em}")


if __name__ == "__main__":
    main()
