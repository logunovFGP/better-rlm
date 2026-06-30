# Phase 1 — RLM engine architecture note

Source studied: `github.com/alexzhang13/rlm` (pip `rlms==0.1.3`, MIT; Zhang, Kraska,
Khattab, MIT CSAIL). Every name below is from the cloned source, not memory.

## Public API — it's a class, not a module function
`from rlm.core.rlm import RLM` (`rlm/__init__.py`). The entry point is the method
`RLM(...).completion(prompt: str | dict, root_prompt: str | None = None) -> RLMChatCompletion`
(`rlm/core/rlm.py:326`). **The brief's `rlm.completion(...)` module function does
not exist** — and the base wrapper we forked already used the class form.

- `prompt` is the (possibly huge) **context**. `root_prompt` is the *small* thing the
  root LM is told to answer (e.g. the question). The big content is **not** inlined into
  the model's messages.
- Result fields (`rlm/core/types.py`): `.response` (final answer str), `.execution_time`,
  `.usage_summary` (a `UsageSummary` whose `.model_usage_summaries` is **keyed by model
  name** — how we prove Sonnet-vs-Haiku routing), `.root_model`, `.metadata`.

## Prompt-as-variable + REPL loop (`completion` → `_completion_turn`)
1. `_spawn_completion_context` builds an LM client + an environment and loads the context
   into the sandbox as the REPL variable **`context`** (`DockerREPL.load_context` →
   `add_context`, `rlm/environments/docker_repl.py:566,588`). The root LM's system prompt
   gets only `QueryMetadata` (context *lengths*), never the content.
2. Each iteration (`max_iterations`, default our config 20): the root LM emits ```python```
   blocks; `find_code_blocks` extracts them; `environment.execute_code(code)` runs them in
   the sandbox and returns a `REPLResult` (`stdout/stderr/locals/final_answer`).
3. The model signals completion by setting `answer["content"]` + `answer["ready"] = True`,
   surfaced as `REPLResult.final_answer`.

## Sub-LLM calls (the recursion)
Inside the REPL the model can call `llm_query` / `llm_query_batched` (single LM) and
`rlm_query` / `rlm_query_batched` (recursive RLM). These hit a **host-side proxy**; the
host's `LMHandler` routes them. Routing of root vs sub model:
- **Root** = `backend` + `backend_kwargs` (`model_name`).
- **Sub** = `other_backends=[...]` + `other_backend_kwargs=[...]` (engine allows exactly
  one extra backend, used at `depth==1`; `_subcall`, `rlm/core/rlm.py:706`). At
  `depth >= max_depth` a sub-call degrades to a plain LM completion on the sub model.

## Anthropic backend (`rlm/clients/anthropic.py`)
`get_client("anthropic", kwargs)` → `AnthropicClient(api_key, model_name, max_tokens=32768)`.
`api_key` is a **required explicit argument** (the client does not read the env itself), and
`_track_cost` records **tokens only, not USD** — so the engine's `max_budget` is a no-op on
Anthropic and we compute cost from token counts × known rates. Backends in the current
`get_client` literal: `openai, vllm, portkey, openrouter, vercel, anthropic, gemini,
azure_openai` — **no `litellm`** (that's the base wrapper's breakage).

## REPL sandboxes (`rlm/environments/__init__.py` `get_environment`)
`local, ipython, docker, modal, daytona, prime, e2b`. We use **docker** by default:
`DockerREPL(image="python:3.11-slim" by default; we pass "rlm-sandbox")`,
driven via the **docker CLI** (`docker run -d --rm -v <tmp>:/workspace --add-host
host.docker.internal:host-gateway ...`). The context is written to a file in the mounted
`/workspace` (str → `context_0.txt`, dict/list → JSON); sub-query calls reach the host at
`http://host.docker.internal:{proxy_port}/rlm_query[_batched]`, so **the API key stays
host-side and never enters the container**. The engine `pip install`s `dill`+`requests`
into the container at startup — we prebake them into `rlm-sandbox` to avoid runtime PyPI.
