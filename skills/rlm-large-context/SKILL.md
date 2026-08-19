---
name: rlm-large-context
description: Reason over an input too large to read directly — multi-MB/GB logs, big JSON/CSV/data exports, PDFs, repo or directory dumps, k8s manifest sets, document corpora — by routing it through the `rlm` MCP tools (load into the external store, then grep/exec/query) instead of reading it inline and blowing the context window. Use when a file is larger than ~200 KB or ~5,000 lines, when a Read/cat would be truncated, and ESPECIALLY for questions whose work grows with the input: "count/label/classify EVERY entry", "aggregate across the whole log", "which pairs contradict each other", "cross-reference all N records", "what's the overall picture across these 1000 documents", "summarize this whole directory or repo". Also for "find X across this huge <log|dump|dataset|dir>", and for LIVE system output — an hour of pod/container logs, a metrics or trace export, a journal, an audit feed — via `rlm_list_sources` / `rlm_load_source` wherever the deployment declares one. NOT for small files (read those directly) or for locating one symbol or definition in a code repo (grep/git grep is faster and free).
---

# RLM — reason over oversized contexts

The `rlm` MCP server holds content in an external on-disk store plus a sandboxed Docker
REPL and **returns only the findings** — a multi-GB log never enters this conversation.
Reach for these tools instead of reading a huge file.

## Route by the shape of the question, not the size of the file

Size decides *whether* to load. The question's complexity decides *which* tool.

| Question shape | Tool | Cost |
|---|---|---|
| "Where is X / does X appear" — one lookup | `rlm_grep(ctx_id, pattern)` | free, no model call |
| Exact counts, sums, buckets, parsing | `rlm_exec(code, ctx_id)` | free, no model call |
| "Label / classify / judge **every** entry"; aggregate over the whole input | `rlm_chunk_context` → `rlm_sub_query_batch` | one cheap call per chunk |
| One targeted semantic question, input under ~200K tokens | `rlm_sub_query(ctx_id, prompt)` | one cheap call |
| Cross-referencing, contradicting pairs, multi-hop over a corpus | `rlm_query(ctx_id, question)` | recursive loop, tens of seconds |
| Hardest reasoning | `rlm_query(..., model_override="opus")` | most expensive |

Start at the top. `rlm_grep` and `rlm_exec` answer more questions than expected and spend
no tokens on the content.

## What this is uniquely good at

From the RLM paper (Zhang, Kraska & Khattab, MIT CSAIL — *Recursive Language Models*),
which measured a median **+26% over context compaction** and **+13% over Claude Code**
across four long-context tasks at comparable cost:

- **Dense aggregation** — every element must be examined (label all, count by category,
  find the peak). Compaction fails here by design: it "presumes that some details that
  appear early in the prompt can safely be forgotten." Measured +28% over the base model.
- **Pairwise / cross-referencing** — work grows quadratically ("which entries conflict?").
  Unaided frontier models scored **≤0.1%** where the recursive path reached 58%.
- **Deep research over a corpus** — multi-hop across ~1000 documents (6–11M tokens).
- **Whole-repo or whole-directory understanding** — reasoning that spans many files at once
  (evaluated up to 4.2M tokens).

## Remote sources — URLs, Google Sheets, whole websites, a remote bundle

**Never pass a URL to `rlm_load_context`.** It has no HTTP support: the link would be
stored as ~60 bytes of *text* and reported as a successful load, so every later answer
would describe the link instead of the data. The tool now refuses URLs outright.

The sandbox has DNS + HTTPS egress with `requests` installed, and its `/workspace` is
bind-mounted to a host directory the store can read. So fetch inside the sandbox, then
load the file it wrote — after that every operation above works normally:

```
1. rlm_exec("import requests; open('/workspace/data','wb').write(requests.get(URL).content)")
2. rlm_exec("print(open('/workspace/owner').read())")   # 3rd line = host path of /workspace
3. rlm_load_file("<that host path>/data")               # -> ctx_id, then chunk/grep/query
```

| Source | How |
|---|---|
| **Google Sheet** | Fetch the export endpoint, not the share link: `https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=<GID>`. Works only if the sheet is link-shared. Several tabs = one fetch per `gid`. |
| **A remote `main.js`, bundle or single asset** | One `requests.get` on the raw URL. For minified JS, `rlm_grep` over the loaded context beats reading it. |
| **A whole website / many pages** | **Sitemap first, not nav-crawling.** Fetch `robots.txt`, follow its `Sitemap:` directives, expand the index, extract every `<loc>` — one `rlm_exec` regex, zero model calls. Measured on google.com: the homepage HTML exposes **17 anchors**, while its sitemap expands to **3,478 pages** (22 sub-sitemaps, 17.6 MB of XML — entries carry ~55 `hreflang` alternates each, so never read that raw). Crawl `<a href>` only if the site publishes no sitemap. Then fetch the pages you selected into ONE file under `/workspace` with `===== FILE: <url> (N bytes) =====` separators and chunk with `strategy="files"`. |
| **A paginated API** | Same loop; write JSON lines to `/workspace`, load once, then `rlm_exec` to parse. |
| **Anything private** | Pass the credential as a header inside the `rlm_exec` code. That code travels through this conversation, so use a short-lived token — never a long-lived password. |

Always sanity-check what you fetched: a private Sheet or an expired session returns an
HTML sign-in page with HTTP **200**. One free `rlm_grep` for `<title>` (or
`rlm_inspect_context`) catches it — otherwise you will analyse a login page as if it
were your data.

**The sandbox has `requests`, not a browser — no JavaScript runs.** For a JS app you get
the delivery shell, not the rendered frontend: google.com's homepage is 206 KB with **435
characters of visible text** and 41% inline JS. If you need the rendered DOM, render it
with a browser tool and save the result to a file, then load *that*. Do not try to make
this server scrape an SPA.

## Live sources — a running system, not a file

The biggest inputs usually are not files. They are what a system is emitting *right now*:
an hour of pod logs across a namespace, a metrics range query, a trace export, a journal,
an audit feed. All far past what fits in this conversation, and all exactly what the
routing table above is for.

`rlm_list_sources` shows what **this** deployment can pull in. The server ships none of its
own: an operator declares them in a registry outside the repo, so the list is site-specific
and already carries that host's tooling, endpoints and credentials.

```
1. rlm_list_sources()                                    # free — what exists here?
2. rlm_load_source("<name>", {"namespace": "...", ...})  # -> ctx_id; output stays on disk
3. route by the table above: rlm_grep / rlm_exec / chunk+batch / rlm_query
```

**Check `rlm_list_sources` before assuming you have to shell out.** When a source is
declared, prefer it: the output never passes through this conversation, the timeout and
byte cap are already set, and nothing is left behind in `/tmp`. When none is declared, the
ordinary route still works — run the command yourself with stdout **redirected to a file**,
then `rlm_load_file` that file. Never let a multi-MB log print into this conversation on
the way to the store.

Once loaded it is an ordinary context:

- *"Which errors dominate this window, and when did they start?"* → `rlm_exec` — free.
- *"Does OOMKilled appear anywhere today?"* → `rlm_grep` — free.
- *"Classify every 5xx in this window by root cause, then total them."* → `rlm_chunk_context` + `rlm_sub_query_batch`.
- *"Which services' error bursts line up with each other?"* → `rlm_query`.

**A partial load is worse than no load.** `rlm_load_source` labels a context
*WITH WARNINGS* when the command exited non-zero, timed out, hit its size cap, **or exited
0 with no output at all**. A truncated log answers "does X appear?" with a confident, wrong
**no** — so narrow the window, raise the cap, or state plainly that the answer covers only
part of the range. And an empty success is not a negative answer: a dead tunnel, a lapsed
session and a wrong selector all look exactly like "nothing matched". Co-verify with a query
that *must* return something before you report a zero. Same trap as the HTTP-200 sign-in
page above.

## What it is *not* for

- Small files — just `Read` them.
- Finding one symbol or definition in a repo — `grep`/`git grep` is faster and free. The
  paper's own result: single-needle tasks degrade to plain keyword heuristics, so paying for
  recursion buys nothing.
- **Code repositories specifically:** prefer `rlm_exec` over `rlm_query`. On the paper's
  code-repository benchmark the *no-sub-calling* variant beat every recursive variant, and
  an ordinary coding agent with context offloading scored higher still. Load the directory,
  then drive it with your own Python.

## Workflow

1. **Load** — `rlm_load_file(path, data_type=text|log|pdf)` for one file,
   `rlm_load_context(source, source_type=auto|text|file|dir)` for inline text, a file, or a
   whole directory, or `rlm_load_source(name, params)` for a live system (see above; list
   them first with `rlm_list_sources`). Returns a `ctx_id`; the content stays on disk.
2. **Sanity-check** (optional) — `rlm_inspect_context(ctx_id)` for metadata plus a head preview.
3. **Route** — pick from the table above.
4. **Housekeeping** — `rlm_list_contexts` to see what is loaded, `rlm_drop_context(ctx_id)`
   to evict one, `rlm_read_chunk(ctx_id, i)` to read exactly what a flagged chunk holds,
   `rlm_status` for mode, resolved models and Docker availability.

## Examples

- *"Most frequent error and its peak hour in this 800 MB log."*
  → `rlm_load_file(path)` → `rlm_exec` to bucket by hour — free, no model needed.
- *"Label every one of these 40k support tickets by root cause, then total them."*
  → `rlm_load_file(path)` → `rlm_chunk_context` → `rlm_sub_query_batch(ctx_id, "label each…")`
- *"Which of these 900 config entries contradict each other?"*
  → `rlm_load_context(dir)` → `rlm_query(ctx_id, "find contradicting pairs")`
- *"What do these 400 k8s manifests configure, and what is misconfigured?"*
  → `rlm_load_context(path, source_type="dir")` → `rlm_query(ctx_id, "summarize workloads, flag security misconfigurations")`
- *"Does OOMKilled appear in this huge log, and where?"*
  → `rlm_load_file(path)` → `rlm_grep(ctx_id, "OOMKilled")` — free.

## Notes

- Content never enters this chat — only the bounded findings (~4 KB for raw content, a
  larger bound for synthesized answers).
- `rlm_query` is a real multi-turn loop (tens of seconds) with a long-tailed cost
  distribution. For a single question prefer `rlm_sub_query`; for exact answers `rlm_exec`.
- Defaults: Sonnet 5 root, Opus 4.8 override, Haiku 4.5 sub. Auth reuses your Claude Code
  login via the `claude` CLI — no API key. Reported input-token counts on that path
  under-count the piped prompt, so treat its cost figures as a floor.
