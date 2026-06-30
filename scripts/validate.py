"""Phase-5 validation harness (no-API-key portions).

Synthesizes a >1M-token log, a k8s manifest set, and a small repo dump, then
exercises the server's load/inspect/chunk/exec tools to prove oversized inputs
are processed in the Docker sandbox WITHOUT the content flowing back into the
caller's context. LLM-routed tools (rlm_query / rlm_sub_query*) are listed but
require Claude auth (claude setup-token), so they are reported as gated here.

Usage:  python scripts/validate.py /tmp/rlm-val
"""
import os
import random
import re
import sys

random.seed(0)
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rlm-val"
LOG = os.path.join(SCRATCH, "val", "big.log")
K8S = os.path.join(SCRATCH, "val", "k8s")
REPO = os.path.join(SCRATCH, "val", "repo")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
os.makedirs(K8S, exist_ok=True)
os.makedirs(REPO, exist_ok=True)

# --- synthesize a large log spread across 24h ---
LEVELS = ["INFO"] * 70 + ["WARN"] * 20 + ["ERROR"] * 10
SVCS = ["auth", "payments", "gateway", "worker", "db", "cache"]
ERRS = ["connection reset by peer", "timeout waiting for upstream",
        "db deadlock detected", "nil pointer dereference",
        "rate limit exceeded", "disk quota exceeded"]
N = 80_000
with open(LOG, "w") as fh:
    for i in range(N):
        hh = (i * 24) // N
        lvl = random.choice(LEVELS)
        svc = random.choice(SVCS)
        msg = (random.choice(ERRS) if lvl == "ERROR"
               else ("request handled ok" if lvl == "INFO" else "elevated latency"))
        fh.write(f"2026-06-30T{hh:02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}Z "
                 f"{lvl} svc={svc} reqid={i:07d} {msg}\n")

# --- k8s manifests with deliberate misconfigs ---
open(os.path.join(K8S, "api.yaml"), "w").write(
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: api\nspec:\n  template:\n"
    "    spec:\n      hostNetwork: true\n      containers:\n      - name: api\n"
    "        image: nginx:latest\n        securityContext:\n          privileged: true\n")
open(os.path.join(K8S, "worker.yaml"), "w").write(
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: worker\nspec:\n  template:\n"
    "    spec:\n      containers:\n      - name: worker\n        image: worker:1.2.3\n"
    "        # no resources.limits set\n")
open(os.path.join(K8S, "db.yaml"), "w").write(
    "apiVersion: v1\nkind: Pod\nmetadata:\n  name: db\nspec:\n  containers:\n  - name: db\n"
    "    image: postgres:latest\n    securityContext:\n      runAsUser: 0\n")

# --- tiny repo dump (cross-file relationship) ---
open(os.path.join(REPO, "models.py"), "w").write("def make_user(name):\n    return {'name': name}\n")
open(os.path.join(REPO, "api.py"), "w").write("from models import make_user\n\ndef handler(n):\n    return make_user(n)\n")

import src.server as s  # noqa: E402


def show(title, out):
    print(f"\n========== {title} ==========")
    print(out)
    return out


def ctxid(tool_output):
    """Extract the ctx_id the load tool actually returned (do NOT rely on store ordering)."""
    m = re.search(r"ctx_[0-9a-f]{8}", tool_output)
    if not m:
        raise SystemExit(f"no ctx_id in tool output:\n{tool_output}")
    return m.group(0)


show("rlm_status", s.rlm_status())

# 1) LARGE LOG — load (in place, no copy), inspect, chunk
ctx = ctxid(show("rlm_load_file(big.log, log)", s.rlm_load_file(LOG, "log")))
show("rlm_inspect_context", s.rlm_inspect_context(ctx, 4))
show("rlm_chunk_context(lines, 5000)", s.rlm_chunk_context(ctx, "lines", 5000))

# 2) exec aggregations IN THE SANDBOX over the loaded `context` (no API key)
agg = r"""
import collections
errs = [l for l in context.splitlines() if ' ERROR ' in l]
by_hour = collections.Counter(l[11:13] for l in errs)
sig = collections.Counter(l.split('reqid=')[1][8:].strip() for l in errs)
print('total_lines      :', context.count(chr(10)))
print('ERROR lines      :', len(errs))
print('errors by hour   :', dict(sorted(by_hour.items())))
print('top error msgs   :', sig.most_common(3))
"""
show("rlm_exec(error patterns + by-hour aggregation)", s.rlm_exec(agg, ctx))

# 3) K8S manifests — load dir, exec misconfig scan
kctx = ctxid(show("rlm_load_context(k8s dir)", s.rlm_load_context(K8S, "dir")))
scan = r"""
issues = []
for marker in ['privileged: true', 'hostNetwork: true', ':latest', 'runAsUser: 0']:
    hits = context.count(marker)
    if hits:
        issues.append((marker, hits))
print('k8s misconfig hits:', issues)
print('files scanned     :', context.count('===== FILE:'))
"""
show("rlm_exec(k8s misconfig scan)", s.rlm_exec(scan, kctx))

# 4) REPO dump — load + cross-file structure via exec (rlm_query is the LLM path, gated)
rctx = ctxid(show("rlm_load_context(repo dir)", s.rlm_load_context(REPO, "dir")))
xfile = r"""
import re
defs = re.findall(r'def (\w+)', context)
imports = re.findall(r'from (\w+) import (\w+)', context)
print('functions defined :', defs)
print('cross-file imports:', imports)
"""
show("rlm_exec(repo cross-file scan)", s.rlm_exec(xfile, rctx))

# cleanup the sandbox container
if s._repl is not None:
    s._repl.close()

print("\n========== AUTH-GATED (require Claude Code OAuth token) ==========")
print(f"rlm_query({ctx}, 'Summarize the dominant error pattern and its peak hour')  -> Sonnet root + Haiku sub")
print(f"rlm_sub_query_batch({ctx}, 'List any error signatures in this chunk')        -> Haiku map-reduce")
import src.auth as _a
print("auth mode:", _a.auth_status(), " (run `claude setup-token`; set CLAUDE_CODE_OAUTH_TOKEN in .env)")
