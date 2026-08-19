import dataclasses
import sys

import pytest

from src.context_store import ContextStore
from src.sources import DEFAULT_MAX_BYTES, DEFAULT_TIMEOUT_S, load_sources, resolve

PY = sys.executable  # never a shell builtin: these tests must pass on Windows too


def _registry(tmp_path, body: str):
    p = tmp_path / "sources.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# --- registry parsing ------------------------------------------------------
def test_missing_registry_is_empty_not_an_error(tmp_path):
    # The server ships no sources; an absent file is the normal state, not a fault.
    assert load_sources(tmp_path / "nope.yaml") == {}


def test_parses_entry_and_infers_params_in_order(tmp_path):
    reg = load_sources(_registry(tmp_path, """
workload-logs:
  description: Logs for one workload
  command: mycli logs -n {namespace} -l app={app} --since={since}
  timeout_s: 30
  max_bytes: 1024
"""))
    src = reg["workload-logs"]
    assert src.command == ("mycli", "logs", "-n", "{namespace}", "-l", "app={app}",
                           "--since={since}")
    assert src.params == ["namespace", "app", "since"]
    assert (src.timeout_s, src.max_bytes) == (30, 1024)


def test_defaults_applied_when_omitted(tmp_path):
    src = load_sources(_registry(tmp_path, "s:\n  command: echo hi\n"))["s"]
    assert src.timeout_s == DEFAULT_TIMEOUT_S
    assert src.max_bytes == DEFAULT_MAX_BYTES
    assert src.description == ""


@pytest.mark.parametrize("body", [
    "s: not-a-mapping\n",               # entry is a scalar
    "s:\n  description: no command\n",  # missing 'command'
    "s:\n  command: ''\n",              # empty command
    "- a\n- b\n",                       # top level is a list
    "s:\n  command: echo hi\n  timeout_s: 0\n",   # a zero bound is not "no bound"
    "s:\n  command: echo hi\n  max_bytes: 0\n",
])
def test_malformed_registry_raises_rather_than_serving_nothing(tmp_path, body):
    # Silently returning {} would look exactly like "none declared" and the operator
    # would never learn their file is being ignored.
    with pytest.raises(ValueError):
        load_sources(_registry(tmp_path, body))


# --- parameter substitution ------------------------------------------------
def test_params_cannot_inject_a_second_command(tmp_path):
    src = load_sources(_registry(tmp_path, "s:\n  command: mycli logs {app}\n"))["s"]
    argv = resolve(src, {"app": "web; rm -rf /"})
    # The whole hostile value stays ONE argv token — the command never sees a shell.
    assert argv == ["mycli", "logs", "web; rm -rf /"]


def test_missing_and_unknown_params_are_rejected(tmp_path):
    src = load_sources(_registry(tmp_path, "s:\n  command: mycli {a} {b}\n"))["s"]
    with pytest.raises(ValueError, match="missing"):
        resolve(src, {"a": "1"})
    # 'bb' is almost certainly a typo for 'b'; running with the literal "{b}" would
    # return plausible output about nothing.
    with pytest.raises(ValueError, match="unknown"):
        resolve(src, {"a": "1", "b": "2", "bb": "3"})


def test_env_expands_in_template_but_never_in_a_param(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_TEST_TOKEN", "s3cret")
    src = load_sources(_registry(
        tmp_path, 's:\n  command: mycli --auth "Bearer ${RLM_TEST_TOKEN}" --q {q}\n'))["s"]
    argv = resolve(src, {"q": "$RLM_TEST_TOKEN"})
    assert argv[2] == "Bearer s3cret"          # template may read the environment
    assert argv[-1] == "$RLM_TEST_TOKEN"       # a parameter value may not


# --- running a source ------------------------------------------------------
def test_load_command_streams_stdout_into_a_context(cfg):
    store = ContextStore(cfg)
    run = store.load_command(
        [PY, "-c", "print('a'); print('b')"],
        source="source:t", timeout_s=30, max_bytes=1 << 20)
    assert run.ok and run.returncode == 0
    assert run.meta.source_type == "command"
    assert run.meta.lines == 2
    assert store.read_text(run.meta.ctx_id).splitlines() == ["a", "b"]


def test_load_command_surfaces_failure_and_keeps_partial_output(cfg):
    store = ContextStore(cfg)
    run = store.load_command(
        [PY, "-c", "import sys; print('partial'); sys.stderr.write('boom\\n'); sys.exit(3)"],
        source="source:t", timeout_s=30, max_bytes=1 << 20)
    assert not run.ok and run.returncode == 3
    assert "boom" in run.stderr_tail
    # The output before the failure is still there — the caller decides what it's worth.
    assert "partial" in store.read_text(run.meta.ctx_id)


def test_merge_stderr_defaults_off_so_a_failure_message_stays_out_of_the_data(tmp_path):
    assert load_sources(_registry(tmp_path, "s:\n  command: echo hi\n"))["s"].merge_stderr is False


def test_merge_stderr_puts_stderr_logs_into_the_context(cfg, tmp_path):
    # postgres (and so `docker logs` on one) writes its LOGS to stderr, and the shell
    # answer 2>&1 does not exist here. Without this the store gets nothing at all.
    code = "import sys; sys.stderr.write('log line\\n'); sys.stdout.write('out line\\n')"
    store = ContextStore(cfg)

    split = store.load_command([PY, "-c", code], source="source:t",
                               timeout_s=30, max_bytes=1 << 20)
    assert store.read_text(split.meta.ctx_id) == "out line\n"
    assert "log line" in split.stderr_tail          # diagnostic tail, not content

    merged = store.load_command([PY, "-c", code], source="source:t",
                                timeout_s=30, max_bytes=1 << 20, merge_stderr=True)
    body = store.read_text(merged.meta.ctx_id)
    assert "log line" in body and "out line" in body
    assert merged.stderr_tail == ""                 # nothing left to tail

    on = load_sources(_registry(tmp_path, "s:\n  command: x\n  merge_stderr: true\n"))["s"]
    assert on.merge_stderr is True


def test_load_command_does_not_deadlock_on_large_stderr(cfg):
    # stderr on a pipe would fill its buffer while we drain only stdout, wedging both.
    store = ContextStore(cfg)
    run = store.load_command(
        [PY, "-c", "import sys; sys.stderr.write('e' * 500_000); print('done')"],
        source="source:t", timeout_s=30, max_bytes=1 << 20)
    assert run.ok
    assert "done" in store.read_text(run.meta.ctx_id)
    assert len(run.stderr_tail) <= 2000     # tail only, never the whole stream


def test_load_command_truncates_at_max_bytes(cfg):
    store = ContextStore(cfg)
    run = store.load_command(
        [PY, "-c", "import sys\nfor _ in range(200): sys.stdout.write('x' * 1000)"],
        source="source:t", timeout_s=30, max_bytes=5000)
    assert run.truncated
    assert run.meta.bytes == 5000


def test_load_command_kills_a_command_that_overruns_its_timeout(cfg):
    store = ContextStore(cfg)
    run = store.load_command(
        [PY, "-c", "import time; time.sleep(30)"],
        source="source:t", timeout_s=1, max_bytes=1 << 20)
    assert run.timed_out and not run.ok


# --- the rlm_load_source tool contract -------------------------------------
def _server(monkeypatch, cfg, tmp_path, body):
    import src.server as S
    monkeypatch.setattr(S, "CFG", dataclasses.replace(cfg, sources_file=_registry(tmp_path, body)))
    monkeypatch.setattr(S, "STORE", ContextStore(S.CFG))
    return S


def test_tool_reports_a_clean_load_without_warnings(monkeypatch, cfg, tmp_path):
    S = _server(monkeypatch, cfg, tmp_path, f"s:\n  command: {PY} -c \"print('data')\"\n")
    out = S.rlm_load_source("s")
    assert "## Source loaded\n" in out and "WARNING" not in out


def test_tool_flags_an_empty_success(monkeypatch, cfg, tmp_path):
    # A dead tunnel, a lapsed session and a wrong selector all exit 0 with no output and
    # look exactly like "nothing matched" — the most dangerous success there is.
    S = _server(monkeypatch, cfg, tmp_path, f"s:\n  command: {PY} -c pass\n")
    out = S.rlm_load_source("s")
    assert "WITH WARNINGS" in out and "EMPTY output" in out


def test_tool_errors_when_a_failed_command_produced_nothing(monkeypatch, cfg, tmp_path):
    S = _server(monkeypatch, cfg, tmp_path,
                f"s:\n  command: {PY} -c \"import sys; sys.exit(4)\"\n")
    out = S.rlm_load_source("s")
    assert out.startswith("ERROR:") and "exit code 4" in out
    assert S.STORE.list_ids() == []   # no empty context left behind to query later


def test_tool_rejects_an_undeclared_source(monkeypatch, cfg, tmp_path):
    S = _server(monkeypatch, cfg, tmp_path, f"s:\n  command: {PY} -c pass\n")
    assert S.rlm_load_source("../../etc/passwd").startswith("ERROR: unknown source")


def test_config_exposes_a_sources_file_path(cfg, tmp_path):
    # The registry lives outside the repo so a site's infrastructure never lands in a diff.
    assert dataclasses.replace(cfg, sources_file=tmp_path / "s.yaml").sources_file.name \
        == "s.yaml"
    from src.config import load_config
    assert load_config().sources_file.name == "sources.yaml"
