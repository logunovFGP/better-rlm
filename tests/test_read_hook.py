"""The oversized-read PreToolUse hook, and its settings.json registrar.

Two things are worth pinning here, and they pull in opposite directions.

The hook must BLOCK when redirecting genuinely helps - that is the whole feature,
and the skill cannot be selected on data size by any other mechanism. But a global
PreToolUse hook that blocks wrongly breaks every Read on the machine, in every
project, including the bounded Read that is the correct *end* of the RLM workflow.
So every fail-open rung gets its own case: a hook that blocks too eagerly is worse
than no hook, because that failure is loud, constant, and unrelated to this repo.

The registrar edits the user's entire Claude Code configuration. Its tests are
about not destroying that file.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _set_home(monkeypatch, home) -> None:
    """Point os.path.expanduser at a throwaway HOME on every platform.

    Both vars, because expanduser reads HOME on POSIX and USERPROFILE on Windows.
    Patching only HOME left the hook reading the developer's REAL ~/.claude.json:
    on a machine that has rlm registered the fail-open cases blocked and failed,
    and the detection cases passed for the wrong reason.
    """
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _load(name: str, relpath: str):
    """Both live outside the package (a hook is invoked by path, not imported)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hook = _load("big_read_to_rlm", "hooks/big-read-to-rlm.py")
registrar = _load("install_hook", "scripts/install_hook.py")


@pytest.fixture
def big(tmp_path):
    """A file comfortably over the hook's threshold."""
    p = tmp_path / "huge.log"
    p.write_text("x" * (hook.LIMIT + 1), encoding="utf-8")
    return p


@pytest.fixture
def rlm_home(tmp_path, monkeypatch):
    """A HOME whose ~/.claude.json registers rlm at user scope."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"rlm": {}}}), encoding="utf-8")
    _set_home(monkeypatch, home)
    return home


# --- the block, which is the point ------------------------------------------


def test_big_file_with_rlm_present_is_blocked_and_told_where_to_go(big, rlm_home, tmp_path):
    code, msg = hook.decide({"tool_input": {"file_path": str(big)}}, cwd=str(tmp_path))
    assert code == 2
    assert "rlm_load_file" in msg and "rlm-large-context" in msg
    assert str(big) in msg, "the model cannot act on a message that omits the path"


# --- every fail-open rung ----------------------------------------------------


def test_small_file_is_never_touched(tmp_path, rlm_home):
    small = tmp_path / "small.log"
    small.write_text("x" * 10, encoding="utf-8")
    assert hook.decide({"tool_input": {"file_path": str(small)}}, cwd=str(tmp_path)) == (0, "")


def test_missing_file_falls_through_rather_than_guessing(tmp_path, rlm_home):
    ghost = tmp_path / "does-not-exist.log"
    assert hook.decide({"tool_input": {"file_path": str(ghost)}}, cwd=str(tmp_path)) == (0, "")


def test_no_file_path_at_all_falls_through(rlm_home, tmp_path):
    assert hook.decide({}, cwd=str(tmp_path)) == (0, "")
    assert hook.decide({"tool_input": {}}, cwd=str(tmp_path)) == (0, "")


def test_bounded_read_of_a_big_file_is_allowed(big, rlm_home, tmp_path):
    """rlm_grep hands back line numbers and the caller Reads that slice. Blocking
    it would make the hook fight the workflow it exists to promote."""
    code, _ = hook.decide(
        {"tool_input": {"file_path": str(big), "offset": 900, "limit": 40}}, cwd=str(tmp_path)
    )
    assert code == 0


def test_media_and_archives_are_not_redirected(tmp_path, rlm_home):
    """The context store holds text. Sending a 5 MB PNG or a tarball to
    rlm_load_file points the caller at something that cannot render it."""
    for name in ("shot.png", "paper.pdf", "notes.ipynb", "dump.tar.gz", "lib.dylib"):
        p = tmp_path / name
        p.write_bytes(b"\0" * (hook.LIMIT + 1))
        code, _ = hook.decide({"tool_input": {"file_path": str(p)}}, cwd=str(tmp_path))
        assert code == 0, f"{name} was redirected to a text store"


def test_never_blocks_when_rlm_is_absent(big, tmp_path, monkeypatch):
    """Blocking with no alternative would break large Reads for anyone who
    installed the skill but not the server, or removed the server later."""
    home = tmp_path / "bare"
    home.mkdir()
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"other": {}}}), encoding="utf-8")
    _set_home(monkeypatch, home)
    assert hook.decide({"tool_input": {"file_path": str(big)}}, cwd=str(tmp_path)) == (0, "")


def test_unreadable_config_never_blocks(big, tmp_path, monkeypatch):
    home = tmp_path / "broken"
    home.mkdir()
    (home / ".claude.json").write_text("{ not json", encoding="utf-8")
    _set_home(monkeypatch, home)
    assert hook.decide({"tool_input": {"file_path": str(big)}}, cwd=str(tmp_path)) == (0, "")


# --- rlm detection across every scope `claude mcp add` can write -------------


def test_rlm_found_at_local_scope(tmp_path, monkeypatch):
    """--scope local keys the server under the project directory, not the top
    level. A hook that only knew the user scope would silently never fire."""
    home, proj = tmp_path / "h", tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    (home / ".claude.json").write_text(
        json.dumps({"projects": {str(proj): {"mcpServers": {"rlm": {}}}}}), encoding="utf-8"
    )
    _set_home(monkeypatch, home)
    assert hook.rlm_configured(str(proj)) is True


def test_rlm_found_in_a_committed_mcp_json_above_the_cwd(tmp_path, monkeypatch):
    """--scope project writes .mcp.json at the repo root; the read may happen from
    a subdirectory, so the probe walks up."""
    home, repo = tmp_path / "h", tmp_path / "repo"
    home.mkdir()
    (repo / "src" / "deep").mkdir(parents=True)
    (home / ".claude.json").write_text("{}", encoding="utf-8")
    (repo / ".mcp.json").write_text(json.dumps({"mcpServers": {"rlm": {}}}), encoding="utf-8")
    _set_home(monkeypatch, home)
    assert hook.rlm_configured(str(repo / "src" / "deep")) is True


# --- the registrar: do not destroy the user's settings -----------------------


OTHERS = {
    "model": "opus",
    "permissions": {"allow": ["Bash(git status)"]},
    "hooks": {
        "PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "python3 /x/other.py"}]}
        ]
    },
}
HOOKPATH = "/opt/claude/hooks/big-read-to-rlm.py"


@pytest.fixture
def settings(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(OTHERS), encoding="utf-8")
    return p


def _entries(p):
    return json.loads(p.read_text(encoding="utf-8")).get("hooks", {}).get("PreToolUse", [])


def test_registering_twice_leaves_one_entry(settings):
    """Without this, every `./install.sh --hook` appends a duplicate and the hook
    runs N times per Read."""
    assert registrar.apply(str(settings), HOOKPATH) == "  registered"
    assert "already registered" in registrar.apply(str(settings), HOOKPATH)
    assert sum(HOOKPATH in json.dumps(e) for e in _entries(settings)) == 1


def test_registering_preserves_everything_else(settings):
    registrar.apply(str(settings), HOOKPATH)
    got = json.loads(settings.read_text(encoding="utf-8"))
    assert got["model"] == "opus"
    assert got["permissions"] == OTHERS["permissions"]
    assert any("other.py" in json.dumps(e) for e in _entries(settings)), "clobbered another hook"


def test_removal_is_idempotent_and_leaves_other_hooks_alone(settings):
    registrar.apply(str(settings), HOOKPATH)
    assert "removed 1" in registrar.apply(str(settings), HOOKPATH, remove=True)
    assert "nothing to remove" in registrar.apply(str(settings), HOOKPATH, remove=True)
    assert any("other.py" in json.dumps(e) for e in _entries(settings))
    assert json.loads(settings.read_text(encoding="utf-8"))["model"] == "opus"


def test_removal_matches_a_hand_edited_command(settings):
    """An entry edited to `timeout 5 python3.12 <path>` is still ours. Leaving it
    behind strands a hook pointing at a file uninstall just deleted, which fails
    every Read on the machine."""
    raw = json.loads(settings.read_text(encoding="utf-8"))
    raw["hooks"]["PreToolUse"].append(
        {
            "matcher": "Read",
            "hooks": [{"type": "command", "command": f"timeout 5 python3.12 {HOOKPATH}"}],
        }
    )
    settings.write_text(json.dumps(raw), encoding="utf-8")
    assert "removed 1" in registrar.apply(str(settings), HOOKPATH, remove=True)
    assert not any(HOOKPATH in json.dumps(e) for e in _entries(settings))


def test_unparseable_settings_is_refused_not_overwritten(tmp_path):
    """The alternative is replacing a file holding the user's whole configuration
    with one containing only our hook."""
    p = tmp_path / "settings.json"
    p.write_text("{ truncated", encoding="utf-8")
    with pytest.raises(SystemExit):
        registrar.apply(str(p), HOOKPATH)
    assert p.read_text(encoding="utf-8") == "{ truncated"


def test_missing_settings_file_is_created(tmp_path):
    p = tmp_path / "nested" / "settings.json"
    assert registrar.apply(str(p), HOOKPATH) == "  registered"
    assert len(_entries(p)) == 1


def test_installer_and_uninstaller_agree_on_the_hook():
    """install.sh registers it; uninstall.sh must unregister it. test_uninstall.py's
    artefact table covers the token, this covers the round trip."""
    sh = (ROOT / "install.sh").read_text(encoding="utf-8")
    un = (ROOT / "uninstall.sh").read_text(encoding="utf-8")
    assert "install_hook.py" in sh and "install_hook.py" in un
    assert "--remove" in un, "uninstall.sh calls the registrar without --remove"


# --- the script owns the whole lifecycle, so install.sh holds one call --------


def test_place_copies_the_shipped_hook_and_makes_it_executable(tmp_path):
    dst = tmp_path / "hooks" / "big-read-to-rlm.py"
    assert "installed" in registrar.place(str(dst))
    assert dst.is_file()
    # Windows has no exec bit for os.chmod to set; the copy is the whole story there.
    if os.name != "nt":
        assert dst.stat().st_mode & 0o111, "copied but not executable"
    assert dst.read_text(encoding="utf-8") == Path(registrar.SOURCE).read_text(encoding="utf-8")


def test_place_resolves_its_source_from_the_repo_not_the_cwd(monkeypatch, tmp_path):
    """install.sh must not have to know the layout, and a worktree checkout has a
    different cwd than the repo root."""
    monkeypatch.chdir(tmp_path)
    assert Path(registrar.SOURCE).is_file()


def test_place_removal_is_idempotent(tmp_path):
    dst = tmp_path / "hooks" / "big-read-to-rlm.py"
    registrar.place(str(dst))
    assert "removed" in registrar.place(str(dst), remove=True)
    assert "nothing to remove" in registrar.place(str(dst), remove=True)
    assert not dst.exists()


def test_dry_run_changes_nothing(tmp_path, capsys):
    """uninstall.sh --dry-run delegates here; printing '[dry-run]' while actually
    deleting would be the worst possible bug in an uninstaller."""
    dst = tmp_path / "hooks" / "big-read-to-rlm.py"
    cfg = tmp_path / "settings.json"
    registrar.place(str(dst))
    registrar.apply(str(cfg), str(dst))
    before = cfg.read_text(encoding="utf-8")

    registrar.main(["--remove", "--dry-run", "--hook", str(dst), "--settings", str(cfg)])
    assert "[dry-run]" in capsys.readouterr().out
    assert dst.is_file(), "dry-run deleted the hook"
    assert cfg.read_text(encoding="utf-8") == before, "dry-run edited settings.json"


def test_uninstall_unregisters_before_deleting(tmp_path, capsys):
    """Order matters: a settings entry pointing at a deleted file fails every Read.
    Asserting on output order is how we pin it without mocking the filesystem."""
    dst = tmp_path / "hooks" / "big-read-to-rlm.py"
    cfg = tmp_path / "settings.json"
    registrar.main(["--hook", str(dst), "--settings", str(cfg)])
    capsys.readouterr()

    registrar.main(["--remove", "--hook", str(dst), "--settings", str(cfg)])
    out = capsys.readouterr().out
    assert out.index("removed 1 hook entry") < out.index(f"removed {dst}")
    assert not dst.exists() and not _entries(cfg)


def test_probe_accepts_the_shipped_hook():
    """The hook must run under bare `python3` -- that is the command settings.json gets."""
    import scripts.install_hook as ih

    assert ih.probe(ih.SOURCE) is None


def test_probe_rejects_a_hook_the_interpreter_cannot_run(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("this is not python(\n", encoding="utf-8")
    import scripts.install_hook as ih

    assert ih.probe(str(bad)) is not None


def test_a_hook_that_cannot_run_is_never_registered(tmp_path, monkeypatch):
    """settings.json must stay untouched: a registered-but-crashing hook prints a
    traceback on every Read in every project on the machine."""
    import scripts.install_hook as ih

    settings = tmp_path / "settings.json"
    settings.write_text('{"model": "opus"}', encoding="utf-8")
    monkeypatch.setattr(ih, "SOURCE", str(tmp_path / "missing.py"))

    rc = ih.main(["--settings", str(settings), "--hook", str(tmp_path / "h.py")])

    assert rc == 1
    assert json.loads(settings.read_text(encoding="utf-8")) == {"model": "opus"}
    assert not (tmp_path / "h.py").exists()
