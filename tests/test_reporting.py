"""What ran, and what authenticated it — reported rather than assumed.

Two failure modes this covers. A sandbox container outlives an image rebuild and
nothing detected it (observed: a 3.11 container serving after the image moved to
3.13, with rlm_status printing only the image NAME). And the model that ANSWERS is
not always the one asked for: on OAuth, models.select maps a configured id to its
closest subscription sibling, so the only way to know is to read it back.
"""

import types

import pytest

import src.server as srv
import src.subquery as sq
import src.transport as tp
from src.sandbox_reap import container_image_status


def _docker(monkeypatch, images_out, inspect_out):
    def run(argv, **kw):
        out = images_out if argv[1] == "images" else inspect_out
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
    monkeypatch.setattr("src.sandbox_reap.subprocess.run", run)


def test_container_on_the_current_image_reads_current(monkeypatch):
    _docker(monkeypatch, "sha256:aaaa111122223333\n", "sha256:aaaa111122223333")
    assert container_image_status("img", "cid").startswith("current")


def test_container_from_an_older_build_is_reported_stale(monkeypatch):
    """The whole point: nothing else in the system notices this."""
    _docker(monkeypatch, "sha256:bbbb444455556666\n", "sha256:aaaa111122223333")
    msg = container_image_status("img", "cid")
    assert msg.startswith("STALE")
    assert "aaaa11112222" in msg and "bbbb44445555" in msg
    assert "reconnect" in msg


def test_no_container_is_not_an_error(monkeypatch):
    assert container_image_status("img", None) == "none created yet"


def test_docker_failure_degrades_to_unknown(monkeypatch):
    def boom(*a, **k):
        raise OSError("no docker")
    monkeypatch.setattr("src.sandbox_reap.subprocess.run", boom)
    assert container_image_status("img", "cid").startswith("unknown")


# --- auth label ---------------------------------------------------------------
def test_auth_label_names_the_method_and_is_cached(monkeypatch):
    monkeypatch.setattr(tp, "_AUTH_LABEL", None)
    monkeypatch.setattr("src.auth.resolve_auth_mode", lambda cfg: "oauth")
    calls = []

    def status(cfg):
        calls.append(1)
        return {"loggedIn": True, "authMethod": "oauth_token"}

    monkeypatch.setattr(tp, "cli_auth_status", status)
    cfg = srv.CFG
    assert tp.auth_label(cfg) == "oauth (claude CLI, oauth_token)"
    tp.auth_label(cfg)
    tp.auth_label(cfg)
    assert len(calls) == 1, "auth_label must not pay the 215ms subprocess per call"


def test_auth_label_says_so_when_the_login_is_dead(monkeypatch):
    monkeypatch.setattr(tp, "_AUTH_LABEL", None)
    monkeypatch.setattr("src.auth.resolve_auth_mode", lambda cfg: "oauth")
    monkeypatch.setattr(tp, "cli_auth_status", lambda cfg: {"loggedIn": False})
    assert "NOT LOGGED IN" in tp.auth_label(srv.CFG)


def test_auth_label_on_the_sdk_path_never_shells_out(monkeypatch):
    monkeypatch.setattr(tp, "_AUTH_LABEL", None)
    monkeypatch.setattr("src.auth.resolve_auth_mode", lambda cfg: "apikey")
    monkeypatch.setattr(tp, "cli_auth_status",
                        lambda cfg: pytest.fail("no CLI is involved on the API-key path"))
    assert tp.auth_label(srv.CFG) == "apikey (anthropic SDK)"


# --- the model that actually answered ----------------------------------------
def test_sub_result_carries_the_model_the_transport_reported(monkeypatch, cfg):
    monkeypatch.setattr(sq, "_call", lambda *a, **k: ("hi", 1, 2, "claude-haiku-4-5"))
    assert sq.sub_query(cfg, "p", "asked-for-id").model == "claude-haiku-4-5"
    assert sq.sub_query_batch(cfg, ["a", "b"], "asked-for-id", concurrency=1)[0].model == \
        "claude-haiku-4-5"


def test_rlm_status_does_not_shadow_the_transport_module(monkeypatch):
    """rlm_status had a LOCAL named `transport`, which shadowed the imported module
    and made transport.auth_label() raise 'str has no attribute auth_label' at
    runtime -- invisible to every unit test that did not call the tool."""
    monkeypatch.setattr(tp, "_AUTH_LABEL", "test-label")
    monkeypatch.setattr(srv, "_container_status", lambda: "test-container")
    out = srv.rlm_status()
    assert not out.lstrip().startswith("ERROR"), out[:200]
    assert "test-label" in out and "test-container" in out
