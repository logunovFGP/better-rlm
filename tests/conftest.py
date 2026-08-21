import dataclasses
from pathlib import Path

import pytest

import src.config as config_mod


@pytest.fixture
def cfg(tmp_path: Path):
    """Config with store/log dirs redirected to a tmp path."""
    base = config_mod.load_config()
    return dataclasses.replace(base, store_dir=tmp_path / "contexts", log_dir=tmp_path / "logs")


@pytest.fixture(autouse=True)
def _no_test_starts_a_real_container(monkeypatch, request):
    """The suite must pass with no Docker daemon running — CI has none, and neither does
    a contributor who has not started Docker Desktop.

    DockerREPL.__init__ calls setup(), which runs `docker run`. A test that constructs one
    without stubbing that took the whole suite down with "failed to connect to the docker
    API" rather than failing by itself. This makes the requirement loud instead: the stub
    raises, so such a test fails alone, pointing at what to do. A test that genuinely wants
    a container marks itself @pytest.mark.docker; a test that only needs the object stubs
    setup itself, and its own monkeypatch replaces this one.
    """
    if "docker" in request.keywords:
        return
    from rlm.environments.docker_repl import DockerREPL

    def refuse(self):
        raise AssertionError(
            "this test constructs a real DockerREPL, whose __init__ starts a container. "
            "Stub setup (monkeypatch.setattr(dr.DockerREPL, 'setup', lambda self: None)) "
            "or mark the test @pytest.mark.docker if it truly needs a daemon."
        )

    monkeypatch.setattr(DockerREPL, "setup", refuse)
