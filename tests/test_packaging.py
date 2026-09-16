"""The properties that make this installable from PyPI rather than only from a checkout.

Publishing is irreversible per version, and the failure modes here are all silent:
a wheel that ships a top-level `src` package, an entry point naming a module that
moved, or a config lookup that reads the repo root and finds site-packages. Each one
installs cleanly and then does the wrong thing at runtime.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from better_rlm import config as cfgmod

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_distribution_is_named_for_the_command_it_installs():
    assert PYPROJECT["project"]["name"] == "better-rlm"
    assert list(PYPROJECT["project"]["scripts"]) == ["better-rlm"]


def test_entry_point_names_a_module_that_exists():
    target = PYPROJECT["project"]["scripts"]["better-rlm"]
    module, _, func = target.partition(":")
    assert (ROOT / Path(*module.split("."))).with_suffix(".py").is_file(), target
    mod = __import__(module, fromlist=[func])
    assert callable(getattr(mod, func))


def test_no_top_level_src_package_is_shipped():
    """`src` is the single worst top-level name to publish: every project that
    forgets `package-dir` claims it, and importing one shadows the others."""
    include = PYPROJECT["tool"]["setuptools"]["packages"]["find"]["include"]
    assert not any(p.startswith("src") for p in include), include
    assert not (ROOT / "src").exists(), "the src/ directory is back"


def test_author_identity_is_set_for_both_author_and_maintainer():
    project = PYPROJECT["project"]
    expected = "logunov.alexander.cs@gmail.com"
    assert project["authors"][0]["email"] == expected
    assert project["maintainers"][0]["email"] == expected


def test_license_declares_no_classifier_alongside_the_spdx_expression():
    # PEP 639: setuptools hard-errors when both are present, and it fails at BUILD
    # time — i.e. during a release, not during verify, unless something pins it here.
    assert PYPROJECT["project"]["license"] == "MIT"
    assert not any(c.startswith("License ::") for c in PYPROJECT["project"]["classifiers"])


# --- checkout vs installed ----------------------------------------------------


def test_this_checkout_is_detected_as_a_checkout():
    assert cfgmod.IS_CHECKOUT is True
    assert cfgmod.config_file() == cfgmod.PKG_ROOT / "config.yaml"
    assert cfgmod.env_file() == cfgmod.PKG_ROOT / ".env"


def test_installed_layout_falls_back_to_the_user_directory(monkeypatch, tmp_path):
    """With no pyproject.toml beside the package -- i.e. site-packages -- config and
    credentials come from ~/.rlm, the directory the tool already owns."""
    monkeypatch.setattr(cfgmod, "IS_CHECKOUT", False)
    monkeypatch.setattr(cfgmod, "USER_DIR", tmp_path)
    assert cfgmod.config_file() == tmp_path / "config.yaml"
    assert cfgmod.env_file() == tmp_path / ".env"


def test_checkout_without_a_config_file_still_falls_back(monkeypatch, tmp_path):
    # A fresh clone that never ran the installer has no config.yaml. Reading the
    # user copy beats reading nothing and silently applying defaults.
    monkeypatch.setattr(cfgmod, "PKG_ROOT", tmp_path / "repo")
    monkeypatch.setattr(cfgmod, "USER_DIR", tmp_path / "user")
    assert cfgmod.config_file() == tmp_path / "user" / "config.yaml"


def test_missing_config_yields_defaults_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(cfgmod, "IS_CHECKOUT", False)
    monkeypatch.setattr(cfgmod, "USER_DIR", tmp_path)
    assert cfgmod._load_yaml() == {}


def test_checkout_without_an_env_file_falls_back_to_the_user_copy(monkeypatch, tmp_path):
    """pip-install, `better-rlm auth` into ~/.rlm/.env, then clone to contribute.
    Without the fallback the token is silently ignored while config.yaml IS still
    read from ~/.rlm -- config looks right, first model call fails on transport."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(cfgmod, "IS_CHECKOUT", True)
    monkeypatch.setattr(cfgmod, "PKG_ROOT", repo)
    monkeypatch.setattr(cfgmod, "USER_DIR", tmp_path / "user")
    assert cfgmod.env_file() == tmp_path / "user" / ".env"
    # An .env in the checkout wins, even an empty one: that is an explicit choice.
    (repo / ".env").write_text("")
    assert cfgmod.env_file() == repo / ".env"


def test_config_and_env_resolve_by_the_same_rule(monkeypatch, tmp_path):
    # env_file()'s docstring says "same rule as config_file()". Assert it, so the
    # two cannot drift into the asymmetry they started with.
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(cfgmod, "IS_CHECKOUT", True)
    monkeypatch.setattr(cfgmod, "PKG_ROOT", repo)
    monkeypatch.setattr(cfgmod, "USER_DIR", tmp_path / "user")
    for resolve, name in ((cfgmod.config_file, "config.yaml"), (cfgmod.env_file, ".env")):
        assert resolve().parent == tmp_path / "user", name
        (repo / name).write_text("")
        assert resolve().parent == repo, name


def test_user_dir_is_the_existing_rlm_directory():
    """Not a new ~/.config/better-rlm store — the same ~/.rlm that already holds
    contexts, logs, cache and the budget ledger. CLAUDE.md bans a second one."""
    assert cfgmod.USER_DIR.name == ".rlm"
    assert str(cfgmod.USER_DIR).startswith(str(Path.home()))
