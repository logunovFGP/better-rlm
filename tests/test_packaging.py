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


def test_the_windows_installer_accepts_every_python_the_package_allows():
    """pyproject and install.ps1 must agree on the supported range.

    They drifted: requires-python said <3.14 and the installer's ValidatePattern said
    (11|12|13), so both refused 3.14 -- a cap inherited from when the engine was a
    dependency shipping wheels, long after the engine became vendored source. The
    suite passes on 3.14.7, so the cap was fiction. The failure this guards is the
    asymmetric one: widening requires-python while the installer still rejects the
    version, so `pip install` works and `install.ps1 -PythonVersion 3.14` does not.
    """
    import re

    spec = PYPROJECT["project"]["requires-python"]
    lo, hi = re.search(r">=3\.(\d+)", spec), re.search(r"<3\.(\d+)", spec)
    assert lo and hi, f"expected a bounded >=3.x,<3.y range, got {spec!r}"
    allowed = {f"3.{n}" for n in range(int(lo.group(1)), int(hi.group(1)))}

    ps1 = (ROOT / "install.ps1").read_text(encoding="utf-8")
    pattern = re.search(r"ValidatePattern\('\^3\\\.\(([0-9|]+)\)\$'\)", ps1)
    assert pattern, "install.ps1 no longer validates -PythonVersion the expected way"
    accepted = {f"3.{n}" for n in pattern.group(1).split("|")}

    assert accepted == allowed, (
        f"pyproject allows {sorted(allowed)} but install.ps1 accepts {sorted(accepted)}"
    )


def test_a_vendored_opentui_wheel_exists_for_every_supported_python():
    """vendor/wheels must cover the whole requires-python range.

    opentui is supplied as a file rather than a dependency -- upstream 0.1.2 lacks
    the scrollbox extent fix and ships no cp314 wheel at all, and its sdist cannot
    self-build (CMakeLists.txt:91 wants the native library downloaded first). A
    file-based dependency has no resolver to notice a gap, so widening
    requires-python without building the matching wheel would leave that
    interpreter silently without a TUI. See vendor/wheels/README.md.
    """
    import re

    spec = PYPROJECT["project"]["requires-python"]
    lo, hi = re.search(r">=3\.(\d+)", spec), re.search(r"<3\.(\d+)", spec)
    assert lo and hi, f"expected a bounded >=3.x,<3.y range, got {spec!r}"
    needed = {f"cp3{n}" for n in range(int(lo.group(1)), int(hi.group(1)))}

    built = {
        w.name.split("-")[2]
        for w in (ROOT / "vendor" / "wheels").glob("opentui-*.whl")
    }

    assert needed <= built, (
        f"requires-python allows {sorted(needed)} but vendor/wheels has "
        f"{sorted(built)} -- rebuild per vendor/wheels/README.md"
    )


def test_the_vendored_wheels_are_one_version_and_not_plain_upstream():
    """A patched build must never be mistakable for upstream 0.1.2.

    The fix lives only in our fork, so the local version segment is the single
    signal that an installed opentui carries it. Mixed versions across the wheel
    set would mean one interpreter silently gets the unpatched binding.
    """
    versions = {
        w.name.split("-")[1] for w in (ROOT / "vendor" / "wheels").glob("opentui-*.whl")
    }
    assert len(versions) == 1, f"vendor/wheels mixes versions: {sorted(versions)}"
    version = versions.pop()
    assert "+" in version, (
        f"{version!r} has no PEP 440 local segment -- it is indistinguishable "
        "from an unpatched upstream build"
    )


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
    """Ambient assertions only on what every clone has.

    pyproject.toml and config.yaml are tracked, so they are present in any checkout
    including a fresh CI one. `.env` is NOT: it is gitignored and created by
    install.sh, so a clone that has never been installed has none and env_file()
    correctly falls back to ~/.rlm/.env. Asserting the repo path here passed on a
    developer machine and failed on every runner -- the same ambient-state trap that
    kept test_rlm_query_reports_a_limit_as_a_failed_tool_call red for weeks. The
    resolution rule is pinned hermetically in the two tests below instead.
    """
    assert cfgmod.IS_CHECKOUT is True
    assert cfgmod.config_file() == cfgmod.PKG_ROOT / "config.yaml"


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
