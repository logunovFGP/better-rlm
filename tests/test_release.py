"""The release is manual by request, and that is the property worth pinning.

A trigger added to release.yml is invisible until the day a merge to main publishes a
release nobody asked for, so these assertions fail on any trigger other than
workflow_dispatch. The rest keeps the workflow honest about what it verifies: the same
command trunk-based.json calls the verify command, on both platforms, since every defect
found merging #2/#4/#5 was Windows-only and green on Linux.
"""
import json
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(wf: dict) -> dict:
    # PyYAML is YAML 1.1, where a bare `on` key parses as the boolean True, not "on".
    # Look under both so this keeps working if the file is ever quoted as "on":.
    return wf.get("on", wf.get(True))


def test_release_is_manual_only():
    assert set(_triggers(_workflow())) == {"workflow_dispatch"}, (
        "release.yml gained a non-manual trigger — a release must only happen by request"
    )


def test_verify_job_runs_the_repos_own_verify_command_on_both_platforms():
    wf = _workflow()
    verify = wf["jobs"]["verify"]
    expected = json.loads((ROOT / "trunk-based.json").read_text(encoding="utf-8"))["verify_command"]
    commands = [step.get("run", "") for step in verify["steps"]]
    assert any(expected in c for c in commands), (
        f"verify job does not run {expected!r} — trunk-based.json and CI have drifted"
    )
    assert set(verify["strategy"]["matrix"]["os"]) >= {"ubuntu-latest", "windows-latest"}


def test_release_job_never_interpolates_the_version_into_a_shell_line():
    # ${{ inputs.version }} inside a run block is textual substitution, so a crafted
    # dispatch input would execute. It must arrive as an environment variable instead.
    release = _workflow()["jobs"]["release"]
    assert release["env"]["VERSION"] == "${{ inputs.version }}"
    for step in release["steps"]:
        assert "${{ inputs.version }}" not in step.get("run", ""), step.get("name")


def test_project_version_is_plain_semver():
    # The workflow refuses anything else, and compares the input against this exact string.
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version


def test_release_reads_the_version_from_the_same_file_the_package_does():
    """The gate must read VERSION, not pyproject's [project] version.

    pyproject declares the version dynamically now, so ['project']['version'] raises
    KeyError -- the old read would have failed every release, and only at release time.
    """
    steps = _workflow()["jobs"]["release"]["steps"]
    gate = [s for s in steps if "VERSION file does not declare" in (s.get("name") or "")]
    assert gate, "the version-agreement gate is gone"
    assert "< VERSION" in gate[0]["run"]
    # The specific thing that would break: parsing a [project] version that no longer exists.
    assert "tomllib" not in gate[0]["run"]
    assert "['project']['version']" not in gate[0]["run"]


# --- PyPI publish -------------------------------------------------------------
# PyPI is the one irreversible step in this workflow: a version number burned there
# can never be reused, even after a yank. These pin the properties that keep an
# accidental or unauthenticated publish from happening.


def test_publish_job_runs_only_after_a_green_release():
    publish = _workflow()["jobs"]["publish"]
    assert publish["needs"] == "release", (
        "publish must depend on release — otherwise a failed verify still reaches PyPI"
    )


def test_publish_is_skipped_on_a_dry_run():
    # The whole point of dry_run is 'change nothing anywhere'. A publish step that
    # ignores it would push to PyPI from a run whose summary says it published nothing.
    assert "!inputs.dry_run" in _workflow()["jobs"]["publish"]["if"].replace(" ", "")


def test_publish_uses_trusted_publishing_and_stores_no_token():
    """OIDC, not a stored API token.

    A PyPI token in repo secrets is a long-lived credential that publishes as you
    from anywhere it leaks. Trusted Publishing mints a short-lived one per run, so
    the repository holds no publishing credential at all.
    """
    publish = _workflow()["jobs"]["publish"]
    assert publish["permissions"] == {"id-token": "write"}, (
        "publish needs exactly id-token: write — job permissions replace the "
        "workflow-level block, so anything extra is over-granting"
    )
    body = WORKFLOW.read_text(encoding="utf-8")
    assert "pypa/gh-action-pypi-publish" in body
    for forbidden in ("PYPI_API_TOKEN", "PYPI_TOKEN", "with:\n          password:"):
        assert forbidden not in body, f"{forbidden} — publishing must not use a stored token"


def test_publish_ships_the_artifacts_the_release_job_built():
    """GitHub and PyPI must receive byte-identical files.

    A publish job that runs its own `uv build` can produce a different wheel than the
    one attached to the GitHub release — same version, different bytes, and no way to
    tell afterwards which one a user installed.
    """
    steps = _workflow()["jobs"]["publish"]["steps"]
    assert any("download-artifact" in (s.get("uses") or "") for s in steps)
    assert not any("uv build" in (s.get("run") or "") for s in steps), (
        "publish rebuilds instead of reusing the release job's dist/"
    )
