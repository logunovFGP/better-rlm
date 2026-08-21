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
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version
