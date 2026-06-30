import dataclasses
from pathlib import Path

import pytest

import src.config as config_mod


@pytest.fixture
def cfg(tmp_path: Path):
    """Config with store/log dirs redirected to a tmp path."""
    base = config_mod.load_config()
    return dataclasses.replace(base, store_dir=tmp_path / "contexts", log_dir=tmp_path / "logs")
