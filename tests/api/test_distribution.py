"""Installed distribution metadata and the module command."""

import subprocess
import sys
from importlib.metadata import version

import pytest

import treams_rs as tr

pytestmark = pytest.mark.interface


def test_version_matches_distribution_metadata():
    assert tr.__version__ == version("treams-rs")


def test_module_help_names_the_invocation():
    result = subprocess.run(
        [sys.executable, "-m", "treams_rs", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.startswith("usage: python -m treams_rs ")
