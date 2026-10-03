"""Conditional oracles must not exclude independent tests or disappear silently."""

import os
import runpy
import subprocess
import sys

import pytest

pytestmark = pytest.mark.interface


def test_shared_helpers_import_without_oracle(monkeypatch, pytestconfig):
    monkeypatch.setitem(sys.modules, "treams", None)
    helpers = runpy.run_path(str(pytestconfig.rootpath / "tests/_support.py"))
    helpers["assert_tree_allclose"]([1], [1])
    with pytest.raises(ModuleNotFoundError, match="treams"):
        helpers["to_oracle"](None)


@pytest.mark.parametrize("required", [False, True])
def test_conditional_oracle_collection(tmp_path, required, pytestconfig):
    tests = tmp_path / "tests"
    nested = tests / "api"
    nested.mkdir(parents=True)
    (tests / "conftest.py").write_text(
        (pytestconfig.rootpath / "tests/conftest.py").read_text()
    )
    comparison = ">" if required else "<"
    (tmp_path / "pyproject.toml").write_text(
        "[dependency-groups]\n"
        f"dev = [\"absent-release-oracle; python_version {comparison} '0'\"]\n"
    )
    # An ordinary shared helper defers the oracle import until it is used.
    (tests / "_helpers.py").write_text(
        "def oracle():\n    import absent_release_oracle\n"
    )
    (nested / "test_independent.py").write_text(
        "import pytest\n"
        "from _helpers import oracle\n"
        "from hypothesis import settings\n"
        "pytestmark = pytest.mark.interface\n"
        "def test_independent(category_markers):\n"
        "    assert 'interface' in category_markers\n"
        "    assert settings.default.max_examples == 100\n"
        "def test_lazy_oracle():\n"
        "    oracle()\n"
    )
    # A top-level oracle import in a root helper still excludes a nested module.
    (tests / "_eager.py").write_text("import absent_release_oracle\n")
    (nested / "test_eager.py").write_text(
        "import _eager\ndef test_oracle():\n    pass\n"
    )
    # The shared script loader imports repository scripts at module load time.
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/needs_oracle.py").write_text("import absent_release_oracle\n")
    (tests / "_scripts.py").write_text(
        "def load(name):\n    raise AssertionError('loader must not execute')\n"
    )
    (nested / "test_script.py").write_text(
        "from _scripts import load\nscript = load('needs_oracle')\n"
        "def test_script():\n    pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tests)],
        cwd=tmp_path,
        env={**os.environ, "HYPOTHESIS_PROFILE": "ci"},
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if required:
        assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
        assert "absent_release_oracle" in output
        assert "are not installed" in output
    else:
        assert result.returncode == pytest.ExitCode.OK, output
        assert "1 passed, 3 skipped" in output
        assert "2 module(s) not collected, 1 test(s) skipped" in output
