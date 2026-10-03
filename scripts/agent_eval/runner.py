# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Wheel-only Codex launcher: runs one attempt in a bubblewrap sandbox with global
instructions, skills and history masked. Configure with the environment variables
listed below.

TREAMS_EVAL_VENV        evaluation environment created by install.py (required).
                        It and its base Python installation (the parent of the
                        ``home`` entry of its pyvenv.cfg) are readable inside the
                        sandbox.
TREAMS_EVAL_CODEX       Codex executable; default: ``codex`` on PATH. The
                        directory two levels above the resolved executable (its
                        installation) is readable inside the sandbox, so it must
                        not contain CODEX_HOME, ~/.claude or other private data;
                        preflight.py fails when it exposes a forbidden path.
TREAMS_EVAL_READ_PATHS  further read-only paths for the sandbox, separated by
                        os.pathsep; default: none.
CODEX_HOME              Codex state directory; default: ~/.codex.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

CODEX = Path(
    os.environ.get("TREAMS_EVAL_CODEX")
    or shutil.which("codex")
    or sys.exit("set TREAMS_EVAL_CODEX or put codex on PATH")
).resolve()
VENV = Path(os.environ["TREAMS_EVAL_VENV"]).resolve()


def read_only_paths():
    """Base Python installation of VENV, resolved and as linked, then extra paths.

    pyvenv.cfg names the base interpreter directory under ``home``; uv often names
    it through a minor-version link to the patch-version directory, so both
    spellings are granted.
    """
    home = None
    for line in (VENV / "pyvenv.cfg").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "home":
            home = Path(value.strip())
    if home is None:
        sys.exit(f"{VENV / 'pyvenv.cfg'} has no home entry")
    extra = os.environ.get("TREAMS_EVAL_READ_PATHS", "").split(os.pathsep)
    return [home.resolve().parent, home.parent, *map(Path, filter(None, extra))]


def codex_home():
    """Codex state directory: CODEX_HOME, or ~/.codex when it is unset or empty."""
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def launch_args(trial, model):
    trial = Path(trial).resolve()
    assert trial.is_dir()
    host = codex_home()
    fds = []
    argv = [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--bind",
        str(host),
        str(host),
        "--bind",
        "/tmp",
        "/tmp",
    ]
    for name in ("AGENTS.md", "AGENTS.override.md", "config.toml"):
        if (host / name).exists():
            with tempfile.TemporaryFile() as empty:
                fd = os.dup(empty.fileno())
            fds.append(fd)
            argv += ["--ro-bind-data", str(fd), str(host / name)]
    for path in (
        host / "skills",
        Path.home() / ".agents" / "skills",
        host / "memories",
        host / "plugins",
    ):
        argv += ["--tmpfs", str(path), "--remount-ro", str(path)]
    argv += [str(CODEX)]
    for flag in (
        "plugins",
        "hooks",
        "shell_snapshot",
        "memories",
        "apps",
        "skill_search",
        "multi_agent",
        "browser_use",
        "browser_use_external",
        "computer_use",
        "image_generation",
        "remote_plugin",
        "workspace_dependencies",
        "recommended_plugins",
        "tool_suggest",
    ):
        argv += ["--disable", flag]
    filesystem = {
        ":minimal": "read",
        str(VENV): "read",
        **dict.fromkeys(map(str, read_only_paths()), "read"),
        str(CODEX.parent.parent): "read",
        str(trial): "write",
    }
    fs_toml = (
        "{"
        + ",".join(json.dumps(k) + "=" + json.dumps(v) for k, v in filesystem.items())
        + "}"
    )
    for setting in (
        "model=" + json.dumps(model),
        'model_reasoning_effort="medium"',
        'approval_policy="never"',
        "project_doc_max_bytes=0",
        'web_search="disabled"',
        "agents.enabled=false",
        'default_permissions="eval"',
        "permissions.eval.filesystem=" + fs_toml,
        "permissions.eval.network.enabled=false",
        "allow_login_shell=false",
        'shell_environment_policy.inherit="none"',
        'shell_environment_policy.set={PATH="' + str(VENV / "bin") + ':/usr/bin:/bin"}',
    ):
        argv += ["-c", setting]
    return argv, tuple(fds)
