# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Wheel-only Codex launcher adapted from the qualified Photonoodle campaign."""

import json
import os
import tempfile
from pathlib import Path

CODEX = Path("~/.local/bin/codex").resolve()
VENV = Path(os.environ["TREAMS_EVAL_VENV"]).resolve()


def launch_args(trial, model):
    trial = Path(trial).resolve()
    assert trial.is_dir()
    host = Path("~/.codex")
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
        Path("~/.agents/skills"),
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
        "~/.local/share/uv/python/cpython-3.13.1-linux-x86_64-gnu": "read",
        "~/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu": "read",
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
