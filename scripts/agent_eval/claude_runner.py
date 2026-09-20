#!/usr/bin/python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Claude Code runner with an external, network-free Bash tool sandbox."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

VENV = Path(os.environ["TREAMS_EVAL_VENV"]).resolve()
CLAUDE = Path("~/.local/bin/claude").resolve()
MODELS = ("claude-sonnet-5", "claude-opus-5")


def tool_command(trial, command):
    """Fresh root and namespaces; only this trial can be written or inspected."""
    args = [
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for directory in ("/usr",):
        args += ["--ro-bind", directory, directory]
    for directory in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(directory)
        if path.is_symlink():
            args += ["--symlink", os.readlink(path), directory]
        elif path.exists():
            args += ["--ro-bind", directory, directory]
    for filename in ("ld.so.cache", "localtime", "passwd", "group", "nsswitch.conf"):
        path = Path("/etc") / filename
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]
    python_link = VENV / "bin/python"
    python_roots = {
        python_link.resolve().parent.parent,
        python_link.readlink().parent.parent,
    }
    for path in sorted(python_roots | {VENV}):
        args += ["--ro-bind", str(path), str(path)]
    args += [
        "--bind",
        str(trial),
        str(trial),
        "--chdir",
        str(trial),
        "--clearenv",
        "--setenv",
        "PATH",
        f"{VENV}/bin:/usr/bin:/bin",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "LC_ALL",
        "C.UTF-8",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--setenv",
        "TMPDIR",
        str(trial / ".tmp"),
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        command,
    ]
    return args


def main():
    # The prefix is a trusted executable chosen by the parent CLI, outside the
    # tool's visible filesystem. Its trial binding cannot be changed by a tool.
    if os.environ.get("TREAMS_EVAL_CLAUDE_TOOL_PREFIX") == "1":
        if len(sys.argv) != 2:
            raise SystemExit("Expected exactly one wrapped shell command")
        trial = Path(os.environ["TREAMS_EVAL_CLAUDE_TRIAL"])
        os.execv("/usr/bin/bwrap", tool_command(trial, sys.argv[1]))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    trial = args.trial_dir.resolve(strict=True)
    (trial / ".tmp").mkdir(exist_ok=True)
    if not (args.model and args.prompt_file and args.log_file):
        parser.error("run requires --model, --prompt-file, and --log-file")
    original_user_dir = Path.home()
    claude_config = original_user_dir / ".claude"
    # OAuth refresh rotates this file atomically, so its parent must be writable.
    # Tool subprocesses never receive this directory in their independent root.
    outer = [
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--bind",
        "/tmp",
        "/tmp",
        "--bind",
        str(claude_config),
        str(claude_config),
        "--chdir",
        str(trial),
    ]
    for name in ("remote-settings.json", "policy-limits.json"):
        path = claude_config / name
        if path.exists():
            outer += ["--ro-bind", str(path), str(path)]
    fds = []
    for name in ("projects", "skills", "plugins", "commands", "agents"):
        path = claude_config / name
        if path.exists():
            outer += ["--tmpfs", str(path)]
    for name in ("CLAUDE.md", "settings.json", "history.jsonl"):
        path = claude_config / name
        if path.exists():
            fd = os.memfd_create("claude-eval-empty-config")
            os.write(fd, b"{}" if name.endswith(".json") else b"")
            os.lseek(fd, 0, os.SEEK_SET)
            fds.append(fd)
            outer += ["--ro-bind-data", str(fd), str(path)]
    # CLI global state may change in memory; the real user's file is untouched.
    global_state = original_user_dir / ".claude.json"
    fd = os.memfd_create("claude-eval-state")
    os.write(fd, global_state.read_bytes())
    os.lseek(fd, 0, os.SEEK_SET)
    fds.append(fd)
    outer += ["--bind-data", str(fd), str(global_state)]
    for name in (
        ".bashrc",
        ".bash_profile",
        ".bash_login",
        ".profile",
        ".zshrc",
        ".zprofile",
        ".zshenv",
    ):
        path = original_user_dir / name
        if path.exists():
            fd = os.memfd_create("claude-eval-empty-startup")
            fds.append(fd)
            outer += ["--ro-bind-data", str(fd), str(path)]

    env = {
        key: value
        for key, value in os.environ.items()
        if key
        in (
            "HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "TERM",
            "TZ",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "NO_PROXY",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        )
    }
    env.update(
        {
            "PATH": f"{VENV}/bin:/usr/bin:/bin",
            "TREAMS_EVAL_VENV": str(VENV),
            "CLAUDE_CODE_SHELL": "/bin/bash",
            "CLAUDE_CODE_SHELL_PREFIX": str(Path(__file__).resolve()),
            "TREAMS_EVAL_CLAUDE_TOOL_PREFIX": "1",
            "TREAMS_EVAL_CLAUDE_TRIAL": str(trial),
            "CLAUDE_CODE_TMPDIR": str(trial / ".tmp"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
        }
    )
    command = [
        *outer,
        str(CLAUDE),
        "--safe-mode",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--disable-slash-commands",
        "--no-chrome",
        "--no-session-persistence",
        "--tools",
        "Bash",
        "--allowedTools",
        "Bash",
        "--permission-mode",
        "dontAsk",
        "--model",
        args.model,
        "--effort",
        "medium",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    with args.log_file.open("xb") as log:
        prompt = args.prompt_file.read_bytes()
        result = subprocess.run(
            command,
            input=prompt,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            pass_fds=fds,
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
