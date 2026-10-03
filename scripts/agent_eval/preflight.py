# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check wheel-only model context and both command sandboxes without model calls.

Sandboxed tools must not read the Codex credentials, instructions and memories
in CODEX_HOME (see runner.py) or the Claude Code credentials in ~/.claude.
Besides the variables of runner.py and claude_runner.py, this script reads:

TREAMS_EVAL_FORBIDDEN        further paths that sandboxed tools must not read,
                             separated by os.pathsep; default: none.
TREAMS_EVAL_CONTEXT_MARKERS  further strings that must not appear in the Codex
                             model context, such as a phrase unique to your
                             global instructions, separated by os.pathsep (so a
                             marker cannot contain it; any distinctive substring
                             works); default: none.
"""

import json
import os
import re
import shlex
import socket
import subprocess
from pathlib import Path

from claude_runner import tool_command
from runner import VENV, codex_home, launch_args

ROOT = Path(__file__).resolve().parents[2]
# Reruns never write into archived evidence; results/local/ is untracked.
OUTPUT = ROOT / "benchmarks/results/local/agent-api/isolation"


def run(command, **kwargs):
    r = subprocess.run(command, text=True, capture_output=True, timeout=60, **kwargs)
    if r.returncode:
        raise RuntimeError(r.stderr or r.stdout)
    return r.stdout


def fill(template, values):
    """Replace each placeholder name in template by the repr of its value.

    One pass, so a value that contains a placeholder name is never rewritten.
    """
    pattern = r"\b(" + "|".join(map(re.escape, values)) + r")\b"
    return re.sub(pattern, lambda match: repr(values[match[1]]), template)


def main():
    extra_forbidden = os.environ.get("TREAMS_EVAL_FORBIDDEN", "").split(os.pathsep)
    extra_markers = os.environ.get("TREAMS_EVAL_CONTEXT_MARKERS", "").split(os.pathsep)
    codex = codex_home()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hidden = OUTPUT / "hidden.txt"
    hidden.write_text("Existing evaluator canary; not an answer.\n")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = Path("/tmp/treams-eval-host-canary.sock")
    if address.exists():
        address.unlink()
    listener.bind(str(address))
    listener.listen()
    results = {}
    try:
        for provider in ("codex", "claude"):
            trial = Path("/tmp/treams-api-eval-preflight") / provider
            trial.mkdir(parents=True, exist_ok=True)
            (trial / ".tmp").mkdir(exist_ok=True)
            link = trial / "hidden-link"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(hidden)
            forbidden = [
                ROOT / "pyproject.toml",
                Path(__file__),
                hidden,
                link,
                codex / "auth.json",
                Path.home() / ".claude/.credentials.json",
                codex / "AGENTS.md",
                codex / "memories/MEMORY.md",
                *map(Path, filter(None, extra_forbidden)),
            ]
            probe = """import errno,json,os,pathlib,socket,treams_rs,advect
for name in FORBIDDEN:
    try:
        with open(name,'rb'): pass
    except (FileNotFoundError,PermissionError): pass
    else: raise AssertionError('forbidden path available: '+name)
for name in ('treams','scipy','pytest'):
    import importlib.util
    assert importlib.util.find_spec(name) is None,name
p=pathlib.Path('own.txt');p.write_text('persistent');assert p.read_text()=='persistent'
try: fd=os.open(VENV_CONFIG,os.O_WRONLY)
except OSError as e: assert e.errno in (errno.EROFS,errno.EPERM,errno.EACCES)
else:
    os.close(fd)
    raise AssertionError('environment writable')
for address in [('1.1.1.1',443),('127.0.0.1',443),('::1',443)]:
    try: s=socket.create_connection(address,timeout=1)
    except OSError: pass
    else:
        s.close()
        raise AssertionError('network available')
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try: s.connect(HOST_SOCKET)
except (FileNotFoundError,PermissionError): pass
else: raise AssertionError('host socket available')
finally: s.close()
for pid in pathlib.Path('/proc').iterdir():
    if pid.name.isdigit():
        try:
            with (pid/'root'/HIDDEN.lstrip('/')).open('rb'): pass
        except (FileNotFoundError,PermissionError): pass
        else: raise AssertionError('host proc root available')
print(json.dumps({'package':treams_rs.__file__,'forbidden_paths':len(FORBIDDEN),
    'network':'blocked','venv':'read-only','workspace':'read-write','oracle_packages':'absent'}))
"""
            probe = fill(
                probe,
                {
                    "FORBIDDEN": [str(p) for p in forbidden],
                    "VENV_CONFIG": str(VENV / "pyvenv.cfg"),
                    "HOST_SOCKET": str(address),
                    "HIDDEN": str(hidden),
                },
            )
            if provider == "codex":
                command, fds = launch_args(trial, "gpt-5.6-luna")
                try:
                    context = run(
                        [
                            *command,
                            "debug",
                            "prompt-input",
                            "No-model isolation check.",
                        ],
                        cwd=trial,
                        pass_fds=fds,
                    )
                finally:
                    for fd in fds:
                        os.close(fd)
                for marker in (
                    "<skills_instructions>",
                    "AGENTS.md instructions",
                    "MEMORY_SUMMARY",
                    *filter(None, extra_markers),
                ):
                    assert marker not in context, marker
                command, fds = launch_args(trial, "gpt-5.6-luna")
                try:
                    output = run(
                        [
                            *command,
                            "sandbox",
                            "--include-managed-config",
                            "-P",
                            "eval",
                            "-C",
                            str(trial),
                            str(VENV / "bin/python"),
                            "-c",
                            probe,
                        ],
                        cwd=trial,
                        pass_fds=fds,
                    )
                finally:
                    for fd in fds:
                        os.close(fd)
            else:
                output = run(tool_command(trial, shlex.join(["python", "-c", probe])))
            results[provider] = json.loads(output)
        (OUTPUT / "preflight.json").write_text(
            json.dumps(
                {"passed": True, "model_calls": 0, "providers": results}, indent=2
            )
            + "\n"
        )
        print(json.dumps(results))
    finally:
        listener.close()
        address.unlink()


if __name__ == "__main__":
    main()
