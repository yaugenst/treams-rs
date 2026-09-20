# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check wheel-only model context and both command sandboxes without model calls."""

import json
import os
import shlex
import socket
import subprocess
from pathlib import Path

from claude_runner import tool_command
from runner import VENV, launch_args

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "benchmarks/results/agent-api-20260920/isolation"


def run(command, **kwargs):
    r = subprocess.run(command, text=True, capture_output=True, timeout=60, **kwargs)
    if r.returncode:
        raise RuntimeError(r.stderr or r.stdout)
    return r.stdout


def main():
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
            trial = Path("/tmp/treams-api-eval-preflight-20260920") / provider
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
                Path("~/personal/advect/pyproject.toml"),
                Path("~/.codex/auth.json"),
                Path("~/.claude/.credentials.json"),
                Path("~/.codex/AGENTS.md"),
                Path("~/.codex/memories/MEMORY.md"),
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
            for key, value in [
                ("FORBIDDEN", [str(p) for p in forbidden]),
                ("VENV_CONFIG", str(VENV / "pyvenv.cfg")),
                ("HOST_SOCKET", str(address)),
                ("HIDDEN", str(hidden)),
            ]:
                probe = probe.replace(key, repr(value))
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
                    "PONYTAIL",
                    "atlas-context",
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
