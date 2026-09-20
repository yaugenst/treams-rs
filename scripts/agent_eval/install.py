# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Install and byte-bind a frozen candidate wheel with baseline dependency versions."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--wheel", type=Path, required=True)
parser.add_argument("--env", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
wheel = args.output / args.wheel.name
shutil.copy2(args.wheel, wheel)
subprocess.run(
    ["uv", "venv", "--python", ".venv/bin/python", str(args.env)], check=True
)
python = args.env / "bin/python"
subprocess.run(
    [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        str(wheel) + "[advect]",
        "numpy==2.5.3",
        "advect==0.2.1",
        "array-api-compat==1.15.0",
    ],
    check=True,
)
site = Path(
    subprocess.check_output(
        [str(python), "-c", 'import sysconfig;print(sysconfig.get_paths()["purelib"])'],
        text=True,
    ).strip()
)
files = {}
with ZipFile(wheel) as archive:
    for name in archive.namelist():
        assert not any(
            p in name for p in ("AGENTS.md", "CLAUDE.md", "__pycache__", ".pyc")
        )
        if name.startswith("treams_rs/") and not name.endswith("/"):
            data = archive.read(name)
            assert (site / name).read_bytes() == data, name
            files[name] = hashlib.sha256(data).hexdigest()
patch = subprocess.check_output(
    ["git", "diff", "HEAD", "--", "python", "pyproject.toml"]
)
(args.output / "source.patch").write_bytes(patch)
binding = {
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "files": files,
    "packages": json.loads(
        subprocess.check_output(
            ["uv", "pip", "list", "--python", str(python), "--format", "json"],
            text=True,
        )
    ),
}
(args.output / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
print("Bound", len(files), "wheel files to", args.env)
