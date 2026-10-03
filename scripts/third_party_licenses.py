# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate or check the third-party license bundle shipped in release wheels.

Release wheels statically link the normal (non-dev, non-build, non-proc-macro)
dependency closure of ``treams-py`` on every release target. This script writes
``THIRD_PARTY_LICENSES.txt`` from the license files of exactly those crate
releases in the local Cargo registry, and the Rust standard-library notices
from the sysroot of the pinned toolchain. It runs offline: fetch the crate
sources with ``cargo fetch --locked`` first.

``--check`` writes nothing and fails when a bundled file is stale, for example
after a ``Cargo.lock`` or toolchain update; rerun without ``--check`` to
refresh the files.

Text is normalised the way the repository's pre-commit hooks normalise every
file (LF line endings, no trailing whitespace, one final newline), so the
generated files stay stable under ``pre-commit run --all-files``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "THIRD_PARTY_LICENSES.txt"
STDLIB_COPYRIGHT = "RUST_STDLIB_COPYRIGHT.html"
STDLIB_MIT = "RUST_STDLIB_LICENSE_MIT.txt"
STDLIB_UNICODE = "RUST_STDLIB_LICENSE_UNICODE_3_0.txt"
# Release wheel targets; keep in sync with the wheel matrix in
# .github/workflows/native-wheels.yml. The crate and Cargo features wheels
# build come from [tool.maturin] in pyproject.toml (see wheel_build).
TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
SEPARATOR = "=" * 79
SUBSEPARATOR = "-" * 79

# Preferred license when a crate offers a choice, and a phrase that the
# license text must contain.
LICENSE_MARKERS = {
    "MIT": "permission is hereby granted, free of charge",
    "Apache-2.0": "apache license",
    "BSD-2-Clause": "redistribution and use in source and binary forms",
}
# License file names by preference (compared case-insensitively). Generic
# names are accepted only when their text matches the selected license.
LICENSE_FILES = {
    "MIT": ("license-mit", "license-mit.md", "license-mit.txt", "license.mit"),
    "Apache-2.0": (
        "license-apache",
        "license-apache.md",
        "license-apache.txt",
        "license-apache-2.0",
    ),
    "BSD-2-Clause": ("license-bsd", "license-bsd.md", "license-bsd.txt"),
}
GENERIC_LICENSE_FILES = ("license", "license.md", "license.txt", "licence")
# Notices a crate ships next to its license, such as the licenses of code it
# ports from other projects. Apache-2.0 requires forwarding NOTICE files.
EXTRA_NOTICE = re.compile(r"(notice.*|copying\..+|.*third[-_]?party.*)", re.IGNORECASE)

# Crate releases that ship no license file take their text from the upstream
# repository's license file at the commit the release was packaged from
# (``git.sha1`` in the release's .cargo_vcs_info.json). Each source names a
# vendored copy under UPSTREAM_DIR, verified identical to the upstream file at
# every listed commit, the upstream file's URL, and the crates packaged from
# each commit. A release from an unregistered commit fails, so a dependency
# update that brings one forces someone to re-verify the upstream license.
UPSTREAM_DIR = "third_party/licenses"
UPSTREAM_SOURCES = (
    (
        "equator.LICENSE",
        "https://github.com/sarah-ek/equator/blob/{commit}/LICENSE",
        {"9d4107bdd6a75598aed4a60fe46200c9e9f4e653": ("equator",)},
    ),
    (
        "faer.LICENSE",
        "https://codeberg.org/sarah-quinones/faer/src/commit/{commit}/LICENSE",
        {
            "0539947ffb757a739d7e703a7d2fa0c792a909c1": ("faer",),
            "8dfcceee8eb3d81231366acc9d3f157ac5e7057a": ("faer-traits",),
        },
    ),
    (
        "nano-gemm.LICENSE",
        "https://github.com/sarah-ek/nano-gemm/blob/{commit}/LICENSE",
        {
            "806b012a6f5ee0615042bf64675180fb97ff2ccc": ("nano-gemm",),
            "d5f1cd77f0af31fb90aa0d128a76e2366c5dabc7": (
                "nano-gemm-c32",
                "nano-gemm-c64",
                "nano-gemm-core",
                "nano-gemm-f32",
                "nano-gemm-f64",
            ),
        },
    ),
    (
        "pulp.LICENSE",
        "https://github.com/sarah-quinones/pulp/blob/{commit}/LICENSE",
        {"5eb07fd7b68edf0a5e19f71737d315f72a510295": ("pulp-wasm-simd-flag",)},
    ),
)
UPSTREAM_LICENSES = {
    (name, commit): (vendored, url.format(commit=commit))
    for vendored, url, releases in UPSTREAM_SOURCES
    for commit, names in releases.items()
    for name in names
}

HEADER = """\
TREAMS-RS THIRD-PARTY SOFTWARE NOTICES

treams-rs release wheels statically link the Rust crates listed below. Each
crate is distributed here under its MIT option, or under its only declared
license where it offers no MIT option. The text following each entry is
copied verbatim from that crate release; where a release ships no license
file, the entry names the upstream file the text was copied from.

The wheels also contain Rust {toolchain} standard-library code. Its official
copyright inventory and the applicable MIT and Unicode license texts ship
beside this file as RUST_STDLIB_COPYRIGHT.html,
RUST_STDLIB_LICENSE_MIT.txt, and RUST_STDLIB_LICENSE_UNICODE_3_0.txt.
"""


class BundleError(RuntimeError):
    """The bundle cannot be generated from the local sources."""


@dataclass(frozen=True)
class Crate:
    """One statically linked crate release."""

    name: str
    version: str
    license: str
    directory: Path
    license_file: str | None


def normalize(data: bytes) -> str:
    """Return text as the trailing-whitespace and end-of-file hooks leave it."""
    lines = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
    text = b"\n".join(line.rstrip() for line in lines).strip(b"\n")
    return text.decode("utf-8") + "\n"


def run(*args: str, root: Path) -> str:
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        command = " ".join(args)
        raise BundleError(
            f"`{command}` failed; run `cargo fetch --locked` first if crate "
            f"sources are missing.\n{result.stderr.strip()}"
        )
    return result.stdout


def wheel_build(root: Path) -> list[str]:
    """Return the Cargo arguments that select the crate and features wheels build.

    They mirror the [tool.maturin] options that decide which crates a wheel
    links: the manifest of the extension crate and its Cargo feature flags.
    """
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    maturin = pyproject.get("tool", {}).get("maturin", {})
    args = ["--manifest-path", str(root / maturin.get("manifest-path", "Cargo.toml"))]
    if maturin.get("all-features"):
        args.append("--all-features")
    if maturin.get("no-default-features"):
        args.append("--no-default-features")
    if features := maturin.get("features"):
        args += ["--features", ",".join(features)]
    return args


def linked_crates(root: Path) -> list[Crate]:
    """Return the crates.io releases a wheel links on any release target."""
    metadata = json.loads(
        run(
            "cargo",
            "metadata",
            "--locked",
            "--offline",
            "--all-features",
            "--format-version",
            "1",
            root=root,
        )
    )
    packages = {(p["name"], p["version"]): p for p in metadata["packages"]}
    build = wheel_build(root)
    linked: set[tuple[str, str]] = set()
    for target in TARGETS:
        tree = run(
            "cargo",
            "tree",
            "--locked",
            "--offline",
            *build,
            "--edges",
            "normal,no-proc-macro",
            "--target",
            target,
            "--prefix",
            "none",
            "--format",
            "{p}",
            root=root,
        )
        for line in tree.splitlines():
            name, version = line.split()[:2]
            linked.add((name, version.removeprefix("v")))
    crates = []
    for key in linked:
        meta = packages[key]
        if meta["source"] is None:
            continue  # workspace crate, covered by the project license
        if meta["source"] != CRATES_IO:
            raise BundleError(f"{key[0]} {key[1]} does not come from crates.io")
        if not meta["license"]:
            raise BundleError(f"{key[0]} {key[1]} declares no SPDX license")
        crates.append(
            Crate(
                name=meta["name"],
                version=meta["version"],
                license=meta["license"],
                directory=Path(meta["manifest_path"]).parent,
                license_file=meta["license_file"],
            )
        )
    return sorted(crates, key=crate_order)


def crate_order(crate: Crate) -> tuple[str, tuple[int, ...]]:
    core = crate.version.split("+")[0].split("-")[0]
    return crate.name, tuple(int(part) for part in core.split("."))


def distributed_license(crate: Crate) -> str:
    """Select the license under which the bundle redistributes ``crate``."""
    expression = crate.license.replace("/", " OR ")
    if "(" in expression or " AND " in expression or " WITH " in expression:
        raise BundleError(
            f"{crate.name} {crate.version}: extend scripts/third_party_licenses.py "
            f"to handle the license expression {crate.license!r}"
        )
    choices = [choice.strip() for choice in expression.split(" OR ")]
    if "MIT" in choices:
        return "MIT"
    if len(choices) == 1 and choices[0] in LICENSE_MARKERS:
        return choices[0]
    raise BundleError(
        f"{crate.name} {crate.version}: no supported license option in "
        f"{crate.license!r}; extend scripts/third_party_licenses.py"
    )


def matches(text: str, license_id: str) -> bool:
    return LICENSE_MARKERS[license_id] in " ".join(text.lower().split())


def license_text(
    crate: Crate, license_id: str, root: Path
) -> tuple[str, Path | None, str | None]:
    """Return the license text, its file in the crate, and any upstream origin."""
    files = {path.name.lower(): path for path in crate.directory.iterdir()}
    candidates = []
    if crate.license_file:
        candidates.append(crate.directory / crate.license_file)
    candidates += [files[n] for n in LICENSE_FILES[license_id] if n in files]
    candidates += [files[n] for n in GENERIC_LICENSE_FILES if n in files]
    for path in candidates:
        text = normalize(path.read_bytes())
        if matches(text, license_id):
            return text, path, None
    if candidates:
        raise BundleError(
            f"{crate.name} {crate.version}: no {license_id} text in "
            + ", ".join(path.name for path in candidates)
        )
    commit = release_commit(crate)
    if (crate.name, commit) not in UPSTREAM_LICENSES:
        raise BundleError(
            f"{crate.name} {crate.version} ships no license file; verify the "
            f"upstream license at commit {commit}, vendor it under "
            f"{UPSTREAM_DIR}/ and register {crate.name} under that commit in "
            "UPSTREAM_SOURCES"
        )
    vendored, origin = UPSTREAM_LICENSES[crate.name, commit]
    text = normalize((root / UPSTREAM_DIR / vendored).read_bytes())
    if not matches(text, license_id):
        raise BundleError(f"{UPSTREAM_DIR}/{vendored} is not a {license_id} text")
    return text, None, origin


def release_commit(crate: Crate) -> str:
    """Return the upstream commit a crate release was packaged from."""
    path = crate.directory / ".cargo_vcs_info.json"
    try:
        commit = json.loads(path.read_text(encoding="utf-8"))["git"]["sha1"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise BundleError(
            f"{crate.name} {crate.version} ships neither a license file nor "
            f"the commit it was packaged from ({path.name}); extend "
            "scripts/third_party_licenses.py"
        ) from error
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise BundleError(
            f"{crate.name} {crate.version}: unexpected git.sha1 {commit!r} in {path.name}"
        )
    return commit


def extra_notices(crate: Crate, license_file: Path | None) -> list[Path]:
    # All paths share one parent, so path order is file-name order.
    return sorted(
        path
        for path in crate.directory.iterdir()
        if path.is_file() and EXTRA_NOTICE.fullmatch(path.name) and path != license_file
    )


def entry(crate: Crate, root: Path) -> str:
    license_id = distributed_license(crate)
    text, license_file, origin = license_text(crate, license_id, root)
    notices = extra_notices(crate, license_file)
    lines = [
        SEPARATOR,
        f"{crate.name} {crate.version}",
        f"Source: https://crates.io/crates/{crate.name}/{crate.version}",
        f"Declared license: {crate.license}; distributed under {license_id}.",
    ]
    if origin is not None:
        lines.append(
            f"License text: the crate release ships none; copied from {origin}"
        )
    if notices:
        names = ", ".join(path.name for path in notices)
        lines.append(f"Additional notices shipped with this crate release: {names}.")
    lines += [SEPARATOR, "", text.rstrip("\n")]
    for path in notices:
        lines += [
            "",
            SUBSEPARATOR,
            f"{crate.name} {crate.version}: {path.name}",
            SUBSEPARATOR,
            "",
            normalize(path.read_bytes()).rstrip("\n"),
        ]
    return "\n".join(lines) + "\n"


def toolchain(root: Path) -> str:
    config = tomllib.loads((root / "rust-toolchain.toml").read_text(encoding="utf-8"))
    return config["toolchain"]["channel"]


def render_bundle(root: Path, header: str) -> str:
    crates = linked_crates(root)
    entries = [entry(crate, root) for crate in crates]
    return header.format(toolchain=toolchain(root)) + "\n" + "\n\n".join(entries)


def stdlib_notices(root: Path) -> dict[str, str]:
    """Return the standard-library notices of the pinned toolchain."""
    channel = toolchain(root)
    version = run("rustc", "--version", root=root).split()
    if version[:2] != ["rustc", channel]:
        raise BundleError(
            f"rustc reports {' '.join(version[:2])}, but rust-toolchain.toml "
            f"pins {channel}; run with the pinned toolchain"
        )
    docs = Path(run("rustc", "--print", "sysroot", root=root).strip())
    docs = docs / "share" / "doc" / "rust"
    sources = {
        STDLIB_COPYRIGHT: docs / "COPYRIGHT-library.html",
        STDLIB_UNICODE: docs / "licenses" / "Unicode-3.0.txt",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise BundleError("toolchain notices missing: " + ", ".join(missing))
    return {name: normalize(path.read_bytes()) for name, path in sources.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when a bundled file is stale",
    )
    args = parser.parse_args(argv)
    try:
        expected = {BUNDLE: render_bundle(ROOT, HEADER)}
        expected |= stdlib_notices(ROOT)
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    # rust-lang/rust ships the standard library's MIT text only in its source
    # tree (LICENSE-MIT at the release tag), not in the toolchain.
    if not (ROOT / STDLIB_MIT).is_file():
        print(f"error: {STDLIB_MIT} is missing", file=sys.stderr)
        return 1
    stale = [
        name
        for name, text in expected.items()
        if not (ROOT / name).is_file()
        or (ROOT / name).read_text(encoding="utf-8") != text
    ]
    if args.check:
        for name in stale:
            print(f"error: {name} is stale", file=sys.stderr)
        if stale:
            print(
                "Regenerate with: uv run --no-sync python scripts/third_party_licenses.py",
                file=sys.stderr,
            )
            return 1
        return 0
    for name in stale:
        (ROOT / name).write_text(expected[name], encoding="utf-8", newline="\n")
        print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
