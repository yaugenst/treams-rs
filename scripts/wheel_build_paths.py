# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Keep local build paths out of distributed wheels.

``rustflags`` prints a ``CARGO_ENCODED_RUSTFLAGS`` value that remaps every local
build root to ``/build``; it keeps any flags already set in the environment.
``check`` scans every member of the given wheels for paths beneath the same
roots and fails when one is found. Both commands derive the roots from one place: the current
home directory, ``CARGO_HOME``, ``RUSTUP_HOME``, a Cargo target directory set
through ``CARGO_TARGET_DIR`` or ``CARGO_BUILD_TARGET_DIR`` (build scripts embed
paths beneath it), the working directory and any ``--home`` given explicitly
(for example the home directory of a manylinux build container, whose Cargo and
rustup directories are covered as well).
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import cast

_REMAPPED_ROOT = "/build"
_SEPARATOR = "\x1f"
# A root leaks as a path beneath it, or whole as a C string such as a DWARF
# compilation directory, so a match must end at a separator, a NUL or the end of
# the member: "/root" must not match "src/roots.rs" or "/rootfs". Rust packs
# string constants without terminators, so any byte may precede a match.
_BOUNDARIES = ("/", "\\", "\x00")


def _specificity(root: str) -> tuple[int, str]:
    return len(root), root


def build_roots(homes: list[str]) -> list[str]:
    """Return the distinct local build roots, least specific first.

    rustc applies the last matching ``--remap-path-prefix``, so a more specific
    root (the checkout, Cargo's registry) must follow the home directory that
    contains it.
    """
    candidates: list[str] = []
    for home in [str(Path.home()), *homes]:
        candidates.extend((home, str(Path(home, ".cargo")), str(Path(home, ".rustup"))))
    candidates.extend(
        value
        for value in (os.environ.get("CARGO_HOME"), os.environ.get("RUSTUP_HOME"))
        if value
    )
    # Cargo resolves a relative target directory against the working directory.
    candidates.extend(
        str(Path(value).absolute())
        for value in (
            os.environ.get("CARGO_TARGET_DIR"),
            os.environ.get("CARGO_BUILD_TARGET_DIR"),
        )
        if value
    )
    candidates.append(str(Path.cwd()))
    roots: set[str] = set()
    for candidate in candidates:
        roots.add(candidate)
        roots.add(str(Path(candidate).resolve()))
    return sorted(roots, key=_specificity)


def encoded_rustflags(homes: list[str]) -> str:
    """Return ``CARGO_ENCODED_RUSTFLAGS`` with remaps for every build root."""
    if encoded := os.environ.get("CARGO_ENCODED_RUSTFLAGS"):
        flags = encoded.split(_SEPARATOR)
    else:
        flags = os.environ.get("RUSTFLAGS", "").split()
    flags.extend(
        f"--remap-path-prefix={root}={_REMAPPED_ROOT}" for root in build_roots(homes)
    )
    return _SEPARATOR.join(flag for flag in flags if flag)


def _needles(roots: list[str]) -> dict[bytes, tuple[str, tuple[bytes, ...]]]:
    """Map each encoded spelling of a root to the root and its boundaries."""
    needles: dict[bytes, tuple[str, tuple[bytes, ...]]] = {}
    for root in roots:
        variants = {root, root.replace("\\", "/"), root.replace("/", "\\")}
        variants |= {variant.replace("\\", "\\\\") for variant in variants}
        if len(root) > 1 and root[1] == ":":
            variants |= {variant[0].swapcase() + variant[1:] for variant in variants}
        for encoding in ("utf-8", "utf-16-le"):
            boundaries = tuple(boundary.encode(encoding) for boundary in _BOUNDARIES)
            for variant in variants:
                needles[variant.encode(encoding)] = (root, boundaries)
    return needles


def _contains_root(
    payload: bytes, needle: bytes, boundaries: tuple[bytes, ...]
) -> bool:
    start = payload.find(needle)
    while start >= 0:
        end = start + len(needle)
        if end == len(payload) or payload.startswith(boundaries, end):
            return True
        start = payload.find(needle, start + 1)
    return False


def find_leaks(wheel: Path, roots: list[str]) -> list[str]:
    """Return ``member: root`` for every wheel member containing a build root."""
    needles = _needles(roots)
    leaks: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            payload = archive.read(name)
            found = {
                root
                for needle, (root, boundaries) in needles.items()
                if _contains_root(payload, needle, boundaries)
            }
            leaks.extend(f"{name}: {root}" for root in sorted(found))
    return leaks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rustflags = commands.add_parser(
        "rustflags", help="Print CARGO_ENCODED_RUSTFLAGS without a trailing newline."
    )
    check = commands.add_parser("check", help="Fail if a wheel contains a build root.")
    check.add_argument("wheels", nargs="+", type=Path)
    for command in (rustflags, check):
        command.add_argument(
            "--home",
            action="append",
            default=[],
            help="Additional build home directory, including its .cargo and .rustup.",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print remapping flags or check wheels for leaked build roots."""
    args = _parser().parse_args(argv)
    homes = cast("list[str]", args.home)
    if args.command == "rustflags":
        sys.stdout.write(encoded_rustflags(homes))
        return 0
    roots = build_roots(homes)
    wheels = cast("list[Path]", args.wheels)
    leaks = [
        f"{wheel.name}: {leak}" for wheel in wheels for leak in find_leaks(wheel, roots)
    ]
    if leaks:
        print("Local build paths found in distributed files:", file=sys.stderr)
        print("\n".join(leaks), file=sys.stderr)
        return 1
    print(f"No local build paths in {len(wheels)} wheel(s); checked {', '.join(roots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
