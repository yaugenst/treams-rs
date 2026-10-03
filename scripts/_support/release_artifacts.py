"""Validate and describe the complete treams-rs release artifact set.

The expected wheel family, packaged license files and source layout are derived
from ``pyproject.toml`` and ``Cargo.toml`` so release checks cannot drift from the
package declarations: CPython tags come from the ``Programming Language :: Python
:: 3.N`` classifiers, license files from ``license-files``, the Python source root
from ``[tool.maturin]`` and the Rust manifests from the Cargo workspace members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import tomllib
import zipfile
from dataclasses import asdict, dataclass
from email.parser import BytesParser
from itertools import product
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from email.message import Message

_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = cast(
    "dict[str, object]",
    tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8")),
)
_CARGO = cast(
    "dict[str, object]",
    tomllib.loads((_ROOT / "Cargo.toml").read_text(encoding="utf-8")),
)
_PROJECT = cast("dict[str, list[str]]", _PYPROJECT["project"])
_MATURIN = cast("dict[str, dict[str, str]]", _PYPROJECT["tool"])["maturin"]
_WORKSPACE = cast("dict[str, list[str]]", _CARGO["workspace"])

DISTRIBUTION = "treams-rs"
_WHEEL_DISTRIBUTION = "treams_rs"
_PACKAGE = "treams_rs"
_PYTHON_TAGS = tuple(
    f"cp3{match[1]}"
    for classifier in _PROJECT["classifiers"]
    if (
        match := re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", classifier)
    )
)
_PLATFORM_FAMILIES = (
    "linux-x86_64",
    "linux-aarch64",
    "macos-x86_64",
    "macos-arm64",
    "windows-x86_64",
)
_REQUIRED_LICENSE_FILES = frozenset(_PROJECT["license-files"])
_EXPECTED_WHEELS = frozenset(
    (python_tag, python_tag, platform)
    for python_tag, platform in product(_PYTHON_TAGS, _PLATFORM_FAMILIES)
)
_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")
_WHEEL_NAME_PARTS = 5
_REQUIRED_PACKAGE_FILES = frozenset({f"{_PACKAGE}/_native.pyi", f"{_PACKAGE}/py.typed"})
_NATIVE_MODULE = re.compile(rf"{_PACKAGE}/_native\.[^/]+\.(?:so|pyd)")
# Development-only files that must never ship in a wheel.
_FORBIDDEN_WHEEL_BASENAMES = frozenset({"AGENTS.md"})
_REQUIRED_SOURCE_FILES = frozenset(
    {
        "Cargo.lock",
        "Cargo.toml",
        "PKG-INFO",
        "pyproject.toml",
        *(f"{member}/Cargo.toml" for member in _WORKSPACE["members"]),
        *_REQUIRED_LICENSE_FILES,
        *(f"{_MATURIN['python-source']}/{name}" for name in _REQUIRED_PACKAGE_FILES),
    }
)
# Repository tooling and evidence that building from source does not need.
# formal/golden is read only by #[cfg(test)] Rust code, never by a package build.
_EXCLUDED_SOURCE_DIRECTORIES = frozenset(
    {".github", "benchmarks", "docs", "formal", "scripts", "tests"}
)


class ReleaseArtifactError(ValueError):
    """Raised when a candidate distribution set is incomplete or inconsistent."""


@dataclass(frozen=True)
class ArtifactRecord:
    """Immutable provenance for one candidate distribution."""

    filename: str
    kind: str
    sha256: str
    size: int
    python_tag: str | None = None
    abi_tag: str | None = None
    platform_family: str | None = None


def _artifact_filename(record: ArtifactRecord) -> str:
    return record.filename


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def cargo_version() -> str:
    """Return the single package version authority from ``Cargo.toml``."""
    package = cast("dict[str, str]", _WORKSPACE["package"])
    return package["version"]


def _platform_family(platform_tag: str) -> str:
    tags = set(platform_tag.split("."))
    if tags & {"manylinux_2_17_x86_64", "manylinux2014_x86_64"}:
        return "linux-x86_64"
    if tags & {"manylinux_2_17_aarch64", "manylinux2014_aarch64"}:
        return "linux-aarch64"
    if platform_tag.startswith("macosx_") and platform_tag.endswith("_x86_64"):
        return "macos-x86_64"
    if platform_tag.startswith("macosx_") and platform_tag.endswith("_arm64"):
        return "macos-arm64"
    if platform_tag == "win_amd64":
        return "windows-x86_64"
    message = f"unsupported release wheel platform tag: {platform_tag}"
    raise ReleaseArtifactError(message)


def _parse_wheel_name(path: Path) -> tuple[str, str, str, str]:
    parts = path.name.removesuffix(".whl").split("-")
    if len(parts) != _WHEEL_NAME_PARTS:
        message = f"release wheel name has an unexpected shape: {path.name}"
        raise ReleaseArtifactError(message)
    distribution, version, python_tag, abi_tag, platform_tag = parts
    if distribution != _WHEEL_DISTRIBUTION:
        message = f"release wheel has the wrong distribution name: {path.name}"
        raise ReleaseArtifactError(message)
    return version, python_tag, abi_tag, platform_tag


def _check_wheel_contents(path: Path, *, version: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                message = f"{path.name} must contain exactly one METADATA file"
                raise ReleaseArtifactError(message)
            metadata_name = metadata_names[0]
            license_root = metadata_name.removesuffix("METADATA") + "licenses/"
            packaged_licenses = {
                name.removeprefix(license_root)
                for name in names
                if name.startswith(license_root)
            }
            if missing := sorted(_REQUIRED_LICENSE_FILES - packaged_licenses):
                message = f"{path.name} is missing packaged license files: {', '.join(missing)}"
                raise ReleaseArtifactError(message)
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
    except zipfile.BadZipFile as error:
        message = f"{path.name} is not a readable wheel"
        raise ReleaseArtifactError(message) from error

    declared_licenses = set(metadata.get_all("License-File") or ())
    if missing := sorted(_REQUIRED_LICENSE_FILES - declared_licenses):
        message = f"{path.name} metadata is missing License-File entries: {', '.join(missing)}"
        raise ReleaseArtifactError(message)
    if missing := sorted(_REQUIRED_PACKAGE_FILES - names):
        message = f"{path.name} is missing required package files: {', '.join(missing)}"
        raise ReleaseArtifactError(message)
    native_modules = [name for name in names if _NATIVE_MODULE.fullmatch(name)]
    if len(native_modules) != 1:
        message = f"{path.name} must contain exactly one native extension module"
        raise ReleaseArtifactError(message)
    if forbidden := sorted(
        name for name in names if PurePosixPath(name).name in _FORBIDDEN_WHEEL_BASENAMES
    ):
        message = f"{path.name} contains development-only files: {', '.join(forbidden)}"
        raise ReleaseArtifactError(message)
    _check_metadata_identity(metadata, path=path, version=version)


def _check_metadata_identity(metadata: Message, *, path: Path, version: str) -> None:
    if (
        _normalized(str(metadata["Name"])) != DISTRIBUTION
        or metadata["Version"] != version
    ):
        message = f"{path.name} metadata does not identify {DISTRIBUTION} {version}"
        raise ReleaseArtifactError(message)


def _validate_wheel(path: Path, *, version: str) -> ArtifactRecord:
    filename_version, python_tag, abi_tag, platform_tag = _parse_wheel_name(path)
    if filename_version != version:
        message = f"{path.name} has version {filename_version}, expected {version}"
        raise ReleaseArtifactError(message)
    platform_family = _platform_family(platform_tag)
    _check_wheel_contents(path, version=version)
    return ArtifactRecord(
        filename=path.name,
        kind="wheel",
        sha256=_sha256(path),
        size=path.stat().st_size,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_family=platform_family,
    )


def _validate_sdist(path: Path, *, version: str) -> ArtifactRecord:
    expected_name = f"{_WHEEL_DISTRIBUTION}-{version}.tar.gz"
    if path.name != expected_name:
        message = f"source distribution is {path.name}, expected {expected_name}"
        raise ReleaseArtifactError(message)

    root = expected_name.removesuffix(".tar.gz")
    try:
        with tarfile.open(path, "r:gz") as archive:
            names = set(archive.getnames())
            if f"{root}/PKG-INFO" not in names:
                message = (
                    f"{path.name} is missing required source files: {root}/PKG-INFO"
                )
                raise ReleaseArtifactError(message)
            stream = archive.extractfile(f"{root}/PKG-INFO")
            if stream is None:
                message = f"{path.name} has an unreadable PKG-INFO"
                raise ReleaseArtifactError(message)
            metadata = BytesParser().parsebytes(stream.read())
    except tarfile.TarError as error:
        message = f"{path.name} is not a readable source distribution"
        raise ReleaseArtifactError(message) from error

    required = {f"{root}/{name}" for name in _REQUIRED_SOURCE_FILES}
    if missing := sorted(required - names):
        message = f"{path.name} is missing required source files: {', '.join(missing)}"
        raise ReleaseArtifactError(message)
    excluded = {f"{root}/{name}" for name in _EXCLUDED_SOURCE_DIRECTORIES}
    if unexpected := sorted(
        name
        for name in names
        if any(name == prefix or name.startswith(f"{prefix}/") for prefix in excluded)
    ):
        message = (
            f"{path.name} contains non-source repository files: {', '.join(unexpected)}"
        )
        raise ReleaseArtifactError(message)
    _check_metadata_identity(metadata, path=path, version=version)

    return ArtifactRecord(
        filename=path.name,
        kind="sdist",
        sha256=_sha256(path),
        size=path.stat().st_size,
    )


def check_source_build(sdist: Path, wheel: Path, *, version: str) -> None:
    """Validate a source distribution and a wheel rebuilt from it.

    The rebuilt wheel targets the build host rather than a release platform, so
    only its contents and metadata are checked, not its platform family.
    """
    _validate_sdist(sdist, version=version)
    filename_version = _parse_wheel_name(wheel)[0]
    if filename_version != version:
        message = f"{wheel.name} has version {filename_version}, expected {version}"
        raise ReleaseArtifactError(message)
    _check_wheel_contents(wheel, version=version)


def assemble_release_artifacts(
    dist_dir: Path,
    *,
    version: str,
    source_revision: str,
    manifest_path: Path,
    checksums_path: Path,
) -> list[ArtifactRecord]:
    """Validate a complete release set and write its immutable provenance."""
    if _SOURCE_REVISION.fullmatch(source_revision) is None:
        message = "source revision must be one full lowercase Git commit SHA"
        raise ReleaseArtifactError(message)
    if not _EXPECTED_WHEELS:
        message = (
            "pyproject.toml declares no CPython classifiers "
            "('Programming Language :: Python :: 3.N'), so no wheel family is defined"
        )
        raise ReleaseArtifactError(message)

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(sdists) != 1:
        message = f"release set must contain exactly one sdist, found {len(sdists)}"
        raise ReleaseArtifactError(message)

    records = [_validate_wheel(path, version=version) for path in wheels]
    actual_wheels = {
        (record.python_tag, record.abi_tag, record.platform_family)
        for record in records
    }
    if actual_wheels != _EXPECTED_WHEELS or len(records) != len(_EXPECTED_WHEELS):
        missing = sorted(_EXPECTED_WHEELS - actual_wheels)
        unexpected = sorted(actual_wheels - _EXPECTED_WHEELS)
        message = (
            "release wheel family mismatch; "
            f"missing={missing}, unexpected={unexpected}, wheels={len(records)}"
        )
        raise ReleaseArtifactError(message)

    records.append(_validate_sdist(sdists[0], version=version))
    records.sort(key=_artifact_filename)

    manifest = {
        "schema_version": 1,
        "package": DISTRIBUTION,
        "package_version": version,
        "source_revision": source_revision,
        "artifacts": [asdict(record) for record in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text(
        "".join(f"{record.sha256}  {record.filename}\n" for record in records),
        encoding="utf-8",
    )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the complete treams-rs release set and write provenance files."
    )
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--checksums-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run release artifact assembly from command-line arguments."""
    args = _parser().parse_args(argv)
    records = assemble_release_artifacts(
        cast("Path", args.dist_dir),
        version=cast("str", args.version),
        source_revision=cast("str", args.source_revision),
        manifest_path=cast("Path", args.manifest_path),
        checksums_path=cast("Path", args.checksums_path),
    )
    print(f"validated {len(records)} distributions for {DISTRIBUTION} {args.version}")
    return 0


def source_build_main(argv: list[str] | None = None) -> int:
    """Check one source distribution and the wheel rebuilt from it."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the treams-rs source distribution and a wheel built from it "
            "against the Cargo.toml version."
        )
    )
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args(argv)
    version = cargo_version()
    sdist = cast("Path", args.sdist)
    wheel = cast("Path", args.wheel)
    check_source_build(sdist, wheel, version=version)
    print(f"validated {sdist.name} and {wheel.name} for {DISTRIBUTION} {version}")
    return 0
