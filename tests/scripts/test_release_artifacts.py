"""Release artifacts are complete, attributable and free of local build paths.

The release workflows build exactly that family and publish it to PyPI only
after the advisory TestPyPI check has run and the operator has approved.
"""

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from io import BytesIO
from itertools import product
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.interface

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their annotations through the registered module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release_artifacts = _load(
    "treams_rs_release_artifacts", SCRIPTS / "_support/release_artifacts.py"
)
build_paths = _load("treams_rs_wheel_build_paths", SCRIPTS / "wheel_build_paths.py")
minimums = _load("treams_rs_dependency_minimums", SCRIPTS / "dependency_minimums.py")
ReleaseArtifactError = release_artifacts.ReleaseArtifactError

_VERSION = "1.2.3"
_REVISION = "a" * 40
_LICENSE_FILES = tuple(sorted(release_artifacts._REQUIRED_LICENSE_FILES))
_PACKAGE_FILES = ("treams_rs/_native.pyi", "treams_rs/py.typed")
_DIST_INFO = f"treams_rs-{_VERSION}.dist-info"
_PLATFORM_TAGS = (
    "manylinux_2_17_x86_64.manylinux2014_x86_64",
    "manylinux_2_17_aarch64.manylinux2014_aarch64",
    "macosx_10_12_x86_64",
    "macosx_11_0_arm64",
    "win_amd64",
)
_SDIST_FILES = tuple(sorted(release_artifacts._REQUIRED_SOURCE_FILES - {"PKG-INFO"}))


@pytest.fixture
def python_tags():
    """Return the CPython family declared by the pyproject classifiers."""
    return release_artifacts._PYTHON_TAGS


def _metadata(*, version=_VERSION, licenses=_LICENSE_FILES, name="treams-rs"):
    return f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n" + "".join(
        f"License-File: {entry}\n" for entry in licenses
    )


def _native_module(python_tag, platform_tag):
    if platform_tag == "win_amd64":
        return f"treams_rs/_native.{python_tag}-win_amd64.pyd"
    return f"treams_rs/_native.cpython-{python_tag.removeprefix('cp')}-native.so"


def _write_wheel(path, update=None):
    _, _, python_tag, _, platform_tag = path.name.removesuffix(".whl").split("-")
    entries = {
        f"{_DIST_INFO}/METADATA": _metadata(),
        **{
            f"{_DIST_INFO}/licenses/{name}": "license fixture\n"
            for name in _LICENSE_FILES
        },
        **dict.fromkeys(_PACKAGE_FILES, "package fixture\n"),
        _native_module(python_tag, platform_tag): "native fixture\n",
    }
    if update is not None:
        update(entries)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def _write_sdist(path, *, missing=None, extra=(), version=_VERSION):
    root = f"treams_rs-{_VERSION}"
    files = {
        f"{root}/PKG-INFO": _metadata(version=version).encode(),
        **{f"{root}/{name}": b"release fixture\n" for name in _SDIST_FILES},
        **{f"{root}/{name}": b"repository fixture\n" for name in extra},
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            if name == f"{root}/{missing}":
                continue
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _wheel_names(python_tags):
    return [
        f"treams_rs-{_VERSION}-{tag}-{tag}-{platform}.whl"
        for tag, platform in product(python_tags, _PLATFORM_TAGS)
    ]


def _write_release_set(dist_dir, python_tags):
    dist_dir.mkdir()
    for name in _wheel_names(python_tags):
        _write_wheel(dist_dir / name)
    _write_sdist(dist_dir / f"treams_rs-{_VERSION}.tar.gz")


def _wheel(dist_dir):
    return sorted(dist_dir.glob("*.whl"))[-1]


def _sdist(dist_dir):
    return dist_dir / f"treams_rs-{_VERSION}.tar.gz"


def _rewrite(update):
    return lambda dist: _write_wheel(_wheel(dist), update)


def _rejection(name, mutate, match, revision=_REVISION):
    return pytest.param(mutate, match, revision, id=name)


def test_release_command_validates_and_hashes_complete_set(
    tmp_path, capsys, python_tags
):
    dist_dir = tmp_path / "dist"
    manifest_path = tmp_path / "RELEASE-PROVENANCE.json"
    checksums_path = tmp_path / "SHA256SUMS"
    _write_release_set(dist_dir, python_tags)

    returncode = release_artifacts.main(
        [
            *("--dist-dir", str(dist_dir), "--version", _VERSION),
            *("--source-revision", _REVISION, "--manifest-path", str(manifest_path)),
            *("--checksums-path", str(checksums_path)),
        ]
    )

    assert returncode == 0
    count = 5 * len(python_tags) + 1
    assert capsys.readouterr().out == (
        f"validated {count} distributions for treams-rs {_VERSION}\n"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["package"] == "treams-rs"
    assert manifest["source_revision"] == _REVISION
    assert manifest["package_version"] == _VERSION
    assert {artifact["filename"] for artifact in manifest["artifacts"]} == {
        path.name for path in dist_dir.iterdir()
    }
    families = {
        artifact["platform_family"]
        for artifact in manifest["artifacts"]
        if artifact["kind"] == "wheel"
    }
    assert families == set(release_artifacts._PLATFORM_FAMILIES)
    for line in checksums_path.read_text().splitlines():
        digest, filename = line.split("  ")
        assert digest == hashlib.sha256((dist_dir / filename).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutate", "match", "revision"),
    [
        _rejection(
            "source-revision", lambda _dist: None, "one full lowercase Git", "A" * 40
        ),
        _rejection(
            "no-sdist", lambda dist: _sdist(dist).unlink(), "exactly one sdist, found 0"
        ),
        _rejection(
            "wheel-family",
            lambda dist: _wheel(dist).unlink(),
            "release wheel family mismatch",
        ),
        _rejection(
            "duplicate-family-wheel",
            lambda dist: shutil.copy(
                next(dist.glob("*manylinux_2_17_x86_64*.whl")),
                dist
                / next(dist.glob("*manylinux_2_17_x86_64*.whl")).name.replace(
                    "manylinux_2_17_x86_64.manylinux2014_x86_64", "manylinux2014_x86_64"
                ),
            ),
            "release wheel family mismatch",
        ),
        *(
            _rejection(
                name,
                lambda dist, filename=filename: shutil.copy(
                    _wheel(dist), dist / filename
                ),
                match,
            )
            for name, filename, match in (
                (
                    "wheel-name-shape",
                    "treams_rs-1.2.3-cp312-cp312.whl",
                    "unexpected shape",
                ),
                (
                    "distribution",
                    "treams-1.2.3-cp312-cp312-win_amd64.whl",
                    "wrong distribution",
                ),
                (
                    "filename-version",
                    "treams_rs-9.9.9-cp312-cp312-win_amd64.whl",
                    "has version 9.9.9",
                ),
                (
                    "platform-tag",
                    "treams_rs-1.2.3-cp312-cp312-linux_x86_64.whl",
                    "platform tag",
                ),
                (
                    "abi3-wheel",
                    "treams_rs-1.2.3-cp312-abi3-win_amd64.whl",
                    "release wheel family mismatch",
                ),
            )
        ),
        _rejection(
            "metadata-count",
            _rewrite(lambda entries: entries.pop(f"{_DIST_INFO}/METADATA")),
            "exactly one METADATA file",
        ),
        _rejection(
            "packaged-license",
            _rewrite(lambda entries: entries.pop(f"{_DIST_INFO}/licenses/LICENSE")),
            "missing packaged license files: LICENSE$",
        ),
        _rejection(
            "declared-license",
            _rewrite(
                lambda entries: entries.update(
                    {f"{_DIST_INFO}/METADATA": _metadata(licenses=())}
                )
            ),
            "metadata is missing License-File entries",
        ),
        *(
            _rejection(
                f"wheel-{name}",
                _rewrite(lambda entries, name=name: entries.pop(name)),
                f"missing required package files: {re.escape(name)}$",
            )
            for name in _PACKAGE_FILES
        ),
        _rejection(
            "native-module",
            _rewrite(
                lambda entries: [
                    entries.pop(name)
                    for name in list(entries)
                    if name.startswith("treams_rs/_native.")
                    and not name.endswith(".pyi")
                ]
            ),
            "exactly one native extension module",
        ),
        _rejection(
            "agent-notes",
            _rewrite(
                lambda entries: entries.update({"treams_rs/AGENTS.md": "notes\n"})
            ),
            "development-only files: treams_rs/AGENTS.md$",
        ),
        _rejection(
            "unreadable-wheel",
            lambda dist: _wheel(dist).write_bytes(b"not a zip"),
            "not a readable wheel",
        ),
        _rejection(
            "metadata-version",
            _rewrite(
                lambda entries: entries.update(
                    {f"{_DIST_INFO}/METADATA": _metadata(version="0")}
                )
            ),
            "metadata does not identify treams-rs 1.2.3",
        ),
        _rejection(
            "metadata-name",
            _rewrite(
                lambda entries: entries.update(
                    {f"{_DIST_INFO}/METADATA": _metadata(name="treams")}
                )
            ),
            "metadata does not identify treams-rs 1.2.3",
        ),
        _rejection(
            "sdist-name",
            lambda dist: _sdist(dist).rename(dist / "treams_rs-9.9.9.tar.gz"),
            "source distribution is treams_rs-9.9.9.tar.gz",
        ),
        _rejection(
            "unreadable-sdist",
            lambda dist: _sdist(dist).write_bytes(b"not a tar"),
            "not a readable source distribution",
        ),
        _rejection(
            "sdist-metadata-version",
            lambda dist: _write_sdist(_sdist(dist), version="0"),
            "treams_rs-1.2.3.tar.gz metadata does not identify treams-rs 1.2.3",
        ),
        *(
            _rejection(
                f"sdist-{name}",
                lambda dist, name=name: _write_sdist(_sdist(dist), missing=name),
                f"missing required source files: treams_rs-{_VERSION}/{re.escape(name)}$",
            )
            for name in ("PKG-INFO", *_SDIST_FILES)
        ),
        *(
            _rejection(
                f"sdist-excludes-{name}",
                lambda dist, name=name: _write_sdist(
                    _sdist(dist), extra=(f"{name}/fixture.txt",)
                ),
                f"non-source repository files: treams_rs-{_VERSION}/{name}/fixture.txt$",
            )
            for name in sorted(release_artifacts._EXCLUDED_SOURCE_DIRECTORIES)
        ),
    ],
)
def test_assemble_release_artifacts_rejects_an_invalid_set_without_provenance(
    tmp_path, python_tags, mutate, match, revision
):
    _write_release_set(tmp_path / "dist", python_tags)
    mutate(tmp_path / "dist")

    with pytest.raises(ReleaseArtifactError, match=match):
        release_artifacts.assemble_release_artifacts(
            tmp_path / "dist",
            version=_VERSION,
            source_revision=revision,
            manifest_path=tmp_path / "provenance" / "RELEASE-PROVENANCE.json",
            checksums_path=tmp_path / "provenance" / "SHA256SUMS",
        )

    assert not (tmp_path / "provenance").exists()


def test_release_family_is_the_classifier_family_on_every_platform(python_tags):
    expected = {
        (tag, tag, family)
        for tag in python_tags
        for family in (
            "linux-x86_64",
            "linux-aarch64",
            "macos-x86_64",
            "macos-arm64",
            "windows-x86_64",
        )
    }
    assert expected == release_artifacts._EXPECTED_WHEELS
    assert {release_artifacts._platform_family(tag) for tag in _PLATFORM_TAGS} == set(
        release_artifacts._PLATFORM_FAMILIES
    )


def test_classifiers_cover_exactly_the_requires_python_range():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = [
        int(match[1])
        for classifier in project["classifiers"]
        if (
            match := re.fullmatch(
                r"Programming Language :: Python :: 3\.(\d+)", classifier
            )
        )
    ]
    bounds = re.fullmatch(r">=3\.(\d+),\s*<3\.(\d+)", project["requires-python"])
    assert bounds is not None, project["requires-python"]
    supported = list(range(int(bounds[1]), int(bounds[2])))
    # Wheels are built for the classifier family; requires-python must not admit
    # an interpreter that would fall back to compiling the source distribution.
    assert sorted(declared) == supported


def test_license_is_a_canonical_spdx_expression():
    # maturin copies the expression verbatim; PyPI rejects a noncanonical one.
    licenses = pytest.importorskip("packaging.licenses")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    expression = project["license"]
    assert licenses.canonicalize_license_expression(expression) == expression
    for path in project["license-files"]:
        assert (ROOT / path).is_file(), path


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_native_wheel_matrix_builds_the_classifier_family():
    matrix = _workflow("native-wheels.yml")["jobs"]["wheelhouse"]["strategy"]["matrix"]
    pythons = matrix["python"]
    tags = tuple(f"cp{entry['version'].replace('.', '')}" for entry in pythons)

    assert tags == release_artifacts._PYTHON_TAGS
    assert [entry["artifact"] for entry in pythons] == [
        tag.replace("cp", "py") for tag in tags
    ]
    assert (
        tuple(platform["artifact"] for platform in matrix["platform"])
        == release_artifacts._PLATFORM_FAMILIES
    )
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    profiles = {"base", "advect", "autograd", "io"}
    assert profiles - {"base"} <= set(extras)
    for entry in pythons:
        smoked = entry["profiles"].split()
        skipped = {reason.partition(":")[0] for reason in [entry["skipped"]] if reason}
        # Every profile is smoke-tested or skipped with a stated reason.
        assert set(smoked) | skipped == profiles, entry
        assert not set(smoked) & skipped, entry


def _needs(job):
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _upstream(jobs, name):
    seen = set()
    pending = _needs(jobs[name])
    while pending:
        dependency = pending.pop()
        if dependency not in seen:
            seen.add(dependency)
            pending.extend(_needs(jobs[dependency]))
    return seen


def test_production_waits_for_testpypi_verification_and_approval():
    jobs = _workflow("publish-release.yml")["jobs"]

    assert jobs["candidate"]["with"]["target"] == "testpypi"
    assert jobs["promote"]["environment"] == "release"
    for name in ("docs", "github-release", "pypi", "verify-pypi"):
        assert {"testpypi", "verify-testpypi", "promote"} <= _upstream(jobs, name), name
    # A job condition such as always() would run a job after a failed dependency.
    assert [name for name, job in jobs.items() if "if" in job] == []


def test_only_publishing_release_jobs_can_write():
    workflow = _workflow("publish-release.yml")
    jobs = workflow["jobs"]

    assert workflow["permissions"] == {"contents": "read"}
    assert all(isinstance(job.get("permissions", {}), dict) for job in jobs.values())
    writes = {
        (name, scope)
        for name, job in jobs.items()
        for scope, level in job.get("permissions", {}).items()
        if level == "write"
    }
    assert writes == {
        ("testpypi", "id-token"),
        ("pypi", "id-token"),
        ("docs", "contents"),
        ("docs", "pages"),
        ("docs", "id-token"),
        ("github-release", "contents"),
    }
    assert jobs["testpypi"]["environment"]["name"] == "testpypi"
    assert jobs["pypi"]["environment"]["name"] == "pypi"


def test_empty_classifier_family_is_rejected(tmp_path, monkeypatch):
    _write_release_set(tmp_path / "dist", ("cp312",))
    monkeypatch.setattr(release_artifacts, "_EXPECTED_WHEELS", frozenset())
    with pytest.raises(ReleaseArtifactError, match="declares no CPython classifiers"):
        release_artifacts.assemble_release_artifacts(
            tmp_path / "dist",
            version=_VERSION,
            source_revision=_REVISION,
            manifest_path=tmp_path / "provenance" / "RELEASE-PROVENANCE.json",
            checksums_path=tmp_path / "provenance" / "SHA256SUMS",
        )


def test_source_build_check_accepts_a_host_wheel_and_rejects_bad_sources(tmp_path):
    sdist = tmp_path / f"treams_rs-{_VERSION}.tar.gz"
    wheel = tmp_path / f"treams_rs-{_VERSION}-cp312-cp312-linux_x86_64.whl"
    _write_sdist(sdist)
    _write_wheel(wheel)
    release_artifacts.check_source_build(sdist, wheel, version=_VERSION)

    _write_wheel(wheel, lambda entries: entries.pop(f"{_DIST_INFO}/licenses/LICENSE"))
    with pytest.raises(ReleaseArtifactError, match="missing packaged license files"):
        release_artifacts.check_source_build(sdist, wheel, version=_VERSION)
    other = wheel.with_name("treams_rs-9.9.9-cp312-cp312-linux_x86_64.whl")
    _write_wheel(other)
    with pytest.raises(ReleaseArtifactError, match=r"has version 9\.9\.9, expected 1"):
        release_artifacts.check_source_build(sdist, other, version=_VERSION)


def test_wheel_path_scan_finds_every_encoding_of_a_build_root(tmp_path):
    wheel = tmp_path / "fixture.whl"
    root = "/home/runner/work/treams-rs/treams-rs"
    windows = r"C:\Users\runneradmin\.cargo"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("clean.so", b"/build/registry/src/pyo3/src/lib.rs")
        archive.writestr("posix.so", f"x{root}/crates/treams-core/src/lib.rs".encode())
        archive.writestr("forward.pyd", b"c:/Users/runneradmin/.cargo/registry/lib.rs")
        archive.writestr("wide.pyd", (windows + r"\registry").encode("utf-16-le"))

    leaks = build_paths.find_leaks(wheel, [root, windows])

    assert leaks == [
        f"posix.so: {root}",
        f"forward.pyd: {windows}",
        f"wide.pyd: {windows}",
    ]


def test_wheel_path_scan_matches_whole_path_components(tmp_path):
    wheel = tmp_path / "fixture.whl"
    remapped = "/build/registry/src/index.crates.io-0/num-integer-0.1.47/src/roots.rs"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("roots.so", f"{remapped}crates/treams-core/src/roots.rs")
        archive.writestr("rootfs.so", b"/rootfs/etc\x00/root.d/x\x00/rooted")
        archive.writestr("wide.pyd", "src/roots.rs".encode("utf-16-le"))
        archive.writestr("cstring.so", b"rustc\x00/root\x00")
        archive.writestr("windows.pyd", b"C:\\\\Users\\\\runneradmin\\\\x")
        archive.writestr("wide-cstring.pyd", "/root\x00".encode("utf-16-le"))
        archive.writestr("end.txt", b"HOME=/root")

    leaks = build_paths.find_leaks(wheel, ["/root", r"C:\Users\runneradmin"])

    assert leaks == [
        "cstring.so: /root",
        r"windows.pyd: C:\Users\runneradmin",
        "wide-cstring.pyd: /root",
        "end.txt: /root",
    ]


def test_wheel_path_check_command_reports_leaks(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    clean = tmp_path / "clean.whl"
    leaky = tmp_path / "leaky.whl"
    with zipfile.ZipFile(clean, "w") as archive:
        archive.writestr("treams_rs/_native.so", b"/build/.cargo/registry/src")
    with zipfile.ZipFile(leaky, "w") as archive:
        archive.writestr("treams_rs/_native.so", b"/root/.cargo/registry/src")

    assert build_paths.main(["check", "--home", "/root", str(clean)]) == 0
    assert build_paths.main(["check", "--home", "/root", str(leaky)]) == 1
    assert "leaky.whl: treams_rs/_native.so: /root" in capsys.readouterr().err


def test_rustflags_keep_existing_flags_and_put_specific_roots_last(monkeypatch, capsys):
    monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=x86-64-v2")
    monkeypatch.setenv("CARGO_HOME", "/opt/cargo")
    monkeypatch.setenv("CARGO_TARGET_DIR", "/scratch/target")
    monkeypatch.delenv("CARGO_BUILD_TARGET_DIR", raising=False)

    assert build_paths.main(["rustflags", "--home", "/root"]) == 0
    flags = capsys.readouterr().out.split("\x1f")

    assert flags[:2] == ["-C", "target-cpu=x86-64-v2"]
    remaps = [flag.removeprefix("--remap-path-prefix=") for flag in flags[2:]]
    assert all(remap.endswith("=/build") for remap in remaps)
    roots = [remap.removesuffix("=/build") for remap in remaps]
    assert {"/root", "/root/.cargo", "/root/.rustup", "/opt/cargo"} <= set(roots)
    # Build scripts write generated sources beneath the Cargo target directory.
    assert "/scratch/target" in roots
    assert str(Path.cwd()) in roots
    assert roots.index("/root") < roots.index("/root/.cargo")
    assert roots == sorted(roots, key=len)

    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "-Dwarnings")
    assert build_paths.encoded_rustflags([]).split("\x1f")[0] == "-Dwarnings"


def test_dependency_minimums_pin_every_declared_floor(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'dependencies = ["numpy>=2.1,<3"]\n'
        "[project.optional-dependencies]\n"
        "io = [\"h5py >= 3.11, <4 ; python_version < '3.15'\"]\n"
        'advect = ["advect[extra]>=0.2.0,<0.3"]\n'
        'loose = ["scipy<2"]\n'
    )

    assert minimums.minimum_pins(["advect", "io"], pyproject) == [
        "numpy==2.1",
        "advect==0.2.0",
        "h5py==3.11; python_version < '3.15'",
    ]
    with pytest.raises(ValueError, match="must declare exactly one >= floor"):
        minimums.minimum_pins(["loose"], pyproject)
    with pytest.raises(ValueError, match="unknown extra 'torch'"):
        minimums.minimum_pins(["torch"], pyproject)


def test_dependency_minimums_cover_the_smoke_profiles():
    pins = minimums.minimum_pins(["advect", "autograd", "io"])
    assert [pin.split("==")[0] for pin in pins][:1] == ["numpy"]
    assert {"advect", "autograd", "h5py"} <= {pin.split("==")[0] for pin in pins}


@pytest.mark.parametrize(
    "script",
    [
        "assemble_release_artifacts.py",
        "check_source_distribution.py",
        "dependency_minimums.py",
        "wheel_build_paths.py",
    ],
)
def test_release_scripts_are_directly_runnable(script):
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
