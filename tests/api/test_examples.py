"""Run the examples gallery and compare it with its recorded output and with treams.

Each ``docs/examples/<name>.py`` prints a table; ``output/<name>.txt`` holds
the printed text (``scripts/generate_docs.py --examples`` rewrites it).
``upstream/<treams name>.py`` holds the matching treams example with the same
sizes, so its results serve as the reference for the treams-rs script.
Examples that treams cannot run, such as gradients, have no treams file.
"""

import re
import runpy
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import yaml

from _support import ROOT, jax_x64

GALLERY = ROOT / "docs/examples"
EXAMPLES = sorted(path.stem for path in GALLERY.glob("*.py"))
UPSTREAM = sorted(path.stem for path in (GALLERY / "upstream").glob("*.py"))
# A first line "# Requires: jax, torch" names optional modules a script imports.
REQUIRES = re.compile(r"\A# Requires: (?P<modules>.+)\n")
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|\bnan\b|\binf\b")


class Oracle(NamedTuple):
    """How to compare a treams-rs example with its treams counterpart.

    ``variables`` names the module-level results to compare: a name shared by
    both scripts, or a ``(treams-rs name, treams name)`` pair where the names
    differ. ``unordered`` marks results that are lists of eigenvalue arrays:
    eigenvalue solvers return them in no fixed order, so each array is
    compared after sorting.
    """

    example: str
    variables: tuple[str | tuple[str, str], ...]
    rtol: float = 1e-10
    atol: float = 0.0
    reason: str = ""
    unordered: bool = False


# Keyed by the treams example name.
ORACLE = {
    "array_spheres": Oracle("array_spheres", (("power", "tr"),)),
    "array_spheres_tmatrixc": Oracle(
        "cylindrical_array",
        (("power", "tr"),),
        rtol=2e-8,
        reason="treams sums the row of chains at its automatic Ewald split, which "
        "leaves errors of about 8e-9; at explicit splits of 0.5 to 1 it agrees with "
        "treams-rs to 1e-14. The decaying axial orders at the lower frequencies "
        "admit no common explicit split.",
    ),
    "band_structure": Oracle("band_structure", ("res",), unordered=True),
    "chain": Oracle("chain", ("ez",)),
    "chain_tmatrixc": Oracle("cylindrical_chain", ("ez",)),
    "cluster": Oracle(
        "cluster",
        (
            "tm",
            "tm_global",
            "tm_rotate",
            "intensity",
            ("intensity_rotate", "intensity_global"),
            ("xs_rotate", "xs"),
        ),
        atol=1e-14,
        reason="T-matrix entries near zero differ by rounding, about 2e-15 for "
        "entries up to 0.44.",
    ),
    "cluster_tmatrixc": Oracle("cylindrical_cluster", ("ez",)),
    "crystal": Oracle(
        "crystal",
        ("res",),
        atol=1e-12,
        reason="At the lowest frequency the smallest singular value is 7.8e-5 of a "
        "matrix with norm 9e3; rounding of the matrix shifts it by about 2e-13.",
    ),
    "crystal_tmatrixc": Oracle("cylindrical_crystal", ("res",)),
    "cylinder_tmatrixc": Oracle(
        "cylinder",
        ("xw_sca", "xw_ext", "xw_sca_mmax0", "xw_ext_mmax0", "intensity"),
    ),
    "grating_tmatrixc": Oracle(
        "grating",
        ("ex",),
        rtol=3e-8,
        reason="treams sums the lattice of cylinders at its automatic Ewald split, "
        "which leaves errors of about 5e-9 in the coupling matrix and 1.3e-8 in "
        "the field; with a larger explicit split it agrees with treams-rs to "
        "1e-12. The propagating and decaying axial orders admit no common "
        "explicit split.",
    ),
    "grid": Oracle(
        "grid",
        ("ez",),
        atol=1e-13,
        reason="Fields that vanish by symmetry come out as rounding noise, about "
        "1e-17, in both codes.",
    ),
    "slab": Oracle("slab", (("power", "tr"),)),
    "sphere": Oracle(
        "sphere",
        ("xs_sca", "xs_ext", "xs_sca_lmax1", "xs_ext_lmax1", "intensity"),
    ),
}


def _requirements(path: Path) -> list[str]:
    match = REQUIRES.match(path.read_text(encoding="utf-8"))
    return [name.strip() for name in match["modules"].split(",")] if match else []


def _template(text: str) -> list[str]:
    """Lines of ``text`` with every number replaced and spaces collapsed."""
    return [" ".join(line.split()) for line in NUMBER.sub("#", text).splitlines()]


def _numbers(text: str) -> np.ndarray:
    return np.array([float(token) for token in NUMBER.findall(text)])


def _results(value, *, unordered: bool) -> list[np.ndarray]:
    if unordered:
        return [np.sort_complex(np.asarray(item)) for item in value]
    return [np.asarray(value)]


@pytest.fixture
def restore_jax_precision():
    """Examples may enable JAX double precision globally; undo it."""
    with jax_x64(enabled=None):
        yield


@pytest.mark.workflows
@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs_and_matches_recorded_output(request, capsys, name):
    script = GALLERY / f"{name}.py"
    for module in _requirements(script):
        pytest.importorskip(module)
        if module == "jax":
            request.getfixturevalue("restore_jax_precision")
    runpy.run_path(str(script), run_name="__main__")
    printed = capsys.readouterr().out
    recorded = (GALLERY / "output" / f"{name}.txt").read_text(encoding="utf-8")
    assert _template(printed) == _template(recorded)
    np.testing.assert_allclose(
        _numbers(printed), _numbers(recorded), rtol=1e-6, atol=1e-12
    )


@pytest.mark.reference
@pytest.mark.parametrize("name", ORACLE)
def test_example_agrees_with_treams(name):
    pytest.importorskip("treams")
    oracle = ORACLE[name]
    ours = runpy.run_path(str(GALLERY / f"{oracle.example}.py"), run_name="__main__")
    theirs = runpy.run_path(str(GALLERY / "upstream" / f"{name}.py"))
    for variable in oracle.variables:
        mine, upstream = (variable, variable) if isinstance(variable, str) else variable
        expected = _results(theirs[upstream], unordered=oracle.unordered)
        actual = _results(ours[mine], unordered=oracle.unordered)
        assert len(actual) == len(expected), f"{name}: {mine}"
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                got,
                want,
                rtol=oracle.rtol,
                atol=oracle.atol,
                err_msg=f"{name}: {mine} against treams {upstream}",
            )


def _nav_pages(entries) -> list[str]:
    pages = []
    for entry in entries:
        value = next(iter(entry.values())) if isinstance(entry, dict) else entry
        pages.extend(_nav_pages(value) if isinstance(value, list) else [value])
    return pages


@pytest.mark.interface
def test_gallery_is_complete():
    nav = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))["nav"]
    section = next(
        entry["Examples"]
        for entry in nav
        if isinstance(entry, dict) and "Examples" in entry
    )
    pages = _nav_pages(section)
    index = (GALLERY / "index.md").read_text(encoding="utf-8")
    assert pages[0] == "examples/index.md"
    assert EXAMPLES
    ported = {oracle.example for oracle in ORACLE.values()}
    for name in EXAMPLES:
        assert (GALLERY / "output" / f"{name}.txt").is_file(), name
        page = (GALLERY / f"{name}.md").read_text(encoding="utf-8")
        assert f'--8<-- "docs/examples/{name}.py"' in page, name
        assert f'--8<-- "docs/examples/output/{name}.txt"' in page, name
        assert f"examples/{name}.md" in pages, name
        assert f"]({name}.md)" in index, name
        # A page shows a treams tab exactly when its script has a treams file.
        has_upstream_tab = '--8<-- "docs/examples/upstream/' in page
        assert has_upstream_tab == (name in ported), name
    assert sorted(ORACLE) == UPSTREAM
    for name, oracle in ORACLE.items():
        assert oracle.example in EXAMPLES, name
        assert oracle.rtol == 1e-10 or oracle.reason, name
        assert oracle.atol == 0 or oracle.reason, name
        page = (GALLERY / f"{oracle.example}.md").read_text(encoding="utf-8")
        assert f'--8<-- "docs/examples/upstream/{name}.py"' in page, name
