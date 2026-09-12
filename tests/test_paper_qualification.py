"""Economical sentinels for the full author-data qualification scripts."""

import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr

ROOT = Path(__file__).resolve().parents[1]
PAPER = run_path(str(ROOT / "scripts/qualify_papers.py"))
REFERENCES = json.loads(
    (ROOT / "benchmarks/papers/ebeam-author-reference.json").read_text()
)["cases"]
THERMAL = run_path(str(ROOT / "scripts/papers_thermal.py"))


@pytest.mark.oracle_numerical
@pytest.mark.parametrize("index", [0, 11, 16, 30, 38, 49])
@pytest.mark.parametrize("name", ["sphere", "cylinder"])
def test_electron_beam_original_author_spectra(name, index):
    cylindrical = name == "cylinder"
    bounds, order = ((2.5, 4.5), 12) if cylindrical else ((2, 5), 4)
    actual = PAPER["electron_spectrum_point"](
        tr, np.linspace(*bounds, 50)[index], cylindrical=cylindrical, order=order
    )
    expected = [REFERENCES[name][quantity][index] for quantity in ("cl", "eels")]
    assert_allclose(actual, expected, rtol=0, atol=1e-8)


@pytest.mark.physics
@settings(max_examples=16, deadline=None)
@given(energy=st.floats(2, 5), phi=st.floats(-np.pi, np.pi))
def test_electron_beam_sphere_rotation_and_passivity(energy, phi):
    calculate = PAPER["electron_spectrum_point"]
    direct = calculate(tr, energy)
    rotated = calculate(tr, energy, phi=phi)
    assert_allclose(rotated, direct, rtol=5e-12, atol=1e-15)
    assert 0 <= direct[0] <= direct[1]


@pytest.mark.physics
@pytest.mark.parametrize("wavelength", [351, 410, 480, 550, 600])
def test_paper_periodic_spheres_above_slab_conserve_power(wavelength):
    transmission, reflection = PAPER["cpc_array_point"](tr, 2 * np.pi / wavelength)
    assert transmission >= 0
    assert reflection >= 0
    assert_allclose(transmission + reflection, 1, rtol=0, atol=1e-10)


@pytest.mark.oracle_numerical
@pytest.mark.parametrize("index", [164, 220])
def test_thermal_chain_original_figure_absorption(index):
    source = json.loads((ROOT / "benchmarks/papers/thermal-source.json").read_text())
    material = np.asarray(source["material"])
    frequency = source["frequencies_hz"][index]
    epsilon = np.interp(frequency, material[:, 0], material[:, 1]) + 1j * np.interp(
        frequency, material[:, 0], material[:, 2]
    )
    matrix = THERMAL["chain_matrix"](tr, frequency, epsilon, 10)
    actual = THERMAL["absorption"](matrix, frequency)
    assert actual > 0
    # The publication's multipole-convergence criterion is 99%, not machine epsilon.
    assert_allclose(actual, source["absorption_um_squared"][index], rtol=0.01, atol=0)
