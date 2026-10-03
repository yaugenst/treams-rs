"""Economical sentinels for the full author-data qualification scripts."""

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr

from _scripts import ROOT, load

paper = load("qualify_papers")
REFERENCES = json.loads(
    (ROOT / "benchmarks/papers/ebeam-author-reference.json").read_text()
)["cases"]
thermal = load("papers_thermal")


@pytest.mark.reference
@pytest.mark.parametrize("index", [0, 11, 16, 30, 38, 49])
@pytest.mark.parametrize("name", ["sphere", "cylinder"])
def test_electron_beam_original_author_spectra(name, index):
    spec = paper.EBEAM_CURVES[name]
    energy = np.linspace(*spec["bounds"], paper.EBEAM_SAMPLES)[index]
    actual = paper.electron_spectrum_point(
        tr, energy, cylindrical=spec["cylindrical"], order=spec["order"]
    )
    expected = [REFERENCES[name][quantity][index] for quantity in ("cl", "eels")]
    assert_allclose(actual, expected, rtol=0, atol=1e-8)


@pytest.mark.physics
@settings(max_examples=16)
@given(energy=st.floats(2, 5), phi=st.floats(-np.pi, np.pi))
def test_electron_beam_sphere_rotation_and_passivity(energy, phi):
    calculate = paper.electron_spectrum_point
    direct = calculate(tr, energy)
    rotated = calculate(tr, energy, phi=phi)
    assert_allclose(rotated, direct, rtol=5e-12, atol=1e-15)
    assert 0 <= direct[0] <= direct[1]


@pytest.mark.physics
@pytest.mark.parametrize("wavelength", [351, 410, 480, 550, 600])
def test_paper_periodic_spheres_above_slab_conserve_power(wavelength):
    transmission, reflection = paper.cpc_array_point(tr, 2 * np.pi / wavelength)
    assert transmission >= 0
    assert reflection >= 0
    assert_allclose(transmission + reflection, 1, rtol=0, atol=1e-10)


@pytest.mark.reference
@pytest.mark.parametrize("index", [164, 220])
def test_thermal_chain_original_figure_absorption(index):
    source = json.loads((ROOT / "benchmarks/papers/thermal-source.json").read_text())
    material = np.asarray(source["material"])
    frequency = source["frequencies_hz"][index]
    epsilon = np.interp(frequency, material[:, 0], material[:, 1]) + 1j * np.interp(
        frequency, material[:, 0], material[:, 2]
    )
    matrix = thermal.chain_matrix(tr, frequency, epsilon, 10)
    actual = thermal.absorption(matrix, frequency)
    assert actual > 0
    # The publication's multipole-convergence criterion is 99%, not machine epsilon.
    assert_allclose(actual, source["absorption_um_squared"][index], rtol=0.01, atol=0)


@pytest.mark.interface
def test_thermal_fixture_does_not_depend_on_working_directory(monkeypatch, tmp_path):
    """Another working directory must not trigger a download or a new fixture."""
    monkeypatch.chdir(tmp_path)
    assert thermal.FIXTURE.is_file()
    expected = json.loads((ROOT / "benchmarks/papers/thermal-source.json").read_text())
    assert thermal.fixture() == expected
    assert list(tmp_path.iterdir()) == []
