"""Incident plane waves, multipole conversion and complete scattering workflows."""

import numpy as np
import pytest
import treams
from hypothesis import example, given
from hypothesis import strategies as st

import treams_rs as tr

# _native: the plane-wave polarization vectors have no public wrapper.
from treams_rs import _native


@pytest.mark.reference
@pytest.mark.parametrize(
    "kvec", [[0, 0, 1], [0, 0, -1], [0.3, 0.4, 0.9], [1.1, 0, 0.3j]]
)
@pytest.mark.parametrize("pol", [0, 1, [0.3 + 0.1j, 0.7 - 0.2j], [1, 0.2, 0.3]])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_plane_to_spherical_reference(kvec, pol, poltype):
    material = (2.0 + 0.1j, 1.1, 0.02 if poltype == "helicity" else 0)
    actual = tr.plane_wave(kvec, pol, k0=1.3, material=material, poltype=poltype)
    expected = treams.plane_wave(
        kvec, pol, k0=1.3, material=treams.Material(material), poltype=poltype
    )
    positions = [[0, 0, 0], [0.1, 0.2, 0.3]]
    basis = tr.SphericalBasis.default(4, 2, positions)
    oracle_basis = treams.SphericalWaveBasis.default(4, 2, positions)
    reference = np.asarray(expected.expand(oracle_basis))
    if kvec[0] == kvec[1] == 0:
        # The upstream high-level path leaks forbidden modes when complex kz/k
        # differs from +/-1 by rounding. Angular coefficients are scale invariant:
        # evaluate them at the exact real unit vector and retain the physical phase.
        reference = np.zeros(len(basis), dtype=complex)
        for amplitude, mode in zip(np.asarray(expected), expected.basis, strict=True):
            direction = np.asarray(mode[:3])
            k = treams.Material(material).ks(1.3)[mode[3]]
            angular = treams.pw.to_sw(
                basis.l, basis.m, basis.pol, *direction, mode[3], poltype=poltype
            )
            reference += (
                amplitude
                * angular
                * np.exp(1j * k * (basis.positions[basis.pidx] @ direction))
            )
        np.testing.assert_array_equal(actual.expand(basis)[abs(basis.m) != 1], 0)
    np.testing.assert_allclose(actual.expand(basis), reference, rtol=2e-10, atol=2e-11)


@pytest.mark.physics
@given(theta=st.floats(0, np.pi), phi=st.floats(-np.pi, np.pi), pol=st.integers(0, 1))
@example(theta=3.037149875588694e-159, phi=1.0, pol=0)
@example(theta=1e-320, phi=1.0, pol=1)
def test_plane_field_reconstruction_and_maxwell(theta, phi, pol):
    wave = tr.plane_wave_angle(theta, phi, pol, k0=1.3)
    basis = tr.SphericalBasis.default(10)
    points = np.array([[0, 0, 0], [0.2, -0.1, 0.3], [0, 0, 0.5]])
    actual, _ = tr.diff.field(wave.expand(basis), points, basis, [1.3, 1.3])
    vector = wave.kvecs[pol]
    polarization = np.array(_native.plane_polarization(tuple(vector), pol, True))
    expected = np.exp(1j * (points @ vector))[:, None] * polarization
    np.testing.assert_allclose(actual, expected, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(np.dot(vector, polarization), 0, atol=1e-12)
    np.testing.assert_allclose(
        1j * np.cross(vector, polarization),
        (2 * pol - 1) * 1.3 * polarization,
        atol=1e-12,
    )


@pytest.mark.workflows
@pytest.mark.reference
@pytest.mark.parametrize("cylindrical", [False, True])
def test_plane_illumination_cross_sections(cylindrical):
    wave = tr.plane_wave([1, 0, 0], [0, 1, 0], k0=1.2)
    reference_wave = treams.plane_wave(
        [1, 0, 0], [0, 1, 0], k0=1.2, material=1, poltype="helicity"
    )
    if cylindrical:
        matrix = tr.CylindricalTMatrix.cylinder([0.0], 4, 1.2, 0.3, [3.0, 1.0])
        reference = treams.TMatrixC.cylinder([0.0], 4, 1.2, 0.3, [3.0, 1.0])
        actual, expected = matrix.xw(wave), reference.xw(reference_wave)
    else:
        matrix = tr.TMatrix.sphere(4, 1.2, 0.3, [3.0, 1.0])
        reference = treams.TMatrix.sphere(4, 1.2, 0.3, [3.0, 1.0])
        actual, expected = matrix.xs(wave), reference.xs(reference_wave)
    np.testing.assert_allclose(
        wave.expand(matrix.basis),
        reference_wave.expand(reference.basis),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        matrix @ wave,
        np.asarray(reference) @ np.asarray(reference_wave.expand(reference.basis)),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual[0], actual[1], rtol=1e-10)


@pytest.mark.interface
def test_plane_metadata_must_match_scattering_problem():
    matrix = tr.TMatrix.sphere(1, 1.2, 0.3, [3.0, 1.0])
    with pytest.raises(ValueError, match="matching"):
        matrix.xs(tr.plane_wave([0, 0, 1], 0, k0=1.3))
    # A plane wave without amplitude expands to zeros in every basis family.
    silent = tr.plane_wave([0.3, 0.4, 0.9], [0, 0], k0=1.2)
    for basis in (
        tr.SphericalBasis.default(1, 2, [[0, 0, 0], [0.3, 0, 0]]),
        tr.CylindricalBasis.default([0.2], 1),
        tr.PlaneWavePorts.default([[0.1, 0.2]]),
    ):
        np.testing.assert_array_equal(silent.expand(basis), np.zeros(len(basis)))
