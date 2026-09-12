"""Numerical coordinate changes must preserve solves and physical observables."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as rs
from treams_rs import diff


@given(exponent=st.integers(30, 399), real=st.floats(-0.5, 0.5))
def test_solve_and_pullback_preserve_independent_equation_and_unknown_units(
    exponent, real
):
    h = np.array([[2 + 1j, 0.5 - 0.25j], [-0.25 + 0.5j, 3 - 1j]])
    rows = np.exp2(np.array([-exponent, exponent], dtype=float))
    columns = np.exp2(np.array([exponent // 2, -(exponent // 2)], dtype=float))
    y = np.array([[1 + real + 0.2j, 1j], [-0.5j, 1 - real]])
    z = np.array([[0.1j, 0.4], [1 + real, 0.2 + 0.3j]])
    a = h / rows[:, None] / columns[None, :]
    b = (h @ y) / rows[:, None]
    value, context = diff.solve(a, b)
    assert_allclose(value / columns[:, None], y, atol=2e-14)
    ga, gb = context.pullback((h.conj().T @ z) / columns[:, None])
    assert_allclose(gb / rows[:, None], z, atol=2e-14)
    assert_allclose(ga / rows[:, None] / columns[None, :], -z @ y.conj().T, atol=2e-14)


def test_small_wavenumber_high_order_chain_absorption_is_stable():
    # PRB 112, 054307 (2025), Fig. 2: four 250 nm SiC spheres with 20 nm gaps.
    # At 3 THz, I-TC has unit diagonal but entries exceeding 1e22 at degree 10.
    # Oracle: independent two-sided equilibration, LAPACK balancing and sqrt(T)
    # basis normalization agree on this absorption; unscaled LU differs by 0.3%.
    frequency = np.logspace(np.log10(3e12), np.log10(1.2e14), 300)[0]
    k0 = 2 * np.pi * frequency / 299792458.0
    sphere = rs.TMatrix.sphere(
        10,
        k0,
        250e-9,
        [rs.Material(12.87707323 + 0.19225647000000007j), rs.Material()],
    )
    positions = [[0, 0, z * 520e-9] for z in (-1.5, -0.5, 0.5, 1.5)]
    cluster = rs.TMatrix.cluster([sphere] * 4, positions).interaction.solve()
    matrix = np.asarray(cluster.expand(rs.SphericalWaveBasis.default(10)))
    absorption = (
        -2 * np.pi * (np.trace(matrix).real + np.vdot(matrix, matrix).real) / k0**2
    )
    assert_allclose(absorption * 1e12, 0.0001754433586134972, rtol=2e-11, atol=0)
    # Axial symmetry forbids coupling between distinct azimuthal orders.
    orders = np.array([mode[2] for mode in rs.SphericalWaveBasis.default(10)])
    assert_allclose(matrix[orders[:, None] != orders[None, :]], 0, atol=2e-20)
