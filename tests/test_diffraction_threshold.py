"""Finite planar limits are distinct from divergent periodic radiation channels."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import treams
from numpy.testing import assert_allclose

import treams_rs as tr

pytestmark = pytest.mark.filterwarnings(
    "ignore:'where' used without 'out'.*:UserWarning"
)


@pytest.mark.parametrize("other", [1.7, 1.7 + 0.2j])
@pytest.mark.parametrize("side", [0, 1])
def test_single_grazing_interface_matches_exact_fresnel(other, side):
    ks = np.array([[1, 1], [other, other]], complex)
    zs = np.array([1, 0.7 + (0.05j if np.iscomplex(other) else 0)], complex)
    if side:
        ks, zs = ks[::-1].copy(), zs[::-1].copy()
    value, context = tr.diff.interface(ks, zs, [0, 1])
    expected = treams.coeffs.fresnel(ks, np.sqrt(ks**2 - 1), zs)
    assert_allclose(value, expected, atol=8e-16, rtol=8e-16)
    with pytest.raises(ValueError, match=r"derivative.*diffraction threshold"):
        context.pullback(np.ones_like(value))


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("direction", [-1, 1, 1j])
def test_grazing_interface_is_the_two_sided_radiation_limit(alignment, direction):
    ks = np.array([[1, 1], [1.7, 1.7]], complex)
    exact = tr.diff.interface(ks, [1, 0.7], [0.6, 0.8], alignment=alignment)[0]
    errors = []
    for delta in (1e-4, 1e-6, 1e-8):
        nearby = ks.copy()
        nearby[0] += direction * delta
        value = tr.diff.interface(nearby, [1, 0.7], [0.6, 0.8], alignment=alignment)[0]
        errors.append(np.max(abs(value - exact)))
    # The first variation is proportional to sqrt(delta), including the
    # outgoing complex-wavevector branch. It need not have a finite derivative.
    assert errors[1] < 0.11 * errors[0]
    assert errors[2] < 0.11 * errors[1]


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("ks", [[[1, 1], [1, 1]], [[1, 1.4], [1, 1.4]]])
def test_identical_grazing_media_are_transparent(alignment, ks):
    value, context = tr.diff.interface(ks, [0.8, 0.8], [0, 1], alignment=alignment)
    expected = np.zeros((2, 2, 2, 2), complex)
    expected[0, 0] = expected[1, 1] = np.eye(2)
    assert_allclose(value, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match=r"derivative.*diffraction threshold"):
        context.pullback(np.ones_like(value))
    if alignment == "xy":
        ks = np.asarray(ks, complex)
        value, context = tr.coeffs.fresnel_with_context(
            ks, np.sqrt(ks**2 - 1), [0.8, 0.8]
        )
        assert_allclose(value, expected, atol=0, rtol=0)
        with pytest.raises(ValueError, match=r"derivative.*grazing"):
            context.pullback(np.ones_like(value))


def test_nonidentical_double_grazing_boundary_remains_explicitly_singular():
    # A general singular tangential system must not silently use a pseudoinverse.
    with pytest.raises(ValueError, match="singular scattering system"):
        tr.diff.interface(np.ones((2, 2)), [1, 0.7], [0, 1])


def test_paper_endpoint_slab_is_finite_but_array_pole_is_explicit():
    k0 = 2 * np.pi / 350
    basis = tr.PlaneWavePorts.diffr_orders([0, 0.3 * k0], tr.Lattice.square(500), 0.02)
    slab = tr.SMatrix.slab(10, basis, k0, [1, 3, 1])
    oracle = treams.SMatrices.slab(
        10, treams.PlaneWaveBasisByComp(basis.modes), k0, [1, 3, 1]
    )
    expected = np.array(
        [[np.asarray(oracle[i, j]) for j in range(2)] for i in range(2)]
    )
    assert_allclose(slab.array, expected, rtol=2e-14, atol=2e-14)
    point = runpy.run_path(Path(__file__).parents[1] / "scripts/qualify_papers.py")[
        "cpc_array_point"
    ]
    with pytest.raises(ValueError, match=r"diffraction threshold"):
        point(tr, k0)
    gaps = []
    for delta in (1e-4, 1e-6):
        values = []
        for side in (-1, 1):
            k = 2 * np.pi / (350 * (1 + side * delta))
            value = point(tr, k)
            assert_allclose(value, point(treams, k), rtol=3e-9, atol=2e-10)
            assert_allclose(value.sum(), 1, atol=2e-10, rtol=0)
            assert np.all((value >= 0) & (value <= 1))
            values.append(value)
        gaps.append(np.max(abs(values[0] - values[1])))
    assert gaps[1] < 0.11 * gaps[0]
