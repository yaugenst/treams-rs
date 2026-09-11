"""Geometry invariants and compatibility through the typed native boundary."""

import numpy as np
import pytest
import treams.lattice as reference_lattice
import treams.misc as reference_misc
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from treams_rs import Material, PlaneWaveBasisByComp, lattice, misc


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_cell_gufuncs_and_strides(dim, dtype):
    rng = np.random.default_rng(37)
    cells = rng.integers(-4, 5, (17, dim, dim)).astype(dtype)
    cells += 12 * np.eye(dim, dtype=dtype)
    expected = reference_lattice.volume(cells)
    value = lattice.volume(cells)
    assert value.dtype == expected.dtype
    assert_array_equal(value, expected)
    target = np.zeros(34, dtype=dtype)[::2]
    lattice.volume(cells[::-1].swapaxes(-1, -2), out=target)
    assert_array_equal(target, expected[::-1])
    reciprocal = lattice.reciprocal(cells)
    assert_allclose(reciprocal, reference_lattice.reciprocal(cells), rtol=4e-15)
    output = np.empty((17, dim, dim * 2))[:, :, ::2].swapaxes(-1, -2)
    lattice.reciprocal(cells, out=output)
    assert_allclose(output, reciprocal)
    lattice.reciprocal(output, out=output)
    assert_allclose(output, cells, atol=4e-14)
    assert lattice.area is lattice.volume


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_single_cell_fast_path_and_numpy_dispatch(dim):
    cell = np.eye(dim) * 1.3
    if dim > 1:
        cell[0, 1] = 0.2
    for function in (lattice.volume, lattice.reciprocal):
        expected = function(cell[None])[0]
        assert_allclose(function(cell), expected, rtol=2e-15)
        assert_allclose(function(cell.T), function(cell.T[None])[0], rtol=2e-15)
        out = np.empty_like(expected)
        assert function(cell, out) is out
        assert_allclose(out, expected, rtol=2e-15)

        class Intercept(np.ndarray):
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return "intercepted"

        assert function(cell.view(Intercept)) == "intercepted"


@given(dim=st.integers(1, 3), pitch=st.floats(0.3, 5), shear=st.floats(-0.8, 0.8))
@settings(max_examples=30, deadline=None)
def test_reciprocal_duality_and_scale(dim, pitch, shear):
    a = np.eye(dim) * pitch
    if dim > 1:
        a[0, 1] = shear
    b = lattice.reciprocal(a)
    assert_allclose(a @ b.T, np.eye(dim) * (2 * np.pi), rtol=2e-15, atol=3e-14)
    assert_allclose(lattice.reciprocal(b), a, atol=2e-14)
    assert_allclose(lattice.volume(1.7 * a), 1.7**dim * lattice.volume(a), rtol=3e-15)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("n", [0, 1, 4])
def test_cube_and_boundary_match_reference(dim, n):
    assert_array_equal(lattice.cube(dim, n), reference_lattice.cube(dim, n))
    assert_array_equal(lattice.cubeedge(dim, n), reference_lattice.cubeedge(dim, n))
    boundary = lattice.cubeedge(dim, n)
    assert len(boundary) == ((2 * n + 1) ** dim - (2 * n - 1) ** dim if n else 1)
    assert np.all(np.max(np.abs(boundary), axis=1) == n)


def test_geometry_invalid_input_and_empty_batch():
    with pytest.raises(ValueError):
        lattice.cube(4, 1)
    with pytest.raises(ValueError):
        lattice.cube(2, -1)
    with pytest.raises(ValueError, match="independent"):
        lattice.reciprocal([[1, 2], [2, 4]])
    assert lattice.volume(np.empty((0, 2, 2))).shape == (0,)
    assert lattice.reciprocal(np.empty((0, 2, 2))).shape == (0, 2, 2)
    assert lattice.diffr_orders_circle(np.eye(2), -1).shape == (0, 2)


@pytest.mark.parametrize("b", [np.diag([1.2, 0.8]), [[0, -1.2], [0.8, 0]]])
def test_diffraction_ordering_matches_orthogonal_reference(b):
    assert_array_equal(
        lattice.diffr_orders_circle(b, 3.5),
        reference_lattice.diffr_orders_circle(np.asarray(b), 3.5),
    )


@given(shear=st.floats(-3, 3), radius=st.floats(0, 4))
@settings(max_examples=35, deadline=None)
def test_skew_diffraction_is_complete_and_used_by_plane_basis(shear, radius):
    b = np.array([[1.0, shear], [0, 1]])
    candidates = np.array([(m, n) for m in range(-4, 5) for n in range(-16, 17)])
    expected = candidates[np.linalg.norm(candidates @ b, axis=1) <= radius]
    orders = lattice.diffr_orders_circle(b, radius)
    assert set(map(tuple, orders)) == set(map(tuple, expected))
    assert_array_equal(orders[1::2], -orders[2::2])
    a = lattice.reciprocal(b)
    basis = PlaneWaveBasisByComp.diffr_orders([0.1, 0.2], a, radius)
    assert_allclose(basis.components[::2], [0.1, 0.2] + orders @ b, atol=2e-14)


@pytest.mark.parametrize(
    "epsilon,mu,kappa",
    [(3.0, 1.2, 0.1), (3 + 0.2j, 1 - 0.1j, 0.2 + 0.3j), (-2 + 0.1j, 1, -0.3j)],
)
def test_material_helpers_and_shared_material_path(epsilon, mu, kappa):
    expected = reference_misc.refractive_index(epsilon, mu, kappa)
    value = misc.refractive_index(epsilon, mu, kappa)
    assert_allclose(value, expected, rtol=2e-15)
    assert value.dtype == expected.dtype
    assert_allclose(Material(epsilon, mu, kappa).nmp, value)
    kx = np.linspace(-2, 2, 27)
    ky = 0.3
    k = np.asarray(1.3 * value[0], complex)
    assert_allclose(
        misc.wave_vec_z(kx, ky, k),
        reference_misc.wave_vec_z(kx, ky, k),
        rtol=3e-15,
        atol=1e-14,
    )
    assert_allclose(
        Material(epsilon, mu, kappa).kzs(1.3, kx, ky, 0), misc.wave_vec_z(kx, ky, k)
    )
    output = np.zeros((5, 4), complex)[:, ::2]
    from treams_rs import _native

    _native.refractive_indices(np.full(5, epsilon), mu, kappa, out=output)
    assert_allclose(output, np.broadcast_to(value, output.shape))


def test_misc_scalar_axis_limits_and_broadcast():
    assert misc.wave_vec_z(0, 0, 0) == 0
    assert misc.wave_vec_z(1, 0, 1) == 0
    assert misc.wave_vec_z(2, 0, 1).imag > 0
    assert_allclose(
        misc.refractive_index(np.array([[2], [3]]), 1, np.array([0.1, 0.2, 0.3])),
        reference_misc.refractive_index(
            np.array([[2], [3]]), 1, np.array([0.1, 0.2, 0.3])
        ),
    )


@given(count=st.integers(0, 20))
def test_mode_change_involution_and_selection(count):
    columns = (
        np.array([(i, 0, pol) for i in range(count) for pol in (0, 1)]).reshape(-1, 3).T
    )
    change = misc.basischange(columns)
    assert_allclose(change @ change, np.eye(2 * count), atol=3e-16)
    assert_array_equal(misc.pickmodes(columns, columns), np.eye(2 * count, dtype=bool))
    if count:
        subset = columns[:, ::3]
        assert_allclose(
            misc.basischange(subset, columns),
            reference_misc.basischange(subset, columns),
        )
        assert_array_equal(
            misc.pickmodes(subset, columns), reference_misc.pickmodes(subset, columns)
        )


@given(k=st.floats(-100, 100), b=st.floats(0.5, 5), shift=st.integers(-20, 20))
def test_brillouin_scalar_periodicity(k, b, shift):
    result = misc.firstbrillouin1d(k, b)
    assert -0.5 * b < result <= 0.5 * b
    shifted = misc.firstbrillouin1d(k + shift * b, b)
    # Equivalent representatives may lie on opposite sides of a floating-point cell boundary.
    assert abs(result - shifted) < 5e-13 or abs(abs(result - shifted) - b) < 5e-13
    assert_allclose(result, reference_misc.firstbrillouin1d(k, b), atol=3e-14)


@pytest.mark.parametrize("dim", [2, 3])
def test_brillouin_native_reference_and_input_ownership(dim):
    b = np.eye(dim) * np.arange(1, dim + 1)
    if dim == 2:
        b[1, 0] = 0.2
    k = np.array([7.1, -6.2, 4.5])[:dim]
    original = k.copy()
    function = getattr(misc, f"firstbrillouin{dim}d")
    expected = getattr(reference_misc, f"firstbrillouin{dim}d")(k.copy(), b.copy())
    result = function(k, b)
    assert_allclose(result, expected, atol=3e-14)
    assert_array_equal(k, original)
    assert_allclose(function(result, b), result, atol=3e-14)
