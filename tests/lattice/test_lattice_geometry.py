"""Lattice geometry through treams_rs.lattice against treams.lattice.

Checks the volume and reciprocal cell gufuncs, cube and cubeedge, and the
enumeration and pairing of the diffraction orders that the plane-wave basis uses.
"""

import numpy as np
import pytest
import treams.lattice as upstream_lattice
from hypothesis import example, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less

from treams_rs import PlaneWavePorts, lattice


@pytest.mark.interface
@pytest.mark.reference
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_cell_gufuncs_and_strides(dim, dtype):
    rng = np.random.default_rng(37)
    cells = rng.integers(-4, 5, (17, dim, dim)).astype(dtype)
    cells += 12 * np.eye(dim, dtype=dtype)
    expected = upstream_lattice.volume(cells)
    value = lattice.volume(cells)
    assert value.dtype == expected.dtype
    assert_array_equal(value, expected)
    # Transposed, reversed cores and strided or aliased outputs of both loops and
    # both core sizes; the ufunc contract's layout cases use only float64 3x3 cells.
    target = np.zeros(34, dtype=dtype)[::2]
    lattice.volume(cells[::-1].swapaxes(-1, -2), out=target)
    assert_array_equal(target, expected[::-1])
    reciprocal = lattice.reciprocal(cells)
    assert_allclose(reciprocal, upstream_lattice.reciprocal(cells), rtol=4e-15)
    output = np.empty((17, dim, dim * 2))[:, :, ::2].swapaxes(-1, -2)
    lattice.reciprocal(cells, out=output)
    assert_allclose(output, reciprocal)
    lattice.reciprocal(output, out=output)
    assert_allclose(output, cells, atol=4e-14)
    assert lattice.area is lattice.volume


@pytest.mark.interface
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_single_cell_fast_path(dim):
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


@pytest.mark.reference
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("n", [0, 1, 4])
def test_cube_and_boundary_match_reference(dim, n):
    assert_array_equal(lattice.cube(dim, n), upstream_lattice.cube(dim, n))
    assert_array_equal(lattice.cubeedge(dim, n), upstream_lattice.cubeedge(dim, n))
    boundary = lattice.cubeedge(dim, n)
    assert len(boundary) == ((2 * n + 1) ** dim - (2 * n - 1) ** dim if n else 1)
    assert np.all(np.max(np.abs(boundary), axis=1) == n)


@pytest.mark.interface
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


@pytest.mark.reference
@pytest.mark.parametrize("b", [np.diag([1.2, 0.8]), [[0, -1.2], [0.8, 0]]])
def test_diffraction_ordering_matches_orthogonal_reference(b):
    assert_array_equal(
        lattice.diffr_orders_circle(b, 3.5),
        upstream_lattice.diffr_orders_circle(np.asarray(b), 3.5),
    )


@pytest.mark.interface
@given(shear=st.floats(-3, 3), radius=st.floats(0, 4))
@settings(max_examples=35)
# (1, -4) and (-1, 4) lie exactly on the cutoff circle: `b` keeps them and the
# reciprocal of `a` drops them.
@example(shear=1.6, radius=2.6)
def test_skew_diffraction_pairs_and_plane_basis(shear, radius):
    # Completeness, pairing and uniqueness against integer enumeration are the
    # Rust geometry::diffraction_orders_are_complete_paired_and_unique property;
    # here the binding keeps opposite orders paired and feeds them to the plane
    # basis.
    b = np.array([[1.0, shear], [0, 1]])
    orders = lattice.diffr_orders_circle(b, radius)
    assert_array_equal(orders[0], [0, 0])
    assert_array_equal(orders[1::2], -orders[2::2])
    a = lattice.reciprocal(b)
    basis = PlaneWavePorts.diffr_orders([0.1, 0.2], a, radius)
    # The basis enumerates the reciprocal of `a`, which is `b` only to rounding, so
    # the two may disagree on orders whose length rounds to the cutoff (26 of 29170
    # decimal shears and radii, each at distance 0 from it). Components are sums of
    # |q| + |n| |b| in magnitude and round to ulps of that (at most 1.5 ulps over
    # 2e4 random and the decimal draws).
    used = lattice.diffr_orders_circle(basis.lattice.reciprocal, radius)
    eps = np.finfo(float).eps
    for order in {tuple(n) for n in orders} ^ {tuple(n) for n in used}:
        floor = np.linalg.norm(np.abs(order) @ np.abs(b))
        assert abs(np.linalg.norm(order @ b) - radius) <= 4 * eps * floor, order
    expected = [0.1, 0.2] + used @ b
    floor = np.abs([0.1, 0.2]) + np.abs(used) @ np.abs(b)
    assert_array_less(
        np.abs(basis.components[::2] - expected),
        1e-7 * np.abs(expected) + 4 * eps * floor,
    )
