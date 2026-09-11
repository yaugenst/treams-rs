"""Metadata algebra and its integration with native periodic operations."""

import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from treams_rs import (
    CylindricalWaveBasis,
    Lattice,
    PlaneWaveBasisByComp,
    SphericalWaveBasis,
    WaveVector,
    expandlattice,
    lattice,
)


@given(
    pitches=st.lists(st.floats(0.5, 4), min_size=3, max_size=3),
    turns=st.integers(-5, 5),
)
@settings(max_examples=25, deadline=None)
def test_lattice_set_and_permutation_algebra(pitches, turns):
    cell = Lattice(pitches)
    x, y, z = (Lattice(cell, axis) for axis in "xyz")
    assert (x | y) | z == cell
    assert cell & (x | y) == Lattice(cell, "xy")
    assert x <= cell
    assert x.isdisjoint(y)
    assert cell.permute(turns).permute(-turns) == cell
    assert (x | z).permute(turns) == x.permute(turns) | z.permute(turns)
    assert_allclose(
        cell.reciprocal @ np.asarray(cell).T, 2 * np.pi * np.eye(3), atol=3e-14
    )
    expected = treams.Lattice(pitches).permute(turns)
    assert_array_equal(np.asarray(cell.permute(turns)), expected[...])
    assert cell.permute(turns).alignment == expected.alignment


@given(
    values=st.lists(st.floats(-3, 3), min_size=3, max_size=3),
    left=st.integers(0, 7),
    right=st.integers(0, 7),
    turns=st.integers(-3, 3),
)
@settings(max_examples=35, deadline=None)
def test_partial_wavevector_constraint_algebra(values, left, right, turns):
    a = [value if left & (1 << i) else np.nan for i, value in enumerate(values)]
    b = [value if right & (1 << i) else np.nan for i, value in enumerate(values)]
    x, y = WaveVector(a), WaveVector(b)
    reference_x, reference_y = treams.WaveVector(a), treams.WaveVector(b)
    for actual, expected in (
        (x & y, reference_x & reference_y),
        (x | y, reference_x | reference_y),
    ):
        assert_allclose(actual, expected, equal_nan=True)
    assert (x <= y) == (reference_x <= reference_y)
    assert x.isdisjoint(y) == reference_x.isdisjoint(reference_y)
    assert x & y == y & x
    assert x | y == y | x
    assert (x & y) <= x
    assert x <= (x | y)
    assert x.permute(turns).permute(-turns) == x


def test_lattice_ownership_and_metadata_errors():
    data = np.diag([1.0, 2, 3])
    cell = Lattice(data)
    data[:] = 9
    assert_array_equal(cell[...], np.diag([1, 2, 3]))
    with pytest.raises(ValueError):
        cell[...][0, 0] = 4
    with pytest.raises(ValueError):
        Lattice([[1, 1], [0, 1]], "xy")._sublattice("x")
    with pytest.raises(ValueError):
        Lattice(1) | Lattice(2)
    with pytest.raises(ValueError):
        WaveVector([1, 2]) & WaveVector([1, 3])
    assert Lattice(1) & None is None
    assert WaveVector(0.3, "x") == WaveVector([0.3, np.nan, np.nan])
    assert Lattice.hexagonal(1, 2).permute().alignment == "xyz"


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
def test_plane_diffraction_metadata_survives_operations(alignment):
    cell = Lattice([2, 3], alignment)
    vector = WaveVector([0.1, 0.2], alignment)
    basis = PlaneWaveBasisByComp.diffr_orders(vector, cell, 2.5)
    assert basis.lattice == cell
    assert basis.kpar == vector
    assert basis.alignment == alignment
    selected = basis[::-2]
    assert selected.lattice == cell
    assert selected.kpar == vector
    assert (basis & selected).lattice == cell
    permuted = selected.permute()
    assert permuted.lattice == cell.permute()
    assert permuted.kpar == vector.permute()
    unit = selected.byunitvector(5)
    assert unit.lattice == cell
    assert unit.kpar == vector
    restored = unit.bycomp(5, alignment)
    assert restored.lattice == cell
    assert restored.kpar == vector
    assert_allclose(restored.components, selected.components, atol=1e-14)


def test_axial_diffraction_metadata_and_sublattice():
    cell = Lattice([2, 3, 4])
    basis = CylindricalWaveBasis.diffr_orders(0.2, 1, cell, 2)
    assert basis.lattice == Lattice(4)
    assert basis.kpar == WaveVector(0.2)
    assert basis[:3].lattice == Lattice(4)
    assert basis[:3].kpar == WaveVector(0.2)


@given(phi=st.floats(-3, 3), pitch=st.floats(1, 3))
@settings(max_examples=25, deadline=None)
def test_rotated_diffraction_basis_preserves_reciprocal_and_bloch_geometry(phi, pitch):
    cell = Lattice.hexagonal(pitch)
    basis = PlaneWaveBasisByComp.diffr_orders([0.1, 0.2], cell, 4)
    c, s = np.cos(phi), np.sin(phi)
    rotation = np.array([[c, -s], [s, c]])
    for original in (basis, basis.byunitvector(10)):
        rotated = original.rotate(phi)
        assert_allclose(
            np.asarray(rotated.lattice), np.asarray(cell) @ rotation.T, atol=2e-14
        )
        assert_allclose(
            rotated.lattice.reciprocal, cell.reciprocal @ rotation.T, atol=2e-14
        )
        assert_allclose(np.asarray(rotated.kpar)[:2], rotation @ [0.1, 0.2])
        assert_allclose(rotated.rotate(-phi).kvecs(10), original.kvecs(10), atol=2e-14)
        assert_allclose(np.asarray(rotated.rotate(-phi).lattice), cell, atol=2e-14)
    assert_allclose(
        (basis.components - [0.1, 0.2]) @ np.asarray(cell).T,
        (basis.rotate(phi).components - rotation @ [0.1, 0.2])
        @ np.asarray(cell.rotate(phi)).T,
        atol=2e-14,
    )


@pytest.mark.parametrize("alignment", ["x", "y", "z", "xy", "yz", "zx", "xyz"])
def test_quarter_turn_metadata_and_unrepresentable_spans(alignment):
    cell = Lattice(np.arange(1, len(alignment) + 1), alignment)
    vector = WaveVector(np.arange(1, len(alignment) + 1), alignment)
    assert cell.rotate(np.pi / 2).rotate(-np.pi / 2) == cell
    assert vector.rotate(np.pi / 2).rotate(-np.pi / 2) == vector
    if alignment in ("x", "y", "yz", "zx"):
        with pytest.raises(ValueError, match="Cartesian alignment"):
            cell.rotate(0.3)
        with pytest.raises(ValueError, match="Cartesian alignment"):
            vector.rotate(0.3)
    else:
        assert_allclose(np.asarray(cell.rotate(0.3).rotate(-0.3)), cell, atol=2e-15)
        assert_allclose(vector.rotate(0.3).rotate(-0.3), vector, atol=2e-15)


@pytest.mark.parametrize("spherical", [True, False])
def test_periodic_geometry_metadata_reaches_native_solver(spherical):
    basis = (
        SphericalWaveBasis.default(1)
        if spherical
        else CylindricalWaveBasis.default([0.2], 1)
    )
    axis = "z" if spherical else "x"
    cell = Lattice(1.7, axis)
    bloch = WaveVector(0.1, axis)
    expected = expandlattice(1.7, 0.1, basis=basis, k0=1.3, eta=0.7)
    assert_allclose(
        expandlattice(cell, bloch, basis=basis, k0=1.3, eta=0.7),
        expected,
        rtol=3e-14,
        atol=1e-13,
    )
    basis.lattice, basis.kpar = cell, bloch
    assert_allclose(
        expandlattice(basis=basis, k0=1.3, eta=0.7), expected, rtol=3e-14, atol=1e-13
    )
    full = Lattice([1.7, 2.1, 1.7])
    assert_allclose(
        expandlattice(full, bloch, basis=basis, k0=1.3, eta=0.7),
        expected,
        rtol=3e-14,
        atol=1e-13,
    )
    wrong = Lattice(1.7, "y")
    with pytest.raises(ValueError, match="sublattice"):
        expandlattice(wrong, bloch, basis=basis, k0=1.3, eta=0.7)


def test_public_lattice_sum_accepts_metadata_and_diagonal_cells():
    cell = Lattice([1.7, 2.1])
    bloch = WaveVector([0.1, 0.2])
    expected = lattice.lsumsw2d(
        2, 1, 1.3 + 0.1j, [0.1, 0.2], np.diag([1.7, 2.1]), [0.2, 0.1], 0.7
    )
    assert_allclose(
        lattice.lsumsw2d(2, 1, 1.3 + 0.1j, bloch, cell, [0.2, 0.1], 0.7), expected
    )
    assert_allclose(
        lattice.lsumsw2d(2, 1, 1.3 + 0.1j, bloch, [1.7, 2.1], [0.2, 0.1], 0.7), expected
    )
