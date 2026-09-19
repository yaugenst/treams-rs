import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

import treams_rs as tr


def _basis(family):
    positions = [[0.2, 0.3, 0.4], [1.0, 0.1, 0.2]]
    if family == "spherical":
        return tr.SphericalBasis.default(
            2, 2, positions
        ), treams.SphericalWaveBasis.default(2, 2, positions)
    if family == "cylindrical":
        return tr.CylindricalBasis.default(
            [0.2, -0.3], 2, 2, positions
        ), treams.CylindricalWaveBasis.default([0.2, -0.3], 2, 2, positions)
    if family == "unit":
        return tr.PlaneWaveBasis.default(
            [[1, 2, 3], [2, 0.3, 1]]
        ), treams.PlaneWaveBasisByUnitVector.default([[1, 2, 3], [2, 0.3, 1]])
    return tr.PlaneWavePorts.default(
        [[0.1, 0.2], [0.3, -0.2]], "yz"
    ), treams.PlaneWaveBasisByComp.default([[0.1, 0.2], [0.3, -0.2]], "yz")


@pytest.mark.parametrize("family", ["spherical", "cylindrical", "unit", "component"])
def test_basis_selection_reference_and_metadata(family):
    basis, expected = _basis(family)
    for i in (0, -1, np.int64(1)):
        assert_allclose(basis[i], expected[i], rtol=3e-16, atol=1e-16)
        assert basis[i] in basis
        assert basis.index(basis[i]) == int(i) % len(basis)
        assert basis.count(basis[i]) == 1
    for key in (
        slice(None),
        slice(None, None, -2),
        [3, 0, 3, 1],
        np.arange(len(basis)) % 2 == 0,
        Ellipsis,
        [],
        np.zeros(len(basis), bool),
    ):
        actual = basis[key]
        reference = expected[key]
        assert type(actual) is type(basis)
        assert_allclose(
            np.asarray(actual.modes),
            np.asarray(list(reference)),
            rtol=3e-16,
            atol=1e-16,
        )
        assert actual == actual[:]
        assert actual == actual[...]
        if family in ("spherical", "cylindrical"):
            assert_array_equal(actual.positions, basis.positions)
        if family == "component":
            # Upstream loses yz alignment when slicing. Preserve the physical basis.
            assert actual.alignment == "yz"
        for column in actual[()]:
            assert column.shape == (len(actual),)
    assert basis[:] == basis
    assert basis[::-1] != basis
    assert basis != 3
    if family == "unit":
        assert_array_equal(basis.directions, basis[:].directions)
    columns = basis[()]
    for actual, reference in zip(columns, expected[()], strict=True):
        assert_allclose(actual, reference, rtol=3e-16, atol=1e-16)
    alias = {
        "spherical": "plms",
        "cylindrical": "pzms",
        "unit": "xyzs",
        "component": "xys",
    }[family]
    for a, b in zip(getattr(basis, alias), columns, strict=True):
        assert_array_equal(a, b)
    with pytest.raises(AttributeError):
        _ = basis.unknown


@pytest.mark.parametrize("family", ["spherical", "cylindrical", "unit", "component"])
@given(indices=st.lists(st.integers(0, 15), max_size=30))
@settings(max_examples=35)
def test_basis_selection_field_reconstruction(family, indices):
    basis, _ = _basis(family)
    indices = [i % len(basis) for i in indices]
    selected = basis[indices]
    assert selected.modes == tuple(dict.fromkeys(basis.modes[i] for i in indices))
    if not len(selected):
        return
    points = [[0.4, 0.3, 0.2], [0.2, -0.1, 0.6]]
    operator = tr.operators.efield(points, basis=basis, k0=1.3)
    columns = [basis.index(mode) for mode in selected]
    assert_allclose(
        tr.operators.efield(points, basis=selected, k0=1.3),
        operator[:, :, columns],
        rtol=2e-13,
        atol=1e-13,
    )


@given(degree=st.integers(0, 8), particles=st.integers(1, 4), mmax=st.integers(0, 8))
@settings(max_examples=35)
def test_basis_ebcm_ordering_and_dimension_inverse(degree, particles, mmax):
    mmax = min(mmax, degree)
    basis = tr.SphericalBasis.ebcm(degree, particles, mmax)
    expected = treams.SphericalWaveBasis.ebcm(
        degree, particles, mmax, np.zeros((particles, 3))
    )
    assert basis.modes == tuple(tuple(row) for row in expected)
    dimension = tr.SphericalBasis.defaultdim(degree, particles)
    assert dimension == len(tr.SphericalBasis.default(degree, particles))
    assert tr.SphericalBasis.defaultlmax(dimension, particles) == degree
    keys = [(p, m, degree, -pol) for p, degree, m, pol in basis]
    assert keys == sorted(keys)


def test_basis_equality_includes_geometry_and_deduplicates_labels():
    spherical = tr.SphericalBasis.default(1)
    assert tr.SphericalBasis([*spherical, *spherical]) == spherical
    assert tr.SphericalBasis(spherical, [[0.1, 0, 0]]) != spherical
    component = tr.PlaneWavePorts.default([[0.1, 0.2]])
    assert tr.PlaneWavePorts(component, "yz") != component
    assert len(component[[]]) == 0
    assert component[[]].components.shape == (0, 2)
    assert component.components is component.components
    assert not component.components.flags.writeable


def test_ebcm_permutation_preserves_surface_integral():
    default = tr.SphericalBasis.default(2)
    blocks = tr.SphericalBasis.ebcm(2)
    indices = [default.index(mode) for mode in blocks]
    kwargs = {
        "r": lambda theta: 0.3 * (1 + 0.1 * np.cos(theta) ** 2),
        "dr": lambda theta: -0.06 * np.sin(theta) * np.cos(theta),
        "ks": [[1.7, 1.9], [1.2, 1.3]],
        "zs": [0.7, 1.1],
    }
    expected = tr.ebcm.qmat(**kwargs, out=default)
    assert_allclose(
        tr.ebcm.qmat(**kwargs, out=blocks),
        expected[np.ix_(indices, indices)],
        rtol=1e-13,
        atol=1e-13,
    )


def test_upstream_component_selection_alignment_regression():
    original = treams.PlaneWaveBasisByComp.default([[0.1, 0.2]], "yz")
    assert original[:].alignment == "xy"
    fixed = tr.PlaneWavePorts.default([[0.1, 0.2]], "yz")
    assert fixed[:].alignment == "yz"
    assert tr.SphericalBasis.default(
        1, positions=[0, 0, 0]
    ) == tr.SphericalBasis.default(1)


@pytest.mark.parametrize("family", ["spherical", "cylindrical", "unit", "component"])
@given(
    left=st.lists(st.integers(0, 15), max_size=25),
    right=st.lists(st.integers(0, 15), max_size=25),
)
@settings(max_examples=35)
def test_basis_ordered_set_algebra(family, left, right):
    basis, _ = _basis(family)
    a = basis[[i % len(basis) for i in left]]
    b = basis[[i % len(basis) for i in right]]
    aset, bset = set(a), set(b)
    assert (a | b).modes == tuple(dict.fromkeys((*a, *b)))
    assert (a & b).modes == tuple(mode for mode in b if mode in aset)
    assert (a - b).modes == tuple(mode for mode in a if mode not in bset)
    assert set(a ^ b) == aset ^ bset
    assert (a <= b) == (aset <= bset)
    assert (a < b) == (aset < bset)
    assert (a >= b) == (aset >= bset)
    assert (a > b) == (aset > bset)
    assert a.isdisjoint(b) == aset.isdisjoint(bset)
    assert ((a | b) - b).modes == (a - b).modes
    assert (a ^ a).modes == ()
    for result in (a | b, a & b, a - b, a ^ b):
        expected = basis[[basis.index(mode) for mode in result]]
        assert result == expected
        if family == "unit":
            assert_array_equal(result.directions, expected.directions)


def test_basis_set_geometry_contract_and_cylindrical_orders():
    a = tr.SphericalBasis.default(1)
    with pytest.raises(ValueError, match="origin"):
        _ = a | tr.SphericalBasis(a, [[0.1, 0, 0]])
    b = tr.PlaneWavePorts.default([[0.1, 0.2]], "yz")
    with pytest.raises(ValueError, match="alignment"):
        _ = b & tr.PlaneWavePorts(b, "zx")
    with pytest.raises(TypeError, match="family"):
        _ = a | b
    for period in (2 * np.pi, 1.7, -3.1):
        for cutoff in (0, 1, 7.0):
            expected = treams.CylindricalWaveBasis.diffr_orders(0.1, 2, period, cutoff)
            actual = tr.CylindricalBasis.diffr_orders(0.1, 2, period, cutoff)
            assert_allclose(np.array(actual.modes), np.array(list(expected)))
            assert np.all(np.abs(actual.kz - 0.1) <= cutoff + 1e-14)
