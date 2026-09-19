"""Matrix-level workflows reuse the explicit native operators and metadata."""

import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

import treams_rs as tr


@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize(
    "field", ["efield", "hfield", "dfield", "bfield", "gfield", "ffield"]
)
@pytest.mark.filterwarnings("ignore:.*scipy.special.sph_harm.*:DeprecationWarning")
def test_tmatrix_field_and_basis_selection(cylindrical, field):
    ours = (
        tr.CylindricalTMatrix.cylinder([0.2], 2, 1.3, 0.2, [3, 1])
        if cylindrical
        else tr.TMatrix.sphere(2, 1.3, 0.2, [3, 1])
    )
    reference = (
        treams.TMatrixC.cylinder([0.2], 2, 1.3, [0.2], [3, 1])
        if cylindrical
        else treams.TMatrix.sphere(2, 1.3, [0.2], [3, 1])
    )
    points = np.array([[0.7, 0.3, 0.4], [-0.3, 0.6, 0.1]])
    args = (1, points) if field in ("gfield", "ffield") else (points,)
    expected = (
        getattr(treams, field)(
            *args,
            basis=reference.basis,
            k0=reference.k0,
            material=reference.material,
            poltype=reference.poltype,
            modetype="singular",
        )
        @ reference
        if field in ("gfield", "ffield")
        else getattr(reference, field)(*args)
    )
    assert_allclose(getattr(ours, field)(*args), expected, atol=2e-13)
    selected = ours.basis[::-2]
    assert_allclose(ours[selected].array, ours.array[::-2, ::-2])
    assert ours[selected].basis == selected
    foreign = type(selected)(list(selected), positions=[[1, 2, 3]])
    assert_array_equal(ours[foreign].basis.positions, ours.basis.positions)
    assert_array_equal(ours[:, 0], ours.array[:, 0])


@given(scale=st.floats(0.3, 3), offset=st.floats(-2, 2))
@settings(max_examples=25)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_exclusion_masks_scale_and_translation(cylindrical, scale, offset):
    family = tr.CylindricalTMatrix if cylindrical else tr.TMatrix
    base = (
        tr.CylindricalTMatrix.cylinder([0.2], 1, 1.3, 0.2, [3, 1])
        if cylindrical
        else tr.TMatrix.sphere(1, 1.3, 0.2, [3, 1])
    )
    origins = np.array([[0, 0, 0], [0.8, 0.3, 0.2]])
    points = np.random.default_rng(32).normal(size=(4, 5, 3))
    radii = np.array([0.2, 0.3])
    first = family._assemble([base, base], origins)
    second = family._assemble([base, base], origins * scale + offset)
    assert_array_equal(
        first.valid_points(points, radii),
        second.valid_points(points * scale + offset, radii * scale),
    )
    assert not first.valid_points(origins, radii).any()
    if cylindrical:
        assert_array_equal(
            first.valid_points(points[..., :2], radii),
            first.valid_points(points, radii),
        )


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.filterwarnings("ignore:.*where.*used without.*out.*:UserWarning")
@given(turns=st.integers(-3, 3), shift=st.floats(-0.4, 0.4))
@settings(max_examples=12, deadline=None)
def test_smatrix_sparse_coordinate_transforms(alignment, poltype, turns, shift):
    directions = [[0.2, 0.3], [0.3, 0.1]]
    basis = tr.PlaneWavePorts.default(directions, alignment)[::-1]
    # Upstream slicing loses non-xy alignment; reconstruct the intended basis.
    oracle_basis = treams.PlaneWaveBasisByComp(list(basis), alignment=alignment)
    rng = np.random.default_rng(3)
    values = rng.normal(size=(2, 2, 4, 4)) + 1j * rng.normal(size=(2, 2, 4, 4))
    materials = (
        ((1.7, 1.1, 0.03), (1.2, 1.3, 0.02)) if poltype == "helicity" else (1.7, 1.2)
    )
    ours = tr.SMatrix(values, k0=1.3, basis=basis, material=materials, poltype=poltype)
    oracle = treams.SMatrices(
        values,
        k0=1.3,
        basis=oracle_basis,
        material=tuple(treams.Material(m) for m in materials),
        poltype=poltype,
    )
    translated = ours.translate([shift, 0.1, -0.2])
    assert_allclose(
        translated.array, np.asarray(oracle.translate([shift, 0.1, -0.2])), atol=2e-13
    )
    assert_allclose(translated.translate([-shift, -0.1, 0.2]).array, values, atol=2e-13)
    changed = ours.permute(turns)
    assert_allclose(changed.array, np.asarray(oracle.permute(turns)), atol=2e-13)
    assert_allclose(changed.permute(-turns).array, values, atol=2e-13)
    assert changed.basis.alignment == basis.permute(turns).alignment
    if alignment == "xy":
        assert_allclose(
            ours.rotate(shift).rotate(-shift).basis.components,
            basis.components,
            atol=2e-14,
        )
        assert_array_equal(ours.rotate(shift).array, values)


def test_material_radial_branch():
    medium = tr.Material(2.3 + 0.1j, 1.2, 0.03)
    assert tr.Material(np.array(medium())) == medium
    kz = np.array([0.2, 0.4, 2.3])[:, None]
    radial = medium.krhos(1.3, kz)
    assert_allclose(radial, treams.Material(*medium()).krhos(1.3, kz, [0, 1]))
    assert_allclose(
        radial**2 + kz**2, np.broadcast_to(medium.ks(1.3) ** 2, radial.shape)
    )
    assert np.all(radial.imag >= 0)
