"""Cylindrical translations: reference, Cartesian limits and analytic derivatives."""

import advect
import numpy as np
import pytest
import treams
import treams.cw
from hypothesis import given, settings
from hypothesis import strategies as st

import treams_rs as tr

# _native: the single-pair translation jet is a test hook without a public wrapper.
from treams_rs import _native
from treams_rs import advect as ad
from treams_rs.testing import check_gradient

from _support import complex_normal


@pytest.mark.reference
@given(
    to=st.integers(-7, 7),
    source=st.integers(-7, 7),
    x=st.floats(-2.0, 2.0),
    y=st.floats(0.5, 2.0),
    singular=st.booleans(),
)
def test_cylindrical_translation_reference(to, source, x, y, singular):
    k, kz, z = 1.2 + 0.1j, 0.3, 0.4
    rho, phi = np.hypot(x, y), np.arctan2(y, x)
    krho = np.sqrt(k * k - kz * kz)
    expected = treams.cw.translate(
        kz, to, 1, kz, source, 1, krho * rho, phi, z, singular=singular
    )
    actual = _native.cylindrical_cartesian_translation_jet(
        (kz, to, 1), (kz, source, 1), k, (x, y, z), singular
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)


@pytest.mark.gradients
def test_cylindrical_cartesian_and_wavenumber_derivatives():
    # Binding smoke test on the axis; the native translation_derivative_recurrences
    # property checks every derivative exactly for all orders, branches and kinds.
    position, order = np.array([0.0, 0.0, 0.4]), 1
    k, kz, h = 1.2 + 0.1j, 0.3, 1e-6

    def translation(k=k, kz=kz, position=position):
        return _native.cylindrical_cartesian_translation_jet(
            (kz, 0, 1), (kz, order, 1), k, tuple(position), False
        )

    _, gradient, dk, dkz = translation()

    def difference(along_k=0, along_kz=0, along_position=0):
        plus, minus = (
            translation(
                k + s * along_k, kz + s * along_kz, position + s * along_position
            )[0]
            for s in (h, -h)
        )
        return (plus - minus) / (2 * h)

    for axis in range(3):
        numeric = difference(along_position=np.eye(3)[axis])
        np.testing.assert_allclose(gradient[axis], numeric, rtol=2e-7, atol=2e-10)
    numeric = difference(along_k=0.3 + 0.2j)
    np.testing.assert_allclose(dk * (0.3 + 0.2j), numeric, rtol=2e-7, atol=2e-10)
    np.testing.assert_allclose(dkz, difference(along_kz=1), rtol=2e-7, atol=2e-10)
    np.testing.assert_allclose(
        np.dot(gradient, position), k * dk + kz * dkz, rtol=2e-10, atol=1e-11
    )


@pytest.mark.interface
def test_cylindrical_mode_selection_and_singular_axis():
    jet = _native.cylindrical_cartesian_translation_jet
    assert jet((0.2, 1, 0), (0.3, 1, 0), 1.2, (1, 1, 0), True)[0] == 0
    assert jet((0.2, 1, 0), (0.2, 1, 1), 1.2, (1, 1, 0), True)[0] == 0
    assert jet((0.2, 1, 0), (0.2, 1, 0), 1.2, (0, 0, 0), True)[0] == 0
    with pytest.raises(ValueError, match="singular"):
        jet((0.2, 1, 0), (0.2, 1, 0), 1.2, (0, 0, 1), True)


@pytest.mark.workflows
@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_cylinder_cluster_and_global_expansion(poltype):
    kzs, k0 = [0.2, -0.3], 1.2
    positions = [[0, 0, 0], [1.2, 0.3, 0.2]]
    radii = [0.2, 0.25]
    material = [(3.0, 1.1, 0.02), (1.0, 1.0, 0.0)]
    native = [
        tr.CylindricalTMatrix.cylinder(kzs, 2, k0, radius, material, poltype=poltype)
        for radius in radii
    ]
    reference = [
        treams.TMatrixC.cylinder(kzs, 2, k0, radius, material) for radius in radii
    ]
    if poltype == "parity":
        reference = [matrix.changepoltype(poltype) for matrix in reference]
    for actual, expected in zip(native, reference, strict=True):
        np.testing.assert_allclose(actual.array, expected, rtol=2e-10, atol=1e-12)
        np.testing.assert_allclose(actual.xw_ext_avg, expected.xw_ext_avg, rtol=1e-10)
        np.testing.assert_allclose(actual.xw_sca_avg, expected.xw_sca_avg, rtol=1e-10)
        np.testing.assert_allclose(actual.xw_ext_avg, actual.xw_sca_avg, rtol=1e-10)
        np.testing.assert_allclose(actual.krhos, expected.krhos, rtol=1e-12)
    actual = tr.Cluster(native, positions=positions).solve()
    expected = treams.TMatrixC.cluster(reference, positions).interaction.solve()
    assert isinstance(actual, tr.CylindricalTMatrix)
    np.testing.assert_allclose(actual.array, expected, rtol=3e-10, atol=2e-12)
    actual = actual.expand(tr.CylindricalBasis.default(kzs, 8))
    expected = expected.expand(treams.CylindricalWaveBasis.default(kzs, 8))
    np.testing.assert_allclose(actual.array, expected, rtol=3e-10, atol=2e-12)
    np.testing.assert_allclose(actual.xw_ext_avg, expected.xw_ext_avg, rtol=3e-10)
    np.testing.assert_allclose(actual.xw_sca_avg, expected.xw_sca_avg, rtol=3e-10)
    rng = np.random.default_rng(773)
    incident = complex_normal(rng, len(actual))
    oracle_incident = treams.PhysicsArray(
        incident,
        basis=expected.basis,
        k0=k0,
        material=expected.material,
        poltype=poltype,
        modetype="regular",
    )
    np.testing.assert_allclose(
        actual.xw(incident), expected.xw(oracle_incident), rtol=3e-10, atol=1e-12
    )


@pytest.mark.gradients
@given(x=st.floats(0.6, 2.0), imaginary=st.floats(0.01, 0.2))
@settings(max_examples=15)
def test_cylindrical_expansion_advect_vjp(x, imaginary):
    basis = tr.CylindricalBasis.default([0.2, -0.3], 2)
    positions = np.array([[x, 0.2, 0.3]])
    ks = np.array([1.2 + imaginary * 1j, 1.3 + imaginary * 1j])

    def loss(position, wavenumbers):
        matrix = ad.expansion(
            position,
            [[0, 0, 0]],
            wavenumbers,
            destination=basis,
            source=basis,
            singular=True,
        )
        return np.sum(np.sin(np.real(matrix)) + np.imag(matrix) * 0.03)

    check_gradient(
        loss,
        advect.grad(loss, argnums=(0, 1)),
        positions,
        ks,
        directions=(np.array([[0.1, -0.2, 0.3]]), np.array([0.2 + 0.1j, -0.1 + 0.3j])),
        step=1e-6,
        rtol=2e-6,
        atol=1e-6,
    )


@pytest.mark.interface
def test_cylindrical_basis_and_matrix_family_contract():
    basis = tr.CylindricalBasis.default([0.0], 1)
    assert next(iter(basis)) == (0, 0.0, -1, 1)
    assert tr.CylindricalBasis.defaultmmax(len(basis)) == 1
    with pytest.raises(ValueError, match="wave family"):
        tr.TMatrix(np.eye(6), k0=1.2, basis=basis)
    with pytest.raises(ValueError, match="wave family"):
        tr.CylindricalTMatrix(np.eye(6), k0=1.2, basis=tr.SphericalBasis.default(1))
    sphere = tr.TMatrix.sphere(1, 1.2, 0.2, [3.0, 1.0])
    cylinder = tr.CylindricalTMatrix.cylinder([0.0], 1, 1.2, 0.2, [3.0, 1.0])
    with pytest.raises(ValueError, match="cluster"):
        tr.Cluster([sphere, cylinder], positions=[[0, 0, 0], [1, 0, 0]])
    assert tr.CylindricalBasis.default([0.2, 0.2], 1) == tr.CylindricalBasis.default(
        [0.2], 1
    )
