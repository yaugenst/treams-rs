"""Cylindrical reference, physical-invariant and full analytic derivative checks."""

import advect
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st

from treams_rs import advect as ad
from treams_rs import coeffs, diff
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal, selecting


@pytest.mark.reference
@pytest.mark.parametrize("order", [-4, -1, 0, 2, 5])
@pytest.mark.parametrize("kz", [0, 0.3])
def test_multilayer_chiral_cylinder(order, kz):
    args = (
        kz,
        order,
        1.2,
        [0.2, 0.5],
        [2 + 0.1j, 3 + 0.2j, 1.1],
        [1.2, 1.1, 1],
        [0.03, 0.02, 0.01],
    )
    np.testing.assert_allclose(
        coeffs.mie_cyl(*args), treams.coeffs.mie_cyl(*args), rtol=3e-10, atol=3e-13
    )


@pytest.mark.physics
@given(
    radius=st.floats(0.1, 1.5),
    epsilon=st.floats(1.1, 6),
    order=st.integers(-4, 4),
    kz=st.floats(-0.5, 0.5),
)
def test_cylinder_zero_contrast_and_layer_split(radius, epsilon, order, kz):
    args = (kz, order, 1.3)
    one = coeffs.mie_cyl(*args, [radius], [epsilon, 1], [1, 1], [0.02, 0])
    split = coeffs.mie_cyl(
        *args, [0.5 * radius, radius], [epsilon, epsilon, 1], [1, 1, 1], [0.02, 0.02, 0]
    )
    zero = coeffs.mie_cyl(*args, [radius], [epsilon, epsilon], [1, 1], [0.02, 0.02])
    np.testing.assert_allclose(one, split, rtol=2e-10, atol=3e-12)
    np.testing.assert_allclose(zero, 0, atol=2e-12)


@pytest.mark.gradients
def test_cylinder_all_input_pullbacks():
    layers = dict(
        radii=np.array([0.2, 0.5]),
        epsilon=np.array([2 + 0.1j, 3 + 0.2j, 1.1 + 0.01j]),
        mu=np.array([1.2, 1.1, 1.0], complex),
        kappa=np.array([0.02, 0.01, 0.005], complex),
    )

    def record(kz, k0, radii, epsilon, mu, kappa):
        return diff.mie_cyl(
            kz=float(kz),
            order=-1,
            k0=float(k0),
            radii=radii,
            epsilon=epsilon,
            mu=mu,
            kappa=kappa,
        )

    value = coeffs.mie_cyl(kz=0.2, m=-1, k0=1.1, **layers)
    check_pullback(
        selecting(record, *range(6)),
        np.array(0.2),
        np.array(1.1),
        *layers.values(),
        cotangents=complex_normal(np.random.default_rng(4), value.shape),
        step=2e-6,
        rtol=2e-6,
        atol=2e-8,
        seed=4,
    )


@pytest.mark.reference
@pytest.mark.parametrize("mmax", [0, 1, 3])
@pytest.mark.parametrize("kzs", [[0.0], [0.2, -0.3]])
def test_full_cylinder_matrix_reference(mmax, kzs):
    materials = [(3.0 + 0.1j, 1.1, 0.02), (2.0 + 0.2j, 1.2, 0.03), (1.0, 1.0, 0.01)]
    epsilon, mu, kappa = np.asarray(materials).T
    actual, _ = diff.cylinder(kzs, mmax, 1.2, [0.2, 0.4], epsilon, mu, kappa)
    expected = treams.TMatrixC.cylinder(kzs, mmax, 1.2, [0.2, 0.4], materials)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-12)


@pytest.mark.gradients
@given(radius=st.floats(0.15, 0.4), epsilon=st.floats(1.2, 5.0), kz=st.floats(0.1, 0.5))
@settings(max_examples=10)
def test_advect_cylinder_all_parameter_derivatives(radius, epsilon, kz):
    values = [
        np.array([kz, -0.3]),
        np.array(1.2),
        np.array([radius]),
        np.array([epsilon + 0.1j, 1.0]),
        np.array([1.1, 1.0]),
        np.array([0.02, 0.0]),
    ]
    rng = np.random.default_rng(732)
    directions = [
        rng.normal(size=v.shape)
        + (1j * rng.normal(size=v.shape) if np.iscomplexobj(v) else 0)
        for v in values
    ]

    def loss(*parameters):
        matrix = ad.cylinder(parameters[0], 2, *parameters[1:])
        return np.sum(np.sin(np.real(matrix)) + 0.2 * np.imag(matrix) ** 2)

    check_gradient(
        loss,
        advect.grad(loss, argnums=tuple(range(6))),
        *values,
        directions=tuple(directions),
        step=1e-6,
        rtol=2e-6,
        atol=1e-8,
    )
