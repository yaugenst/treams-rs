"""Cylindrical reference, physical-invariant and full analytic derivative checks."""

import numpy as np
import pytest
import scipy.special as sc
import treams
from hypothesis import given
from hypothesis import strategies as st

from treams_rs import _native, coeffs


@pytest.mark.oracle_numerical
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


@pytest.mark.oracle_numerical
@pytest.mark.parametrize("order", [-8, -1, 0, 1, 2, 10, 30])
@pytest.mark.parametrize("z", [1e-7 + 1e-8j, 0.2 + 0.1j, 5 + 0.3j])
def test_cylindrical_bessel_jets(order, z):
    for outgoing in (False, True):
        actual = _native.cylindrical(order, z, outgoing)
        function = sc.h1vp if outgoing else sc.jvp
        expected = [function(order, z, n) for n in range(3)]
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)


@pytest.mark.physics
@given(
    radius=st.floats(0.1, 1.5),
    epsilon=st.floats(1.1, 6),
    order=st.integers(-4, 4),
    kz=st.floats(-0.5, 0.5),
)
def test_cylinder_conservation_and_layer_split(radius, epsilon, order, kz):
    args = (kz, order, 1.3)
    one = coeffs.mie_cyl(*args, [radius], [epsilon, 1], [1, 1], [0.02, 0])
    split = coeffs.mie_cyl(
        *args, [0.5 * radius, radius], [epsilon, epsilon, 1], [1, 1, 1], [0.02, 0.02, 0]
    )
    zero = coeffs.mie_cyl(*args, [radius], [epsilon, epsilon], [1, 1], [0.02, 0.02])
    np.testing.assert_allclose(one, split, rtol=2e-10, atol=3e-12)
    np.testing.assert_allclose(zero, 0, atol=2e-12)
    np.testing.assert_allclose(
        -np.trace(one).real, np.sum(abs(one) ** 2), rtol=2e-10, atol=2e-12
    )


@pytest.mark.ad_contract
@pytest.mark.parametrize("parameter", ["kz", "k0", "radii", "epsilon", "mu", "kappa"])
def test_cylinder_all_input_pullbacks(parameter):
    args = dict(
        kz=0.2,
        m=-1,
        k0=1.1,
        radii=np.array([0.2, 0.5]),
        epsilon=np.array([2 + 0.1j, 3 + 0.2j, 1.1 + 0.01j]),
        mu=np.array([1.2, 1.1, 1.0], complex),
        kappa=np.array([0.02, 0.01, 0.005], complex),
    )
    value, context = coeffs.mie_cyl_with_context(**args)
    rng = np.random.default_rng(4)
    cotangent = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = dict(
        zip(
            ["kz", "k0", "radii", "epsilon", "mu", "kappa"],
            context.pullback(cotangent),
            strict=True,
        )
    )
    direction = rng.normal(size=np.shape(args[parameter]))
    if parameter in ("epsilon", "mu", "kappa"):
        direction = direction + 1j * rng.normal(size=direction.shape)
    plus, minus = dict(args), dict(args)
    h = 2e-6
    plus[parameter] = args[parameter] + h * direction
    minus[parameter] = args[parameter] - h * direction
    numerical = np.vdot(
        cotangent, coeffs.mie_cyl(**plus) - coeffs.mie_cyl(**minus)
    ).real / (2 * h)
    np.testing.assert_allclose(
        np.vdot(gradients[parameter], direction).real, numerical, atol=2e-8, rtol=2e-6
    )
