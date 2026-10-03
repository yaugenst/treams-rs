"""Spherical and cylindrical vector-wave fields: references, identities, pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
import treams.special as upstream_special
from hypothesis import example, given, settings
from hypothesis import strategies as st

# _native: the spherical-wave jet is a test hook without a public wrapper.
from treams_rs import CylindricalBasis, Material, SphericalBasis, _native, diff
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal, to_oracle


def _mode_sum_floors(coefficients, points, basis, ks, cotangent, **kwargs):
    """Sums over modes and points of the term moduli of the field and its VJP.

    Reordering the modes reorders these sums, which may then move by ulps of the
    moduli of their terms however much those cancel. Returns the floors of the
    field and of the (coefficients, points, origins, ks) cotangents.
    """
    value = np.zeros((len(points), 3))
    floors = [np.zeros(len(basis)), np.zeros((len(points), 3)), 0.0, 0.0]
    for p, point in enumerate(points):
        for i in range(len(basis)):
            single = np.zeros_like(coefficients)
            single[i] = coefficients[i]
            term, context = diff.field(single, [point], basis, ks, **kwargs)
            value[p] += np.abs(term[0])
            terms = [np.abs(t) for t in context.pullback(cotangent[p : p + 1])]
            if i == 0:
                # Coefficient cotangents do not depend on the coefficients.
                floors[0] += terms[0]
            floors[1][p] += terms[1][0]
            floors[2] += terms[2]
            floors[3] += terms[3]
    return value, floors


@pytest.mark.physics
@pytest.mark.gradients
@given(
    order=st.integers(1, 4),
    x=st.floats(0.1, 0.8),
    singular=st.booleans(),
    helicity=st.booleans(),
    chiral=st.booleans(),
)
# Origin cotangents of about 1e4 whose sum has a -4.27 entry, 3.6e-12 apart.
@example(order=4, x=0.31478578000682234, singular=True, helicity=True, chiral=False)
def test_cylindrical_polarization_pair_permutation(
    order, x, singular, helicity, chiral
):
    basis = CylindricalBasis.default([0.2], order)
    permutation = np.r_[np.arange(0, len(basis), 2), np.arange(1, len(basis), 2)]
    separated = CylindricalBasis(np.asarray(basis.modes)[permutation])
    coefficients = np.arange(1, len(basis) + 1) * (0.07 + 0.02j)
    points = [[x, 0.15, 0.3], [-0.4, 0.2, 0.1]]
    ks = [1.3 + 0.03j, 1.3 + 0.03j + (0.2 if chiral and helicity else 0)]
    kwargs = {"poltype": "helicity" if helicity else "parity", "singular": singular}
    value, context = diff.field(coefficients, points, basis, ks, **kwargs)
    other, other_context = diff.field(
        coefficients[permutation], points, separated, ks, **kwargs
    )
    cotangent = np.array([[0.3 + 0.2j, -0.1j, 0.8], [0.1, -0.2, 0.2j]])
    gradient = context.pullback(cotangent)
    other_gradient = other_context.pullback(cotangent)
    value_floor, floors = _mode_sum_floors(
        coefficients, points, basis, ks, cotangent, **kwargs
    )
    # The permutation reorders sums over modes: the worst change over 3e4 draws was
    # 4.7 ulps of their term moduli.
    pairs = zip(
        [value, gradient[0][permutation], *gradient[1:]],
        [other, *other_gradient],
        [value_floor, floors[0][permutation], *floors[1:]],
        strict=True,
    )
    for expected, actual, floor in pairs:
        np.testing.assert_array_less(
            np.abs(np.asarray(actual) - expected),
            2e-13 * np.abs(expected) + 16 * np.finfo(float).eps * floor,
        )


@pytest.mark.gradients
@pytest.mark.parametrize("offset", [1e-9, 1e-100, 1e-200, 1e-300])
def test_cylindrical_field_near_axis_gradient(offset):
    basis = CylindricalBasis.default([0.2], 3)

    def evaluate(x):
        value, context = diff.field(
            np.ones(len(basis), complex), [[x, -x, 0.1]], basis, [1.3, 1.3]
        )
        return value, context.pullback(np.ones_like(value))

    actual, gradients = evaluate(offset)
    expected, axis_gradients = evaluate(0)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    for a, b in zip(gradients, axis_gradients, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-8)


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("axis", [False, True])
def test_cylindrical_field_reference_and_pullback(poltype, singular, axis):
    rng = np.random.default_rng(913)
    positions = np.array([[0, 0, 0], [0, 0, -0.2]])
    basis = CylindricalBasis.default([0.2, -0.3], 3, 2, positions)
    material = Material((2.3 + 0.1j, 1.1, 0.07 if poltype == "helicity" else 0))
    ks = material.ks(1.3)
    coefficients = complex_normal(rng, len(basis))
    points = np.array([[0.2, 0.1, 0.4], [-0.3, 0.2, -0.1]])
    if axis:
        points[:, :2] = 0
    if axis and singular:
        with pytest.raises(ValueError, match="singular"):
            diff.field(coefficients, points, basis, ks, poltype=poltype, singular=True)
        return

    value = diff.field(
        coefficients, points, basis, ks, poltype=poltype, singular=singular
    )[0]
    expected = (
        treams.efield(
            points,
            basis=to_oracle(basis),
            k0=1.3,
            material=treams.Material(material.epsilon, material.mu, material.kappa),
            poltype=poltype,
            modetype="singular" if singular else "regular",
        )
        @ coefficients
    )
    np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
    g = complex_normal(rng, value.shape)
    inputs = [coefficients, points, positions, ks]
    directions = [
        0.1
        * (
            rng.normal(size=v.shape)
            + (1j * rng.normal(size=v.shape) if np.iscomplexobj(v) else 0)
        )
        for v in inputs
    ]
    if poltype == "parity":
        # The parity basis requires equal wavenumbers.
        directions[-1][:] = directions[-1][0]
    check_pullback(
        lambda c, p, o, k: diff.field(
            c,
            p,
            CylindricalBasis(basis.modes, o),
            k,
            poltype=poltype,
            singular=singular,
        ),
        *inputs,
        directions=tuple(directions),
        cotangents=g,
        step=1e-6,
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.gradients
@given(radius=st.floats(0.1, 0.4), kz=st.floats(-0.5, 0.5))
def test_cylinder_scattered_field_advect(radius, kz):
    basis = CylindricalBasis.default([kz], 2)

    def objective(radius):
        matrix = ad.cylinder([kz], 2, 1.3, anp.reshape(radius, (1,)), [3 + 0.1j, 1])
        amplitudes = matrix @ anp.ones(len(basis))
        field = ad.field(
            amplitudes,
            [[0.8, 0.4, 0.1]],
            [[0, 0, 0]],
            [1.3, 1.3],
            basis=basis,
            singular=True,
        )
        return anp.sum(anp.real(field * anp.conj(field)))

    check_gradient(
        objective,
        advect.grad(objective),
        np.array(radius),
        directions=(np.array(1.0),),
        step=1e-5,
        rtol=1e-6,
        atol=1e-10,
    )


@pytest.mark.reference
@pytest.mark.parametrize(
    "position",
    [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0.2, -0.1, 0.3), (1e-9, -1e-9, 1e-9)],
)
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("helicity", [False, True])
def test_vector_waves_reference(position, singular, helicity):
    if singular and np.linalg.norm(position) == 0:
        with pytest.raises(ValueError, match="singular"):
            _native.spherical_wave_jet(
                (1, 0, 1), 1.2 + 0.1j, position, helicity, singular
            )
        return
    spherical = upstream_special.car2sph(position)
    for degree in range(1, 5):
        for order in range(-degree, degree + 1):
            for pol in (0, 1):
                args = (degree, order, (1.2 + 0.1j) * spherical[0], *spherical[1:])
                if helicity:
                    function = (
                        upstream_special.vsw_A if singular else upstream_special.vsw_rA
                    )
                    expected = function(*args, pol)
                else:
                    function = {
                        (False, 0): upstream_special.vsw_rM,
                        (False, 1): upstream_special.vsw_rN,
                        (True, 0): upstream_special.vsw_M,
                        (True, 1): upstream_special.vsw_N,
                    }[singular, pol]
                    expected = function(*args)
                expected = upstream_special.vsph2car(expected, spherical)
                actual = _native.spherical_wave_jet(
                    (degree, order, pol), 1.2 + 0.1j, position, helicity, singular
                )[0]
                # Cartesian components can cancel near the outgoing singularity.
                # Judge error relative to the whole vector, including zero components.
                error = np.linalg.norm(np.asarray(actual) - expected)
                assert error <= 3e-13 + 2e-11 * np.linalg.norm(expected)


@pytest.mark.gradients
@pytest.mark.parametrize("position", [(0, 0, 0), (0, 0, 1), (0.2, -0.3, 0.4)])
@pytest.mark.parametrize("mode", [(1, 0, 0), (1, -1, 1), (2, 1, 1), (3, -2, 0)])
def test_field_position_and_wavenumber_derivatives(position, mode):
    k = 1.2 + 0.1j
    _, jacobian, dk = _native.spherical_wave_jet(mode, k, position, True, False)
    step = 1e-6
    for axis in range(3):
        direction = np.eye(3)[axis]
        plus = _native.spherical_wave_jet(
            mode, k, tuple(np.asarray(position) + step * direction), True, False
        )[0]
        minus = _native.spherical_wave_jet(
            mode, k, tuple(np.asarray(position) - step * direction), True, False
        )[0]
        numerical = (np.array(plus) - minus) / (2 * step)
        np.testing.assert_allclose(
            np.array(jacobian)[:, axis], numerical, atol=2e-10, rtol=3e-8
        )
    direction = 0.3 - 0.2j
    plus = _native.spherical_wave_jet(
        mode, k + step * direction, position, True, False
    )[0]
    minus = _native.spherical_wave_jet(
        mode, k - step * direction, position, True, False
    )[0]
    np.testing.assert_allclose(
        np.array(dk) * direction,
        (np.array(plus) - minus) / (2 * step),
        atol=2e-10,
        rtol=3e-8,
    )


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
def test_batched_fields_reference(poltype, singular):
    basis = SphericalBasis.default(3, nmax=2, positions=[[0, 0, 0], [0.1, 0.2, 0.3]])
    oracle_basis = to_oracle(basis)
    points = np.array([[0.7, -0.1, 1.0], [-0.5, 0.4, 0.2], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(491)
    amplitudes = complex_normal(rng, len(basis))
    material = treams.Material(2.0 + 0.1j, 1.1, 0.05 if poltype == "helicity" else 0)
    ks = material.ks(1.2)
    actual, _ = diff.field(
        amplitudes, points, basis, ks, poltype=poltype, singular=singular
    )
    expected = (
        np.asarray(
            treams.efield(
                points,
                basis=oracle_basis,
                k0=1.2,
                material=material,
                poltype=poltype,
                modetype="singular" if singular else "regular",
            )
        )
        @ amplitudes
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.physics
@pytest.mark.gradients
@given(scale=st.floats(0.7, 1.4), offset=st.floats(-0.2, 0.2), singular=st.booleans())
@settings(max_examples=10)
def test_batched_field_pullback_all_inputs_and_translation_invariance(
    scale, offset, singular
):
    rng = np.random.default_rng(945)
    basis = SphericalBasis.default(2, nmax=2, positions=[[0, 0, 0], [0.1, 0.2, 0.3]])
    amplitudes = scale * complex_normal(rng, len(basis))
    points = np.array([[0.8 + offset, 0.1, 1.0], [0, 0, 1.0]])
    ks = np.array([1.1 + 0.1j, 1.3 + 0.1j])
    weight = complex_normal(rng, (2, 3))
    inputs = [amplitudes, points, basis.positions, ks]
    directions = [
        rng.normal(size=v.shape)
        + (1j * rng.normal(size=v.shape) if np.iscomplexobj(v) else 0)
        for v in inputs
    ]

    def forward(values):
        return diff.field(
            values[0],
            values[1],
            SphericalBasis(basis.modes, positions=values[2]),
            values[3],
            singular=singular,
        )

    gradients = forward(inputs)[1].pullback(weight)
    check_pullback(
        lambda *values: forward(values),
        *inputs,
        directions=tuple(directions),
        cotangents=weight,
        step=1e-6,
        rtol=3e-7,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        gradients[1].sum(axis=0) + gradients[2].sum(axis=0), 0, atol=2e-8
    )


@pytest.mark.gradients
def test_advect_field_composes_through_native_interaction():
    basis = SphericalBasis.default(1)
    incident = np.arange(6) * (0.02 + 0.03j)
    points = np.array([[0.1, 0.2, 1.1], [0, 0, 1.0]])

    def loss(radii):
        matrix = ad.sphere(1, 1.2, radii, [3.0, 1.0])
        field = ad.field(
            matrix @ incident,
            points,
            [[0, 0, 0]],
            [1.2, 1.2],
            basis=basis,
            singular=True,
        )
        return np.sum(np.abs(field) ** 2)

    check_gradient(
        loss,
        advect.grad(loss),
        np.array([0.3]),
        directions=(np.array([1.0]),),
        step=1e-6,
        rtol=2e-6,
        atol=1e-10,
    )
