"""Vector spherical waves checked against treams and Maxwell identities."""

import numpy as np
import pytest
import treams.special as sc
from hypothesis import given
from hypothesis import strategies as st

from treams_rs import _native


@pytest.mark.parametrize("offset", [1e-9, 1e-100, 1e-200, 1e-300])
def test_cylindrical_field_near_axis_gradient(offset):
    from treams_rs import CylindricalWaveBasis, diff

    basis = CylindricalWaveBasis.default([0.2], 3)

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


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("outgoing", [False, True])
@pytest.mark.parametrize("axis", [False, True])
def test_cylindrical_field_reference_and_pullback(poltype, outgoing, axis):
    import treams

    from treams_rs import CylindricalWaveBasis, Material, diff

    rng = np.random.default_rng(913)
    positions = np.array([[0, 0, 0], [0, 0, -0.2]])
    basis = CylindricalWaveBasis.default([0.2, -0.3], 3, 2, positions)
    material = Material((2.3 + 0.1j, 1.1, 0.07 if poltype == "helicity" else 0))
    ks = material.ks(1.3)
    coefficients = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    points = np.array([[0.2, 0.1, 0.4], [-0.3, 0.2, -0.1]])
    if axis:
        points[:, :2] = 0
    if axis and outgoing:
        with pytest.raises(ValueError, match="singular"):
            diff.field(coefficients, points, basis, ks, poltype=poltype, singular=True)
        return

    value, ctx = diff.field(
        coefficients, points, basis, ks, poltype=poltype, singular=outgoing
    )
    expected = (
        treams.efield(
            points,
            basis=treams.CylindricalWaveBasis(basis.modes, positions),
            k0=1.3,
            material=treams.Material(material.epsilon, material.mu, material.kappa),
            poltype=poltype,
            modetype="singular" if outgoing else "regular",
        )
        @ coefficients
    )
    np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
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
        directions[-1][:] = directions[-1][0]

    def shifted(h):
        c, p, o, k = [v + h * d for v, d in zip(inputs, directions, strict=True)]
        return diff.field(
            c,
            p,
            CylindricalWaveBasis(basis.modes, o),
            k,
            poltype=poltype,
            singular=outgoing,
        )[0]

    gradients = ctx.pullback(g)
    numeric = np.vdot(g, (shifted(1e-6) - shifted(-1e-6)) / (2e-6)).real
    analytic = sum(
        np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True)
    )
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-7)


@given(radius=st.floats(0.1, 0.4), kz=st.floats(-0.5, 0.5))
def test_cylinder_scattered_field_advect(radius, kz):
    import advect
    import advect.numpy as anp

    from treams_rs import CylindricalWaveBasis
    from treams_rs import advect as ad

    basis = CylindricalWaveBasis.default([kz], 2)

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

    gradient = advect.grad(objective)(np.array(radius))
    h = 1e-5
    np.testing.assert_allclose(
        gradient,
        (objective(radius + h) - objective(radius - h)) / (2 * h),
        rtol=1e-6,
        atol=1e-10,
    )


@pytest.mark.oracle_numerical
@pytest.mark.filterwarnings(
    "ignore:`scipy.special.sph_harm` is deprecated.*:DeprecationWarning"
)
@pytest.mark.parametrize(
    "position",
    [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0.2, -0.1, 0.3), (1e-9, -1e-9, 1e-9)],
)
@pytest.mark.parametrize("outgoing", [False, True])
@pytest.mark.parametrize("helicity", [False, True])
def test_vector_waves_reference(position, outgoing, helicity):
    if outgoing and np.linalg.norm(position) == 0:
        with pytest.raises(ValueError, match="singular"):
            _native.spherical_wave((1, 0, 1), 1.2 + 0.1j, position, helicity, outgoing)
        return
    spherical = sc.car2sph(position)
    for degree in range(1, 5):
        for order in range(-degree, degree + 1):
            for pol in (0, 1):
                args = (degree, order, (1.2 + 0.1j) * spherical[0], *spherical[1:])
                if helicity:
                    function = sc.vsw_A if outgoing else sc.vsw_rA
                    expected = function(*args, pol)
                else:
                    function = {
                        (False, 0): sc.vsw_rM,
                        (False, 1): sc.vsw_rN,
                        (True, 0): sc.vsw_M,
                        (True, 1): sc.vsw_N,
                    }[outgoing, pol]
                    expected = function(*args)
                expected = sc.vsph2car(expected, spherical)
                actual = _native.spherical_wave(
                    (degree, order, pol), 1.2 + 0.1j, position, helicity, outgoing
                )[0]
                # Cartesian components can cancel near the outgoing singularity.
                # Judge error relative to the whole vector, including zero components.
                error = np.linalg.norm(np.asarray(actual) - expected)
                assert error <= 3e-13 + 2e-11 * np.linalg.norm(expected)


@pytest.mark.physics
@given(
    degree=st.integers(1, 6),
    selector=st.integers(0, 20),
    x=st.floats(-1, 1),
    y=st.floats(-1, 1),
    z=st.floats(0.5, 2),
    pol=st.integers(0, 1),
    outgoing=st.booleans(),
)
def test_helicity_eigenfield_and_zero_divergence(
    degree, selector, x, y, z, pol, outgoing
):
    k = 1.2 + 0.1j
    order = selector % (2 * degree + 1) - degree
    value, derivative, _ = _native.spherical_wave(
        (degree, order, pol), k, (x, y, z), True, outgoing
    )
    derivative = np.array(derivative)
    curl = np.array(
        [
            derivative[2, 1] - derivative[1, 2],
            derivative[0, 2] - derivative[2, 0],
            derivative[1, 0] - derivative[0, 1],
        ]
    )
    scale = max(1, np.linalg.norm(value))
    np.testing.assert_allclose(
        curl, (2 * pol - 1) * k * np.array(value), rtol=2e-10, atol=2e-10 * scale
    )
    np.testing.assert_allclose(np.trace(derivative), 0, atol=2e-10 * scale)


@pytest.mark.ad_contract
@pytest.mark.parametrize("position", [(0, 0, 0), (0, 0, 1), (0.2, -0.3, 0.4)])
@pytest.mark.parametrize("mode", [(1, 0, 0), (1, -1, 1), (2, 1, 1), (3, -2, 0)])
def test_field_position_and_wavenumber_derivatives(position, mode):
    k = 1.2 + 0.1j
    _, jacobian, dk = _native.spherical_wave(mode, k, position, True, False)
    step = 1e-6
    for axis in range(3):
        direction = np.eye(3)[axis]
        plus = _native.spherical_wave(
            mode, k, tuple(np.asarray(position) + step * direction), True, False
        )[0]
        minus = _native.spherical_wave(
            mode, k, tuple(np.asarray(position) - step * direction), True, False
        )[0]
        numerical = (np.array(plus) - minus) / (2 * step)
        np.testing.assert_allclose(
            np.array(jacobian)[:, axis], numerical, atol=2e-10, rtol=3e-8
        )
    direction = 0.3 - 0.2j
    plus = _native.spherical_wave(mode, k + step * direction, position, True, False)[0]
    minus = _native.spherical_wave(mode, k - step * direction, position, True, False)[0]
    np.testing.assert_allclose(
        np.array(dk) * direction,
        (np.array(plus) - minus) / (2 * step),
        atol=2e-10,
        rtol=3e-8,
    )


@pytest.mark.oracle_numerical
@pytest.mark.filterwarnings(
    "ignore:`scipy.special.sph_harm` is deprecated.*:DeprecationWarning"
)
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
def test_batched_fields_reference(poltype, singular):
    import treams

    from treams_rs import SphericalWaveBasis, diff

    basis = SphericalWaveBasis.default(
        3, nmax=2, positions=[[0, 0, 0], [0.1, 0.2, 0.3]]
    )
    oracle_basis = treams.SphericalWaveBasis(basis.modes, positions=basis.positions)
    points = np.array([[0.7, -0.1, 1.0], [-0.5, 0.4, 0.2], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(491)
    amplitudes = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
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


@pytest.mark.ad_contract
@given(scale=st.floats(0.7, 1.4), offset=st.floats(-0.2, 0.2), singular=st.booleans())
def test_batched_field_pullback_all_inputs_and_translation_invariance(
    scale, offset, singular
):
    from treams_rs import SphericalWaveBasis, diff

    rng = np.random.default_rng(945)
    basis = SphericalWaveBasis.default(
        2, nmax=2, positions=[[0, 0, 0], [0.1, 0.2, 0.3]]
    )
    amplitudes = scale * (
        rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    )
    points = np.array([[0.8 + offset, 0.1, 1.0], [0, 0, 1.0]])
    ks = np.array([1.1 + 0.1j, 1.3 + 0.1j])
    weight = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
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
            SphericalWaveBasis(basis.modes, positions=values[2]),
            values[3],
            singular=singular,
        )

    _, context = forward(inputs)
    gradients = context.pullback(weight)
    step = 1e-6
    numerical = np.real(
        np.vdot(
            weight,
            (
                forward(
                    [v + step * d for v, d in zip(inputs, directions, strict=True)]
                )[0]
                - forward(
                    [v - step * d for v, d in zip(inputs, directions, strict=True)]
                )[0]
            )
            / (2 * step),
        )
    )
    analytic = sum(
        np.real(np.vdot(g, d)) for g, d in zip(gradients, directions, strict=True)
    )
    np.testing.assert_allclose(analytic, numerical, rtol=3e-7, atol=2e-7)
    np.testing.assert_allclose(
        gradients[1].sum(axis=0) + gradients[2].sum(axis=0), 0, atol=2e-8
    )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(weight)


@pytest.mark.ad_contract
def test_advect_field_composes_through_native_interaction():
    import advect

    from treams_rs import SphericalWaveBasis
    from treams_rs import advect as ad

    basis = SphericalWaveBasis.default(1)
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

    radii = np.array([0.3])
    h = 1e-6
    np.testing.assert_allclose(
        advect.grad(loss)(radii)[0],
        (loss(radii + h) - loss(radii - h)) / (2 * h),
        rtol=2e-6,
        atol=1e-10,
    )
