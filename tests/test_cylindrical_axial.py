import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs import diff


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
def test_cylindrical_field_axial_pullbacks_and_operator(poltype, singular):
    origins = np.array([[0.1, -0.2, 0.0], [0.3, 0.4, -0.1]])
    basis = tr.CylindricalWaveBasis.default([-0.3, 0.25], 2, 2, origins)
    ks = np.array(
        [1.3 + 0.05j, 1.5 + 0.07j] if poltype == "helicity" else [1.3 + 0.05j] * 2
    )
    points = np.array([[0.5, -0.3, 0.2], [0.8, 0.5, 0.3], [0.1, -0.2, 0.4]])
    if singular:
        points[-1, 0] += 0.5
    rng = np.random.default_rng(55)
    amplitudes = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    kwargs = {"poltype": poltype, "singular": singular}
    field, context = diff.field(amplitudes, points, basis, ks, **kwargs)
    g = rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)
    gradients = context.pullback_axial(g)
    _, context = diff.field(amplitudes, points, basis, ks, **kwargs)
    for actual, expected in zip(gradients[:4], context.pullback(g), strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    operator, context = diff.field_operator(points, basis, ks, **kwargs)
    assert_allclose(operator @ amplitudes, field, rtol=1e-13, atol=1e-12)
    operator_gradient = context.pullback_axial(g[:, :, None] * amplitudes.conj())
    for actual, expected in zip(operator_gradient, gradients[1:], strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
    direction = rng.uniform(-0.1, 0.1, size=len(basis))
    h = 1e-6
    perturbed = [
        tr.CylindricalWaveBasis(
            [
                (p, kz + sign * h * step, m, pol)
                for (p, kz, m, pol), step in zip(basis.modes, direction, strict=True)
            ],
            origins,
        )
        for sign in (1, -1)
    ]
    difference = (
        diff.field(amplitudes, points, perturbed[0], ks, **kwargs)[0]
        - diff.field(amplitudes, points, perturbed[1], ks, **kwargs)[0]
    ) / (2 * h)
    assert_allclose(
        np.dot(gradients[-1], direction),
        np.vdot(g, difference).real,
        rtol=2e-7,
        atol=1e-7,
    )
    _, context = diff.field(amplitudes, points, basis, ks, **kwargs)
    amplitudes[:] = 0
    points[:] = 0
    ks[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback_axial(g[:1])
    for actual, expected in zip(
        context.pullback_axial(np.asfortranarray(g)), gradients, strict=True
    ):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@pytest.mark.parametrize(
    "name", ["field", "hfield", "gfield", "ffield", "field_operator"]
)
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_advect_field_family_axial(name, poltype):
    basis = tr.CylindricalWaveBasis.default([-0.2, 0.3], 1)
    amplitudes = np.arange(len(basis)) * 0.1 + 0.2j
    points = np.array([[0.4, -0.3, 0.2], [0, 0, 0.3]])
    ks = np.array(
        [1.3 + 0.05j, 1.5 + 0.07j] if poltype == "helicity" else [1.3 + 0.05j] * 2
    )

    def objective(kzs):
        kwargs = {"basis": basis, "poltype": poltype, "kzs": kzs}
        if name == "field_operator":
            value = (
                ad.field_operator(points, basis.positions, ks, **kwargs) @ amplitudes
            )
        else:
            args = (amplitudes, points, basis.positions, ks)
            if name == "hfield":
                args = (*args, 0.8)
            if name in ("gfield", "ffield"):
                args = (1, *args)
            value = getattr(ad, name)(*args, **kwargs)
        return anp.sum(anp.abs(value) ** 2)

    kzs = basis.kz
    gradient = advect.grad(objective)(kzs)
    direction = np.linspace(-0.2, 0.3, len(basis))
    h = 1e-6
    assert_allclose(
        np.dot(gradient, direction),
        (objective(kzs + h * direction) - objective(kzs - h * direction)) / (2 * h),
        rtol=2e-7,
        atol=1e-8,
    )


@given(scale=st.floats(0.6, 1.8), kz=st.floats(-0.4, 0.4))
@settings(max_examples=25)
def test_cylindrical_axial_geometric_scale_invariance(scale, kz):
    basis = tr.CylindricalWaveBasis.default([kz], 2)
    points = np.array([[0.4, -0.3, 0.2], [0, 0, 0.3]])
    ks = np.array([1.3 + 0.05j, 1.5 + 0.07j])
    amplitudes = np.full(len(basis), 0.2 + 0.3j)
    expected, _ = diff.field(amplitudes, points, basis, ks)
    scaled = tr.CylindricalWaveBasis.default([kz / scale], 2)
    actual, context = diff.field(amplitudes, points * scale, scaled, ks / scale)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    gradient = context.pullback_axial(np.full_like(actual, 0.3 + 0.2j))
    assert_allclose(
        np.sum(gradient[1] * points * scale),
        np.vdot(gradient[3], ks / scale).real + np.dot(gradient[4], scaled.kz),
        rtol=1e-11,
        atol=1e-12,
    )


def test_complete_cylinder_scattered_field_axial_gradient():
    kzs = np.array([-0.2, 0.3])
    basis = tr.CylindricalWaveBasis.default(kzs, 1)
    points = np.array([[0.6, 0.3, 0.2], [0.4, -0.5, 0.7]])
    illumination = np.linspace(0.1, 0.5, len(basis)) + 0.2j

    def objective(kzs, radius):
        response = ad.cylinder(
            kzs,
            1,
            1.3,
            anp.reshape(radius, (1,)),
            [3.1 + 0.1j, 1.1 + 0.04j],
            [1.2, 1.0],
            [0.07, 0.02],
        )
        medium = tr.Material(1.1 + 0.04j, 1, 0.02)
        value = ad.field(
            response @ illumination,
            points,
            basis.positions,
            medium.ks(1.3),
            basis=basis,
            singular=True,
            kzs=anp.repeat(kzs, 6),
        )
        return anp.sum(anp.abs(value) ** 2)

    gradient = advect.grad(objective, argnums=(0, 1))(kzs, np.array(0.2))
    h = 1e-6
    direction = np.array([0.1, -0.2])
    expected = (
        objective(kzs + h * direction, 0.2 + h * 0.1)
        - objective(kzs - h * direction, 0.2 - h * 0.1)
    ) / (2 * h)
    assert_allclose(
        np.dot(gradient[0], direction) + gradient[1] * 0.1,
        expected,
        rtol=2e-7,
        atol=1e-10,
    )


def test_axial_field_empty_samples_and_parameter_contract():
    basis = tr.CylindricalWaveBasis.default([0.1, 0.3], 1)
    coefficients = np.ones(len(basis), complex)
    points = np.empty((0, 3))
    _, context = diff.field(coefficients, points, basis, [1.3, 1.3])
    for g in context.pullback_axial(np.empty((0, 3), complex)):
        assert_allclose(g, 0)
    _, context = diff.field_operator(points, basis, [1.3, 1.3])
    for g in context.pullback_axial(np.empty((0, 3, len(basis)), complex)):
        assert_allclose(g, 0)
    for kzs, message in [
        (np.ones(len(basis) - 1), "one real"),
        (np.zeros(len(basis)), "distinct"),
        (basis.kz + 0.1j, "real"),
    ]:
        with pytest.raises(ValueError, match=message):
            ad.field_operator(points, basis.positions, [1.3, 1.3], basis=basis, kzs=kzs)
    spherical = tr.SphericalWaveBasis.default(1)
    with pytest.raises(ValueError, match="cylindrical"):
        ad.field_operator(
            points,
            spherical.positions,
            [1.3, 1.3],
            basis=spherical,
            kzs=np.ones(len(spherical)),
        )
