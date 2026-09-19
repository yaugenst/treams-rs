"""Native grouped power flux, complete port pullbacks and scattering composition."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("direction", ["up", "down"])
@settings(max_examples=12, deadline=None)
@given(scale=st.floats(0.7, 1.4))
def test_batched_power_complete_advect_and_scale(axis, direction, scale):
    rng = np.random.default_rng(114)
    matrices = (
        rng.normal(size=(2, 2, 4, 4)) + 1j * rng.normal(size=(2, 2, 4, 4))
    ) * 0.03
    matrices[0, 0] += np.eye(4) * 0.8
    matrices[1, 1] += np.eye(4) * 0.9
    incident = (rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))) * scale
    ks = np.array([[1.3 + 0.02j, 1.5 + 0.02j], [1.2 + 0.01j, 1.4 + 0.01j]])
    zs = np.array([0.9 + 0.01j, 1.1 - 0.01j])
    q = np.array([[0.2, 0.13], [0.31, -0.17]])
    modes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    arguments = (matrices, incident, ks, zs, q)
    weights = np.array([[0.3, -0.2], [0.1, 0.4]])

    def objective(*args):
        return anp.real(
            anp.sum(
                ad.smatrix_tr(*args, modes=modes, axis=axis, modetype=direction)
                * weights
            )
        )

    gradients = advect.grad(objective, argnums=(0, 1, 2, 3, 4))(*arguments)
    assert_allclose(np.vdot(gradients[1], incident).real, 0, atol=2e-12)
    value = objective(*arguments)
    assert_allclose(
        objective(matrices, incident * (1.3 + 0.2j), ks, zs, q), value, atol=2e-13
    )
    assert_allclose(
        objective(matrices, incident, ks * scale, zs, q * scale), value, atol=2e-13
    )
    spectral = np.vdot(gradients[2], ks).real + np.vdot(gradients[4], q).real
    assert_allclose(spectral, 0, atol=2e-12)
    for parameter in range(5):
        direction_array = rng.normal(size=arguments[parameter].shape)
        if np.iscomplexobj(arguments[parameter]):
            direction_array = direction_array + 1j * rng.normal(
                size=direction_array.shape
            )
        h = 2e-6
        plus, minus = list(arguments), list(arguments)
        plus[parameter] = arguments[parameter] + h * direction_array
        minus[parameter] = arguments[parameter] - h * direction_array
        expected = (objective(*plus) - objective(*minus)) / (2 * h)
        actual = np.vdot(gradients[parameter], direction_array).real
        assert_allclose(actual, expected, rtol=3e-7, atol=2e-8)


def test_power_context_owns_inputs_strides_and_invalid_cotangent_is_retriable():
    matrices = np.zeros((2, 2, 2, 2), complex)
    matrices[0, 0] = matrices[1, 1] = np.eye(2)
    incident = np.array([[0.2 + 0.3j], [0.4 - 0.1j]])
    ks, zs, q = np.full((2, 2), 1.3 + 0j), np.ones(2, complex), np.array([[0.2, 0.1]])
    modes = [(0, 0), (0, 1)]
    args = (matrices, incident, ks, zs, q)
    value, residual = tr.diff.smatrix_tr(*args, modes=modes)
    _, duplicate = tr.diff.smatrix_tr(*args, modes=modes)
    expected = duplicate.pullback(np.ones_like(value, dtype=complex))
    assert_allclose(value[:, 0], [1, 0], atol=1e-14)
    for a in args:
        a[:] = 0
    with pytest.raises(ValueError, match="shape"):
        residual.pullback(np.ones((3, 1), complex))
    with pytest.raises(ValueError, match="finite"):
        residual.pullback(np.full((2, 1), np.nan, complex))
    for actual, reference in zip(
        residual.pullback(np.ones_like(value, dtype=complex)), expected, strict=True
    ):
        assert_allclose(actual, reference, atol=0)
    with pytest.raises(ValueError, match="consumed"):
        residual.pullback(np.ones_like(value, dtype=complex))


@pytest.mark.parametrize("alignment,axis", [("xy", 2), ("yz", 0), ("zx", 1)])
@pytest.mark.parametrize("fixed_q", [False, True])
def test_interface_power_differentiated_energy_conservation(alignment, axis, fixed_q):
    ks = np.array([[1.3, 1.5], [1.1, 1.4]])
    zs = np.array([1.2, 0.9])
    q = np.array([[0, 0] if fixed_q else [0.2, 0.1]])
    incident = np.array([[0.3 + 0.2j], [0.4 - 0.1j]])
    modes = [(0, 0), (0, 1)]

    def objective(ks, zs, incident):
        matrix = ad.interface_coefficients(
            ks, zs, q[0], alignment=alignment, fixed_q=fixed_q
        )
        # Interface input media are below/above; S-matrix ports are above/below.
        powers = ad.smatrix_tr(
            matrix,
            incident,
            ks[::-1],
            zs[::-1],
            q,
            modes=modes,
            axis=axis,
            fixed_q=fixed_q,
        )
        return anp.real(anp.sum(powers))

    assert_allclose(objective(ks, zs, incident), 1, atol=2e-13)
    for gradient in advect.grad(objective, argnums=(0, 1, 2))(ks, zs, incident):
        assert_allclose(gradient, 0, atol=3e-12)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_cd_native_batch_reference_complete_gradient_and_undefined_limit(
    poltype, direction
):
    basis = tr.PlaneWavePorts.default([[0.2, 0.1]])
    stack = tr.SMatrix.slab(
        0.4,
        basis,
        1.3,
        [1, (2.3, 1.1, 0.08) if poltype == "helicity" else (2.3, 1.1), 1],
        poltype=poltype,
    )
    matrices, incident = stack.array, np.array([0.3 + 0.2j, 0.7 - 0.1j])
    ks, zs, q = np.full((2, 2), 1.3 + 0j), np.ones(2, complex), np.array([[0.2, 0.1]])
    modes = [(0, 0), (0, 1)]
    actual = ad.smatrix_cd(
        matrices, incident, ks, zs, q, modes=modes, poltype=poltype, modetype=direction
    )
    assert_allclose(actual, stack.cd(incident, modetype=direction), atol=2e-13)

    def objective(matrices, incident):
        return anp.real(
            anp.sum(
                ad.smatrix_cd(
                    matrices,
                    incident,
                    ks,
                    zs,
                    q,
                    modes=modes,
                    poltype=poltype,
                    modetype=direction,
                )
            )
        )

    gradients = advect.grad(objective, argnums=(0, 1))(matrices, incident)
    rng = np.random.default_rng(115)
    for i, value in enumerate((matrices, incident)):
        delta = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
        args = [matrices, incident]
        plus, minus = args.copy(), args.copy()
        plus[i] = value + 2e-6 * delta
        minus[i] = value - 2e-6 * delta
        assert_allclose(
            np.vdot(gradients[i], delta).real,
            (objective(*plus) - objective(*minus)) / 4e-6,
            rtol=2e-7,
            atol=2e-9,
        )
    assert_allclose(np.vdot(gradients[1], incident).real, 0, atol=1e-12)
    reflected = np.zeros_like(matrices)
    reflected[0, 1] = reflected[1, 0] = np.eye(2)
    with pytest.raises(ValueError, match="undefined"):
        objective(reflected, incident)
    with pytest.raises(ValueError, match="undefined"):
        advect.grad(objective)(reflected, incident)
