import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad


def test_density_boundary_contract():
    for alignment in ("yz", "zx"):
        with pytest.raises(ValueError, match="xy-aligned"):
            tr.chirality_density(
                tr.PlaneWaveBasisByComp.default([[0.2, 0.3]], alignment), 1.3
            )
    with pytest.raises(ValueError, match="nonzero"):
        tr.diff.chirality_density([0], [1])
    with pytest.raises(ValueError, match="matching"):
        tr.diff.chirality_density([1], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        tr.diff.chirality_density([1], [1], [0, np.nan])


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_density_reordered_partial_basis(poltype):
    basis = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [1.7, 0.4]])
    indices = [3, 0, 1]
    partial = tr.PlaneWaveBasisByComp([basis.modes[i] for i in indices])
    complete = tr.chirality_density(basis, 1.3, poltype=poltype, z=(-0.2, 0.7))
    assert_allclose(
        tr.chirality_density(partial, 1.3, poltype=poltype, z=(-0.2, 0.7)),
        np.asarray(complete)[:, indices][:, :, indices],
    )


def _contract(forms, up, down):
    return np.real(
        np.vdot(up, forms[0] @ up)
        + np.vdot(down, forms[1] @ down)
        + np.vdot(down, forms[2] @ up)
    )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("epsilon", [2.3, 2.3 + 0.2j])
def test_chirality_at_origin_reference(poltype, epsilon):
    basis = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [3.0, 0.4]])
    material = (epsilon, 1.2, 0.06 if poltype == "helicity" else 0)
    expected = treams.chirality_density(
        treams.PlaneWaveBasisByComp(basis.modes), 1.3, material, poltype
    )
    assert_allclose(tr.chirality_density(basis, 1.3, material, poltype), expected)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("q", [[0.2, 0.3], [3.0, 0.4]])
def test_density_agrees_with_coherent_cartesian_field_integral(poltype, q):
    basis = tr.PlaneWaveBasisByComp.default([q])
    material = tr.Material(
        (2.3 + 0.2j, 1.2 + 0.1j, 0.07 if poltype == "helicity" else 0)
    )
    up, down = np.array([0.7 + 0.2j, -0.3j]), np.array([0.4j, 0.1 + 0.3j])
    nodes, weights = np.polynomial.legendre.leggauss(40)
    for interval in [(0.0, 0.0), (0.4, 0.4), (-0.2, 0.7), (0.2, 0.2 + 1e-10)]:
        z = 0.5 * (sum(interval) + nodes * (interval[1] - interval[0]))
        points = z[:, None] * np.eye(3)[basis.normal_axis]
        fields = [
            sum(
                kind(
                    points,
                    basis=basis,
                    k0=1.3,
                    material=material,
                    poltype=poltype,
                    modetype=direction,
                )
                @ amplitudes
                for direction, amplitudes in [("up", up), ("down", down)]
            )
            for kind in (tr.efield, tr.hfield)
        ]
        expected = weights @ np.real(
            np.sum(fields[0].conj() * (1j * material.impedance * fields[1]), axis=-1)
        )
        forms = tr.chirality_density(basis, 1.3, material, poltype, interval)
        assert_allclose(_contract(forms, up, down), expected, rtol=1e-12, atol=1e-12)
        assert_allclose(
            forms[:2], np.asarray(forms[:2]).conj().swapaxes(-1, -2), atol=1e-14
        )


@given(k0=st.floats(0.8, 2.0), start=st.floats(-2, 2), stop=st.floats(-2, 2))
def test_single_propagating_wave_density_is_interval_independent(k0, start, stop):
    basis = tr.PlaneWaveBasisByComp.default([[0.2, 0.3]])
    up, down, _ = tr.chirality_density(basis, k0, z=(start, stop))
    assert_allclose(up, np.diag(2 * (2 * basis.pol - 1)), atol=1e-14)
    assert_allclose(down, up, atol=1e-14)


@pytest.mark.parametrize("z", [[-0.2, 0.7], [0.4, 0.4], [0.2, 0.2 + 1e-10]])
def test_chirality_native_complex_pullback(z):
    ks = np.array([1.3 + 0.2j, 1.6 + 0.1j, 1.0])
    normal = np.array([1.2 + 0.1j, 0.3 + 0.5j, 1.0j])
    values = [ks, normal, np.array(z)]
    rng = np.random.default_rng(86)
    g = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    _, context = tr.diff.chirality_density(*values)
    gradients = context.pullback(g)
    for i, value in enumerate(values):
        direction = rng.normal(size=value.shape) * 0.1
        if i < 2:
            direction = direction + 0.03j
        plus, minus = list(values), list(values)
        h = 1e-5
        plus[i], minus[i] = value + h * direction, value - h * direction
        expected = np.vdot(
            g,
            (tr.diff.chirality_density(*plus)[0] - tr.diff.chirality_density(*minus)[0])
            / (2 * h),
        ).real
        assert_allclose(
            np.vdot(gradients[i], direction).real, expected, rtol=1e-7, atol=1e-9
        )
    assert_allclose(
        np.vdot(gradients[0], ks).real
        + np.vdot(gradients[1], normal).real
        - np.dot(gradients[2], z),
        0,
        atol=1e-12,
    )
    _, context = tr.diff.chirality_density(*values)
    with pytest.raises(ValueError, match="shape"):
        context.pullback(g[:1])
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full_like(g, np.nan))
    context.pullback(g[::-1, ::-1])
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@settings(max_examples=30)
@given(k0=st.floats(0.9, 1.6), thickness=st.floats(0.1, 0.5))
def test_complete_material_and_interval_advect(k0, thickness):
    def objective(k0, thickness, epsilon):
        k = k0 * anp.sqrt(epsilon)
        normal = anp.sqrt(k * k - 0.2**2 - 0.3**2)
        forms = ad.chirality_density(
            anp.reshape(k, (1,)),
            anp.reshape(normal, (1,)),
            anp.stack([0.1, 0.1 + thickness]),
        )
        return anp.real(forms[0, 0] + 0.3j * forms[2, 0])

    values = [np.array(k0), np.array(thickness), np.array(2.3 + 0.2j)]
    gradients = advect.grad(objective, argnums=(0, 1, 2))(*values)
    for i, direction in enumerate([0.1, -0.2, 0.1 + 0.2j]):
        h = 1e-5
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * direction, values[i] - h * direction
        assert_allclose(
            np.vdot(gradients[i], direction).real,
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=1e-7,
            atol=1e-9,
        )


def test_upstream_interval_regression():
    basis = tr.PlaneWaveBasisByComp.default([[0.2, 0.3]])
    oracle = treams.PlaneWaveBasisByComp(basis.modes)
    expected = np.diag([2.0, -2.0])
    assert_allclose(tr.chirality_density(basis, 1.3, z=(0, 1))[0], expected)
    assert (
        np.max(abs(treams.chirality_density(oracle, 1.3, z=(0, 1))[0] - expected)) > 0.5
    )
