"""Chirality density of plane-wave ports: tr.chirality_density against upstream and
Cartesian field quadrature, and the diff chirality-density pullbacks."""

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
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal


@pytest.mark.interface
def test_density_boundary_contract():
    with pytest.raises(ValueError, match="nonzero"):
        tr.diff.chirality_density([0], [1])
    with pytest.raises(ValueError, match="matching"):
        tr.diff.chirality_density([1], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        tr.diff.chirality_density([1], [1], [0, np.nan])


@pytest.mark.interface
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
def test_density_reordered_partial_basis(poltype, alignment):
    basis = tr.PlaneWavePorts.default([[0.2, 0.3], [1.7, 0.4]], alignment)
    indices = [3, 0, 1]
    partial = tr.PlaneWavePorts([basis.modes[i] for i in indices], alignment)
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


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("epsilon", [2.3, 2.3 + 0.2j])
def test_chirality_at_origin_reference(poltype, epsilon):
    basis = tr.PlaneWavePorts.default([[0.2, 0.3], [3.0, 0.4]])
    material = (epsilon, 1.2, 0.06 if poltype == "helicity" else 0)
    expected = treams.chirality_density(
        treams.PlaneWaveBasisByComp(basis.modes), 1.3, material, poltype
    )
    assert_allclose(tr.chirality_density(basis, 1.3, material, poltype), expected)


@pytest.mark.physics
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("q", [[0.2, 0.3], [3.0, 0.4], [0.0, 0.0]])
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
def test_density_agrees_with_coherent_cartesian_field_integral(poltype, q, alignment):
    basis = tr.PlaneWavePorts.default([q], alignment)
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
            for kind in (tr.operators.efield, tr.operators.hfield)
        ]
        expected = weights @ np.real(
            np.sum(fields[0].conj() * (1j * material.impedance * fields[1]), axis=-1)
        )
        forms = tr.chirality_density(basis, 1.3, material, poltype, interval)
        assert_allclose(_contract(forms, up, down), expected, rtol=1e-12, atol=1e-12)
        assert_allclose(
            forms[:2], np.asarray(forms[:2]).conj().swapaxes(-1, -2), atol=1e-14
        )


@pytest.mark.physics
@given(k0=st.floats(0.8, 2.0), start=st.floats(-2, 2), stop=st.floats(-2, 2))
def test_single_propagating_wave_density_is_interval_independent(k0, start, stop):
    basis = tr.PlaneWavePorts.default([[0.2, 0.3]])
    up, down, _ = tr.chirality_density(basis, k0, z=(start, stop))
    assert_allclose(up, np.diag(2 * (2 * basis.pol - 1)), atol=1e-14)
    assert_allclose(down, up, atol=1e-14)


@pytest.mark.gradients
@pytest.mark.parametrize("z", [[-0.2, 0.7], [0.4, 0.4], [0.2, 0.2 + 1e-10]])
def test_chirality_native_complex_pullback(z):
    ks = np.array([1.3 + 0.2j, 1.6 + 0.1j, 1.0])
    normal = np.array([1.2 + 0.1j, 0.3 + 0.5j, 1.0j])
    values = [ks, normal, np.array(z)]
    rng = np.random.default_rng(86)
    g = complex_normal(rng, (3, 3))
    _, context = tr.diff.chirality_density(*values)
    gradients = context.pullback(g)
    check_pullback(
        tr.diff.chirality_density,
        *values,
        directions=tuple(
            rng.normal(size=v.shape) * 0.1 + (0.03j if i < 2 else 0)
            for i, v in enumerate(values)
        ),
        cotangents=g,
        step=1e-5,
        rtol=1e-7,
        atol=1e-9,
    )
    # The forms are homogeneous of degree zero in (ks, normal, 1/z).
    assert_allclose(
        np.vdot(gradients[0], ks).real
        + np.vdot(gradients[1], normal).real
        - np.dot(gradients[2], z),
        0,
        atol=1e-12,
    )


@pytest.mark.gradients
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

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2)),
        np.array(k0),
        np.array(thickness),
        np.array(2.3 + 0.2j),
        directions=(np.array(0.1), np.array(-0.2), np.array(0.1 + 0.2j)),
        step=1e-5,
        rtol=1e-7,
        atol=1e-9,
    )


@pytest.mark.reference
def test_chirality_interval_average_is_invariant_unlike_upstream():
    ours = tr.PlaneWavePorts.default([[0.2, 0.3]])
    upstream = treams.PlaneWaveBasisByComp(ours.modes)
    expected = np.diag([2.0, -2.0])
    assert_allclose(tr.chirality_density(ours, 1.3, z=(0, 1))[0], expected)
    assert (
        np.max(abs(treams.chirality_density(upstream, 1.3, z=(0, 1))[0] - expected))
        > 0.5
    )


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("axial", [False, True])
@given(start=st.floats(-0.5, 0.5), stop=st.floats(0.5, 1.0))
@settings(max_examples=15)
def test_oriented_native_pullback_and_interval_invariants(axis, axial, start, stop):
    q = np.array([[0.2, 0.3], [0.3, -0.4]]) if not axial else np.zeros((2, 2))
    normal = np.array([1.2 + 0.1j, 0.3 + 0.5j])
    z = np.array([start, stop])
    metadata = {"polarizations": [0, 1], "axis": axis}
    values = [q, normal, z]
    rng = np.random.default_rng(173)
    g = complex_normal(rng, (3, 2))

    def record(*values):
        return tr.diff.oriented_chirality(*values, **metadata)

    value, context = record(*values)
    gradients = context.pullback(g)
    check_pullback(
        record,
        *values,
        directions=tuple(
            rng.normal(size=v.shape) * 0.1 + (0.04j if i == 1 else 0)
            for i, v in enumerate(values)
        ),
        cotangents=g,
        step=1e-5,
        rtol=2e-7,
        atol=3e-9,
    )
    assert_allclose(
        np.vdot(gradients[0], q).real
        + np.vdot(gradients[1], normal).real
        - np.dot(gradients[2], z),
        0,
        atol=2e-12,
    )
    reverse = tr.diff.oriented_chirality(q, normal, z[::-1], **metadata)[0]
    assert_allclose(reverse, value, atol=1e-13)
    middle = (start + stop) / 2
    left = tr.diff.oriented_chirality(q, normal, [start, middle], **metadata)[0]
    right = tr.diff.oriented_chirality(q, normal, [middle, stop], **metadata)[0]
    assert_allclose(value, (left + right) / 2, atol=1e-13)


@pytest.mark.physics
def test_oriented_xy_formula():
    q = np.array([[0.2, 0.3], [0.4, 0.5]])
    normal = np.array([1.2 + 0.1j, 0.3 + 0.5j])
    z = np.array([-0.2, 0.4])
    pol = np.array([0, 1])
    value, _ = tr.diff.oriented_chirality(q, normal, z, polarizations=pol, axis=2)
    ks = np.sqrt(np.sum(q * q, axis=-1) + normal * normal)
    compact, _ = tr.diff.chirality_density(ks, normal, z)
    assert_allclose(value, compact * (2 * pol - 1), atol=2e-14)


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_oriented_wavenumber_scale(axis, scale):
    q = np.array([[0.2, 0.3]])
    normal = np.array([1.2 + 0.1j])
    kwargs = {"polarizations": [1], "axis": axis}
    value, context = tr.diff.oriented_chirality(q, normal, **kwargs)
    wanted = context.pullback(np.ones_like(value))
    scaled, context = tr.diff.oriented_chirality(q * scale, normal * scale, **kwargs)
    assert_allclose(scaled, value, atol=2e-14)
    gradient = context.pullback(np.ones_like(value))
    assert_allclose(gradient[0] * scale, wanted[0], atol=2e-14)
    assert_allclose(gradient[1] * scale, wanted[1], atol=2e-14)
    assert_allclose(gradient[2] / scale, wanted[2], atol=2e-14)


@pytest.mark.gradients
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_oriented_advect_material_incidence_interval(axis):
    def objective(k0, q, epsilon, interval):
        normal = anp.sqrt(k0 * k0 * epsilon - anp.sum(q * q, axis=-1))
        value = ad.oriented_chirality(
            q, normal, interval, polarizations=[0, 1], axis=axis
        )
        return anp.real(anp.sum(value[0]) + 0.3j * anp.sum(value[2]))

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2, 3)),
        np.array(1.3),
        np.array([[0.2, 0.3], [0.4, 0.5]]),
        np.array(2.3 + 0.2j),
        np.array([-0.2, 0.5]),
        directions=(
            np.array(0.1),
            np.array([[0.1, -0.2], [0.3, 0.1]]),
            np.array(0.1 + 0.2j),
            np.array([-0.1, 0.3]),
        ),
        step=1e-5,
        rtol=2e-7,
        atol=2e-9,
    )


@pytest.mark.gradients
@pytest.mark.interface
def test_oriented_parallel_and_strided_paths():
    q = np.tile([[0.2, 0.3], [0.3, -0.4]], (600, 1))
    normal = np.tile([1.2 + 0.1j, 0.3 + 0.5j], 600)
    pol = np.tile([0, 1], 600)
    value, context = tr.diff.oriented_chirality(
        q[::-1], normal[::-1], polarizations=pol[::-1], axis=0
    )
    g = np.tile([[1 + 0.2j], [0.3j], [0.4]], (1, 1200))[:, ::-1]
    gradient = context.pullback(g)
    small, context = tr.diff.oriented_chirality(
        q[1::-1], normal[1::-1], polarizations=pol[1::-1], axis=0
    )
    expected = context.pullback(g[:, :2])
    assert_allclose(value, np.tile(small, 600))
    assert_allclose(gradient[0], np.tile(expected[0], (600, 1)))
    assert_allclose(gradient[1], np.tile(expected[1], 600))
    assert_allclose(gradient[2], expected[2] * 600)
