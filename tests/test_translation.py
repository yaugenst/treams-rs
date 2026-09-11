"""Explicit translation operators, physical plane identities and native pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad

pytestmark = pytest.mark.filterwarnings(
    "ignore:'where' used without 'out'.*:UserWarning"
)


@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_translation_reference_all_families(family, poltype):
    positions = [[0.1, 0.2, -0.1], [-0.3, 0.1, 0.4]]
    pairs = []
    for package in (tr, treams):
        if family == "sw":
            source = package.SphericalWaveBasis.default(2, 2, positions)
            destination = package.SphericalWaveBasis(list(source)[::-2], positions)
        elif family == "cw":
            source = package.CylindricalWaveBasis.default([0.2, -0.3], 2, 2, positions)
            destination = package.CylindricalWaveBasis(list(source)[::-2], positions)
        elif family == "unit":
            source = package.PlaneWaveBasisByUnitVector.default(
                [[0.2, 0.3, 1], [0.1j, 0.2, 1]]
            )
            destination = package.PlaneWaveBasisByUnitVector(list(source)[::-2])
        else:
            source = package.PlaneWaveBasisByComp.default(
                [[0.2, 0.3], [2.1, 0.1]], family
            )
            destination = package.PlaneWaveBasisByComp(list(source)[::-2], family)
        pairs.append((destination, source))
    r = np.array([[[0.1, 0.2, -0.3], [-0.2, 0.1, 0.3], [0, 0, 0]]])
    mask = np.random.default_rng(97).random((len(pairs[0][0]), len(pairs[0][1]))) > 0.2
    common = dict(
        k0=1.3,
        material=(1.4 + 0.1j, 1.2, 0.04 if poltype == "helicity" else 0),
        poltype=poltype,
        modetype="down",
        where=mask,
    )
    actual = tr.translate(r, basis=pairs[0], **common)
    expected = treams.translate(r, basis=pairs[1], **common)
    assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("side", ["up", "down"])
@given(scale=st.floats(0.01, 100), phase=st.floats(-2, 2))
def test_plane_basis_conversion_and_translation_preserve_fields(
    poltype, side, scale, phase
):
    k0, material = 1.3 / scale, (1.4, 1.2, 0.04 if poltype == "helicity" else 0)
    source = tr.PlaneWaveBasisByComp.default(
        np.array([[0.2, 0.3], [2.1, -0.1]]) / scale
    )
    full = source.byunitvector(k0, material, side)
    destination = tr.PlaneWaveBasisByUnitVector(full.modes[::-1])
    assert_allclose(
        tr.expand(source, side, k0=k0, material=material, poltype=poltype),
        np.eye(len(source)),
        atol=0,
    )
    mapping = tr.expand(
        (destination, source), side, k0=k0, material=material, poltype=poltype
    )
    assert_allclose(mapping, np.eye(len(source))[::-1], atol=0)
    assert_allclose(
        tr.expand(
            (source, destination), side, k0=k0, material=material, poltype=poltype
        )
        @ mapping,
        np.eye(len(source)),
        atol=0,
    )
    amplitudes = np.array([0.3, 0.1j, -0.2, 0.4 + 0.1j]) * np.exp(1j * phase)
    points = np.array([[0.2, 0.1, 0.3], [-0.1, 0.3, -0.2]]) * scale
    shift = np.array([0.1, -0.2, 0.3]) * scale
    options = dict(k0=k0, material=material, poltype=poltype, modetype=side)
    moved = tr.translate(shift, basis=(destination, source), **options) @ amplitudes
    expected = tr.efield(points + shift, basis=source, **options) @ amplitudes
    assert_allclose(
        tr.efield(points, basis=destination, **options) @ moved,
        expected,
        rtol=2e-12,
        atol=2e-12,
    )
    zero = tr.expand(
        (source, source), ("up", "down"), k0=k0, material=material, poltype=poltype
    )
    assert_allclose(zero, 0, atol=0)


@given(scale=st.floats(0.1, 3), imaginary=st.floats(-0.2, 0.2))
def test_plane_phase_values_and_all_pullbacks(scale, imaginary):
    points = np.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]]) * scale
    vectors = (
        np.array(
            [[1.2 + imaginary * 1j, 0.1j, 0.3], [0, 0, 1.3 + imaginary * 1j], [0, 0, 0]]
        )
        / scale
    )
    value, context = tr.diff.plane_phases(points, vectors)
    assert_allclose(value, np.exp(1j * points @ vectors.T), rtol=1e-13, atol=1e-13)
    g = np.arange(6).reshape(3, 2).T.astype(complex) + 0.2j
    gradients = context.pullback(g)
    directions = [np.full(points.shape, 0.1), np.full(vectors.shape, 0.03 + 0.02j)]
    h = 1e-5
    for i in range(2):
        plus, minus = [points, vectors], [points, vectors]
        plus[i], minus[i] = plus[i] + h * directions[i], minus[i] - h * directions[i]
        numeric = np.vdot(
            g,
            (tr.diff.plane_phases(*plus)[0] - tr.diff.plane_phases(*minus)[0])
            / (2 * h),
        ).real
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real, numeric, rtol=3e-8, atol=3e-9
        )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
def test_parallel_plane_phase_contractions(layout):
    rng = np.random.default_rng(817)
    points = rng.normal(size=(257, 3)) * 0.2
    vectors = rng.normal(size=(17, 3)) + 0.1j * rng.normal(size=(17, 3))
    expected = np.exp(1j * points @ vectors.T)
    value, context = tr.diff.plane_phases(points, vectors)
    raw = rng.normal(size=(257, 34)) + 1j * rng.normal(size=(257, 34))
    g = raw[:, ::2] if layout == "strided" else raw[:, :17]
    g = (
        g[::-1]
        if layout == "reversed"
        else np.array(g, order=layout)
        if layout in ("C", "F")
        else g
    )
    gp, gk = context.pullback(g)
    assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
    assert_allclose(
        gp, ((g.conj() * 1j * expected) @ vectors).real, rtol=2e-12, atol=2e-12
    )
    assert_allclose(gk, (g * (1j * expected).conj()).T @ points, rtol=2e-12, atol=2e-12)


def test_advect_plane_translation_and_amplitudes():
    points = np.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]])
    vectors = np.array([[1.2 + 0.1j, 0, 0.3], [0, 0, 1.3 - 0.1j]])
    coefficients = np.array([0.2 + 0.1j, -0.3 + 0.4j])

    def objective(points, vectors, coefficients):
        value = ad.plane_phases(points, vectors) @ coefficients
        return anp.sum(anp.real(value * anp.conj(value)))

    values = [points, vectors, coefficients]
    gradients = advect.grad(objective, argnums=(0, 1, 2))(*values)
    directions = [
        np.full(points.shape, 0.1),
        np.full(vectors.shape, 0.02 + 0.03j),
        np.array([0.04, 0.03j]),
    ]
    h = 1e-5
    for i in range(3):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real,
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=3e-8,
            atol=3e-9,
        )


def test_plane_phase_context_shape_validation_does_not_consume():
    value, context = tr.diff.plane_phases([[0, 0, 0]], [[0, 0, 1]])
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.zeros((2, 1), complex))
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full(value.shape, np.nan + 0j))
    gp, gk = context.pullback(np.ones_like(value))
    assert_allclose(gp, 0, atol=0)
    assert_allclose(gk, 0, atol=0)
    empty, context = tr.diff.plane_phases(np.empty((0, 3)), [[0, 0, 1]])
    assert empty.shape == (0, 1)
    assert context.pullback(empty)[0].shape == (0, 3)
