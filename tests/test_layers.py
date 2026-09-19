import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("partial", [False, True])
def test_compact_slab_preserves_basis_order_and_partial_semantics(
    alignment, poltype, partial
):
    modes = [(0.2, 0.3, 0), (2.7, -0.1, 1), (0.2, 0.3, 1), (2.7, -0.1, 0)]
    if partial:
        modes = modes[:-1]
    basis = tr.PlaneWavePorts(modes, alignment)
    materials = [1, (2.3 + 0.1j, 1.1, 0.1 if poltype == "helicity" else 0), 1.7, 1.2]
    thickness = [0.4, 0.2]
    actual = tr.SMatrix.slab(thickness, basis, 1.3, materials, poltype)
    expected = tr.SMatrix.interface(basis, 1.3, materials[:2], poltype)
    for i, d in enumerate(thickness):
        expected = expected.add(
            tr.SMatrix.propagation(d, basis, 1.3, materials[i + 1], poltype)
        ).add(tr.SMatrix.interface(basis, 1.3, materials[i + 1 : i + 3], poltype))
    assert_allclose(actual.array, expected.array, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("fixed_q", [False, True])
def test_compact_layer_pullback_all_parameters(alignment, fixed_q):
    values = [
        np.array(
            [
                [1.3 + 0.1j, 1.5 + 0.12j],
                [1.8 + 0.07j, 2.0 + 0.15j],
                [1.7 + 0.2j, 1.9 + 0.2j],
                [1.5 + 0.1j, 1.7 + 0.13j],
            ]
        ),
        np.array([0.9 + 0.03j, 0.7 + 0.04j, 1.1 + 0.1j, 0.8 + 0.03j]),
        np.array([[0.2, 0.3], [0, 0], [2.8, 0.2]]),
        np.array([0.4, 0.6]),
    ]
    rng = np.random.default_rng(84)
    directions = [
        rng.normal(size=v.shape) * 0.1 + (0.03j if np.iscomplexobj(v) else 0)
        for v in values
    ]
    value, context = tr.diff.layer_stack(*values, alignment=alignment, fixed_q=fixed_q)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    with pytest.raises(ValueError, match="shape"):
        context.pullback(g[:1])
    gradients = context.pullback(g)
    for i in range(4):
        if fixed_q and i == 2:
            assert_allclose(gradients[i], 0, atol=0)
            continue
        plus, minus = list(values), list(values)
        h = 1e-5
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        numeric = np.vdot(
            g,
            (
                tr.diff.layer_stack(*plus, alignment=alignment)[0]
                - tr.diff.layer_stack(*minus, alignment=alignment)[0]
            )
            / (2 * h),
        ).real
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real, numeric, rtol=3e-7, atol=2e-9
        )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@settings(max_examples=20, deadline=None)
@given(epsilon=st.floats(1.3, 4), d=st.floats(0.1, 0.8))
def test_compact_stack_advect_and_lossless_power(alignment, epsilon, d):
    q = np.array([[0.2, 0.3], [0, 0], [0.5, -0.1]])

    def objective(eps, thickness, q):
        k = 1.3 * anp.sqrt(eps)
        ks = anp.stack(
            [anp.array([1.3, 1.3]), anp.stack([k, k]), anp.array([1.3, 1.3])]
        )
        z = anp.stack([1.0, 1.3 / k, 1.0])
        value = ad.layer_stack(
            ks, z, q, anp.reshape(thickness, (1,)), alignment=alignment
        )
        reflected = value[:, 1, 0, :, 0]
        return anp.sum(anp.real(reflected * anp.conj(reflected)))

    values = [np.array(epsilon), np.array(d), q]
    directions = [0.1, 0.03, np.full_like(q, 0.1)]
    gradients = advect.grad(objective, argnums=(0, 1, 2))(*values)
    h = 1e-5
    for i in range(3):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real,
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=3e-6,
            atol=2e-9,
        )
    k = 1.3 * np.sqrt(epsilon)
    value = tr.diff.layer_stack(
        [[1.3, 1.3], [k, k], [1.3, 1.3]], [1, 1.3 / k, 1], q, [d], alignment=alignment
    )[0]
    assert_allclose(np.sum(abs(value[:, :, 0, :, 0]) ** 2, axis=(1, 2)), 1, atol=2e-12)


def test_empty_layer_stack_is_interface_and_constant_q_advect():
    ks = np.array([[1.3, 1.5], [1.8, 2.0]])
    zs = np.array([0.9, 0.7])
    q = np.array([[0.2, 0.3]])
    actual = tr.diff.layer_stack(ks, zs, q, [], alignment="zx")[0][0]
    assert_allclose(
        actual, tr.diff.interface(ks, zs, q[0], alignment="zx")[0], atol=1e-14
    )

    def objective(z):
        value = ad.layer_stack(ks, z, q, [], alignment="zx", fixed_q=True)
        return anp.sum(anp.real(value))

    gradient = advect.grad(objective)(zs)
    h = 1e-5
    assert_allclose(
        np.sum(gradient),
        (objective(zs + h) - objective(zs - h)) / (2 * h),
        rtol=1e-7,
        atol=1e-9,
    )
