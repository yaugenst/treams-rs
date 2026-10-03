"""Planar layer stacks over compact port bases: basis order, reciprocity and lossless
power, and the diff.layer_stack pullbacks through Advect."""

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

from _support import (
    assert_unitary_ports,
    complex_normal,
    oracle_smatrix_array,
    to_oracle,
    varying,
)


@pytest.mark.interface
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("partial", [False, True])
def test_compact_slab_preserves_basis_order_and_partial_semantics(
    alignment, poltype, partial
):
    full = tr.PlaneWavePorts(
        [(0.2, 0.3, 0), (2.7, -0.1, 1), (0.2, 0.3, 1), (2.7, -0.1, 0)], alignment
    )
    basis = full[:-1] if partial else full
    materials = [1, (2.3 + 0.1j, 1.1, 0.1 if poltype == "helicity" else 0), 1.7, 1.2]
    thickness = [0.4, 0.2]
    actual = tr.SMatrix.slab(thickness, basis, 1.3, materials, poltype)
    factors = [tr.SMatrix.interface(full, 1.3, materials[:2], poltype)]
    for i, d in enumerate(thickness):
        factors += [
            tr.SMatrix.propagation(d, full, 1.3, materials[i + 1], poltype),
            tr.SMatrix.interface(full, 1.3, materials[i + 1 : i + 3], poltype),
        ]
    # Composition is associative: fold the non-commuting factors from the
    # left and from the right.
    left, right = factors[0], factors[-1]
    for factor in factors[1:]:
        left = left.add(factor)
    for factor in factors[-2::-1]:
        right = factor.add(right)
    # Unrepresented external channels can still carry the internal reflections.
    size = len(basis)
    assert_allclose(
        actual.array, left.array[:, :, :size, :size], rtol=3e-12, atol=3e-12
    )
    assert_allclose(
        actual.array, right.array[:, :, :size, :size], rtol=3e-12, atol=3e-12
    )


@pytest.mark.physics
@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_partial_slab_projects_full_channels_and_retains_only_their_power(poltype):
    full = tr.PlaneWavePorts.default([[0.6, 0.2]])
    partial = full[:1]
    reference = treams.SMatrices.slab(0.4, to_oracle(full), 1.3, [1, 2.3, 1])
    if poltype == "parity":
        reference = reference.changepoltype(poltype)
    slab = tr.SMatrix.slab(0.4, partial, 1.3, [1, 2.3, 1], poltype)
    expected = oracle_smatrix_array(reference)
    assert_allclose(slab.array, expected[:, :, :1, :1], rtol=2e-13, atol=2e-13)
    power = slab.power([1.0])
    full_power = reference.tr([1.0, 0.0])
    assert_allclose(sum(full_power), 1.0, atol=2e-13)
    # Each selected power is a subset of the lossless full-channel power.
    assert power.transmission <= full_power[0] + 2e-13
    assert power.reflection <= full_power[1] + 2e-13
    assert_allclose(power.transmission, abs(expected[0, 0, 0, 0]) ** 2, atol=2e-13)
    assert_allclose(power.reflection, abs(expected[1, 0, 0, 0]) ** 2, atol=2e-13)


@pytest.mark.gradients
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

    def record(*values):
        return tr.diff.layer_stack(*values, alignment=alignment, fixed_q=fixed_q)

    value, context = record(*values)
    g = complex_normal(rng, value.shape)
    gradients = context.pullback(g)
    # A fixed q is not differentiated: its gradient is exactly zero.
    free = [0, 1, 3] if fixed_q else [0, 1, 2, 3]
    if fixed_q:
        assert_allclose(gradients[2], 0, atol=0)
    check_pullback(
        varying(record, values, *free),
        *(values[i] for i in free),
        directions=tuple(directions[i] for i in free),
        cotangents=g,
        step=1e-5,
        rtol=3e-7,
        atol=2e-9,
    )


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@settings(max_examples=20)
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

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2)),
        np.array(epsilon),
        np.array(d),
        q,
        directions=(np.array(0.1), np.array(0.03), np.full_like(q, 0.1)),
        step=1e-5,
        rtol=3e-6,
        atol=2e-9,
    )
    k = 1.3 * np.sqrt(epsilon)
    value = tr.diff.layer_stack(
        [[1.3, 1.3], [k, k], [1.3, 1.3]], [1, 1.3 / k, 1], q, [d], alignment=alignment
    )[0]
    for array in value:
        assert_unitary_ports(array, atol=2e-12)


_lossless = st.tuples(st.floats(1.0, 3.0), st.floats(0.8, 1.5))
_passive = st.tuples(
    st.floats(1.2, 4.0), st.floats(0.0, 0.5), st.floats(0.8, 1.5), st.floats(-0.3, 0.3)
)


@pytest.mark.physics
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@settings(max_examples=15)
@given(
    below=_lossless,
    above=_lossless,
    interior=st.lists(_passive, min_size=2, max_size=2),
    thickness=st.lists(st.floats(0.1, 1.0), min_size=2, max_size=2),
    q=st.tuples(st.floats(-0.35, 0.35), st.floats(-0.35, 0.35)),
    chiral=st.booleans(),
)
def test_slab_between_lossless_media_transmits_reciprocally(
    alignment, below, above, interior, thickness, q, chiral
):
    """Lossy stacks between distinct lossless media transmit equal power up and down.

    Equality holds per helicity (and per TE/TM for xy ports) for achiral interiors
    and summed over helicities for chiral ones; reflectances generally differ.
    """
    materials = [
        (below[0], below[1], 0),
        *((re + 1j * im, mu, kappa if chiral else 0) for re, im, mu, kappa in interior),
        (above[0], above[1], 0),
    ]
    basis = tr.PlaneWavePorts.default([list(q)], alignment)

    def transmitted(poltype):
        slab = tr.SMatrix.slab(thickness, basis, 1.0, materials, poltype)
        illuminations = np.eye(len(basis))
        return [
            np.array([slab.tr(illu, modetype=side)[0] for illu in illuminations])
            for side in ("up", "down")
        ]

    up, down = transmitted("helicity")
    if chiral:
        assert_allclose(up.sum(), down.sum(), rtol=0, atol=1e-12)
    else:
        assert_allclose(up, down, rtol=0, atol=1e-12)
        if alignment == "xy":
            up, down = transmitted("parity")
            assert_allclose(up, down, rtol=0, atol=1e-12)


@pytest.mark.physics
@pytest.mark.gradients
def test_empty_layer_stack_is_interface_and_constant_q_advect():
    ks = np.array([[1.3, 1.5], [1.8, 2.0]])
    zs = np.array([0.9, 0.7])
    q = np.array([[0.2, 0.3]])
    actual = tr.diff.layer_stack(ks, zs, q, [], alignment="zx")[0][0]
    assert_allclose(
        actual,
        tr.diff.interface_coefficients(ks, zs, q[0], alignment="zx")[0],
        atol=1e-14,
    )

    def objective(z):
        value = ad.layer_stack(ks, z, q, [], alignment="zx", fixed_q=True)
        return anp.sum(anp.real(value))

    check_gradient(
        objective,
        advect.grad(objective),
        zs,
        directions=(np.ones_like(zs),),
        step=1e-5,
        rtol=1e-7,
        atol=1e-9,
    )
