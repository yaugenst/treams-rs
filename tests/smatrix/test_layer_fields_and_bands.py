"""Fields inside planar stacks and Bloch bands: internal illumination, transfer and
bands against upstream, interface continuity, and native pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment

import treams_rs as tr

# _native: the non-recording illumination forward has no public wrapper.
from treams_rs import _native
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import LAYOUTS, arrange, complex_normal


def _blocks(seed, n=4):
    rng = np.random.default_rng(seed)
    result = 0.1 * complex_normal(rng, (2, 2, n, n))
    result[0, 0] += np.eye(n)
    result[1, 1] += np.eye(n)
    return result


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("columns", [1, 3])
def test_internal_illumination_reference_and_all_pullbacks(columns):
    lower, upper = _blocks(5), _blocks(6)
    rng = np.random.default_rng(7)
    up = complex_normal(rng, (4, columns))
    down = complex_normal(rng, up.shape)
    values = [lower, upper, up, down]
    value, context = tr.diff.smatrix_illuminate(*values)
    basis = treams.PlaneWaveBasisByComp.default([[0.2, 0.3], [0.4, 0.1]])
    reference_lower = treams.SMatrices(lower, basis=basis, k0=1.3)
    reference_upper = treams.SMatrices(upper, basis=basis, k0=1.3)
    # Upstream forces its AnnotationWarning to "always" inside this call, so no
    # filter can silence it; record it rather than leak it into the summary.
    with pytest.warns(
        treams.util.AnnotationWarning, match="overwriting key 'modetype'"
    ):
        reference = reference_lower.illuminate(up, down, smat=reference_upper)
    assert_allclose(value, reference, atol=1e-12)
    g = complex_normal(rng, value.shape)
    gradients = context.pullback(g)
    check_pullback(
        tr.diff.smatrix_illuminate,
        *values,
        directions=tuple(rng.normal(size=v.shape) * 0.1 + 0.03j for v in values),
        cotangents=g,
        step=1e-5,
        rtol=3e-8,
        atol=1e-9,
    )
    # The fields are linear in the illumination (Euler identity).
    assert_allclose(
        np.vdot(gradients[2], up).real + np.vdot(gradients[3], down).real,
        np.vdot(g, value).real,
        atol=1e-11,
    )


@pytest.mark.physics
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_internal_fields_satisfy_both_interface_boundaries(alignment, poltype):
    basis = tr.PlaneWavePorts.default([[0.2, 0.3]], alignment)
    k0, left, right = 1.3, 0.17, 0.23
    middle = tr.Material(2.3 + 0.1j, 1.2, 0.08 if poltype == "helicity" else 0)
    top_medium = tr.Material(1.4)
    lower = tr.SMatrix.interface(basis, k0, [1, middle], poltype).add(
        tr.SMatrix.propagation(left, basis, k0, middle, poltype)
    )
    upper = tr.SMatrix.propagation(right, basis, k0, middle, poltype).add(
        tr.SMatrix.interface(basis, k0, [middle, top_medium], poltype)
    )
    incident_up = np.array([1, 0.2j])
    incident_down = np.array([0.1j, 0.3])
    outgoing_up, outgoing_down, internal_up, internal_down = lower.illuminate(
        incident_up, incident_down, smat=upper
    )
    assert_allclose(
        [outgoing_up, outgoing_down],
        lower.add(upper).illuminate(incident_up, incident_down),
        atol=1e-12,
    )
    axis = basis.normal_axis
    tangential = [(axis + 1) % 3, (axis + 2) % 3]

    def fields(operator, distance, material, up, down):
        point = np.eye(3)[axis] * distance
        common = dict(basis=basis, k0=k0, material=material, poltype=poltype)
        return (
            operator(point, modetype="up", **common) @ up
            + operator(point, modetype="down", **common) @ down
        )[tangential]

    for operator in (tr.operators.efield, tr.operators.hfield):
        assert_allclose(
            fields(operator, 0, 1, incident_up, outgoing_down),
            fields(operator, -left, middle, internal_up, internal_down),
            atol=2e-12,
        )
        assert_allclose(
            fields(operator, right, middle, internal_up, internal_down),
            fields(operator, 0, top_medium, outgoing_up, incident_down),
            atol=2e-12,
        )


@pytest.mark.physics
def test_internal_downward_plane_uses_top_outer_medium():
    wave = tr.plane_wave([0.1, 0.2, -1], 1, k0=1.3, material=4)
    basis = tr.PlaneWavePorts.default([wave.kvecs[0, :2].real])
    lower = tr.SMatrix.interface(basis, 1.3, [1, 2.3])
    upper = tr.SMatrix.interface(basis, 1.3, [2.3, 4])
    actual = lower.illuminate(wave, smat=upper)
    expected = lower.add(upper).illuminate(wave)
    assert_allclose(actual[:2], expected, atol=1e-12)


@pytest.mark.gradients
@pytest.mark.reference
def test_transfer_reference_and_pullback():
    value = _blocks(11)
    transfer = tr.diff.smatrix_periodic(value)[0]
    basis = treams.PlaneWaveBasisByComp.default([[0.2, 0.3], [0.4, 0.1]])
    expected = treams.SMatrices(value, basis=basis, k0=1.3).periodic()
    assert_allclose(transfer, expected, atol=1e-12)
    rng = np.random.default_rng(12)
    g = complex_normal(rng, transfer.shape)
    check_pullback(
        tr.diff.smatrix_periodic,
        value,
        directions=(rng.normal(size=value.shape) * 0.1 + 0.03j,),
        cotangents=g,
        step=1e-5,
        rtol=3e-8,
        atol=1e-9,
    )


@pytest.mark.physics
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@given(period=st.floats(0.1, 0.7), k0=st.floats(1.0, 2.0))
def test_uniform_bloch_bands(alignment, period, k0):
    basis = tr.PlaneWavePorts.default([[0.2, 0.3], [2.5, 0.1]], alignment)
    smats = tr.SMatrix.propagation(period, basis, k0)
    wavenumbers, vectors = smats.bands_kz(period)
    expected = np.sqrt(k0**2 - np.sum(basis.components**2, axis=1) + 0j)
    expected = np.concatenate([expected, -expected])
    _, order = linear_sum_assignment(abs(wavenumbers[:, None] - expected))
    assert_allclose(wavenumbers, expected[order], atol=3e-12)
    assert_allclose(
        smats.periodic() @ vectors,
        vectors * np.exp(1j * wavenumbers * period),
        atol=3e-12,
    )


@pytest.mark.physics
@settings(max_examples=20)
@given(
    q=hnp.arrays(np.float64, (2, 2), elements=st.floats(-0.4, 0.4)),
    kappa=st.tuples(st.floats(-0.2, 0.2), st.floats(-0.2, 0.2)),
    thickness=st.tuples(st.floats(0.1, 0.6), st.floats(0.1, 0.6)),
    loss=st.floats(0, 0.3),
)
def test_bloch_factors_of_reciprocal_stacks_pair_with_their_inverses(
    q, kappa, thickness, loss
):
    # A reciprocal stack of isotropic layers transmits a Bloch wave with
    # factor exp(i k d) in one direction and exp(-i k d) in the other, so the
    # eigenvalues of the transfer matrix pair as lambda <-> 1 / lambda, with
    # loss and chirality. The second layer is lossy, which keeps every draw
    # away from a lossless band edge, where lambda = 1 / lambda is defective.
    materials = [
        1,
        (2.3 + loss * 1j, 1.2, kappa[0]),
        (1.7 + 0.2j, 1.0, kappa[1]),
        1,
    ]
    stack = tr.SMatrix.slab(
        list(thickness), tr.PlaneWavePorts.default(q), 1.3, materials
    )
    period = sum(thickness)
    wavenumbers, vectors = stack.bands_kz(period)
    factors = np.exp(1j * period * wavenumbers)
    _, order = linear_sum_assignment(abs(factors[:, None] - 1 / factors[None, :]))
    assert_allclose(factors, 1 / factors[order], rtol=1e-11, atol=1e-12)
    assert_allclose(stack.periodic() @ vectors, vectors * factors, atol=3e-12)


@pytest.mark.reference
def test_slab_band_reference():
    q = [[0.2, 0.3], [0.4, 0.1]]
    materials = [1, (2.3 + 0.1j, 1.2, 0.13), 1]
    ours = tr.SMatrix.slab(0.4, tr.PlaneWavePorts.default(q), 1.3, materials)
    oracle = treams.SMatrices.slab(
        0.4, treams.PlaneWaveBasisByComp.default(q), 1.3, materials
    )
    assert_allclose(ours.periodic(), oracle.periodic(), atol=2e-12)
    wavenumbers, vectors = ours.bands_kz(0.4)
    expected, _ = oracle.bands_kz(0.4)
    _, order = linear_sum_assignment(abs(wavenumbers[:, None] - expected))
    assert_allclose(wavenumbers, expected[order], atol=3e-12)
    assert_allclose(
        ours.periodic() @ vectors, vectors * np.exp(0.4j * wavenumbers), atol=3e-12
    )


@pytest.mark.gradients
@pytest.mark.parametrize("vector_objective", [False, True])
def test_band_native_all_pullbacks(vector_objective):
    smats, period = _blocks(22, 3), 0.7
    (k, vectors), context = tr.diff.bands(smats, period)
    rng = np.random.default_rng(23)
    gk = complex_normal(rng, k.shape)
    gv = (
        complex_normal(rng, vectors.shape)
        if vector_objective
        else np.zeros_like(vectors)
    )
    gradients = context.pullback(gk, gv)

    def objective(s, p):
        (ks, vs), _ = tr.diff.bands(s, p)
        _, order = linear_sum_assignment(abs(k[:, None] - ks))
        return np.vdot(gk, ks[order]).real + np.vdot(gv, vs[:, order]).real

    direction = 0.1 * complex_normal(rng, smats.shape)
    h = 1e-5
    assert_allclose(
        np.vdot(gradients[0], direction).real,
        (
            objective(smats + h * direction, period)
            - objective(smats - h * direction, period)
        )
        / (2 * h),
        rtol=2e-7,
        atol=2e-8,
    )
    assert_allclose(
        gradients[1],
        (objective(smats, period + h) - objective(smats, period - h)) / (2 * h),
        rtol=2e-8,
        atol=2e-9,
    )


@pytest.mark.gradients
def test_complete_advect_uniform_bands_at_polarization_degeneracy():
    def objective(k0, period):
        normal = anp.sqrt(k0**2 - 0.2**2 - 0.3**2)
        vectors = anp.stack([anp.stack([0.2, 0.3, normal])] * 2)
        smats = ad.propagation_matrix(vectors, anp.stack([0.0, 0.0, period]))
        k, _ = ad.bands(smats, period)
        return anp.sum(anp.real(k * anp.conj(k)))

    gk, gp = advect.grad(objective, argnums=(0, 1))(np.array(1.3), np.array(0.4))
    assert_allclose(gk, 8 * 1.3, atol=2e-11)
    assert_allclose(gp, 0, atol=2e-11)


@pytest.mark.gradients
@pytest.mark.parametrize("observable", ["bands", "internal"])
def test_complete_advect_layer_observables(observable):
    def objective(k0, epsilon, thickness, q):
        indices = anp.stack([1.0, anp.sqrt(epsilon), anp.sqrt(1.7), 1.0])
        chirality = anp.array([0.0, 0.13, 0.07, 0.0])
        ks = k0 * anp.stack([indices - chirality, indices + chirality], axis=-1)
        zs = 1 / indices
        if observable == "bands":
            smats = ad.layer_stack(ks, zs, q[None, :], thickness)[0]
            k, vectors = ad.bands(smats, anp.sum(thickness))
            return anp.sum(anp.real(k * anp.conj(k))) + 0.1 * anp.sum(
                anp.real(vectors * anp.conj(vectors)) ** 2
            )
        # The shared interior medium has no interface at the observation plane.
        lower = ad.layer_stack(ks[:3], zs[:3], q[None, :], thickness[:1])[0]
        upper = ad.layer_stack(ks[2:], zs[2:], q[None, :], anp.array([]))[0]
        vectors = anp.stack(
            [
                q[0] * anp.ones(2),
                q[1] * anp.ones(2),
                anp.sqrt(ks[2] ** 2 - anp.sum(q**2)),
            ],
            axis=-1,
        )
        upper = ad.smatrix_add(
            ad.propagation_matrix(vectors, anp.stack([0.0, 0.0, thickness[1]])), upper
        )
        fields = ad.smatrix_illuminate(
            lower, upper, np.array([[1.0], [0.2j]]), np.array([[0.1j], [0.3]])
        )[2:]
        return anp.sum(anp.real(fields * anp.conj(fields)))

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2, 3)),
        np.array(1.3),
        np.array(2.3 + 0.1j),
        np.array([0.2, 0.3]),
        np.array([0.2, 0.3]),
        directions=(
            np.array(0.1),
            np.array(0.1 + 0.02j),
            np.array([0.03, -0.02]),
            np.array([0.01, 0.02]),
        ),
        step=1e-5,
        rtol=3e-6,
        atol=3e-8,
    )


@pytest.mark.gradients
@pytest.mark.interface
@pytest.mark.parametrize("columns", [1, 4])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_internal_borrowed_forward_and_owned_adjoint(layout, columns):
    # Both bindings read the blocks and illumination columns in every layout,
    # flipped ones included; layout invariance needs only one fixed draw.
    rng = np.random.default_rng(0)
    values = [_blocks(0), _blocks(1)] + [
        rng.normal(size=(4, columns)).astype(complex) + 0.2j for _ in range(2)
    ]
    expected, context = tr.diff.smatrix_illuminate(*values)
    g = rng.normal(size=expected.shape).astype(complex) + 0.13j
    expected_gradients = context.pullback(g)

    inputs = [arrange(a, layout) for a in values]
    assert_allclose(_native.smatrix_illuminate_value(*inputs), expected, atol=1e-12)
    value, residual = tr.diff.smatrix_illuminate(*inputs)
    assert_allclose(value, expected, atol=1e-12)
    for a in inputs:
        a[:] = np.nan
    gradients = residual.pullback(arrange(g, layout))
    for gradient, wanted in zip(gradients, expected_gradients, strict=True):
        assert_allclose(gradient, wanted, atol=1e-12)


@pytest.mark.interface
@pytest.mark.parametrize("record", [False, True])
def test_internal_nonfinite_input_is_rejected(record):
    function = (
        tr.diff.smatrix_illuminate if record else _native.smatrix_illuminate_value
    )
    lower = _blocks(5)
    lower[0, 1, 1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        function(lower, _blocks(6), np.ones((4, 1), complex), np.zeros((4, 1), complex))


@pytest.mark.gradients
@pytest.mark.parametrize("columns,seed", [(1, 0), (3, 1)])
def test_large_thin_illumination_matches_dense_batch_and_all_pullbacks(columns, seed):
    n = 512
    rng = np.random.default_rng(seed)
    lower, upper = [_blocks(seed + i, n) / np.sqrt(n) for i in range(2)]
    up, down = [complex_normal(rng, (n, columns)) / np.sqrt(n) for _ in range(2)]
    # Nine RHS use the direct path; padding with unilluminated columns leaves
    # both the original fields and their parameter derivatives unchanged.
    padded = [np.pad(a, ((0, 0), (0, 9 - columns))) for a in (up, down)]
    expected, dense = tr.diff.smatrix_illuminate(lower, upper, *padded)
    value, iterative = tr.diff.smatrix_illuminate(lower, upper, up, down)
    assert_allclose(value, expected[:, :, :columns], rtol=3e-12, atol=2e-14)
    assert_allclose(
        _native.smatrix_illuminate_value(lower, upper, up, down),
        expected[:, :, :columns],
        rtol=3e-12,
        atol=2e-14,
    )
    g = complex_normal(rng, value.shape)
    expected_gradients = dense.pullback(np.pad(g, ((0, 0), (0, 0), (0, 9 - columns))))
    lower[:] = upper[:] = np.nan
    up[:] = down[:] = np.nan
    for i, (actual, wanted) in enumerate(
        zip(iterative.pullback(g), expected_gradients, strict=True)
    ):
        if i >= 2:
            wanted = wanted[:, :columns]
        assert_allclose(actual, wanted, rtol=3e-11, atol=2e-13)
