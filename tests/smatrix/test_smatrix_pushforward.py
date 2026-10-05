"""Directional derivatives of the complete planar native-record surface."""

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from treams_rs import diff
from treams_rs.testing import check_pushforward

from _support import arrange, assert_tree_allclose, complex_normal

pytestmark = pytest.mark.gradients


def _cases():
    rng = np.random.default_rng(612)

    def scattering():
        value = 0.05 * complex_normal(rng, (2, 2, 2, 2))
        value[0, 0] += np.diag([0.8 + 0.1j, 0.7 + 0.2j])
        value[1, 1] += np.diag([0.6 + 0.2j, 0.9 + 0.1j])
        return value

    lower, upper = scattering(), scattering()
    ks = np.array([[1.3 + 0.02j, 1.5 + 0.03j], [1.1 + 0.01j, 1.4 + 0.04j]])
    zs = np.array([1.2 + 0.01j, 0.9 + 0.02j])
    q = np.array([0.2, 0.1])
    normal = np.array([1.2 + 0.1j, 0.3 + 0.5j])
    interval = np.array([-0.2, 0.4])
    yield (
        "from_array",
        diff.smatrix_from_array,
        (
            0.2 * complex_normal(rng, (3, 3)),
            complex_normal(rng, (2, 2, 3, 2)),
        ),
    )
    yield "add", diff.smatrix_add, (lower, upper)
    yield (
        "illuminate",
        diff.smatrix_illuminate,
        (
            lower,
            upper,
            *complex_normal(rng, (2, 2, 3)),
        ),
    )
    yield "periodic", diff.smatrix_periodic, (lower,)
    yield "bands", diff.bands, (lower, np.array(1.3))
    yield (
        "fresnel",
        diff.fresnel,
        (
            ks,
            np.sqrt(ks**2 - np.dot(q, q)),
            zs,
        ),
    )
    yield (
        "propagation",
        diff.propagation_matrix,
        (
            np.array([[0.2, 0.1, 1.2 + 0.01j], [0.2, 0.1, 1.3 + 0.02j]]),
            np.array([0.0, 0.1, 0.4]),
        ),
    )
    yield "chirality", diff.chirality_density, (ks[0], normal, interval)
    for axis in range(3):
        yield (
            f"oriented_chirality_{axis}",
            partial(diff.oriented_chirality, polarizations=[0, 1], axis=axis),
            (np.array([[0.2, 0.3], [0.4, 0.5]]), normal, interval),
        )
    for alignment in ("xy", "yz", "zx"):
        yield (
            f"interface_{alignment}",
            partial(diff.interface_coefficients, alignment=alignment),
            (ks, zs, q),
        )
        yield (
            f"layers_{alignment}",
            partial(diff.layer_stack, alignment=alignment),
            (
                np.array(
                    [
                        [1.0 + 0.02j, 1.0 + 0.02j],
                        [1.5 + 0.1j, 1.6 + 0.1j],
                        [1.2 + 0.03j, 1.2 + 0.03j],
                    ]
                ),
                np.array([1.0, 0.7, 0.9], dtype=complex),
                np.array([[0.2, 0.1], [0.1, -0.2]]),
                np.array([0.4]),
            ),
        )
    for direction in ("up", "down"):
        for poltype in ("helicity", "parity"):
            yield (
                f"tr_{direction}_{poltype}",
                partial(
                    diff.smatrix_tr,
                    modes=[(0, 0), (0, 1)],
                    modetype=direction,
                    poltype=poltype,
                ),
                (
                    lower,
                    complex_normal(rng, (2, 3)),
                    ks if poltype == "helicity" else np.repeat(ks[:, :1], 2, axis=1),
                    zs,
                    q[None],
                ),
            )


CASES = [
    pytest.param(record, arguments, id=name) for name, record, arguments in _cases()
]


def _directions(record, arguments, seed):
    rng = np.random.default_rng(seed)
    directions = tuple(
        complex_normal(rng, value.shape)
        if np.iscomplexobj(value)
        else np.asarray(rng.normal(size=value.shape))
        for value in arguments
    )
    if isinstance(record, partial) and record.keywords.get("poltype") == "parity":
        # Parity ports require achiral media, including along a perturbation.
        directions[2][:, 1] = directions[2][:, 0]
    return directions


@pytest.mark.parametrize("record, arguments", CASES)
def test_planar_pushforward_matches_reference_and_adjoint(record, arguments):
    check_pushforward(
        record, *arguments, directions=_directions(record, arguments, 713)
    )


@pytest.mark.interface
@pytest.mark.parametrize("record, arguments", CASES)
def test_planar_tangents_validate_and_reuse_context_with_strided_inputs(
    record, arguments
):
    directions = _directions(record, arguments, 27)
    value, context = record(*arguments)
    for i, direction in enumerate(directions):
        invalid = list(directions)
        invalid[i] = np.zeros((*direction.shape, 1), dtype=direction.dtype)
        with pytest.raises(ValueError, match=r"tangent.*shape"):
            context.pushforward(*invalid)
        invalid[i] = np.full(direction.shape, np.nan, dtype=direction.dtype)
        with pytest.raises(ValueError, match=r"tangent.*finite"):
            context.pushforward(*invalid)
        if not np.iscomplexobj(direction):
            invalid[i] = direction.astype(complex) + 1j
            with pytest.raises(ValueError, match="real"):
                context.pushforward(*invalid)
    expected = context.pushforward(*directions)
    outputs = value if isinstance(value, tuple) else (value,)
    cotangents = tuple(np.ones_like(output) for output in outputs)
    gradients = context.pullback(*cotangents)
    strided = tuple(
        arrange(value, "reversed") if value.ndim else value for value in directions
    )
    actual = context.pushforward(*strided)
    assert_tree_allclose(actual, expected)
    repeated = context.pullback(*(-g for g in cotangents))
    if not isinstance(gradients, tuple):
        gradients, repeated = (gradients,), (repeated,)
    for result, reference in zip(repeated, gradients, strict=True):
        assert_allclose(result, -reference)
    negative = context.pushforward(*(-d for d in directions))
    assert_tree_allclose(
        negative,
        tuple(-item for item in expected) if isinstance(expected, tuple) else -expected,
    )


@pytest.mark.physics
def test_layer_scaling_has_zero_directional_derivative():
    ks = np.array([[1.0, 1.0], [1.5 + 0.1j, 1.6 + 0.1j], [1.2, 1.2]])
    zs = np.array([1.0, 0.7, 0.9], dtype=complex)
    q = np.array([[0.2, 0.1], [0.1, -0.2]])
    thickness = np.array([0.4])
    _, context = diff.layer_stack(ks, zs, q, thickness)
    tangent = context.pushforward(ks, np.zeros_like(zs), q, -thickness)
    assert_allclose(tangent, 0, atol=2e-15)
