"""Cluster JVP context lifecycle, including reusable dense and block factors."""

import numpy as np
import pytest

import treams_rs as tr
from treams_rs import _native
from treams_rs.testing import check_pushforward

from _support import assert_saved_context, assert_tree_allclose


def _record(kind):
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.3, -0.2]])
    if kind == "interaction":
        local = np.eye(3, dtype=complex) * 0.1
        coupling = np.ones((3, 3), complex) * (0.02 + 0.01j)
        value, context = _native.interaction(local, coupling)
        return value, context, (local * 0, coupling * 0), "pushforward", 1
    if kind == "spheres":
        radii = np.array([0.2, 0.3])
        epsilon = np.array([2.0 + 0.1j, 3.0 + 0.2j])
        value, context = _native.sphere_cluster(1, 1.1, radii, epsilon, positions)
        return (
            value,
            context,
            (0.0, radii * 0, epsilon * 0, positions * 0),
            "pushforward",
            1,
        )
    if kind in {"spherical-particles", "cylindrical-particles"}:
        if kind == "spherical-particles":
            basis = tr.SphericalBasis.default(1, nmax=2, positions=positions)
            record = _native.particle_cluster
        else:
            basis = tr.CylindricalBasis.default([0.1], 1, nmax=2, positions=positions)
            record = _native.cylindrical_particle_cluster
        blocks = [np.eye(len(basis) // 2, dtype=complex) * 0.02] * 2
        value, context = record(
            blocks, list(basis.modes), positions.tolist(), (1.1, 1.2), True
        )
        return (
            value,
            context,
            ([b * 0 for b in blocks], positions * 0, (0j, 0j)),
            "pushforward",
            1,
        )
    if kind == "sphere-factor":
        factor = _native.sphere_cluster_factor(
            1,
            1.1,
            np.array([0.2, 0.3]),
            np.array([2.0 + 0.1j, 3.0 + 0.2j]),
            positions,
        )
        local = [np.zeros((6, 6), complex)] * 2
        method = "pushforward_blocks"
    elif kind == "block-factor":
        blocks = [np.eye(1, dtype=complex) * 0.1, np.eye(2, dtype=complex) * 0.2]
        factor = _native.InteractionFactor.from_blocks(
            blocks, np.ones((3, 3), complex) * 0.02
        )
        local = [b * 0 for b in blocks]
        method = "pushforward_blocks"
    else:
        factor = _native.InteractionFactor(
            np.eye(3, dtype=complex) * 0.1, np.ones((3, 3), complex) * 0.02
        )
        local = np.zeros((3, 3), complex)
        method = "pushforward"
    incident = np.ones((factor.dimension, 2), complex)
    value, context = factor.record(incident)
    return (
        value,
        context,
        (local, np.zeros((factor.dimension, factor.dimension), complex), incident * 0),
        method,
        1,
    )


_KINDS = [
    "interaction",
    "spheres",
    "spherical-particles",
    "cylindrical-particles",
    "dense-factor",
    "block-factor",
    "sphere-factor",
]


@pytest.mark.interface
@pytest.mark.parametrize("kind", _KINDS)
def test_cluster_saved_arrays_restore_the_same_linearization(kind):
    value, context, tangents, method, _ = _record(kind)
    cls = type(context)
    if kind == "interaction":
        size = cls._state_spec(3)
    elif kind == "spheres":
        size = cls._state_spec(1, 2)
    elif kind.endswith("particles"):
        size = cls._state_spec([6, 6], kind == "cylindrical-particles")
    else:
        sizes = {"dense-factor": [3], "block-factor": [1, 2], "sphere-factor": [6, 6]}
        size = cls._state_spec(sizes[kind], value.shape[1])
    assert_saved_context(
        context,
        tangents,
        np.ones_like(value),
        size=size,
        suffix=method.removeprefix("pushforward"),
    )


@pytest.mark.interface
@pytest.mark.parametrize("kind", _KINDS)
def test_cluster_pushforward_accepts_real_arraylike_directions(kind):
    value, context, tangents, method, _ = _record(kind)

    def real_arraylike(value):
        if isinstance(value, np.ndarray):
            return value.real.tolist()
        if isinstance(value, (tuple, list)):
            return [real_arraylike(v) for v in value]
        return float(np.real(value))

    tangent = getattr(context, method)(*(real_arraylike(t) for t in tangents))
    np.testing.assert_array_equal(tangent, np.zeros_like(value))


@pytest.mark.interface
@pytest.mark.parametrize(
    "kind,index",
    [
        ("spheres", 0),
        ("spheres", 1),
        ("spheres", 3),
        ("spherical-particles", 1),
        ("cylindrical-particles", 1),
    ],
)
def test_cluster_pushforward_rejects_complex_real_input_direction(kind, index):
    value, context, tangents, method, _ = _record(kind)
    bad = list(tangents)
    bad[index] = np.asarray(bad[index]) + 1j
    with pytest.raises(ValueError, match="real input must be real"):
        getattr(context, method)(*bad)
    np.testing.assert_array_equal(
        getattr(context, method)(*tangents), np.zeros_like(value)
    )


@pytest.mark.interface
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_cluster_context_reuses_saved_work_after_rejected_and_successful_derivatives(
    kind, invalid
):
    value, context, tangents, method, index = _record(kind)
    bad = list(tangents)
    bad[index] = (
        tangents[index][:-1]
        if invalid == "shape"
        else np.full_like(tangents[index], np.nan)
    )
    with pytest.raises(ValueError):
        getattr(context, method)(*bad)
    tangent = getattr(context, method)(*tangents)
    np.testing.assert_array_equal(tangent, np.zeros_like(value))
    pullback = getattr(
        context, "pullback_blocks" if method.endswith("blocks") else "pullback"
    )
    gradient = pullback(np.ones_like(value))
    np.testing.assert_array_equal(getattr(context, method)(*tangents), tangent)
    assert_tree_allclose(pullback(np.ones_like(value)), gradient)


@pytest.mark.interface
@pytest.mark.parametrize("kind", ["dense-factor", "block-factor", "sphere-factor"])
def test_illuminate_rejects_wrong_block_method_without_changing_context(kind):
    value, context, tangents, method, _ = _record(kind)
    local, coupling, incident = tangents
    if method == "pushforward":
        with pytest.raises(ValueError, match="use pushforward for dense-local"):
            context.pushforward_blocks([local], coupling, incident)
    else:
        with pytest.raises(ValueError, match="use pushforward_blocks for block-local"):
            context.pushforward(np.zeros_like(coupling), coupling, incident)
    np.testing.assert_array_equal(
        getattr(context, method)(*tangents), np.zeros_like(value)
    )


@pytest.mark.gradients
def test_dense_factor_jvp_matches_differences_and_pullback():
    def record(local, coupling, incident):
        return _native.InteractionFactor(local, coupling).record(incident)

    check_pushforward(
        record,
        np.array([[0.1 + 0.02j, 0.03j], [0.02, 0.2 - 0.01j]]),
        np.array([[0.0, 0.1j], [0.12 + 0.03j, 0.0]]),
        np.array([[1.0, 0.3j], [0.2 + 0.1j, 0.8]]),
        seed=42,
    )
