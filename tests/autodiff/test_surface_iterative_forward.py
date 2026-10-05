"""All framework bridges carry native surface and matrix-free JVPs."""

import contextlib
import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs._records import DerivativeContext
from treams_rs.iterative import SphereCluster

from _support import complex_normal, jax_x64

pytestmark = pytest.mark.gradients


def iterative_case():
    rng = np.random.default_rng(601)
    inputs = (
        np.asarray(1.3),
        np.array([0.2, 0.24]),
        np.array([2.3 + 0.04j, 3.1 + 0.02j]),
        np.array([[0.1, 0.0, -0.1], [1.2, 0.2, 0.4]]),
        complex_normal(rng, (12, 2)),
    )
    directions = (
        np.asarray(0.07),
        np.array([0.03, -0.02]),
        np.array([0.2 - 0.03j, -0.1 + 0.04j]),
        np.array([[0.03, 0.02, -0.04], [-0.02, 0.01, 0.03]]),
        complex_normal(rng, (12, 2)) * 0.03,
    )

    def record(k0, radii, epsilon, positions, incident):
        operator = SphereCluster(1, k0, radii, epsilon, positions)
        value, context = operator.record(incident, rtol=2e-12)
        # Reports remain available on the physical solver's context. Framework
        # dynamic outputs contain only the differentiable scattered coefficients.
        return value.coefficients, DerivativeContext(
            lambda cotangent: context.pullback(cotangent)[:-1],
            lambda *tangents: context.pushforward(*tangents).coefficients,
        )

    return record, inputs, directions


def surface_case():
    nodes, weights = np.polynomial.legendre.leggauss(16)
    theta = (nodes + 1) * np.pi / 2
    inputs = (
        0.3 * (1 + 0.2 * np.cos(theta) ** 2),
        -0.12 * np.cos(theta) * np.sin(theta),
        np.array([[1.7 + 0.1j, 1.8 + 0.1j], [1.2 + 0.02j, 1.3 + 0.02j]]),
        np.array([0.8 + 0.01j, 1.1 - 0.02j]),
    )
    directions = (
        0.02 * np.cos(theta),
        -0.02 * np.sin(theta),
        np.array([[0.1 + 0.03j, -0.2 + 0.01j], [-0.1 - 0.02j, 0.15 - 0.01j]]),
        np.array([0.07 + 0.02j, -0.04 + 0.01j]),
    )

    def record(*args):
        return tr.diff.ebcm_qmat(
            *args,
            theta=theta,
            weights=weights * np.pi / 2,
            destination=tr.SphericalBasis.default(1),
        )

    return record, inputs, directions


@pytest.mark.parametrize("case", [iterative_case, surface_case])
@pytest.mark.parametrize("framework", ["jax", "torch", "autograd", "advect"])
def test_framework_jvp_matches_native(case, framework):
    engine = pytest.importorskip(framework)
    adapter = importlib.import_module(f"treams_rs.{framework}")
    record, inputs, directions = case()
    expected, context = record(*inputs)
    expected_tangent = context.pushforward(*directions)
    scope = jax_x64() if framework == "jax" else contextlib.nullcontext()
    with scope:
        if framework == "jax":
            operation = adapter.wrap(record, *inputs)
            value, tangent = engine.jit(lambda xs, ds: engine.jvp(operation, xs, ds))(
                inputs,
                directions,
            )
        elif framework == "torch":
            value, tangent = engine.func.jvp(
                adapter.wrap(record),
                tuple(engine.tensor(x) for x in inputs),
                tuple(engine.tensor(x) for x in directions),
            )
            value, tangent = value.detach().numpy(), tangent.detach().numpy()
        elif framework == "autograd":
            value, tangent = engine.make_jvp(
                adapter.wrap(record), tuple(range(len(inputs)))
            )(
                *inputs,
            )(directions)
        else:
            # Advect's shared record bridge is the layer behind its named physics
            # functions; it has no public generic wrap helper.
            value, tangent = engine.jvp(
                lambda *values: adapter._operation(record, *values),
                argnums=tuple(range(len(inputs))),
            )(*inputs, tangents=directions)
    assert_allclose(value, expected, rtol=2e-12, atol=1e-13)
    assert_allclose(tangent, expected_tangent, rtol=2e-11, atol=1e-13)
