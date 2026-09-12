"""CUDA is opt-in; run hardware tests with TREAMS_TEST_CUDA=1."""

import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import treams_rs as rs
from treams_rs import cuda, diff


def test_compiled_out() -> None:
    if cuda.compiled():
        pytest.skip("this extension explicitly includes CUDA")
    with pytest.raises(RuntimeError, match="compiled out"):
        cuda.Device()


@pytest.mark.skipif(
    os.environ.get("TREAMS_TEST_CUDA") != "1", reason="explicit CUDA hardware test"
)
@settings(max_examples=16, deadline=None)
@given(n=st.integers(1, 24), columns=st.integers(1, 6), seed=st.integers(0, 10000))
def test_solve_and_pullback(n: int, columns: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a += (2 * n + 1) * np.eye(n)
    b = rng.normal(size=(n, columns)) + 1j * rng.normal(size=(n, columns))
    g = rng.normal(size=(n, columns)) + 1j * rng.normal(size=(n, columns))
    if seed % 2 == 0:
        a = np.asfortranarray(a)[:, ::-1]
        b = np.asfortranarray(b)[:, ::-1]
        g = np.asfortranarray(g)[:, ::-1]
    device = cuda.Device()
    factor = device.factor(a)
    np.testing.assert_allclose(
        factor.solve(np.eye(n)), np.linalg.solve(a, np.eye(n)), rtol=2e-12, atol=2e-13
    )
    x, context = factor.solve_with_pullback(b)
    reference, cpu_context = diff.solve(a, b)
    np.testing.assert_allclose(x, reference, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        factor.solve(b, adjoint=True),
        np.linalg.solve(a.conj().T, b),
        rtol=2e-12,
        atol=2e-13,
    )
    resident = factor.solve_device(device.upload(b))
    np.testing.assert_allclose(
        device.matmul(device.upload(a), resident).numpy(), b, rtol=2e-12, atol=2e-13
    )
    with pytest.raises(ValueError, match="invalid CUDA solve cotangent"):
        context.pullback(np.zeros((n + 1, columns), dtype=np.complex128))
    actual = context.pullback(g)
    expected = cpu_context.pullback(g)
    for found, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(found, wanted, rtol=4e-12, atol=2e-13)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@pytest.mark.skipif(
    os.environ.get("TREAMS_TEST_CUDA") != "1", reason="explicit CUDA hardware test"
)
@pytest.mark.parametrize("order", [2, 5, 8])
def test_physical_sphere_cluster(order: int) -> None:
    sphere = rs.TMatrix.sphere(
        order, 1.0, [0.35], [rs.Material(2.3 + 0.02j), rs.Material()]
    )
    cluster = rs.TMatrix.cluster([sphere, sphere], [[0, 0, 0], [1.1, 0.2, 0]])
    incident = np.column_stack(
        [rs.plane_wave([0, 0, 1], pol, k0=1.0).expand(cluster.basis) for pol in (0, 1)]
    )
    operator = cluster.interaction()
    rhs = cluster.array @ incident
    gpu = cuda.Device().factor(operator).solve(rhs)
    reference = cluster.interaction.solve().array @ incident
    np.testing.assert_allclose(gpu, reference, rtol=3e-9, atol=3e-13)


@pytest.mark.skipif(
    os.environ.get("TREAMS_TEST_CUDA_TILE") != "1",
    reason="explicit CUDA Tile hardware test",
)
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_tile_fields(poltype: str) -> None:
    wavevectors = np.asarray([[0.2, 0.3, 1.1 + 0.02j], [-0.1, 0.4, 0.8 + 0.01j]])
    polarizations = np.asarray([0, 1])
    coefficients = np.asarray([0.3 + 0.1j, -0.2j])
    points = np.random.default_rng(12).normal(size=(129, 3))
    expansion = cuda.PlaneWaves(
        wavevectors, polarizations, coefficients, poltype=poltype
    )
    expected, _ = diff.plane_field(
        coefficients,
        points,
        wavevectors,
        polarizations,
        poltype=poltype,
        fixed_vectors=True,
    )
    np.testing.assert_allclose(
        expansion.evaluate(points), expected, rtol=3e-12, atol=3e-13
    )
