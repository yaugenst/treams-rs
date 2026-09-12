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
@settings(max_examples=20, deadline=None)
@given(
    rows=st.integers(1, 12),
    inner=st.integers(15, 26),
    columns=st.integers(1, 5),
    seed=st.integers(0, 10000),
    strided=st.booleans(),
)
def test_rectangular_matmul_adjoint_flags(
    rows: int, inner: int, columns: int, seed: int, strided: bool
) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(rows, inner)) + 1j * rng.normal(size=(rows, inner))
    b = rng.normal(size=(inner, columns)) + 1j * rng.normal(size=(inner, columns))
    if strided:
        a = np.asfortranarray(a)[::-1, ::-1]
        b = np.asfortranarray(b)[::-1, ::-1]
    expected = a @ b
    device = cuda.Device()
    for adjoint_left in (False, True):
        for adjoint_right in (False, True):
            left = device.upload(a.conj().T if adjoint_left else a)
            right = device.upload(b.conj().T if adjoint_right else b)
            result = device.matmul(
                left,
                right,
                adjoint_left=adjoint_left,
                adjoint_right=adjoint_right,
            )
            assert result.shape == (rows, columns)
            actual = result.numpy()
            assert actual.dtype == np.complex128
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
    with pytest.raises(ValueError, match="dimensions do not match"):
        device.matmul(device.upload(a), device.upload(b), adjoint_left=True)


@pytest.mark.skipif(
    os.environ.get("TREAMS_TEST_CUDA") != "1", reason="explicit CUDA hardware test"
)
@settings(max_examples=20, deadline=None)
@given(
    modes=st.integers(1, 12),
    extra_rows=st.integers(1, 20),
    columns=st.integers(1, 4),
    seed=st.integers(0, 10000),
)
def test_fixed_operator_coefficient_pullback(
    modes: int, extra_rows: int, columns: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    rows = modes + extra_rows
    operator = rng.normal(size=(rows, modes)) + 1j * rng.normal(size=(rows, modes))
    direction = rng.normal(size=(modes, columns)) + 1j * rng.normal(
        size=(modes, columns)
    )
    g = rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
    h = rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
    alpha, beta = 0.3 + 0.7j, -0.4 + 0.2j
    device = cuda.Device()
    resident_operator = device.upload(operator)
    derivative = device.matmul(resident_operator, device.upload(direction)).numpy()
    pullback_g = device.matmul(
        resident_operator, device.upload(g), adjoint_left=True
    ).numpy()
    pullback_h = device.matmul(
        resident_operator, device.upload(h), adjoint_left=True
    ).numpy()
    combined = device.matmul(
        resident_operator, device.upload(alpha * g + beta * h), adjoint_left=True
    ).numpy()
    np.testing.assert_allclose(
        pullback_g, operator.conj().T @ g, rtol=3e-12, atol=3e-13
    )
    # Equality of the full complex pairing also checks the native real-loss pairing.
    np.testing.assert_allclose(
        np.vdot(g, derivative),
        np.vdot(pullback_g, direction),
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        combined, alpha * pullback_g + beta * pullback_h, rtol=3e-12, atol=3e-12
    )
    assert resident_operator.shape == operator.shape
    assert resident_operator.nbytes == operator.nbytes
    np.testing.assert_array_equal(resident_operator.numpy(), operator)


@pytest.mark.skipif(
    os.environ.get("TREAMS_TEST_CUDA") != "1", reason="explicit CUDA hardware test"
)
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_physical_plane_coefficient_pullback(poltype: str) -> None:
    wavevectors = np.asarray(
        [
            [0.2 + 0.01j, 0.3 - 0.02j, 1.1 + 0.04j],
            [-0.4 + 0.02j, 0.25 + 0.03j, 0.9 - 0.01j],
            [0.35 - 0.04j, -0.2 + 0.01j, 0.7 + 0.02j],
        ]
    )
    polarizations = np.asarray([0, 1, 0])
    coefficients = np.asarray([0.3 + 0.2j, -0.4j, -0.1 + 0.5j])
    points = np.asarray(
        [[0.1, -0.2, 0.3], [-0.3, 0.4, 0.2], [0.5, 0.1, -0.2], [0.2, 0.3, 0.4]]
    )
    operator, _ = diff.plane_field(
        None, points, wavevectors, polarizations, poltype=poltype
    )
    field, context = diff.plane_field(
        coefficients, points, wavevectors, polarizations, poltype=poltype
    )
    rng = np.random.default_rng(203)
    g = rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)
    expected = context.pullback(g)[0]
    device = cuda.Device()
    resident_operator = device.upload(
        operator.reshape(3 * len(points), len(coefficients))
    )
    sampled = device.matmul(
        resident_operator, device.upload(coefficients.reshape(-1, 1))
    ).numpy()
    actual = device.matmul(
        resident_operator, device.upload(g.reshape(-1, 1)), adjoint_left=True
    ).numpy()
    np.testing.assert_allclose(
        sampled.reshape(field.shape), field, rtol=3e-12, atol=3e-13
    )
    np.testing.assert_allclose(actual[:, 0], expected, rtol=3e-12, atol=3e-13)


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
