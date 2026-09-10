"""Public scattering workflows and expansion adjoints against treams."""

import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st

import treams_rs as rust
from treams_rs import _native, diff


@pytest.mark.public_e2e
@given(
    degree=st.integers(1, 4),
    radius=st.floats(0.1, 1.5),
    epsilon=st.floats(1.1, 8),
    parity=st.booleans(),
)
def test_sphere_api_and_optical_theorem(degree, radius, epsilon, parity):
    poltype = "parity" if parity else "helicity"
    actual = rust.TMatrix.sphere(
        degree, 1.3, radius, [rust.Material(epsilon), 1], poltype
    )
    expected = treams.TMatrix.sphere(degree, 1.3, radius, [epsilon, 1], poltype)
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-11)
    np.testing.assert_allclose(
        actual.xs_ext_avg, expected.xs_ext_avg, atol=1e-12, rtol=3e-11
    )
    np.testing.assert_allclose(
        actual.xs_sca_avg, expected.xs_sca_avg, atol=1e-12, rtol=3e-11
    )
    np.testing.assert_allclose(
        actual.xs_ext_avg, actual.xs_sca_avg, atol=1e-12, rtol=3e-11
    )
    np.testing.assert_array_equal(actual.basis.lms, expected.basis.lms)
    np.testing.assert_allclose(
        actual.changepoltype().changepoltype(), actual, atol=2e-14
    )


@pytest.mark.public_e2e
@pytest.mark.parametrize(
    "poltype,embedding", [("helicity", (1.2, 1.1, 0.02)), ("parity", (1.2, 1.1, 0))]
)
def test_heterogeneous_cluster_global_expansion_and_cross_sections(poltype, embedding):
    positions = [[0, 0, 0], [0.3, 0.2, 1.4]]
    matrices = []
    for package in (rust, treams):
        spheres = [
            package.TMatrix.sphere(
                lmax,
                1.1,
                [0.1, radius],
                [
                    (2.1, 1.2, 0.03 if poltype == "helicity" else 0),
                    (3 + 0.1j, 1, 0.01 if poltype == "helicity" else 0),
                    embedding,
                ],
                poltype,
            )
            for lmax, radius in [(1, 0.2), (2, 0.3)]
        ]
        cluster = package.TMatrix.cluster(spheres, positions)
        solved = cluster.interaction.solve()
        global_matrix = solved.expand(package.SphericalWaveBasis.default(4))
        incident = np.random.default_rng(1).normal(size=len(solved)) + 0.3j
        if package is treams:
            incident = treams.PhysicsArray(
                incident,
                basis=solved.basis,
                k0=solved.k0,
                material=solved.material,
                poltype=solved.poltype,
                modetype="regular",
            )
        matrices.append((cluster, solved, global_matrix, solved.xs(incident)))
    for actual, expected in zip(*matrices, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=3e-12)
    for name in ("xs_ext_avg", "xs_sca_avg"):
        np.testing.assert_allclose(
            getattr(matrices[0][2], name),
            getattr(matrices[1][2], name),
            rtol=2e-10,
            atol=1e-12,
        )


@pytest.mark.ad_contract
@given(
    to_degree=st.integers(1, 3),
    source_degree=st.integers(1, 3),
    to_order=st.integers(-1, 1),
    source_order=st.integers(-1, 1),
    parity=st.booleans(),
)
def test_regular_translation_derivative_at_origin(
    to_degree, source_degree, to_order, source_order, parity
):
    to, source = (
        (to_degree, to_order, 1),
        (source_degree, source_order, 0 if parity else 1),
    )
    k = 1.2 + 0.1j
    _, analytic, _ = _native.translation(to, source, k, (0, 0, 0), not parity, False)
    for axis in range(3):
        step = np.eye(3)[axis] * 1e-6
        finite_difference = (
            _native.translation(to, source, k, tuple(step), not parity, False)[0]
            - _native.translation(to, source, k, tuple(-step), not parity, False)[0]
        ) / 2e-6
        np.testing.assert_allclose(
            analytic[axis], finite_difference, atol=3e-10, rtol=2e-8
        )


@pytest.mark.ad_contract
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("parameter", ["destination", "source", "ks"])
def test_expansion_pullback(parameter, singular):
    destination = np.array([[0.2, -0.1, 0.3], [0.2, 1.1, 0.4]])
    source = np.array([[-0.1, 0.2, 1.1]])
    ks = np.array([1.1 + 0.02j, 1.2 + 0.1j])
    args = dict(destination=destination, source=source, ks=ks)

    def forward(destination, source, ks):
        return diff.expansion(
            rust.SphericalWaveBasis.default(1, 2, destination),
            rust.SphericalWaveBasis.default(2, positions=source),
            ks,
            singular=singular,
        )

    value, residual = forward(**args)
    rng = np.random.default_rng(21)
    cotangent = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = dict(zip(args, residual.pullback(cotangent), strict=True))
    direction = rng.normal(size=args[parameter].shape)
    if parameter == "ks":
        direction = direction + 1j * rng.normal(size=direction.shape)
    plus, minus = dict(args), dict(args)
    step = 1e-6
    plus[parameter] = args[parameter] + step * direction
    minus[parameter] = args[parameter] - step * direction
    numerical = np.vdot(cotangent, forward(**plus)[0] - forward(**minus)[0]).real / (
        2 * step
    )
    np.testing.assert_allclose(
        np.vdot(gradients[parameter], direction).real, numerical, rtol=3e-7, atol=3e-6
    )
    np.testing.assert_allclose(
        gradients["destination"].sum(axis=0) + gradients["source"].sum(axis=0),
        0,
        atol=1e-9,
    )


@pytest.mark.python_contract
def test_invalid_gradient_does_not_consume_residual():
    basis = rust.SphericalWaveBasis.default(1)
    value, residual = diff.expansion(basis, basis, [1, 1])
    with pytest.raises(ValueError, match="shape"):
        residual.pullback(np.zeros((2, 2), dtype=complex))
    residual.pullback(np.ones_like(value))
    with pytest.raises(ValueError, match="consumed"):
        residual.pullback(np.ones_like(value))
