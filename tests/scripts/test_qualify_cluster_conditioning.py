"""The cluster diagnostic separates equation residuals from entrywise agreement.

Its reference uncertainty must bound the encoded equation, even for wrong answers.
"""

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from _scripts import load

mp = pytest.importorskip("mpmath", reason="high-precision oracle")
conditioning = load("qualify_cluster_conditioning")


@pytest.mark.physics
@given(
    scale=st.floats(
        min_value=1e-8, max_value=1e8, allow_nan=False, allow_infinity=False
    )
)
def test_backward_errors_are_scale_invariant_and_detect_perturbations(scale):
    system = np.array([[2, 0.3j], [-0.1j, 3]], dtype=complex)
    solution = np.array([[1 + 0.2j, 0.4], [0.1, -0.5j]])
    rhs = system @ solution
    exact = conditioning.backward_errors(system, solution, rhs)
    assert max(exact.values()) < 1e-15
    perturbed = solution.copy()
    perturbed[0, 0] += 0.01
    original = conditioning.backward_errors(system, perturbed, rhs)
    scaled = conditioning.backward_errors(scale * system, perturbed, scale * rhs)
    for metric in original:
        assert original[metric] > 1e-4
        np.testing.assert_allclose(
            scaled[metric], original[metric], rtol=1e-11, atol=1e-14
        )


@pytest.mark.reference
def test_lapack_condition_estimate_matches_diagonal_system():
    diagonal = np.diag(np.array([1, 1e-3, 2e-7], dtype=complex))
    result = conditioning.condition_estimate(diagonal)
    np.testing.assert_allclose(
        result["system_reciprocal_condition_1_estimate"], 2e-7, rtol=1e-14
    )
    np.testing.assert_allclose(result["system_condition_1_estimate"], 5e6, rtol=1e-14)


@pytest.mark.physics
def test_small_chain_retains_both_backends_and_cutoff_metadata():
    result = conditioning.qualify(2, (1, 2))
    assert result["complete"] and result["passed"]
    assert result["source_unchanged"]
    rows = result["observations"]
    assert len({row["id"] for row in rows}) == len(rows)
    residuals = [
        row
        for row in rows
        if row["metric"] == "illuminated_componentwise_backward_error"
    ]
    assert {row["backend"] for row in residuals} == {"treams", "treams-rs"}
    assert all(row["error"] < 1e-12 for row in residuals)
    cutoff = [row for row in rows if row["reference_kind"] == "self_convergence"]
    assert cutoff and all(row["parameters"]["reference_lmax"] == 2 for row in cutoff)
    assert all(row["parameters"]["lmax"] < 2 for row in cutoff)
    json.dumps(result, allow_nan=False)


def exact_solution(a, rhs):
    with mp.workdps(100):
        matrix = mp.matrix([[mp.mpc(complex(z)) for z in row] for row in a])
        return [
            mp.lu_solve(matrix, mp.matrix([mp.mpc(complex(z)) for z in rhs[:, j]]))
            for j in range(rhs.shape[1])
        ]


def complex_matrix(data, size, bound):
    """Complex entries with real and imaginary parts in [-bound, bound]."""
    parts = st.floats(-bound, bound)
    count = size * size
    pairs = data.draw(st.lists(st.tuples(parts, parts), min_size=count, max_size=count))
    values = np.array(pairs).reshape(size, size, 2)
    return values[..., 0] + 1j * values[..., 1]


@settings(max_examples=20)
@given(
    data=st.data(),
    size=st.integers(2, 5),
    contraction=st.sampled_from([0.5, 0.25, 1.1, 3.0]),
    perturbation=st.sampled_from([0.0, 1e-8, -1e-6]),
)
@pytest.mark.interface
def test_encoded_equation_bound_under_coordinate_changes(
    data, size, contraction, perturbation
):
    """The certificate bounds every column of A X = T in its declared norm.

    A = D B D^-1 with B = I + E, |E| row sums <= contraction and D = diag(2^k),
    k in [-20, 20]. T's rows peak on the diagonal at d_i^2 / 4, so the code's
    coefficient scaling recovers B; a noncontractive B must be refused.
    """
    exponents = st.lists(st.integers(-20, 20), min_size=size, max_size=size)
    d = 2.0 ** np.array(data.draw(exponents))
    e = complex_matrix(data, size, 1.0)
    e[np.arange(size), (np.arange(size) + 1) % size] += 0.1  # no empty rows
    rows = np.sum(abs(e.real) + abs(e.imag), axis=1, keepdims=True)
    e *= contraction / (rows if contraction > 1 else np.maximum(rows, 1))
    a = d[:, None] * (np.eye(size) + e) / d[None, :]
    mixing = complex_matrix(data, size, 0.6)
    np.fill_diagonal(mixing, 1)
    rhs = (d * d)[:, None] * mixing / 4
    reference, scale, b = conditioning._coefficient_scaled_reference(a, rhs)
    if contraction > 1:
        with pytest.raises(ValueError, match="No Neumann certificate"):
            conditioning._encoded_system_bound(a, rhs, reference, scale, b)
        return
    row, column = data.draw(st.tuples(*[st.integers(0, size - 1)] * 2))
    reference[row, column] += perturbation * np.max(abs(reference[:, column]))
    bounds, certificate = conditioning._encoded_system_bound(
        a, rhs, reference, scale, b
    )
    assert certificate["scaled_off_diagonal_norm_upper"] < 1
    answers = exact_solution(a, rhs)
    with mp.workdps(100):
        for j, answer in enumerate(answers):
            differences = [
                mp.mpc(complex(reference[i, j])) - answer[i] for i in range(size)
            ]
            error = max(abs(mp.re(x)) + abs(mp.im(x)) for x in differences)
            assert error <= mp.mpf(float(bounds[j]))


@pytest.mark.interface
def test_certificate_refuses_noncontractive_matrix_and_checks_high_precision():
    a = np.array([[1, 2], [2, 1]], complex)
    with pytest.raises(ValueError, match="No Neumann certificate"):
        conditioning._encoded_system_bound(a, np.eye(2), np.eye(2), np.ones(2), a)
    a = np.array([[1, 1 / 3], [1 / 7, 1]], complex)
    rhs = np.eye(2, dtype=complex)
    x = np.linalg.solve(a, rhs)
    rows = conditioning._high_precision_residuals(
        a, rhs, x, np.array([3.0, 7.0]), [0, 1]
    )
    assert all(row["precision_change"] < 1e-60 for row in rows)
    assert any(row["precision_change"] > 0 for row in rows)
    assert all(len(row["weighted_residual_120_digits_text"]) > 40 for row in rows)


@pytest.mark.interface
def test_public_certificate_rejects_wrong_answers_and_broadcast_shapes():
    import treams

    radii = np.array([0.2, 0.23])
    epsilon = np.array([4 + 0.1j, 4 + 0.1j])
    positions = np.array([[0, 0, 0], [0.8, 0, 0]])
    local = treams.TMatrix.cluster(
        [
            treams.TMatrix.sphere(1, 1.3, r, [e, 1])
            for r, e in zip(radii, epsilon, strict=True)
        ],
        positions,
    )
    upstream = np.asarray(local.interaction.solve())
    reference, _, _ = conditioning._coefficient_scaled_reference(
        np.asarray(local.interaction()), np.asarray(local)
    )
    args = dict(lmax=1, k0=1.3, radii=radii, epsilon=epsilon, positions=positions)
    proof = conditioning.certify_cluster_reference(reference, upstream, **args)
    assert proof["passed"]
    assert proof["high_precision_residual_checks_passed"]
    assert proof["certificate"]["scaled_off_diagonal_norm_upper"] < 1
    bad = reference.copy()
    bad[0, 0] += 1e-6
    assert not conditioning.certify_cluster_reference(bad, upstream, **args)["passed"]
    shape = conditioning.certify_cluster_reference(reference[:1], upstream, **args)
    assert not shape["passed"] and "shape" in shape["failure"]
