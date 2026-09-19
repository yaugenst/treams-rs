"""Reference uncertainty must bound the encoded equation, even for wrong answers."""

import importlib.util
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "conditioning_reference", DIRECTORY / "qualify_cluster_conditioning.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def exact_solution(a, rhs):
    with mp.workdps(100):
        matrix = mp.matrix([[mp.mpc(complex(z)) for z in row] for row in a])
        return [
            mp.lu_solve(matrix, mp.matrix([mp.mpc(complex(z)) for z in rhs[:, j]]))
            for j in range(rhs.shape[1])
        ]


@settings(max_examples=20, deadline=None)
@given(exponent=st.integers(-20, 20), perturbation=st.sampled_from([0.0, 1e-8, -1e-6]))
def test_encoded_equation_bound_under_coordinate_changes(exponent, perturbation):
    d = np.array([2.0**exponent, 2.0**-exponent])
    balanced = np.array([[1, 0.125 + 0.0625j], [-0.125j, 1]], complex)
    a = d[:, None] * balanced / d[None, :]
    rhs = np.diag(d * d * 0.25).astype(complex)
    reference, scale, b = MODULE._coefficient_scaled_reference(a, rhs)
    reference[0, 0] += perturbation
    bounds, certificate = MODULE._encoded_system_bound(a, rhs, reference, scale, b)
    answers = exact_solution(a, rhs)
    with mp.workdps(100):
        for j, answer in enumerate(answers):
            error = max(
                abs(mp.mpc(complex(reference[i, j])) - answer[i]) for i in range(2)
            )
            assert error <= mp.mpf(float(bounds[j]))
    assert certificate["scaled_off_diagonal_norm_upper"] < 1


def test_certificate_refuses_noncontractive_matrix_and_checks_high_precision():
    a = np.array([[1, 2], [2, 1]], complex)
    with pytest.raises(ValueError, match="No Neumann certificate"):
        MODULE._encoded_system_bound(a, np.eye(2), np.eye(2), np.ones(2), a)
    a = np.array([[1, 1 / 3], [1 / 7, 1]], complex)
    rhs = np.eye(2, dtype=complex)
    x = np.linalg.solve(a, rhs)
    rows = MODULE._high_precision_residuals(a, rhs, x, np.array([3.0, 7.0]), [0, 1])
    assert all(row["precision_change"] < 1e-60 for row in rows)
    assert any(row["precision_change"] > 0 for row in rows)
    assert all(len(row["weighted_residual_120_digits_text"]) > 40 for row in rows)


def test_public_certificate_rejects_wrong_answers_and_broadcast_shapes(monkeypatch):
    import treams

    monkeypatch.syspath_prepend(str(DIRECTORY))
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
    reference, _, _ = MODULE._coefficient_scaled_reference(
        np.asarray(local.interaction()), np.asarray(local)
    )
    args = dict(lmax=1, k0=1.3, radii=radii, epsilon=epsilon, positions=positions)
    proof = MODULE.certify_cluster_reference(reference, upstream, **args)
    assert proof["passed"]
    assert proof["high_precision_residual_checks_passed"]
    assert proof["certificate"]["scaled_off_diagonal_norm_upper"] < 1
    bad = reference.copy()
    bad[0, 0] += 1e-6
    assert not MODULE.certify_cluster_reference(bad, upstream, **args)["passed"]
    shape = MODULE.certify_cluster_reference(reference[:1], upstream, **args)
    assert not shape["passed"] and "shape" in shape["failure"]
