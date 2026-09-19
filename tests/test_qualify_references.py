"""Small independent checks of reference formulas and qualification bookkeeping."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

mp = pytest.importorskip(
    "mpmath", reason="optional high-precision qualification dependency"
)
REFERENCE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/qualify_references.py")
)


@pytest.mark.parametrize("kind", ["j", "y", "h1", "h2"])
def test_spherical_degree_zero_matches_elementary_functions(kind):
    with mp.workdps(70):
        z = mp.mpc("1.3", "0.2")
        elementary = {
            "j": lambda x: mp.sin(x) / x,
            "y": lambda x: -mp.cos(x) / x,
            "h1": lambda x: -mp.j * mp.exp(mp.j * x) / x,
            "h2": lambda x: mp.j * mp.exp(-mp.j * x) / x,
        }[kind]
        assert abs(REFERENCE["radial"](kind, 0, z, True) - elementary(z)) < mp.mpf(
            "1e-65"
        )
        assert abs(
            REFERENCE["radial"](kind, 0, z, True, True) - mp.diff(elementary, z)
        ) < mp.mpf("1e-65")


@settings(max_examples=5, deadline=None)
@given(size=st.floats(0.1, 4), epsilon=st.floats(1.2, 8), mu=st.floats(0.8, 2))
def test_reference_mie_lossless_power_and_electric_magnetic_duality(size, epsilon, mu):
    with mp.workdps(70):
        matrix = REFERENCE["homogeneous_mie"](2, size, epsilon, mu)
        dual = REFERENCE["homogeneous_mie"](2, size, mu, epsilon)
        assert abs(matrix[0] - dual[0]) < mp.mpf("1e-65")
        assert abs(matrix[1] + dual[1]) < mp.mpf("1e-65")
        for value in (matrix[0] + matrix[1], matrix[0] - matrix[1]):
            assert abs(abs(1 + 2 * value) ** 2 - 1) < mp.mpf("1e-65")
        assert REFERENCE["homogeneous_mie"](2, size, 1, 1) == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "epsilon,mu", [(2.25, 1), (3.1 + 0.2j, 1), (2.5 + 0.2j, 1.3 + 0.1j), (-8 + 0.4j, 1)]
)
def test_high_precision_mie_uses_public_helicity_convention(epsilon, mu):
    case = {
        "id": "smoke-mie",
        "family": "mie_sphere",
        "function": "mie",
        "parameters": {
            "degree": 2,
            "size": 1.3,
            "epsilon": REFERENCE["pair"](epsilon),
            "mu": REFERENCE["pair"](mu),
        },
        "regime": "test",
    }
    result = REFERENCE["qualify_case"](case, 60)
    assert result["reference_stable"]
    assert result["float64_no_overflow"]
    assert all(row["status"] == "passed" for row in result["backends"].values()), result


def test_error_metrics_handle_near_zeros_nonfinite_values_and_tiny_errors():
    with mp.workdps(70):
        result = REFERENCE["errors"]([1e-15], [mp.mpf("1e-40")], 2e-13, 2e-11)
        assert result["status"] == "passed"
        assert result["relative_l2_error"] > 1e24
        assert result["max_abs_error"] == pytest.approx(1e-15)
        tiny = REFERENCE["errors"]([0], [mp.mpf("1e-500")], 2e-13, 2e-11)
        assert tiny["max_abs_error"] is None
        assert mp.mpf(tiny["errors_high_precision"]["max_abs_error"]) > 0
        assert tiny["relative_l2_error"] == 1
        assert (
            REFERENCE["errors"]([0], [mp.mpf(0)], 1e-13, 1e-11)["relative_l2_error"]
            is None
        )
        with pytest.raises(ValueError, match="nonfinite"):
            REFERENCE["errors"]([np.nan], [mp.mpf(1)], 1e-13, 1e-11)
        json.dumps(tiny, allow_nan=False)


def test_exception_and_reference_overflow_are_retained(monkeypatch):
    qualify = REFERENCE["qualify_case"]
    monkeypatch.setitem(qualify.__globals__, "reference", lambda _: [mp.mpf("1e500")])
    monkeypatch.setitem(
        qualify.__globals__,
        "evaluate",
        lambda *_: (_ for _ in ()).throw(ValueError("intentional failure")),
    )
    case = {
        "id": "overflow",
        "family": "spherical_bessel",
        "parameters": {},
        "function": "spherical_yn",
        "regime": "test",
    }
    result = qualify(case, 60)
    assert result["reference_stable"]
    assert not result["float64_no_overflow"]
    assert result["backends"]["treams-rs"]["error"] == "intentional failure"
    observations = list(REFERENCE["observations"](result))
    assert all(
        row["error"] is None and row["status"] == "error" for row in observations
    )
    assert all(row["message"] == "intentional failure" for row in observations)
    json.dumps(result, allow_nan=False)


def test_planned_grid_has_all_families_and_meaningful_stress_regimes():
    cases = list(REFERENCE["case_grid"]())
    assert set(REFERENCE["FAMILIES"]) == {case["family"] for case in cases}
    assert {
        "small",
        "near_regular_zero",
        "cut_above",
        "cut_below",
        "very_large",
        "near_endpoint",
        "metallic",
        "near_zero_contrast",
    } <= {case["regime"] for case in cases}
    near_zero = [case for case in cases if case["regime"] == "near_regular_zero"]
    assert len(near_zero) == 2
    assert all(
        case["parameters"]["order"] == 0
        and case["parameters"]["kind"] == "j"
        and not case["parameters"]["derivative"]
        for case in near_zero
    )
    encoded = [json.dumps(case, sort_keys=True) for case in cases]
    assert len(encoded) == len(set(encoded))


def test_completed_failed_qualification_emits_json_and_success_exit(
    monkeypatch, capsys
):
    main = REFERENCE["main"]
    globals_ = main.__globals__
    case = {
        "family": "cylindrical_bessel",
        "function": "jv",
        "parameters": {},
        "regime": "test",
    }
    monkeypatch.setitem(globals_, "case_grid", lambda: iter([case]))
    monkeypatch.setitem(globals_, "fingerprints", lambda: {})
    monkeypatch.setitem(
        globals_,
        "qualify_case",
        lambda case, _digits: {
            **case,
            "reference_stable": True,
            "float64_no_overflow": True,
            "backends": {
                "treams-rs": {"status": "failed", "max_scaled_error": 2.0},
                "treams": {"status": "passed", "max_scaled_error": 0.5},
            },
        },
    )
    monkeypatch.setattr(sys, "argv", ["qualify_references.py", "--threads", "1"])
    for name in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] and not result["passed"]
    assert result["observations"][2]["status"] == "failed"
    assert result["observations"][2]["error"] == 2.0


def test_reference_range_distinguishes_overflow_subnormal_and_underflow():
    with mp.workdps(70):
        result = REFERENCE["reference_range"](
            [mp.mpc("1e-500", "1e-310"), mp.mpc(0, 1)]
        )
        assert result["float64_no_overflow"]
        assert result["reference_component_ranges"] == [
            {"real": "below_min_subnormal", "imag": "subnormal"},
            {"real": "zero", "imag": "normal"},
        ]
        assert result["float64_below_min_subnormal_components"] == 1
        assert result["float64_subnormal_components"] == 1
        assert result["float64_rounds_to_zero_components"] == 1
        assert not REFERENCE["reference_range"]([mp.mpf("1e500")])[
            "float64_no_overflow"
        ]


def test_tiny_references_are_qualified_with_absolute_error_gate(monkeypatch):
    qualify = REFERENCE["qualify_case"]
    monkeypatch.setitem(qualify.__globals__, "reference", lambda _: [mp.mpf("1e-500")])
    monkeypatch.setitem(qualify.__globals__, "evaluate", lambda *_: [0])
    case = {
        "id": "tiny",
        "family": "spherical_bessel",
        "parameters": {},
        "function": "spherical_jn",
        "regime": "test",
    }
    result = qualify(case, 60)
    assert result["float64_no_overflow"]
    assert result["float64_rounds_to_zero_components"] == 1
    native = result["backends"]["treams-rs"]
    assert native["status"] == "passed"
    assert native["relative_l2_error"] == 1
    assert native["max_abs_error"] is None
    assert mp.mpf(native["errors_high_precision"]["max_abs_error"]) > 0


def test_wigner_factorial_sum_low_order_sign_and_orthogonality():
    with mp.workdps(70):
        beta = mp.mpf("0.7")
        function = REFERENCE["wigner_smalld"]
        assert function(0, 0, 0, beta) == 1
        assert abs(function(1, -1, 0, beta) - mp.sin(beta) / mp.sqrt(2)) < mp.mpf(
            "1e-65"
        )
        assert abs(function(1, 0, -1, beta) + mp.sin(beta) / mp.sqrt(2)) < mp.mpf(
            "1e-65"
        )
        matrix = mp.matrix(
            [[function(2, m, k, beta) for k in range(-2, 3)] for m in range(-2, 3)]
        )
        assert mp.norm(matrix.T * matrix - mp.eye(5)) < mp.mpf("1e-65")
        case = {
            "family": "wigner_rotation",
            "function": "wignerd",
            "parameters": {
                "degree": 1,
                "m_out": -1,
                "m_in": 0,
                "angles": [0.2, 0.7, -0.3],
            },
        }
        expected = mp.exp(mp.j * mp.mpf(0.2)) * mp.sin(mp.mpf(0.7)) / mp.sqrt(2)
        assert abs(REFERENCE["reference"](case)[0] - expected) < mp.mpf("1e-65")


def test_wigner_diagnostics_append_without_changing_original_grid():
    cases = list(REFERENCE["case_grid"]())
    original = json.dumps(cases[:1930], sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(original).hexdigest()
        == "7dfaf41b66da6ca3a8711aff27ed2ed93f540776e06d9d75f85d058f88608fd1"
    )
    diagnostics = cases[1930:]
    assert len(cases) == 2050 and len(diagnostics) == 120
    assert all(case["selection"] == "post_failure_diagnostic" for case in diagnostics)
    failed_pairs = [
        case
        for case in diagnostics
        if case["function"] == "rotation_entry"
        and case["parameters"]["degree"] == 24
        and (case["parameters"]["m_out"], case["parameters"]["m_in"])
        in ((-15, -11), (15, 11))
    ]
    assert len(failed_pairs) == 4
    assert {case["parameters"]["polarization"] for case in failed_pairs} == {0, 1}


def test_rotation_disagreement_matching_gate_does_not_evaluate_reference(monkeypatch):
    certify = REFERENCE["certify_rotation_disagreements"]
    monkeypatch.setitem(
        certify.__globals__, "reference", lambda _: pytest.fail("unneeded HP reference")
    )
    result = certify(np.eye(2), np.eye(2), [(0, 1, -1, 0), (0, 1, 0, 0)], [0, 0, 0])
    assert result["passed"] and result["original_gate_passed"]
    assert result["upstream_matched_entries"] == 4 and result["mismatch_count"] == 0
    with pytest.raises(ValueError, match="matching square"):
        certify(np.eye(2), np.ones((2, 1)), [(0, 1, -1, 0)], [0, 0, 0])


def test_rotation_disagreement_checks_particle_degree_polarization_and_nonfinite():
    certify = REFERENCE["certify_rotation_disagreements"]
    # Deliberately shuffled modes distinguish particle, degree and polarization.
    basis = [(1, 1, -1, 0), (0, 2, 0, 0), (1, 1, 0, 0), (1, 1, 0, 1), (2, 1, 0, 0)]
    native, upstream = np.eye(5, dtype=complex), np.eye(5, dtype=complex)
    for row, column in ((0, 1), (2, 3), (2, 4)):
        upstream[row, column] = 1e-5
    result = certify(native, upstream, basis, [0, 0, 0])
    assert result["passed"] and not result["original_gate_passed"]
    assert result["mismatch_count"] == 3
    assert all(entry["structural_zero"] for entry in result["entries"])
    assert all(entry["treams"]["status"] == "failed" for entry in result["entries"])
    native[0, 1] = 1e-5
    upstream[0, 1] = 0
    assert not certify(native, upstream, basis, [0, 0, 0])["passed"]
    native[0, 1] = complex(float("nan"), 0)
    result = certify(native, upstream, basis, [0, 0, 0])
    assert (
        not result["passed"] and result["entries"][0]["treams-rs"]["status"] == "error"
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.filterwarnings("ignore:\x27where\x27 used without \x27out\x27:UserWarning")
def test_rotation_disagreement_adjudicates_actual_l24_production_entries():
    certify = REFERENCE["certify_rotation_disagreements"]
    angles = [0.2, 0.7, -0.3]
    basis = [(0, 24, m, 1) for m in range(-24, 25)]
    native = REFERENCE["rotation_block"]("treams-rs", 24, tuple(angles), 1)
    upstream = REFERENCE["rotation_block"]("treams", 24, tuple(angles), 1)
    result = certify(native, upstream, basis, angles)
    assert result["passed"], result
    assert not result["original_gate_passed"] and result["mismatch_count"] == 2
    assert {
        (entry["output_mode"][2], entry["input_mode"][2]) for entry in result["entries"]
    } == {(-15, -11), (15, 11)}
    assert all(
        entry["treams-rs"]["status"] == "passed"
        and entry["treams"]["status"] == "failed"
        for entry in result["entries"]
    )
    assert all(
        entry["original_comparison"]["status"] == "failed"
        for entry in result["entries"]
    )
    json.dumps(result, allow_nan=False)
