"""Small independent checks of reference formulas and qualification bookkeeping."""

import hashlib
import json
import math
import sys

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from _scripts import load

pytestmark = pytest.mark.reference

mp = pytest.importorskip("mpmath", reason="high-precision oracle")
references = load("qualify_references")


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
        assert abs(references.radial(kind, 0, z, True) - elementary(z)) < mp.mpf(
            "1e-65"
        )
        assert abs(
            references.radial(kind, 0, z, True, True) - mp.diff(elementary, z)
        ) < mp.mpf("1e-65")


def assert_identity(residual, *terms):
    """Zero to 40 of the 50 working digits, relative to the largest term."""
    scale = max([mp.mpf(1), *(abs(term) for term in terms)])
    assert abs(residual) <= mp.mpf("1e-40") * scale, (residual, scale)


ORDERS = st.floats(-5, 20)
ARGUMENTS = st.tuples(st.floats(0.1, 20), st.floats(-5, 5))


@settings(max_examples=25)
@given(order=ORDERS, argument=ARGUMENTS)
def test_cylindrical_references_satisfy_wronskians_and_recurrence(order, argument):
    """DLMF 10.5.2, 10.5.5, 10.6.1 and 10.6.2 for real orders, complex arguments."""
    with mp.workdps(50):
        nu, z = mp.mpf(order), mp.mpc(*argument)

        def value(kind, degree=nu, derivative=False):
            return references.radial(kind, degree, z, False, derivative)

        for first, second, expected in (
            ("j", "y", 2 / (mp.pi * z)),
            ("h1", "h2", -4j / (mp.pi * z)),
        ):
            left = value(first) * value(second, derivative=True)
            right = value(first, derivative=True) * value(second)
            assert_identity(left - right - expected, left, right)
        for kind in ("j", "y", "h1", "h2"):
            lower, upper = value(kind, nu - 1), value(kind, nu + 1)
            middle = 2 * nu / z * value(kind)
            assert_identity(lower + upper - middle, lower, upper, middle)
            # A multiple of C itself cancels from the Wronskians; this does not.
            derivative, shifted = value(kind, derivative=True), nu / z * value(kind)
            assert_identity(derivative - lower + shifted, derivative, lower, shifted)


@settings(max_examples=25)
@given(degree=st.integers(0, 40), argument=ARGUMENTS)
def test_spherical_references_satisfy_wronskians_and_recurrence(degree, argument):
    """DLMF 10.50.1, 10.50.2, 10.51.1 and 10.51.2 for complex arguments."""
    with mp.workdps(50):
        z = mp.mpc(*argument)

        def value(kind, order=degree, derivative=False):
            return references.radial(kind, order, z, True, derivative)

        for first, second, expected in (("j", "y", 1 / z**2), ("h1", "h2", -2j / z**2)):
            left = value(first) * value(second, derivative=True)
            right = value(first, derivative=True) * value(second)
            assert_identity(left - right - expected, left, right)
        for kind in ("j", "y", "h1", "h2"):
            lower, upper = value(kind, degree - 1), value(kind, degree + 1)
            middle = (2 * degree + 1) / z * value(kind)
            assert_identity(lower + upper - middle, lower, upper, middle)
            # DLMF 10.51.2 from below, independent of the implemented form.
            derivative = value(kind, derivative=True)
            shifted = (degree + 1) / z * value(kind)
            assert_identity(derivative - lower + shifted, derivative, lower, shifted)


@settings(max_examples=15)
@given(
    degree=st.integers(0, 8),
    first=st.floats(-np.pi, np.pi),
    second=st.floats(-np.pi, np.pi),
)
@example(degree=2, first=0.7, second=-0.7)  # d(0.7) is orthogonal, d(0) is 1
def test_wigner_reference_is_an_orthogonal_one_parameter_group(degree, first, second):
    """d(b1) d(b2) = d(b1 + b2), d(-b) = d(b)^T and d^T d = 1 for every degree."""
    with mp.workdps(70):
        orders = range(-degree, degree + 1)

        def matrix(beta):
            return mp.matrix(
                [
                    [references.wigner_smalld(degree, m, k, beta) for k in orders]
                    for m in orders
                ]
            )

        b1, b2 = mp.mpf(first), mp.mpf(second)
        d1, d2 = matrix(b1), matrix(b2)
        identity = mp.eye(2 * degree + 1)
        assert mp.mnorm(d1 * d2 - matrix(b1 + b2), 1) < mp.mpf("1e-65")
        assert mp.mnorm(matrix(-b1) - d1.T, 1) < mp.mpf("1e-65")
        assert mp.mnorm(d1.T * d1 - identity, 1) < mp.mpf("1e-65")


# The hypergeometric reference in (1 - x) / 2 loses all digits at the parity
# zeros x = 0 of integer degrees; the qualification grid uses |x| >= 0.3.
LEGENDRE_ARGUMENTS = st.floats(-0.99, 0.99).filter(lambda x: abs(x) >= 0.01)


@settings(max_examples=25)
@given(
    order=st.integers(-5, 5),
    offset=st.floats(1e-3, 34),
    x=LEGENDRE_ARGUMENTS,
)
def test_ferrers_reference_satisfies_the_degree_recurrence(order, offset, x):
    """DLMF 14.10.3 for real degrees above |m| + 1 on (-1, 1)."""
    ferrers = load("qualify_legendre").ferrers
    with mp.workdps(50):
        nu, x = abs(order) + 1 + mp.mpf(offset), mp.mpf(x)
        upper = (nu - order + 1) * ferrers(nu + 1, order, x)
        middle = (2 * nu + 1) * x * ferrers(nu, order, x)
        lower = (nu + order) * ferrers(nu - 1, order, x)
        assert_identity(upper - middle + lower, upper, middle, lower)


@settings(max_examples=25)
@given(order=st.integers(-5, 5), extra=st.integers(0, 15), x=LEGENDRE_ARGUMENTS)
def test_ferrers_reference_matches_mpmath_at_integer_degrees(order, extra, x):
    ferrers = load("qualify_legendre").ferrers
    with mp.workdps(50):
        degree = abs(order) + extra
        expected = mp.legenp(degree, order, mp.mpf(x), type=2)
        actual = ferrers(degree, order, mp.mpf(x))
        assert_identity(actual - expected, expected)


@settings(max_examples=5)
@given(size=st.floats(0.1, 4), epsilon=st.floats(1.2, 8), mu=st.floats(0.8, 2))
def test_reference_mie_lossless_power_and_electric_magnetic_duality(size, epsilon, mu):
    with mp.workdps(70):
        matrix = references.homogeneous_mie(2, size, epsilon, mu)
        dual = references.homogeneous_mie(2, size, mu, epsilon)
        assert abs(matrix[0] - dual[0]) < mp.mpf("1e-65")
        assert abs(matrix[1] + dual[1]) < mp.mpf("1e-65")
        for value in (matrix[0] + matrix[1], matrix[0] - matrix[1]):
            assert abs(abs(1 + 2 * value) ** 2 - 1) < mp.mpf("1e-65")
        assert references.homogeneous_mie(2, size, 1, 1) == [0, 0, 0, 0]


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
            "epsilon": references.pair(epsilon),
            "mu": references.pair(mu),
        },
        "regime": "test",
    }
    result = references.qualify_case(case, 60)
    assert result["reference_stable"]
    assert result["float64_no_overflow"]
    assert all(row["status"] == "passed" for row in result["backends"].values()), result


def test_error_metrics_handle_near_zeros_nonfinite_values_and_tiny_errors():
    with mp.workdps(70):
        result = references.errors([1e-15], [mp.mpf("1e-40")], 2e-13, 2e-11)
        assert result["status"] == "passed"
        assert result["relative_l2_error"] > 1e24
        assert result["max_abs_error"] == pytest.approx(1e-15)
        tiny = references.errors([0], [mp.mpf("1e-500")], 2e-13, 2e-11)
        assert tiny["max_abs_error"] is None
        assert mp.mpf(tiny["errors_high_precision"]["max_abs_error"]) > 0
        assert tiny["relative_l2_error"] == 1
        assert (
            references.errors([0], [mp.mpf(0)], 1e-13, 1e-11)["relative_l2_error"]
            is None
        )
        with pytest.raises(ValueError, match="nonfinite"):
            references.errors([np.nan], [mp.mpf(1)], 1e-13, 1e-11)
        json.dumps(tiny, allow_nan=False)


def test_exception_and_reference_overflow_are_retained(monkeypatch):
    qualify = references.qualify_case
    monkeypatch.setattr(references, "reference", lambda _: [mp.mpf("1e500")])
    monkeypatch.setattr(
        references,
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
    observations = list(references.observations(result))
    assert all(
        row["error"] is None and row["status"] == "error" for row in observations
    )
    assert all(row["message"] == "intentional failure" for row in observations)
    json.dumps(result, allow_nan=False)


def test_planned_grid_has_all_families_and_meaningful_stress_regimes():
    cases = list(references.case_grid())
    assert set(references.FAMILIES) == {case["family"] for case in cases}
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
    main = references.main
    case = {
        "family": "cylindrical_bessel",
        "function": "jv",
        "parameters": {},
        "regime": "test",
    }
    monkeypatch.setattr(references, "case_grid", lambda: iter([case]))
    monkeypatch.setattr(references, "fingerprints", lambda: {})
    monkeypatch.setattr(
        references,
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
    for name in load("_harness").THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    # main() sets the process budget; the block restores it for later tests.
    with references.treams_rs.threads(None):
        assert main() == 0
    assert references.treams_rs.thread_info()["source"] != "set_num_threads"
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] and not result["passed"]
    assert result["observations"][2]["status"] == "failed"
    assert result["observations"][2]["error"] == 2.0


def test_reference_range_distinguishes_overflow_subnormal_and_underflow():
    with mp.workdps(70):
        result = references.reference_range([mp.mpc("1e-500", "1e-310"), mp.mpc(0, 1)])
        assert result["float64_no_overflow"]
        assert result["reference_component_ranges"] == [
            {"real": "below_min_subnormal", "imag": "subnormal"},
            {"real": "zero", "imag": "normal"},
        ]
        assert result["float64_below_min_subnormal_components"] == 1
        assert result["float64_subnormal_components"] == 1
        assert result["float64_rounds_to_zero_components"] == 1
        assert not references.reference_range([mp.mpf("1e500")])["float64_no_overflow"]


def test_tiny_references_are_qualified_with_absolute_error_gate(monkeypatch):
    qualify = references.qualify_case
    monkeypatch.setattr(references, "reference", lambda _: [mp.mpf("1e-500")])
    monkeypatch.setattr(references, "evaluate", lambda *_: [0])
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


def test_wigner_factorial_sum_low_order_closed_forms_and_signs():
    with mp.workdps(70):
        beta = mp.mpf("0.7")
        function = references.wigner_smalld
        assert function(0, 0, 0, beta) == 1
        assert abs(function(1, -1, 0, beta) - mp.sin(beta) / mp.sqrt(2)) < mp.mpf(
            "1e-65"
        )
        assert abs(function(1, 0, -1, beta) + mp.sin(beta) / mp.sqrt(2)) < mp.mpf(
            "1e-65"
        )
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
        assert abs(references.reference(case)[0] - expected) < mp.mpf("1e-65")


def test_wigner_diagnostics_append_without_changing_original_grid():
    cases = list(references.case_grid())
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
    certify = references.certify_rotation_disagreements
    monkeypatch.setattr(
        references, "reference", lambda _: pytest.fail("unneeded HP reference")
    )
    result = certify(np.eye(2), np.eye(2), [(0, 1, -1, 0), (0, 1, 0, 0)], [0, 0, 0])
    assert result["passed"] and result["original_gate_passed"]
    assert result["upstream_matched_entries"] == 4 and result["mismatch_count"] == 0
    with pytest.raises(ValueError, match="matching square"):
        certify(np.eye(2), np.ones((2, 1)), [(0, 1, -1, 0)], [0, 0, 0])


def test_rotation_disagreement_checks_particle_degree_polarization_and_nonfinite():
    certify = references.certify_rotation_disagreements
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


def test_rotation_disagreement_adjudicates_actual_l24_production_entries():
    certify = references.certify_rotation_disagreements
    angles = [0.2, 0.7, -0.3]
    basis = [(0, 24, m, 1) for m in range(-24, 25)]
    native = references.rotation_block("treams-rs", 24, tuple(angles), 1)
    upstream = references.rotation_block("treams", 24, tuple(angles), 1)
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


COMPONENTS = st.floats(-1e3, 1e3, allow_subnormal=False)


@settings(max_examples=60)
@given(
    actual=st.lists(st.tuples(COMPONENTS, COMPONENTS), min_size=1, max_size=5),
    exponents=st.lists(st.integers(-400, 400), min_size=5, max_size=5),
    relative=st.sampled_from([0, 1e-12, 1e-10, 3e-11, 1e-3]),
    tolerances=st.sampled_from([(2e-13, 2e-11), (2e-13, 1e-10)]),
)
# The exact error, 1e-145 to 54 bits, lies halfway between two floats. Rounded
# directly it gives 1.0000000000000001e-145, but its 60-digit string reads as 1e-145.
@example(
    actual=[(2.942727237314098e-101, 0.0)],
    exponents=[-145, 0, 0, 0, 0],
    relative=0,
    tolerances=(2e-13, 2e-11),
)
# Its error string reads as 1e-134. Parsed by mpmath at 60 digits, it lands on the
# midpoint between two floats and rounds down to 9.999999999999999e-135.
@example(
    actual=[(7.399046684673866e-90, 0.0)],
    exponents=[-134, 0, 0, 0, 0],
    relative=0,
    tolerances=(2e-13, 2e-11),
)
def test_error_summaries_are_strict_json_and_gate_on_high_precision(
    actual, exponents, relative, tolerances
):
    """References beyond float64 range keep exact strings and a strict status."""
    atol, rtol = tolerances
    actual = [complex(*pair) for pair in actual]
    with mp.workdps(60):
        expected = [
            mp.mpc(value) * (1 + mp.mpf(relative)) + mp.mpf(10) ** exponent
            for value, exponent in zip(actual, exponents, strict=False)
        ]
        result = references.errors(actual, expected, atol, rtol)
        json.dumps(result, allow_nan=False)
        exact = {
            key: None if text is None else mp.mpf(text)
            for key, text in result["errors_high_precision"].items()
        }
        assert result["status"] == (
            "passed" if exact["max_scaled_error"] <= 1 else "failed"
        )
        for key, text in result["errors_high_precision"].items():
            number = None if text is None else float(text)
            representable = (
                number is not None
                and math.isfinite(number)
                and (number != 0 or exact[key] == 0)
            )
            assert result[key] == (number if representable else None)
        # Relaxing both tolerances twofold halves every scaled error.
        relaxed = references.errors(actual, expected, 2 * atol, 2 * rtol)
        halved = mp.mpf(relaxed["errors_high_precision"]["max_scaled_error"])
        assert abs(2 * halved - exact["max_scaled_error"]) <= mp.mpf("1e-45") * max(
            1, exact["max_scaled_error"]
        )
        assert result["status"] != "passed" or relaxed["status"] == "passed"
