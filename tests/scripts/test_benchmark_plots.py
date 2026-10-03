"""Check evidence normalization; plotting never invents or pools measurements."""

import hashlib
import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from _scripts import load

pytestmark = pytest.mark.interface

plot = load("plot_benchmarks")


@pytest.fixture
def plt():
    """Matplotlib's pyplot; rendering tests skip where it is not installed."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as pyplot

    return pyplot


@pytest.fixture
def write_suite(tmp_path):
    """Write (case, raw result) pairs next to a suite manifest; return its path."""

    def write(cases, suite_id="test"):
        for case, raw in cases:
            if raw is not None:
                text = raw if isinstance(raw, str) else json.dumps(raw)
                (tmp_path / case["result"]).write_text(text)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"suite_id": suite_id, "cases": [case for case, _ in cases]})
        )
        return manifest

    return write


def test_paired_samples_replace_unpaired_medians_and_keep_recording():
    data = {
        "results": [
            {"backend": "treams", "median_seconds": 999, "samples_seconds": [999]},
            {
                "backend": "rust",
                "median_seconds": 999,
                "samples_seconds": [999],
                "peak_rss_mib": 42,
                "baseline_rss_mib": 43,
                "forward_records_adjoint": True,
            },
        ],
        "timing_comparison": {
            "pairs_seconds": [{"treams": 4, "rust": 2}, {"treams": 8, "rust": 4}]
        },
    }
    upstream, native = plot.read_measurements(data, "cluster")
    assert upstream["median"] == 6 and native["median"] == 3
    assert native["seconds"] == [2, 4]
    assert native["growth_rss_mib"] == 0
    assert native["forward_records_adjoint"] is True
    assert upstream["forward_records_adjoint"] is False


def test_failed_missing_and_different_builds_are_preserved(write_suite):
    written = []
    for i in range(3):
        raw = {
            "results": [
                {"backend": "rust", "samples_seconds": [1], "native_sha256": str(i)}
            ]
        }
        if i == 2:
            raw["validation"] = {"passed": False}
        case = {"id": str(i), "kind": "cluster", "status": "passed"}
        written.append(({**case, "result": f"{i}.json"}, raw))
    missing = {"id": "missing", "kind": "cluster", "result": "absent.json"}
    written.append(({**missing, "status": "passed"}, None))
    written.append(({"id": "timeout", "kind": "cluster", "status": "timeout"}, None))
    cases = plot.load_manifest(write_suite(written))["cases"]
    assert len(cases) == 5
    assert cases[0]["fingerprint"] != cases[1]["fingerprint"]
    assert [c["status"] for c in cases] == [
        "passed",
        "passed",
        "accuracy_failed",
        "missing",
        "timeout",
    ]
    assert plot.cpu_pair(cases[2]) is None
    assert cases[3]["measurements"] == []


def test_gradient_total_is_not_sum_of_record_and_reverse_medians():
    data = {
        "measurements": [
            {
                "backend": "rust",
                "phase": "adjoint",
                "elapsed_seconds": [4, 5, 6],
                "record_seconds": [1, 2, 3],
                "reverse_seconds": [1, 1, 1],
            },
            {
                "backend": "treams",
                "phase": "finite_difference",
                "elapsed_seconds": [40],
            },
            {"backend": "treams", "phase": "linear_adjoint", "elapsed_seconds": [2]},
        ]
    }
    rows = plot.read_measurements(data, "gradient")
    assert [(r["phase"], r["median"]) for r in rows] == [
        ("adjoint", 5),
        ("record", 2),
        ("reverse", 1),
        ("finite_difference", 40),
        ("linear_adjoint", 2),
    ]


@pytest.mark.parametrize(
    "values", [[1, float("nan")], [float("inf")], [0], [-1], 0.0, "x", (1.0, 2.0)]
)
def test_samples_reject_nonfinite_nonpositive_or_unlisted_values(values):
    with pytest.raises(ValueError, match="finite positive"):
        plot.samples(values)


def test_samples_accept_missing_scalar_and_listed_seconds():
    assert plot.samples(None) == []
    assert plot.samples(2) == [2.0]
    assert plot.samples([1, 2.5]) == [1.0, 2.5]


def test_speedup_uses_median_of_paired_ratios():
    left = {"median": 100, "seconds": [2, 100, 100], "paired": True}
    right = {"median": 2, "seconds": [1, 100, 2], "paired": True}
    assert plot.speedup(left, right) == 2
    assert plot.ratio_interval(left, right, paired=True)[0] == 2


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    "values",
    [
        [2e-9],
        [0.0, 1e-14, 1e-8],
        # Finite absolute-error ranges of the recorded Linux reference run.
        [0.0, 4.9963674950853214e-306, 3.715707027309196e281],
        [0.0, 7e-323, 5.424334802529538e226],
    ],
)
def test_error_axis_keeps_real_zeros_and_spaces_ticks_and_markers(axis, values, plt):
    fig, ax = plt.subplots()
    coordinates = (values, range(len(values)))
    (line,) = ax.plot(*(coordinates if axis == "x" else coordinates[::-1]))
    plot.error_axis(ax, values, axis)
    np.testing.assert_array_equal(getattr(line, f"get_{axis}data")(), values)
    coordinate = getattr(ax, f"{axis}axis")
    transform = coordinate.get_transform()
    assert np.isfinite(transform.transform(values)).all()
    lower, upper = transform.transform(getattr(ax, f"get_{axis}lim")())
    assert lower == 0
    assert transform.transform(max(values)) < upper
    assert np.isfinite(getattr(ax, f"get_{axis}lim")()).all()
    ticks = coordinate.get_majorticklocs()
    assert 0 in ticks
    assert all(transform.transform(t) / upper >= 0.08 for t in ticks if t > 0)
    with np.errstate(over="raise", invalid="raise"):
        fig.canvas.draw()
    plt.close(fig)


def test_illumination_accuracy_aggregates_certificates_and_excludes_self_comparison():
    case = {
        "id": "illumination-n128",
        "kind": "illumination",
        "suite": "test",
        "fingerprint": "build",
        "series": "illumination-particles",
        "x": {"name": "Particles", "value": 128},
        "raw": {
            "case": {"particles": 128, "columns": 1, "rtol": 1e-10},
            "oracle_relative_error": 1e-12,
            "comparison_reference": "native recorded full-T cluster API",
            "comparison_relative_errors": {
                "full/adjoint/value": 0,
                "full/adjoint/radii": 0,
                "full/forward/value": 0,
                "matrix-free/forward/value": 1e-9,
                "matrix-free/adjoint/value": 2e-9,
                "matrix-free/adjoint/radii": 2e-8,
                "treams-full/forward/value": 1e-12,
            },
            "measurements": [
                {
                    "backend": "full",
                    "phase": "adjoint",
                    "numerical": {"scale_ward_absolute": -2e-10},
                },
                {
                    "backend": "matrix-free",
                    "phase": "forward",
                    "forward_convergence": [[4, 2e-12, 0.1]],
                },
                {
                    "backend": "matrix-free",
                    "phase": "adjoint",
                    "forward_convergence": [[4, 3e-12, 0.1]],
                    "adjoint_convergence": [[5, 2e-11, 0.1]],
                    "numerical": {"radius_direction_relative_error": 1e-8},
                },
            ],
        },
    }
    rows = plot.accuracy_records([case])
    by_metric = {row["metric"]: row for row in rows}
    assert len(rows) == 7
    assert (
        by_metric["illumination_upstream_forward_relative_l2_error"]["reference_kind"]
        == "upstream"
    )
    comparison = by_metric["illumination_forward_relative_l2_error"]
    assert comparison["error"] == 2e-9
    assert comparison["reference_kind"] == "native_full_t"
    assert comparison["compared_phases"] == ["forward", "adjoint"]
    forward = by_metric["illumination_forward_true_residual_relative"]
    assert forward["error"] == pytest.approx(3e-11)
    assert forward["certificate_count"] == 2
    assert forward["independent_input_count"] == 1
    assert forward["x"]["value"] == 128
    assert (
        by_metric["illumination_adjoint_true_residual_relative"]["status"] == "failed"
    )
    ward = by_metric["scale_ward_absolute"]
    assert ward["error"] == 2e-10 and ward["signed_value"] == -2e-10
    assert ward["status"] == "diagnostic" and ward["tolerance"] is None
    assert "1 requested-illumination input definitions" in plot.accuracy_html([case])


def test_illumination_without_oracle_retains_missing_certificate_status():
    case = {
        "id": "large",
        "raw": {
            "oracle_relative_error": None,
            "upstream_skipped_reason": "upstream dimension cap",
            "comparison_relative_errors": {"matrix-free/adjoint/value": 0},
            "measurements": [
                {
                    "backend": "matrix-free",
                    "phase": "forward",
                    "forward_convergence": None,
                }
            ],
        },
    }
    rows = plot.illumination_accuracy_records(case)
    assert len(rows) == 2
    assert rows[0]["status"] == "not_measured" and rows[0]["error"] is None
    assert rows[1]["status"] == "not_recorded" and rows[1]["error"] is None


def test_shared_suite_id_gets_distinct_os_labels(tmp_path):
    names = []
    for os_name in ["Darwin", "Linux"]:
        path = tmp_path / f"{os_name}.json"
        path.write_text(
            json.dumps(
                {
                    "suite_id": "shared",
                    "phase": "broad",
                    "environment": {"os": os_name},
                    "cases": [],
                }
            )
        )
        names.append(plot.load_manifest(path)["name"])
    assert names == ["shared · macOS · broad", "shared · Linux · broad"]


def test_concise_figure_environment_keeps_suite_identity_separate():
    suite = {
        "name": "technical-suite-id",
        "metadata": {
            "environment": {
                "os": "Linux",
                "cpu_model": "AMD Ryzen 9 9950X 16-Core Processor",
            },
            "phase": "broad",
        },
        "cases": [{"group": "broad"}],
    }
    assert plot.environment_label(suite) == "Ryzen 9 9950X · Linux"
    assert plot.suite_caption(suite) == "Ryzen 9 9950X · Linux · Feature coverage"
    assert suite["name"] == "technical-suite-id"
    assert plot.family_for("ebcm-forward") == "EBCM"
    suite["metadata"]["environment"]["cpu_model"] = None
    assert plot.environment_label(suite) == "Linux"


def test_controller_recovery_archive_checks_hashes(tmp_path):
    folder = tmp_path / "mac" / "scaling"
    folder.mkdir(parents=True)
    previous = folder / "manifest-before-controller-recovery.json"
    previous.write_text('{"status": "interrupted"}\n')
    controller = folder.parent / "controller-recovery" / "runner-before.py.txt"
    controller.parent.mkdir()
    controller.write_text("# archived controller\n")
    recovery = {
        "previous_manifest": previous.name,
        "previous_manifest_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "previous_runner_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
    }
    suite = {
        "name": "mac-scaling",
        "path": folder / "manifest.json",
        "metadata": {"controller_recovery": recovery},
    }
    output = tmp_path / "report"
    archived = plot.archive_controller_recovery(suite, output)
    assert (
        output / archived["recovery-1-previous_manifest"]["path"]
    ).read_bytes() == previous.read_bytes()
    assert (
        output / archived["recovery-1-previous_controller"]["path"]
    ).read_bytes() == controller.read_bytes()
    recovery["previous_runner_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum differs"):
        plot.archive_controller_recovery(suite, output)


def test_two_controller_recoveries_preserve_each_manifest_and_source(tmp_path):
    folder = tmp_path / "mac" / "scaling"
    folder.mkdir(parents=True)
    controller_folder = folder.parent / "controller-recovery"
    controller_folder.mkdir()
    recoveries = []
    originals = {}
    for index in (1, 2):
        previous = folder / f"manifest-before-recovery-{index}.json"
        previous.write_text(json.dumps({"recovery": index}))
        source = controller_folder / f"runner-before-{index}.py.txt"
        source.write_text(f"# controller {index}\n")
        recovery = {
            "previous_manifest": previous.name,
            "previous_manifest_sha256": hashlib.sha256(
                previous.read_bytes()
            ).hexdigest(),
            "previous_runner_source": "../controller-recovery/" + source.name,
            "previous_runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        recoveries.append(recovery)
        originals[f"recovery-{index}-previous_manifest"] = previous.read_bytes()
        originals[f"recovery-{index}-previous_controller"] = source.read_bytes()
    suite = {
        "name": "mac-scaling",
        "path": folder / "manifest.json",
        "metadata": {"controller_recoveries": recoveries},
    }
    output = tmp_path / "report"
    archived = plot.archive_controller_recovery(suite, output)
    assert archived.keys() == originals.keys()
    for key, expected in originals.items():
        assert (output / archived[key]["path"]).read_bytes() == expected
        assert archived[key]["sha256"] == hashlib.sha256(expected).hexdigest()
    recoveries[1]["previous_runner_sha256"] = "wrong-second-hash"
    with pytest.raises(ValueError, match="runner-before-2"):
        plot.archive_controller_recovery(suite, output)


def test_accuracy_availability_keeps_failed_numeric_errors_and_explains_omissions():
    rows = [
        {"backend": "treams-rs", "error": 0, "status": "passed"},
        {"backend": "treams-rs", "error": 3, "status": "failed"},
        {
            "backend": "treams-rs",
            "error": None,
            "conditioning": {"float64_no_overflow": False},
        },
        {"backend": "treams", "error": None, "status": "error"},
        {"backend": "treams", "error": None, "error_high_precision": "1e400"},
    ]
    note = plot.accuracy_availability(rows)
    assert "treams-rs: 2/3 finite (omitted 1 reference overflow)" in note
    assert (
        "treams: 0/2 finite (omitted 1 evaluation error, 1 high-precision-only residual)"
        in note
    )


def test_gradient_package_identity_and_invalid_output(write_suite):
    row = plot.read_measurements(
        {
            "measurements": [
                {
                    "backend": "rust",
                    "phase": "adjoint",
                    "elapsed_seconds": [1],
                    "package_sha256": "python-package",
                }
            ]
        },
        "gradient",
    )[0]
    assert row["python_source_sha256"] == "python-package"
    bad = {"id": "bad", "kind": "cluster", "result": "bad.json"}
    manifest = write_suite(
        [({**bad, "status": "invalid_output"}, "incomplete output")], "bad-output"
    )
    case = plot.load_manifest(manifest)["cases"][0]
    assert case["status"] == "invalid_output"
    assert case["measurements"] == []


def test_accuracy_preserves_zero_failed_and_unmeasured_observations():
    observations = [
        {"id": "zero", "error": 0.0, "status": "passed"},
        {"id": "wrong", "error": 2.0, "status": "failed"},
        {"id": "overflow", "error": None, "status": "error"},
    ]
    case = {
        "id": "reference",
        "kind": "accuracy",
        "suite": "test",
        "fingerprint": "build",
        "raw": {"observations": observations},
    }
    rows = plot.accuracy_records([case])
    assert [r["error"] for r in rows] == [0.0, 2.0, None]
    assert [r["status"] for r in rows] == ["passed", "failed", "error"]
    assert plot.read_measurements(case["raw"], "accuracy") == []


def test_slowdown_counts_distinguish_observed_from_bounded():
    cases = []
    for timings in [[0.4, 0.5, 0.6], [0.4, 0.8, 1.8], [0.1]]:
        rows = plot.read_measurements(
            {
                "results": [
                    {"backend": "treams", "samples_seconds": timings},
                    {"backend": "rust", "samples_seconds": [1] * len(timings)},
                ]
            },
            "cluster",
        )
        cases.append({"kind": "cluster", "status": "success", "measurements": rows})
    assert plot.slowdown_summary(cases) == (3, 1)


def test_accuracy_ledger_is_separate_and_script_data_is_escaped(tmp_path):
    case = {
        "id": "case",
        "kind": "accuracy",
        "suite": "test",
        "fingerprint": "build",
        "raw": {
            "observations": [
                {"id": "observation</script>", "error": 0, "status": "passed"}
            ]
        },
    }
    summary = plot.accuracy_html([case])
    assert "accuracy-test.html" in summary
    assert "observation</script>" not in summary
    plot.write_accuracy_pages([{"name": "test", "cases": [case]}], tmp_path)
    page = (tmp_path / "accuracy-test.html").read_text()
    assert "observation</script>" not in page
    assert "observation\\u003c/script>" in page
    assert "slice(page*100,(page+1)*100)" in page
    case.update(status="success", measurements=[], family="finite", tier="small")
    suite = {"name": "test", "cases": [case], "metadata": {}, "historical": False}
    plot.write_html([suite], [[]], tmp_path, [])
    index = (tmp_path / "index.html").read_text()
    assert "per-input, per-backend acceptance" in index
    assert "Among 0 matched CPU cases" not in index


def test_certified_native_accuracy_keeps_original_upstream_failure(
    tmp_path, write_suite
):
    original = {
        "id": "original-upstream-comparison",
        "backend": "treams-rs",
        "reference_kind": "upstream",
        "metric": "max_scaled_error",
        "error": 20.0,
        "tolerance": 1.0,
        "status": "failed",
    }
    certificate = {
        "id": "independent-certificate",
        "backend": "treams-rs",
        "reference_kind": "independently_balanced_upstream_system",
        "metric": "certified_reference_max_scaled_error",
        "error": 0.01,
        "tolerance": 1.0,
        "status": "passed",
        "certificate": {"passed": True, "encoded_system_sha256": "system-hash"},
    }
    raw = {
        "passed": False,
        "native_reference_passed": True,
        "observations": [original, certificate],
    }
    case = {"id": "conditioning", "kind": "accuracy", "status": "success"}
    manifest = write_suite([({**case, "result": "result.json"}, raw)])
    suite = plot.load_manifest(manifest)
    (case,) = suite["cases"]
    case["raw_link"] = "raw/conditioning.json"
    assert case["status"] == "accuracy_failed"
    rows = plot.accuracy_records([case])
    assert [(r["status"], r["error"]) for r in rows] == [
        ("failed", 20.0),
        ("passed", 0.01),
    ]
    assert rows[1]["certificate"] == certificate["certificate"]
    summary = plot.accuracy_html([case])
    assert "Native response-matrix qualification passed" in summary
    assert "Original upstream comparison failures remain visible" in summary
    assert "raw/conditioning.json" in summary
    plot.write_html([suite], [[]], tmp_path, [])
    index = (tmp_path / "index.html").read_text()
    assert "accuracy_failed" in index and "native_reference_passed" in index
    assert "Native response-matrix qualification passed" in index
    assert (
        "Original upstream disagreements remain shown"
        in plot.accuracy_reference_note("upstream", rows, {"conditioning"})
    )
    assert plot.accuracy_reference_note("upstream", rows, {"other-case"}) == ""
    note = plot.accuracy_reference_note(
        "independently_balanced_upstream_system", rows, {"conditioning"}
    )
    assert "certified reference uncertainty" in note
    assert "not physical truncation accuracy" in note

    observed = case["raw"]["observations"][1]
    observed["certificate"]["passed"] = False
    assert plot.native_qualification_note(case) == ""
    observed["certificate"]["passed"] = True
    observed["status"] = "failed"
    assert plot.native_qualification_note(case) == ""
    case["raw"]["observations"] = [original]
    assert plot.native_qualification_note(case) == ""


def test_timing_independent_reference_proof_remains_in_case_ledger(tmp_path):
    validation = {
        "accuracy_check": "passed",
        "reference_kind": "upstream_with_independent_disagreement_checks",
        "independent_reference": {
            "passed": True,
            "original_disagreement": {"max_scaled_error": 7.0},
            "certificate": {"encoded_system_sha256": "certified-system"},
        },
    }
    case = {
        "id": "cluster",
        "kind": "cluster",
        "suite": "test",
        "fingerprint": "build",
        "status": "success",
        "measurements": [],
        "family": "cluster",
        "tier": "large",
        "raw_link": "raw/cluster.json",
        "raw": {"validation": validation},
    }
    suite = {"name": "test", "cases": [case], "metadata": {}, "historical": False}
    plot.write_html([suite], [[]], tmp_path, [])
    index = (tmp_path / "index.html").read_text()
    assert "Independent native qualification passed" in index
    assert "original_disagreement" in index and "certified-system" in index
    assert "raw/cluster.json" in index
    assert case["status"] == "success" and case["raw"]["validation"] == validation
    validation["independent_reference"]["passed"] = False
    assert plot.native_qualification_note(case) == ""


def test_gradient_gates_and_trial_diagnostics_are_distinct():
    case = {
        "id": "gradient",
        "kind": "gradient",
        "suite": "test",
        "family": "cluster",
        "fingerprint": "build",
        "raw": {
            "case": {"gradient_rtol": 1e-5},
            "validation": {
                "passed": True,
                "selected_fd_step": 1e-6,
                "gradient_relative_error": 1e-7,
                "fd_step_sweep": [{"step": 0.01, "gradient_relative_error": 0.1}],
                "directional_checks": [{"step": 0.01, "relative_error": 0.1}],
                "gradient_block_relative_errors": {"radii": 0.1},
                "invariants": {"translation_relative_residual": 1e-10},
            },
        },
    }
    rows = {row["metric"]: row for row in plot.accuracy_records([case])}
    assert rows["gradient_selected_relative_l2_error"]["status"] == "passed"
    assert rows["gradient_relative_l2_error"]["status"] == "diagnostic"
    assert rows["directional_relative_error"]["status"] == "diagnostic"
    assert rows["gradient_block_relative_l2_error"]["status"] == "failed"
    assert rows["translation_relative_residual"]["tolerance"] == 3e-8
    assert (
        rows["translation_relative_residual"]["reference_kind"] == "physical_invariant"
    )


def test_raw_and_normalized_error_classification():
    assert plot.raw_error_metric("passivity_nonnegative_violation")
    assert plot.raw_error_metric("scattering_exceeds_extinction")
    assert plot.raw_error_metric("max_abs_error")
    assert not plot.raw_error_metric("layer_split_scaled_absolute")
    assert not plot.raw_error_metric("divergence_scaled_absolute")
    assert not plot.raw_error_metric("ewald_split_scaled_absolute")
    assert not plot.raw_error_metric("max_scaled_error")


def test_finite_difference_pages_keep_every_case_separate_with_six_panel_limit():
    curves = {}
    expected_series = []
    for family, metric, count in [
        ("cluster", "gradient_relative_l2_error", 8),
        ("cluster", "directional_relative_error", 6),
        ("sphere", "gradient_relative_l2_error", 1),
    ]:
        for index in range(count):
            series = f"{family}/{metric}/{index}"
            rows = [
                {
                    "family": family,
                    "scope": "Full-coordinate step sweep",
                    "case_id": series,
                    "error": 0,
                    "x": {"value": 1e-6},
                }
            ]
            curves[("finite_difference", metric, series, "Finite-difference step")] = (
                rows
            )
            expected_series.append(series)
    curves[("analytic", "relative_error", "other", "Order")] = [{"family": "cluster"}]
    pages = plot.finite_difference_pages(curves)
    assert len(pages) == 4
    assert all(1 <= len(panels) <= 6 for *_, panels in pages)
    observed = []
    for family, metric, x_name, _, panels in pages:
        for series, rows in panels:
            assert rows is curves[("finite_difference", metric, series, x_name)]
            assert rows[0]["family"] == family
            assert rows[0]["scope"] == "Full-coordinate step sweep"
            observed.append(series)
    assert sorted(observed) == sorted(expected_series)


def test_reference_summary_counts_inputs_and_separates_float64_ranges():
    definitions = [
        {
            "reference_stable": True,
            "float64_no_overflow": True,
            "float64_subnormal_components": 1,
            "float64_below_min_subnormal_components": 1,
            "float64_rounds_to_zero_components": 1,
            "backends": {
                "treams-rs": {"status": "passed"},
                "treams": {"status": "failed"},
            },
        },
        {
            "reference_stable": True,
            "float64_no_overflow": False,
            "backends": {
                "treams-rs": {"status": "error"},
                "treams": {"status": "error"},
            },
        },
        {"reference_stable": False, "backends": {}},
    ]
    observations = [{"id": str(i), "status": "error"} for i in range(18)]
    case = {
        "id": "precision",
        "kind": "accuracy",
        "suite": "test",
        "fingerprint": "build",
        "raw": {"cases": definitions, "observations": observations},
    }
    summary = plot.accuracy_html([case])
    assert "3 high-precision input definitions" in summary
    assert "18 metric observations" in summary
    assert "1 stable references without float64 overflow" in summary
    assert "1 references above float64 range" in summary
    assert "1 unstable or unavailable references" in summary
    assert "treams-rs: passed 1, failed 0, error 0" in summary
    assert "treams: passed 0, failed 1, error 0" in summary
    assert "1 inputs with subnormal components" in summary
    assert "1 inputs with components below the smallest subnormal" in summary
    assert "1 inputs with components rounding to zero" in summary
    assert case["raw"]["observations"] == observations


def test_illumination_caps_do_not_split_one_build(write_suite):
    written = []
    native = {
        "native_sha256": "native",
        "python_source_sha256": "python",
        "benchmark_sha256": "harness",
    }
    for index, backends in enumerate(
        [["full", "selected", "matrix-free", "treams-full"], ["matrix-free"]]
    ):
        rows = [
            {
                "backend": backend,
                "phase": "forward",
                "fresh_seconds": [1],
                **(
                    {"treams_package_sha256": "upstream", "benchmark_sha256": "harness"}
                    if backend.startswith("treams-")
                    else native
                ),
            }
            for backend in backends
        ]
        case = {"id": str(index), "kind": "illumination", "status": "success"}
        written.append(({**case, "result": f"{index}.json"}, {"measurements": rows}))
    cases = plot.load_manifest(write_suite(written, "illumination"))["cases"]
    assert cases[0]["fingerprint"] == cases[1]["fingerprint"]
    assert len(cases[0]["measurements"]) == 4
    assert len(cases[1]["measurements"]) == 1


def test_scaling_marks_planned_failure_without_a_numeric_y(monkeypatch):
    import sys
    from types import ModuleType

    class Axis:
        def __init__(self):
            self.calls = []
            self.transform = object()

        def get_xaxis_transform(self):
            return self.transform

        def __getattr__(self, name):
            def record(*args, **kwargs):
                self.calls.append((name, args, kwargs))

            return record

    axes = [Axis(), Axis(), Axis()]
    plt = ModuleType("matplotlib.pyplot")
    plt.subplots = lambda *args, **kwargs: ("figure", axes)
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = plt
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    lines = []
    monkeypatch.setattr(
        plot,
        "timing_line",
        lambda ax, rows, *args: lines.append([x for x, _ in rows]),
    )
    monkeypatch.setattr(plot, "ratio_interval", lambda *args: (2, 2, 2))
    common = {
        "kind": "cluster",
        "series": "cluster-order",
        "group": "scaling",
        "parameters": {"threads": 4},
    }
    measured = []
    for x in (1, 3):
        rows = plot.read_measurements(
            {
                "results": [
                    {"backend": backend, "samples_seconds": [1], "threads": 4}
                    for backend in ("treams", "rust")
                ]
            },
            "cluster",
        )
        measured.append(
            {
                **common,
                "id": str(x),
                "x": {"name": "Order", "value": x},
                "status": "success",
                "measurements": rows,
            }
        )
    failed = {
        **common,
        "id": "failed",
        "x": {"name": "Order", "value": 2},
        "status": "error",
        "measurements": [],
        "error": "AssertionError: Not equal to tolerance rtol=2e-9, atol=1e-12",
    }
    assert plot.scaling_segments([measured[0], failed, measured[1]]) == [
        [measured[0]],
        [measured[1]],
    ]
    saved = []
    plot.plot_scaling(measured, lambda *args: saved.append(args), [*measured, failed])
    assert lines == [[1], [1], [3], [3]]
    for ax in axes:
        markers = [
            (args, kwargs) for name, args, kwargs in ax.calls if name == "scatter"
        ]
        assert len(markers) == 1
        args, kwargs = markers[0]
        assert args[0] == [2]
        assert kwargs["transform"] is ax.transform
        assert kwargs["label"] == "Reference-check failure (no measurement)"
    assert "Reference-check failure: Order=2" in saved[0][-1]
    assert plot.scaling_failure_label({"status": "timeout"}) == "Timeout"
    assert plot.scaling_failure_label({"status": "memory_limit"}) == "Memory limit"


@pytest.mark.parametrize(
    ("values", "gate"),
    [([0.0, 1e-280, 1e100], 1e200), ([0.0, 1e-280, 1e280], 1.0)],
)
def test_error_axis_includes_recorded_gate_without_changing_measurements(
    values, gate, plt
):
    fig, axes = plt.subplots(1, 2)
    try:
        for axis, ax in zip(("x", "y"), axes, strict=True):
            coordinates = (values, range(len(values)))
            (line,) = ax.plot(*(coordinates if axis == "x" else coordinates[::-1]))
            plot.error_axis(ax, values, axis=axis, gate=gate)
            np.testing.assert_array_equal(getattr(line, f"get_{axis}data")(), values)
            lower, upper = getattr(ax, f"get_{axis}lim")()
            assert lower == 0 and upper >= gate
            ticks = getattr(ax, f"{axis}axis").get_majorticklocs()
            assert 0 in ticks and gate in ticks and len(ticks) <= 9
            assert max(ticks) >= max(values) / 100
        with np.errstate(over="raise", invalid="raise"):
            fig.canvas.draw()
    finally:
        plt.close(fig)


def test_accuracy_curve_draws_only_a_complete_common_positive_tolerance(plt):
    tolerances = [
        (1e-2, 1e-2),
        (1e-2, None),
        (1e-2, 2e-2),
        (None, None),
        (0.0, 0.0),
        (float("nan"), float("nan")),
    ]
    fig, axes = plt.subplots(2, 3)
    try:
        for index, (ax, thresholds) in enumerate(
            zip(axes.flat, tolerances, strict=True)
        ):
            rows = [
                {
                    "backend": "treams-rs",
                    "x": {"value": x},
                    "error": error,
                    "tolerance": threshold,
                }
                for x, error, threshold in zip(
                    (1, 2), (0.0, 1e-14), thresholds, strict=True
                )
            ]
            plot.draw_accuracy_curve(
                ax, rows, "iterative_residual", "true_residual_relative", "Order"
            )
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2])
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.0, 1e-14])
            assert len(ax.lines) == (2 if index == 0 else 1)
            if index == 0:
                np.testing.assert_array_equal(ax.lines[1].get_ydata(), [1e-2, 1e-2])
                assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] >= 1e-2
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_high_order_plot_uses_observed_ticks_and_nonnegative_bounds(plt):
    grids = {
        "Degree": [23, 24, 25],
        "Order": [0, 1, 2, 4, 8, 16, 24, 32, 64, 96, 128, 256],
    }
    observations = []
    for coordinate, values in grids.items():
        for index, value in enumerate(values):
            error = [0.0, 0.5, 2.0][index % 3]
            observations.append(
                {
                    "family": "wigner_rotation" if coordinate == "Degree" else "bessel",
                    "backend": "treams-rs",
                    "reference_kind": "high_precision",
                    "metric": "max_scaled_error",
                    "parameters": {coordinate.lower(): value},
                    "error": error,
                    "tolerance": 1.0,
                    "status": "failed" if error > 1 else "passed",
                }
            )
    case = {
        "kind": "accuracy",
        "id": "observed-orders",
        "suite": "test",
        "fingerprint": "test-build",
        "raw": {"observations": observations},
    }
    captured = set()

    def save(fig, name, title, caption):
        try:
            if not name.startswith("high-order-"):
                return
            (ax,) = fig.axes
            coordinate = ax.get_xlabel()
            observed = grids[coordinate]
            captured.add(coordinate)
            lower, upper = ax.get_xlim()
            assert 0 <= lower <= min(observed) and upper >= max(observed)
            ticks = ax.get_xticks()
            assert 1 <= len(ticks) <= 9 and set(ticks) <= set(observed)
            if coordinate == "Degree":
                np.testing.assert_array_equal(ticks, [23, 24, 25])
            offsets = ax.collections[0].get_offsets()
            np.testing.assert_array_equal(offsets[:, 0], observed)
            np.testing.assert_array_equal(
                offsets[:, 1], [[0.0, 0.5, 2.0][i % 3] for i in range(len(observed))]
            )
            assert ax.get_ylim()[0] == 0
            fig.canvas.draw()
        finally:
            plt.close(fig)

    plot.plot_accuracy([case], save)
    assert captured == set(grids)


@pytest.mark.parametrize(
    "coordinates",
    [
        [4, 16, 64, 128, 256, 512, 1024, 2048, 4096],
        [2, 8, 32, 128, 512, 1024, 2048, 4096],
    ],
)
def test_illumination_ticks_keep_endpoints_without_removing_measurements(
    coordinates, plt
):
    cases = [
        {
            "kind": "illumination",
            "status": "success",
            "series": "count",
            "x": {"name": "Particles", "value": x},
            "parameters": {"particles": x, "lmax": 1, "strength": "weak"},
            "measurements": [
                {
                    "backend": "selected",
                    "phase": phase,
                    "median": x / 10,
                    "seconds": [],
                    "peak_rss_mib": x,
                }
                for phase in ["fresh", "reuse"]
            ],
        }
        for x in coordinates
    ]

    def save(fig, name, title, caption):
        try:
            assert title == (
                "Scattering solve · varying particles · multipole cutoff 1, weak coupling"
            )
            for ax in fig.axes:
                ticks = ax.get_xticks()
                assert len(ticks) <= 6
                assert ticks[0] == coordinates[0] and ticks[-1] == coordinates[-1]
                assert set(ticks) <= set(coordinates)
                assert (
                    np.log(ticks[-1] / ticks[-2])
                    >= np.log(coordinates[-1] / coordinates[0]) / 6
                )
                np.testing.assert_array_equal(ax.lines[0].get_xdata(), coordinates)
            fig.canvas.draw()
        finally:
            plt.close(fig)

    plot.plot_illumination(cases, save)


@given(
    outcomes=st.lists(
        st.tuples(st.integers(0, 50), st.booleans()),
        unique_by=lambda outcome: outcome[0],
        max_size=12,
    ),
    shuffle=st.randoms(use_true_random=False),
)
def test_scaling_segments_split_exactly_at_unmeasured_positions(outcomes, shuffle):
    """Curves never bridge a planned size without a measured native/upstream pair."""
    pair = [{"backend": backend, "phase": "forward"} for backend in ("treams", "rust")]
    cases = [
        {
            "kind": "cluster",
            "status": "success" if measured else "error",
            "measurements": pair if measured else [],
            "x": {"value": x},
        }
        for x, measured in outcomes
    ]
    shuffle.shuffle(cases)
    ordered = sorted(cases, key=lambda case: case["x"]["value"])
    segments = plot.scaling_segments(cases)
    measured = [case for case in ordered if case["status"] == "success"]
    assert [case for segment in segments for case in segment] == measured
    positions = [case["x"]["value"] for case in ordered]
    for segment in segments:
        first = positions.index(segment[0]["x"]["value"])
        # Each segment is a maximal run of consecutive measured positions.
        assert ordered[first : first + len(segment)] == segment
        assert first == 0 or ordered[first - 1]["status"] == "error"
        end = first + len(segment)
        assert end == len(ordered) or ordered[end]["status"] == "error"
