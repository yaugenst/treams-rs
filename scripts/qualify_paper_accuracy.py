# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1", "scipy>=1.16,<1.17", "treams==0.4.7", "threadpoolctl>=3.7"]
# ///
"""Record full paper spectra, source comparisons and physical residuals.

Physics comes from qualify_papers.py. Unlike that fail-fast test script, this
collector preserves every computed spectrum and failed comparison. The historical
normalization option only transforms saved data and never claims a fresh run.
"""

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

from _harness import file_sha256, pinned_threads

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks/papers"


def references():
    return {
        name: json.loads((FIXTURES / name).read_text())
        for name in (
            "ebeam-author-reference.json",
            "ebeam-notebook-execution.json",
            "source-provenance.json",
        )
    }


def normalize_curves(raw, fixtures):
    """Give recorded and newly collected curves the same fields for plotting."""
    curves = []
    author = fixtures["ebeam-author-reference.json"]
    notebooks = fixtures["ebeam-notebook-execution.json"]
    for name, curve in raw.get("ebeam", {}).get("curves", {}).items():
        base = {
            "family": f"ebeam-{name}",
            "paper": author["paper"],
            "quantity": ["Cathodoluminescence", "Electron energy loss"],
            "unit": curve["unit"],
            "x_label": "Photon energy (eV)",
            "parameters": {
                "order": curve["order"],
                "radius_nm": 50,
                "impact_nm": 60,
                "beta": 0.7,
            },
        }
        original = author["cases"][name]
        curves.append(
            {
                **base,
                **curve,
                "id": f"ebeam-{name}-author-table",
                "references": [
                    {
                        "reference_kind": "author_data",
                        "label": "Author stored regression data",
                        "values": [
                            list(row)
                            for row in zip(
                                original["cl"], original["eels"], strict=True
                            )
                        ],
                        "source_url": original["source_url"],
                        "source_sha256": original["source_sha256"],
                        "rtol": 0.0,
                        "atol": 1e-8,
                    }
                ],
            }
        )
        fine = curve.get("author_notebook_grid")
        if fine is not None:
            corrected = name == "cylinder"
            notebook = notebooks["cases"][name]
            curves.append(
                {
                    **base,
                    **fine,
                    "id": f"ebeam-{name}-author-notebook",
                    "references": [
                        {
                            "reference_kind": "corrected_author_notebook"
                            if corrected
                            else "author_notebook",
                            "label": "Corrected author notebook execution"
                            if corrected
                            else "Original author notebook execution",
                            "values": notebook["repaired" if corrected else "original"][
                                "values_cl_eels"
                            ],
                            "source_url": f"https://github.com/tfp-photonics/treams_ebeam/blob/{notebooks['source_commit']}/docs/{name}.ipynb",
                            "source_sha256": notebook["notebook_sha256"],
                            "rtol": 2e-9,
                            "atol": 2e-11,
                        }
                    ],
                }
            )
        # The separately flattened notebook is authoritative in this format.
        curves[-(2 if fine is not None else 1)].pop("author_notebook_grid", None)
    labels = {
        "sphere": (
            "Scattering efficiency",
            "Extinction efficiency",
            "Dipole scattering",
            "Dipole extinction",
        ),
        "chiral_slab": (
            "Transmission +",
            "Reflection +",
            "Transmission -",
            "Reflection -",
        ),
        "sphere_array_above_slab": ("Transmission", "Reflection"),
    }
    for name, curve in raw.get("cpc", {}).get("curves", {}).items():
        base = {
            "id": f"cpc-{name}",
            "family": f"cpc-{name}",
            "paper": raw["cpc"]["paper"],
            "quantity": labels[name],
            "unit": "Efficiency" if name == "sphere" else "Power fraction",
            "x_label": "Wavelength (nm)",
            "parameters": {
                "order": 3
                if name == "sphere_array_above_slab"
                else 4
                if name == "sphere"
                else None
            },
            "references": [],
            "source_url": "https://github.com/tfp-photonics/treams/blob/"
            "1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/"
            + {
                "sphere": "sphere.py",
                "chiral_slab": "slab.py",
                "sphere_array_above_slab": "array_spheres.py",
            }[name],
        }
        curves.append(
            {**base, **curve, "plot_grid": [2 * math.pi / x for x in curve["grid"]]}
        )
        if "threshold_approach" in curve:
            approach = curve["threshold_approach"]
            curves.append(
                {
                    **base,
                    **approach,
                    "id": f"cpc-{name}-threshold-approach",
                    "plot_grid": approach["wavelength_nm"],
                }
            )
            curves[-2].pop("threshold_approach", None)
    return curves


def observe(curves):
    observations = []
    for curve in curves:

        def add(metric, error, tolerance, reference_kind, *, curve=curve, **details):
            finite = error is not None and math.isfinite(error)
            observations.append(
                {
                    "id": f"{curve['id']}/{reference_kind}/{metric}",
                    "family": curve["family"],
                    "backend": "treams-rs",
                    "reference_kind": reference_kind,
                    "metric": metric,
                    "error": error if finite else None,
                    "tolerance": tolerance,
                    "status": (
                        "diagnostic"
                        if tolerance is None
                        else "passed"
                        if error <= tolerance
                        else "failed"
                    )
                    if finite
                    else "error",
                    "parameters": {
                        **curve["parameters"],
                        "samples": len(curve["grid"]),
                    },
                    "series": curve["id"],
                    **details,
                }
            )

        comparisons = [
            {
                "reference_kind": "upstream",
                "values": curve["upstream"],
                "rtol": 2e-9,
                "atol": 2e-11,
            },
            *curve["references"],
        ]
        for reference in comparisons:
            pairs = [
                (a, b)
                for row_a, row_b in zip(
                    curve["native"], reference["values"], strict=True
                )
                for a, b in zip(row_a, row_b, strict=True)
            ]
            finite = [
                (a, b)
                for a, b in pairs
                if a is not None
                and b is not None
                and math.isfinite(a)
                and math.isfinite(b)
            ]
            complete = len(finite) == len(pairs)
            difference = [abs(a - b) for a, b in finite]
            denom = math.hypot(*(b for _, b in finite))
            relative = (
                math.hypot(*difference) / denom
                if denom
                else 0.0
                if not any(difference)
                else None
            )
            values = {
                "max_abs_error": max(difference, default=0.0),
                "relative_l2_error": relative,
                "max_scaled_error": max(
                    (
                        d / (reference["atol"] + reference["rtol"] * abs(b))
                        for d, (_, b) in zip(difference, finite, strict=True)
                    ),
                    default=0.0,
                ),
            }
            for metric, value in values.items():
                add(
                    metric,
                    value if complete else None,
                    1.0 if metric == "max_scaled_error" else None,
                    reference["reference_kind"],
                    nonfinite_count=len(pairs) - len(finite),
                    element_count=len(pairs),
                    atol=reference["atol"],
                    rtol=reference["rtol"],
                )
        values = curve["native"]
        valid = all(v is not None and math.isfinite(v) for row in values for v in row)
        add(
            "passivity_nonnegative_violation",
            max([0.0, *(-v for row in values for v in row)]) if valid else None,
            1e-12,
            "physical_invariant",
        )
        family = curve["family"]
        if family.startswith("ebeam") or family == "cpc-sphere":
            add(
                "scattering_exceeds_extinction",
                max(
                    [
                        0.0,
                        *(
                            row[i] - row[i + 1]
                            for row in values
                            for i in range(0, len(row), 2)
                        ),
                    ]
                )
                if valid
                else None,
                1e-12,
                "physical_invariant",
            )
        elif family == "cpc-chiral_slab":
            add(
                "passivity_power_excess",
                max([0.0, *(row[i] + row[i + 1] - 1 for row in values for i in (0, 2))])
                if valid
                else None,
                1e-12,
                "physical_invariant",
            )
        else:
            add(
                "power_balance_error",
                max(abs(sum(row) - 1) for row in values) if valid else None,
                2e-10 if "threshold-approach" in curve["id"] else 1e-10,
                "physical_invariant",
            )
        if "threshold-approach" in curve["id"]:
            far = abs(values[0][0] - values[1][0]) if valid else 0
            near = abs(values[2][0] - values[3][0]) if valid else 0
            add(
                "two_sided_threshold_gap_ratio",
                near / far if valid and far else None,
                1.0,
                "truncation",
                scope="Gap at relative offsets 1e-6 versus 1e-4; threshold itself excluded",
            )
        convergence = curve.get("convergence")
        if convergence:
            pairs = [
                (a, b)
                for row, index in zip(
                    convergence["refined_native"], convergence["indices"], strict=True
                )
                for a, b in zip(row, values[index], strict=True)
            ]
            valid_refined = all(
                a is not None
                and b is not None
                and math.isfinite(a)
                and math.isfinite(b)
                for a, b in pairs
            )
            add(
                "truncation_max_absolute_change",
                max(abs(a - b) for a, b in pairs) if valid_refined else None,
                None,
                "truncation",
                refined_order=convergence["refined_order"],
            )
            add(
                "truncation_max_relative_change",
                max(abs(a - b) / max(abs(a), 1e-30) for a, b in pairs)
                if valid_refined
                else None,
                None,
                "truncation",
                refined_order=convergence["refined_order"],
                scope="Sensitivity at five frequencies; no claim of converged absolute accuracy",
            )
    return observations


def recompute():
    import warnings

    import numpy as np
    import qualify_papers as paper

    failures = []

    def sample(function, lib, grid, columns, **kwargs):
        values = []
        for index, x in enumerate(grid):
            try:
                row = function(lib, float(x), **kwargs).tolist()
                if len(row) != columns or not all(math.isfinite(v) for v in row):
                    raise ValueError("Wrong shape or nonfinite spectrum")
            except Exception as error:
                row = [None] * columns
                failures.append(
                    {
                        "function": function.__name__,
                        "backend": lib.__name__,
                        "index": index,
                        "x": float(x),
                        "parameters": kwargs,
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
            values.append(row)
        return values

    def curve(function, grid, columns, **kwargs):
        return {
            "grid": np.asarray(grid).tolist(),
            "native": sample(function, paper.tr, grid, columns, **kwargs),
            "upstream": sample(function, paper.treams, grid, columns, **kwargs),
        }

    raw = {
        "ebeam": {"curves": {}},
        "cpc": {"paper": "https://doi.org/10.1016/j.cpc.2023.109076", "curves": {}},
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "'where' used without 'out'.*",
            UserWarning,
            module=r"treams\._operators",
        )
        for name, spec in paper.EBEAM_CURVES.items():
            cylindrical, order = spec["cylindrical"], spec["order"]
            energy = np.linspace(*spec["bounds"], paper.EBEAM_SAMPLES)
            result = curve(
                paper.electron_spectrum_point,
                energy,
                2,
                cylindrical=cylindrical,
                order=order,
            )
            selected = paper.EBEAM_REFINED
            result.update(
                {
                    "order": order,
                    "unit": "1/(eV nm)" if cylindrical else "1/eV",
                    "convergence": {
                        "indices": selected,
                        "refined_order": order + 2,
                        "refined_native": sample(
                            paper.electron_spectrum_point,
                            paper.tr,
                            energy[selected],
                            2,
                            cylindrical=cylindrical,
                            order=order + 2,
                        ),
                    },
                }
            )
            notebook_energy, wavelength, dispersion_hbar = paper.notebook_grid(
                cylindrical
            )
            notebook = curve(
                paper.electron_spectrum_point,
                notebook_energy,
                2,
                cylindrical=cylindrical,
                order=order,
                dispersion_hbar=dispersion_hbar,
            )
            notebook.update(
                {
                    "wavelength_nm": wavelength.tolist(),
                    "dispersion_hbar_eV_seconds": dispersion_hbar,
                    "source_correction": 'Explicit .changepoltype("parity") in the cylinder T-matrix, matching the author regression test; omitted by the original notebook.'
                    if cylindrical
                    else None,
                }
            )
            result["author_notebook_grid"] = notebook
            raw["ebeam"]["curves"][name] = result
        for name, function, bounds, count, columns in paper.CPC_CURVES:
            grid = paper.cpc_grid(name, bounds, count)
            result = curve(function, grid, columns)
            if name == "sphere_array_above_slab":
                selected = paper.ARRAY_REFINED
                result["convergence"] = {
                    "indices": selected,
                    "original_order": paper.ARRAY_ORDER,
                    "refined_order": paper.ARRAY_REFINED_ORDER,
                    "refined_native": sample(
                        function,
                        paper.tr,
                        grid[selected],
                        columns,
                        order=paper.ARRAY_REFINED_ORDER,
                    ),
                }
                result["excluded_author_samples"] = [dict(paper.EXCLUDED_ARRAY_SAMPLE)]
                wavelength = paper.threshold_wavelengths()
                result["threshold_approach"] = {
                    **curve(function, 2 * np.pi / wavelength, columns),
                    "wavelength_nm": wavelength.tolist(),
                    "scope": "Two-sided approach, not evaluation at the singular endpoint.",
                }
            raw["cpc"]["curves"][name] = result
    return raw, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--normalize-existing",
        type=Path,
        help="Transform a saved qualification without executing solvers; clearly historical",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    started = time.perf_counter()
    fixture = references()
    if args.normalize_existing:
        raw = json.loads(args.normalize_existing.read_text())
        failures = []
        source = {
            key: raw[key]
            for key in (
                "native_sha256",
                "python_source_sha256",
                "script_sha256",
                "versions",
            )
            if key in raw
        }
        source["historical_result_sha256"] = file_sha256(args.normalize_existing)
        source["historical_platform"] = raw.get("platform")
    else:
        os.environ.update(pinned_threads(args.threads))
        from benchmark_illumination import fingerprints
        from threadpoolctl import threadpool_limits

        from treams_rs import _native

        with threadpool_limits(limits=args.threads):
            raw, failures = recompute()
        source = {
            **fingerprints(),
            **fingerprints(upstream=True),
            "native_profile": _native.build_profile(),
        }
    curves = normalize_curves(raw, fixture)
    observations = observe(curves)
    result = {
        "kind": "accuracy",
        "fresh": not bool(args.normalize_existing),
        "passed": bool(observations)
        and all(row["status"] in ("passed", "diagnostic") for row in observations)
        and not failures,
        "complete": len(curves) == 8 and not failures,
        "observations": observations,
        "paper_curves": curves,
        "failures": failures,
        "source": {
            **source,
            "collector_sha256": file_sha256(__file__),
            "reference_harness_sha256": file_sha256(ROOT / "scripts/qualify_papers.py"),
            "fixture_sha256": {name: file_sha256(FIXTURES / name) for name in fixture},
            "provenance": fixture["source-provenance.json"],
        },
        "environment": {
            "os": platform.system(),
            "python": platform.python_version(),
            "threads_requested": args.threads,
        },
        "execution_seconds_not_a_benchmark": time.perf_counter() - started,
        "protocol": {
            "scope": "Full electron-beam sphere/cylinder notebook and regression spectra; CPC companion sphere, chiral slab and periodic array spectra. Not the CPC quasi-BIC figure or all published applications.",
            "upstream": "Fresh treams 0.4.5 calculations are implementation agreement, separately labeled from stored author data and notebook executions.",
            "correction": "Cylinder notebook uses the explicitly repaired parity basis; original uncorrected results are not claimed reproduced.",
            "threshold": "Exact 350 nm array endpoint excluded; four two-sided approach samples retained.",
            "truncation": "Orders +2 at five energies are sensitivity diagnostics, not imposed convergence guarantees.",
            "scaled_error": "max(abs(native-reference)/(atol+rtol*abs(reference))); threshold 1. Zero remains zero.",
            "warning_filter": "Known upstream NumPy where-without-out warning suppressed only in treams._operators.",
        },
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
