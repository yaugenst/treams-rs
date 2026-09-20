# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2.1,<3"]
# ///
"""Render auditable CPU benchmark and numerical qualification reports.

uv --no-config run --no-project --script scripts/plot_benchmarks.py \
    --manifest benchmarks/campaign/linux.json --output benchmarks/report

A manifest contains suite_id, environment, source, protocol and cases. Each case
has id, kind (cluster/illumination/gradient/gpu), result, status, family, tier,
and optionally series and x={name,value,unit}. Paths are relative to the manifest
directory, or --root when supplied. --manifest may be repeated. The archived
complete-qualification.json format is also accepted, clearly marked historical.

No benchmark executes here. Measurements from different suites/builds are never
pooled. A missing/failed result is a table entry, never a zero-valued data point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

COLORS = {"treams": "#b84b3a", "rust": "#2166ac", "gpu": "#308267"}
PALETTE = ["#2166ac", "#b84b3a", "#308267", "#8856a7", "#b27918", "#525b66"]
LABELS = {"treams": "treams", "rust": "treams-rs"}
PASS = {"passed", "complete", "ok", "verified", "success"}


def backend_color(backend, index=0):
    name = backend.lower()
    if "legacy" in name:
        return "#b27918"
    if "gpu" in name or "cuda" in name:
        return COLORS["gpu"]
    if name == "rust" or name.startswith("treams-rs"):
        return COLORS["rust"]
    if name == "treams":
        return COLORS["treams"]
    return ["#8856a7", "#525b66", "#b27918"][index % 3]


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def samples(values):
    if values is None:
        return []
    values = [values] if finite(values) else values
    if not isinstance(values, list) or any(not finite(x) or x <= 0 for x in values):
        raise ValueError("Runtime samples must be finite positive seconds")
    return [float(x) for x in values]


def family_for(workload):
    for prefix, label in [
        ("coordinate", "Coordinates"),
        ("angular", "Special functions"),
        ("bessel", "Special functions"),
        ("wigner", "Special functions"),
        ("incgamma", "Special functions"),
        ("intkambe", "Special functions"),
        ("callback", "Particle coefficients"),
        ("particle-cluster", "Finite clusters"),
        ("cylindrical-particle-cluster", "Finite clusters"),
        ("cylindrical-field", "Field sampling"),
        ("internal-field", "Field sampling"),
        ("field", "Field sampling"),
        ("cylindrical-periodic", "Periodic scattering"),
        ("periodic", "Periodic scattering"),
        ("oriented-chirality", "Power observables"),
        ("plane-permutation", "Wave transforms"),
        ("cylindrical-expansion", "Wave transforms"),
        ("geometry", "Geometry / material"),
        ("namespace", "Wave transforms"),
        ("lattice", "Lattice sums"),
        ("wave", "Local vector waves"),
        ("polar", "Translation coefficients"),
        ("operator", "Field operators"),
        ("power", "Power observables"),
        ("coefficient", "Particle coefficients"),
        ("ebcm", "EBCM"),
    ]:
        if workload.startswith(prefix):
            return label
    return workload.removesuffix("-forward").replace("-", " ").capitalize()


def measurement(backend, phase, timing, metadata, median=None):
    timing = samples(timing)
    if not timing and median is None:
        return None
    center = statistics.median(timing) if timing else float(median)
    if not finite(center) or center <= 0:
        raise ValueError("Runtime median must be finite and positive")
    peak, baseline = metadata.get("peak_rss_mib"), metadata.get("baseline_rss_mib")
    return {
        "backend": backend,
        "phase": phase,
        "seconds": timing,
        "median": center,
        "peak_rss_mib": peak,
        "baseline_rss_mib": baseline,
        "growth_rss_mib": max(0.0, peak - baseline)
        if finite(peak) and finite(baseline)
        else None,
        "threads": metadata.get(
            "threads", metadata.get("actual_cpu_threads", metadata.get("cpu_threads"))
        ),
        "native_sha256": metadata.get("native_sha256"),
        "python_source_sha256": metadata.get(
            "python_source_sha256", metadata.get("package_sha256")
        ),
        "treams_package_sha256": metadata.get("treams_package_sha256"),
        "benchmark_sha256": metadata.get("benchmark_sha256"),
        "platform": metadata.get("platform"),
        "paired": metadata.get("paired", False),
        "forward_records_adjoint": metadata.get("forward_records_adjoint")
        if backend == "rust"
        else False,
        "workflow": metadata.get("workflow", metadata.get("method")),
    }


def read_measurements(data, kind):
    """Normalize native harness outputs while preserving each timing boundary."""
    result = []

    def add(backend, phase, timing, meta, median=None):
        row = measurement(backend, phase, timing, meta, median)
        if row:
            result.append(row)

    if kind == "cluster":
        paired = data.get("timing_comparison", {}).get("pairs_seconds", [])
        for raw in data.get("results", []):
            backend = raw["backend"]
            timing = (
                [p[backend] for p in paired]
                if paired
                else raw.get("samples_seconds", [])
            )
            add(
                backend,
                "forward",
                timing,
                {**raw, "paired": bool(paired)},
                raw.get("median_seconds"),
            )
            add(
                backend,
                "reverse",
                raw.get("backward_samples_seconds", []),
                {**raw, "peak_rss_mib": raw.get("forward_and_backward_peak_rss_mib")},
                raw.get("backward_median_seconds"),
            )
    elif kind == "illumination":
        for raw in data.get("measurements", []):
            backend = raw["backend"] + (
                " + recording" if raw["phase"] == "adjoint" else ""
            )
            for phase, key in [
                ("fresh", "fresh"),
                ("reuse", "reuse"),
                ("reverse", "backward"),
                ("setup", "setup"),
            ]:
                add(
                    backend,
                    phase,
                    raw.get(f"{key}_seconds", []),
                    raw,
                    raw.get(f"{key}_median_seconds"),
                )
    elif kind == "gradient":
        for raw in data.get("measurements", []):
            add(
                raw["backend"],
                raw["phase"],
                raw.get("elapsed_seconds", []),
                raw,
                raw.get("median_seconds"),
            )
            if raw["phase"] == "adjoint":
                for phase, key in [("record", "record"), ("reverse", "reverse")]:
                    add(
                        raw["backend"],
                        phase,
                        raw.get(f"{key}_seconds", []),
                        raw,
                        raw.get(f"{key}_median_seconds"),
                    )
    elif kind == "accuracy":
        return []
    elif kind == "gpu":
        normalized = data.get("results", []) if "raw" in data else []
        for raw in normalized:
            label = " · ".join([raw["backend"], raw["phase"], raw["path"]])
            add(
                label,
                "warm",
                raw.get("samples_seconds"),
                {
                    **data.get("source", {}),
                    **data.get("parameters", {}),
                    "workflow": data.get("scope"),
                },
                raw["median_seconds"],
            )
        if "raw" in data:
            data = data["raw"]
        raw_samples = data.get("samples", [])
        if normalized:
            pass
        elif raw_samples and isinstance(raw_samples[0], dict):
            # GPU harnesses name every timed boundary, including host/resident.
            for key in raw_samples[0]:
                if key.endswith("_seconds"):
                    add(
                        key.removesuffix("_seconds"),
                        "warm",
                        [r[key] for r in raw_samples],
                        data,
                    )
        else:
            for key in ["cpu_seconds", "cpu_preweighted_seconds", "gpu_seconds"]:
                if key in data:
                    add(key.removesuffix("_seconds"), "warm", [], data, data[key])
        for key in [
            "context_seconds",
            "setup_seconds",
            "assembly_seconds",
            "operator_upload_seconds",
            "cold_gpu_seconds",
            "cold_application_seconds",
            "cpu_preparation_seconds",
            "cpu_prepare_geometry_seconds",
            "cpu_row_major_pack_seconds",
            "mode_upload_seconds",
            "point_upload_seconds",
            "output_allocation_seconds",
            "cold_kernel_seconds",
            "download_seconds",
        ]:
            if finite(data.get(key)) and data[key] > 0:
                add(key.removesuffix("_seconds"), "setup / cold", [], data, data[key])
    else:
        raise ValueError(f"Unsupported result kind: {kind}")
    return result


def load_manifest(path, root=None):
    manifest = json.loads(path.read_text())
    historical = "verified" in manifest and "cases" not in manifest
    if historical:
        entries = [
            dict(
                row,
                result=row["output"],
                id=Path(row["output"]).stem,
                kind="cluster",
                status="passed",
            )
            for row in manifest["verified"]
        ]
        suite = "Archived reference qualification"
        base = root or path.parent.parent
    else:
        entries = manifest["cases"]
        suite = manifest["suite_id"]
        environment = manifest.get("environment", {})
        if environment.get("os"):
            platform = "macOS" if environment["os"] == "Darwin" else environment["os"]
            suite += f" · {platform} · {manifest.get('phase', 'campaign')}"
        base = root or path.parent
    cases = []
    for entry in entries:
        case = dict(
            entry,
            suite=suite,
            historical=historical or manifest.get("historical", False),
            campaign_phase="broad" if historical else manifest.get("phase"),
        )
        case["id"] = case["id"] if "id" in case else Path(case["result"]).stem
        case["status"] = case.get("status", "passed")
        case["measurements"] = []
        case["raw_path"] = None
        if case.get("result"):
            raw_path = base / case["result"]
            if raw_path.exists():
                case["raw_path"] = raw_path
                try:
                    data = json.loads(raw_path.read_text())
                except (ValueError, UnicodeError) as exc:
                    data = {}
                    case["status"], case["error"] = "invalid_output", str(exc)
                case["raw"] = data
                case["kind"] = case.get("kind", "cluster")
                if data.get("validation", {}).get("passed") is False:
                    case["status"] = "accuracy_failed"
                if case["kind"] == "accuracy" and data.get("passed") is False:
                    case["status"] = "accuracy_failed"
                case["measurements"] = read_measurements(data, case["kind"])
                first = next(iter(data.get("results", [])), {})
                params = {**first, **data.get("case", {}), **case.get("parameters", {})}
                case["parameters"] = {
                    k: v
                    for k, v in params.items()
                    if k
                    in {
                        "particles",
                        "lmax",
                        "samples",
                        "layers",
                        "channels",
                        "dimension",
                        "threads",
                        "columns",
                        "strength",
                        "parameter_count",
                        "parameters",
                    }
                }
                workload = params.get("workload", data.get("workload", case["id"]))
                case["family"] = case.get("family", family_for(workload))
                case["feature_group"] = (
                    family_for(case["family"])
                    if case["family"] == workload
                    else case["family"]
                )
                case["series"] = case.get("series", workload)
                dimension = data.get(
                    "dimension",
                    params.get(
                        "dimension", data.get("matrix_dimension", data.get("points", 1))
                    ),
                )
                if case["kind"] == "gradient":
                    dimension = params.get("parameter_count", dimension)
                case["x"] = case.get(
                    "x",
                    {
                        "name": "Real parameter coordinates"
                        if case["kind"] == "gradient"
                        else "Problem dimension",
                        "value": dimension or 1,
                        "unit": "",
                    },
                )
                case["tier"] = case.get("tier", "unspecified")
                case["paired"] = bool(
                    data.get("timing_comparison", {}).get("pairs_seconds")
                )
            elif case["status"] in PASS:
                case["status"], case["error"] = (
                    "missing",
                    "Declared result file is missing",
                )
        case.setdefault("family", case.get("group", "Unclassified"))
        case.setdefault("tier", "unspecified")
        # Different executable/source/platform identities get independent panels.
        fingerprints = sorted(
            {
                tuple(
                    str(m.get(k) or "")
                    for k in (
                        "backend",
                        "native_sha256",
                        "python_source_sha256",
                        "treams_package_sha256",
                        "benchmark_sha256",
                        "platform",
                    )
                )
                for m in case["measurements"]
                if any(
                    m.get(k)
                    for k in [
                        "native_sha256",
                        "python_source_sha256",
                        "treams_package_sha256",
                        "benchmark_sha256",
                    ]
                )
            }
        )
        if not fingerprints:
            raw = case.get("raw", {})
            fingerprints = raw.get(
                "source",
                raw.get(
                    "fingerprints",
                    manifest.get("source", manifest.get("fingerprints", {})),
                ),
            )
        case["fingerprint"] = hashlib.sha256(
            json.dumps(fingerprints, sort_keys=True).encode()
        ).hexdigest()[:10]
        cases.append(case)
    illumination = [c for c in cases if c.get("kind") == "illumination"]
    upstream_hashes = {
        m["treams_package_sha256"]
        for c in illumination
        for m in c["measurements"]
        if m.get("treams_package_sha256")
    }
    if len(upstream_hashes) <= 1:
        # An intentionally skipped dense/upstream method is not a new build.
        # Preserve all per-method hashes while connecting one native build's
        # scaling sweep across caps; never combine distinct upstream packages.
        for case in illumination:
            native = sorted(
                {
                    tuple(
                        str(m.get(key) or "")
                        for key in (
                            "native_sha256",
                            "python_source_sha256",
                            "benchmark_sha256",
                            "platform",
                        )
                    )
                    for m in case["measurements"]
                    if m.get("native_sha256")
                }
            )
            if native:
                identity = {"native": native, "upstream": sorted(upstream_hashes)}
                case["fingerprint"] = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest()[:10]
    return {
        "name": suite,
        "historical": historical or manifest.get("historical", False),
        "path": path,
        "metadata": manifest,
        "cases": cases,
    }


def cpu_pair(case):
    if case["status"] not in PASS or case.get("kind") != "cluster":
        return None
    rows = {m["backend"]: m for m in case["measurements"] if m["phase"] == "forward"}
    return (rows["treams"], rows["rust"]) if {"treams", "rust"} <= rows.keys() else None


def speedup(left, right):
    if (
        left.get("paired")
        and left["seconds"]
        and len(left["seconds"]) == len(right["seconds"])
    ):
        return statistics.median(
            a / b for a, b in zip(left["seconds"], right["seconds"], strict=True)
        )
    return left["median"] / right["median"]


def ratio_interval(left, right, paired=False):
    """Deterministic sample bootstrap; no confidence band for summary-only data."""
    import numpy as np

    estimate = speedup(left, right)
    a, b = np.array(left["seconds"]), np.array(right["seconds"])
    if min(len(a), len(b)) < 3:
        return estimate, estimate, estimate
    rng = np.random.default_rng(76183)
    ia = rng.integers(0, len(a), (2000, len(a)))
    ib = ia if paired and len(a) == len(b) else rng.integers(0, len(b), (2000, len(b)))
    ratios = (
        np.median(a[ia] / b[ib], axis=1)
        if paired and len(a) == len(b)
        else np.median(a[ia], axis=1) / np.median(b[ib], axis=1)
    )
    low, high = np.quantile(ratios, [0.025, 0.975])
    return estimate, min(estimate, float(low)), max(estimate, float(high))


def slowdown_summary(cases):
    observed = bounded = 0
    for case in cases:
        pair = cpu_pair(case)
        if pair and speedup(*pair) < 1:
            observed += 1
            if min(len(row["seconds"]) for row in pair) >= 3:
                bounded += ratio_interval(*pair, case.get("paired", False))[2] < 1
    return observed, bounded


def chart_style():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def environment_label(suite):
    environment = suite["metadata"].get("environment", {})
    os_name = environment.get("os", "")
    if os_name == "Darwin":
        os_name = "macOS"
    cpu = re.sub(
        r"\s+\d+-Core Processor$", "", environment.get("cpu_model") or ""
    ).removeprefix("AMD ")
    return " · ".join(part for part in [cpu, os_name] if part) or suite["name"]


def suite_caption(suite):
    groups = {c.get("group") for c in suite["cases"] if c.get("group")}
    group = (
        next(iter(groups)) if len(groups) == 1 else suite["metadata"].get("phase", "")
    )
    label = {
        "broad": "Feature coverage",
        "scaling": "Problem-size scaling",
        "thread-scaling": "Thread scaling",
        "gradient": "Gradient benchmarks",
        "illumination": "Requested illuminations",
        "gpu": "GPU benchmarks",
    }.get(group, (group or "").replace("-", " ").capitalize())
    return " · ".join(part for part in [environment_label(suite), label] if part)


def save_figure(
    fig,
    destination,
    name,
    title,
    description,
    gallery,
    pdf,
    *,
    subtitle=None,
    provenance=None,
):
    import matplotlib.pyplot as plt

    wrapped_title = textwrap.fill(title.replace("_", " "), int(fig.get_figwidth() * 7))
    title_space = wrapped_title.count("\n") * 20 / (72 * fig.get_figheight())
    fig.suptitle(wrapped_title, x=0.06, ha="left", fontsize=15, fontweight="bold")
    if subtitle:
        fig.text(0.06, 0.921 - title_space, subtitle, fontsize=10, color="#525b66")
    for ax in fig.axes:
        ax.set_title(textwrap.fill(ax.get_title(loc="left"), 36), loc="left")
    fig.text(
        0.06, 0.014, description, fontsize=8, color="#525b66", wrap=True, va="bottom"
    )
    fig.tight_layout(
        rect=(0.02, 0.06, 0.99, (0.93 if subtitle else 0.98) - title_space), pad=1.5
    )
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(destination / f"{name}.{ext}", dpi=180)
    pdf.savefig(fig)
    gallery.append(
        {
            "name": name,
            "title": title,
            "description": description,
            "subtitle": subtitle,
            "provenance": provenance,
        }
    )
    plt.close(fig)


def plot_overview(cases, save):
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [(c, cpu_pair(c)) for c in cases if cpu_pair(c)]
    if not valid:
        return
    ratios = np.array([speedup(a, b) for _, (a, b) in valid])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    variants = [
        (False, "Forward only", COLORS["rust"]),
        (True, "Forward + retained adjoint context", COLORS["gpu"]),
    ]
    for recorded, label, color in variants:
        order = np.sort(
            [
                speedup(a, b)
                for _, (a, b) in valid
                if bool(b.get("forward_records_adjoint")) == recorded
            ]
        )
        if len(order):
            axes[0].step(
                order,
                np.arange(1, len(order) + 1) / len(order),
                where="post",
                color=color,
                lw=2,
                label=f"{label} · {len(order)} cases",
            )
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].axvspan(min(ratios.min(), 0.5), 1, color=COLORS["treams"], alpha=0.07)
    axes[0].axvline(1, color="#555", ls="--", lw=1)
    axes[0].set(
        xscale="log",
        xlabel="treams time / treams-rs time · higher is faster",
        ylabel="Fraction of measured cases",
        title=f"Every matched CPU case · n={len(valid)}",
        ylim=(0, 1.03),
    )
    observed, bounded = slowdown_summary(cases)
    axes[0].text(
        0.03,
        0.94,
        f"{observed} median slowdowns; {bounded} with 95% interval below 1\nmedian {np.median(ratios):.2f}x · range {ratios.min():.2f}-{ratios.max():.1f}x",
        va="top",
        transform=axes[0].transAxes,
    )
    for recorded, label, color in variants:
        points = [
            (speedup(a, b), b["peak_rss_mib"] / a["peak_rss_mib"])
            for c, (a, b) in valid
            if bool(b.get("forward_records_adjoint")) == recorded
            and finite(a["peak_rss_mib"])
            and finite(b["peak_rss_mib"])
        ]
        if points:
            axes[1].scatter(
                *zip(*points, strict=True), s=21, alpha=0.65, color=color, label=label
            )
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].axvline(1, color="#555", ls="--", lw=1)
    axes[1].axhline(1, color="#555", ls="--", lw=1)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Speedup · higher is faster",
        ylabel="Rust / upstream process peak RSS · lower is smaller",
        title="Runtime and memory together",
    )
    save(
        fig,
        "overview",
        "CPU comparison at a glance",
        "Each point is one accuracy-qualified case. Full process peak RSS includes imports. Suites/builds are separate; no total-workload weighting.",
    )
    grouped = defaultdict(list)
    for c, pair in valid:
        grouped[c.get("feature_group", c["family"])].append((c, pair))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.48 * len(grouped) + 2)))
    names = sorted(grouped)
    tier_order = {"small": 0, "medium": 1, "large": 2, "very-large": 3, "very_large": 3}
    tiers_present = sorted(
        {c["tier"] for c, _ in valid}, key=lambda tier: (tier_order.get(tier, 4), tier)
    )
    for i, name in enumerate(names):
        items = grouped[name]
        values = np.array([speedup(a, b) for _, (a, b) in items])
        jitter = np.linspace(-0.17, 0.17, len(values))
        axes[0].scatter(
            values,
            i + jitter,
            s=15,
            alpha=0.5,
            color=np.where(values < 1, COLORS["treams"], COLORS["rust"]),
        )
        axes[0].scatter(
            np.median(values), i, s=44, marker="D", color="#202833", zorder=4
        )
        tiers = Counter(c["tier"] for c, _ in items)
        start = 0
        for j, tier in enumerate(tiers_present):
            n = tiers.get(tier, 0)
            axes[1].barh(
                i,
                n,
                left=start,
                color=PALETTE[j % len(PALETTE)],
                label=tier.replace("_", " ") if i == 0 else None,
            )
            start += n
        axes[1].text(start + 0.2, i, str(len(items)), va="center", fontsize=8)
    axes[0].set(
        xscale="log",
        xlabel="Speedup · diamond = family median",
        yticks=range(len(names)),
        yticklabels=names,
        title="Coverage across numerical features",
    )
    axes[0].axvline(1, color="#555", ls="--", lw=1)
    axes[1].set(
        yticks=range(len(names)),
        yticklabels=[],
        xlabel="Number of measured cases",
        title="Input tiers from the declared manifest",
    )
    axes[1].legend(fontsize=8, loc="lower right")
    save(
        fig,
        "features",
        "Feature coverage and speedup distribution",
        "Every qualified result is visible, including regressions. A family median is descriptive, not an application speedup; cases have equal weight.",
    )


def plot_frontier(cases, save):
    import matplotlib.pyplot as plt

    valid = [(c, cpu_pair(c)) for c in cases if cpu_pair(c)]
    if not valid:
        return
    # Label the worst ratios, not only the favorable cases.
    ranked = sorted(valid, key=lambda p: speedup(*p[1]))[:18]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, len(ranked) * 0.34 + 2)))
    for i, (case, (a, b)) in enumerate(ranked):
        mid, lo, hi = ratio_interval(a, b, case.get("paired", False))
        has_interval = min(len(a["seconds"]), len(b["seconds"])) >= 3
        color = (
            "#6b7280"
            if not has_interval
            else COLORS["treams"]
            if hi < 1
            else "#b27918"
            if lo <= 1
            else COLORS["rust"]
        )
        axes[0].errorbar(
            mid,
            i,
            xerr=[[mid - lo], [hi - mid]],
            fmt="o",
            color=color,
            capsize=3,
        )
        for row, shift in [(a, -0.12), (b, 0.12)]:
            if finite(row["peak_rss_mib"]):
                axes[1].scatter(
                    row["peak_rss_mib"], i + shift, color=COLORS[row["backend"]], s=25
                )
    labels = [c["id"] for c, _ in ranked]
    axes[0].set(
        xscale="log",
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="Speedup · bootstrap 95% interval",
        title="Smallest speedups, including regressions",
    )
    axes[0].axvline(1, color="#555", ls="--", lw=1)
    axes[1].set(
        yticks=range(len(labels)),
        yticklabels=[],
        xlabel="Peak RSS (MiB) · blue Rust / red upstream",
        title="Absolute process memory for those cases",
    )
    save(
        fig,
        "smallest-speedups",
        "Where the performance margin is smallest",
        "Red: entire 95% sample interval below 1; amber: interval crosses 1; gray: fewer than 3 samples. Resampling describes this run, not independent sessions or fixed-order drift.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for _, (a, b) in valid:
        for row in [a, b]:
            if finite(row["peak_rss_mib"]):
                axes[0].scatter(
                    row["median"],
                    row["peak_rss_mib"],
                    s=15,
                    alpha=0.5,
                    color=COLORS[row["backend"]],
                )
        if all(finite(row["growth_rss_mib"]) for row in [a, b]):
            axes[1].scatter(
                a["growth_rss_mib"],
                b["growth_rss_mib"],
                s=18,
                alpha=0.55,
                color=COLORS["rust"],
            )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Time per call (s)",
        ylabel="Process peak RSS (MiB)",
        title="Time-memory tradeoff · blue Rust / red upstream",
    )
    axes[1].set(
        xscale="symlog",
        yscale="symlog",
        xlabel="treams after-setup high-water growth (MiB)",
        ylabel="treams-rs after-setup high-water growth (MiB)",
        title="Memory growth, including zero measurements",
    )
    bound = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, bound], [0, bound], color="#555", ls="--", lw=1)
    save(
        fig,
        "memory",
        "Absolute memory and after-setup growth",
        "RSS growth is peak RSS minus after-setup baseline, clamped at zero. It is not live allocations or numerical working memory; allocator retention remains included.",
    )


def timing_line(ax, items, label, color, marker="o", linestyle="-"):
    import numpy as np

    items = sorted(items, key=lambda p: p[0])
    xs, ys = [x for x, _ in items], [m["median"] for _, m in items]
    low = [min(m["seconds"]) if m["seconds"] else m["median"] for _, m in items]
    high = [max(m["seconds"]) if m["seconds"] else m["median"] for _, m in items]
    distinct = len(set(xs)) == len(xs)
    ax.plot(
        xs,
        ys,
        marker=marker,
        label=label,
        color=color,
        lw=1.6,
        ms=4,
        linestyle=linestyle if distinct else "none",
    )
    if distinct:
        ax.fill_between(
            xs, np.minimum(low, ys), np.maximum(high, ys), alpha=0.13, color=color
        )


def scaling_failure_label(case):
    error = case.get("error", "") or ""
    if "Not equal to tolerance" in error or "assert_allclose" in error:
        return "Reference-check failure"
    return {
        "accuracy_failed": "Accuracy-check failure",
        "error": "Execution error",
        "timeout": "Timeout",
        "memory_limit": "Memory limit",
    }.get(case["status"], case["status"].replace("_", " ").capitalize())


def scaling_segments(entries):
    """Split measured curves at every planned position lacking a valid pair."""
    segments = [[]]
    for case in sorted(entries, key=lambda c: c["x"]["value"]):
        if cpu_pair(case):
            segments[-1].append(case)
        elif segments[-1]:
            segments.append([])
    return [segment for segment in segments if segment]


def plot_scaling(cases, save, planned_cases=None):
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for c in cases:
        # The broad harness's nominal dimension can be a batch length, radius,
        # or even an unchanged scalar argument. Only prespecified size sweeps
        # have an unambiguous axis and fixed-parameter contract.
        if (
            cpu_pair(c)
            and c.get("campaign_phase") != "broad"
            and c.get("group") != "broad"
        ):
            groups[(c["series"], c["x"]["name"])].append(c)
    # Failures often have no raw JSON and therefore no measured-build identity.
    # Overlay their declared positions only on matching measured series.
    for c in planned_cases if planned_cases is not None else cases:
        key = (c.get("series"), c.get("x", {}).get("name"))
        if c.get("kind") == "cluster" and not cpu_pair(c) and key in groups:
            groups[key].append(c)
    for (series, axis_name), entries in sorted(groups.items()):
        if len({c["x"]["value"] for c in entries}) < 2:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
        thread_axis = "thread" in axis_name.lower()
        # Hold the thread budget fixed unless it is the explicitly varied axis.
        threads = (
            [None]
            if thread_axis
            else sorted(
                {
                    str(
                        cpu_pair(c)[0]["threads"]
                        if cpu_pair(c)
                        else c.get("parameters", {}).get("threads")
                    )
                    for c in entries
                }
            )
        )
        for ti, thread in enumerate(threads):
            subset = (
                entries
                if thread_axis
                else [
                    c
                    for c in entries
                    if str(
                        cpu_pair(c)[0]["threads"]
                        if cpu_pair(c)
                        else c.get("parameters", {}).get("threads")
                    )
                    == thread
                ]
            )
            for si, segment in enumerate(scaling_segments(subset)):
                for bi, backend in enumerate(["treams", "rust"]):
                    rows = [(c["x"]["value"], cpu_pair(c)[bi]) for c in segment]
                    label = (
                        LABELS[backend]
                        if thread_axis
                        else f"{LABELS[backend]} · {thread} threads"
                    )
                    if si:
                        label = "_nolegend_"
                    color = (
                        COLORS[backend]
                        if len(threads) == 1
                        else PALETTE[(2 * ti + bi) % len(PALETTE)]
                    )
                    timing_line(axes[0], rows, label, color, "o" if bi else "s")
                    memory = [
                        (x, m["peak_rss_mib"])
                        for x, m in rows
                        if finite(m["peak_rss_mib"])
                    ]
                    if memory:
                        axes[1].plot(
                            *zip(*memory, strict=True),
                            marker="o" if bi else "s",
                            color=color,
                            label=label,
                        )
                points = [
                    (
                        c["x"]["value"],
                        ratio_interval(*cpu_pair(c), c.get("paired", False)),
                    )
                    for c in segment
                ]
                axes[2].errorbar(
                    [x for x, _ in points],
                    [r[0] for _, r in points],
                    yerr=[
                        [r[0] - r[1] for _, r in points],
                        [r[2] - r[0] for _, r in points],
                    ],
                    marker="o",
                    color=PALETTE[ti % len(PALETTE)],
                    label="_nolegend_"
                    if si
                    else "Matched thread budgets"
                    if thread_axis
                    else f"{thread} threads",
                    capsize=2,
                )
        failures = defaultdict(list)
        for case in entries:
            if not cpu_pair(case):
                failures[scaling_failure_label(case)].append(case["x"]["value"])
        for i, (label, positions) in enumerate(sorted(failures.items())):
            for ax in axes:
                # y is an axes fraction, never a fabricated timing/memory value.
                ax.scatter(
                    positions,
                    [0.025] * len(positions),
                    transform=ax.get_xaxis_transform(),
                    color="#525b66",
                    marker=["x", "+", "1", "2"][i % 4],
                    s=55,
                    label=label + " (no measurement)",
                    zorder=5,
                )
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel(axis_name)
        axes[0].set(
            yscale="log",
            ylabel="Time (s)",
            title="Runtime · full observed sample range",
        )
        axes[1].set(yscale="log", ylabel="Peak RSS (MiB)", title="Process memory")
        axes[2].set(
            yscale="log",
            ylabel="treams / Rust time",
            title="Speedup · 95% sample-bootstrap interval",
        )
        axes[2].axhline(1, color="#555", ls="--", lw=1)
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=8)
        save(
            fig,
            f"scaling-{slug(series)}-{slug(axis_name)}",
            f"Scaling: {series}",
            "Lines break at planned positions without valid measurements. Bottom-edge markers identify planned failures/statuses without assigning runtime or memory or blaming either backend. Full errors remain in the ledger. "
            + "; ".join(
                f"{label}: {axis_name}={','.join(map(str, sorted(set(positions))))}"
                for label, positions in sorted(failures.items())
            ),
        )


def plot_gradients(cases, save):
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for c in cases:
        if c.get("kind") == "gradient" and c["status"] in PASS:
            groups[c["series"]].append(c)
    for series, entries in sorted(groups.items()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
        variants = sorted(
            {
                (m["backend"], m["phase"])
                for c in entries
                for m in c["measurements"]
                if m["phase"] not in {"record", "reverse"}
            }
        )
        styles = {
            ("rust", "adjoint"): (COLORS["rust"], "o"),
            ("rust", "forward"): ("#75a7ce", "s"),
            ("treams", "finite_difference"): (COLORS["treams"], "^"),
            ("treams", "forward"): ("#df9c90", "s"),
            ("treams", "linear_adjoint"): ("#8856a7", "D"),
        }
        for backend, phase in variants:
            rows = [
                (c["x"]["value"], m)
                for c in entries
                for m in c["measurements"]
                if (m["backend"], m["phase"]) == (backend, phase)
            ]
            label = f"{LABELS.get(backend, backend)} · {phase.replace('_', ' ')}"
            color, marker = styles[(backend, phase)]
            timing_line(axes[0], rows, label, color, marker)
            mem = [
                (x, m["peak_rss_mib"])
                for x, m in sorted(rows, key=lambda pair: pair[0])
                if finite(m["peak_rss_mib"])
            ]
            if mem:
                axes[1].plot(
                    *zip(*mem, strict=True),
                    marker=marker,
                    label=label,
                    color=color,
                )
        for phase, color in [("record", COLORS["rust"]), ("reverse", COLORS["gpu"])]:
            rows = [
                (c["x"]["value"], m)
                for c in entries
                for m in c["measurements"]
                if m["backend"] == "rust" and m["phase"] == phase
            ]
            if rows:
                timing_line(axes[2], rows, phase, color)
        for ax in axes:
            ax.set(
                xscale="log",
                yscale="log",
                xlabel=entries[0]["x"]["name"],
            )
            ax.legend(fontsize=7)
        axes[0].set(ylabel="Time (s)", title="Full gradient cost, including forward")
        axes[1].set(
            ylabel="Process peak RSS (MiB)", title="Memory through the measured phase"
        )
        axes[2].set(ylabel="Time (s)", title="Native recording and reverse separately")
        linear_ratios = []
        for case in entries:
            timings = {
                (m["backend"], m["phase"]): m["median"] for m in case["measurements"]
            }
            if ("treams", "linear_adjoint") in timings and (
                "rust",
                "adjoint",
            ) in timings:
                linear_ratios.append(
                    timings[("treams", "linear_adjoint")] / timings[("rust", "adjoint")]
                )
        linear_note = ""
        if linear_ratios:
            linear_note = f" Exact linear adjoint: upstream/native total-time ratios {min(linear_ratios):.3g}-{max(linear_ratios):.3g}x; native has {sum(r < 1 for r in linear_ratios)}/{len(linear_ratios)} slower observed medians (not a significance claim)."
        family = entries[0]["family"].replace("-", " ").replace("/", " · ")
        scope = []
        for key, label in [("particles", "N"), ("lmax", "L"), ("threads", "threads")]:
            values = sorted({c["parameters"][key] for c in entries})
            value = str(values[0]) if len(values) == 1 else f"{values[0]}-{values[-1]}"
            scope.append(f"{label}={value}")
        save(
            fig,
            f"gradients-{slug(series)}",
            f"{family.capitalize()} gradients · {', '.join(scope)}",
            "treams finite differences evaluate every parameter coordinate; this is not upstream automatic differentiation. Exact linear adjoints are shown when available."
            + linear_note,
        )


def plot_illumination(cases, save):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, StrMethodFormatter

    groups = defaultdict(list)
    for c in cases:
        if c.get("kind") == "illumination" and c["status"] in PASS:
            groups[c["series"]].append(c)
    for series, entries in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
        backends = sorted({m["backend"] for c in entries for m in c["measurements"]})
        for backend in backends:
            recorded = backend.endswith(" + recording")
            method = backend.removesuffix(" + recording")
            upstream = method.startswith("treams-")
            method = method.removeprefix("treams-")
            label = f"{'treams' if upstream else 'Rust'} {method}"
            if recorded:
                label += " · recording"
            color = COLORS["treams"] if upstream else COLORS["rust"]
            marker = {"full": "o", "selected": "s", "matrix-free": "^"}[method]
            linestyle = "--" if recorded else "-"
            for j, phase in enumerate(["fresh", "reuse"]):
                rows = [
                    (c["x"]["value"], m)
                    for c in entries
                    for m in c["measurements"]
                    if m["backend"] == backend and m["phase"] == phase
                ]
                if rows:
                    timing_line(axes[j], rows, label, color, marker, linestyle)
                    if j == 0:
                        mem = [
                            (x, m["peak_rss_mib"])
                            for x, m in sorted(rows, key=lambda pair: pair[0])
                            if finite(m["peak_rss_mib"])
                        ]
                        if mem:
                            axes[2].plot(
                                *zip(*mem, strict=True),
                                marker=marker,
                                linestyle=linestyle,
                                color=color,
                                label=label,
                            )
        coordinates = sorted({c["x"]["value"] for c in entries})
        ticks = coordinates[:: max(1, math.ceil((len(coordinates) - 1) / 5))]
        if ticks[-1] != coordinates[-1]:
            ticks.append(coordinates[-1])
        if (
            len(ticks) > 2
            and math.log(ticks[-1] / ticks[-2])
            < math.log(coordinates[-1] / coordinates[0]) / 6
        ):
            ticks.pop(-2)
        for ax in axes:
            ax.set(xscale="log", yscale="log", xlabel=entries[0]["x"]["name"])
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        axes[0].set(ylabel="Time (s)", title="Fresh setup + requested illuminations")
        axes[1].set(ylabel="Time (s)", title="Reuse for new incident amplitudes")
        axes[2].set(
            ylabel="Process peak RSS (MiB)",
            title="Memory and retained derivative context",
        )
        axes[0].legend(fontsize=7)
        scope = []
        for key, label in [
            ("particles", "particles"),
            ("lmax", "multipole cutoff"),
            ("strength", "coupling"),
        ]:
            values = {c.get("parameters", {}).get(key) for c in entries}
            if len(values) == 1 and None not in values:
                value = next(iter(values))
                scope.append(
                    f"{label} {value}" if key == "lmax" else f"{value} {label}"
                )
        title = f"Scattering solve · varying {entries[0]['x']['name'].lower()}"
        if scope:
            title += " · " + ", ".join(scope)
        save(
            fig,
            f"illumination-{slug(series)}",
            title,
            "treams-full: public full T matrix; treams-selected: custom public coupling + SciPy LU. Native full/selected/matrix-free differ in setup and storage; upstream may be capped.",
        )


def plot_gpu(cases, save):
    import matplotlib.pyplot as plt

    for case in cases:
        if case.get("kind") != "gpu" or case["status"] not in PASS:
            continue
        warm = [m for m in case["measurements"] if m["phase"] == "warm"]
        cold = [m for m in case["measurements"] if m["phase"] == "setup / cold"]
        if not warm:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5.5, len(warm) * 0.3 + 2)))
        for ax, rows, title in [
            (axes[0], warm, "Warm boundaries · transfers named explicitly"),
            (axes[1], cold, "Setup and cold-call costs · separate boundaries"),
        ]:
            for i, m in enumerate(rows):
                color = COLORS["gpu"] if "gpu" in m["backend"] else COLORS["rust"]
                ax.barh(i, m["median"], color=color, alpha=0.8)
                if m["seconds"]:
                    ax.scatter(
                        m["seconds"],
                        [i] * len(m["seconds"]),
                        s=9,
                        color="#15202e",
                        zorder=3,
                    )
            ax.set(
                xscale="log",
                yticks=range(len(rows)),
                yticklabels=[m["backend"].replace("_", " ") for m in rows],
                xlabel="Seconds",
                title=title,
            )
        data = case["raw"]
        precision = data.get("precision", "precision unspecified")
        parameters = data.get("parameters", {})
        workload = {
            "sampling": "Cached field operator",
            "fields": "Fused plane-wave fields",
            "field-phases": "Fused-field kernel diagnostic",
        }.get(parameters.get("workload"), "CUDA comparison")
        scope = ", ".join(
            f"{label}={parameters[key]:,}"
            for key, label in [
                ("points", "P"),
                ("modes", "M"),
                ("threads", "CPU threads"),
            ]
            if key in parameters
        )
        device = (
            data.get("raw", data).get("device")
            or data.get("environment", {})
            .get("gpu_before", "GPU device unspecified")
            .split(",")[0]
        )
        save(
            fig,
            f"gpu-{slug(case['id'])}",
            f"{workload} · {scope}",
            f"{device} · {precision}. CPU baseline is native Rust, not upstream treams. Dots are recorded samples; a bar alone is a summary without retained samples. Device array bytes are not measured peak VRAM.",
        )


def plot_native_reverse(cases, save):
    import matplotlib.pyplot as plt

    rows = []
    for c in cases:
        pair = cpu_pair(c)
        reverse = next(
            (
                m
                for m in c["measurements"]
                if m["backend"] == "rust" and m["phase"] == "reverse"
            ),
            None,
        )
        if pair and reverse:
            rows.append((c, pair[1], reverse))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    grouped = defaultdict(list)
    for case, forward, reverse in rows:
        grouped[case.get("feature_group", case["family"])].append(
            reverse["median"] / forward["median"]
        )
        axes[1].scatter(
            forward["median"], reverse["median"], s=22, color=COLORS["rust"], alpha=0.6
        )
    names = sorted(grouped)
    for i, name in enumerate(names):
        axes[0].scatter(
            grouped[name],
            [i] * len(grouped[name]),
            s=22,
            color=COLORS["rust"],
            alpha=0.6,
        )
    axes[0].set(
        xscale="log",
        yticks=range(len(names)),
        yticklabels=names,
        xlabel="Reverse / recorded-forward time",
        title=f"Native pullback coverage · {len(rows)} cases",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Recorded forward time (s)",
        ylabel="Reverse time (s)",
        title="Absolute cost of one analytic pullback",
    )
    bounds = [
        min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
        max(axes[1].get_xlim()[1], axes[1].get_ylim()[1]),
    ]
    axes[1].plot(bounds, bounds, ls="--", color="#555", lw=1)
    save(
        fig,
        "native-reverse",
        "Analytic reverse across the feature suite",
        "Reverse timing excludes the fresh forward context. This is native forward/reverse overhead, not a derivative-speedup comparison against upstream.",
    )


def plot_host_comparison(suites, output, pdf):
    import matplotlib.pyplot as plt

    gallery = []
    for i, first in enumerate(suites):
        if first["historical"]:
            continue
        a = {c["id"]: c for c in first["cases"] if cpu_pair(c)}
        for second in suites[i + 1 :]:
            if second["historical"]:
                continue
            b = {c["id"]: c for c in second["cases"] if cpu_pair(c)}
            matched = [
                (a[key], b[key])
                for key in sorted(a.keys() & b.keys())
                if a[key].get("parameters") == b[key].get("parameters")
            ]
            if not matched:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
            for left, right in matched:
                p, q = cpu_pair(left), cpu_pair(right)
                axes[0].scatter(
                    speedup(*p), speedup(*q), s=18, color=COLORS["rust"], alpha=0.55
                )
                axes[1].scatter(
                    p[1]["median"],
                    q[1]["median"],
                    s=18,
                    color=COLORS["rust"],
                    alpha=0.55,
                )
                axes[2].scatter(
                    p[0]["median"],
                    q[0]["median"],
                    s=18,
                    color=COLORS["treams"],
                    alpha=0.55,
                )
            for ax, title in zip(
                axes,
                [
                    "Relative speedup on each CPU",
                    "Absolute native runtime (s)",
                    "Absolute upstream runtime (s)",
                ],
                strict=True,
            ):
                ax.set(
                    xscale="log",
                    yscale="log",
                    xlabel=environment_label(first),
                    ylabel=environment_label(second),
                    title=title,
                )
                bounds = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(bounds, bounds, color="#555", ls="--", lw=1)
            save_figure(
                fig,
                output,
                f"matched-{slug(first['name'])}-{slug(second['name'])}",
                f"Matched cases across environments · n={len(matched)}",
                "Identical case IDs and parameters only. Each axis retains its own environment/build; no cross-host timing samples are pooled. Diagonal means equal value.",
                gallery,
                pdf,
                provenance=f"X dataset: {first['name']}. Y dataset: {second['name']}. Build identities remain in each dataset's raw evidence.",
            )
    return gallery


def illumination_accuracy_records(case):
    """Normalize saved final-solve certificates, without rerunning a solver."""
    data = case.get("raw", {})
    if not data:
        return []
    parameters = {**data.get("case", {}), **case.get("parameters", {})}
    common = {
        "family": "Requested illuminations",
        "parameters": parameters,
        "x": case.get("x"),
        "independent_input_count": 1,
    }
    observations = []

    def add(
        metric,
        error,
        backend,
        reference_kind,
        tolerance=None,
        *,
        strict=False,
        **details,
    ):
        valid = finite(error) and error >= 0
        status = (
            "error"
            if not valid
            else "diagnostic"
            if not finite(tolerance)
            else "passed"
            if (error < tolerance if strict else error <= tolerance)
            else "failed"
        )
        observations.append(
            {
                **common,
                "id": f"{case['id']}/{backend}/{metric}",
                "series": case.get("series", case["id"]) + " · " + metric,
                "backend": backend,
                "reference_kind": reference_kind,
                "metric": metric,
                "error": error if valid else None,
                "tolerance": tolerance,
                "status": status,
                **details,
            }
        )

    measured = data.get("measurements", [])
    reference_method = (
        "full" if any(m["backend"] == "full" for m in measured) else "matrix-free"
    )
    native_reference = "treams-rs " + reference_method
    if "oracle_relative_error" in data:
        oracle = data["oracle_relative_error"]
        add(
            "illumination_upstream_forward_relative_l2_error",
            oracle,
            native_reference,
            "upstream",
            3e-9,
            strict=True,
            status="not_measured"
            if oracle is None and data.get("upstream_skipped_reason")
            else "error"
            if not finite(oracle)
            else "passed"
            if oracle < 3e-9
            else "failed",
            message=data.get("upstream_skipped_reason"),
            normalization="||native reference - upstream full T incident|| / ||upstream full T incident||",
        )
    comparisons = defaultdict(list)
    for key, error in data.get("comparison_relative_errors", {}).items():
        backend, phase, component = key.split("/")
        # Reference against itself, including a repeated forward mode, is not
        # independent accuracy. The upstream full comparison is already above.
        if backend not in {reference_method, "treams-full"}:
            comparisons[(backend, component)].append((phase, error))
    for (backend, component), values in comparisons.items():
        error = (
            max(value for _, value in values)
            if all(finite(value) for _, value in values)
            else None
        )
        upstream = backend.startswith("treams-")
        add(
            f"illumination_{'forward' if component == 'value' else 'gradient_' + component}_relative_l2_error",
            error,
            native_reference + " vs " + backend if upstream else "treams-rs " + backend,
            "upstream"
            if upstream
            else "native_full_t"
            if reference_method == "full"
            else "native_matrix_free",
            3e-9 if component == "value" else 3e-8,
            strict=True,
            reference_method=data.get("comparison_reference"),
            compared_phases=[phase for phase, _ in values],
            aggregation="Maximum across saved forward/recorded modes, not independent inputs",
            normalization="||method - native reference|| / max(||native reference||, 1e-15)",
            upstream_workflow="Custom upstream public coupling + SciPy selected LU"
            if upstream
            else None,
        )

    matrix_free = [m for m in measured if m["backend"] == "matrix-free"]
    for phase in ("forward", "adjoint"):
        workers = [
            m for m in matrix_free if phase == "forward" or m["phase"] == "adjoint"
        ]
        if not workers:
            continue
        reports = [
            report for m in workers for report in (m.get(phase + "_convergence") or [])
        ]
        valid = bool(reports) and all(
            isinstance(r, list) and len(r) == 3 and all(finite(v) and v >= 0 for v in r)
            for r in reports
        )
        zero_rhs_failure = valid and any(r[2] == 0 and r[1] > 0 for r in reports)
        relative = (
            max(r[1] / r[2] if r[2] else 0.0 for r in reports)
            if valid and not zero_rhs_failure
            else None
        )
        options = (
            {"status": "failed"}
            if zero_rhs_failure
            else {"status": "not_recorded"}
            if not reports
            else {}
        )
        add(
            f"illumination_{phase}_true_residual_relative",
            relative,
            "treams-rs matrix-free",
            "iterative_residual",
            parameters.get("rtol"),
            certificate_count=len(reports),
            worker_phases=[m["phase"] for m in workers],
            iterations_max=max(r[0] for r in reports) if valid else None,
            aggregation="Maximum across columns in the final saved solve from each worker phase; timing repeats are not separately archived certificates",
            normalization="||A x - b|| / ||b||; zero RHS with zero residual is 0, nonzero residual with zero RHS fails",
            stopping_criterion="||A x - b|| <= rtol ||b||, atol=0",
            **options,
        )

    for m in measured:
        numerical = m.get("numerical", {})
        for metric in (
            "translation_ward_absolute",
            "scale_ward_absolute",
            "illumination_ward_absolute",
        ):
            if metric in numerical:
                value = numerical[metric]
                add(
                    metric,
                    abs(value) if finite(value) else None,
                    "treams-rs " + m["backend"],
                    "physical_invariant",
                    signed_value=value,
                    scope="Incident multipoles held fixed. Collector checked |Ward residual| <= 1e-8 max(1, ||gradient_radii|| ||radii||); this scale was not archived, so its normalized gate cannot be reconstructed",
                )
        if "radius_direction_relative_error" in numerical:
            add(
                "illumination_radius_direction_relative_error",
                numerical["radius_direction_relative_error"],
                "treams-rs " + m["backend"],
                "finite_difference",
                2e-6,
                strict=True,
                scope="One radius direction, not a complete coordinate gradient; central step 1e-4",
                numerical_derivative=numerical.get("radius_direction_fd"),
                analytic_derivative=numerical.get("radius_direction_adjoint"),
                normalization="|FD - adjoint| / max(|adjoint|, 1e-12)",
            )
    return observations


def gpu_accuracy_records(case):
    """Keep CUDA comparison envelopes and unsaved scale-dependent gates explicit."""
    if case.get("kind") != "gpu":
        return []
    data = case.get("raw", {})
    raw = data.get("raw", data)
    parameters = {
        **{
            key: raw[key]
            for key in (
                "points",
                "modes",
                "operator_shape",
                "lossless",
                "variant",
                "real_k_specialized",
                "actual_cpu_threads",
                "selected_cpu_baseline",
                "selected_cpu_layout",
                "selected_cpu_adjoint_layout",
            )
            if key in raw
        },
        **data.get("parameters", {}),
        **case.get("parameters", {}),
    }
    workload = parameters.get("workload")
    if not workload:
        workload = (
            "field-phases"
            if "variant" in raw
            else "fields"
            if raw.get("workload") == "weighted-plane-fields"
            else "sampling"
            if "operator_shape" in raw
            else None
        )
    if workload not in {"sampling", "fields", "field-phases"}:
        return []
    common = {
        "family": case.get("family", f"GPU {workload}"),
        "backend": "treams-rs-cpu-and-cuda"
        if workload == "sampling"
        else "treams-rs-gpu",
        "parameters": parameters,
        "case_status": case.get("status"),
        "reported_scope": {
            key: raw[key] for key in ("scope", "adjoint_scope") if key in raw
        },
    }
    for context in (raw, data, case):
        common.update(
            {
                key: context[key]
                for key in (
                    "source",
                    "environment",
                    "precision",
                    "comparison",
                    "selection",
                    "provenance",
                )
                if key in context
            }
        )
    if data is not raw and "scope" in data:
        common["wrapper_scope"] = data["scope"]
    if workload == "sampling":
        metrics = [
            (
                "max_abs_error",
                "forward",
                "native_cpu",
                None,
                "Worst checked forward discrepancy across prepared/cached CPU and CUDA variants versus native-core fused fields; not a GPU-only residual.",
            ),
            (
                "max_adjoint_abs_error",
                "coefficient-adjoint",
                "native_cpu",
                None,
                "Worst checked adjoint discrepancy across CPU row-layout and CUDA variants versus CPU column-layout F^H g; fixed-operator coefficients only.",
            ),
            (
                "max_adjoint_normalized_pairing_error",
                "coefficient-adjoint-pairing",
                "analytic",
                2e-12,
                "Worst CPU/CUDA Hermitian pairing residual: abs(g^H(Fc) - gradient^H c) / max(1, norm(g) * norm(Fc)); fixed-operator coefficients only.",
            ),
        ]
    else:
        metrics = [
            (
                "max_abs_error",
                "forward-diagnostic" if workload == "field-phases" else "forward",
                "native_cpu",
                None,
                "Selected diagnostic CUDA kernel variant's final warm output versus native-core fields; not the production API."
                if workload == "field-phases"
                else "First production CUDA field evaluation versus native-core fused fields; maximum complex component discrepancy.",
            ),
        ]
    observations = []
    for metric, phase, reference, tolerance, scope in metrics:
        if metric not in raw:
            continue
        value = raw[metric]
        valid = finite(value) and value >= 0
        observations.append(
            {
                **common,
                "id": f"{case['id']}/{metric}",
                "metric": metric,
                "phase": phase,
                "reference_kind": reference,
                "error": value if valid else None,
                "tolerance": tolerance,
                "status": "error"
                if not valid
                else "diagnostic"
                if tolerance is None
                else "passed"
                if value <= tolerance
                else "failed",
                "source_field": ("raw." if data is not raw else "") + metric,
                "scope": scope
                + (
                    " Reference magnitudes were not retained, so the scale-dependent gate and relative error cannot be reconstructed."
                    if tolerance is None
                    else ""
                ),
            }
        )
    return observations


def accuracy_records(cases):
    """Collect measured residuals; timing success is never accuracy evidence."""
    records = []
    for case in cases:
        data = case.get("raw", {})
        observations = (
            list(data.get("observations", [])) if case.get("kind") == "accuracy" else []
        )
        if case.get("kind") == "illumination":
            observations = illumination_accuracy_records(case)
        elif case.get("kind") == "gpu":
            observations = gpu_accuracy_records(case)
        if case.get("kind") == "gradient":
            validation = data.get("validation", {})
            common = {
                "id": case["id"],
                "family": case["family"],
                "backend": "treams-rs",
                "reference_kind": "finite_difference",
                "parameters": case.get("parameters", {}),
                "status": "diagnostic",
                "validation_passed": validation.get("passed"),
                "selected_fd_step": validation.get("selected_fd_step"),
            }
            tolerance = data.get("case", {}).get("gradient_rtol")

            def checked(
                metric,
                error,
                threshold,
                *,
                common=common,
                observations=observations,
                **details,
            ):
                status = (
                    "error"
                    if not finite(error)
                    else "diagnostic"
                    if not finite(threshold)
                    else "passed"
                    if error <= threshold
                    else "failed"
                )
                observations.append(
                    {
                        **common,
                        "metric": metric,
                        "error": error,
                        "tolerance": threshold,
                        "status": status,
                        **details,
                    }
                )

            if "gradient_relative_error" in validation:
                checked(
                    "gradient_selected_relative_l2_error",
                    validation["gradient_relative_error"],
                    tolerance,
                    scope="Selected full-coordinate gradient check",
                )
            for step in validation.get("fd_step_sweep", []):
                observations.append(
                    {
                        **common,
                        "metric": "gradient_relative_l2_error",
                        "error": step.get("gradient_relative_error"),
                        "series": case["id"] + " · full coordinate FD",
                        "x": {
                            "name": "Finite-difference step",
                            "value": step["step"],
                            "unit": "relative",
                        },
                        "scope": "Full-coordinate step sweep"
                        if validation.get("full_coordinate_step_sweep")
                        else "Single full-coordinate check after independent directional step selection",
                    }
                )
            for step in validation.get("directional_checks", []):
                observations.append(
                    {
                        **common,
                        "metric": "directional_relative_error",
                        "error": step.get("relative_error"),
                        "series": case["id"] + " · directional FD",
                        "x": {
                            "name": "Finite-difference step",
                            "value": step["step"],
                            "unit": "relative",
                        },
                    }
                )
            for block, error in validation.get(
                "gradient_block_relative_errors", {}
            ).items():
                checked(
                    "gradient_block_relative_l2_error",
                    error,
                    tolerance,
                    id=case["id"] + " · " + block,
                )
            if validation.get("linear_adjoint_relative_error") is not None:
                checked(
                    "linear_adjoint_relative_l2_error",
                    validation["linear_adjoint_relative_error"],
                    3e-9,
                    reference_kind="analytic",
                )
            for name, error in validation.get("invariants", {}).items():
                checked(
                    name,
                    error,
                    3e-8,
                    reference_kind="physical_invariant",
                    scope="Recorded gradient symmetry residual; collector tolerance 3e-8",
                )
        records.extend(
            {
                **observation,
                "suite": case["suite"],
                "case_id": case["id"],
                "fingerprint": case["fingerprint"],
                "raw_link": case.get("raw_link", ""),
            }
            for observation in observations
        )
    return records


def common_tolerance(rows):
    tolerances = {r.get("tolerance") for r in rows}
    value = next(iter(tolerances)) if len(tolerances) == 1 else None
    return value if finite(value) and value > 0 else None


def native_qualification_note(case):
    data = case.get("raw", {})
    proof = data.get("validation", {}).get("independent_reference", {})
    if proof.get("passed") is True:
        return (
            "Independent native qualification passed for the recorded upstream "
            "disagreements. Original discrepancies and the proof remain in validation."
        )
    certificates = [
        row
        for row in data.get("observations", [])
        if row.get("reference_kind") == "independently_balanced_upstream_system"
        and row.get("metric") == "certified_reference_max_scaled_error"
    ]
    if (
        data.get("native_reference_passed") is True
        and certificates
        and all(
            row.get("status") == "passed"
            and row.get("certificate", {}).get("passed") is True
            for row in certificates
        )
    ):
        return (
            "Native response-matrix qualification passed for the recorded certified "
            "cutoffs of the encoded interaction system. Original upstream comparison "
            "failures remain visible; they are distinct from this native qualification."
        )
    return ""


def accuracy_reference_note(reference, rows, qualified_case_ids):
    if reference == "independently_balanced_upstream_system":
        return (
            " Native response error plus certified reference uncertainty is compared "
            "with the unchanged elementwise tolerance. The independently balanced "
            "upstream system, arithmetic enclosure, Neumann inverse bound and selected "
            "80/120-digit residual checks are retained in the evidence ledger. This "
            "certifies the encoded linear system, not physical truncation accuracy."
        )
    if reference == "upstream" and any(
        row["case_id"] in qualified_case_ids for row in rows
    ):
        return (
            " Original upstream disagreements remain shown. Passing independent "
            "native response-matrix certificates for this collector are shown "
            "separately in the evidence ledger. Upstream agreement and native "
            "qualification are distinct checks."
        )
    return ""


def error_axis(ax, values, axis="x", gate=None):
    from matplotlib.ticker import FixedLocator, LogFormatterSciNotation, NullFormatter

    positive = [v for v in values if finite(v) and v > 0]
    threshold = min(positive) / 10 if positive else 1e-16
    # Symmetric-log's linear region retains real zero measurements without
    # replacing them by a fabricated epsilon error.
    anchors = [0.0, gate] if finite(gate) and gate > 0 else [0.0]
    upper = min(max([*positive, *anchors]) or 1e-16, sys.float_info.max / 10) * 10
    getattr(ax, f"set_{axis}lim")(0, upper)
    getattr(ax, f"set_{axis}scale")("symlog", linthresh=max(threshold, 1e-300))
    coordinate = getattr(ax, f"{axis}axis")
    transform = coordinate.get_transform()
    displayed_upper = transform.transform(upper)
    # Suppress crowded near-zero labels, not measurements. Directly setting one
    # decade of headroom avoids overflowing the inverse scale on stress grids.
    candidates = [
        tick
        for tick in coordinate.get_majorticklocs()
        if finite(tick) and 0 < tick <= upper
    ]
    # Keep a high-scale label so failed gates far above one remain interpretable.
    if candidates and all(
        abs(transform.transform(candidates[-1]) - transform.transform(other))
        / displayed_upper
        >= 0.08
        for other in anchors
    ):
        anchors.append(candidates[-1])
    stride = max(1, math.ceil(len(candidates) / (9 - len(anchors))))
    for tick in candidates[::stride]:
        if all(
            abs(transform.transform(tick) - transform.transform(other))
            / displayed_upper
            >= 0.08
            for other in anchors
        ):
            anchors.append(tick)
    coordinate.set_major_locator(FixedLocator(sorted(anchors)))
    coordinate.set_major_formatter(
        LogFormatterSciNotation(minor_thresholds=(math.inf, math.inf))
    )
    coordinate.set_minor_formatter(NullFormatter())


def raw_error_metric(metric):
    """Keep raw-unit residuals apart from the collectors' normalized errors."""
    if metric in {
        "layer_split_scaled_absolute",
        "divergence_scaled_absolute",
        "ewald_split_scaled_absolute",
    }:
        return False
    return (
        "absolute" in metric
        or "_abs_" in metric
        or metric
        in {"passivity_nonnegative_violation", "scattering_exceeds_extinction"}
    )


def accuracy_availability(rows):
    """Describe each metric's finite plotting denominator without changing data."""
    notes = []
    for backend in sorted({r["backend"] for r in rows}):
        selected = [r for r in rows if r["backend"] == backend]
        missing = [r for r in selected if not finite(r.get("error")) or r["error"] < 0]
        reasons = Counter()
        for row in missing:
            conditioning = row.get("conditioning", {})
            if conditioning.get("float64_no_overflow") is False:
                reason = "reference overflow"
            elif conditioning.get("reference_stable") is False:
                reason = "unstable reference"
            elif row.get("error_high_precision") is not None:
                reason = "high-precision-only residual"
            elif row.get("status") == "error":
                reason = "evaluation error"
            else:
                reason = "unavailable residual"
            reasons[reason] += 1
        note = f"{backend}: {len(selected) - len(missing)}/{len(selected)} finite"
        if reasons:
            note += (
                " (omitted "
                + ", ".join(f"{n} {reason}" for reason, n in sorted(reasons.items()))
                + ")"
            )
        notes.append(note)
    return "; ".join(notes)


def draw_accuracy_curve(ax, rows, reference, metric, x_name):
    for i, backend in enumerate(sorted({r["backend"] for r in rows})):
        selected = sorted(
            (r for r in rows if r["backend"] == backend), key=lambda r: r["x"]["value"]
        )
        xs, ys = [r["x"]["value"] for r in selected], [r["error"] for r in selected]
        ax.plot(
            xs,
            ys,
            marker=["o", "s", "^", "D"][i % 4],
            ms=4,
            color=backend_color(backend, i),
            label=backend,
            linestyle="-" if len(set(xs)) == len(xs) else "none",
        )
        failed = [r for r in selected if r.get("status") == "failed"]
        if failed:
            ax.scatter(
                [r["x"]["value"] for r in failed],
                [r["error"] for r in failed],
                marker="x",
                s=65,
                color=backend_color(backend, i),
                zorder=4,
            )
    tolerance = common_tolerance(rows)
    error_axis(ax, [r["error"] for r in rows], "y", tolerance)
    if all(r["x"]["value"] > 0 for r in rows) and (
        "step" in x_name.lower() or "tolerance" in x_name.lower()
    ):
        ax.set_xscale("log")
    if tolerance is not None:
        ax.axhline(tolerance, color="#555", ls="--", lw=1, label="Recorded tolerance")
    ax.set(
        xlabel=x_name,
        ylabel="Error / tolerance"
        if metric.endswith("max_scaled_error")
        else metric.replace("_", " "),
    )
    ax.legend(fontsize=8)


def finite_difference_pages(curves):
    groups = defaultdict(list)
    for (reference, metric, series, x_name), rows in sorted(curves.items()):
        if reference == "finite_difference" and "step" in x_name.lower():
            groups[(rows[0]["family"], metric, x_name)].append((series, rows))
    return [
        (family, metric, x_name, start // 6 + 1, entries[start : start + 6])
        for (family, metric, x_name), entries in sorted(groups.items())
        for start in range(0, len(entries), 6)
    ]


def plot_accuracy(cases, save):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import NullFormatter, StrMethodFormatter

    records = accuracy_records(cases)
    qualified_case_ids = {c["id"] for c in cases if native_qualification_note(c)}
    groups = defaultdict(list)
    for row in records:
        # Counts and condition estimates are diagnostic context, not error norms.
        if row["metric"] in {
            "reciprocal_condition_1_estimate",
            "native_tiny_upstream_nonzero_entry_count",
        }:
            continue
        absolute = raw_error_metric(row["metric"])
        if absolute and row["reference_kind"] == "upstream":
            # Absolute units differ across hundreds of operations. Preserve all
            # raw values/gates in the ledger; plot the comparable scaled errors.
            continue
        # Absolute errors have different physical units and output scales.
        # Only normalized/relative metrics get a cross-family distribution.
        family = family_for(row["family"]) if absolute else ""
        groups[
            (
                row["reference_kind"],
                row["metric"],
                family,
                row.get("selection", "collector_scope"),
            )
        ].append(row)
    for (reference, metric, absolute_family, selection), all_rows in sorted(
        groups.items()
    ):
        rows = [r for r in all_rows if finite(r.get("error")) and r["error"] >= 0]
        if not rows:
            continue
        families = sorted({family_for(r["family"]) for r in rows})
        fig, axes = plt.subplots(
            1, 2, figsize=(13, max(5.8, len(families) * 0.38 + 2.6))
        )
        backends = sorted({r["backend"] for r in rows})
        for i, backend in enumerate(backends):
            selected = [r for r in rows if r["backend"] == backend]
            errors = np.sort([r["error"] for r in selected])
            color = backend_color(backend, i)
            axes[0].step(
                errors,
                np.arange(1, len(errors) + 1) / len(errors),
                where="post",
                color=color,
                label=f"{backend} · {len(errors)}/{sum(r['backend'] == backend for r in all_rows)} finite observations",
                lw=1.8,
            )
            for j, family in enumerate(families):
                points = [r for r in selected if family_for(r["family"]) == family]
                if points:
                    axes[1].scatter(
                        [r["error"] for r in points],
                        j + np.linspace(-0.15, 0.15, len(points)),
                        s=16,
                        alpha=0.6,
                        color=color,
                    )
            failed = [r for r in selected if r.get("status") == "failed"]
            if failed:
                axes[1].scatter(
                    [r["error"] for r in failed],
                    [families.index(family_for(r["family"])) for r in failed],
                    marker="x",
                    s=48,
                    color=color,
                    zorder=4,
                )
        for ax in axes:
            tolerance = common_tolerance(rows)
            error_axis(ax, [r["error"] for r in rows], gate=tolerance)
            ax.set_xlabel(
                metric.replace("_", " ")
                + (" · raw workload units" if absolute_family else " · lower is better")
            )
            if tolerance is not None:
                ax.axvline(tolerance, ls="--", color="#555", lw=1)
        axes[0].set(
            ylabel="Fraction of available finite error observations",
            ylim=(0, 1.02),
            title="Measured residual distribution",
        )
        axes[0].legend(fontsize=8, loc="lower right")
        axes[1].set(
            yticks=range(len(families)),
            yticklabels=families,
            title="Feature coverage · failed gates marked x",
        )
        exact_zeros = sum(r["error"] == 0 for r in rows)
        scale_note = (
            "Absolute errors retain workload-dependent units and magnitudes; compare this family separately, not as one accuracy score."
            if absolute_family
            else "Metric is dimensionless; relative error near zero needs the accompanying absolute-error and conditioning evidence."
        )
        save(
            fig,
            f"accuracy-{slug(reference)}-{slug(metric)}{('-' + slug(absolute_family)) if absolute_family else ''}-{slug(selection)}",
            f"Accuracy: {reference.replace('_', ' ')} · {metric.replace('_', ' ')}{(' · ' + absolute_family) if absolute_family else ''}{' · post-failure diagnostic' if selection == 'post_failure_diagnostic' else ''}",
            f"{exact_zeros} floating-point zeros remain zero. {scale_note} {accuracy_availability(all_rows)}. CDFs condition on finite residuals; backend cohorts may differ. Dashed line: common tolerance."
            + (
                " Selected after observed failures; excluded from the predeclared grid."
                if selection == "post_failure_diagnostic"
                else ""
            )
            + accuracy_reference_note(reference, all_rows, qualified_case_ids),
        )
    high_orders = defaultdict(list)
    for row in records:
        if (
            row["reference_kind"] != "high_precision"
            or row["metric"] != "max_scaled_error"
        ):
            continue
        parameters = row.get("parameters", {})
        coordinate = "degree" if "degree" in parameters else "order"
        if finite(parameters.get(coordinate)):
            high_orders[
                (row["family"], row.get("selection", "collector_scope"), coordinate)
            ].append(row)
    for (family, selection, coordinate), all_rows in sorted(high_orders.items()):
        rows = [r for r in all_rows if finite(r.get("error")) and r["error"] >= 0]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.8))
        for i, backend in enumerate(sorted({r["backend"] for r in rows})):
            selected = [r for r in rows if r["backend"] == backend]
            ax.scatter(
                [r["parameters"][coordinate] for r in selected],
                [r["error"] for r in selected],
                s=20,
                alpha=0.55,
                marker="s" if backend == "treams" else "o",
                color=backend_color(backend, i),
                label=f"{backend} · {len(selected)}/{sum(r['backend'] == backend for r in all_rows)} finite",
            )
        error_axis(ax, [r["error"] for r in rows], "y", 1)
        coordinates = sorted({r["parameters"][coordinate] for r in rows})
        if coordinates[0] > 0:
            ax.set_xscale("log")
        else:
            ax.set_xscale("symlog", linthresh=1)
            if coordinates[0] == 0:
                ax.set_xlim(left=0)
        stride = max(1, math.ceil(len(coordinates) / 8))
        ticks = sorted({*coordinates[::stride], coordinates[-1]})
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axhline(1, color="#555", ls="--", lw=1)
        ax.set(
            xlabel=coordinate.capitalize(),
            ylabel="Maximum tolerance-scaled error",
            title=family.replace("_", " "),
        )
        ax.legend(fontsize=8)
        save(
            fig,
            f"high-order-{slug(family)}-{slug(selection)}-{coordinate}",
            "High-order accuracy"
            + (
                " · post-failure diagnostic"
                if selection == "post_failure_diagnostic"
                else ""
            ),
            "Points only: arguments and materials also vary, so this is not a convergence curve. Error 1 is the mixed-tolerance bound. Axes are logarithmic with linear regions near zero. "
            + accuracy_availability(all_rows)
            + (
                ". Selection follows observed failures."
                if selection == "post_failure_diagnostic"
                else "."
            ),
        )
    curves = defaultdict(list)
    for row in records:
        if row["metric"] == "native_tiny_upstream_nonzero_entry_count":
            continue
        if (
            row.get("series")
            and row.get("x")
            and finite(row.get("error"))
            and row["error"] >= 0
        ):
            curves[
                (row["reference_kind"], row["metric"], row["series"], row["x"]["name"])
            ].append(row)
    for (reference, metric, series, x_name), rows in sorted(curves.items()):
        if reference == "finite_difference" and "step" in x_name.lower():
            continue
        if len({r["x"]["value"] for r in rows}) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.8))
        draw_accuracy_curve(ax, rows, reference, metric, x_name)
        conditioning = metric == "reciprocal_condition_1_estimate"
        ax.set(
            title="Conditioning of the interaction system"
            if conditioning
            else "Independently bounded linear system"
            if reference == "independently_balanced_upstream_system"
            else "Reference: upstream treams"
            if reference == "upstream"
            else reference.replace("_", " "),
        )
        title_kind = (
            "Conditioning"
            if conditioning
            else "Numerical qualification"
            if reference == "independently_balanced_upstream_system"
            else "Reference agreement"
            if reference == "upstream"
            else "Convergence"
        )
        save(
            fig,
            f"{'conditioning' if conditioning else 'convergence'}-{slug(reference)}-{slug(metric)}-{slug(series)}",
            f"{title_kind}: {series}",
            (
                "Lower reciprocal condition estimates mean a more ill-conditioned system. This diagnostic has no accuracy pass gate; it provides context for residuals and upstream disagreements. The cutoff sweep was selected after observed reference-check failures."
                if conditioning
                else "Worst archived per-column true residual divided by RHS norm; dashed line is the requested stopping tolerance. Algebraic convergence is not a bound on truncation error or physical accuracy. These are final saved certificates, not all timing repeats."
                if reference == "iterative_residual"
                else "Dashed line: acceptance bound 1."
                if reference == "independently_balanced_upstream_system"
                else "Measured comparisons against upstream."
                if reference == "upstream"
                else "Measured points only; zero residuals remain zero. A refined numerical solution measures cutoff/tolerance sensitivity, not proof of exactness. Finite-difference step sweeps expose truncation and roundoff effects."
            )
            + accuracy_reference_note(reference, rows, qualified_case_ids),
        )
    for family, metric, x_name, page, panels in finite_difference_pages(curves):
        nrows = math.ceil(len(panels) / 2)
        fig, axes = plt.subplots(
            nrows, 2, figsize=(13, 3.1 * nrows + 1.5), squeeze=False
        )
        for ax, (_, rows) in zip(axes.flat, panels, strict=False):
            draw_accuracy_curve(ax, rows, "finite_difference", metric, x_name)
            parameters = rows[0].get("parameters", {})
            labels = [
                ("particles", "N"),
                ("lmax", "L"),
                ("samples", "samples"),
                ("parameter_count", "parameters"),
                ("threads", "threads"),
            ]
            title = (
                ", ".join(
                    f"{label}={parameters[key]}"
                    for key, label in labels
                    if key in parameters
                )
                or rows[0]["case_id"]
            )
            scope = (
                "directional step sweep"
                if metric == "directional_relative_error"
                else "full-coordinate step sweep"
                if rows[0].get("scope") == "Full-coordinate step sweep"
                else "selected full-coordinate check"
            )
            ax.set_title(title + " · " + scope, fontsize=10, loc="left")
            ax.set_ylabel(
                "Relative gradient error"
                if metric != "directional_relative_error"
                else "Relative directional error"
            )
        for ax in list(axes.flat)[len(panels) :]:
            ax.set_visible(False)
        save(
            fig,
            f"finite-difference-{slug(family)}-{slug(metric)}-page-{page}",
            f"Derivative checks: {family} · page {page}",
            "Each panel is a separate input with its own diagnostic step sweep or selected full-coordinate check. No unrelated series are pooled. All recorded points and scopes remain in the ledger; selected-gradient acceptance gates are reported separately.",
        )


def plot_paper_spectra(cases, save):
    import matplotlib.pyplot as plt
    import numpy as np

    wanted = {
        "ebeam-sphere-author-notebook": ("Electron beam / sphere", 1),
        "ebeam-cylinder-author-notebook": ("Electron beam / cylinder", 1),
        "cpc-sphere_array_above_slab": ("Periodic sphere array above a slab", 0),
    }
    curves = [
        curve
        for case in cases
        for curve in case.get("raw", {}).get("paper_curves", [])
        if curve["id"] in wanted
    ]
    if not curves:
        return
    fig, axes = plt.subplots(
        1, len(curves), figsize=(4.8 * len(curves), 5.8), squeeze=False
    )
    for ax, curve in zip(axes.flat, curves, strict=True):
        title, column = wanted[curve["id"]]
        x = curve.get("plot_grid", curve["grid"])
        for key, label, color, style in [
            ("native", "treams-rs", COLORS["rust"], "-"),
            ("upstream", "treams", COLORS["treams"], "--"),
        ]:
            if curve.get(key):
                ax.plot(
                    x,
                    np.asarray(curve[key], dtype=float)[:, column],
                    color=color,
                    ls=style,
                    lw=1.8,
                    label=label,
                )
        for reference in curve.get("references", []):
            ax.plot(
                x,
                np.asarray(reference["values"], dtype=float)[:, column],
                color="#252a31",
                lw=0.8,
                ls=":",
                marker="o",
                markevery=max(1, len(x) // 24),
                ms=3,
                markerfacecolor="white",
                label=reference["label"],
            )
        ax.set(
            xlabel=curve["x_label"],
            ylabel=f"{curve['quantity'][column]} ({curve['unit']})",
            title=title,
        )
        ax.legend(fontsize=7)
    save(
        fig,
        "paper-spectra",
        "Published workflows: spectra on matching calculation grids",
        "Electron-beam overlays use their same-grid executed author notebooks (corrected cylinder notebook explicitly labelled). Periodic-array comparison is the upstream recipe; exact diffraction endpoints are excluded. Overlap is not an error bound.",
    )


def accuracy_html(cases):
    rows = accuracy_records(cases)
    if not rows:
        return ""
    counts = Counter(r.get("status", "unspecified") for r in rows)
    link = f"accuracy-{slug(rows[0]['suite'])}.html"
    coverage = Counter()
    reference_strata = Counter()
    reference_backend_status = defaultdict(Counter)
    underflow = Counter()
    reference_selection = Counter()
    for case in cases:
        data = case.get("raw", {})
        if case.get("kind") == "gradient" and data.get("validation"):
            coverage["gradient input definitions"] += 1
        elif case.get("kind") == "illumination" and data.get("measurements"):
            coverage["requested-illumination input definitions"] += 1
        elif case.get("kind") == "gpu" and data:
            coverage["CPU/CUDA benchmark input definitions"] += 1
        elif case.get("kind") != "accuracy":
            continue
        elif data.get("cases"):
            coverage[
                "high-precision input definitions (each compared for both backends)"
            ] += len(data["cases"])
            for reference in data["cases"]:
                reference_selection[reference.get("selection", "predeclared_grid")] += 1
                if (
                    reference.get("reference_stable") is not True
                    or reference.get("float64_no_overflow") is None
                ):
                    reference_strata["unstable or unavailable references"] += 1
                elif reference["float64_no_overflow"] is False:
                    reference_strata["references above float64 range"] += 1
                else:
                    reference_strata["stable references without float64 overflow"] += 1
                    for backend in ("treams-rs", "treams"):
                        reference_backend_status[backend][
                            reference.get("backends", {})
                            .get(backend, {})
                            .get("status", "error")
                        ] += 1
                    for flag, label in [
                        ("float64_subnormal_components", "subnormal components"),
                        (
                            "float64_below_min_subnormal_components",
                            "components below the smallest subnormal",
                        ),
                        (
                            "float64_rounds_to_zero_components",
                            "components rounding to zero",
                        ),
                    ]:
                        underflow[label] += reference.get(flag, 0) > 0
        elif data.get("paper_curves"):
            coverage["primary spectrum samples per backend"] += sum(
                len(curve["grid"]) for curve in data["paper_curves"]
            )
            coverage["additional native refinement samples"] += sum(
                len(curve.get("convergence", {}).get("refined_native", []))
                for curve in data["paper_curves"]
            )
        elif data.get("suite") == "physical identities and convergence":
            coverage["physics input definitions"] += len(
                {r["id"].split("/")[0] for r in data.get("observations", [])}
            )
        elif any(
            r.get("reference_kind") == "upstream" for r in data.get("observations", [])
        ):
            coverage["upstream comparison commands"] += 1
    inputs = "; ".join(f"{count} {label}" for label, count in coverage.items())
    reference_summary = ""
    if reference_strata:
        inputs += "; " + ", ".join(
            f"{n} {selection.replace('_', ' ')} reference inputs"
            for selection, n in sorted(reference_selection.items())
        )
        strata = "; ".join(
            f"{reference_strata[label]} {label}"
            for label in (
                "stable references without float64 overflow",
                "references above float64 range",
                "unstable or unavailable references",
            )
        )
        statuses = "; ".join(
            f"{backend}: "
            + ", ".join(
                f"{state} {states[state]}" for state in ("passed", "failed", "error")
            )
            for backend, states in sorted(reference_backend_status.items())
        )
        tiny = "; ".join(
            f"{count} inputs with {label}" for label, count in underflow.items()
        )
        reference_summary = f"<p><strong>Reference input strata:</strong> {esc(strata)}. Backend statuses on stable references without float64 overflow only: {esc(statuses)}. Within that stratum, overlapping tiny-value subsets: {esc(tiny)}. These remain part of absolute/tolerance-scaled qualification; a mixed-tolerance pass does not certify relative accuracy for tiny values. Original ledger statuses remain unchanged.</p>"
    for case in cases:
        note = native_qualification_note(case)
        if note:
            evidence = (
                f' <a href="{esc(case["raw_link"])}">Raw qualification evidence</a>.'
                if case.get("raw_link")
                else ""
            )
            reference_summary += (
                f"<p><strong>{esc(case['id'])}:</strong> {esc(note)}{evidence}</p>"
            )
    return f'<h3>Accuracy evidence ledger</h3><p>{esc(inputs)}.</p>{reference_summary}<p><strong>{len(rows)} metric observations · observation statuses {esc(dict(counts))}</strong>. Multiple metrics and backends describe one input; these are not independent test-case or pass counts. A mixed absolute/relative tolerance pass does not imply a fixed number of relative digits. Diagnostic rows have no implied pass gate; counts and condition estimates stay in the ledger as context, outside error distributions. Upstream absolute errors remain fully available here and in CSV; their distributions are omitted because their units differ across operations. Scaled/relative plots retain failed gates, and targeted absolute-error curves remain plotted.</p><p><a href="{link}">Open searchable accuracy ledger</a> · <a href="accuracy.csv">Download complete CSV</a> · <a href="accuracy-observations.json">Download complete JSON</a>. The separate ledger shows at most 100 matches per page; all failures, diagnostics and undefined values remain searchable.</p>'


def write_accuracy_pages(suites, output):
    template = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Accuracy ledger — __SUITE__</title><style>
    body{font:15px/1.5 system-ui,sans-serif;color:#192738;background:#f5f7fa;max-width:1400px;margin:auto;padding:24px}a{color:#2166ac}input,button{font:inherit;padding:9px;margin:5px;border:1px solid #bdcbd7;border-radius:4px}input{width:min(550px,90%)}table{border-collapse:collapse;width:100%;background:white;font-size:.85rem}th,td{padding:9px;text-align:left;border-bottom:1px solid #d8e0e8;vertical-align:top}details{min-width:220px;max-width:600px}summary{cursor:pointer}pre{font-size:.75rem;white-space:pre-wrap;overflow-wrap:anywhere;max-height:450px;overflow:auto}.failed,.error{color:#a12420;font-weight:700}.table{overflow:auto}
    </style><p><a href="index.html">← Benchmark report</a> · <a href="accuracy.csv">Complete accuracy CSV</a> · <a href="accuracy-observations.json">Complete accuracy JSON</a></p><h1>Accuracy ledger</h1><h2>__SUITE__</h2><p>Search all observations; only 100 matching rows enter the page at once. Expand a row for parameters, reference precision, conditioning and the raw evidence link. Error metrics and reference types remain separate.</p><label>Filter <input id="query" placeholder="Function, backend, metric, status, parameter…"></label><div><button id="previous" type="button">Previous</button><button id="next" type="button">Next</button><span id="count" role="status" aria-live="polite"></span></div><div class="table"><table><thead><tr><th>Observation and details</th><th>Backend</th><th>Reference</th><th>Metric</th><th>Error</th><th>Status</th></tr></thead><tbody></tbody></table></div><script type="application/json" id="records">__DATA__</script><script>
    const rows=JSON.parse(document.querySelector('#records').textContent);
    const search=rows.map(r=>[r.id,r.family,r.backend,r.reference_kind,r.metric,r.status,JSON.stringify(r.parameters)].join(' ').toLowerCase());
    const body=document.querySelector('tbody'),query=document.querySelector('#query'),previous=document.querySelector('#previous'),next=document.querySelector('#next'),count=document.querySelector('#count');
    let matches=rows.map((r,i)=>i),page=0;
    function render(){
      body.replaceChildren();
      for(const i of matches.slice(page*100,(page+1)*100)){
        const r=rows[i],tr=document.createElement('tr'),td=document.createElement('td'),details=document.createElement('details'),summary=document.createElement('summary');
        summary.textContent=r.id;details.append(summary);td.append(details);tr.append(td);
        details.addEventListener('toggle',()=>{if(details.open&&details.childElementCount===1){const pre=document.createElement('pre');pre.textContent=JSON.stringify(r,null,2);details.append(pre);if(r.raw_link){const link=document.createElement('a');link.href=r.raw_link;link.textContent='Raw JSON';details.append(link);}}});
        for(const [name,value] of [['backend',r.backend],['reference',r.reference_kind],['metric',r.metric],['error',Number.isFinite(r.error)?r.error.toExponential(5):(r.error_high_precision??'not measured')],['status',r.status]]){const cell=document.createElement('td');cell.textContent=value??'';if(name==='status')cell.className=r.status??'';tr.append(cell);}
        body.append(tr);
      }
      previous.disabled=page===0;next.disabled=(page+1)*100>=matches.length;
      count.textContent=`${matches.length? page*100+1:0}-${Math.min((page+1)*100,matches.length)} of ${matches.length} matches (${rows.length} total observations)`;
    }
    query.addEventListener('input',()=>{const q=query.value.toLowerCase();matches=search.flatMap((text,i)=>text.includes(q)?[i]:[]);page=0;render();});
    previous.addEventListener('click',()=>{page--;render();});next.addEventListener('click',()=>{page++;render();});render();
    </script></html>"""
    for suite in suites:
        rows = accuracy_records(suite["cases"])
        if rows:
            # Escape '<' inside inert JSON so source data cannot close its script.
            data = json.dumps(rows, separators=(",", ":")).replace("<", "\\u003c")
            page = template.replace("__SUITE__", esc(suite["name"])).replace(
                "__DATA__", data
            )
            (output / f"accuracy-{slug(suite['name'])}.html").write_text(page)


def write_ledger(suites, output):
    fields = [
        "suite",
        "id",
        "kind",
        "family",
        "tier",
        "status",
        "fingerprint",
        "backend",
        "phase",
        "threads",
        "median",
        "peak_rss_mib",
        "baseline_rss_mib",
        "growth_rss_mib",
        "seconds",
        "raw",
    ]
    with (output / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for suite in suites:
            for case in suite["cases"]:
                for m in case["measurements"] or [{}]:
                    row = {k: case.get(k, "") for k in fields}
                    row.update({k: v for k, v in m.items() if k in fields})
                    row["raw"] = case.get("raw_link", "")
                    row["seconds"] = json.dumps(m.get("seconds", []))
                    writer.writerow(row)
    accuracy = accuracy_records([c for suite in suites for c in suite["cases"]])
    (output / "accuracy-observations.json").write_text(
        json.dumps(accuracy, indent=2) + "\n"
    )
    with (output / "accuracy.csv").open("w", newline="") as file:
        fields = [
            "suite",
            "case_id",
            "id",
            "family",
            "backend",
            "reference_kind",
            "metric",
            "error",
            "error_high_precision",
            "tolerance",
            "status",
            "fingerprint",
            "raw_link",
            "parameters",
            "x",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for observation in accuracy:
            writer.writerow(
                {
                    k: json.dumps(observation[k])
                    if isinstance(observation.get(k), (dict, list))
                    else observation.get(k)
                    for k in fields
                }
            )


def esc(value):
    return html.escape(str(value), quote=True)


def figure_html(gallery):
    return "".join(
        f'<figure><a href="{g["name"]}.svg"><img loading="lazy" src="{g["name"]}.png" alt="{esc(g["title"])}"></a><figcaption>{esc(g["description"])} <a href="{g["name"]}.svg">SVG</a> · <a href="{g["name"]}.pdf">PDF</a>{("<br><small>" + esc(g["provenance"]) + "</small>") if g.get("provenance") else ""}</figcaption></figure>'
        for g in gallery
    )


def write_html(suites, galleries, output, cross_gallery):
    sections = []
    for suite, gallery in zip(suites, galleries, strict=True):
        cases = suite["cases"]
        states = Counter(c["status"] for c in cases)
        pairs = [cpu_pair(c) for c in cases if cpu_pair(c)]
        observed, bounded = slowdown_summary(cases)
        figures = figure_html(gallery) + accuracy_html(cases)
        rows = []
        for c in cases:
            pair = cpu_pair(c)
            ratio = f"{speedup(*pair):.3f}x" if pair else "—"
            if pair and speedup(*pair) < 1:
                if min(len(m["seconds"]) for m in pair) >= 3:
                    _, lo, hi = ratio_interval(*pair, c.get("paired", False))
                    ratio += f" · 95% [{lo:.3f}, {hi:.3f}] · " + (
                        "interval below 1"
                        if hi < 1
                        else "median slowdown; interval crosses 1"
                    )
                else:
                    ratio += " · observed median slowdown; interval unavailable"
            detail = esc(
                json.dumps(
                    {
                        "parameters": c.get("parameters"),
                        "telemetry": c.get("telemetry"),
                        "command": c.get("command"),
                        "validation": c.get("raw", {}).get("validation"),
                        "native_reference_passed": c.get("raw", {}).get(
                            "native_reference_passed"
                        ),
                        "dense_skipped_reason": c.get("raw", {}).get(
                            "dense_skipped_reason"
                        ),
                        "upstream_skipped_reason": c.get("raw", {}).get(
                            "upstream_skipped_reason"
                        ),
                        "error": c.get("error"),
                        "build_group": c["fingerprint"],
                        "measurements": c["measurements"],
                    },
                    indent=2,
                )
            )
            raw = (
                f'<a href="{c["raw_link"]}">raw JSON</a>'
                if c.get("raw_link")
                else "no result"
            )
            qualification = native_qualification_note(c)
            if qualification:
                raw += f"<br><small>{esc(qualification)}</small>"
            rows.append(
                f'<tr><td><details><summary>{esc(c["id"])}</summary><pre>{detail}</pre></details></td><td>{esc(c["family"])}</td><td>{esc(c["tier"])}</td><td class="{esc(c["status"])}">{esc(c["status"])}</td><td>{ratio}</td><td>{raw}</td></tr>'
            )
        metadata = {
            k: v for k, v in suite["metadata"].items() if k not in {"cases", "verified"}
        }
        recovery_links = " · ".join(
            f'<a href="{artifact["path"]}">{esc(name.replace("_", " "))}</a>'
            for name, artifact in suite.get("recovery_artifacts", {}).items()
        )
        timing_summary = (
            f"Among {len(pairs)} matched CPU cases: <strong>{observed} observed median slowdowns; {bounded} have their entire 95% sample-bootstrap interval below 1.</strong>"
            if pairs
            else ""
        )
        sections.append(
            f'<section id="{slug(suite["name"])}"><p class="eyebrow">{"Historical evidence — separate build and protocol" if suite["historical"] else "Fresh campaign — independent environment"}</p><h2>{esc(suite["name"])}</h2><p>{len(cases)} declared cases · case outcomes {esc(dict(states))}. The accuracy ledger reports per-input, per-backend acceptance separately from collection errors. {timing_summary}</p><details><summary>Environment, build provenance and protocol</summary><pre>{esc(json.dumps(metadata, indent=2))}</pre>{("<p>Controller recovery evidence: " + recovery_links + "</p>") if recovery_links else ""}</details>{figures}<h3>Complete case ledger</h3><p>Expand a case for all recorded timing samples, validation and memory. Search filters this table only.</p><input aria-label="Filter case ledger" placeholder="Filter case name, family, tier, status…" oninput="for (const r of this.nextElementSibling.tBodies[0].rows) r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase())"><table><thead><tr><th>Case and measurements</th><th>Family</th><th>Tier</th><th>Status</th><th>CPU speedup / slowdown evidence</th><th>Evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>'
        )
    navigation = " · ".join(
        f'<a href="#{slug(s["name"])}">{esc(s["name"])}</a>' for s in suites
    )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>treams / treams-rs benchmark atlas</title><style>
    :root{font-family:system-ui,sans-serif;color:#192738;background:#f5f7fa;line-height:1.55}body{max-width:1260px;margin:auto;padding:40px 24px}h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.1;letter-spacing:-.04em}h2{font-size:2rem}h3{margin-top:2rem}a{color:#2166ac}.eyebrow{text-transform:uppercase;letter-spacing:.1em;font-size:.75rem;font-weight:700;color:#526b82}header{padding:1rem 0 2rem}section{border-top:2px solid #c5d1dc;padding-top:2rem;margin-top:3rem}figure{background:white;padding:12px;border:1px solid #d8e0e8;margin:24px 0;border-radius:8px}img{width:100%;height:auto}figcaption{font-size:.85rem;color:#536579;padding:0 15px 12px}details{background:#fff;padding:10px;border:1px solid #d8e0e8;border-radius:4px}summary{cursor:pointer}pre{font-size:.75rem;white-space:pre-wrap;overflow-wrap:anywhere;max-height:30rem;overflow:auto}table{border-collapse:collapse;width:100%;font-size:.8rem;margin-top:10px}th,td{text-align:left;border-bottom:1px solid #d8e0e8;padding:8px;vertical-align:top}th{background:#e5edf4;position:sticky;top:0}td:first-child{min-width:240px;max-width:600px}td details{padding:0;background:none;border:0}input{padding:12px;font:inherit;max-width:100%;width:480px;box-sizing:border-box;border:1px solid #bccad6;border-radius:5px}.failed,.accuracy_failed,.error,.invalid_output,.timeout,.memory_limit,.missing{color:#a72f27;font-weight:700}.note{border-left:4px solid #52799d;padding-left:18px;max-width:1000px}@media(max-width:650px){body{padding:20px 12px}table{display:block;overflow:auto}figure{padding:2px}figcaption{padding:8px}th{position:static}}
    </style><header><p class="eyebrow">Measured performance · numerical qualification · reproducible evidence</p><h1>treams → treams-rs<br>Benchmark atlas</h1><p>CPU runtime, memory, scaling and gradients — alongside independent reference error, physical conservation and numerical convergence.</p><p><a href="benchmark-atlas.pdf">All plots as PDF</a> · <a href="measurements.csv">Timing/memory CSV</a> · <a href="accuracy.csv">Accuracy CSV</a> · <a href="accuracy-observations.json">Accuracy JSON</a> · <a href="manifest-index.json">Manifest index</a></p><nav>"""
        + navigation
        + """</nav></header><div class="note"><p><strong>Reading these plots.</strong> CPU speedup is the median of paired treams/Rust ratios when paired samples exist, otherwise the ratio of independent medians; values below 1 are observed median slowdowns. Counts additionally distinguish those whose entire 95% sample-bootstrap interval lies below 1. The complete case ledger retains failures, timeouts and skipped baselines. Native forwards that retain derivative context are identified separately from forward-only calls. Size labels come from the manifest and have workload-specific meanings. Generic broad-suite dimensions can change batch count, truncation, lattice radius or no actual scalar argument; those cases contribute to feature coverage but never to size-scaling curves. Scaling plots use the separately declared sweeps. Lines connect measured points only.</p><p><strong>What uncertainty means.</strong> Runtime bands show observed sample minima/maxima. Speedup intervals bootstrap the recorded samples, preserving pairs where the harness recorded alternating pairs. They describe this sample set, not host-to-host or day-to-day variation. Larger cases use fixed upstream-then-native isolated-process order; these intervals do not capture order-dependent drift. Summary-only records have no invented uncertainty.</p><p><strong>Memory.</strong> Absolute process high-water RSS includes imports, allocator retention and bookkeeping. After-setup growth subtracts the recorded baseline and is not a measurement of live numerical allocations.</p><p><strong>Comparisons remain separate.</strong> Each environment and build fingerprint gets its own figures. Dimensionless tolerance-scaled and normwise relative errors are the primary cross-feature accuracy comparisons. Absolute-error plots are separate by family because their raw units and magnitudes depend on the workload. Accuracy reference kinds and error metrics stay separate: agreement with upstream is not independent proof of correctness, and a passing speed benchmark supplies no error estimate. Native gradient timings include the forward pass when stated; upstream finite differences are a different algorithm, and exact upstream linear adjoints appear separately.</p></div>"""
        + figure_html(cross_gallery)
        + "".join(sections)
        + "</html>"
    )
    (output / "index.html").write_text(page)


def archive_controller_recovery(suite, output):
    recoveries = suite["metadata"].get("controller_recoveries")
    if recoveries is None:
        previous = suite["metadata"].get("controller_recovery")
        recoveries = [previous] if previous else []
    if not recoveries:
        return {}
    target = Path("source-manifests") / (slug(suite["name"]) + "-controller-recovery")
    (output / target).mkdir(parents=True, exist_ok=True)
    archived = {}
    for index, recovery in enumerate(recoveries, 1):
        sources = {
            "previous_manifest": (
                suite["path"].parent / recovery["previous_manifest"],
                recovery["previous_manifest_sha256"],
            ),
            "previous_controller": (
                suite["path"].parent
                / recovery.get(
                    "previous_runner_source",
                    "../controller-recovery/runner-before.py.txt",
                ),
                recovery["previous_runner_sha256"],
            ),
        }
        for name, (source, expected_hash) in sources.items():
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError(
                    f"Controller recovery evidence checksum differs: {source.name}"
                )
            destination = target / source.name
            shutil.copyfile(source, output / destination)
            archived[f"recovery-{index}-{name}"] = {
                "path": destination.as_posix(),
                "sha256": digest,
            }
    return archived


def generate(manifest_paths, output, root=None):
    import matplotlib.backends.backend_pdf as pdf_backend

    chart_style()
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    (output / "source-manifests").mkdir(exist_ok=True)
    suites = [load_manifest(path, root) for path in manifest_paths]
    if len({slug(s["name"]) for s in suites}) != len(suites):
        raise ValueError(
            "Manifest suite_id/environment/phase labels must identify distinct datasets"
        )
    galleries = []
    with pdf_backend.PdfPages(
        output / "benchmark-atlas.pdf",
        metadata={
            "Title": "treams / treams-rs benchmark atlas",
            "Author": "treams-rs contributors",
        },
    ) as pdf:
        for suite in suites:
            gallery = []
            caption = suite_caption(suite)
            suite["recovery_artifacts"] = archive_controller_recovery(suite, output)
            shutil.copyfile(
                suite["path"],
                output / "source-manifests" / (slug(suite["name"]) + ".json"),
            )
            for c in suite["cases"]:
                if c["raw_path"]:
                    directory = output / "raw" / slug(suite["name"])
                    directory.mkdir(exist_ok=True)
                    target = f"raw/{slug(suite['name'])}/{c['raw_path'].name}"
                    shutil.copyfile(c["raw_path"], output / target)
                    c["raw_link"] = target
                    arrays = (
                        c.get("raw", {}).get("validation", {}).get("check_arrays", {})
                    )
                    if arrays.get("path"):
                        source = c["raw_path"].parent / arrays["path"]
                        if source.exists():
                            shutil.copyfile(source, directory / source.name)
            by_build = defaultdict(list)
            for case in suite["cases"]:
                by_build[case["fingerprint"]].append(case)
            for fingerprint, cases in by_build.items():
                prefix = slug(suite["name"]) + "--" + fingerprint

                def save(
                    fig,
                    name,
                    title,
                    description,
                    prefix=prefix,
                    fingerprint=fingerprint,
                    suite_name=suite["name"],
                    caption=caption,
                    gallery=gallery,
                ):
                    save_figure(
                        fig,
                        output,
                        f"{prefix}--{name}",
                        title,
                        description,
                        gallery,
                        pdf,
                        subtitle=caption,
                        provenance=f"Dataset: {suite_name}. Build group: {fingerprint}.",
                    )

                plot_overview(cases, save)
                plot_frontier(cases, save)
                plot_scaling(cases, save, suite["cases"])
                plot_native_reverse(cases, save)
                plot_gradients(cases, save)
                plot_illumination(cases, save)
                plot_gpu(cases, save)
                plot_accuracy(cases, save)
                plot_paper_spectra(cases, save)
            galleries.append(gallery)
        cross_gallery = plot_host_comparison(suites, output, pdf)
    write_ledger(suites, output)
    write_accuracy_pages(suites, output)
    write_html(suites, galleries, output, cross_gallery)
    (output / "README.md").write_text(
        "# Benchmark and accuracy report\n\n"
        "Open index.html. The accuracy ledgers load separately and show at most "
        "100 matching rows; complete observations remain in accuracy.csv and "
        "accuracy-observations.json. Timing/memory samples are in measurements.csv. "
        "Raw JSON, gradient NPZ arrays, source manifests, and per-figure PNG/SVG/PDF "
        "files accompany the combined benchmark-atlas.pdf.\n\n"
        "Runtime intervals bootstrap recorded samples, preserving pairs where available. "
        "They do not measure host/day variation. Larger cases use fixed upstream-then-native "
        "isolated-process order; intervals do not capture order-dependent drift. "
        "Observed median slowdowns and intervals wholly below one are counted separately. "
        "A 0.75x speedup means 1/0.75 times the runtime, about 33% more time.\n\n"
        "Absolute RSS includes imports, input setup and allocator retention. "
        "Additional RSS is high-water growth after setup, not live numerical allocations. "
        "\n"
        "Accuracy references and metrics stay separate. Dimensionless tolerance-scaled "
        "and normwise relative errors support cross-feature comparison; absolute errors "
        "retain workload-dependent units and magnitudes and are plotted per family. "
        "Upstream absolute-error distributions are omitted; all absolute values and "
        "statuses remain in the ledger/CSV, alongside targeted convergence curves. "
        "Agreement with upstream, physical conservation, independent high-precision "
        "references and numerical-cutoff sensitivity are different evidence. "
        "Timing success and passing test counts are not accuracy measurements. "
        "Failed numerical observations, diagnostics and undefined residuals remain visible.\n\n"
        "Every suite retains its own environment and build identity. Historical data, "
        "when included, are labelled separately. Original manifest paths use the "
        "repository layout; source-manifests and manifest-index.json identify the inputs.\n"
    )
    index = [
        {
            "suite_id": s["name"],
            "manifest": "source-manifests/" + slug(s["name"]) + ".json",
            "manifest_sha256": hashlib.sha256(s["path"].read_bytes()).hexdigest(),
            "cases": len(s["cases"]),
            "figures": len(g),
            "controller_recovery_artifacts": s.get("recovery_artifacts", {}),
        }
        for s, g in zip(suites, galleries, strict=True)
    ]
    (output / "manifest-index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"report": str(output / "index.html"), "suites": index}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, help="Resolve result paths relative to this directory"
    )
    args = parser.parse_args()
    generate(args.manifest, args.output, args.root)
