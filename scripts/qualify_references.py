# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["mpmath>=1.3", "numpy>=2.1", "treams==0.4.5", "threadpoolctl>=3.6"]
# ///
"""Independent high-precision references, retaining failures and domain exclusions.

Run in the release development environment after timing experiments:
uv run --no-sync --with mpmath python scripts/qualify_references.py --output result.json
The default grid is fixed before execution; --limit is only for quick test runs of the script.
"""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import runpy
import sys
import time
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import numpy as np
import treams
from _harness import file_sha256, package_sha256, pinned_threads, threadpools
from threadpoolctl import threadpool_limits

import treams_rs
from treams_rs import _native, coeffs, diff, special

ROOT = Path(__file__).resolve().parents[1]
ferrers = runpy.run_path(str(Path(__file__).with_name("qualify_legendre.py")))[
    "ferrers"
]
FAMILIES = (
    "cylindrical_bessel",
    "spherical_bessel",
    "ferrers",
    "mie_sphere",
    "wigner_rotation",
)
TOLERANCES = {
    family: {"atol": 2e-13, "rtol": 1e-10 if family == "ferrers" else 2e-11}
    for family in FAMILIES
}
SOURCES = [
    {
        "title": "NIST DLMF spherical Bessel definitions",
        "url": "https://dlmf.nist.gov/10.47",
    },
    {"title": "NIST DLMF Bessel derivatives", "url": "https://dlmf.nist.gov/10.6"},
    {
        "title": "NIST DLMF spherical Bessel derivatives",
        "url": "https://dlmf.nist.gov/10.51",
    },
    {
        "title": "NIST DLMF Ferrers hypergeometric representation",
        "url": "https://dlmf.nist.gov/14.3",
    },
    {
        "title": "Feng et al. (2015), Wigner factorial sum, equation (1)",
        "url": "https://arxiv.org/pdf/1507.04535",
    },
    {
        "title": "Broadband suppression of backscattering at optical frequencies using low permittivity dielectric spheres",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5677120/",
    },
]


def package_digest(package, suffixes):
    return package_sha256(Path(package.__file__).parent, suffixes)


def fingerprints():
    return {
        "native_sha256": file_sha256(_native.__file__),
        "native_profile": _native.build_profile(),
        "python_source_sha256": package_digest(treams_rs, (".py",)),
        "upstream_package_sha256": package_digest(
            treams, (".py", ".so", ".dylib", ".pyd")
        ),
        "mpmath_source_sha256": package_digest(mp, (".py",)),
        "script_sha256": file_sha256(__file__),
        "ferrers_reference_script_sha256": file_sha256(
            Path(__file__).with_name("qualify_legendre.py")
        ),
    }


def neutral(message):
    return (
        str(message)
        .replace(str(ROOT), "<checkout>")
        .replace(str(Path.home()), "<home>")
    )


def pair(value):
    value = complex(value)
    return [value.real, value.imag]


def radial(kind, order, z, spherical=False, derivative=False):
    """DLMF 10.47 and 10.6, evaluated without the native/SciPy radial code."""
    function = {"j": mp.besselj, "y": mp.bessely, "h1": mp.hankel1, "h2": mp.hankel2}[
        kind
    ]
    order, z = mp.mpf(order), mp.mpc(z)
    if spherical:
        value = mp.sqrt(mp.pi / (2 * z)) * function(order + mp.mpf("0.5"), z)
        return (
            order * value / z
            - mp.sqrt(mp.pi / (2 * z)) * function(order + mp.mpf("1.5"), z)
            if derivative
            else value
        )
    return (
        (function(order - 1, z) - function(order + 1, z)) / 2
        if derivative
        else function(order, z)
    )


def homogeneous_mie(degree, size, epsilon, mu):
    """Direct scalar Mie ratios, vacuum host, then the treams helicity convention.

    With psi=z*j_l(z), xi=z*h_l^(1)(z), m=sqrt(epsilon*mu):
    a=(m*psi(mx)*psi'(x)-mu*psi(x)*psi'(mx)) /
      (m*psi(mx)*xi'(x)-mu*xi(x)*psi'(mx)); b interchanges m and mu.
    In treams T=-Mie: diagonal=-(a+b)/2, off-diagonal=-(a-b)/2.
    This avoids the native multilayer 4-by-4 transfer matrices.
    """
    size, epsilon, mu = mp.mpf(size), mp.mpc(epsilon), mp.mpc(mu)
    index = mp.sqrt(epsilon * mu)

    def riccati(z, kind):
        value = radial(kind, degree, z, spherical=True)
        return z * value, value + z * radial(
            kind, degree, z, spherical=True, derivative=True
        )

    inner, dinner = riccati(index * size, "j")
    outer, douter = riccati(size, "j")
    outgoing, doutgoing = riccati(size, "h1")
    electric = (index * inner * douter - mu * outer * dinner) / (
        index * inner * doutgoing - mu * outgoing * dinner
    )
    magnetic = (mu * inner * douter - index * outer * dinner) / (
        mu * inner * doutgoing - index * outgoing * dinner
    )
    diagonal, cross = -(electric + magnetic) / 2, -(electric - magnetic) / 2
    return [diagonal, cross, cross, diagonal]


def wigner_smalld(degree, m_out, m_in, beta):
    """Factorial sum, Feng et al. (2015), Eq. (1), in z-y-z convention.

    Rows label m_out, columns m_in. In particular d^1[-1,0]=sin(beta)/sqrt(2).
    High working precision handles cancellation; no native eigenproblem or
    native scalar Jacobi recurrence contributes to this reference.
    """
    cosine, sine = mp.cos(mp.mpf(beta) / 2), mp.sin(mp.mpf(beta) / 2)
    normalization = mp.sqrt(
        mp.factorial(degree + m_out)
        * mp.factorial(degree - m_out)
        * mp.factorial(degree + m_in)
        * mp.factorial(degree - m_in)
    )
    return normalization * mp.fsum(
        (-1) ** (s + m_out - m_in)
        * cosine ** (2 * degree - 2 * s + m_in - m_out)
        * sine ** (2 * s + m_out - m_in)
        / (
            mp.factorial(degree - m_out - s)
            * mp.factorial(degree + m_in - s)
            * mp.factorial(s + m_out - m_in)
            * mp.factorial(s)
        )
        for s in range(max(0, m_in - m_out), min(degree - m_out, degree + m_in) + 1)
    )


@lru_cache(maxsize=12)
def rotation_block(backend, degree, angles, polarization):
    """Evaluate the matrix API path that the failed benchmark check used."""
    library = treams_rs if backend == "treams-rs" else treams
    basis = (
        treams_rs.SphericalBasis
        if backend == "treams-rs"
        else treams.SphericalWaveBasis
    )([(degree, m, polarization) for m in range(-degree, degree + 1)])
    return np.asarray(
        (library.operators if backend == "treams-rs" else library).rotate(
            *angles, basis=basis
        )
    )


def certify_rotation_disagreements(
    actual, upstream, basis, angles, *, atol=1e-12, rtol=2e-9
):
    """Resolve every rotation mismatch with treams, keeping the treams tolerance.

    ``basis`` explicitly lists (particle, degree, order, polarization) for each
    row/column. Entries that match treams keep the treams comparison as evidence;
    only mismatches receive independent 80/120-digit factorial-sum references.
    Load this collector lazily after a benchmark's agreement check fails so normal
    benchmark validation does not require mpmath. Reference work is untimed.
    """
    actual, upstream = np.asarray(actual), np.asarray(upstream)
    modes = np.asarray(basis)
    if (
        actual.ndim != 2
        or actual.shape != upstream.shape
        or actual.shape[0] != actual.shape[1]
        or modes.shape != (actual.shape[0], 4)
        or not np.issubdtype(modes.dtype, np.integer)
    ):
        raise ValueError(
            "require matching square arrays and integer (particle,l,m,pol) basis"
        )
    if (
        np.any(modes[:, :2] < 0)
        or np.any(abs(modes[:, 2]) > modes[:, 1])
        or np.any((modes[:, 3] != 0) & (modes[:, 3] != 1))
        or len(angles) != 3
        or not all(math.isfinite(x) for x in angles)
        or not all(math.isfinite(x) and x >= 0 for x in (atol, rtol))
        or atol + rtol == 0
    ):
        raise ValueError("invalid spherical modes, rotation angles, or tolerance")
    entries = []
    for start in range(0, actual.size, 65536):
        left, right = (
            actual.flat[start : start + 65536],
            upstream.flat[start : start + 65536],
        )
        close = (
            np.isfinite(left)
            & np.isfinite(right)
            & np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=False)
        )
        for offset in np.flatnonzero(~close):
            row, column = divmod(start + int(offset), actual.shape[1])
            to, source = modes[row].tolist(), modes[column].tolist()
            entry = {
                "row": row,
                "column": column,
                "output_mode": to,
                "input_mode": source,
            }
            case = {
                "family": "wigner_rotation",
                "function": "wignerd",
                "parameters": {
                    "degree": to[1],
                    "m_out": to[2],
                    "m_in": source[2],
                    "angles": angles,
                },
            }
            structural_zero = any(to[index] != source[index] for index in (0, 1, 3))
            entry["structural_zero"] = structural_zero
            try:
                with mp.workdps(80):
                    first = mp.mpc(0) if structural_zero else reference(case)[0]
                    entry["reference_80_digits"] = {
                        "real": str(mp.re(first)),
                        "imag": str(mp.im(first)),
                    }
                with mp.workdps(120):
                    precise = mp.mpc(0) if structural_zero else reference(case)[0]
                    delta = abs(first - precise) / max(mp.mpf(1), abs(precise))
                    entry.update(
                        reference_120_digits={
                            "real": str(mp.re(precise)),
                            "imag": str(mp.im(precise)),
                        },
                        reference_precision_delta=str(delta),
                        reference_stable=bool(
                            mp.isfinite(precise) and delta < mp.mpf("1e-40")
                        ),
                    )
                    for backend, value in (
                        ("treams-rs", actual[row, column]),
                        ("treams", upstream[row, column]),
                    ):
                        try:
                            entry[backend] = errors([value], [precise], atol, rtol)
                        except ValueError as exc:
                            entry[backend] = {
                                "status": "error",
                                "error": neutral(exc),
                                "value": {
                                    "real": str(np.real(value)),
                                    "imag": str(np.imag(value)),
                                },
                            }
                    try:
                        entry["original_comparison"] = errors(
                            [actual[row, column]],
                            [mp.mpc(complex(upstream[row, column]))],
                            atol,
                            rtol,
                        )
                    except ValueError as exc:
                        entry["original_comparison"] = {
                            "status": "error",
                            "error": neutral(exc),
                        }
            except (
                Exception
            ) as exc:  # Keep an unresolvable disagreement as a failed check.
                entry.update(reference_stable=False, reference_error=neutral(exc))
            entries.append(entry)
    with mp.workdps(120):
        precise_angles = [str(mp.mpf(angle)) for angle in angles]
    return {
        "passed": all(
            entry["reference_stable"]
            and entry.get("treams-rs", {}).get("status") == "passed"
            for entry in entries
        ),
        "original_gate_passed": not entries,
        "total_entries": int(actual.size),
        "upstream_matched_entries": int(actual.size) - len(entries),
        "mismatch_count": len(entries),
        "entries": entries,
        "protocol": {
            "reference_scope": "Only original failing entries have independent high-precision references; other entries retain the original upstream agreement gate.",
            "atol": atol,
            "rtol": rtol,
            "angles": list(angles),
            "angles_high_precision": precise_angles,
            "basis_order": ["particle", "degree", "order", "polarization"],
            "reference_precision_digits": [80, 120],
            "source": "https://arxiv.org/pdf/1507.04535 (equation 1)",
            "script_sha256": file_sha256(__file__),
            "mpmath_version": mp.__version__,
        },
    }


def case_grid():
    """Fixed reference grid followed by explicitly tagged diagnostic cases."""
    arguments = [
        ("small", 0.01 + 0.002j),
        ("real", 0.3),
        ("complex", 1.3 + 0.2j),
        ("lower_half_plane", 3 - 0.7j),
        ("oscillatory", 10 + 2j),
        ("large", 80 + 0.1j),
        ("very_large", 300 + 0.1j),
        ("cut_above", -3 + 1e-9j),
        ("cut_below", -3 - 1e-9j),
        ("evanescent", 1 + 20j),
        ("growing", 1 - 20j),
    ]
    for spherical, orders in (
        (False, [-3.4, -0.2, 0, 0.5, 3, 12, 48]),
        (True, [0, 1, 2, 6, 16, 48, 96, 128]),
    ):
        family = "spherical_bessel" if spherical else "cylindrical_bessel"
        for degree in orders:
            for regime, z in [
                *arguments,
                ("near_regular_zero", math.pi if spherical else 2.404825557695773),
            ]:
                for kind in ("j", "y", "h1", "h2"):
                    for derivative in (False, True):
                        function = (
                            ("spherical_" if spherical else "")
                            + {
                                "j": "jn" if spherical else "jv",
                                "y": "yn" if spherical else "yv",
                                "h1": "hankel1",
                                "h2": "hankel2",
                            }[kind]
                            + ("_d" if derivative else "")
                        )
                        yield {
                            "family": family,
                            "function": function,
                            "parameters": {
                                "order": degree,
                                "argument": pair(z),
                                "kind": kind,
                                "derivative": derivative,
                            },
                            "regime": (
                                "shared_low_order_zero_argument"
                                if regime == "near_regular_zero"
                                and (degree != 0 or kind != "j" or derivative)
                                else regime
                            ),
                        }
    for degree in (0.2, 2.3, math.nextafter(10, 11), 32.3, 65.2, 127.2):
        orders = sorted(
            {
                sign * order
                for sign in (-1, 1)
                for order in (0, 1, 3, 12, 64)
                if order < degree
            }
        )
        for order in orders:
            for x in (-0.999999, -0.9, -0.3501, 0.3, 0.999999):
                for derivative in (False, True):
                    yield {
                        "family": "ferrers",
                        "function": "lpmv_d" if derivative else "lpmv",
                        "parameters": {
                            "degree": degree,
                            "order": order,
                            "argument": x,
                            "derivative": derivative,
                        },
                        "regime": "near_endpoint" if abs(x) > 0.99 else "interior",
                    }
    for regime, epsilon, mu in (
        ("dielectric", 2.25, 1),
        ("high_index", 16, 1),
        ("lossy", 3.1 + 0.2j, 1),
        ("magnetic_lossy", 2.5 + 0.2j, 1.3 + 0.1j),
        ("metallic", -8 + 0.4j, 1),
        ("near_zero_contrast", 1 + 1e-8, 1),
    ):
        for size in (0.01, 0.1, 0.7, 2, 8, 30, 80):
            for degree in sorted(
                {
                    1,
                    3,
                    max(1, round(size)),
                    min(128, max(6, round(size + 4 * size ** (1 / 3) + 2))),
                }
            ):
                yield {
                    "family": "mie_sphere",
                    "function": "mie",
                    "parameters": {
                        "degree": degree,
                        "size": size,
                        "epsilon": pair(epsilon),
                        "mu": pair(mu),
                        "kappa": 0,
                        "host_epsilon": 1,
                        "host_mu": 1,
                    },
                    "regime": regime,
                }

    # Appended after the original 1,930 definitions to preserve their IDs/inputs.
    # Selection followed an observed upstream/native discrepancy, not a random grid.
    pairs = [(-16, -12), (-15, -12), (-15, -11), (-15, -10), (-14, -11)]
    pairs += [(-m, -k) for m, k in pairs]
    for degree in (23, 24, 25):
        for m_out, m_in in pairs:
            for function, polarization in (
                ("wignersmalld", None),
                ("wignerd", None),
                ("rotation_entry", 0),
                ("rotation_entry", 1),
            ):
                yield {
                    "family": "wigner_rotation",
                    "function": function,
                    "parameters": {
                        "degree": degree,
                        "m_out": m_out,
                        "m_in": m_in,
                        "angles": [0.2, 0.7, -0.3],
                        "polarization": polarization,
                    },
                    "regime": "post_failure_diagnostic",
                    "selection": "post_failure_diagnostic",
                    "selection_reason": "Investigates rotation L24 discrepancies at m_out,m_in=(-15,-11),(15,11) observed in the Linux scaling gate; nearby pairs/orders are diagnostic context.",
                }


def reference(case):
    p = case["parameters"]
    if case["family"] == "wigner_rotation":
        alpha, beta, gamma = map(mp.mpf, p["angles"])
        value = wigner_smalld(p["degree"], p["m_out"], p["m_in"], beta)
        if case["function"] != "wignersmalld":
            value *= mp.exp(-mp.j * (p["m_out"] * alpha + p["m_in"] * gamma))
        return [value]
    if case["family"].endswith("bessel"):
        return [
            radial(
                p["kind"],
                p["order"],
                complex(*p["argument"]),
                case["family"] == "spherical_bessel",
                p["derivative"],
            )
        ]
    if case["family"] == "mie_sphere":
        return homogeneous_mie(
            p["degree"], p["size"], complex(*p["epsilon"]), complex(*p["mu"])
        )
    degree, order, z = mp.mpf(p["degree"]), p["order"], mp.mpf(p["argument"])
    value = ferrers(degree, order, z)
    if p["derivative"]:
        value = (
            -mp.sqrt(1 - z * z) * ferrers(degree, order + 1, z) - order * z * value
        ) / (1 - z * z)
    return [value]


def evaluate(case, backend):
    p = case["parameters"]
    if case["family"] == "wigner_rotation":
        if case["function"] == "rotation_entry":
            matrix = rotation_block(
                backend, p["degree"], tuple(p["angles"]), p["polarization"]
            )
            return [matrix[p["m_out"] + p["degree"], p["m_in"] + p["degree"]]]
        library = special if backend == "treams-rs" else treams.special
        angles = p["angles"] if case["function"] == "wignerd" else [p["angles"][1]]
        return [
            getattr(library, case["function"])(
                p["degree"], p["m_out"], p["m_in"], *angles
            )
        ]
    if case["family"].endswith("bessel"):
        library = special if backend == "treams-rs" else treams.special
        return [getattr(library, case["function"])(p["order"], complex(*p["argument"]))]
    if case["family"] == "mie_sphere":
        library = coeffs if backend == "treams-rs" else treams.coeffs
        return np.asarray(
            library.mie(
                p["degree"],
                [p["size"]],
                [complex(*p["epsilon"]), 1],
                [complex(*p["mu"]), 1],
                [0, 0],
            )
        ).ravel()
    if backend == "treams-rs":
        if p["derivative"]:
            _, context = diff.angular(p["degree"], p["order"], p["argument"])
            return [context.pullback(np.asarray(1, complex))]
        return [special.lpmv(p["order"], p["degree"], p["argument"])]
    value = treams.special.lpmv(p["order"], p["degree"], p["argument"])
    if p["derivative"]:
        z, order = p["argument"], p["order"]
        value = (
            -np.sqrt(1 - z * z) * treams.special.lpmv(order + 1, p["degree"], z)
            - order * z * value
        ) / (1 - z * z)
    return [value]


def finite_number(text):
    """The float Python, json and NumPy read from `text`.

    None if float64 cannot hold it: the value overflows or underflows to zero.
    """
    if text is None:
        return None
    result = float(text)
    if math.isfinite(result) and (result != 0 or mp.mpf(text) == 0):
        return result
    return None


def errors(actual, expected, atol, rtol):
    actual = [complex(x) for x in actual]
    if len(actual) != len(expected) or not all(
        math.isfinite(x.real) and math.isfinite(x.imag) for x in actual
    ):
        raise ValueError("wrong shape or nonfinite backend output")
    residuals = [abs(mp.mpc(a) - b) for a, b in zip(actual, expected, strict=True)]
    norm = mp.sqrt(sum(abs(x) ** 2 for x in expected))
    metrics = {
        "max_abs_error": max(residuals),
        "relative_l2_error": mp.sqrt(sum(x * x for x in residuals)) / norm
        if norm
        else None,
        "max_scaled_error": max(
            err / (mp.mpf(atol) + mp.mpf(rtol) * abs(ref))
            for err, ref in zip(residuals, expected, strict=True)
        ),
    }
    published = {
        key: str(value) if value is not None else None for key, value in metrics.items()
    }
    return {
        "value": [pair(x) for x in actual],
        # Read each number from its string so both forms give the same float.
        **{key: finite_number(text) for key, text in published.items()},
        "errors_high_precision": published,
        "status": "passed" if metrics["max_scaled_error"] <= 1 else "failed",
    }


def reference_range(values):
    """Classify reference components without discarding underflow-scale values."""

    def category(value):
        magnitude = abs(value)
        if magnitude == 0:
            return "zero"
        if magnitude > sys.float_info.max:
            return "overflow"
        if magnitude < math.ulp(0.0):
            return "below_min_subnormal"
        if magnitude < sys.float_info.min:
            return "subnormal"
        return "normal"

    components = [part for value in values for part in (mp.re(value), mp.im(value))]
    ranges = [
        {"real": category(mp.re(value)), "imag": category(mp.im(value))}
        for value in values
    ]
    counts = Counter(category(part) for part in components)
    return {
        "float64_no_overflow": counts["overflow"] == 0,
        "reference_component_ranges": ranges,
        "float64_subnormal_components": counts["subnormal"],
        "float64_below_min_subnormal_components": counts["below_min_subnormal"],
        "float64_rounds_to_zero_components": sum(
            part != 0 and float(part) == 0 for part in components
        ),
    }


def qualify_case(case, digits):
    result = {**case, "tolerance": TOLERANCES[case["family"]], "backends": {}}
    try:
        with mp.workdps(digits):
            first = reference(case)
        with mp.workdps(digits + 40):
            precise = reference(case)
            if not all(mp.isfinite(x) for x in precise):
                raise ValueError("nonfinite high precision reference")
            delta = max(
                abs(a - b) / max(mp.mpf(1), abs(b))
                for a, b in zip(first, precise, strict=True)
            )
            result["reference"] = [
                {"real": str(mp.re(x)), "imag": str(mp.im(x))} for x in precise
            ]
            result["reference_precision_delta"] = str(delta)
            result["reference_precision_digits"] = digits + 40
            result["reference_stable"] = delta < mp.mpf("1e-40")
            result.update(reference_range(precise))
            for backend in ("treams-rs", "treams"):
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        values = evaluate(case, backend)
                    row = {"warnings": sorted({neutral(w.message) for w in captured})}
                    if result["reference_stable"] and result["float64_no_overflow"]:
                        row.update(errors(values, precise, **result["tolerance"]))
                    else:
                        row.update(
                            status="error",
                            error="reference exceeds finite float64 range or has unstable precision",
                        )
                except Exception as exc:  # Qualification records backend failures, including mpmath convergence errors.
                    row = {"status": "error", "error": neutral(exc)}
                result["backends"][backend] = row
    except (
        Exception
    ) as exc:  # Keep invalid references visible instead of dropping the case.
        result.update(reference_error=neutral(exc), reference_stable=False)
    return result


def observations(case):
    for backend in ("treams-rs", "treams"):
        measured = case["backends"].get(backend, {"status": "error"})
        for metric in ("max_abs_error", "relative_l2_error", "max_scaled_error"):
            yield {
                "id": f"{case['id']}/{backend}/{metric}",
                "family": case["family"],
                "backend": backend,
                "reference_kind": "high_precision",
                "selection": case.get("selection", "predeclared_grid"),
                "selection_reason": case.get("selection_reason"),
                "metric": metric,
                "error": measured.get(metric),
                "error_high_precision": measured.get("errors_high_precision", {}).get(
                    metric
                ),
                "parameters": {
                    **case["parameters"],
                    "function": case["function"],
                    "regime": case["regime"],
                },
                "tolerance": 1.0 if metric == "max_scaled_error" else None,
                "status": measured["status"],
                "message": measured.get("error", case.get("reference_error")),
                "warnings": measured.get("warnings", []),
                "reference_precision_digits": case.get("reference_precision_digits"),
                "series": case["function"],
                "conditioning": {
                    "regime": case["regime"],
                    "float64_no_overflow": case.get("float64_no_overflow"),
                    "float64_subnormal_components": case.get(
                        "float64_subnormal_components"
                    ),
                    "float64_below_min_subnormal_components": case.get(
                        "float64_below_min_subnormal_components"
                    ),
                    "float64_rounds_to_zero_components": case.get(
                        "float64_rounds_to_zero_components"
                    ),
                    "reference_stable": case.get("reference_stable"),
                },
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--family", choices=FAMILIES, action="append")
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if (
        args.digits < 60
        or args.threads < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error(
            "reference precision must be >=60 digits; threads and limit positive"
        )
    os.environ.update(pinned_threads(args.threads))
    # treams_rs is imported above, so its environment has been read.
    treams_rs.set_num_threads(args.threads)
    selected = [
        {"id": f"reference-{index:05d}", **case}
        for index, case in enumerate(case_grid())
        if not args.family or case["family"] in args.family
    ]
    planned = len(selected)
    if args.limit is not None:
        selected = selected[: args.limit]
    identity, started = fingerprints(), time.monotonic()
    with threadpool_limits(limits=args.threads):
        pools = threadpools()
        rows = [qualify_case(case, args.digits) for case in selected]
    if fingerprints() != identity:
        raise RuntimeError("reference or solver source changed during qualification")
    result = {
        "kind": "accuracy",
        "schema_version": 1,
        "source": identity,
        "environment": {
            "threads": args.threads,
            "threadpools": pools,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mpmath": mp.__version__,
            "treams": importlib.metadata.version("treams"),
        },
        "protocol": {
            "grid": "Original 1,930 fixed cross-product definitions followed by 120 explicitly tagged post-failure Wigner diagnostics. Original inputs and IDs are unchanged. The appended diagnostic set was selected after an observed rotation mismatch and must not be described as a representative prespecified population.",
            "planned_cases": planned,
            "complete": len(rows) == planned,
            "precision_digits": [args.digits, args.digits + 40],
            "reference_precision_gate": "max |reference_low-reference_high|/max(1,|reference_high|) < 1e-40",
            "metric": "max_scaled_error = max_i |actual_i-reference_i|/(atol+rtol*|reference_i|); pass iff <=1",
            "float64_range_policy": "float64_no_overflow only checks each reference component against the largest finite float64 value. Subnormal components, values below the smallest subnormal, and nonzero components that round to zero are counted separately. Tiny references remain in absolute/tolerance-scaled qualification; only upper-overflow cases are excluded from its finite-output gate.",
            "relative_error_caveat": "Relative errors near zeros or underflow can be large. Errors outside JSON float64 range are null with full high-precision strings retained, never replaced by zero.",
            "ferrers_upstream_derivative": "Derived from upstream lpmv values via the displayed argument-derivative identity; not an upstream exposed derivative API.",
            "wigner_scope": "Orders 23,24,25 near the observed L24 discrepancy; independent factorial sum at both working precisions. Matrix entries use the actual public rotate API and complete single-polarization block; scalar wignersmalld/wignerd are separate implementation paths. D[m_out,m_in]=exp(-i*(m_out*alpha+m_in*gamma))*d[m_out,m_in](beta).",
            "mie_scope": "Homogeneous isotropic nonchiral spheres in vacuum, including lossy/magnetic media; no independent multilayer/chiral formula in this collector.",
            "independence": "mpmath arbitrary-precision Bessel/hypergeometric routines, direct Mie ratios and factorial Wigner sums; no SciPy/native values in the high-precision reference.",
            "tolerances": TOLERANCES,
            "sources": SOURCES,
        },
        "elapsed_seconds": time.monotonic() - started,
        "complete": len(rows) == planned,
        "passed": all(
            row.get("reference_stable")
            and (
                not row.get("float64_no_overflow", False)
                or row["backends"].get("treams-rs", {}).get("status") == "passed"
            )
            for row in rows
        ),
        "counts": {
            "cases": len(rows),
            "reference_overflow_cases": sum(
                row.get("float64_no_overflow") is False for row in rows
            ),
            "reference_subnormal_cases": sum(
                row.get("float64_subnormal_components", 0) > 0 for row in rows
            ),
            "reference_below_min_subnormal_cases": sum(
                row.get("float64_below_min_subnormal_components", 0) > 0 for row in rows
            ),
            "reference_rounds_to_zero_cases": sum(
                row.get("float64_rounds_to_zero_components", 0) > 0 for row in rows
            ),
            "native_status": dict(
                Counter(
                    row["backends"].get("treams-rs", {}).get("status", "error")
                    for row in rows
                )
            ),
            "upstream_status": dict(
                Counter(
                    row["backends"].get("treams", {}).get("status", "error")
                    for row in rows
                )
            ),
        },
        "cases": rows,
        "observations": [
            observation for row in rows for observation in observations(row)
        ],
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    # A completed qualification is data, including failed observations.
    return 0


if __name__ == "__main__":
    sys.exit(main())
