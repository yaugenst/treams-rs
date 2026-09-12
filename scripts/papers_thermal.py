"""Reproduce PRB 112, 054307 (2025), Fig. 2, against the authors' data.

Run from the release-built checkout with PYTHONPATH=python. The default selects
52 of the original 300 frequencies; --samples 300 evaluates the entire curve.
The fixture contains numerical data, not a copy of the authors' implementation.
"""

import argparse
import hashlib
import io
import json
import platform
import time
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

import treams_rs as rs
from treams_rs import _native

COMMIT = "ec710fc1f15f379d126659043ef1c2ce4f0eafae"
SOURCE = "https://github.com/jdmazo-vasquez/TMatricesThermalRadiation"
RAW = f"https://raw.githubusercontent.com/jdmazo-vasquez/TMatricesThermalRadiation/{COMMIT}"
FIXTURE = Path("benchmarks/papers/thermal-source.json")
C0 = 299792458.0
HBAR = 1.05457182e-34
KB = 1.380649e-23
CUTS = [164, 187, 220]


def fixture():
    if FIXTURE.exists():
        return json.loads(FIXTURE.read_text())
    # URLs are fixed to the inspected author commit, never caller-controlled.
    with urllib.request.urlopen(f"{RAW}/Data.zip") as stream:  # noqa: S310
        archive = stream.read()
    with urllib.request.urlopen(f"{RAW}/Example-SiCSpheresChain.ipynb") as stream:  # noqa: S310
        notebook = stream.read()
    names = (
        "Data_SiC_Larruquert.txt",
        "DataFig1a.dat",
        "DataFig1b.dat",
        "DataFig1c.dat",
        "kplotvals_SiCChain.dat",
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        files = {name: source.read(name) for name in names}
    values = {name: np.loadtxt(io.BytesIO(data)) for name, data in files.items()}
    angular = values["DataFig1b.dat"][:, CUTS]
    np.testing.assert_array_equal(angular, values["DataFig1c.dat"][:, 1:])
    result = {
        "paper": "Studying thermal radiation with T matrices",
        "doi": "10.1103/41m5-9ztm",
        "figure": "2(a), 2(c); source files use the earlier numbering 1(a), 1(c)",
        "repository": SOURCE,
        "commit": COMMIT,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "notebook_sha256": hashlib.sha256(notebook).hexdigest(),
        "file_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
        "material_columns": ["frequency_hz", "epsilon_real", "epsilon_imag"],
        "material": values["Data_SiC_Larruquert.txt"].tolist(),
        "frequencies_hz": np.logspace(np.log10(3e12), np.log10(1.2e14), 300).tolist(),
        "wavenumbers_cm_inverse": values["kplotvals_SiCChain.dat"].tolist(),
        "absorption_um_squared": values["DataFig1a.dat"][:, 1].tolist(),
        "thermal_energy_j_per_m3_per_cm_inverse": values["DataFig1a.dat"][
            :, 2
        ].tolist(),
        "polar_angles": np.linspace(0.001, np.pi, 200).tolist(),
        "angular_cut_frequency_indices": CUTS,
        "angular_photons_per_second_per_cm_inverse": angular.tolist(),
        "source_notes": [
            "The committed angular cuts equal columns 164, 187, 220 of DataFig1b exactly.",
            "They already include 200*pi conversion; the notebook's final unscaled save cell is stale.",
            "Q uses T.conj().T @ T, matching executable source, despite its markdown writing T @ T.conj().T.",
        ],
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(result, indent=2) + "\n")
    return result


def chain_matrix(module, frequency, epsilon, order):
    """Four 250 nm SiC spheres spaced 520 nm, expanded about the origin."""
    k0 = 2 * np.pi * frequency / C0
    sphere = module.TMatrix.sphere(
        order,
        k0,
        250e-9,
        [module.Material(epsilon), module.Material()],
        poltype="helicity",
    )
    positions = [[0.0, 0.0, z * 520e-9] for z in (-1.5, -0.5, 0.5, 1.5)]
    cluster = module.TMatrix.cluster([sphere] * 4, positions).interaction.solve()
    return np.asarray(cluster.expand(module.SphericalWaveBasis.default(order)))


def absorption(matrix, frequency):
    k0 = 2 * np.pi * frequency / C0
    # Trace of Q = -2T† - 2T - 4T†T, without materializing another dense product.
    return float(
        -2
        * np.pi
        * (np.trace(matrix).real + np.vdot(matrix, matrix).real)
        / k0**2
        * 1e12
    )


def angular_cut(matrix, frequency, angles, order, *, phi=np.pi / 4, helicity=1):
    """Directional Kirchhoff law from -k, doubled for this achiral object."""
    basis = rs.SphericalWaveBasis.default(order)
    modes = list(basis)
    phi += np.pi
    coefficients = np.asarray(
        [
            [
                np.sqrt((2 * degree + 1) / (4 * np.pi))
                * rs.special.wignerd(degree, m, helicity, phi, np.pi - angle, 0)
                if polarization == int(helicity == 1)
                else 0.0
                for _, degree, m, polarization in modes
            ]
            for angle in angles
        ],
        dtype=complex,
    )
    q = -2 * matrix.conj().T - 2 * matrix - 4 * matrix.conj().T @ matrix
    # The paper's modal formula uses right singular vectors. This retains that
    # convention, including the absolute eigenvalues of roundoff-size Q modes.
    _, values, right = np.linalg.svd(q, hermitian=True)
    projection = coefficients.conj() @ right.T
    weighted = np.abs(projection) ** 2 @ values
    occupation = np.expm1(HBAR * 2 * np.pi * frequency / (KB * 500.0))
    return 200.0 * C0 * weighted / occupation


def relative_error(actual, expected):
    return float(
        np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-300))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--backend", choices=("both", "rust"), default="both")
    parser.add_argument(
        "--upstream-reference",
        type=Path,
        help="Reuse the identical-parameter upstream rows of an earlier qualification",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/papers/thermal-result.json")
    )
    parser.add_argument(
        "--indices",
        help="Explicit comma-separated source indices for a bounded diagnostic",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Evaluate degrees 6..12 at representative points, and 14/16 at the low-frequency endpoint",
    )
    args = parser.parse_args()
    if not 1 <= args.samples <= 300:
        parser.error("--samples must be between 1 and 300")
    data = fixture()
    frequencies = np.asarray(data["frequencies_hz"])
    material = np.asarray(data["material"])
    epsilons = np.interp(frequencies, material[:, 0], material[:, 1]) + 1j * np.interp(
        frequencies, material[:, 0], material[:, 2]
    )
    expected = np.asarray(data["absorption_um_squared"])
    if args.indices:
        indices = sorted({int(index) for index in args.indices.split(",")})
    else:
        indices = sorted(
            set(np.linspace(0, 299, args.samples).round().astype(int)) | set(CUTS)
        )
    old_result = None
    reference_rows = {}
    if args.upstream_reference:
        old_result = json.loads(args.upstream_reference.read_text())
        assert old_result["complete"]
        assert old_result["source_commit"] == COMMIT
        assert old_result["lmax"] == args.lmax
        assert (
            old_result["fixture_sha256"]
            == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        )
        reference_rows = {
            row["index"]: row
            for row in old_result["rows"]
            if "upstream_absorption_um_squared" in row
        }
    upstream = None
    if args.backend == "both" and not args.upstream_reference:
        import treams

        upstream = treams
    source_digest = hashlib.sha256()
    for path in sorted(Path(rs.__file__).parent.glob("*.py")):
        source_digest.update(path.name.encode())
        source_digest.update(hashlib.sha256(path.read_bytes()).digest())
    result = {
        "paper_doi": data["doi"],
        "source_commit": COMMIT,
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(
            Path(_native.__file__).read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_source_sha256": source_digest.hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "lmax": args.lmax,
        "parameters": {
            "sphere_count": 4,
            "radius_m": 250e-9,
            "center_spacing_m": 520e-9,
            "temperature_k": 500.0,
            "polarization": "helicity",
        },
        "timing_note": "Wall times from a numerical qualification, not isolated performance benchmarks.",
        "source_frequency_count": 300,
        "selected_frequency_count": len(indices),
        "upstream_version": old_result["upstream_version"]
        if old_result
        else None
        if upstream is None
        else version("treams"),
        "upstream_reference_path": None
        if args.upstream_reference is None
        else args.upstream_reference.name,
        "upstream_reference_sha256": None
        if args.upstream_reference is None
        else hashlib.sha256(args.upstream_reference.read_bytes()).hexdigest(),
        "rows": [],
        "angular_cuts": [],
        "convergence": [],
    }
    for index in indices:
        frequency, epsilon = frequencies[index], epsilons[index]
        start = time.perf_counter()
        matrix = chain_matrix(rs, frequency, epsilon, args.lmax)
        sigma = absorption(matrix, frequency)
        row = {
            "index": int(index),
            "frequency_hz": float(frequency),
            "author_absorption_um_squared": float(expected[index]),
            "rust_absorption_um_squared": sigma,
            "rust_seconds": time.perf_counter() - start,
            "author_relative_error": relative_error(sigma, expected[index]),
        }
        if upstream is not None:
            start = time.perf_counter()
            reference_matrix = chain_matrix(upstream, frequency, epsilon, args.lmax)
            reference = absorption(reference_matrix, frequency)
            row.update(
                upstream_absorption_um_squared=reference,
                upstream_seconds=time.perf_counter() - start,
                upstream_relative_error=relative_error(sigma, reference),
                upstream_author_relative_error=relative_error(
                    reference, expected[index]
                ),
            )
        elif index in reference_rows:
            previous = reference_rows[index]
            assert previous["frequency_hz"] == float(frequency)
            reference = previous["upstream_absorption_um_squared"]
            row.update(
                upstream_absorption_um_squared=reference,
                upstream_seconds=previous["upstream_seconds"],
                upstream_relative_error=relative_error(sigma, reference),
                upstream_author_relative_error=relative_error(
                    reference, expected[index]
                ),
            )
        if index in CUTS:
            values = angular_cut(matrix, frequency, data["polar_angles"], args.lmax)
            sample = np.array([0, 33, 66, 99, 132, 165, 199])
            angles = np.asarray(data["polar_angles"])[sample]
            rotated = angular_cut(matrix, frequency, angles, args.lmax, phi=0.37)
            opposite = angular_cut(matrix, frequency, angles, args.lmax, helicity=-1)
            original = np.asarray(data["angular_photons_per_second_per_cm_inverse"])[
                :, CUTS.index(index)
            ]
            result["angular_cuts"].append(
                {
                    "index": int(index),
                    "rust_values": values.tolist(),
                    "author_max_relative_error": relative_error(values, original),
                    "azimuth_invariance_relative_error": relative_error(
                        rotated, values[sample]
                    ),
                    "helicity_symmetry_relative_error": relative_error(
                        opposite, values[sample]
                    ),
                }
            )
        result["rows"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            f"{len(result['rows'])}/{len(indices)} index={index} paper_error={row['author_relative_error']:.3g}",
            flush=True,
        )
    if args.convergence:
        for index in [0, *CUTS, 299]:
            orders = (6, 8, 10, 12, 14, 16) if index == 0 else (6, 8, 10, 12)
            for order in orders:
                matrix = chain_matrix(rs, frequencies[index], epsilons[index], order)
                result["convergence"].append(
                    {
                        "index": index,
                        "lmax": order,
                        "absorption_um_squared": absorption(matrix, frequencies[index]),
                    }
                )
    rows = result["rows"]
    comparisons = [row for row in rows if "upstream_relative_error" in row]
    convergence_errors = []
    if args.convergence:
        for index in [0, *CUTS, 299]:
            checks = [row for row in result["convergence"] if row["index"] == index]
            original_order = next(row for row in checks if row["lmax"] == 10)
            convergence_errors.append(
                relative_error(
                    original_order["absorption_um_squared"],
                    checks[-1]["absorption_um_squared"],
                )
            )
    summary = {
        "finite_nonnegative_absorption": all(
            np.isfinite(row["rust_absorption_um_squared"])
            and row["rust_absorption_um_squared"] >= 0
            for row in rows
        ),
        "max_author_relative_error": max(row["author_relative_error"] for row in rows),
        "upstream_comparison_count": len(comparisons),
        "max_upstream_relative_error": max(
            (row["upstream_relative_error"] for row in comparisons), default=None
        ),
        "max_angular_author_relative_error": max(
            (cut["author_max_relative_error"] for cut in result["angular_cuts"]),
            default=None,
        ),
        "max_angular_symmetry_error": max(
            (
                max(
                    cut["azimuth_invariance_relative_error"],
                    cut["helicity_symmetry_relative_error"],
                )
                for cut in result["angular_cuts"]
            ),
            default=0.0,
        ),
        "rust_total_seconds": sum(row["rust_seconds"] for row in rows),
        "upstream_total_seconds": sum(row["upstream_seconds"] for row in comparisons),
        "max_lmax10_convergence_relative_error": max(convergence_errors, default=None),
    }
    # The paper selected jmax=10 by a >99% multipole-convergence criterion.
    # Its stored low-frequency curve also contains unbalanced-solve roundoff.
    # Record exact errors; do not claim float64 agreement with that old data.
    summary["paper_relative_tolerance"] = 0.01
    summary["paper_agreement"] = summary["max_author_relative_error"] <= 0.01 and (
        summary["max_angular_author_relative_error"] is None
        or summary["max_angular_author_relative_error"] <= 0.01
    )
    summary["passed"] = (
        summary["finite_nonnegative_absorption"]
        and summary["paper_agreement"]
        and summary["max_angular_symmetry_error"] <= 1e-8
        and (not convergence_errors or max(convergence_errors) <= 0.01)
    )
    result["summary"] = summary
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    assert summary["finite_nonnegative_absorption"], (
        "Passive SiC must not produce negative absorption"
    )
    assert summary["paper_agreement"], (
        "Published figure agreement exceeds the paper's 1% truncation criterion"
    )
    assert summary["max_angular_symmetry_error"] <= 1e-8, (
        "Axial, achiral chain must preserve azimuth and helicity symmetry"
    )
    assert not convergence_errors or max(convergence_errors) <= 0.01, (
        "jmax=10 multipole convergence exceeds 1%"
    )


if __name__ == "__main__":
    main()
