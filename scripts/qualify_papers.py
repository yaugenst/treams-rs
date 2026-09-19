# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1", "scipy>=1.14,<1.17", "treams==0.4.5"]
# ///
"""Recompute author-provided optical spectra with the installed Rust extension.

Run from the checkout: uv run --no-sync python scripts/qualify_papers.py
See docs/paper-qualification.md for provenance and the limits of each comparison.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import treams
from numpy.testing import assert_allclose
from scipy import constants

import treams_rs as tr
from treams_rs import _native

ROOT = Path(__file__).resolve().parents[1]
CPC_COMMIT = "1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39"
EBEAM_COMMIT = "9aa9974d0e56c60873dc880f11557e622d233ea9"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def python_source_digest():
    checksum = hashlib.sha256()
    for path in sorted(Path(tr.__file__).parent.glob("*.py")):
        checksum.update(path.name.encode())
        checksum.update(hashlib.sha256(path.read_bytes()).digest())
    return checksum.hexdigest()


def electron_spectrum_point(
    lib,
    energy,
    *,
    cylindrical=False,
    order=4,
    phi=0.0,
    dispersion_hbar=constants.hbar / constants.e,
):
    """Paper Eqs. 6, 21, 25 and 26: an aloof electron in vacuum, beta=0.7.

    Distances are nm. Return CL/EELS in 1/eV (sphere) or 1/(eV nm)
    (infinite cylinder). Only the requested incident field is constructed.
    """
    k0 = energy / (constants.hbar / constants.e) / constants.c * 1e-9
    beta = 0.7
    kz = k0 / beta
    impact = [60 * np.cos(phi), 60 * np.sin(phi), 0]
    source = (tr.CylindricalBasis if lib is tr else lib.CylindricalWaveBasis).default(
        [kz], 0, positions=[impact]
    )
    destination = (
        tr.CylindricalBasis if lib is tr else lib.CylindricalWaveBasis
    ).default([kz], order)
    coefficients = np.zeros(len(source), complex)
    coefficients[np.asarray(source.pol) == 1] = (
        1j
        * constants.e
        * kz
        * np.sqrt(1 - beta**2)
        / (4 * constants.c * constants.epsilon_0)
    )
    incident = (tr.operators if lib is tr else lib).expand(
        (destination, source), ("regular", "singular"), k0=k0, poltype="parity"
    ) @ coefficients
    if cylindrical:
        material_energy = energy * dispersion_hbar / (constants.hbar / constants.e)
        epsilon = 3.3 - 81 / (material_energy**2 + 0.022j * material_energy)
        tm = (
            (tr.CylindricalTMatrix if lib is tr else lib.TMatrixC)
            .cylinder([kz], order, k0, 50, [epsilon, 1])
            .changepoltype("parity")
        )
        incident = np.asarray(incident)
        scattered = np.asarray(tm) @ incident
        factor = (
            4
            * constants.epsilon_0
            * 1e-9
            / (k0**2 * np.pi * (constants.hbar / constants.e) * constants.hbar)
        )
        # Every axial channel is closed because kz > k0: no radiated CL.
        # The electron's evanescent EELS pairing is bilinear, without conjugation.
        return np.array([0.0, (incident @ scattered).real * factor])
    tm = lib.TMatrix.sphere(order, k0, 50, [16 + 0.5j, 1], poltype="parity")
    incident = np.asarray(
        (tr.operators if lib is tr else lib).expand(
            (tm.basis, destination), k0=k0, poltype="parity"
        )
        @ incident
    )
    scattered = np.asarray(tm) @ incident
    factor = 1 / (
        k0**2
        * np.pi
        * energy
        * np.sqrt(constants.mu_0 / constants.epsilon_0)
        * constants.hbar
    )
    return np.array(
        [
            np.vdot(scattered, scattered).real * factor,
            -np.vdot(incident, scattered).real * factor,
        ]
    )


def cpc_sphere_point(lib, k0, order=4):
    tm = lib.TMatrix.sphere(order, k0, 75, [16 + 0.5j, 1])
    dipole = tm[(tr.SphericalBasis if lib is tr else lib.SphericalWaveBasis).default(1)]
    return np.array(
        [tm.xs_sca_avg, tm.xs_ext_avg, dipole.xs_sca_avg, dipole.xs_ext_avg]
    ) / (np.pi * 75**2)


def cpc_slab_point(lib, k0):
    basis = (tr.PlaneWavePorts if lib is tr else lib.PlaneWaveBasisByComp).default(
        [0, 0.5 * k0]
    )
    layer = (tr.SMatrix if lib is tr else lib.SMatrices).slab(
        50, basis, k0, [1, (12.4 + 1j, 1 + 0.1j, 0.5 + 0.05j), (2, 2)]
    )
    return np.asarray([layer.tr([1, 0]), layer.tr([0, 1])]).reshape(-1)


def cpc_array_point(lib, k0, order=3):
    lattice = lib.Lattice.square(500)
    kpar = [0, 0.3 * k0]
    unit_cell = lib.TMatrix.sphere(order, k0, 100, [(4, 1, 0.05), 1])
    basis = (tr.PlaneWavePorts if lib is tr else lib.PlaneWaveBasisByComp).diffr_orders(
        kpar, lattice, 0.02
    )
    if lib is tr:
        incident = lib.plane_wave(
            [0, 0.3 * k0, k0 * np.sqrt(1 - 0.3**2)], [1, 0, 0], k0=k0
        ).expand(basis)
    else:
        incident = lib.plane_wave(kpar, [1, 0, 0], k0=k0, basis=basis, material=1)
    slab = (tr.SMatrix if lib is tr else lib.SMatrices).slab(10, basis, k0, [1, 3, 1])
    gap = (tr.SMatrix if lib is tr else lib.SMatrices).propagation(
        [0, 0, 100], basis, k0, 1
    )
    if lib is tr:
        # rs makes coupling explicit at the constructor; upstream expects it done.
        array = tr.solve_periodic(unit_cell, lattice=lattice, kpar=kpar).to_smatrix(
            basis
        )
    else:
        response = unit_cell.latticeinteraction.solve(lattice, kpar)
        array = (tr.SMatrix if lib is tr else lib.SMatrices).from_array(response, basis)
    return np.asarray(
        (tr.SMatrix if lib is tr else lib.SMatrices)
        .stack([slab, gap, array])
        .tr(incident)
    )


def compare_curve(function, grid, *, oracle_rtol=2e-9, oracle_atol=2e-11, **kwargs):
    started = time.perf_counter()
    native = np.array([function(tr, float(x), **kwargs) for x in grid])
    native_seconds = time.perf_counter() - started
    started = time.perf_counter()
    upstream = np.array([function(treams, float(x), **kwargs) for x in grid])
    upstream_seconds = time.perf_counter() - started
    assert_allclose(native, upstream, rtol=oracle_rtol, atol=oracle_atol)
    return {
        "grid": np.asarray(grid).tolist(),
        "native": native.tolist(),
        "upstream": upstream.tolist(),
        "oracle_rtol": oracle_rtol,
        "oracle_atol": oracle_atol,
        "max_absolute_oracle_error": float(np.max(abs(native - upstream))),
        "execution_seconds_not_a_benchmark": {
            "native": native_seconds,
            "upstream": upstream_seconds,
        },
    }


def qualify_ebeam():
    fixture = ROOT / "benchmarks/papers/ebeam-author-reference.json"
    references = json.loads(fixture.read_text())
    executed_path = ROOT / "benchmarks/papers/ebeam-notebook-execution.json"
    executed = json.loads(executed_path.read_text())
    results = {}
    for name, cylindrical, bounds, order in (
        ("sphere", False, (2.0, 5.0), 4),
        ("cylinder", True, (2.5, 4.5), 12),
    ):
        energies = np.linspace(*bounds, 50)
        curve = compare_curve(
            electron_spectrum_point, energies, cylindrical=cylindrical, order=order
        )
        original = references["cases"][name]
        expected = np.column_stack([original["cl"], original["eels"]])
        actual = np.asarray(curve["native"])
        # Original author tests print sphere values to 8 decimal places.
        assert_allclose(actual, expected, rtol=0, atol=1e-8)
        assert np.all(actual >= 0)
        assert np.all(actual[:, 0] <= actual[:, 1] + 1e-12)
        selected = np.array([0, 16, 30, 38, 49])
        refined = np.array(
            [
                electron_spectrum_point(
                    tr, float(energies[i]), cylindrical=cylindrical, order=order + 2
                )
                for i in selected
            ]
        )
        curve.update(
            {
                "quantity": ["cathodoluminescence", "electron_energy_loss"],
                "unit": "1/(eV nm)" if cylindrical else "1/eV",
                "energy_unit": "eV",
                "order": order,
                "published_reference": expected.tolist(),
                "published_atol": 1e-8,
                "max_absolute_published_error": float(np.max(abs(actual - expected))),
                "checks": {"nonnegative": True, "cl_bounded_by_eels": True},
                "convergence": {
                    "indices": selected.tolist(),
                    "refined_order": order + 2,
                    "refined_native": refined.tolist(),
                    "relative_change": (
                        abs(refined - actual[selected])
                        / np.maximum(abs(refined), 1e-30)
                    ).tolist(),
                    "scope": "Measured truncation sensitivity, not an imposed convergence claim.",
                },
            }
        )
        results[name] = curve
        # The figure notebooks use a different grid from the stored tests.
        # The cylinder notebook also rounds hbar in its material dispersion.
        inverse_wavelength = np.linspace(
            1 / (500 if cylindrical else 600), 1 / 250, 200
        )
        notebook_energy = (
            2
            * np.pi
            * inverse_wavelength
            * ((constants.hbar / constants.e) * constants.c * 1e9)
        )
        dispersion_hbar = 6.582e-16 if cylindrical else constants.hbar / constants.e
        notebook = compare_curve(
            electron_spectrum_point,
            notebook_energy,
            cylindrical=cylindrical,
            order=order,
            dispersion_hbar=dispersion_hbar,
        )
        notebook["wavelength_nm"] = (1 / inverse_wavelength).tolist()
        notebook["dispersion_hbar_eV_seconds"] = dispersion_hbar
        notebook["scope"] = (
            "Full original 200-point notebook grid and dispersion constants; fresh upstream calculation, not digitized figure pixels."
        )
        notebook["source_correction"] = (
            'Explicit .changepoltype("parity") for the cylinder T-matrix, matching the author regression test; omitted by the original notebook.'
            if cylindrical
            else None
        )
        original_values = np.asarray(
            executed["cases"][name]["repaired" if cylindrical else "original"][
                "values_cl_eels"
            ]
        )
        assert_allclose(notebook["native"], original_values, rtol=2e-9, atol=2e-11)
        notebook["max_absolute_executed_notebook_error"] = float(
            np.max(abs(np.asarray(notebook["native"]) - original_values))
        )
        curve["author_notebook_grid"] = notebook
    return {
        "paper": references["paper"],
        "code_commit": EBEAM_COMMIT,
        "reference_sha256": digest(fixture),
        "executed_notebook_sha256": digest(executed_path),
        "curves": results,
    }


def qualify_cpc():
    curves = {}
    for name, function, bounds, count in (
        ("sphere", cpc_sphere_point, (1 / 700, 1 / 300), 200),
        ("chiral_slab", cpc_slab_point, (1 / 1000, 1 / 300), 50),
        ("sphere_array_above_slab", cpc_array_point, (1 / 600, 1 / 350), 100),
    ):
        grid = 2 * np.pi * np.linspace(*bounds, count)
        if name == "sphere_array_above_slab":
            # The author's final sample is an exact Rayleigh anomaly. Upstream
            # regularizes lattice/radiation poles with finite surrogates.
            grid = grid[:-1]
        curve = compare_curve(function, grid)
        actual = np.asarray(curve["native"])
        assert np.all(actual >= -1e-12)
        if name == "sphere":
            assert np.all(actual[:, 0] <= actual[:, 1] + 1e-12)
            assert np.all(actual[:, 2] <= actual[:, 3] + 1e-12)
        elif name == "chiral_slab":
            assert np.all(actual[:, [0, 2]] + actual[:, [1, 3]] <= 1 + 1e-12)
        else:
            assert_allclose(actual.sum(axis=1), 1, rtol=0, atol=1e-10)
            selected = [0, 25, 50, 75, 98]
            refined = np.array(
                [function(tr, float(grid[i]), order=5) for i in selected]
            )
            curve["convergence"] = {
                "indices": selected,
                "original_order": 3,
                "refined_order": 5,
                "refined_native": refined.tolist(),
                "max_absolute_change": float(np.max(abs(refined - actual[selected]))),
                "scope": "Measured truncation sensitivity of the original author's lmax=3 recipe.",
            }
            curve["excluded_author_samples"] = [
                {
                    "index": 99,
                    "wavelength_nm": 350,
                    "reason": "Exact lattice/radiation threshold poles are not assigned upstream's finite surrogates.",
                }
            ]
            wavelengths = 350 * (1 + np.array([-1e-4, 1e-4, -1e-6, 1e-6]))
            approach = compare_curve(function, 2 * np.pi / wavelengths)
            transmission = np.asarray(approach["native"])[:, 0]
            power_error = float(np.max(abs(np.sum(approach["native"], axis=1) - 1)))
            assert power_error < 2e-10
            assert abs(transmission[2] - transmission[3]) < abs(
                transmission[0] - transmission[1]
            )
            approach["wavelength_nm"] = wavelengths.tolist()
            approach["max_power_balance_error"] = power_error
            approach["scope"] = (
                "Two-sided approach to the Rayleigh anomaly, not evaluation at the singular endpoint."
            )
            curve["threshold_approach"] = approach
        curve["grid_unit"] = "vacuum_wave_number_per_nm"
        curves[name] = curve
    return {
        "paper": "https://doi.org/10.1016/j.cpc.2023.109076",
        "code_commit": CPC_COMMIT,
        "scope": "Original companion-code spectra referenced by Table 3; array endpoint excluded explicitly. Not Figure 5 quasi-BIC reproduction.",
        "curves": curves,
    }


def plot_result(result, output):
    """Optional plotting dependency; the numerical qualification needs no GUI."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for axis, name, title in zip(
        axes[:2],
        ("sphere", "cylinder"),
        ("Electron-excited dielectric sphere", "Electron-excited metallic cylinder"),
        strict=True,
    ):
        curve = result["ebeam"]["curves"][name]
        fine = curve["author_notebook_grid"]
        for column, label, color in ((0, "CL", "#c54c39"), (1, "EELS", "#2265a0")):
            axis.plot(
                fine["grid"],
                np.asarray(fine["native"])[:, column],
                color=color,
                label=f"Rust {label}",
            )
            axis.scatter(
                curve["grid"],
                np.asarray(curve["published_reference"])[:, column],
                facecolors="none",
                edgecolors=color,
                s=15,
                label=f"Author data {label}",
            )
        axis.set(xlabel="Photon energy (eV)", ylabel=curve["unit"], title=title)
        axis.legend(fontsize=7)
    curve = result["cpc"]["curves"]["sphere_array_above_slab"]
    wavelength = 2 * np.pi / np.asarray(curve["grid"])
    for column, label, color in (
        (0, "Transmission", "#2265a0"),
        (1, "Reflection", "#c54c39"),
    ):
        axes[2].plot(
            wavelength,
            np.asarray(curve["native"])[:, column],
            color=color,
            label=f"Rust {label}",
        )
        axes[2].plot(
            wavelength,
            np.asarray(curve["upstream"])[:, column],
            "--",
            color="black",
            linewidth=0.7,
            label="Original treams recipe" if column == 0 else None,
        )
    axes[2].set(
        xlabel="Wavelength (nm)",
        ylabel="Power fraction",
        title="Periodic spheres above a slab",
    )
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "ebeam", "cpc"), default="all")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save PNG/SVG (requires matplotlib and --case all)",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/papers/qualification.json"
    )
    args = parser.parse_args()
    if args.plot and args.case != "all":
        parser.error("--plot requires --case all")
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "treams")
        },
        "native_sha256": digest(_native.__file__),
        "python_source_sha256": python_source_digest(),
        "script_sha256": digest(__file__),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "'where' used without 'out'.*", UserWarning)
        if args.case in ("all", "ebeam"):
            result["ebeam"] = qualify_ebeam()
            print(
                "Passed electron-beam sphere and cylinder author-reference spectra.",
                flush=True,
            )
        if args.case in ("all", "cpc"):
            result["cpc"] = qualify_cpc()
            print(
                "Passed CPC sphere, chiral slab, and periodic sphere/slab spectra.",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot_result(result, args.output)


if __name__ == "__main__":
    main()
