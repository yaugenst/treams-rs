# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["treams==0.4.5", "numpy==2.5.3", "scipy==1.16.3"]
# ///
"""Generate the browser fixture using only upstream treams.

Run from the repository root: uv run --script scripts/generate_wasm_reference.py
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import treams


def interleaved(value, order="C"):
    flat = np.asarray(value, dtype=np.complex128).ravel(order=order)
    return np.column_stack((flat.real, flat.imag)).ravel().tolist()


def material(epsilon, mu=1.0, kappa=0.0):
    return {
        "epsilon": interleaved(epsilon),
        "mu": interleaved(mu),
        "kappa": interleaved(kappa),
    }


def upstream_material(value):
    return treams.Material(
        *(complex(*value[name]) for name in ("epsilon", "mu", "kappa"))
    )


def qualify(case):
    embedding = treams.Material()
    matrices = [
        treams.TMatrix.sphere(
            case["lmax"],
            case["k0"],
            sphere["radii"],
            [*(upstream_material(m) for m in sphere["materials"]), embedding],
            poltype="helicity",
        )
        for sphere in case["spheres"]
    ]
    matrix = matrices[0]
    if len(matrices) > 1:
        matrix = treams.TMatrix.cluster(matrices, case["positions"]).interaction.solve()
    wave = treams.plane_wave(
        case["direction"],
        case["polarization"],
        k0=case["k0"],
        material=embedding,
        poltype="helicity",
    )
    incident = wave.expand(matrix.basis)
    scattered = matrix @ incident
    field = (
        treams.efield(
            case["points"],
            basis=matrix.basis,
            k0=case["k0"],
            material=embedding,
            modetype="singular",
            poltype="helicity",
        )
        @ scattered
    )
    sca, ext = (float(value) for value in matrix.xs(wave))
    assert sca > 0 and ext > 0
    if case["lossless"]:
        np.testing.assert_allclose(sca, ext, rtol=2e-12, atol=1e-15)
    else:
        assert ext > sca
    for value in (matrix, incident, scattered, field):
        assert np.isfinite(value).all()
    return {
        **case,
        "mode_count": len(matrix),
        "tmatrix": interleaved(matrix, order="F"),
        "incident": interleaved(incident),
        "scattered": interleaved(scattered),
        "electric_field": interleaved(field),
        "cross_sections": {"scattering": sca, "extinction": ext},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("crates/treams-wasm/tests/reference.json")
    )
    args = parser.parse_args()
    common = {
        "lmax": 3,
        "k0": 1.3,
        "embedding": material(1.0),
        "direction": [0.36, -0.48, 0.8],
        "polarization": 1,
        "positions": [[0.0, 0.0, 0.0]],
        "points": [[1.1, 0.8, 0.7], [-0.9, -0.8, 0.6], [0.1, 0.7, -1.2]],
        "lossless": False,
    }
    cases = [
        {
            **common,
            "name": "multilayer_chiral_sphere",
            "spheres": [
                {
                    "radii": [0.12, 0.3],
                    "materials": [
                        material(2.1 + 0.02j, 1.2, 0.03),
                        material(3.0 + 0.04j, 1.0, 0.08),
                    ],
                }
            ],
        },
        {
            **common,
            "name": "two_sphere_cluster",
            "spheres": [
                {"radii": [0.2], "materials": [material(2.1 + 0.02j)]},
                {"radii": [0.3], "materials": [material(3.2 + 0.01j)]},
            ],
            "positions": [[-0.5, 0.1, -0.2], [0.55, -0.2, 0.35]],
        },
        {
            **common,
            "name": "lossless_sphere",
            "lmax": 4,
            "spheres": [{"radii": [0.35], "materials": [material(4.0)]}],
            "lossless": True,
        },
    ]
    result = {
        "reference": {
            name: importlib.metadata.version(name)
            for name in ("treams", "numpy", "scipy")
        },
        "conventions": {
            "complex": "interleaved real/imaginary float64",
            "tmatrix": "column-major (mode_count, mode_count)",
            "electric_field": "point-major Cartesian xyz, scattered field only",
            "basis": "particle, l=1..lmax, m=-l..l, helicity=1 then 0",
            "materials": "sphere interiors only; the embedding is appended outside",
            "cross_sections": "unit-amplitude plane wave, incident flux 0.5",
        },
        "cases": [qualify(case) for case in cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
