# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Freeze follow-up cases whose powers distinguish helicity and incident side."""

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from cases import task
from reference import derivative, jsonable, native, upstream, z

OUT = Path(__file__).resolve().parents[2] / "benchmarks/agent-api"
TASKS = [
    task(
        "chiral_stack_sides",
        "forward",
        "A lossy reciprocal chiral two-layer stack is ordered negative-z to positive-z. For both normal-incidence directions and both helicities, compute transmitted, reflected and absorbed power fractions. Preserve layer order when changing incidence. epsilon_layers and kappa_layers contain [real,imag] values; mu=1 throughout. The exterior permittivities are negative_epsilon and positive_epsilon. Return exactly the keys transmission, reflection, absorption, each with shape (wavelengths, 2 sides, 2 helicities): sides negative then positive, helicities positive then negative relative to propagation. Preserve the supplied wavelength order. The complex chirality and asymmetric losses distinguish the cases.",
        {
            "thicknesses": [0.08, 0.13],
            "epsilon_layers": [[2.3, 0.1], [3.8, 0.2]],
            "kappa_layers": [[0.12, 0.015], [-0.08, 0.005]],
            "negative_epsilon": 1.0,
            "positive_epsilon": 2.25,
            "wavelengths": [0.73, 0.61, 0.86],
        },
        {
            "thicknesses": [0.11, 0.09],
            "kappa_layers": [[-0.05, 0.01], [0.09, -0.008]],
            "negative_epsilon": 1.44,
            "positive_epsilon": 1.0,
            "wavelengths": [0.69, 0.83],
        },
        transfer=True,
    ),
    task(
        "circular_absorption_sensitivity",
        "sensitivity",
        "A homogeneous lossy reciprocal chiral slab in vacuum has epsilon=[real,imag] and kappa=kappa_real+1j*kappa_imag, with mu=1. For normal incidence along +z, calculate the absorbed-power difference A_positive_helicity minus A_negative_helicity. Return exactly two scalar keys: value (that difference) and gradient (its analytic derivative with respect to kappa_imag). Hold all other inputs fixed, including kappa_real. The two helicities are defined relative to propagation. Use native analytic derivatives; a finite difference is only an optional validation.",
        {
            "thickness": 0.17,
            "epsilon": [2.3, 0.1],
            "kappa_real": 0.12,
            "kappa_imag": 0.015,
            "wavelength": 0.73,
        },
        {
            "thickness": 0.13,
            "epsilon": [3.8, 0.2],
            "kappa_real": -0.08,
            "kappa_imag": -0.005,
            "wavelength": 0.81,
        },
        transfer=True,
    ),
]


def powers(config, lib=upstream):
    basis_type = lib.PlaneWaveBasisByComp if lib is upstream else lib.PlaneWavePorts
    matrix_type = lib.SMatrices if lib is upstream else lib.SMatrix
    ports = basis_type.default([0, 0])
    media = [
        config["negative_epsilon"],
        *[
            lib.Material(z(epsilon), 1, z(kappa))
            for epsilon, kappa in zip(
                config["epsilon_layers"], config["kappa_layers"], strict=True
            )
        ],
        config["positive_epsilon"],
    ]
    rows = []
    for wavelength in config["wavelengths"]:
        k0 = 2 * np.pi / wavelength
        network = matrix_type.slab(config["thicknesses"], ports, k0, media)
        sides = []
        for side, direction, sign, medium in (
            ("negative", "up", 1, media[0]),
            ("positive", "down", -1, media[-1]),
        ):
            helicities = []
            for pol in (1, 0):
                if lib is upstream:
                    incident = lib.PhysicsArray(
                        (ports.pol == pol).astype(complex), modetype=direction
                    )
                    transmission, reflection = network.tr(incident)
                else:
                    incident = lib.plane_wave([0, 0, sign], pol, k0=k0, medium=medium)
                    balance = network.power(incident, side=side)
                    transmission, reflection = balance.transmission, balance.reflection
                helicities.append(
                    [transmission, reflection, 1 - transmission - reflection]
                )
            sides.append(helicities)
        rows.append(sides)
    array = np.asarray(rows)
    return {
        key: array[..., i]
        for i, key in enumerate(("transmission", "reflection", "absorption"))
    }


def evaluate(case_id, config, lib=upstream):
    if case_id == "chiral_stack_sides":
        return powers(config, lib)

    def difference(kappa_imag):
        absorption = powers(
            {
                "thicknesses": [config["thickness"]],
                "epsilon_layers": [config["epsilon"]],
                "kappa_layers": [[config["kappa_real"], kappa_imag]],
                "negative_epsilon": 1.0,
                "positive_epsilon": 1.0,
                "wavelengths": [config["wavelength"]],
            },
            lib,
        )["absorption"]
        return absorption[0, 0, 0] - absorption[0, 0, 1]

    return {
        "value": difference(config["kappa_imag"]),
        "gradient": derivative(difference, config["kappa_imag"]),
    }


def main():
    warnings.filterwarnings("ignore", message="'where' used without 'out'")
    for case in TASKS:
        case["expected"] = []
        for config in case["inputs"]:
            expected = evaluate(case["id"], config)
            actual = evaluate(case["id"], config, native)
            for key in expected:
                np.testing.assert_allclose(
                    actual[key], expected[key], rtol=3e-6, atol=1e-8
                )
            if case["id"] == "chiral_stack_sides":
                assert (
                    np.max(
                        np.abs(
                            expected["transmission"][..., 0]
                            - expected["transmission"][..., 1]
                        )
                    )
                    > 0.01
                )
                assert (
                    np.max(
                        np.abs(
                            expected["reflection"][:, 0] - expected["reflection"][:, 1]
                        )
                    )
                    > 1e-4
                )
            case["expected"].append(jsonable(expected))
    files = {
        "polarization-reference.json": TASKS,
        "polarization-prompts.json": [
            {"id": t["id"], "prompt": t["prompt"]} for t in TASKS
        ],
    }
    for name, data in files.items():
        with (OUT / name).open("x") as file:
            file.write(json.dumps(data, indent=2) + "\n")
    record = {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": "Source audit found reversed port-order guidance. Earlier real-chirality/lossless power cases cannot distinguish helicity or incident side reliably; these supplementary cases explicitly do. Preserve every original task, reference and result.",
        "design": "Two new cases x four models x two fresh attempts after corrected wheel freezes. Target >=90% full passes. Existing transfer tasks become confirmation tasks after the documentation correction; do not call those unseen again.",
        "hashes": {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in files
        },
    }
    with (OUT / "polarization-protocol.json").open("x") as file:
        file.write(json.dumps(record, indent=2) + "\n")
    print(
        "Frozen two nondegenerate polarization cases; upstream and physical-wave controls agree."
    )


if __name__ == "__main__":
    main()
