# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1", "threadpoolctl>=3.6"]
# ///
"""Record physical identities and convergence, independently of upstream treams.

Run against the installed native extension from this checkout. Individual numerical
failures remain in the JSON; exit status only reports inability to collect a report.
This is accuracy qualification, not a timing benchmark or an exhaustive proof.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from contextlib import contextmanager
from numbers import Real
from pathlib import Path

FAMILIES = (
    "coefficients",
    "fields",
    "rotation",
    "finite",
    "planar",
    "periodic",
    "ebcm",
)


class Collector:
    """Keep failed evaluations and nonfinite residuals visible in strict JSON."""

    def __init__(self):
        self.observations = []
        self.current = {}

    @contextmanager
    def case(self, identifier, family, parameters):
        self.current = {"id": identifier, "family": family, "parameters": parameters}
        start = len(self.observations)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield
            except Exception as error:
                self.observations.append(
                    {
                        **self.current,
                        "backend": "treams-rs",
                        "reference_kind": "physical_invariant",
                        "metric": "evaluation",
                        "error": None,
                        "tolerance": None,
                        "status": "error",
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
        if caught:
            messages = sorted({str(item.message) for item in caught})
            for observation in self.observations[start:]:
                observation["warnings"] = messages

    def add(self, metric, error, tolerance, **details):
        value = float(error)
        finite = math.isfinite(value) and value >= 0
        status = (
            (
                "diagnostic"
                if tolerance is None
                else "passed"
                if value <= tolerance
                else "failed"
            )
            if finite
            else "error"
        )
        row = {
            **self.current,
            "id": f"{self.current['id']}/{metric}",
            "backend": "treams-rs",
            "reference_kind": "physical_invariant",
            "metric": metric,
            "error": value if finite else None,
            "tolerance": tolerance,
            "status": status,
            **details,
        }
        if not finite:
            row["message"] = f"invalid residual: {value}"
        invalid_details = []

        def strict_numbers(detail, path):
            if isinstance(detail, Real) and not isinstance(detail, (int, bool)):
                numeric = float(detail)
                if not math.isfinite(numeric):
                    invalid_details.append({"path": path, "value": str(numeric)})
                    return None
                return numeric
            if isinstance(detail, dict):
                return {
                    key: strict_numbers(item, f"{path}/{key}")
                    for key, item in detail.items()
                }
            if isinstance(detail, (list, tuple)):
                return [
                    strict_numbers(item, f"{path}/{index}")
                    for index, item in enumerate(detail)
                ]
            return detail

        row = strict_numbers(row, "")
        if invalid_details:
            row["status"] = "error"
            row["unqualified_residual"] = row["error"]
            row["error"] = None
            row["nonfinite_details"] = invalid_details
        self.observations.append(row)


def relative(actual, expected):
    import numpy as np

    return float(
        np.linalg.norm(np.asarray(actual) - expected)
        / max(np.linalg.norm(expected), 1e-30)
    )


def unitarity(matrix):
    import numpy as np

    return float(np.max(abs(matrix.conj().T @ matrix - np.eye(len(matrix)))))


def coefficients(c, rng):
    import numpy as np

    from treams_rs import coeffs

    for index in range(24):
        size = float(rng.uniform(0.2, 3))
        epsilon = float(rng.uniform(1.1, 8))
        degree = int(rng.integers(1, 9))
        fraction = float(rng.uniform(0.2, 0.8))
        with c.case(
            f"sphere-{index}",
            "sphere coefficients",
            dict(size=size, epsilon=epsilon, degree=degree, split_fraction=fraction),
        ):
            value = coeffs.mie(degree, [size], [epsilon, 1], [1, 1], [0, 0])
            sca, ext = float(np.sum(abs(value) ** 2)), float(-np.trace(value).real)
            c.add(
                "optical_theorem_absolute",
                abs(sca - ext),
                2e-11,
                observables={"scattering": sca, "extinction": ext},
            )
            zero = coeffs.mie(degree, [size], [epsilon, epsilon], [1, 1], [0, 0])
            c.add("zero_contrast_max_absolute", np.max(abs(zero)), 2e-13)
            plain = coeffs.mie(degree, [size], [epsilon, 1], [1, 1], [0.03, 0])
            split = coeffs.mie(
                degree,
                [fraction * size, size],
                [epsilon, epsilon, 1],
                [1, 1, 1],
                [0.03, 0.03, 0],
            )
            c.add(
                "layer_split_scaled_absolute",
                np.max(abs(split - plain)) / max(1, np.max(abs(plain))),
                3.3e-11,
            )
            absorbing = coeffs.mie(degree, [size], [epsilon + 0.2j, 1], [1, 1], [0, 0])
            sca, ext = (
                float(np.sum(abs(absorbing) ** 2)),
                float(-np.trace(absorbing).real),
            )
            c.add(
                "passivity_violation_absolute",
                max(0, sca - ext, -sca),
                2e-12,
                observables={
                    "scattering": sca,
                    "extinction": ext,
                    "absorption": ext - sca,
                },
            )
    for index in range(18):
        radius = float(rng.uniform(0.1, 1.5))
        epsilon = float(rng.uniform(1.1, 6))
        order = int(rng.integers(-4, 5))
        kz = float(rng.uniform(-0.5, 0.5))
        with c.case(
            f"cylinder-{index}",
            "cylinder coefficients",
            dict(radius=radius, epsilon=epsilon, order=order, kz=kz, k0=1.3),
        ):
            args = (kz, order, 1.3)
            one = coeffs.mie_cyl(*args, [radius], [epsilon, 1], [1, 1], [0.02, 0])
            split = coeffs.mie_cyl(
                *args,
                [0.5 * radius, radius],
                [epsilon, epsilon, 1],
                [1, 1, 1],
                [0.02, 0.02, 0],
            )
            zero = coeffs.mie_cyl(
                *args, [radius], [epsilon, epsilon], [1, 1], [0.02, 0.02]
            )
            sca, ext = float(np.sum(abs(one) ** 2)), float(-np.trace(one).real)
            c.add(
                "optical_theorem_absolute",
                abs(sca - ext),
                4.1e-10,
                observables={"scattering": sca, "extinction": ext},
            )
            c.add("zero_contrast_max_absolute", np.max(abs(zero)), 2e-12)
            c.add(
                "layer_split_scaled_absolute",
                np.max(abs(split - one)) / max(1, np.max(abs(one))),
                2.1e-10,
            )


def fields(c, rng):
    import numpy as np

    import treams_rs as tr
    from treams_rs import _native

    points = np.array([[0, 0, 0], [0.2, -0.1, 0.3], [0, 0, 0.5], [-0.3, 0.3, 0.2]])
    for axis in range(3):
        direction, polarization = np.eye(3)[axis], np.eye(3)[(axis + 1) % 3]
        for degree in (2, 4, 6, 8, 10):
            with c.case(
                f"plane-{axis}-l{degree}",
                "plane expansion convergence",
                dict(
                    k0=1.3,
                    direction=direction.tolist(),
                    polarization=polarization.tolist(),
                    lmax=degree,
                    points=points.tolist(),
                ),
            ):
                basis = tr.SphericalWaveBasis.default(degree)
                wave = tr.plane_wave(direction, polarization, k0=1.3)
                actual = tr.diff.field(wave.expand(basis), points, basis, [1.3, 1.3])[0]
                expected = np.exp(1.3j * (points @ direction))[:, None] * polarization
                c.add(
                    "field_relative_l2",
                    relative(actual, expected),
                    3e-9 if degree == 10 else None,
                    reference_kind="analytic",
                    series=f"Cartesian plane wave axis {axis}",
                    x={"name": "lmax", "value": degree, "unit": "order"},
                )
    for index in range(24):
        degree = int(rng.integers(1, 7))
        order = int(rng.integers(-degree, degree + 1))
        pol = index % 2
        point = [
            float(rng.uniform(-1, 1)),
            float(rng.uniform(-1, 1)),
            float(rng.uniform(0.5, 1.5)),
        ]
        outgoing = index % 3 == 0
        k = 1.2 + 0.1j
        with c.case(
            f"maxwell-{index}",
            "spherical Maxwell identities",
            dict(
                degree=degree,
                order=order,
                polarization=pol,
                point=point,
                k=[k.real, k.imag],
                outgoing=outgoing,
            ),
        ):
            value, derivative, _ = _native.spherical_wave(
                (degree, order, pol), k, tuple(point), True, outgoing
            )
            value, derivative = np.asarray(value), np.asarray(derivative)
            curl = np.array(
                [
                    derivative[2, 1] - derivative[1, 2],
                    derivative[0, 2] - derivative[2, 0],
                    derivative[1, 0] - derivative[0, 1],
                ]
            )
            scale = max(1, np.linalg.norm(value))
            c.add(
                "curl_helicity_scaled_l2",
                np.linalg.norm(curl - (2 * pol - 1) * k * value) / scale,
                2e-10,
            )
            c.add(
                "divergence_scaled_absolute", abs(np.trace(derivative)) / scale, 2e-10
            )


def rotation(c, _rng):
    import numpy as np

    import treams_rs as tr

    for degree in (1, 3, 10, 30, 60):
        with c.case(
            f"rotation-l{degree}",
            "rotation",
            dict(degree=degree, angles=[0.2, 1.3, -0.4]),
        ):
            basis = tr.SphericalWaveBasis(
                [(degree, order, 1) for order in range(-degree, degree + 1)]
            )
            value = tr.rotate(0.2, 1.3, -0.4, basis=basis)
            inverse = tr.rotate(0.4, -1.3, -0.2, basis=basis)
            c.add(
                "unitarity_max_absolute",
                unitarity(value),
                2e-12,
                x={"name": "degree", "value": degree, "unit": "order"},
            )
            c.add(
                "inverse_max_absolute",
                np.max(abs(value @ inverse - np.eye(len(basis)))),
                2e-12,
            )
    with c.case(
        "sphere-rotation",
        "rotation",
        dict(lmax=4, k0=1.3, radius=0.4, epsilon=[3.1, 0.2], angles=[0.2, 0.7, -0.3]),
    ):
        sphere = tr.TMatrix.sphere(4, 1.3, 0.4, [3.1 + 0.2j, 1])
        c.add(
            "isotropic_rotation_relative_l2",
            relative(sphere.rotate(0.2, 0.7, -0.3).array, sphere.array),
            2e-12,
        )


def finite(c, _rng):
    import numpy as np

    import treams_rs as tr

    positions = np.array([[0, 0, 0], [0.4, -0.2, 1.4], [-0.7, 0.5, 0.3]])
    radii, epsilon, k0 = np.array([0.2, 0.25, 0.18]), [3.1, 2.5, 4], 1.3
    angle, scale = 0.73, 1.4
    rz = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    for degree in (1, 2, 3, 4):
        with c.case(
            f"cluster-l{degree}",
            "finite cluster",
            dict(
                lmax=degree,
                k0=k0,
                radii=radii.tolist(),
                epsilon=epsilon,
                positions=positions.tolist(),
                translation=[0.4, -0.3, 0.7],
                rotation_z=angle,
                scale=scale,
            ),
        ):
            particles = [
                tr.TMatrix.sphere(degree, k0, radius, [eps, 1])
                for radius, eps in zip(radii, epsilon, strict=True)
            ]
            local = tr.TMatrix.cluster(particles, positions)
            response = local.interaction.solve()
            system = local.interaction()
            residual = np.linalg.norm(system @ response.array - local.array) / (
                np.linalg.norm(system) * np.linalg.norm(response.array)
                + np.linalg.norm(local.array)
            )
            conditioning = {
                "system_condition_2": float(np.linalg.cond(system)),
                "matrix_dimension": len(local),
            }
            c.add(
                "linear_backward_error",
                residual,
                2e-14,
                reference_kind="analytic",
                conditioning=conditioning,
            )
            shifted = tr.TMatrix.cluster(
                particles, positions + np.array([0.4, -0.3, 0.7])
            ).interaction.solve()
            c.add(
                "translation_relative_l2",
                relative(shifted.array, response.array),
                2e-12,
            )
            rotated = tr.TMatrix.cluster(
                particles, positions @ rz.T
            ).interaction.solve()
            basis_rotation = tr.rotate(angle, 0, 0, basis=local.basis)
            c.add(
                "rotation_covariance_relative_l2",
                relative(
                    rotated.array,
                    basis_rotation @ response.array @ basis_rotation.conj().T,
                ),
                2e-12,
            )
            scaled_particles = [
                tr.TMatrix.sphere(degree, k0 / scale, radius * scale, [eps, 1])
                for radius, eps in zip(radii, epsilon, strict=True)
            ]
            scaled = tr.TMatrix.cluster(
                scaled_particles, positions * scale
            ).interaction.solve()
            c.add(
                "scale_invariance_relative_l2",
                relative(scaled.array, response.array),
                2e-12,
            )
            wave = tr.plane_wave([0, 0, 1], [1, 0, 0], k0=k0)
            sca, ext = response.xs(wave)
            c.add(
                "optical_theorem_relative",
                abs(sca - ext) / max(abs(ext), 1e-30),
                2e-10,
                observables={"scattering": sca, "extinction": ext},
                x={"name": "lmax", "value": degree, "unit": "order"},
            )


def planar(c, rng):
    import numpy as np

    import treams_rs as tr

    for index in range(12):
        epsilon, mu, chirality = (
            float(rng.uniform(1.2, 4)),
            float(rng.uniform(0.7, 1.3)),
            float(rng.uniform(-0.2, 0.2)),
        )
        thickness, q = float(rng.uniform(0.05, 3)), float(rng.uniform(0, 0.5))
        with c.case(
            f"slab-{index}",
            "planar slab",
            dict(
                epsilon=epsilon,
                mu=mu,
                chirality=chirality,
                thickness=thickness,
                q=q,
                k0=1.8,
            ),
        ):
            basis = tr.PlaneWaveBasisByComp.default([[q, 0.2], [-0.4, q]])
            material = (epsilon, mu, chirality)
            layer = tr.SMatrices.slab(thickness, basis, 1.8, [1, material, 1])
            split = tr.SMatrices.slab(
                [thickness * 0.3, thickness * 0.7],
                basis,
                1.8,
                [1, material, material, 1],
            )
            c.add(
                "layer_split_relative_l2",
                relative(split.array, layer.array),
                2e-12,
            )
            incident = np.array([0.2 + 0.7j, 0.5, -0.1j, 0.3])
            for direction in ("up", "down"):
                trans, refl = layer.tr(incident, modetype=direction)
                c.add(
                    f"energy_balance_{direction}_absolute",
                    abs(trans + refl - 1),
                    2e-12,
                    observables={"transmittance": trans, "reflectance": refl},
                )
            absorbing = tr.SMatrices.slab(
                thickness, basis, 1.8, [1, (epsilon + 0.2j, mu, chirality), 1]
            )
            trans, refl = absorbing.tr(incident, modetype="up")
            c.add(
                "passivity_violation_absolute",
                max(0, trans + refl - 1, -trans, -refl),
                2e-12,
                observables={
                    "transmittance": trans,
                    "reflectance": refl,
                    "absorption": 1 - trans - refl,
                },
            )
    for epsilon in (1.5, 3, 8):
        for thickness in (0, 0.1, 0.7, 2):
            with c.case(
                f"fabry-perot-e{epsilon}-d{thickness}",
                "normal incidence slab",
                dict(epsilon=epsilon, mu=1, k0=1.3, thickness=thickness),
            ):
                basis = tr.PlaneWaveBasisByComp.default([[0, 0]])
                layer = tr.SMatrices.slab(thickness, basis, 1.3, [1, epsilon, 1])
                trans, refl = layer.tr([1, 0], modetype="up")
                n = np.sqrt(epsilon)
                r = (1 - n) / (1 + n)
                numerator = 4 * r**2 * np.sin(1.3 * n * thickness) ** 2
                expected = numerator / ((1 - r**2) ** 2 + numerator)
                c.add(
                    "fresnel_reflectance_absolute",
                    abs(refl - expected),
                    2e-12,
                    reference_kind="analytic",
                    observables={
                        "actual": refl,
                        "reference": float(expected),
                        "transmittance": trans,
                    },
                )


def periodic(c, _rng):
    import numpy as np

    import treams_rs as tr

    period, bloch, k0 = 1.7, 0.2, 1.3
    for radius in (0.1, 0.2, 0.3):
        with c.case(
            f"periodic-sphere-r{radius}",
            "periodic sphere array",
            dict(
                radius=radius,
                lmax=3,
                period=period,
                bloch=bloch,
                k0=k0,
                scale=1.4,
                propagating_axial_orders=[0],
            ),
        ):
            basis = tr.CylindricalWaveBasis.default([bloch], 3)
            particle = tr.TMatrix.sphere(3, k0, radius, [3, 1])
            value = tr.TMatrixC.from_array(
                particle, basis, lattice=period, kpar=bloch
            ).array
            c.add(
                "unitarity_max_absolute",
                unitarity(np.eye(len(basis)) + 2 * value),
                2e-10,
            )
            scale = 1.4
            scaled = tr.TMatrixC.from_array(
                tr.TMatrix.sphere(3, k0 / scale, radius * scale, [3, 1]),
                tr.CylindricalWaveBasis.default([bloch / scale], 3),
                lattice=period * scale,
                kpar=bloch / scale,
            )
            c.add("scale_invariance_relative_l2", relative(scaled.array, value), 2e-10)
        with c.case(
            f"periodic-cylinder-r{radius}",
            "periodic cylinder array",
            dict(
                radius=radius,
                mmax=3,
                period=period,
                bloch=0.1,
                kz=0.2,
                k0=k0,
                propagating_port_indices=[0, 1],
                closed_port_indices=[2, 3, 4, 5],
            ),
        ):
            basis = tr.CylindricalWaveBasis.default([0.2], 3)
            ports = tr.PlaneWaveBasisByComp.default(
                [
                    [0.2, 0.1],
                    [0.2, 0.1 + 2 * np.pi / period],
                    [0.2, 0.1 - 2 * np.pi / period],
                ],
                "zx",
            )
            local = tr.diff.cylinder([0.2], 3, k0, [radius], [4, 1])[0]
            coupling = tr.lattice.expansion_with_context(
                basis, basis, [k0, k0], [[period]], [0.1]
            )[0]
            response = tr.diff.interaction(local, coupling)[0]
            channels = tr.diff.cylindrical_channels(
                basis, [k0, k0], ports.components, ports.pol, period
            )[0]
            scattering = tr.diff.smatrix_from_array(response, channels)[0]
            power = float(
                np.sum(abs(scattering[0, 0, :2, 0]) ** 2)
                + np.sum(abs(scattering[1, 0, :2, 0]) ** 2)
            )
            c.add(
                "propagating_energy_balance_absolute",
                abs(power - 1),
                4e-10,
                observables={"outgoing_power": power, "incident_power": 1},
                conditioning={
                    "nearest_closed_order_margin": float(2 * np.pi / period - 0.1 - k0)
                },
            )
    for period in (1.7, 7.2, 12.8):
        for eta in (0.4, 0.7, 1.0):
            with c.case(
                f"ewald-p{period}-eta{eta}",
                "Ewald split invariance",
                dict(
                    period=period,
                    eta=eta,
                    order=-6,
                    k0=1.3,
                    kz=0.2,
                    bloch=0.1,
                    displacement=[0.8, 0],
                ),
            ):
                k = np.sqrt(1.3**2 - 0.2**2)
                automatic = tr.lattice.lsumcw(1, -6, k, 0.1, period, [0.8, 0])
                explicit = tr.lattice.lsumcw(1, -6, k, 0.1, period, [0.8, 0], eta=eta)
                c.add(
                    "ewald_split_scaled_absolute",
                    abs(explicit - automatic) / max(1, abs(automatic)),
                    2e-10,
                    reference_kind="analytic",
                    series=f"period {period}",
                    x={"name": "eta", "value": eta, "unit": "split parameter"},
                    conditioning={"reference_magnitude": float(abs(automatic))},
                )


def ebcm(c, _rng):
    import numpy as np

    import treams_rs as tr

    def solve(degree, order, *, legacy=False, deformation=0.23, epsilon=3.1):
        basis = tr.SphericalWaveBasis.default(degree)
        ks = 1.3 * np.array(
            [[np.sqrt(epsilon) - 0.07, np.sqrt(epsilon) + 0.07], [1, 1]]
        )
        zs = [1 / np.sqrt(epsilon), 1]
        args = dict(
            r=lambda theta: 0.3 * (1 + deformation * np.cos(theta) ** 2),
            dr=lambda theta: -0.6 * deformation * np.cos(theta) * np.sin(theta),
            ks=ks,
            zs=zs,
            out=basis,
            order=order,
            legacy=legacy,
        )
        q = tr.ebcm.qmat(**args)
        regular = tr.ebcm.qmat(**args, singular=False)
        value = -tr.diff.solve(q, regular)[0]
        return value, q

    for degree in (2, 4, 6):
        for legacy in (False, True):
            with c.case(
                f"ebcm-l{degree}-legacy{legacy}",
                "EBCM deformed lossless convergence",
                dict(
                    lmax=degree,
                    quadrature_order=96,
                    radius=0.3,
                    deformation=0.23,
                    epsilon=3.1,
                    chirality=0.07,
                    k0=1.3,
                    legacy=legacy,
                ),
            ):
                value, q = solve(degree, 96, legacy=legacy)
                c.add(
                    "unitarity_max_absolute",
                    unitarity(np.eye(len(value)) + 2 * value),
                    2e-8 if degree == 6 and not legacy else None,
                    backend="treams-rs legacy integral"
                    if legacy
                    else "treams-rs corrected",
                    series="deformed lossless surface",
                    x={"name": "lmax", "value": degree, "unit": "order"},
                    conditioning={"q_condition_2": float(np.linalg.cond(q))},
                )
    with c.case(
        "ebcm-quadrature",
        "EBCM quadrature convergence",
        dict(
            lmax=6,
            reference_quadrature_order=192,
            radius=0.3,
            deformation=0.23,
            epsilon=3.1,
            chirality=0.07,
            k0=1.3,
        ),
    ):
        reference, _ = solve(6, 192)
        for order in (48, 96):
            value, _ = solve(6, order)
            c.add(
                "quadrature_relative_l2",
                relative(value, reference),
                2e-9,
                reference_kind="converged_native",
                series="deformed surface",
                id=f"ebcm-quadrature-q{order}/quadrature_relative_l2",
                x={"name": "quadrature order", "value": order, "unit": "nodes"},
            )
    for radius in (0.2, 0.3, 0.45):
        with c.case(
            f"ebcm-sphere-r{radius}",
            "EBCM spherical limit",
            dict(
                lmax=2,
                quadrature_order=48,
                radius=radius,
                epsilon=3.1,
                mu=1.2,
                chirality=0.08,
                k0=1.3,
            ),
        ):
            basis = tr.SphericalWaveBasis.default(2)
            materials = [tr.Material((3.1, 1.2, 0.08)), tr.Material(1)]
            args = dict(
                r=lambda _, radius=radius: radius,
                dr=lambda _: 0,
                ks=[m.ks(1.3) for m in materials],
                zs=[m.impedance for m in materials],
                out=basis,
                order=48,
            )
            q = tr.ebcm.qmat(**args)
            regular = tr.ebcm.qmat(**args, singular=False)
            value = -tr.diff.solve(q, regular)[0]
            c.add(
                "sphere_recovery_relative_l2",
                relative(value, tr.TMatrix.sphere(2, 1.3, radius, materials).array),
                2e-11,
                reference_kind="converged_native",
                reference_note="Independent formulations within one native implementation: surface integral versus layered-sphere coefficient recurrence.",
            )
            c.add(
                "unitarity_max_absolute",
                unitarity(np.eye(len(basis)) + 2 * value),
                2e-12,
            )
    with c.case(
        "ebcm-zero-contrast",
        "EBCM zero contrast",
        dict(lmax=2, quadrature_order=64, radius=0.3, deformation=0.23, k=1.3),
    ):
        basis = tr.SphericalWaveBasis.default(2)
        args = dict(
            r=lambda theta: 0.3 * (1 + 0.23 * np.cos(theta) ** 2),
            dr=lambda theta: -0.138 * np.cos(theta) * np.sin(theta),
            ks=np.full((2, 2), 1.3),
            zs=[1, 1],
            out=basis,
            order=64,
        )
        regular = tr.ebcm.qmat(**args, singular=False)
        value = -tr.diff.solve(tr.ebcm.qmat(**args), regular)[0]
        c.add("zero_contrast_max_absolute", np.max(abs(value)), 1e-14)


def source_metadata():
    import treams_rs as tr
    from treams_rs import _native

    checksum = hashlib.sha256()
    for path in sorted(Path(tr.__file__).parent.glob("*.py")):
        checksum.update(path.name.encode())
        checksum.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "native_sha256": hashlib.sha256(
            Path(_native.__file__).read_bytes()
        ).hexdigest(),
        "native_profile": _native.build_profile(),
        "python_source_sha256": checksum.hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def qualify(seed=20260912, families=FAMILIES):
    import numpy as np

    source = source_metadata()
    c = Collector()
    for name in families:
        # Independent streams make --families reproduce the same samples as a full run.
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, FAMILIES.index(name)])
        )
        globals()[name](c, rng)
    after = source_metadata()
    stable = source == after
    return {
        "kind": "accuracy",
        "suite": "physical identities and convergence",
        "source": source,
        "source_after": after,
        "source_unchanged": stable,
        "environment": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "protocol": {
            "seed": seed,
            "families": list(families),
            "reference": "Analytic solutions, conservation laws, covariance, and separately labelled native self-convergence; no upstream treams is imported.",
            "relative_l2": "Euclidean/Frobenius difference norm divided by max(reference norm, 1e-30).",
            "scaled_absolute": "Maximum elementwise residual divided by max(1, reference maximum magnitude), unless the metric names l2.",
            "passivity": "One-sided violation; signed scattering/extinction/absorption or reflected/transmitted power are retained.",
            "diagnostic": "A null tolerance is a convergence/legacy observation, not a passing correctness check. Only final truncation orders have convergence acceptance tolerances.",
            "periodic_power": "Lossless media away from diffraction cutoffs; flux sums include only explicitly identified propagating ports, never evanescent amplitudes.",
            "legacy_ebcm": "The explicitly labelled native legacy integral reproduces upstream's omitted radius factor in the surface-area element. Its nonunitarity is an expected reference defect, not evidence of an error in corrected treams-rs. Low-order corrected residuals include multipole truncation.",
            "limitations": "Finite deterministic grids and seeded samples are not exhaustive; conservation and self-convergence alone do not establish correctness, and this suite does not qualify GPU or WASM builds.",
        },
        "observations": c.observations,
        "complete": True,
        "passed": stable
        and all(row["status"] not in ("failed", "error") for row in c.observations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(args.threads)
    from threadpoolctl import threadpool_info, threadpool_limits

    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        result = qualify(args.seed, args.families)
        result["environment"]["threadpools"] = [
            {**pool, "filepath": Path(pool["filepath"]).name}
            for pool in threadpool_info()
        ]
    result["environment"]["requested_threads"] = args.threads
    result["elapsed_seconds"] = time.perf_counter() - started
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    sys.stdout.write(encoded + "\n")


if __name__ == "__main__":
    main()
