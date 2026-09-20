"""Hidden upstream numerical references and analytic planar controls, never shipped."""

import numpy as np
import treams as upstream

import treams_rs as native


def z(value):
    return complex(*value) if isinstance(value, list) else value


def sphere_efficiency(radius, epsilon, wavelength, lmax, lib=upstream):
    tm = lib.TMatrix.sphere(lmax, 2 * np.pi / wavelength, radius, [epsilon, 1])
    return tm.xs_sca_avg / (np.pi * radius**2)


def derivative(function, x):
    h = max(abs(x) * 1e-4, 1e-6)
    # Fourth-order independent test oracle, never an execution pullback.
    return (
        function(x - 2 * h)
        - 8 * function(x - h)
        + 8 * function(x + h)
        - function(x + 2 * h)
    ) / (12 * h)


def planar(c, lib=upstream, reverse=False):
    result = []
    cls = lib.SMatrices if lib is upstream else lib.SMatrix
    ports_cls = lib.PlaneWaveBasisByComp if lib is upstream else lib.PlaneWavePorts
    for wavelength in c["wavelengths"]:
        k0 = 2 * np.pi / wavelength
        ports = ports_cls.default([0, 0])
        if "epsilon_layers" in c:
            layers = c["epsilon_layers"]
            thicknesses = c["thicknesses"]
        else:
            layers = [(z(c["epsilon"]), 1, c["kappa"])]
            thicknesses = [c["thickness"]]
        slab = cls.slab(thicknesses, ports, k0, [1, *layers, 1])
        if lib is upstream:
            result.append(
                [
                    slab.tr(
                        lib.PhysicsArray(incident, modetype="down" if reverse else "up")
                    )
                    for incident in ([0, 1], [1, 0])
                ]
            )
        else:
            result.append(
                [
                    (p.transmission, p.reflection)
                    for p in [
                        slab.power([0, 1], side="positive" if reverse else "negative"),
                        slab.power([1, 0], side="positive" if reverse else "negative"),
                    ]
                ]
            )
    a = np.asarray(result)
    return {
        "transmission": a[:, :, 0],
        "reflection": a[:, :, 1],
        "absorption": 1 - a[:, :, 0] - a[:, :, 1],
    }


def dimer(c, lib=upstream):
    k0 = 2 * np.pi / c["wavelength"]
    d = c["separation"]
    radii = c.get("radii", [c.get("radius")] * 2)
    eps = c["epsilon"] if "radii" in c else [c["epsilon"]] * 2
    tms = [
        lib.TMatrix.sphere(c["lmax"], k0, r, [z(e), 1])
        for r, e in zip(radii, eps, strict=True)
    ]
    points = c.get("points", [c.get("point")])
    if lib is native:
        tm = lib.Cluster(tms, positions=[[-d / 2, 0, 0], [d / 2, 0, 0]]).solve()
        inc = lib.plane_wave([0, 0, 1], 1, k0=k0)
        field = inc.efield(points) + tm.scatter(inc).efield(points)
        power = tm.cross_sections(inc)
        sca, ext = power.scattering, power.extinction
    else:
        tm = lib.TMatrix.cluster(
            tms, [[-d / 2, 0, 0], [d / 2, 0, 0]]
        ).interaction.solve()
        inc = lib.plane_wave([0, 0, 1], 1, k0=k0)
        scattered = np.asarray(tm) @ np.asarray(inc.expand(tm.basis))
        field = (
            np.asarray(
                lib.efield(
                    points, basis=tm.basis, k0=k0, material=1, modetype="singular"
                )
            )
            @ scattered
        )
        field += np.asarray(
            lib.efield(points, basis=inc.basis, k0=k0, material=1)
        ) @ np.asarray(inc)
        sca, ext = tm.xs(inc)
    return {
        "total_field_real": field.real,
        "total_field_imag": field.imag,
        "scattering": sca,
        "extinction": ext,
        "absorption": ext - sca,
    }


def coating_reflection(thickness, wavelength, epsilon, substrate):
    # Independent normal-incidence Fresnel/Airy formula, not package output.
    n = np.sqrt(epsilon)
    ns = np.sqrt(substrate)
    a = (1 - n) / (1 + n)
    b = (n - ns) / (n + ns)
    phase = np.exp(4j * np.pi * n * thickness / wavelength)
    return float(abs((a + b * phase) / (1 + a * b * phase)) ** 2)


def evaluate(case_id, c, lib=upstream):
    if case_id in ("sphere_spectrum", "coated_sphere"):
        rows = []
        for wavelength in c["wavelengths"]:
            radii = c["radii"] if case_id == "coated_sphere" else c["radius"]
            materials = (
                [*[z(e) for e in c["epsilon_layers"]], c["epsilon_medium"]]
                if case_id == "coated_sphere"
                else [z(c["epsilon"]), 1]
            )
            tm = lib.TMatrix.sphere(c["lmax"], 2 * np.pi / wavelength, radii, materials)
            rows.append([tm.xs_sca_avg, tm.xs_ext_avg])
        a = np.asarray(rows)
        result = {
            "scattering": a[:, 0],
            "extinction": a[:, 1],
            "absorption": a[:, 1] - a[:, 0],
        }
        if case_id == "sphere_spectrum":
            result["efficiency"] = a[:, 0] / (np.pi * c["radius"] ** 2)
        return result
    if case_id == "cylinder_width":
        rows = []
        for wavelength in c["wavelengths"]:
            k0 = 2 * np.pi / wavelength
            cls = lib.TMatrixC if lib is upstream else lib.CylindricalTMatrix
            tm = cls.cylinder([0], c["mmax"], k0, c["radius"], [z(c["epsilon"]), 1])
            inc = lib.plane_wave([1, 0, 0], 1, k0=k0)
            rows.append(tm.xw(inc))
        a = np.asarray(rows)
        return {
            "scattering": a[:, 0],
            "extinction": a[:, 1],
            "absorption": a[:, 1] - a[:, 0],
        }
    if case_id == "dimer_field":
        return dimer(c, lib)
    if case_id == "chiral_slab":
        return planar(c, lib)
    if case_id == "array_on_slab":
        rows = []
        for wavelength in c["wavelengths"]:
            k0 = 2 * np.pi / wavelength
            lattice = lib.Lattice.square(c["period"])
            ports = (
                lib.PlaneWaveBasisByComp if lib is upstream else lib.PlaneWavePorts
            ).default([0, 0])
            sphere = lib.TMatrix.sphere(
                c["lmax"], k0, c["radius"], [c["sphere_epsilon"], 1]
            )
            cls = lib.SMatrices if lib is upstream else lib.SMatrix
            if lib is upstream:
                response = sphere.latticeinteraction.solve(lattice, [0, 0])
                array = cls.from_array(response, ports)
            else:
                array = lib.solve_periodic(
                    sphere, lattice=lattice, kpar=[0, 0]
                ).to_smatrix(ports)
            slab = cls.slab(c["slab_thickness"], ports, k0, [1, c["slab_epsilon"], 1])
            gap = cls.propagation([0, 0, c["gap"]], ports, k0, 1)
            network = cls.stack([slab, gap, array])
            rows.append(network.tr([0, 1]))
        a = np.asarray(rows)
        return {
            "transmission": a[:, 0],
            "reflection": a[:, 1],
            "absorption": 1 - a[:, 0] - a[:, 1],
        }
    if case_id == "sphere_sensitivity":

        def f(radius, epsilon):
            return sphere_efficiency(
                radius,
                epsilon + 1j * c["epsilon_imag"],
                c["wavelength"],
                c["lmax"],
                lib,
            )

        r, e = c["radius"], c["epsilon_real"]
        return {
            "value": f(r, e),
            "gradient": [
                derivative(lambda x: f(x, e), r),
                derivative(lambda x: f(r, x), e),
            ],
        }
    if case_id == "dimer_sensitivity":

        def f(d):
            result = dimer({**c, "separation": d}, lib)
            return np.sum(
                result["total_field_real"] ** 2 + result["total_field_imag"] ** 2
            )

        return {"value": f(c["separation"]), "gradient": derivative(f, c["separation"])}
    if case_id == "reverse_stack_transfer":
        return {
            k: v[:, 0]
            for k, v in planar(c, lib, reverse=True).items()
            if k != "absorption"
        }
    if case_id == "coated_sphere_sensitivity_transfer":

        def f(r):
            tm = lib.TMatrix.sphere(
                c["lmax"],
                2 * np.pi / c["wavelength"],
                [c["core_radius"], r],
                [z(c["core_epsilon"]), z(c["shell_epsilon"]), 1],
            )
            return tm.xs_ext_avg - tm.xs_sca_avg

        return {
            "value": f(c["outer_radius"]),
            "gradient": derivative(f, c["outer_radius"]),
        }
    if case_id in ("antireflection_design", "wavelength_design_transfer"):
        wavelength = (
            c["wavelength"]
            if case_id == "antireflection_design"
            else 4 * np.sqrt(c["epsilon"]) * c["thickness"]
        )
        thickness = wavelength / (4 * np.sqrt(c["epsilon"]))
        initial = coating_reflection(
            c.get("initial_thickness", thickness),
            c.get("initial_wavelength", wavelength),
            c["epsilon"],
            c["substrate_epsilon"],
        )
        return {
            "optimum": thickness if case_id == "antireflection_design" else wavelength,
            "initial_reflection": initial,
        }
    if case_id == "radius_inverse_design":

        def f(r):
            q = np.array(
                [
                    sphere_efficiency(r, c["epsilon"], w, c["lmax"], lib)
                    for w in c["wavelengths"]
                ]
            )
            return np.mean(
                ((q - c["target_efficiencies"]) / c["target_efficiencies"]) ** 2
            )

        return {"initial_loss": f(c["initial_radius"])}
    raise ValueError(case_id)


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return np.asarray(value).tolist()
