# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Frozen physical tasks; reference answers never enter model-visible prompts."""

import copy
import json

COMMON = """Only a treams-rs wheel with its declared [advect] dependencies is installed. Use python; do not install packages or access external sources. The installed package and your own workspace are available. Scratch files persist in your workspace and TMPDIR.

Create solution.py exposing run(config), returning a JSON-serializable dictionary. Its CLI must read one config dictionary from standard input and print the returned dictionary as JSON to standard output. Actually execute your solution for the example config, save results.json, and write a short REPORT.md explaining units, conventions and checks. Your function will also be called with different values of the same input keys; do not hard-code the example answers. Use treams-rs for numerical scattering, not a reimplementation. For sensitivity/optimization tasks use its analytic derivatives (via native pullbacks or the installed Advect adapter); finite differences may validate derivatives but must not supply the requested derivatives or optimization steps. Avoid private API imports. No plotting is required.

"""
PAPER = "Beutel, Fernandez-Corbaton and Rockstuhl, treams, Computer Physics Communications 297, 109076 (2024), doi:10.1016/j.cpc.2023.109076."
EXAMPLES = "https://github.com/tfp-photonics/treams/tree/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples"


def task(case_id, category, text, config, variant, *, transfer=False):
    other = copy.deepcopy(config)
    other.update(variant)
    return {
        "id": case_id,
        "category": category,
        "transfer": transfer,
        "source": PAPER,
        "inspiration": EXAMPLES,
        "adaptation": "Small self-contained evaluation inputs; no claim to reproduce a published figure or optimized design.",
        "prompt": COMMON
        + text
        + "\n\nInspiration: "
        + PAPER
        + " The physical problem is fully specified here; literature access is unnecessary. All lengths are in micrometers; permittivities, permeabilities and chirality are relative/dimensionless. Unless specified, mu=1, chirality=0 and the exterior is vacuum.\n\nExample config:\n"
        + json.dumps(config, indent=2),
        "inputs": [config, other],
    }


TASKS = [
    task(
        "sphere_spectrum",
        "forward",
        "Compute a homogeneous dielectric sphere spectrum, illuminated along +z with positive helicity. Use the specified multipole cutoff. Return scattering, extinction and absorption cross sections (areas), and scattering efficiency divided by pi*radius**2, as lists in wavelength order. Keys: scattering, extinction, absorption, efficiency. epsilon=[real,imag] is relative permittivity, not refractive index.",
        {
            "radius": 0.1,
            "epsilon": [12, 0.05],
            "wavelengths": [0.45, 0.55, 0.65, 0.75],
            "lmax": 4,
        },
        {"radius": 0.085, "epsilon": [9, 0.12], "wavelengths": [0.48, 0.62, 0.79]},
    ),
    task(
        "coated_sphere",
        "forward",
        "Compute a concentric core-shell sphere spectrum in a dielectric exterior. radii are increasing outer boundaries, and epsilon_layers supplies one [real,imag] permittivity per layer, inside-out. epsilon_medium describes the exterior. Return rotationally averaged scattering, extinction and absorption cross sections as lists in wavelength order. Do not append the exterior as an extra finite layer.",
        {
            "radii": [0.07, 0.12],
            "epsilon_layers": [[2.25, 0.03], [4, 0.06]],
            "epsilon_medium": 1.7689,
            "wavelengths": [0.58, 0.7, 0.82],
            "lmax": 4,
        },
        {
            "radii": [0.05, 0.105],
            "epsilon_layers": [[3, 0.01], [5, 0.08]],
            "epsilon_medium": 1.44,
        },
    ),
    task(
        "cylinder_width",
        "forward",
        "Compute scattering by a homogeneous infinite cylinder parallel to z, with kz=0 and positive-helicity illumination along +x. mmax is the azimuthal cutoff. Return scattering, extinction and absorption cross widths (length units), as wavelength-ordered lists. epsilon is [real,imag]. Do not use sphere area normalization.",
        {
            "radius": 0.08,
            "epsilon": [5, 0.1],
            "wavelengths": [0.5, 0.65, 0.8],
            "mmax": 4,
        },
        {"radius": 0.1, "epsilon": [3.5, 0.03], "wavelengths": [0.58, 0.73]},
    ),
    task(
        "dimer_field",
        "forward",
        "Two spheres sit at x=-separation/2 and +separation/2, y=z=0, with the radii and epsilon values in that order. Include all multiple scattering at the fixed lmax. Illuminate along +z with positive helicity and evaluate the TOTAL electric field at points (incident plus scattered, using the package incident normalization). Return total_field_real and total_field_imag as point-by-3 lists, plus scalar scattering, extinction and absorption cross sections. epsilon entries are [real,imag].",
        {
            "radii": [0.08, 0.11],
            "epsilon": [[4, 0.02], [6, 0.08]],
            "separation": 0.32,
            "wavelength": 0.72,
            "lmax": 3,
            "points": [[0, 0, 0.3], [0.35, 0.2, 0.1]],
        },
        {
            "separation": 0.38,
            "wavelength": 0.8,
            "points": [[0.1, 0, 0.4], [-0.4, 0.1, 0.2]],
        },
    ),
    task(
        "chiral_slab",
        "forward",
        "A uniform reciprocal chiral slab is normal to z, bounded by vacuum on both sides. For +z normal incidence, compute transmitted, reflected and absorbed power fractions for positive and negative helicity at each wavelength. epsilon is [real,imag]; kappa is the package material chirality. Return transmission, reflection and absorption as lists with shape (wavelengths, 2), ordered positive then negative helicity. These are power fractions, not complex amplitude coefficients.",
        {
            "thickness": 0.16,
            "epsilon": [2.25, 0.06],
            "kappa": 0.08,
            "wavelengths": [0.55, 0.65, 0.8],
        },
        {"thickness": 0.21, "epsilon": [3, 0.08], "kappa": -0.05},
    ),
    task(
        "array_on_slab",
        "forward",
        "A square periodic array of identical spheres is above a uniform dielectric slab, with a vacuum propagation gap between the slab top and sphere-center plane. Order the network along +z as slab, vacuum gap, sphere array. All exterior media are vacuum. Illuminate from the negative-z side along +z with positive helicity. The wavelengths exceed the period, so retain only the zeroth propagating diffraction order (both helicities), with Bloch vector [0,0]. Solve the full periodic interaction at the fixed lmax. Return wavelength-ordered transmission, reflection and absorption power fractions. sphere_epsilon and slab_epsilon are real permittivities.",
        {
            "radius": 0.06,
            "sphere_epsilon": 9.0,
            "period": 0.4,
            "gap": 0.12,
            "slab_thickness": 0.09,
            "slab_epsilon": 2.25,
            "wavelengths": [0.65, 0.74, 0.83],
            "lmax": 2,
        },
        {"period": 0.43, "radius": 0.055, "gap": 0.14, "wavelengths": [0.68, 0.78]},
    ),
    task(
        "sphere_sensitivity",
        "sensitivity",
        "For a lossy sphere under positive-helicity +z incidence, calculate scattering efficiency Q=scattering_cross_section/(pi*radius**2). Return scalar value and gradient=[dQ/dradius,dQ/depsilon_real], holding epsilon_imag, wavelength and lmax fixed. Differentiate the normalization as well as the cross section, using analytic derivatives. Verify the gradient separately if useful.",
        {
            "radius": 0.1,
            "epsilon_real": 4.0,
            "epsilon_imag": 0.06,
            "wavelength": 0.72,
            "lmax": 3,
        },
        {"radius": 0.09, "epsilon_real": 5.0, "wavelength": 0.79},
    ),
    task(
        "dimer_sensitivity",
        "sensitivity",
        "Two identical spheres at x=-separation/2 and +separation/2 are illuminated along +z with positive helicity. Include multiple scattering. At the fixed point, define value=sum(abs(E_total)**2), including the incident field. Return scalar value and scalar gradient=dvalue/dseparation. The sphere positions both move as separation changes; point, wavelength, radii, material and lmax stay fixed. epsilon is [real,imag]. Use analytic derivatives.",
        {
            "radius": 0.08,
            "epsilon": [4.0, 0.04],
            "separation": 0.3,
            "wavelength": 0.72,
            "lmax": 2,
            "point": [0.05, 0.02, 0.28],
        },
        {"separation": 0.36, "wavelength": 0.8, "point": [0.03, 0.06, 0.3]},
    ),
    task(
        "antireflection_design",
        "optimization",
        "Use analytic thickness gradients to minimize reflected power of one lossless coating at normal incidence from vacuum (+z) into a substrate. The coating epsilon and positive-side substrate epsilon are supplied; no vacuum gap after the coating. Start at initial_thickness and stay within bounds. Return thickness, reflection, initial_reflection, gradient (final dR/dthickness), and loss_history. Achieve reflection <1e-7. Do not substitute a closed-form quarter-wave answer for running gradient-based optimization.",
        {
            "wavelength": 0.7,
            "epsilon": 2.25,
            "substrate_epsilon": 5.0625,
            "initial_thickness": 0.05,
            "bounds": [0.025, 0.21],
        },
        {"wavelength": 0.81, "initial_thickness": 0.19, "bounds": [0.03, 0.24]},
    ),
    task(
        "radius_inverse_design",
        "optimization",
        "Infer a homogeneous lossless sphere radius from target scattering efficiencies at the given wavelengths. epsilon is fixed, vacuum exterior, and lmax fixed. Minimize mean(((Q-target)/target)**2) with analytic radius gradients; start at initial_radius and stay in bounds. Return radius, loss, initial_loss, gradient (final derivative of loss), fitted_efficiencies, and loss_history. Achieve loss <1e-7. The targets are measurements supplied with the config, not a reference radius.",
        {
            "epsilon": 4.0,
            "lmax": 3,
            "wavelengths": [0.58, 0.69, 0.81],
            "initial_radius": 0.075,
            "bounds": [0.06, 0.14],
            "target_efficiencies": [],
        },
        {"wavelengths": [0.61, 0.74, 0.85], "initial_radius": 0.13},
    ),
    task(
        "reverse_stack_transfer",
        "forward",
        "A two-layer lossless stack is ordered negative-z to positive-z with the supplied thicknesses and permittivities. Both exterior media are vacuum. Compute wavelength-ordered transmission and reflection for incidence from POSITIVE z toward negative z, with positive helicity relative to propagation. Return transmission and reflection as lists. Preserve layer order when reversing incidence.",
        {
            "thicknesses": [0.08, 0.13],
            "epsilon_layers": [2.25, 4.0],
            "wavelengths": [0.6, 0.73, 0.87],
        },
        {"thicknesses": [0.11, 0.07], "epsilon_layers": [3.24, 1.69]},
        transfer=True,
    ),
    task(
        "coated_sphere_sensitivity_transfer",
        "sensitivity",
        "For a concentric core-shell sphere in vacuum, compute absorption cross section and its analytic derivative with respect to outer_radius. core_radius remains fixed; the shell permittivity and all other quantities stay fixed. Positive-helicity +z incidence. Return scalar value and gradient. Materials are supplied as [real,imag].",
        {
            "core_radius": 0.06,
            "outer_radius": 0.11,
            "core_epsilon": [4.0, 0.05],
            "shell_epsilon": [2.25, 0.1],
            "wavelength": 0.71,
            "lmax": 3,
        },
        {"outer_radius": 0.12, "wavelength": 0.8},
        transfer=True,
    ),
    task(
        "wavelength_design_transfer",
        "optimization",
        "A fixed lossless single-layer coating lies between vacuum on negative z and a dielectric substrate on positive z. Minimize reflected power by varying wavelength using analytic derivatives, starting from initial_wavelength within bounds. Normal +z positive-helicity incidence. Return wavelength, reflection, initial_reflection, gradient (final dR/dwavelength) and loss_history. Achieve reflection <1e-7. Geometry stays fixed; remember k0 depends on wavelength. Run gradient-based optimization rather than a closed-form quarter-wave answer.",
        {
            "thickness": 0.12,
            "epsilon": 2.25,
            "substrate_epsilon": 5.0625,
            "initial_wavelength": 0.61,
            "bounds": [0.55, 0.88],
        },
        {"thickness": 0.135, "initial_wavelength": 0.92, "bounds": [0.62, 0.98]},
        transfer=True,
    ),
]
