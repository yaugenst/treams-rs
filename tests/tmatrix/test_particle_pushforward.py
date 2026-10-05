"""Particle JVPs against directional differences, adjoints and scale identities."""

import numpy as np
import pytest

import treams_rs as tr
from treams_rs import diff
from treams_rs.testing import check_pushforward

from _support import complex_normal

pytestmark = pytest.mark.gradients


def _layers():
    return (
        np.array([0.2, 0.5]),
        np.array([2.0 + 0.1j, 3.0 + 0.2j, 1.1 + 0.01j]),
        np.array([1.2 + 0.02j, 1.1 + 0.01j, 1.0 + 0.005j]),
        np.array([0.03 + 0.004j, 0.02 + 0.006j, 0.01 + 0.002j]),
    )


def _directions(values):
    rng = np.random.default_rng(437)
    return tuple(
        0.1
        * (
            complex_normal(rng, np.shape(value))
            if np.iscomplexobj(value)
            else rng.normal(size=np.shape(value))
        )
        for value in values
    )


@pytest.mark.parametrize(
    ("degree", "sizes"), [(1, [0.2, 0.5]), (3, [0.7, 1.4]), (8, [2.5, 5.0])]
)
def test_multilayer_chiral_mie_pushforward(degree, sizes):
    # Higher degrees need larger particles to retain appreciable scattering.
    values = (np.array(sizes), *_layers()[1:])
    check_pushforward(
        lambda *layers: diff.mie(degree, *layers),
        *values,
        directions=_directions(values),
    )


@pytest.mark.parametrize("order", [-3, 0, 2])
@pytest.mark.parametrize("kz", [-0.3, 0.0, 0.3])
def test_multilayer_chiral_cylinder_coefficients_pushforward(order, kz):
    values = (np.asarray(kz), np.asarray(1.2), *_layers())

    def record(kz, k0, *layers):
        return diff.mie_cyl(float(kz), order, float(k0), *layers)

    check_pushforward(record, *values, directions=_directions(values))


@pytest.mark.parametrize("lmax", [1, 3])
def test_multilayer_chiral_sphere_pushforward(lmax):
    values = (np.asarray(1.2), *_layers())

    def record(k0, *layers):
        return diff.sphere(lmax, float(k0), *layers)

    check_pushforward(record, *values, directions=_directions(values))


@pytest.mark.parametrize("mmax", [0, 2])
def test_cylinder_pushforward_has_independent_axial_directions(mmax):
    # Mirrored primal channels may share coefficients, but their tangent inputs
    # are independent. The zero channel also has a nonzero axial perturbation.
    values = (np.array([0.3, -0.3, 0.0]), np.asarray(1.2), *_layers())
    directions = (np.array([0.13, -0.29, 0.21]), *_directions(values)[1:])

    def record(kzs, k0, *layers):
        return diff.cylinder(kzs, mmax, float(k0), *layers)

    check_pushforward(record, *values, directions=directions)


@pytest.mark.parametrize(
    ("record", "prefix"),
    [
        (lambda *layers: diff.mie(2, *layers), ()),
        (
            lambda kz, k0, *layers: diff.mie_cyl(float(kz), 1, float(k0), *layers),
            (np.asarray(0.3), np.asarray(1.2)),
        ),
        (
            lambda k0, *layers: diff.sphere(1, float(k0), *layers),
            (np.asarray(1.2),),
        ),
        (
            lambda kzs, k0, *layers: diff.cylinder(kzs, 1, float(k0), *layers),
            (np.array([0.3, -0.3, 0.0]), np.asarray(1.2)),
        ),
    ],
    ids=["mie", "mie_cyl", "sphere", "cylinder"],
)
def test_particle_pushforward_accepts_real_material_directions(record, prefix):
    # Lossless materials are naturally passed as real arrays even though the
    # native material representation and the resulting matrices are complex.
    values = (*prefix, *(value.real.copy() for value in _layers()))
    check_pushforward(record, *values, directions=_directions(values))


@pytest.mark.parametrize("kind", ["cd", "db", "chi"])
def test_metric_pushforward_includes_matrix_and_wavenumber_directions(kind):
    matrix = -0.15 * np.eye(6) + 0.02 * complex_normal(
        np.random.default_rng(91), (6, 6)
    )
    values = (matrix, np.array([1.2, 1.4]))

    def record(matrix, ks):
        return diff.tmatrix_metric(
            matrix, ks, polarizations=tr.SphericalBasis.default(1).pol, kind=kind
        )

    # Keep the probe below the matrix scale, away from chirality's zero contrast.
    directions = tuple(0.1 * direction for direction in _directions(values))
    check_pushforward(record, *values, directions=directions)


@pytest.mark.physics
@pytest.mark.parametrize("cylindrical", [False, True])
def test_particle_pushforward_preserves_dimensionless_scattering(cylindrical):
    radii, epsilon, mu, kappa = _layers()
    k0 = 1.2
    material_directions = tuple(np.zeros_like(value) for value in (epsilon, mu, kappa))
    if cylindrical:
        kzs = np.array([0.3, -0.3, 0.0])
        _, context = diff.cylinder(kzs, 2, k0, radii, epsilon, mu, kappa)
        tangent = context.pushforward(-kzs, -k0, radii, *material_directions)
    else:
        _, context = diff.sphere(3, k0, radii, epsilon, mu, kappa)
        tangent = context.pushforward(-k0, radii, *material_directions)
    # Scaling every length and inversely scaling every wavenumber preserves
    # all radial arguments and material contrasts.
    np.testing.assert_allclose(tangent, 0, rtol=0, atol=3e-13)


@pytest.mark.physics
@pytest.mark.parametrize("kind", ["db", "chi"])
@pytest.mark.parametrize("factor", [1.0, 1.0j])
def test_metric_pushforward_preserves_scale_and_global_phase(kind, factor):
    matrix = -0.15 * np.eye(6) + 0.02 * complex_normal(
        np.random.default_rng(91), (6, 6)
    )
    _, context = diff.tmatrix_metric(
        matrix, polarizations=tr.SphericalBasis.default(1).pol, kind=kind
    )
    tangent = context.pushforward(factor * matrix, np.zeros(2))
    np.testing.assert_allclose(tangent, 0, rtol=0, atol=3e-13)
