"""Native grouped power flux, complete port pullbacks and scattering composition."""

import warnings

import advect
import advect.numpy as anp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import assert_tree_allclose, complex_normal

pytestmark = pytest.mark.gradients


@pytest.mark.physics
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_batched_power_complete_advect_and_scale(axis, direction):
    rng = np.random.default_rng(114)
    matrices = complex_normal(rng, (2, 2, 4, 4)) * 0.03
    matrices[0, 0] += np.eye(4) * 0.8
    matrices[1, 1] += np.eye(4) * 0.9
    incident = complex_normal(rng, (4, 2))
    ks = np.array([[1.3 + 0.02j, 1.5 + 0.02j], [1.2 + 0.01j, 1.4 + 0.01j]])
    zs = np.array([0.9 + 0.01j, 1.1 - 0.01j])
    q = np.array([[0.2, 0.13], [0.31, -0.17]])
    modes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    arguments = (matrices, incident, ks, zs, q)
    weights = np.array([[0.3, -0.2], [0.1, 0.4]])

    def objective(*args):
        return anp.real(
            anp.sum(
                ad.smatrix_tr(*args, modes=modes, axis=axis, modetype=direction)
                * weights
            )
        )

    gradients = advect.grad(objective, argnums=(0, 1, 2, 3, 4))(*arguments)
    assert_allclose(np.vdot(gradients[1], incident).real, 0, atol=2e-12)
    value = objective(*arguments)
    assert_allclose(
        objective(matrices, incident * (1.3 + 0.2j), ks, zs, q), value, atol=2e-13
    )
    # Powers are dimensionless: scaling all wavenumbers together changes nothing.
    assert_allclose(
        objective(matrices, incident, ks * 1.3, zs, q * 1.3), value, atol=2e-13
    )
    spectral = np.vdot(gradients[2], ks).real + np.vdot(gradients[4], q).real
    assert_allclose(spectral, 0, atol=2e-12)
    check_gradient(
        objective,
        lambda *_: gradients,
        *arguments,
        directions=tuple(
            complex_normal(rng, v.shape)
            if np.iscomplexobj(v)
            else rng.normal(size=v.shape)
            for v in arguments
        ),
        step=2e-6,
        rtol=3e-7,
        atol=2e-8,
    )


def _power_case(rng, direction, illuminations=3):
    """Arguments and a recorder of a lossy four-mode, two-group power problem."""
    matrices = complex_normal(rng, (2, 2, 4, 4)) * 0.1
    matrices[0, 0] += np.eye(4)
    matrices[1, 1] += np.eye(4)
    incident = complex_normal(rng, (4, illuminations))
    ks = np.array([[1.3 + 0.02j, 1.5 + 0.02j], [1.2 + 0.01j, 1.4 + 0.01j]])
    zs, q = np.array([0.9 + 0.01j, 1.1 - 0.01j]), np.array([[0.2, 0.13], [0.31, -0.17]])
    modes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def record(*args):
        return tr.diff.smatrix_tr(*args, modes=modes, modetype=direction)

    return record, (matrices, incident, ks, zs, q)


@pytest.mark.parametrize("illuminations", [3, 1])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_power_pullback_takes_real_cotangents_of_its_real_output(
    direction, illuminations
):
    """A real cotangent equals its complex cast, so check_pullback applies.

    With one illumination the rows of a real cotangent have size 1. NumPy
    before 2.5 turns such a row into a complex scalar with a DeprecationWarning,
    so an element-wise conversion would flatten the cotangent; ignoring the
    warning checks that the whole array converts on those versions too.
    """
    rng = np.random.default_rng(29)
    record, arguments = _power_case(rng, direction, illuminations)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        check_pullback(record, *arguments, rtol=1e-6, atol=1e-8)
        value, context = record(*arguments)
        g = rng.normal(size=value.shape)
        assert_tree_allclose(
            context.pullback(g), record(*arguments)[1].pullback(g + 0j), rtol=0, atol=0
        )


@pytest.mark.physics
@pytest.mark.parametrize("alignment,axis", [("xy", 2), ("yz", 0), ("zx", 1)])
@pytest.mark.parametrize("fixed_q", [False, True])
def test_interface_power_differentiated_energy_conservation(alignment, axis, fixed_q):
    ks = np.array([[1.3, 1.5], [1.1, 1.4]])
    zs = np.array([1.2, 0.9])
    q = np.array([[0, 0] if fixed_q else [0.2, 0.1]])
    incident = np.array([[0.3 + 0.2j], [0.4 - 0.1j]])
    modes = [(0, 0), (0, 1)]

    def objective(ks, zs, incident):
        matrix = ad.interface_coefficients(
            ks, zs, q[0], alignment=alignment, fixed_q=fixed_q
        )
        # Interface input media are below/above; S-matrix ports are above/below.
        powers = ad.smatrix_tr(
            matrix,
            incident,
            ks[::-1],
            zs[::-1],
            q,
            modes=modes,
            axis=axis,
            fixed_q=fixed_q,
        )
        return anp.real(anp.sum(powers))

    assert_allclose(objective(ks, zs, incident), 1, atol=2e-13)
    for gradient in advect.grad(objective, argnums=(0, 1, 2))(ks, zs, incident):
        assert_allclose(gradient, 0, atol=3e-12)


@pytest.mark.interface
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_cd_native_batch_reference_complete_gradient_and_undefined_limit(
    poltype, direction
):
    basis = tr.PlaneWavePorts.default([[0.2, 0.1]])
    stack = tr.SMatrix.slab(
        0.4,
        basis,
        1.3,
        [1, (2.3, 1.1, 0.08) if poltype == "helicity" else (2.3, 1.1), 1],
        poltype=poltype,
    )
    matrices, incident = stack.array, np.array([0.3 + 0.2j, 0.7 - 0.1j])
    ks, zs, q = np.full((2, 2), 1.3 + 0j), np.ones(2, complex), np.array([[0.2, 0.1]])
    modes = [(0, 0), (0, 1)]
    actual = ad.smatrix_cd(
        matrices, incident, ks, zs, q, modes=modes, poltype=poltype, modetype=direction
    )
    assert_allclose(actual, stack.cd(incident, modetype=direction), atol=2e-13)

    def objective(matrices, incident):
        return anp.real(
            anp.sum(
                ad.smatrix_cd(
                    matrices,
                    incident,
                    ks,
                    zs,
                    q,
                    modes=modes,
                    poltype=poltype,
                    modetype=direction,
                )
            )
        )

    gradients = advect.grad(objective, argnums=(0, 1))(matrices, incident)
    rng = np.random.default_rng(115)
    check_gradient(
        objective,
        lambda *_: gradients,
        matrices,
        incident,
        directions=(
            complex_normal(rng, matrices.shape),
            complex_normal(rng, incident.shape),
        ),
        step=2e-6,
        rtol=2e-7,
        atol=2e-9,
    )
    assert_allclose(np.vdot(gradients[1], incident).real, 0, atol=1e-12)
    reflected = np.zeros_like(matrices)
    reflected[0, 1] = reflected[1, 0] = np.eye(2)
    with pytest.raises(ValueError, match="undefined"):
        objective(reflected, incident)
    with pytest.raises(ValueError, match="undefined"):
        advect.grad(objective)(reflected, incident)
