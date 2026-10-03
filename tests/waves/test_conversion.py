"""Cylindrical-to-spherical expansion: operators.expand against upstream, fields rebuilt
through diff.expansion and diff.field, and native and Advect pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal


@pytest.mark.reference
@pytest.mark.parametrize(
    "poltype,epsilon,kappa",
    [
        ("helicity", 2, 0),
        ("parity", 2, 0),
        ("helicity", 2 + 0.2j, 0),
        ("parity", 2 + 0.2j, 0),
        ("helicity", 2 + 0.2j, 0.1),
    ],
)
def test_conversion_reference(poltype, epsilon, kappa):
    material = tr.Material(epsilon, 1.2, kappa)
    to = tr.SphericalBasis.default(5)
    source = tr.CylindricalBasis.default([-0.4, 0, 0.3, 2.7], 4)
    actual = tr.operators.expand(
        (to, source), k0=1.3, material=material, poltype=poltype
    )
    expected = treams.expand(
        (
            treams.SphericalWaveBasis(to.modes),
            treams.CylindricalWaveBasis(source.modes),
        ),
        k0=1.3,
        material=treams.Material(epsilon, 1.2, kappa),
        poltype=poltype,
    )
    assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.physics
@given(
    m=st.integers(-3, 3),
    kz=st.floats(-0.7, 0.7),
    pol=st.integers(0, 1),
    offset=st.lists(st.floats(-0.3, 0.3), min_size=3, max_size=3),
    helicity=st.booleans(),
)
def test_displaced_conversion_reconstructs_field(m, kz, pol, offset, helicity):
    to = tr.SphericalBasis.default(10, positions=[offset])
    source = tr.CylindricalBasis([(kz, m, pol)], positions=[[-0.2, 0.1, 0.3]])
    poltype = "helicity" if helicity else "parity"
    ks = np.array([1.3 + 0.1j, 1.5 + 0.2j]) if helicity else np.full(2, 1.3 + 0.1j)
    points = np.array(offset) + np.array(
        [[0, 0, 0], [0.1, -0.2, 0.1], [-0.2, 0.1, 0.05]]
    )
    converted = tr.diff.expansion(to, source, ks, poltype=poltype)[0][:, 0]
    actual = tr.diff.field(converted, points, to, ks, poltype=poltype)[0]
    expected = tr.diff.field([1], points, source, ks, poltype=poltype)[0]
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("coincident", [False, True])
def test_conversion_pullback(poltype, coincident):
    destination = tr.SphericalBasis.default(3, 2)
    source = tr.CylindricalBasis.default([-0.3, 0.2], 2, 2)
    rng = np.random.default_rng(52)
    origins = np.zeros((2, 3)) if coincident else rng.normal(size=(2, 3)) * 0.2
    source_origins = np.zeros((2, 3)) if coincident else rng.normal(size=(2, 3)) * 0.2
    ks = (
        np.array([1.3 + 0.1j, 1.5 + 0.2j])
        if poltype == "helicity"
        else np.full(2, 1.3 + 0.1j)
    )
    values = [origins, source_origins, ks]
    directions = [rng.normal(size=x.shape) * 0.2 for x in values]
    directions[2] = directions[2].astype(complex) + 0.1j
    if poltype == "parity":
        # The parity basis requires equal wavenumbers.
        directions[2][:] = directions[2][0]

    def record(origins, source_origins, ks):
        return tr.diff.expansion(
            type(destination)(destination.modes, origins),
            type(source)(source.modes, source_origins),
            ks,
            poltype=poltype,
        )

    value, context = record(*values)
    cotangent = complex_normal(rng, value.shape)
    gradients = context.pullback(cotangent)
    assert_allclose(gradients[0].sum(axis=0) + gradients[1].sum(axis=0), 0, atol=2e-12)
    check_pullback(
        record,
        *values,
        directions=tuple(directions),
        cotangents=cotangent,
        step=1e-5,
        rtol=3e-7,
        atol=2e-8,
    )


@pytest.mark.gradients
def test_advect_conversion_field_composition():
    source = tr.CylindricalBasis.default([0.3], 2)
    destination = tr.SphericalBasis.default(4)
    coefficients = np.arange(len(source)) * (0.1 + 0.03j)
    origins = np.array([[0.1, -0.2, 0.3]])
    points = np.array([[0.2, 0.1, 0.1], [-0.1, 0.3, 0.2]])

    def objective(k):
        ks = anp.stack([k, k * 1.1])
        expansion = ad.expansion(
            origins, source.positions, ks, destination=destination, source=source
        )
        field = ad.field(
            expansion @ coefficients, points, origins, ks, basis=destination
        )
        return anp.sum(anp.real(field * anp.conj(field)))

    check_gradient(
        objective,
        advect.grad(objective),
        np.array(1.3 + 0.1j),
        directions=(np.array(0.2 - 0.3j),),
        step=1e-5,
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.physics
@pytest.mark.reference
@pytest.mark.parametrize(
    "modetype",
    [("regular", "regular"), ("regular", "singular"), ("singular", "singular")],
)
@pytest.mark.parametrize("cylindrical", [True, False])
def test_public_expand_addition_theorem(modetype, cylindrical):
    basis_type = tr.CylindricalBasis if cylindrical else tr.SphericalBasis
    basis = basis_type.default([0.2], 2) if cylindrical else basis_type.default(2)
    destination = type(basis)(basis.modes, [[0.3, 0.2, -0.1]])
    oracle = treams.CylindricalWaveBasis if cylindrical else treams.SphericalWaveBasis
    actual = tr.operators.expand((destination, basis), modetype, k0=1.3)
    expected = treams.expand(
        (
            oracle(destination.modes, destination.positions),
            oracle(basis.modes, basis.positions),
        ),
        modetype,
        k0=1.3,
    )
    assert_allclose(actual, expected, rtol=2e-11, atol=1e-11)
