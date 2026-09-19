"""Explicit polarization conventions are independent between calculations."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
@given(radius=st.floats(0.1, 0.5), k0=st.floats(0.8, 1.8))
def test_explicit_polarization_and_stable_default(kind, radius, k0):
    def particle(poltype=None):
        if kind == "sphere":
            return tr.TMatrix.sphere(
                2, k0, radius, [(2.3, 1.1, 0.07), 1], poltype=poltype
            )
        return tr.CylindricalTMatrix.cylinder(
            [0.2], 2, k0, radius, [(2.3, 1.1, 0.07), 1], poltype=poltype
        )

    helicity = particle()
    parity = particle("parity")
    assert_allclose(parity.array, helicity.changepoltype("parity").array, atol=0)
    assert_allclose(particle().array, helicity.array, atol=0)
    assert particle().poltype == "helicity"
    with pytest.raises(ValueError, match="polarization"):
        particle("invalid")


def test_default_polarization_complete_advect_field_gradient():
    basis = tr.SphericalBasis.default(2)
    amplitudes = np.linspace(0.1, 0.4, len(basis)).astype(complex)
    points = np.array([[0.1, 0.2, 0.3]])

    def objective(points, explicit):
        kwargs = {"poltype": "helicity"} if explicit else {}
        e = ad.field(
            amplitudes, points, basis.positions, [1.3, 1.3], basis=basis, **kwargs
        )
        h = ad.hfield(
            amplitudes, points, basis.positions, [1.3, 1.3], 1.0, basis=basis, **kwargs
        )
        return anp.sum(anp.real(e * anp.conj(e) + h * anp.conj(h)))

    assert_allclose(objective(points, False), objective(points, True), atol=0)
    assert_allclose(
        advect.grad(lambda x: objective(x, False))(points),
        advect.grad(lambda x: objective(x, True))(points),
        atol=0,
    )


def test_sources_and_operators_share_the_explicit_convention():
    for poltype in ("helicity", "parity"):
        wave = tr.spherical_wave(2, -1, 0, k0=1.3, poltype=poltype)
        points = [[0.1, 0.2, 0.3]]
        assert_allclose(
            wave.efield(points),
            tr.operators.efield(points, basis=wave.basis, k0=1.3, poltype=poltype)
            @ wave.array,
            atol=1e-14,
        )
    assert tr.spherical_wave(1, 0, 1).poltype == "helicity"
