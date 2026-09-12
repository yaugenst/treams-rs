import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(radius=st.floats(0.1, 0.5), k0=st.floats(0.8, 1.8))
def test_global_polarization_defaults_and_owned_object_conventions(poltype, radius, k0):
    previous = tr.config.POLTYPE
    try:
        tr.config.POLTYPE = poltype
        sphere = tr.TMatrix.sphere(2, k0, radius, [2.3, 1])
        cylinder = tr.TMatrixC.cylinder([0.2], 2, k0, [radius], [2.3, 1])
        plane = tr.plane_wave([0.2, 0.3, 0.9], [0.2, 0.8], k0=k0)
        source = tr.spherical_wave(2, -1, 0, k0=k0)
        array = tr.PhysicsArray(source.array, basis=source.basis, k0=k0)
        ports = tr.PlaneWaveBasisByComp.default([[0.2, 0.3]])
        slab = tr.SMatrices.slab(radius, ports, k0, [1, 2.3, 1])
        for obj in (sphere, cylinder, plane, source, array, slab):
            assert obj.poltype == poltype
        assert_allclose(
            sphere.array,
            tr.TMatrix.sphere(2, k0, radius, [2.3, 1], poltype=poltype).array,
            atol=0,
        )
        points = [[0.1, 0.2, 0.3]]
        assert_allclose(
            tr.efield(points, basis=source.basis, k0=k0),
            tr.efield(points, basis=source.basis, k0=k0, poltype=poltype),
            atol=0,
        )
        assert_allclose(
            tr.pw.to_sw(2, -1, 0, 0.2, 0.3, 1.2, 1),
            tr.pw.to_sw(2, -1, 0, 0.2, 0.3, 1.2, 1, poltype=poltype),
            atol=0,
        )
        assert_allclose(
            tr.sw.translate(2, 0, 0, 2, 1, 1, 1.3, 0.2, 0.3),
            tr.sw.translate(2, 0, 0, 2, 1, 1, 1.3, 0.2, 0.3, poltype=poltype),
            atol=0,
        )
        expected = source.efield(points)
        tr.config.POLTYPE = "parity" if poltype == "helicity" else "helicity"
        assert_allclose(source.efield(points), expected, atol=0)
        assert source.poltype == poltype
        assert_allclose(
            tr.efield(points, basis=source.basis, k0=k0, poltype=poltype)
            @ source.array,
            expected,
            atol=1e-14,
        )
    finally:
        tr.config.POLTYPE = previous


def test_default_polarization_complete_advect_field_gradient():
    basis = tr.SphericalWaveBasis.default(2)
    amplitudes = np.linspace(0.1, 0.4, len(basis)).astype(complex)
    points = np.array([[0.1, 0.2, 0.3]])
    previous = tr.config.POLTYPE
    try:
        tr.config.POLTYPE = "parity"

        def objective(points, explicit):
            kwargs = {"poltype": "parity"} if explicit else {}
            e = ad.field(
                amplitudes, points, basis.positions, [1.3, 1.3], basis=basis, **kwargs
            )
            h = ad.hfield(
                amplitudes,
                points,
                basis.positions,
                [1.3, 1.3],
                1.0,
                basis=basis,
                **kwargs,
            )
            return anp.sum(anp.real(e * anp.conj(e) + h * anp.conj(h)))

        assert_allclose(objective(points, False), objective(points, True), atol=0)
        assert_allclose(
            advect.grad(lambda x: objective(x, False))(points),
            advect.grad(lambda x: objective(x, True))(points),
            atol=0,
        )
    finally:
        tr.config.POLTYPE = previous


def test_invalid_global_default_is_rejected_and_explicit_choice_is_independent():
    previous = tr.config.POLTYPE
    try:
        tr.config.POLTYPE = "invalid"
        with pytest.raises(ValueError, match="polarization"):
            tr.plane_wave([0, 0, 1], 0)
        with pytest.raises(ValueError, match="poltype"):
            tr.sw.translate(1, 0, 0, 1, 0, 0, 1, 0, 0)
        assert tr.plane_wave([0, 0, 1], 0, poltype="helicity").poltype == "helicity"
    finally:
        tr.config.POLTYPE = previous


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
@given(radius=st.floats(0.1, 0.5), k0=st.floats(0.8, 1.8))
def test_particle_constructors_convert_from_native_helicity_independent_of_default(
    kind, radius, k0
):
    def particle(poltype=None):
        if kind == "sphere":
            return tr.TMatrix.sphere(
                2, k0, radius, [(2.3, 1.1, 0.07), 1], poltype=poltype
            )
        return tr.TMatrixC.cylinder(
            [0.2], 2, k0, radius, [(2.3, 1.1, 0.07), 1], poltype=poltype
        )

    previous = tr.config.POLTYPE
    try:
        tr.config.POLTYPE = "helicity"
        helicity = particle()
        expected = helicity.changepoltype("parity")
        tr.config.POLTYPE = "parity"
        assert_allclose(particle().array, expected.array, rtol=0, atol=0)
        assert_allclose(particle("parity").array, expected.array, rtol=0, atol=0)
        assert_allclose(particle("helicity").array, helicity.array, rtol=0, atol=0)
        tr.config.POLTYPE = "invalid"
        assert_allclose(particle("helicity").array, helicity.array, rtol=0, atol=0)
    finally:
        tr.config.POLTYPE = previous
