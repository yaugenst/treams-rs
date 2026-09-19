import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr


@pytest.mark.parametrize("family", ["sw", "cw"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("pol", [0, 1])
@pytest.mark.filterwarnings("ignore:.*scipy.special.sph_harm.*:DeprecationWarning")
def test_single_mode_source_and_all_fields_reference(family, poltype, singular, pol):
    maker, oracle, mode = (
        (tr.spherical_wave, treams.spherical_wave, (2, -1, pol))
        if family == "sw"
        else (tr.cylindrical_wave, treams.cylindrical_wave, (0.2, -1, pol))
    )
    kwargs = dict(
        k0=1.3,
        material=(2.3 + 0.2j, 1.2 + 0.1j, 0.04 if poltype == "helicity" else 0),
        poltype=poltype,
        modetype="singular" if singular else "regular",
    )
    wave = maker(*mode, **kwargs)
    reference = oracle(
        *mode, **(kwargs | {"material": treams.Material(kwargs["material"])})
    )
    assert_allclose(wave.array, np.asarray(reference))
    points = np.array([[[0.7, 0.8, 0.1], [-0.5, 0.3, 0.6]]])
    for kind in ("efield", "hfield", "dfield", "bfield"):
        assert_allclose(
            getattr(wave, kind)(points),
            getattr(reference, kind)(points),
            rtol=1e-11,
            atol=1e-11,
        )
    for kind in ("gfield", "ffield"):
        for helicity in (-1, 1):
            assert_allclose(
                getattr(wave, kind)(helicity, points),
                getattr(reference, kind)(helicity, r=points),
                rtol=1e-11,
                atol=1e-11,
            )


@pytest.mark.parametrize("family", ["sw", "cw"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@settings(max_examples=30)
@given(
    scale=st.complex_numbers(
        min_magnitude=0.2, max_magnitude=2, allow_nan=False, allow_infinity=False
    )
)
def test_weighted_fields_linearity_and_operator_agreement(family, poltype, scale):
    basis = (
        tr.SphericalBasis.default(2)
        if family == "sw"
        else tr.CylindricalBasis.default([0.2], 2)
    )
    amplitudes = np.arange(len(basis)) * (0.03 + 0.04j)
    wave = tr.Wave(
        amplitudes * scale, basis=basis, k0=1.3, material=(2.3, 1.2), poltype=poltype
    )
    points = np.array([[0.5, 0.3, 0.2], [-0.2, 0.5, 0.1]])
    for kind in ("efield", "hfield", "dfield", "bfield"):
        operator = getattr(tr.operators, kind)(
            points, basis=basis, k0=1.3, material=(2.3, 1.2), poltype=poltype
        )
        assert_allclose(
            getattr(wave, kind)(points), (operator @ amplitudes) * scale, atol=1e-13
        )


@pytest.mark.parametrize("family", ["sw", "cw"])
def test_source_expansion_and_complete_tmatrix_illumination(family):
    points = np.array([[0.02, 0.01, 0.03], [-0.03, 0.01, 0.02]])
    if family == "sw":
        source_basis = tr.SphericalBasis.default(2, positions=[[1.1, 0.3, 0.2]])
        wave = tr.spherical_wave(
            2, 1, 1, basis=source_basis, k0=1.3, modetype="singular"
        )
        tm = tr.TMatrix.sphere(8, 1.3, 0.1, [3, 1])
    else:
        source_basis = tr.CylindricalBasis.default(
            [0.2], 2, positions=[[1.1, 0.3, 0.2]]
        )
        wave = tr.cylindrical_wave(
            0.2, 1, 1, basis=source_basis, k0=1.3, modetype="singular"
        )
        tm = tr.CylindricalTMatrix.cylinder([0.2], 8, 1.3, [0.1], [3, 1])
    expanded = wave.expand(tm.basis)
    reconstructed = tr.Wave(expanded, basis=tm.basis, k0=1.3)
    assert_allclose(
        reconstructed.efield(points), wave.efield(points), atol=1e-9, rtol=1e-9
    )
    assert_allclose(tm @ wave, tm.array @ expanded, atol=1e-13)
    wrong = tr.Wave(wave.array, basis=wave.basis, k0=1.4)
    with pytest.raises(ValueError, match="matching"):
        tm @ wrong


def test_source_owned_inputs_and_mode_selection():
    basis = tr.SphericalBasis([(3, 2, -1, 0)], positions=np.zeros((4, 3)))
    wave = tr.spherical_wave(2, -1, 0, basis=basis)
    assert_allclose(wave.array, [1])
    coefficients = np.array([0.3 + 0.2j])
    owned = tr.Wave(coefficients, basis=basis)
    coefficients[:] = 0
    assert_allclose(np.asarray(owned), [0.3 + 0.2j])
    assert not owned.array.flags.writeable
    with pytest.raises(ValueError):
        tr.spherical_wave(2, 0, 0, basis=basis)
    with pytest.raises(ValueError, match="global"):
        tr.spherical_wave(1, 0, 0, basis=tr.SphericalBasis.default(1, 2))
