import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr


def _basis(family):
    if family == "sw":
        return tr.SphericalWaveBasis.default(2, 2)
    if family == "cw":
        return tr.CylindricalWaveBasis.default([0.2], 2, 2)
    if family == "unit":
        return tr.PlaneWaveBasisByUnitVector.default([[0.2, 0.3, 1], [0.1, 0.2, -1]])
    return tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [1.7, 0.2]], family)


def _copy(basis, indices, oracle=False):
    cls = getattr(treams, type(basis).__name__) if oracle else type(basis)
    kwargs = {}
    if isinstance(basis, (tr.SphericalWaveBasis, tr.CylindricalWaveBasis)):
        kwargs["positions"] = basis.positions
    elif isinstance(basis, tr.PlaneWaveBasisByComp):
        kwargs["alignment"] = basis.alignment
    return cls([basis.modes[i] for i in indices], **kwargs)


@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
@pytest.mark.parametrize(
    "poltype", [None, "helicity", "parity", ("parity", "helicity")]
)
def test_rectangular_masked_polarization_reference(family, poltype):
    basis = _basis(family)
    to, source = list(range(len(basis)))[::2], list(range(len(basis)))[::-3]
    where = np.arange(len(to) * len(source)).reshape(len(to), len(source)) % 3 != 0
    actual = tr.changepoltype(
        poltype, basis=(_copy(basis, to), _copy(basis, source)), where=where
    )
    expected = treams.changepoltype(
        poltype, basis=(_copy(basis, to, True), _copy(basis, source, True)), where=where
    )
    assert_allclose(actual, expected, atol=0)


@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
@settings(max_examples=25)
@given(seed=st.integers(0, 1000))
def test_polarization_involution_and_field_invariance(family, seed):
    basis = _basis(family)
    rng = np.random.default_rng(seed)
    basis = _copy(basis, rng.permutation(len(basis)))
    change = tr.changepoltype(basis=basis)
    coefficients = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    assert_allclose(change @ change, np.eye(len(basis)), atol=1e-14)
    point = [0.7, 0.5, 0.3]
    for kind in (tr.efield, tr.hfield):
        helicity = kind(point, basis=basis, k0=1.3, material=(2.3, 1.2)) @ coefficients
        parity = (
            kind(point, basis=basis, k0=1.3, material=(2.3, 1.2), poltype="parity")
            @ change
            @ coefficients
        )
        assert_allclose(helicity, parity, atol=1e-12)


@pytest.mark.parametrize("family", ["sw", "cw"])
def test_source_polarization_preserves_fields(family):
    basis = _basis(family)
    wave = tr.MultipoleWave(
        np.arange(len(basis)) * (0.1 + 0.2j), basis=basis, k0=1.3, material=(2.3, 1.2)
    )
    converted = wave.changepoltype()
    assert converted.poltype == "parity"
    assert_allclose(converted.changepoltype().array, wave.array, atol=1e-13)
    for kind in ("efield", "hfield", "dfield", "bfield"):
        assert_allclose(
            getattr(converted, kind)([0.7, 0.3, 0.1]),
            getattr(wave, kind)([0.7, 0.3, 0.1]),
            atol=1e-12,
        )


def test_polarization_requires_matching_families_and_complete_source_pairs():
    with pytest.raises(ValueError, match="wave family"):
        tr.changepoltype(basis=(_basis("sw"), _basis("cw")))
    with pytest.raises(ValueError, match="alignments"):
        tr.changepoltype(basis=(_basis("xy"), _basis("yz")))
    with pytest.raises(ValueError, match="switch"):
        tr.changepoltype(("helicity", "helicity"), basis=_basis("sw"))
    wave = tr.spherical_wave(1, 0, 1, basis=tr.SphericalWaveBasis([(1, 0, 1)]))
    with pytest.raises(ValueError, match="both polarizations"):
        wave.changepoltype()
