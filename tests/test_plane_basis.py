import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr


@pytest.mark.parametrize("kind", ["efield", "hfield", "dfield", "bfield"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_full_plane_basis_fields_and_expansion(kind, poltype):
    vectors = [[0, 0, 1], [0.2, 0.3, -0.8], [1.5, 0.1, 0.3j]]
    basis = tr.PlaneWaveBasisByUnitVector.default(vectors)
    oracle = treams.PlaneWaveBasisByUnitVector.default(vectors)
    material = (1.3 + 0.1j, 1.2, 0.03 if poltype == "helicity" else 0)
    points = [[0.1, 0.2, 0.3], [-0.1, 0.3, 0.2]]
    kwargs = dict(k0=1.3, material=material, poltype=poltype)
    assert_allclose(
        getattr(tr, kind)(points, basis=basis, **kwargs),
        getattr(treams, kind)(points, basis=oracle, **kwargs),
        rtol=1e-12,
        atol=1e-12,
    )
    # Avoid the separate upstream complex-axis angular rounding defect.
    source = tr.PlaneWaveBasisByUnitVector.default(vectors[1:])
    destination = tr.SphericalWaveBasis.default(
        3, 2, [[0.1, 0.2, 0.3], [0.2, -0.1, 0.3]]
    )
    actual = tr.expand((destination, source), **kwargs)
    expected = treams.expand(
        (
            treams.SphericalWaveBasis(destination.modes, destination.positions),
            treams.PlaneWaveBasisByUnitVector(source.modes),
        ),
        **kwargs,
    )
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("modetype", ["up", "down"])
@pytest.mark.parametrize("material", [1, (1.3 + 0.1j, 1.2, 0.03)])
def test_component_unit_round_trip(modetype, material, alignment):
    basis = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [1.7, -0.2]], alignment)
    full = basis.byunitvector(1.3, material, modetype)
    assert_allclose(full.directions**2 @ np.ones(3), 1, rtol=1e-12, atol=1e-12)
    assert_allclose(
        np.column_stack(full.kvecs(1.3, material)),
        np.column_stack(basis.kvecs(1.3, material, modetype)),
        rtol=1e-12,
        atol=1e-12,
    )
    partial = full.bycomp(1.3, alignment, material=material)
    assert_allclose(partial.modes, basis.modes, rtol=1e-12, atol=1e-12)


@given(n=st.integers(-30, 30), scale=st.floats(0.01, 100))
def test_unit_basis_permutation_and_normalization(n, scale):
    vectors = np.array([[0.2, 0.3, 0.9], [1.5, 0.1, 0.3j]])
    basis = tr.PlaneWaveBasisByUnitVector.default(vectors)
    scaled = tr.PlaneWaveBasisByUnitVector.default(vectors * scale)
    assert_allclose(scaled.directions, basis.directions, rtol=1e-12, atol=1e-12)
    assert_allclose(
        basis.permute(n).permute(-n).directions,
        basis.directions,
        rtol=1e-12,
        atol=1e-12,
    )
    oracle = treams.PlaneWaveBasisByUnitVector.default(vectors).permute(n)
    assert_allclose(
        basis.permute(n).directions,
        np.column_stack([oracle.qx, oracle.qy, oracle.qz]),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_plane_wave_extreme_scale_invariance(scale):
    vector = np.array([0.2, 0.3, 0.9])
    wave = tr.plane_wave(vector * scale, [0.2 + 0.1j, 0.7], k0=1.3)
    full = tr.PlaneWaveBasisByUnitVector.default([vector * scale])
    assert_allclose(wave.expand(full), [0.7, 0.2 + 0.1j], rtol=1e-12, atol=1e-12)
    other = tr.plane_wave(vector, [0.2 + 0.1j, 0.7], k0=1.3)
    assert_allclose(
        wave.efield([[0.1, 0.2, 0.3]]),
        other.efield([[0.1, 0.2, 0.3]]),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_aligned_plane_fields_illumination_and_permutation(alignment, poltype):
    basis = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [1.7, -0.2]], alignment)
    oracle = treams.PlaneWaveBasisByComp(basis.modes, alignment)
    points = [[0.1, 0.2, 0.3], [-0.1, 0.3, -0.2]]
    medium = (1.3 + 0.1j, 1.2, 0.03 if poltype == "helicity" else 0)
    for side in ("up", "down"):
        assert_allclose(
            basis.kvecs(1.3, medium, side),
            oracle.kvecs(1.3, medium, side),
            rtol=1e-12,
            atol=1e-12,
        )
        for kind in ("efield", "hfield", "dfield", "bfield"):
            args = dict(k0=1.3, material=medium, modetype=side, poltype=poltype)
            assert_allclose(
                getattr(tr, kind)(points, basis=basis, **args),
                getattr(treams, kind)(points, basis=oracle, **args),
                rtol=1e-12,
                atol=1e-12,
            )
    for axis in "xyz":
        component = getattr(basis, "k" + axis)
        if axis in alignment:
            assert_allclose(component, getattr(oracle, "k" + axis))
        else:
            assert component is None
    vectors = np.column_stack(basis.kvecs(1.3, medium))
    for n in (-4, -1, 1, 2, 3, 7):
        assert_allclose(
            np.column_stack(basis.permute(n).kvecs(1.3, medium)),
            np.roll(vectors, n % 3, axis=1),
            rtol=1e-12,
            atol=1e-12,
        )
    wave = tr.plane_wave([0.2, 0.3, 0.9], [0.2 + 0.1j, 0.7], k0=1.3, poltype=poltype)
    basis = tr.PlaneWaveBasisByUnitVector.default([[0.2, 0.3, 0.9]]).bycomp(
        1.3, alignment
    )
    assert_allclose(wave.expand(basis), [0.7, 0.2 + 0.1j], rtol=1e-12, atol=1e-12)
