"""Single-mode spherical and cylindrical waves (tr.spherical_wave, tr.cylindrical_wave):
upstream references, T-matrix illumination and batched field sampling."""

import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr

# The batch-field test patches the internal operator chunk size of the field sampler.
from treams_rs import _fields

from _support import complex_normal


@pytest.mark.reference
@pytest.mark.parametrize("family", ["sw", "cw"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("pol", [0, 1])
def test_single_mode_source_reference(family, poltype, singular, pol):
    # Field sampling of these objects is checked against the field operators in
    # tests/api/test_operators.py; each operator family has an upstream oracle.
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
    assert wave.basis.modes == tuple(tuple(mode) for mode in reference.basis)


@pytest.mark.workflows
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


@pytest.mark.interface
def test_source_owned_inputs_and_mode_selection():
    basis = tr.SphericalBasis([(3, 2, -1, 0)], positions=np.zeros((4, 3)))
    wave = tr.spherical_wave(2, -1, 0, k0=1, basis=basis)
    assert_allclose(wave.array, [1])
    coefficients = np.array([0.3 + 0.2j])
    owned = tr.Wave(coefficients, basis=basis, k0=1)
    coefficients[:] = 0
    assert_allclose(np.asarray(owned), [0.3 + 0.2j])
    assert not owned.array.flags.writeable
    with pytest.raises(ValueError):
        tr.spherical_wave(2, 0, 0, k0=1, basis=basis)
    with pytest.raises(ValueError, match="global"):
        tr.spherical_wave(1, 0, 0, k0=1, basis=tr.SphericalBasis.default(1, 2))


@pytest.mark.physics
@pytest.mark.parametrize("family", ["sw", "cw", "unit", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@settings(max_examples=10)
@given(
    seed=st.integers(0, 2**32 - 1),
    shape=st.sampled_from([(), (1,), (4,), (2, 3)]),
    columns=st.integers(1, 3),
)
def test_batch_fields_are_the_weighted_single_column_samples(
    family, poltype, seed, shape, columns
):
    # A (modes, B) batch samples all columns with one operator per chunk of
    # points; each column must equal the native weighted kernel for that
    # vector alone, for every field kind and any point shape or chunking.
    rng = np.random.default_rng(seed)
    if family == "sw":
        basis, kind = tr.SphericalBasis.default(1, 2, [[0, 0, 0], [0.4, 0, 0]]), {}
    elif family == "cw":
        basis, kind = tr.CylindricalBasis.default([0.2, -0.1], 1), {}
    elif family == "unit":
        basis = tr.PlaneWaveBasis.default([[0.2, 0.3, 1], [0.1j, 0.2, 1]])
        kind = {"modetype": "up"}
    else:
        basis = tr.PlaneWavePorts.default([[0.2, 0.3], [2.1, 0.1]], "zx")
        kind = {"modetype": "down"}
    material = (2.3 + 0.1j, 1.2, 0.05 if poltype == "helicity" else 0)
    coefficients = complex_normal(rng, (len(basis), columns))
    options = dict(basis=basis, k0=1.3, material=material, poltype=poltype, **kind)
    batch = tr.Wave(coefficients, **options)
    points = rng.normal(size=(*shape, 3))
    for field in ("efield", "hfield", "dfield", "bfield", "gfield", "ffield"):
        args = (1, points) if field in ("gfield", "ffield") else (points,)
        with pytest.MonkeyPatch.context() as patch:
            # One point per operator chunk exercises the chunked assembly.
            patch.setattr(_fields, "BATCH_OPERATOR_ENTRIES", 1)
            actual = getattr(batch, field)(*args)
        assert actual.shape == (*shape, 3, columns)
        assert_allclose(getattr(batch, field)(*args), actual, rtol=1e-14, atol=1e-15)
        for column in range(columns):
            single = tr.Wave(coefficients[:, column], **options)
            assert_allclose(
                actual[..., column],
                getattr(single, field)(*args),
                rtol=1e-12,
                atol=1e-13,
            )
