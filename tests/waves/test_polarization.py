"""Polarization conventions (helicity and parity): operators.changepoltype and the
object methods against upstream, involution and field invariance, and the explicit
per-call convention with its helicity default."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

import treams_rs as tr
from treams_rs import _operator_objects
from treams_rs import advect as ad

from _support import complex_normal, to_oracle


def _basis(family):
    if family == "sw":
        return tr.SphericalBasis.default(2, 2)
    if family == "cw":
        return tr.CylindricalBasis.default([0.2], 2, 2)
    if family == "unit":
        return tr.PlaneWaveBasis.default([[0.2, 0.3, 1], [0.1, 0.2, -1]])
    return tr.PlaneWavePorts.default([[0.2, 0.3], [1.7, 0.2]], family)


def _copy(basis, indices):
    kwargs = {}
    if isinstance(basis, (tr.SphericalBasis, tr.CylindricalBasis)):
        kwargs["positions"] = basis.positions
    elif isinstance(basis, tr.PlaneWavePorts):
        kwargs["alignment"] = basis.alignment
    return type(basis)([basis.modes[i] for i in indices], **kwargs)


@pytest.mark.reference
@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
@pytest.mark.parametrize(
    "poltype", [None, "helicity", "parity", ("parity", "helicity")]
)
def test_rectangular_masked_polarization_reference(family, poltype):
    basis = _basis(family)
    to, source = list(range(len(basis)))[::2], list(range(len(basis)))[::-3]
    where = np.arange(len(to) * len(source)).reshape(len(to), len(source)) % 3 != 0
    actual = tr.operators.changepoltype(
        poltype, basis=(_copy(basis, to), _copy(basis, source)), where=where
    )
    expected = treams.changepoltype(
        poltype,
        basis=tuple(
            to_oracle(basis, [basis.modes[i] for i in x]) for x in (to, source)
        ),
        where=where,
    )
    assert_allclose(actual, expected, atol=0)


@pytest.mark.physics
@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
@settings(max_examples=25)
@given(data=st.data())
def test_polarization_involution_and_field_invariance(family, data):
    basis = _basis(family)
    order = data.draw(st.permutations(range(len(basis))), label="mode order")
    basis = _copy(basis, order)
    change = tr.operators.changepoltype(basis=basis)
    coefficients = complex_normal(np.random.default_rng(17), len(basis))
    assert_allclose(change @ change, np.eye(len(basis)), atol=1e-14)
    point = [0.7, 0.5, 0.3]
    for kind in (tr.operators.efield, tr.operators.hfield):
        helicity = kind(point, basis=basis, k0=1.3, material=(2.3, 1.2)) @ coefficients
        parity = (
            kind(point, basis=basis, k0=1.3, material=(2.3, 1.2), poltype="parity")
            @ change
            @ coefficients
        )
        assert_allclose(helicity, parity, atol=1e-12)


@pytest.mark.physics
@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
@settings(max_examples=20)
@given(seed=st.integers(0, 2**32 - 1), source=st.sampled_from(["helicity", "parity"]))
def test_object_polarization_changes_apply_the_operator(family, seed, source):
    # Waves, T-matrices and S-matrices change convention by the explicit
    # (symmetric, involutive) operator C: C x, C T C^T and C S_ij C^T.
    rng = np.random.default_rng(seed)
    basis = _copy(_basis(family), rng.permutation(len(_basis(family))))
    change = tr.operators.changepoltype(basis=basis)
    target = "parity" if source == "helicity" else "helicity"
    plane = family not in ("sw", "cw")
    coefficients = complex_normal(rng, (len(basis), 3))
    wave = tr.Wave(
        coefficients,
        basis=basis,
        k0=1.3,
        material=(2.3, 1.2),
        modetype="up" if plane else "regular",
        poltype=source,
    )
    converted = wave.changepoltype()
    assert converted.poltype == target
    assert_allclose(converted.array, change @ coefficients, rtol=0, atol=1e-14)
    assert_allclose(wave.changepoltype(target).array[:, 0], converted.array[:, 0])
    assert wave.changepoltype(source) is wave
    if family in ("sw", "cw"):
        matrix = complex_normal(rng, (len(basis), len(basis)))
        family_type = tr.TMatrix if family == "sw" else tr.CylindricalTMatrix
        tm = family_type(matrix, basis=basis, k0=1.3, poltype=source)
        expected = change @ matrix @ change.T
        assert_allclose(tm.changepoltype().array, expected, rtol=0, atol=1e-14)
    elif family != "unit":
        blocks = complex_normal(rng, (2, 2, len(basis), len(basis)))
        sm = tr.SMatrix(blocks, basis=basis, k0=1.3, poltype=source)
        expected = change @ blocks @ change.T
        assert_allclose(sm.changepoltype().array, expected, rtol=0, atol=1e-14)


@pytest.mark.physics
@pytest.mark.parametrize("family", ["sw", "cw"])
def test_source_polarization_preserves_fields(family):
    basis = _basis(family)
    wave = tr.Wave(
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


@pytest.mark.interface
def test_polarization_requires_matching_families_and_complete_source_pairs():
    with pytest.raises(ValueError, match="wave family"):
        tr.operators.changepoltype(basis=(_basis("sw"), _basis("cw")))
    with pytest.raises(ValueError, match="alignments"):
        tr.operators.changepoltype(basis=(_basis("xy"), _basis("yz")))
    with pytest.raises(ValueError, match="switch"):
        tr.operators.changepoltype(("helicity", "helicity"), basis=_basis("sw"))
    wave = tr.spherical_wave(1, 0, 1, k0=1, basis=tr.SphericalBasis([(1, 0, 1)]))
    with pytest.raises(ValueError, match="both polarizations"):
        wave.changepoltype()
    partial = tr.PlaneWavePorts([(0.1, 0.2, 1), (0.1, 0.2, 0), (0.3, 0.2, 1)])
    network = tr.SMatrix(
        np.eye(3)[None, None].repeat(2, 0).repeat(2, 1), basis=partial, k0=1
    )
    with pytest.raises(ValueError, match="both polarizations"):
        network.changepoltype()
    with pytest.raises(ValueError, match="switch helicity and parity"):
        network.changepoltype("chirality")


@pytest.mark.interface
@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
def test_changepoltype_defaults_to_helicity(family):
    basis = _basis(family)
    assert_array_equal(
        tr.operators.changepoltype(basis=basis),
        tr.operators.changepoltype("helicity", basis=basis),
    )
    array = tr.operators.PhysicsArray(np.ones(len(basis)), basis=basis)
    assert array.poltype == "helicity"
    assert tr.operators.ChangePoltype().get_kwargs(array) == {
        "basis": basis,
        "poltype": "parity",
    }


@pytest.mark.interface
@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
@pytest.mark.parametrize("source", ["helicity", "parity"])
def test_change_poltype_inverse_switches_back(family, source, monkeypatch):
    basis = _basis(family)
    target = "parity" if source == "helicity" else "helicity"
    values = complex_normal(np.random.default_rng(5), len(basis))
    array = tr.operators.PhysicsArray(values, basis=basis, poltype=source)
    change = tr.operators.ChangePoltype()
    assert change.get_kwargs(array)["poltype"] == target
    converted = tr.operators.PhysicsArray(change @ array, basis=basis, poltype=target)
    assert_allclose(change.inv @ converted, values, rtol=0, atol=1e-14)

    # The inverse names its destination: the convention it converts back to.
    destinations = []

    def spy(poltype=None, **kwargs):
        destinations.append(poltype)
        return tr.operators.changepoltype(poltype, **kwargs)

    monkeypatch.setattr(_operator_objects, "changepoltype", spy)
    change.inv(basis=basis, poltype=target)
    change.inv(basis=basis)
    tr.operators.ChangePoltype(target).inv(basis=basis)
    to, source_modes = _copy(basis, range(0, len(basis), 2)), _copy(basis, range(1))
    pair = tr.operators.ChangePoltype((target, source)).inv(basis=(to, source_modes))
    assert destinations == [source, "parity", source, (source, target)]
    assert_array_equal(
        pair,
        tr.operators.changepoltype((source, target), basis=(source_modes, to)),
    )
    with pytest.raises(ValueError, match="helicity or parity"):
        tr.operators.ChangePoltype("chirality").inv(basis=basis)
    array.poltype = "chirality"
    with pytest.raises(ValueError, match="helicity or parity"):
        tr.operators.ChangePoltype().get_kwargs(array)


@pytest.mark.interface
@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_explicit_polarization_and_stable_default(kind):
    radius, k0 = 0.3, 1.3

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
    # Guards against a direct parity construction losing the chiral coupling.
    assert_allclose(parity.array, helicity.changepoltype("parity").array, atol=0)
    assert parity.poltype == "parity"
    assert_allclose(particle().array, helicity.array, atol=0)
    assert particle().poltype == "helicity"
    assert tr.spherical_wave(1, 0, 1, k0=1).poltype == "helicity"
    with pytest.raises(ValueError, match="polarization"):
        particle("invalid")


@pytest.mark.gradients
@pytest.mark.interface
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
