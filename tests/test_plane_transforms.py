import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import _native
from treams_rs import advect as ad

pytestmark = pytest.mark.filterwarnings(
    "ignore:'where' used without 'out'.*:UserWarning"
)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("turns", [-2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("alignment", ["unit", "xy", "yz", "zx"])
def test_permutation_reference_and_cartesian_field(poltype, turns, alignment):
    if alignment == "unit":
        basis = tr.PlaneWaveBasisByUnitVector.default(
            [[0.2, 0.3, 0.8], [-0.1, 0.7, -0.5]]
        )
        reference = treams.PlaneWaveBasisByUnitVector(basis.modes)
    else:
        basis = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [0.4, 0.6]], alignment)
        reference = treams.PlaneWaveBasisByComp(basis.modes, alignment)
    kwargs = dict(k0=1.3, material=(2.1 + 0.1j, 1.2, 0), poltype=poltype)
    value = tr.permute(turns, basis=basis, **kwargs)
    assert_allclose(value, treams.permute(turns, basis=reference, **kwargs), atol=1e-12)
    points = np.array([[0.1, 0.2, -0.3], [-0.1, 0.15, 0.04]])
    output = tr.efield(
        np.roll(points, turns % 3, axis=1), basis=basis.permute(turns), **kwargs
    )
    expected = np.roll(tr.efield(points, basis=basis, **kwargs), turns % 3, axis=-2)
    assert_allclose(output @ value, expected, atol=1e-12)
    inverse = tr.permute(-turns, basis=basis.permute(turns), **kwargs)
    assert_allclose(inverse @ value, np.eye(len(basis)), atol=1e-12)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(x=st.floats(-0.2, 0.2), scale=st.floats(0.1, 10), n=st.integers(1, 2))
def test_permutation_vector_pullback_and_scale_invariant(poltype, x, scale, n):
    vectors = np.array([[0.4 + x + 0.1j, 0.3, 0.8], [-0.2 + 0.1j, 0.6, -0.4]]) * scale
    pols = [0, 1]
    value, context = tr.diff.plane_permutation(vectors, pols, n, poltype=poltype)
    g = np.array([[0.2 + 0.3j, -0.1j], [0.7, 0.1 - 0.2j]])
    gradient = context.pullback(g)
    direction = np.array([[0.3 + 0.1j, -0.2j, 0.1], [0.1, 0.4, -0.2j]]) * scale
    h = 1e-5
    difference = (
        tr.diff.plane_permutation(vectors + h * direction, pols, n, poltype=poltype)[0]
        - tr.diff.plane_permutation(vectors - h * direction, pols, n, poltype=poltype)[
            0
        ]
    ) / (2 * h)
    assert_allclose(
        np.vdot(gradient, direction).real,
        np.vdot(g, difference).real,
        rtol=1e-7,
        atol=1e-9,
    )
    assert_allclose(np.vdot(gradient, vectors), 0, atol=2e-13)
    assert_allclose(
        value,
        tr.diff.plane_permutation(vectors / scale, pols, n, poltype=poltype)[0],
        atol=1e-12,
    )


def test_permutation_advect_and_context_validation():
    vectors = np.array([[0.3 + 0.1j, 0.4, 0.8]])
    weights = np.array([[0.2j], [0.8]])

    def objective(x):
        output = ad.plane_permutation(x, polarizations=[1], n=1)
        return anp.real(anp.sum(anp.conj(weights) * output))

    _, context = tr.diff.plane_permutation(vectors, [1])
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.ones((1, 1), complex))
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full((2, 1), np.nan, complex))
    expected = context.pullback(weights)
    assert_allclose(advect.grad(objective)(vectors), expected, atol=1e-12)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(weights)
    with pytest.raises(ValueError, match="polarization"):
        tr.diff.plane_permutation(vectors, [0.1])
    # Non-contiguous native input and cotangent do not change the map.
    value, context = _native.plane_permutation(
        vectors[:, ::-1], np.array([1.0]), 2, True
    )
    assert np.isfinite(context.pullback(np.ones_like(value)[:, ::-1])).all()


@pytest.mark.parametrize("unit", [True, False])
@given(phi=st.floats(-3, 3), psi=st.floats(-3, 3))
def test_plane_rotation_labels_reference_and_composition(unit, phi, psi):
    if unit:
        basis = tr.PlaneWaveBasisByUnitVector.default(
            [[0.2, 0.3, 0.8], [0.1, -0.7, -0.5]]
        )
        reference = treams.PlaneWaveBasisByUnitVector(basis.modes)
    else:
        basis = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [0.4, 0.1]])
        reference = treams.PlaneWaveBasisByComp(basis.modes)
    expected = treams.rotate(phi, psi=psi, basis=reference)
    assert_allclose(tr.rotate(phi, psi=psi, basis=basis), expected, atol=1e-12)
    rotated = basis.rotate(phi + psi)
    assert_allclose(
        rotated.modes,
        list(
            expected.basis[0] if isinstance(expected.basis, tuple) else expected.basis
        ),
        atol=1e-12,
    )
    assert_allclose(basis.rotate(phi).rotate(psi).modes, rotated.modes, atol=1e-12)
    assert_allclose(rotated.rotate(-phi - psi).modes, basis.modes, atol=1e-12)
    mask = np.eye(len(basis), dtype=bool)
    mask[0, 0] = False
    assert_allclose(tr.rotate(phi, basis=basis, where=mask), mask.astype(complex))
    with pytest.raises(ValueError, match="theta"):
        tr.rotate(phi, theta=0.1, basis=basis)


def test_plane_permutation_identity_axis_and_invalid_geometry():
    for direction in [[0, 0, 1], [1, 0, 0], [0, -1, 0]]:
        basis = tr.PlaneWaveBasisByUnitVector.default([direction])
        for n in (0, 1, 2):
            transform = tr.permute(n, basis=basis)
            inverse = tr.permute(-n, basis=basis.permute(n))
            assert_allclose(inverse @ transform, np.eye(2), atol=1e-12)
    with pytest.raises(ValueError, match="integer"):
        basis.permute(0.5)
    with pytest.raises(ValueError, match="xy"):
        tr.PlaneWaveBasisByComp.default([[0.2, 0.3]], "yz").rotate(0.3)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_permutation_extreme_scales_and_shared_directions(poltype, scale):
    k = np.array([0.3 + 0.1j, 0.4, 0.8])
    vectors = np.array([k, k, k * np.array([1, -1, 1])])
    pols = [0, 1, 1]
    value, context = tr.diff.plane_permutation(vectors * scale, pols, poltype=poltype)
    expected, original = tr.diff.plane_permutation(vectors, pols, poltype=poltype)
    g = np.array([[0.2 + 0.1j, 0.3, 0.2], [0.4, 0.2j, -0.1]])
    gradient = context.pullback(g) * scale
    assert_allclose(value, expected, atol=1e-12)
    assert_allclose(gradient, original.pullback(g), atol=1e-12)
    # Adjacent equal directions share arithmetic, but are independent AD inputs.
    h = 1e-5
    for i in (0, 1):
        direction = np.zeros_like(vectors)
        direction[i] = [0.1j, 0.2, -0.1j]
        plus = tr.diff.plane_permutation(
            vectors + h * direction, pols, poltype=poltype
        )[0]
        minus = tr.diff.plane_permutation(
            vectors - h * direction, pols, poltype=poltype
        )[0]
        assert_allclose(
            np.vdot(gradient, direction).real,
            np.vdot(g, (plus - minus) / (2 * h)).real,
            atol=1e-9,
        )
