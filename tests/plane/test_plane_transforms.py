"""Plane-wave permutations and rotations: diff.plane_permutation and the operators
against upstream, exact quarter-turn label maps, and pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr

# _native: the permutation record is also checked at the binding, in every layout.
from treams_rs import _native
from treams_rs import advect as ad
from treams_rs.testing import check_pullback

from _support import LAYOUTS, arrange


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("turns", [-2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("alignment", ["unit", "xy", "yz", "zx"])
def test_permutation_reference_and_cartesian_field(poltype, turns, alignment):
    if alignment == "unit":
        basis = tr.PlaneWaveBasis.default([[0.2, 0.3, 0.8], [-0.1, 0.7, -0.5]])
        reference = treams.PlaneWaveBasisByUnitVector(basis.modes)
    else:
        basis = tr.PlaneWavePorts.default([[0.2, 0.3], [0.4, 0.6]], alignment)
        reference = treams.PlaneWaveBasisByComp(basis.modes, alignment)
    kwargs = dict(k0=1.3, material=(2.1 + 0.1j, 1.2, 0), poltype=poltype)
    value = tr.operators.permute(turns, basis=basis, **kwargs)
    assert_allclose(value, treams.permute(turns, basis=reference, **kwargs), atol=1e-12)
    points = np.array([[0.1, 0.2, -0.3], [-0.1, 0.15, 0.04]])
    output = tr.operators.efield(
        np.roll(points, turns % 3, axis=1), basis=basis.permute(turns), **kwargs
    )
    expected = np.roll(
        tr.operators.efield(points, basis=basis, **kwargs), turns % 3, axis=-2
    )
    assert_allclose(output @ value, expected, atol=1e-12)
    inverse = tr.operators.permute(-turns, basis=basis.permute(turns), **kwargs)
    assert_allclose(inverse @ value, np.eye(len(basis)), atol=1e-12)


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(x=st.floats(-0.2, 0.2), scale=st.floats(0.1, 10), n=st.integers(1, 2))
def test_permutation_vector_pullback_and_scale_invariant(poltype, x, scale, n):
    vectors = np.array([[0.4 + x + 0.1j, 0.3, 0.8], [-0.2 + 0.1j, 0.6, -0.4]]) * scale
    pols = [0, 1]

    def record(vectors):
        return tr.diff.plane_permutation(vectors, pols, n, poltype=poltype)

    value, context = record(vectors)
    g = np.array([[0.2 + 0.3j, -0.1j], [0.7, 0.1 - 0.2j]])
    check_pullback(
        record,
        vectors,
        directions=(np.array([[0.3 + 0.1j, -0.2j, 0.1], [0.1, 0.4, -0.2j]]) * scale,),
        cotangents=g,
        step=1e-5,
        rtol=1e-7,
        atol=1e-9,
    )
    # The map is homogeneous of degree zero in the wavevectors (Euler).
    assert_allclose(np.vdot(context.pullback(g), vectors), 0, atol=2e-13)
    assert_allclose(
        value,
        tr.diff.plane_permutation(vectors / scale, pols, n, poltype=poltype)[0],
        atol=1e-12,
    )


@pytest.mark.gradients
@pytest.mark.interface
def test_permutation_advect_and_context_validation():
    vectors = np.array([[0.3 + 0.1j, 0.4, 0.8]])
    weights = np.array([[0.2j], [0.8]])

    def objective(x):
        output = ad.plane_permutation(x, polarizations=[1], n=1)
        return anp.real(anp.sum(anp.conj(weights) * output))

    _, context = tr.diff.plane_permutation(vectors, [1])
    expected = context.pullback(weights)
    assert_allclose(advect.grad(objective)(vectors), expected, atol=1e-12)
    with pytest.raises(ValueError, match="polarization"):
        tr.diff.plane_permutation(vectors, [0.1])
    # The native binding reads every memory layout of the same logical inputs
    # and cotangent exactly as the contiguous record does.
    vectors = np.array([[0.3 + 0.1j, 0.4, 0.8], [-0.2 + 0.1j, 0.6, -0.4]])
    polarizations = np.array([1.0, 0.0])
    g = np.array([[0.2 + 0.1j, 0.3], [0.7 - 0.3j, -0.1j]])
    reference, context = _native.plane_permutation(vectors, polarizations, 2, True)
    expected = context.pullback(g)
    for layout in LAYOUTS:
        value, context = _native.plane_permutation(
            arrange(vectors, layout), arrange(polarizations, layout), 2, True
        )
        assert_allclose(value, reference, rtol=0, atol=0)
        assert_allclose(context.pullback(arrange(g, layout)), expected, rtol=0, atol=0)


@pytest.mark.physics
@pytest.mark.reference
@pytest.mark.parametrize("unit", [True, False])
@given(phi=st.floats(-3, 3), psi=st.floats(-3, 3))
def test_plane_rotation_labels_reference_and_composition(unit, phi, psi):
    if unit:
        basis = tr.PlaneWaveBasis.default([[0.2, 0.3, 0.8], [0.1, -0.7, -0.5]])
        reference = treams.PlaneWaveBasisByUnitVector(basis.modes)
    else:
        basis = tr.PlaneWavePorts.default([[0.2, 0.3], [0.4, 0.1]])
        reference = treams.PlaneWaveBasisByComp(basis.modes)
    expected = treams.rotate(phi, psi=psi, basis=reference)
    assert_allclose(
        tr.operators.rotate(phi, psi=psi, basis=basis), expected, atol=1e-12
    )
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
    assert_allclose(
        tr.operators.rotate(phi, basis=basis, where=mask), mask.astype(complex)
    )
    with pytest.raises(ValueError, match="theta"):
        tr.operators.rotate(phi, theta=0.1, basis=basis)


@pytest.mark.physics
@pytest.mark.interface
def test_plane_permutation_identity_axis_and_invalid_geometry():
    for direction in [[0, 0, 1], [1, 0, 0], [0, -1, 0]]:
        basis = tr.PlaneWaveBasis.default([direction])
        for n in (0, 1, 2):
            transform = tr.operators.permute(n, basis=basis)
            inverse = tr.operators.permute(-n, basis=basis.permute(n))
            assert_allclose(inverse @ transform, np.eye(2), atol=1e-12)
    with pytest.raises(ValueError, match="integer"):
        basis.permute(0.5)
    with pytest.raises(ValueError, match="xy"):
        tr.PlaneWavePorts.default([[0.2, 0.3]], "yz").rotate(0.3)


@pytest.mark.gradients
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
    # The per-row check does not depend on the scale: the scaled gradient is
    # tied to it through original.pullback(g) above, so run it once.
    if scale == 1.0:
        # Adjacent equal directions share arithmetic, but are independent AD
        # inputs: probe the first and the second row of the pair separately.
        for i in (0, 1):
            direction = np.zeros_like(vectors)
            direction[i] = [0.1j, 0.2, -0.1j]
            check_pullback(
                lambda v: tr.diff.plane_permutation(v, pols, poltype=poltype),
                vectors,
                directions=(direction,),
                cotangents=g,
                step=1e-5,
                rtol=1e-7,
                atol=1e-9,
            )


@pytest.mark.physics
@pytest.mark.interface
@given(
    quarter=st.integers(-8, 8),
    components=st.lists(
        st.tuples(st.floats(-2, 2), st.floats(-2, 2)), min_size=1, max_size=3
    ),
)
def test_plane_quarter_turns_are_exact_label_maps(quarter, components):
    # Quarter turns map (x, y) -> (-y, x) exactly, like the lattice metadata, so
    # rotated ports equal directly constructed ones and cascade with them.
    def exact(x, y):
        for _ in range(quarter % 4):
            x, y = -y, x
        return x, y

    phi = quarter * np.pi / 2
    ports = tr.PlaneWavePorts.default(components)
    expected = tr.PlaneWavePorts(
        [(*exact(x, y), pol) for x, y, pol in ports.modes], ports.alignment
    )
    assert ports.rotate(phi) == expected
    unit = tr.PlaneWaveBasis.default([[x, y, 0.8] for x, y in components])
    assert unit.rotate(phi) == tr.PlaneWaveBasis(
        [(*exact(x, y), z, pol) for x, y, z, pol in unit.modes]
    )
    cell = tr.PlaneWavePorts.diffr_orders([0.1, 0.2], tr.Lattice.square(2.5), 3)
    rotated = cell.rotate(phi)
    assert rotated.lattice == cell.lattice.rotate(phi)
    assert rotated.kpar == cell.kpar.rotate(phi)
    assert rotated == type(cell)(
        [(*exact(x, y), pol) for x, y, pol in cell.modes], cell.alignment
    )


@pytest.mark.workflows
def test_quarter_turned_layer_cascades_with_constructed_ports():
    ports = tr.PlaneWavePorts.default([[0.2, 0.3]])
    layer = tr.slab(basis=ports, k0=1.3, thickness=0.4, material=2.1)
    turned = tr.PlaneWavePorts.default([[-0.3, 0.2]])
    other = tr.slab(basis=turned, k0=1.3, thickness=0.2, material=1.7)
    network = tr.stack([layer.rotate(np.pi / 2), other])
    assert network.basis == turned
    # A slab is isotropic around its normal, so the rotated layer is the layer
    # built directly on the rotated ports.
    direct = tr.slab(basis=turned, k0=1.3, thickness=0.4, material=2.1)
    assert_allclose(network.array, tr.stack([direct, other]).array, atol=1e-14)
