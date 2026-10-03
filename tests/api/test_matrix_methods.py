"""Matrix-level workflows reuse the explicit native operators and metadata."""

import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import block_diag

import treams_rs as tr
from treams_rs import propagation, stack

from _support import complex_normal

pytestmark = pytest.mark.interface


MEDIUM = (1.2, 1.1, 0.05)


def _particle(cylindrical):
    if cylindrical:
        return tr.CylindricalTMatrix.cylinder([0.2], 2, 1.3, 0.2, [3, MEDIUM])
    return tr.TMatrix.sphere(2, 1.3, 0.2, [3, MEDIUM])


@pytest.mark.parametrize("cylindrical", [False, True])
def test_tmatrix_scattered_fields_follow_the_field_operators(cylindrical):
    # Each operator family has its own upstream oracle; the scattered wave must
    # pass its metadata: outgoing modes, k0, the chiral medium and helicity.
    ours = _particle(cylindrical)
    points = np.array([[0.7, 0.3, 0.4], [-0.3, 0.6, 0.1]])
    outgoing = ours.scatter(np.eye(len(ours.basis)))
    metadata = dict(
        basis=ours.basis,
        k0=1.3,
        material=MEDIUM,
        modetype="singular",
        poltype="helicity",
    )
    for field in ("efield", "hfield", "dfield", "bfield", "gfield", "ffield"):
        for args in [(points,)] if field[0] in "ehdb" else [(0, points), (1, points)]:
            assert_allclose(
                getattr(outgoing, field)(*args),
                getattr(tr.operators, field)(*args, **metadata) @ ours.array,
                rtol=1e-14,
                atol=1e-15,
            )


@pytest.mark.parametrize("cylindrical", [False, True])
def test_tmatrix_basis_selection(cylindrical):
    ours = _particle(cylindrical)
    selected = ours.basis[::-2]
    assert_allclose(ours[selected].array, ours.array[::-2, ::-2])
    assert ours[selected].basis == selected
    foreign = type(selected)(list(selected), positions=[[1, 2, 3]])
    assert_array_equal(ours[foreign].basis.positions, ours.basis.positions)
    assert_array_equal(ours[:, 0], ours.array[:, 0])


@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize(
    "r", [[0.1, -0.2, 0.3], [0, 0, 0.4], [-0.25, 0.05, 0]], ids=["xyz", "z", "xy"]
)
def test_tmatrix_translate_applies_the_translation_operators(cylindrical, r):
    ours = _particle(cylindrical)
    metadata = dict(basis=ours.basis, k0=1.3, material=MEDIUM, poltype="helicity")
    moved = ours.translate(r)
    assert type(moved) is type(ours)
    assert moved.basis == ours.basis
    assert (moved.k0, moved.medium, moved.polarization) == (
        ours.k0,
        ours.medium,
        ours.polarization,
    )
    assert_array_equal(
        moved.array,
        tr.operators.translate(r, **metadata)
        @ ours.array
        @ tr.operators.translate(np.negative(r), **metadata),
    )


@settings(max_examples=20)
@given(r=st.lists(st.floats(-0.06, 0.06), min_size=3, max_size=3))
@pytest.mark.parametrize("cylindrical", [False, True])
def test_tmatrix_translate_round_trips(cylindrical, r):
    # Truncating the basis at degree or order 6 limits the round trip to about
    # 1e-13 for displacements up to 0.1.
    ours = (
        tr.CylindricalTMatrix.cylinder([0.2], 6, 1.3, 0.2, [3, MEDIUM])
        if cylindrical
        else tr.TMatrix.sphere(6, 1.3, 0.2, [3, MEDIUM])
    )
    back = ours.translate(r).translate(np.negative(r))
    assert type(back) is type(ours)
    assert_allclose(back.array, ours.array, rtol=0, atol=5e-13)


@pytest.mark.parametrize(
    "cylindrical,lattice,kpar",
    [(False, np.diag([1.6, 1.7]), [0.1, 0.2]), (True, 1.7, 0.1)],
)
def test_tmatrix_expandlattice_applies_the_lattice_expansion(
    cylindrical, lattice, kpar
):
    ours = _particle(cylindrical)
    expected = (
        tr.operators.expandlattice(
            lattice,
            kpar,
            basis=ours.basis,
            k0=1.3,
            material=MEDIUM,
            poltype="helicity",
            modetype=("regular", "singular"),
        )
        @ ours.array
    )
    assert_array_equal(ours.expandlattice(lattice=lattice, kpar=kpar), expected)


def _pair(particle, positions):
    """Two uncoupled copies of a particle at the positions, as one block-diagonal matrix."""
    modes = [(index, *mode[1:]) for index in (0, 1) for mode in particle.basis]
    return type(particle)(
        block_diag(particle.array, particle.array),
        basis=type(particle.basis)(modes, positions),
        k0=particle.k0,
        material=particle.medium,
        poltype=particle.polarization,
    )


@given(scale=st.floats(0.3, 3), offset=st.floats(-2, 2))
@settings(max_examples=25)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_exclusion_masks_scale_and_translation(cylindrical, scale, offset):
    base = (
        tr.CylindricalTMatrix.cylinder([0.2], 1, 1.3, 0.2, [3, 1])
        if cylindrical
        else tr.TMatrix.sphere(1, 1.3, 0.2, [3, 1])
    )
    origins = np.array([[0, 0, 0], [0.8, 0.3, 0.2]])
    points = np.random.default_rng(32).normal(size=(4, 5, 3))
    radii = np.array([0.2, 0.3])
    first = _pair(base, origins)
    second = _pair(base, origins * scale + offset)
    assert_array_equal(
        first.valid_points(points, radii),
        second.valid_points(points * scale + offset, radii * scale),
    )
    assert not first.valid_points(origins, radii).any()
    if cylindrical:
        assert_array_equal(
            first.valid_points(points[..., :2], radii),
            first.valid_points(points, radii),
        )


def _smatrix(alignment, poltype):
    directions = [[0.2, 0.3], [0.3, 0.1]]
    basis = tr.PlaneWavePorts.default(directions, alignment)[::-1]
    values = complex_normal(np.random.default_rng(3), (2, 2, 4, 4))
    materials = (
        ((1.7, 1.1, 0.03), (1.2, 1.3, 0.02)) if poltype == "helicity" else (1.7, 1.2)
    )
    return tr.SMatrix(values, k0=1.3, basis=basis, material=materials, poltype=poltype)


@pytest.mark.reference
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("turns", [-1, 1, 2])
def test_smatrix_sparse_coordinate_transforms(alignment, poltype, turns):
    ours = _smatrix(alignment, poltype)
    # Upstream slicing loses non-xy alignment; reconstruct the intended basis.
    oracle = treams.SMatrices(
        ours.array,
        k0=1.3,
        basis=treams.PlaneWaveBasisByComp(list(ours.basis), alignment=alignment),
        material=tuple(treams.Material(m()) for m in ours.material),
        poltype=poltype,
    )
    shift = [0.3, 0.1, -0.2]
    assert_allclose(
        ours.translate(shift).array, np.asarray(oracle.translate(shift)), atol=2e-13
    )
    changed = ours.permute(turns)
    assert_allclose(changed.array, np.asarray(oracle.permute(turns)), atol=2e-13)
    assert changed.basis.alignment == ours.basis.permute(turns).alignment
    if alignment == "xy":
        assert_allclose(
            ours.rotate(0.3).rotate(-0.3).basis.components,
            ours.basis.components,
            atol=2e-14,
        )
        assert_array_equal(ours.rotate(0.3).array, ours.array)


@pytest.mark.physics
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(
    first=st.integers(-3, 3),
    second=st.integers(-3, 3),
    a=st.tuples(*(st.floats(-0.4, 0.4) for _ in range(3))),
    b=st.tuples(*(st.floats(-0.4, 0.4) for _ in range(3))),
    distance=st.floats(-0.5, 0.5),
)
def test_smatrix_translations_and_permutations_compose(
    alignment, poltype, first, second, a, b, distance
):
    ours = _smatrix(alignment, poltype)
    composed = ours.permute(first).permute(second)
    direct = ours.permute(first + second)
    assert_allclose(composed.array, direct.array, rtol=0, atol=2e-14)
    assert composed.basis.alignment == direct.basis.alignment
    assert_allclose(ours.permute(3).array, ours.array, rtol=0, atol=1e-14)
    assert ours.permute(3).basis == ours.basis
    assert_allclose(
        ours.translate(a).translate(b).array,
        ours.translate(np.add(a, b)).array,
        rtol=0,
        atol=2e-14,
    )
    # A shift along the normal equals propagating through the outer media on
    # both sides: exterior (material[1]) below, interior (material[0]) above.
    normal = np.eye(3)[{"xy": 2, "yz": 0, "zx": 1}[alignment]]
    options = dict(basis=ours.basis, k0=1.3, polarization=poltype)
    cascade = stack(
        [
            propagation(distance=-distance, medium=ours.material[1], **options),
            ours,
            propagation(distance=distance, medium=ours.material[0], **options),
        ]
    )
    assert_allclose(
        ours.translate(normal * distance).array, cascade.array, rtol=0, atol=2e-14
    )


@pytest.mark.reference
def test_material_radial_branch():
    medium = tr.Material(2.3 + 0.1j, 1.2, 0.03)
    assert tr.Material(np.array(medium())) == medium
    # A 0-d array is a scalar permittivity, like a Python or NumPy scalar.
    assert tr.Material(np.array(2.0)) == tr.Material(np.float64(2.0))
    assert tr.Material(np.array(2.0)) == tr.Material(2.0)
    kz = np.array([0.2, 0.4, 2.3])[:, None]
    radial = medium.krhos(1.3, kz)
    assert_allclose(radial, treams.Material(*medium()).krhos(1.3, kz, [0, 1]))
    assert_allclose(
        radial**2 + kz**2, np.broadcast_to(medium.ks(1.3) ** 2, radial.shape)
    )
    assert np.all(radial.imag >= 0)


@pytest.mark.parametrize("cylindrical", [False, True])
def test_derived_responses_own_read_only_arrays(cylindrical):
    # Internal producers adopt fresh arrays without a copy; every result must
    # still own read-only storage that shares no memory with its inputs.
    particle = (
        tr.CylindricalTMatrix.cylinder([0.2], 1, 1.3, 0.2, [3, 1], "parity")
        if cylindrical
        else tr.TMatrix.sphere(1, 1.3, 0.2, [3, 1], "parity")
    )
    positions = [[0, 0, 0], [0.8, 0, 0]]
    cluster = _pair(particle, positions)
    solved = tr.Cluster([particle, particle], positions=positions).solve()
    derived = [
        particle,
        particle.changepoltype(),
        particle.rotate(0.3),
        particle.expand(particle.basis),
        particle[particle.basis[::2]],
        cluster,
        solved,
        solved.changepoltype(),
    ]
    for response in derived:
        assert not response.array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            response.array[0, 0] = 1
        for other in derived:
            if other is not response:
                assert not np.shares_memory(response.array, other.array)
    coupling = tr.diff.expansion(
        cluster.basis, cluster.basis, cluster.ks, poltype="parity", singular=True
    )[0]
    assert_allclose(
        solved.array,
        tr.diff.interaction(cluster.array, coupling)[0],
        rtol=1e-12,
        atol=1e-14,
    )


_PORTS = tr.PlaneWavePorts.default([0.1, 0.2])
_SPHERICAL = tr.SphericalBasis.default(1)
_CHIRAL = tr.Material(2, 1, 0.1)


@pytest.mark.parametrize(
    "build",
    [
        lambda k0, m, p: tr.TMatrix(np.eye(6), k0=k0, material=m, poltype=p),
        lambda k0, m, p: tr.SMatrix(
            np.zeros((2, 2, 2, 2)), k0=k0, basis=_PORTS, material=m, poltype=p
        ),
        lambda k0, m, p: tr.SMatrix.interface(_PORTS, k0, [1, m], p),
        lambda k0, m, p: tr.SMatrix.slab(0.2, _PORTS, k0, [1, m, 1], p),
        lambda k0, m, p: tr.SMatrix.propagation(0.2, _PORTS, k0, m, p),
        lambda k0, m, p: tr.Wave(
            np.ones(6), basis=_SPHERICAL, k0=k0, material=m, poltype=p
        ),
        lambda k0, m, p: tr.plane_wave([0, 0, 1], 0, k0=k0, material=m, poltype=p),
        lambda k0, m, p: tr.operators.PhysicsArray(
            np.eye(6), basis=_SPHERICAL, k0=k0, material=m, poltype=p
        ),
        lambda k0, m, p: tr.operators.expand(_SPHERICAL, k0=k0, material=m, poltype=p),
        lambda k0, m, p: tr.operators.translate(
            [0, 0, 1], basis=_SPHERICAL, k0=k0, material=m, poltype=p
        ),
        lambda k0, m, p: tr.operators.efield(
            [0, 0, 1], basis=_SPHERICAL, k0=k0, material=m, poltype=p
        ),
    ],
)
def test_frequency_and_parity_medium_validation_is_shared(build):
    build(1.3, _CHIRAL, "helicity")
    build(1.3, 2, "parity")
    for k0 in (0, -1.3, np.inf, np.nan):
        with pytest.raises(ValueError, match="k0 must be finite and positive"):
            build(k0, 2, "helicity")
    # Values without a float64 representation are not wavenumbers.
    for k0 in (10**400, -(10**400), "1.3"):
        with pytest.raises(TypeError):
            build(k0, 2, "helicity")
    with pytest.raises(ValueError, match="parity polarization requires an achiral"):
        build(1.3, _CHIRAL, "parity")
