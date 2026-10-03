"""Upstream-style operator objects and PhysicsArray attributes (treams_rs.operators):
upstream references, inverses, metadata, and fields that satisfy Maxwell's equations."""

import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr

from _support import complex_normal

pytestmark = pytest.mark.interface


@pytest.mark.reference
@pytest.mark.parametrize(
    "name,args",
    [
        ("Rotate", (0.3, 0.4, -0.2)),
        ("Translate", ([0.1, -0.2, 0.3],)),
        ("ChangePoltype", ("parity",)),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_operator_reference_and_owned_parameters(name, args, inverse):
    args = tuple(np.array(x, copy=True) if isinstance(x, list) else x for x in args)
    basis = tr.SphericalBasis.default(2)
    oracle = treams.SphericalWaveBasis.default(2)
    op, ref = getattr(tr.operators, name)(*args), getattr(treams, name)(*args)
    kwargs = (
        {} if name in ("Rotate", "ChangePoltype") else dict(k0=1.3, material=(2.1, 1.2))
    )
    if inverse:
        op, ref = op.inv, ref.inv
    assert_allclose(op(basis=basis, **kwargs), ref(basis=oracle, **kwargs), atol=1e-13)
    assert op.inv.inv.isinv == op.isinv
    if name == "Translate":
        expected = op(basis=basis, **kwargs)
        args[0][:] = [9, 8, 7]
        assert_allclose(op(basis=basis, **kwargs), expected, atol=0)


@settings(max_examples=20)
@given(phi=st.floats(-3, 3), theta=st.floats(-2, 2), psi=st.floats(-3, 3))
def test_operator_rotation_inverse_and_metadata_matmul(phi, theta, psi):
    tm = tr.TMatrix.sphere(2, 1.3, 0.4, [2.3, 1])
    rotation = tr.operators.Rotate(phi, theta, psi)
    matrix = rotation(basis=tm.basis)
    inverse = rotation.inv(basis=tm.basis)
    assert_allclose(matrix @ inverse, np.eye(len(tm)), atol=3e-14)
    assert_allclose(rotation @ tm, matrix @ tm.array, atol=1e-14)
    assert_allclose(tm @ rotation, tm.array @ matrix, atol=1e-14)
    assert rotation.get_kwargs(tm) == {"basis": tm.basis}


def test_expand_object_source_metadata_rectangular_and_explicit_pair():
    source = tr.SphericalBasis.default(2, positions=[[0.3, 0.2, 0.1]])
    destination = tr.SphericalBasis.default(1)
    wave = tr.spherical_wave(2, 1, 0, basis=source, k0=1.2)
    op = tr.operators.Expand(destination)
    expected = tr.operators.expand((destination, source), k0=1.2)
    assert_allclose(op @ wave, expected @ wave.array, atol=1e-13)
    assert_allclose(
        op.inv(basis=source, k0=1.2),
        tr.operators.expand((source, destination), k0=1.2),
        atol=1e-13,
    )
    assert_allclose(
        tr.operators.Expand((destination, source))(basis=source, k0=1.2),
        expected,
        atol=0,
    )
    plane = tr.plane_wave([0.3, -0.2, 1], [0.2 + 0.1j, 0.7], k0=1.2)
    assert_allclose(op @ plane, plane.expand(destination), atol=1e-13)


@pytest.mark.parametrize(
    "family,modetype",
    [
        ("plane", None),
        ("sphere", "regular"),
        ("sphere", "singular"),
        ("cylinder", "regular"),
        ("cylinder", "singular"),
    ],
)
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_all_field_objects_apply_the_field_operators(family, modetype, poltype):
    # Each operator family has its own upstream oracle; the objects must pass
    # the wave's basis, k0, medium, polarization and mode type.
    material = tr.Material(2.3 + 0.2j, 1.2, 0.03 if poltype == "helicity" else 0)
    constructor, args = {
        "plane": ("plane_wave", ([0.2, 0.3, 0.9], [0.2 + 0.1j, 0.8])),
        "sphere": ("spherical_wave", (2, -1, 1)),
        "cylinder": ("cylindrical_wave", (0.2, -1, 1)),
    }[family]
    options = {} if modetype is None else {"modetype": modetype}
    wave = getattr(tr, constructor)(
        *args, k0=1.3, material=material, poltype=poltype, **options
    )
    metadata = dict(
        basis=wave.basis,
        k0=1.3,
        material=material,
        modetype=modetype,
        poltype=poltype,
    )
    points = np.array([[[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]]])
    for name in ["EField", "HField", "DField", "BField", "GField", "FField"]:
        for args in [(points,)] if name[0] in "EHDB" else [(0, points), (1, points)]:
            op = getattr(tr.operators, name)(*args)
            expected = getattr(tr.operators, name.lower())(*args, **metadata)
            assert_allclose(op @ wave, expected @ wave.array, rtol=1e-14, atol=1e-15)
            assert_allclose(
                op(**op.get_kwargs(wave)) @ wave.array, op @ wave, atol=1e-13
            )
            with pytest.raises(NotImplementedError):
                _ = op.inv


def test_polarization_and_permutation_operator_inverses():
    basis = tr.PlaneWaveBasis.default([[0.2, 0.3, 0.9], [0.4, 0.5, -0.9]])
    op = tr.operators.Permute()
    assert_allclose(
        op.inv(basis=basis) @ op(basis=basis), np.eye(len(basis)), atol=1e-13
    )
    tm = tr.TMatrix.sphere(1, 1.3, 0.4, [2.3, 1])
    change = tr.operators.ChangePoltype()
    assert change.get_kwargs(tm)["poltype"] == "parity"
    assert_allclose(
        change @ tm,
        tr.operators.changepoltype("parity", basis=tm.basis) @ tm.array,
        atol=0,
    )


def test_periodic_operator_inferred_and_explicit_metadata():
    basis = tr.SphericalBasis.default(1)
    basis.lattice, basis.kpar = tr.Lattice(1.7), tr.WaveVector(0.13)
    inferred = tr.operators.ExpandLattice()
    explicit = tr.operators.ExpandLattice(basis.lattice, basis.kpar, basis, eta=0.7)
    wave = tr.spherical_wave(1, 0, 1, basis=basis, k0=1.3, modetype="singular")
    assert_allclose(
        inferred @ wave,
        tr.operators.expandlattice(basis=basis, k0=1.3) @ wave.array,
        atol=1e-11,
    )
    assert_allclose(
        explicit(**explicit.get_kwargs(wave), lattice=basis.lattice, kpar=basis.kpar),
        tr.operators.expandlattice(basis=basis, k0=1.3, eta=0.7),
        atol=1e-11,
    )


def test_physics_array_bound_attribute_ownership_and_keyword_arguments():
    first_basis = tr.SphericalBasis.default(1)
    second_basis = tr.SphericalBasis.default(2)
    first = tr.operators.PhysicsArray(np.eye(6), basis=first_basis, k0=1.3)
    second = tr.operators.PhysicsArray(np.eye(16), basis=second_basis, k0=1.3)
    saved = first.rotate
    _ = second.rotate
    assert_allclose(
        saved.eval(0.3, 0.2, 0.1),
        tr.operators.rotate(0.3, 0.2, 0.1, basis=first_basis),
        atol=0,
    )
    assert_allclose(saved(0.3, 0.2, 0.1), np.eye(6), atol=1e-14)
    assert_allclose(
        first.expand.eval(basis=second_basis),
        tr.operators.expand((second_basis, first_basis), k0=1.3),
        atol=0,
    )
    assert_allclose(
        first.expand.eval_inv(second_basis),
        tr.operators.expand((first_basis, second_basis), k0=1.3),
        atol=0,
    )
    assert_allclose(
        first @ tr.operators.Rotate(0.2),
        first.array @ tr.operators.rotate(0.2, basis=first_basis),
        atol=0,
    )
    assert_allclose(
        tr.operators.Rotate(0.2) @ first,
        tr.operators.rotate(0.2, basis=first_basis) @ first.array,
        atol=0,
    )
    assert isinstance(first * 2, np.ndarray)
    assert not first.array.flags.writeable
    with pytest.raises(ValueError, match="dimension"):
        tr.operators.PhysicsArray([1, 2], basis=first_basis)


@pytest.mark.parametrize(
    "name", ["efield", "hfield", "dfield", "bfield", "gfield", "ffield"]
)
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_physics_array_weighted_field_attributes(name, poltype):
    wave = tr.spherical_wave(2, -1, 0, k0=1.3, poltype=poltype)
    obj = tr.operators.PhysicsArray(
        wave.array, basis=wave.basis, k0=wave.k0, poltype=poltype
    )
    points = [[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]]
    args = (1,) if name in ("gfield", "ffield") else ()
    assert_allclose(
        getattr(obj, name)(*args, r=points),
        getattr(wave, name)(*args, r=points),
        atol=1e-13,
    )
    assert_allclose(
        getattr(obj, name).eval(*args, r=points) @ obj.array,
        getattr(wave, name)(*args, r=points),
        atol=1e-13,
    )
    with pytest.raises(NotImplementedError):
        getattr(obj, name).apply_right(*args, r=points)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_smatrix_block_views_port_metadata_and_fields(poltype):
    basis = tr.PlaneWavePorts.default([[0.2, 0.1]])
    stack = tr.SMatrix.interface(basis, 1.3, [1.2, 2.1], poltype)
    for i in range(2):
        for j in range(2):
            block = stack.block(i, j)
            assert isinstance(block, tr.ScatteringBlock)
            assert np.shares_memory(block.array, stack.array)
            assert not block.array.flags.writeable
            assert block.material == (stack.material[i], stack.material[1 - j])
            assert block.modetype == (("up", "down")[i], ("up", "down")[j])
            points = [[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]]
            expected = (
                tr.operators.efield(
                    points,
                    basis=basis,
                    k0=1.3,
                    material=block.material[0],
                    modetype=block.modetype[0],
                    poltype=poltype,
                )
                @ stack.array[i, j]
            )
            assert_allclose(block.efield(points), expected, atol=0)
            standalone = tr.ScatteringBlock(
                block.array,
                basis=basis,
                k0=1.3,
                material=block.material,
                poltype=poltype,
                modetype=block.modetype,
            )
            assert_allclose(standalone.efield(points), expected, atol=0)


# Maxwell's equations for every wave family ----------------------------------------

FAMILIES = {
    "spherical": (tr.SphericalBasis.default(2), ("regular", "singular")),
    "cylindrical": (
        tr.CylindricalBasis.default([0.2, -0.3], 2),
        ("regular", "singular"),
    ),
    "unit": (tr.PlaneWaveBasis.default([[0.2, 0.3, 0.9], [0.4, -0.5, 0.1]]), (None,)),
    # The second component exceeds some wavenumbers: evanescent modes included.
    **{
        alignment: (
            tr.PlaneWavePorts.default([[0.1, 0.2], [1.9, -0.2]], alignment),
            ("up", "down"),
        )
        for alignment in ("xy", "yz", "zx")
    },
}


@st.composite
def waves(draw):
    """A random superposition of one family in a passive, possibly chiral medium.

    Returns the wave, its medium, k0 and a point away from the spherical origin
    and the cylindrical axis. Unit-vector plane waves ignore the mode type.
    """
    basis, modetypes = FAMILIES[draw(st.sampled_from(sorted(FAMILIES)))]
    poltype = draw(st.sampled_from(["helicity", "parity"]))
    epsilon, mu = (draw(st.floats(1, 4)) + 1j * draw(st.floats(0, 0.2)) for _ in "em")
    kappa = draw(st.floats(-0.1, 0.1)) if poltype == "helicity" else 0
    medium = tr.Material(epsilon, mu, kappa)
    k0 = draw(st.floats(0.8, 1.6))
    rng = np.random.default_rng(draw(st.integers(0, 2**16)))
    modetype = draw(st.sampled_from(modetypes))
    # Wave objects weight the amplitudes; PhysicsArray applies field operators.
    wave = draw(st.sampled_from([tr.Wave, tr.operators.PhysicsArray]))(
        complex_normal(rng, len(basis)),
        basis=basis,
        k0=k0,
        material=medium,
        poltype=poltype,
        modetype=modetype or "up",
    )
    point = rng.normal(size=3)
    point[:2] *= draw(st.floats(0.6, 1.5)) / np.linalg.norm(point[:2])
    return wave, medium, k0, point


@pytest.mark.physics
@given(waves())
def test_fields_satisfy_the_constitutive_relations_and_superpose(sample):
    wave, medium, k0, point = sample
    e, h, d, b = (getattr(wave, f"{name}field")(r=point) for name in "ehdb")
    epsilon, mu, kappa = medium()
    scale = max(abs(e).max(), abs(h).max())
    assert_allclose(d, epsilon * e + 1j * kappa * h, rtol=0, atol=1e-13 * scale)
    assert_allclose(b, mu * h - 1j * kappa * e, rtol=0, atol=1e-13 * scale)
    # The samples of a superposition are the operator times its coefficients.
    for name, value in zip("ehdb", (e, h, d, b), strict=True):
        operator = getattr(tr.operators, f"{name}field")(
            point,
            basis=wave.basis,
            k0=k0,
            material=medium,
            modetype=wave.modetype,
            poltype=wave.poltype,
        )
        assert_allclose(value, operator @ wave.array, rtol=0, atol=1e-13 * scale)


def _curl_and_divergence(field, point, step=5e-4):
    """Fourth-order central differences of ``field(r=...)`` at ``point``."""
    offsets = np.array([-2, -1, 1, 2])
    weights = np.array([1, -8, 8, -1]) / (12 * step)
    # samples[s, j, i]: component i at point + offsets[s] * step * e_j.
    samples = field(r=point + step * offsets[:, None, None] * np.eye(3))
    jacobian = np.einsum("s,sji->ij", weights, samples)
    curl = jacobian[[2, 0, 1], [1, 2, 0]] - jacobian[[1, 2, 0], [2, 0, 1]]
    return curl, np.trace(jacobian)


@pytest.mark.physics
@given(waves())
def test_fields_satisfy_maxwells_equations(sample):
    # exp(-i omega t): curl E = i k0 B and curl H = -i k0 D in the package's
    # units, with divergence-free D and B; the Riemann-Silberstein fields of
    # helicity s = +1 and -1 are eigenfields, curl G = s k_s G.
    wave, medium, k0, point = sample
    fields = {name: getattr(wave, f"{name}field") for name in "ehdb"}
    values = {name: field(r=point) for name, field in fields.items()}
    derivatives = {
        name: _curl_and_divergence(field, point) for name, field in fields.items()
    }
    tolerance = 1e-9 * k0 * max(abs(value).max() for value in values.values())
    assert_allclose(derivatives["e"][0], 1j * k0 * values["b"], atol=tolerance)
    assert_allclose(derivatives["h"][0], -1j * k0 * values["d"], atol=tolerance)
    assert_allclose(derivatives["d"][1], 0, atol=tolerance)
    assert_allclose(derivatives["b"][1], 0, atol=tolerance)
    ks = medium.ks(k0)
    for pol in (0, 1):
        curl, _ = _curl_and_divergence(lambda r, pol=pol: wave.gfield(pol, r=r), point)
        assert_allclose(
            curl, (2 * pol - 1) * ks[pol] * wave.gfield(pol, r=point), atol=tolerance
        )
