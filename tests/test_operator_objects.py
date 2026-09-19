import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr


@pytest.mark.parametrize(
    "name,args",
    [
        ("Rotate", (0.3, 0.4, -0.2)),
        ("Translate", ([0.1, -0.2, 0.3],)),
        ("ChangePoltype", ("parity",)),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.filterwarnings("ignore:.*where.*used without.*out.*:UserWarning")
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
    "name", ["EField", "HField", "DField", "BField", "GField", "FField"]
)
@pytest.mark.parametrize("family", ["plane", "sphere", "cylinder"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.filterwarnings("ignore:.*scipy.special.sph_harm.*:DeprecationWarning")
def test_all_field_objects_weighted_samples_and_reference(name, family, poltype):
    kwargs = dict(
        k0=1.3,
        material=(2.3 + 0.2j, 1.2, 0.03 if poltype == "helicity" else 0),
        poltype=poltype,
    )
    constructor, args = {
        "plane": ("plane_wave", ([0.2, 0.3, 0.9], [0.2 + 0.1j, 0.8])),
        "sphere": ("spherical_wave", (2, -1, 0)),
        "cylinder": ("cylindrical_wave", (0.2, -1, 0)),
    }[family]
    wave = getattr(tr, constructor)(*args, **kwargs)
    oracle = getattr(treams, constructor)(
        *args, **(kwargs | {"material": treams.Material(kwargs["material"])})
    )
    points = np.array([[[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]]])
    helicities = [0, 1] if name in ("GField", "FField") else [None]
    for pol in helicities:
        args = (points,) if pol is None else (pol, points)
        op = getattr(tr.operators, name)(*args)
        method = name.lower()
        expected = (
            getattr(oracle, method)(points)
            if pol is None
            else getattr(oracle, method)(pol, r=points)
        )
        assert_allclose(op @ wave, expected, atol=2e-12, rtol=2e-12)
        assert_allclose(op(**op.get_kwargs(wave)) @ wave.array, op @ wave, atol=1e-13)
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
