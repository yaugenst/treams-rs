"""HDF5 and mesh interchange (treams_rs.io): unit conversion, round trips, compatibility
with the upstream reader and writer, and input validation before writing."""

from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import treams
import treams.io
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

import treams_rs as tr
from treams_rs import io

from _support import complex_normal

pytestmark = pytest.mark.interface


def _assert_same(actual, expected, scale=1):
    assert_allclose(actual.array, expected.array, rtol=0, atol=0)
    assert actual.basis.modes == expected.basis.modes
    # Unit conversions cost a few rounding errors.
    assert_allclose(
        actual.basis.positions, expected.basis.positions * scale, rtol=2e-15, atol=0
    )
    assert_allclose(actual.k0, expected.k0 / scale, rtol=2e-15, atol=0)
    assert actual.material == expected.material
    assert actual.poltype == expected.poltype


def _memory(name="memory.h5"):
    return h5py.File(name, "w", driver="core", backing_store=False)


@st.composite
def sweeps(draw):
    """Sweeps of dense, non-symmetric T-matrices sharing a multi-origin basis."""
    particles = draw(st.integers(1, 3))
    rng = np.random.default_rng(draw(st.integers(0, 2**16)))
    positions = rng.uniform(-2, 2, (particles, 3))
    basis = tr.SphericalBasis.default(draw(st.integers(1, 2)), particles, positions)
    poltype = draw(st.sampled_from(["helicity", "parity"]))
    shape = draw(st.sampled_from([(), (2,), (2, 1), (1, 3)]))
    matrices = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        epsilon, mu = rng.uniform(1, 4, 2) + 1j * rng.uniform(0, 0.3, 2)
        kappa = rng.uniform(-0.15, 0.15) if poltype == "helicity" else 0
        matrices[index] = tr.TMatrix(
            complex_normal(rng, (len(basis), len(basis))),
            basis=basis,
            k0=rng.uniform(0.5, 3),
            material=(epsilon, mu, kappa),
            poltype=poltype,
        )
    return matrices[()] if shape == () else matrices


# Plain str units: h5py cannot store NumPy string scalars as attributes.
length_units = st.sampled_from(sorted(io.LENGTHS))


@given(sweeps(), length_units, length_units)
def test_hdf5_roundtrip_converts_units_and_keeps_every_entry(sweep, saved, loaded):
    scale = io.LENGTHS[saved] / io.LENGTHS[loaded]
    expected = np.asarray(sweep) if isinstance(sweep, np.ndarray) else sweep
    uuid = bytes(range(16))
    with _memory() as handle:
        io.save_hdf5(handle, sweep, name="sweep", lunit=saved, uuid=uuid)
        result = io.load_hdf5(handle, lunit=loaded)
        if isinstance(sweep, tr.TMatrix):
            assert isinstance(result, tr.TMatrix)
            pairs = [(result, sweep)]
        else:
            assert result.shape == expected.shape
            pairs = [(result[i], expected[i]) for i in np.ndindex(expected.shape)]
        for actual, reference in pairs:
            _assert_same(actual, reference, scale)
            # The physical phase k0 * r does not depend on the unit.
            assert_allclose(
                actual.k0 * actual.basis.positions,
                reference.k0 * reference.basis.positions,
                rtol=2e-15,
                atol=0,
            )
        assert handle.attrs["name"] == "sweep"
        first = pairs[0][1]
        if np.any(first.basis.pidx != 0):
            assert handle["modes/pidx"].id == handle["modes/position_index"].id
        if first.poltype == "helicity":
            chirality = handle["embedding/chirality"]
            assert chirality.id == handle["embedding/chirality_parameter"].id
        # Saving what was loaded, in the same unit, reproduces the file.
        again = io.load_hdf5(handle, lunit=saved)
        with _memory("again.h5") as copy:
            io.save_hdf5(copy, again, name="sweep", lunit=saved, uuid=uuid)
            _assert_same_files(handle, copy)


def _assert_same_files(first, second):
    assert set(first) == set(second)
    for key in first:
        a, b = first[key], second[key]
        if isinstance(a, h5py.Group):
            _assert_same_files(a, b)
            continue
        assert dict(a.attrs) == dict(b.attrs), key
        if a.dtype.kind in "fc":
            assert_allclose(a[()], b[()], rtol=2e-15, atol=0, err_msg=key)
        else:
            assert np.array_equal(a[()], b[()]), key


@given(
    data=st.data(),
    k0=st.floats(0.01, 100),
    lunit=length_units,
)
def test_all_frequency_representations_agree(data, k0, lunit):
    # One physical frequency written in every representation and unit table.
    tm = tr.TMatrix.sphere(1, k0, 0.2 / k0, [3, 1])
    wavenumber = k0 / io.LENGTHS[lunit]  # angular vacuum wavenumber in 1/m
    c = 299792458.0
    quantities = {
        "frequency": (io.FREQUENCIES, c * wavenumber / (2 * np.pi)),
        "angular_frequency": (io.FREQUENCIES, c * wavenumber),
        "vacuum_wavelength": (io.LENGTHS, 2 * np.pi / wavenumber),
        "vacuum_wavenumber": (io.INVLENGTHS, wavenumber / (2 * np.pi)),
        "angular_vacuum_wavenumber": (io.INVLENGTHS, wavenumber),
    }
    for kind, (units, value) in quantities.items():
        unit = data.draw(st.sampled_from(sorted(units)), label=kind)
        with _memory() as handle:
            io.save_hdf5(handle, tm, lunit=lunit)
            del handle["angular_vacuum_wavenumber"]
            handle[kind] = value / units[unit]
            handle[kind].attrs["unit"] = unit
            assert_allclose(io.load_hdf5(handle, lunit=lunit).k0, k0, rtol=1e-14)


def test_hertz_and_inverse_seconds_are_one_table():
    assert io.FREQUENCIES["ks^{-1}"] == io.FREQUENCIES["mHz"]
    assert io.FREQUENCIES["s^{-1}"] == io.FREQUENCIES["Hz"]


def test_hdf5_single_matrix_and_path(tmp_path):
    tm = tr.TMatrix.sphere(2, 1.3, 0.2, [(3, 1), (1.3, 1.1, 0.08)])
    filename = tmp_path / "single.h5"
    with h5py.File(filename, "w") as handle:
        io.save_hdf5(handle, tm)
    _assert_same(io.load_hdf5(filename), tm)


@pytest.mark.reference
def test_legacy_upstream_writer_and_reader_compatibility():
    basis = tr.SphericalBasis.default(1, 2, [[0, 0, 0], [0.7, 0.2, 0.1]])
    tm = tr.TMatrix(
        np.eye(len(basis)) * (0.1 + 0.2j),
        basis=basis,
        k0=1.3,
        material=(1.3, 1.1, 0.08),
    )
    reference = treams.TMatrix(
        tm.array,
        basis=treams.SphericalWaveBasis(basis.modes, basis.positions),
        k0=tm.k0,
        material=treams.Material(tm.material()),
    )
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        treams.io.save_hdf5(handle, [reference])
        loaded = io.load_hdf5(handle)[0]
        _assert_same(loaded, tm)
    # Upstream's loader loses origins in its basis union and cannot read local
    # multi-origin matrices even when the file carries the correct indices.
    tm = tr.TMatrix.sphere(1, 1.3, 0.2, [3, (1.3, 1.1, 0.08)])
    basis = tm.basis
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        io.save_hdf5(handle, [tm])
        loaded = treams.io.load_hdf5(handle)[0]
        assert_allclose(loaded, tm.array, atol=0)
        assert_allclose(loaded.basis.positions, basis.positions)
        assert_allclose(loaded.basis.pidx, basis.pidx)
        assert loaded.material.kappa == 0.08


@given(
    seed=st.integers(0, 2**16),
    incident=st.lists(st.integers(0, 15), min_size=1, max_size=16, unique=True),
    scattered=st.lists(st.integers(0, 15), min_size=1, max_size=16, unique=True),
)
def test_rectangular_modes_embed_into_the_union_basis(seed, incident, scattered):
    tm = tr.TMatrix.sphere(2, 1.3, 0.2, [3, 1])
    data = complex_normal(np.random.default_rng(seed), (len(scattered), len(incident)))
    with _memory() as handle:
        io.save_hdf5(handle, tm)
        del handle["tmatrix"]
        handle["tmatrix"] = data
        for name in ["l", "m", "polarization"]:
            values = handle[f"modes/{name}"][()]
            del handle[f"modes/{name}"]
            handle[f"modes/{name}_incident"] = values[incident]
            handle[f"modes/{name}_scattered"] = values[scattered]
        del handle["embedding/relative_permittivity"]
        del handle["embedding/relative_permeability"]
        handle["embedding/refractive_index"] = 2
        handle["embedding/relative_impedance"] = 0.5
        loaded = io.load_hdf5(handle)
    assert loaded.material == tr.Material(4, 1)
    modes = [tm.basis.modes[i] for i in (*incident, *scattered)]
    assert loaded.basis.modes == tuple(dict.fromkeys(modes))
    incoming = [loaded.basis.modes.index(tm.basis.modes[i]) for i in incident]
    outgoing = [loaded.basis.modes.index(tm.basis.modes[i]) for i in scattered]
    expected = np.zeros_like(loaded.array)
    expected[np.ix_(outgoing, incoming)] = data
    # Coefficients missing from the file are zero.
    assert_array_equal(loaded.array, expected)


_SPHERE = tr.TMatrix.sphere(1, 1.3, 0.2, [3, 1])


@pytest.mark.parametrize(
    "tms,kwargs,error,match",
    [
        (
            [_SPHERE, tr.TMatrix.sphere(2, 1.3, 0.2, [3, 1])],
            {},
            ValueError,
            "share mode",
        ),
        ([[_SPHERE], [_SPHERE, _SPHERE]], {}, ValueError, "must be rectangular"),
        ([], {}, ValueError, "nonempty rectangular"),
        (_SPHERE, {"lunit": "inch"}, ValueError, "unrecognized length unit"),
        (
            _SPHERE,
            {"scatterers": [{}, {"geometry": {"unit": "inch", "radius": 0.2}}]},
            ValueError,
            "unrecognized geometry length unit: inch",
        ),
        (
            _SPHERE,
            {"computation": {"mesh": "/nonexistent/scene.msh"}},
            FileNotFoundError,
            "scene.msh",
        ),
        (
            _SPHERE,
            {"computation": {"files": [{"path": "/nonexistent/run.py"}]}},
            FileNotFoundError,
            "run.py",
        ),
    ],
)
def test_hdf5_rejects_invalid_input_before_writing(tms, kwargs, error, match):
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        with pytest.raises(error, match=match):
            io.save_hdf5(handle, tms, **kwargs)
        assert list(handle) == []
        assert not handle.attrs


def _set_attribute(handle, name, key, value):
    handle[name].attrs[key] = value


def _replace(handle, name, value):
    del handle[name]
    handle[name] = value


def _mixed_polarizations(handle):
    handle["modes/polarization_scattered"] = ["magnetic", "electric"] * 3


@pytest.mark.parametrize(
    "mutate,lunit,match",
    [
        (lambda h: None, "inch", "unrecognized length unit: inch"),
        (
            lambda h: (
                h["modes"]
                .create_dataset("positions", data=np.zeros((1, 3)))
                .attrs.create("unit", "inch")
            ),
            "nm",
            "unrecognized positions unit: inch",
        ),
        (
            lambda h: h["angular_vacuum_wavenumber"].attrs.__delitem__("unit"),
            "nm",
            "angular_vacuum_wavenumber requires a unit attribute",
        ),
        (
            lambda h: _set_attribute(h, "angular_vacuum_wavenumber", "unit", "furlong"),
            "nm",
            "unrecognized angular_vacuum_wavenumber unit: furlong",
        ),
        (
            lambda h: h.__delitem__("angular_vacuum_wavenumber"),
            "nm",
            "no frequency, wavelength or wavenumber",
        ),
        (
            lambda h: _replace(h, "modes/polarization", ["negative", "electric"] * 3),
            "nm",
            "consistently name helicity or parity",
        ),
        (_mixed_polarizations, "nm", "same polarization type"),
        (
            lambda h: _replace(h, "tmatrix", np.zeros(6, complex)),
            "nm",
            "at least two matrix dimensions",
        ),
        (
            lambda h: _replace(h, "tmatrix", np.zeros((6, 5), complex)),
            "nm",
            "matrix shape does not match",
        ),
        (
            lambda h: _replace(h, "modes/l", [[1] * 6]),
            "nm",
            "missing or not one-dimensional",
        ),
    ],
)
def test_hdf5_loader_names_each_invalid_file(mutate, lunit, match):
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        io.save_hdf5(handle, _SPHERE)
        mutate(handle)
        with pytest.raises(ValueError, match=match):
            io.load_hdf5(handle, lunit=lunit)


@pytest.mark.parametrize("radius,shift", [(0.3, 1.5), (1.1, -2.5)])
def test_extended_scatterer_metadata_and_numerical_roundtrip(radius, shift):
    tm = tr.TMatrix.sphere(1, 1.3, radius, [2.3, 1])
    metadata = {
        "name": "sphere",
        "keywords": "passive, reciprocal",
        "material": {
            "name": "dielectric",
            "relative_permittivity": 2.3,
            "relative_permeability": 1,
            "chirality": 0,
        },
        "geometry": {"shape": "sphere", "radius": radius},
        "position": [shift, 0, 0],
    }
    with h5py.File("metadata.h5", "w", driver="core", backing_store=False) as file:
        io.save_hdf5(
            file,
            tm,
            scatterers=metadata,
            computation={"method": "Mie", "keywords": "semi-analytical"},
        )
        assert file.attrs["storage_format_version"] == "v1"
        assert file["scatterer/geometry"].attrs["shape"] == "sphere"
        assert file["scatterer/geometry/radius"].attrs["unit"] == "nm"
        assert_allclose(file["scatterer/geometry/radius"][()], radius)
        assert_allclose(file["scatterer/geometry/position"][()], [shift, 0, 0])
        assert file["scatterer/material"].attrs["name"] == "dielectric"
        assert_allclose(io.load_hdf5(file).array, tm.array, atol=0)
        assert "treams-rs=" in file["computation"].attrs["software"]


def test_mesh_and_reproducibility_files(tmp_path):
    tm = tr.TMatrix.sphere(1, 1.3, 0.2, [2.3, 1])
    mesh = tmp_path / "scene.msh"
    mesh.write_text("mesh text\nµm\n", encoding="utf-8")
    script = tmp_path / "simulation.py"
    script.write_text("print('reproduce')\n", encoding="utf-8")
    for mesh_input in (mesh, {"mesh.msh": mesh.read_text(encoding="utf-8")}):
        with h5py.File("metadata.h5", "w", driver="core", backing_store=False) as file:
            io.save_hdf5(
                file,
                tm,
                lunit="um",
                scatterers=[
                    {"geometry": {"shape": "sphere", "radius": 0.2}},
                    {"position": [1, 2, 3]},
                ],
                computation={
                    "method": "test",
                    "mesh": mesh_input,
                    "files": [script, {"path": script, "name": "copy.py"}],
                },
            )
            assert file["scatterer_0/geometry/radius"].attrs["unit"] == "um"
            assert_allclose(file["scatterer_1/geometry/position"][()], [1, 2, 3])
            assert isinstance(file.get("mesh", getlink=True), h5py.SoftLink)
            target = (
                file["mesh/mesh.msh"] if isinstance(mesh_input, dict) else file["mesh"]
            )
            assert target.asstr()[()] == mesh.read_text(encoding="utf-8")
            assert file["computation/files/copy.py"].asstr()[()] == script.read_text(
                encoding="utf-8"
            )
    with h5py.File("metadata.h5", "w", driver="core", backing_store=False) as file:
        io.save_hdf5(
            file,
            tm,
            scatterers={"geometry": {"shape": "sphere"}},
            computation={"method": "test"},
        )
        assert "storage_format_version" not in file.attrs


@pytest.mark.parametrize("tag,radius", [(10, 0.3), (987, 1.7)])
def test_gmsh_helper_actual_surface_tags_and_complete_geometry(tag, radius):
    model = MagicMock()
    model.occ.addSphere.side_effect = [tag, tag + 1]
    model.getBoundary.side_effect = [
        [(2, tag + 1000)],
        [(2, tag + 2000)],
        [(0, 11), (0, 12)],
    ]
    model.getEntities.return_value = [(0, 11), (0, 12)]
    assert (
        io.mesh_spheres([radius, radius * 2], [[0, 0, 0], [10, 0, 0]], model) is model
    )
    assert [call.args for call in model.addPhysicalGroup.call_args_list] == [
        (3, [tag]),
        (2, [tag + 1000]),
        (3, [tag + 1]),
        (2, [tag + 2000]),
    ]
    assert model.occ.addSphere.call_args_list[0].args == (0, 0, 0, radius)
    assert_allclose(model.mesh.setSize.call_args_list[0].args[1], radius * 0.4)
    model = MagicMock()
    with pytest.raises(ValueError, match="one radius"):
        io.mesh_spheres([1, 2], [[0, 0, 0]], model)
    model.occ.addSphere.assert_not_called()
