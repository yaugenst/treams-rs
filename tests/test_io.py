import h5py
import numpy as np
import pytest
import treams
import treams.io
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import io


def _assert_same(actual, expected, scale=1):
    assert_allclose(actual.array, expected.array, atol=0)
    assert actual.basis.modes == expected.basis.modes
    assert_allclose(actual.basis.positions, expected.basis.positions * scale)
    assert_allclose(actual.k0, expected.k0 / scale)
    assert actual.material == expected.material
    assert actual.poltype == expected.poltype


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@settings(max_examples=30)
@given(kappa=st.floats(-0.15, 0.15), offset=st.floats(-2, 2), k0=st.floats(0.8, 2))
def test_hdf5_roundtrip_sweeps_origins_and_chirality(poltype, kappa, offset, k0):
    basis = tr.SphericalBasis.default(1, 2, [[offset, 0, 0], [0, 0, 0]])
    matrices = [
        tr.TMatrix(
            np.diag(np.arange(len(basis))) * (0.1 + 0.2j),
            basis=basis,
            k0=frequency,
            material=(1.3 + 0.1j, 1.2, kappa if poltype == "helicity" else 0),
            poltype=poltype,
        )
        for frequency in [k0, k0 * 1.3]
    ]
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        io.save_hdf5(handle, [[matrices[0]], [matrices[1]]], name="sweep", lunit="nm")
        loaded = io.load_hdf5(handle)
        assert loaded.shape == (2, 1)
        for i, expected in enumerate(matrices):
            _assert_same(loaded[i, 0], expected)
        converted = io.load_hdf5(handle, lunit="um")
        for i, expected in enumerate(matrices):
            _assert_same(converted[i, 0], expected, 1e-3)
        assert handle.attrs["name"] == "sweep"
        assert handle["modes/pidx"].id == handle["modes/position_index"].id
        if poltype == "helicity":
            assert (
                handle["embedding/chirality"].id
                == handle["embedding/chirality_parameter"].id
            )


def test_hdf5_single_matrix_and_path(tmp_path):
    tm = tr.TMatrix.sphere(2, 1.3, 0.2, [(3, 1), (1.3, 1.1, 0.08)])
    filename = tmp_path / "single.h5"
    with h5py.File(filename, "w") as handle:
        io.save_hdf5(handle, tm)
    _assert_same(io.load_hdf5(filename), tm)


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


@pytest.mark.parametrize(
    "kind,unit,value",
    [
        ("frequency", "THz", 299792458.0 / (2 * np.pi) * 1.3e-3),
        ("angular_frequency", "GHz", 299792458.0 * 1.3),
        ("vacuum_wavelength", "nm", 2 * np.pi / 1.3),
        ("vacuum_wavenumber", "um^{-1}", 1.3e3 / (2 * np.pi)),
        ("angular_vacuum_wavenumber", "m^{-1}", 1.3e9),
    ],
)
def test_all_frequency_representations(kind, unit, value):
    tm = tr.TMatrix.sphere(1, 1.3, 0.2, [3, 1])
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        io.save_hdf5(handle, tm)
        del handle["angular_vacuum_wavenumber"]
        handle[kind] = value
        handle[kind].attrs["unit"] = unit
        _assert_same(io.load_hdf5(handle), tm)


def test_rectangular_modes_and_refractive_material():
    tm = tr.TMatrix.sphere(2, 1.3, 0.2, [3, 1])
    incident = [0, 2, 3]
    scattered = [1, 0]
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        io.save_hdf5(handle, tm)
        del handle["tmatrix"]
        data = tm.array[np.ix_(scattered, incident)]
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
        incoming = [loaded.basis.modes.index(tm.basis.modes[i]) for i in incident]
        outgoing = [loaded.basis.modes.index(tm.basis.modes[i]) for i in scattered]
        assert_allclose(loaded.array[np.ix_(outgoing, incoming)], data)


def test_hdf5_rejects_mixed_bases_before_writing():
    first = tr.TMatrix.sphere(1, 1.3, 0.2, [3, 1])
    second = tr.TMatrix.sphere(2, 1.3, 0.2, [3, 1])
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        with pytest.raises(ValueError, match="share mode"):
            io.save_hdf5(handle, [first, second])
        assert list(handle) == []


@given(radius=st.floats(0.1, 1.2), shift=st.floats(-3, 3))
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


@given(tag=st.integers(10, 1000), radius=st.floats(0.1, 2))
def test_gmsh_helper_actual_surface_tags_and_complete_geometry(tag, radius):
    from unittest.mock import MagicMock

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
