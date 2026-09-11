"""HDF5 T-matrix interchange. Install the optional ``treams-rs[io]`` extra."""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import h5py
import numpy as np

from ._core import Material, SphericalWaveBasis
from ._tmatrix import TMatrix

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

type MatrixSet = TMatrix | Sequence[MatrixSet] | NDArray[np.object_]

_PREFIXES = {
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "µ": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "": 0,
    "da": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
}
LENGTHS = {prefix + "m": 10.0**power for prefix, power in _PREFIXES.items()}
INVLENGTHS = {unit + "^{-1}": 1 / scale for unit, scale in LENGTHS.items()}
FREQUENCIES = {prefix + "Hz": 10.0**power for prefix, power in _PREFIXES.items()} | {
    prefix + "s^{-1}": 10.0**-power for prefix, power in _PREFIXES.items()
}


def mesh_spheres(
    radii: ArrayLike,
    positions: ArrayLike,
    model: Any,
    meshsize: float | None = None,
    meshsize_boundary: float | None = None,
) -> Any:
    """Add spheres, physical volume/surface groups and mesh sizes to a Gmsh model.

    The caller owns Gmsh initialization, mesh generation, writing and finalization.
    Entity tags are allocated by Gmsh so pre-existing geometry remains usable.
    """
    radii, positions = (
        np.atleast_1d(np.asarray(radii, dtype=float)),
        np.atleast_2d(np.asarray(positions, dtype=float)),
    )
    if radii.ndim != 1 or radii.size == 0 or positions.shape != (radii.size, 3):
        raise ValueError("require one radius and Cartesian position per sphere")
    if (
        not np.isfinite(radii).all()
        or np.any(radii <= 0)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("sphere radii must be positive and geometry finite")
    meshsize = float(np.max(radii) * 0.2) if meshsize is None else meshsize
    meshsize_boundary = meshsize if meshsize_boundary is None else meshsize_boundary
    if (
        not np.isfinite([meshsize, meshsize_boundary]).all()
        or min(meshsize, meshsize_boundary) <= 0
    ):
        raise ValueError("mesh sizes must be finite and positive")
    volumes = [
        (3, model.occ.addSphere(*position, radius))
        for radius, position in zip(radii, positions, strict=True)
    ]
    model.occ.synchronize()
    for volume in volumes:
        model.addPhysicalGroup(3, [volume[1]])
        surfaces = model.getBoundary([volume], combined=False, oriented=False)
        model.addPhysicalGroup(2, [tag for dim, tag in surfaces if dim == 2])
    model.mesh.setSize(model.getEntities(0), meshsize)
    model.mesh.setSize(
        model.getBoundary(volumes, combined=False, oriented=False, recursive=True),
        meshsize_boundary,
    )
    return model


def _save_scatterers(
    group: h5py.Group,
    scatterers: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    lunit: str,
) -> None:
    items = [scatterers] if isinstance(scatterers, Mapping) else scatterers
    for index, item in enumerate(items):
        scatterer = group.require_group(
            "scatterer" if len(items) == 1 else f"scatterer_{index}"
        )
        _description(
            scatterer,
            *(str(item.get(key, "")) for key in ("name", "description", "keywords")),
        )
        for name in ("material", "geometry"):
            values = item.get(name, {})
            child = scatterer.require_group(name)
            _description(
                child,
                *(
                    str(values.get(key, ""))
                    for key in ("name", "description", "keywords")
                ),
            )
            if name == "material":
                for key in (
                    "relative_permittivity",
                    "relative_permeability",
                    "chirality",
                ):
                    if key in values:
                        child[key] = values[key]
                continue
            unit = values.get("unit", lunit)
            if unit not in LENGTHS:
                raise ValueError(f"unrecognized geometry length unit: {unit}")
            child.attrs["unit"] = unit
            if "shape" in values:
                child.attrs["shape"] = values["shape"]
            data = {
                key: value
                for key, value in values.items()
                if key not in ("name", "description", "keywords", "shape", "unit")
            }
            if "position" in item:
                data["position"] = item["position"]
            for key, value in data.items():
                child.create_dataset(key, data=value).attrs["unit"] = unit


def _save_computation(
    group: h5py.Group, computation: Mapping[str, Any], lunit: str
) -> None:
    target = group.require_group("computation")
    _description(
        target,
        *(str(computation.get(key, "")) for key in ("name", "description", "keywords")),
    )
    if "method" in computation:
        target.attrs["method"] = computation["method"]
    target.attrs["software"] = (
        computation.get("software")
        or f"python={platform.python_version()}, treams-rs={version('treams-rs')}, h5py={version('h5py')}, numpy={np.__version__}"
    )
    mesh = computation.get("mesh")
    if mesh is not None:
        if isinstance(mesh, Mapping):
            saved_mesh = target.require_group("mesh")
            saved_mesh.attrs["unit"] = lunit
            for name, content in mesh.items():
                saved_mesh[name] = content
        else:
            path = Path(mesh)
            saved_mesh = target.create_dataset(
                "mesh" + path.suffix, data=path.read_text(encoding="utf-8")
            )
            saved_mesh.attrs["unit"] = lunit
        group["mesh"] = h5py.SoftLink(saved_mesh.name)
    files = computation.get("files", ())
    if files:
        file_group = target.require_group("files")
        for item in files:
            path = Path(item["path"] if isinstance(item, Mapping) else item)
            name = (
                item.get("name", path.name) if isinstance(item, Mapping) else path.name
            )
            file_group[name] = path.read_text(encoding="utf-8")


def _collect(value: MatrixSet) -> tuple[tuple[int, ...], list[TMatrix]]:
    if isinstance(value, TMatrix):
        return (), [value]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _collect(value.item())
    if (
        not isinstance(value, (Sequence, np.ndarray))
        or isinstance(value, (str, bytes))
        or len(value) == 0
    ):
        raise ValueError(
            "require a TMatrix or a nonempty rectangular sequence of TMatrix objects"
        )
    children = [_collect(item) for item in value]
    shape = children[0][0]
    if any(child_shape != shape for child_shape, _ in children):
        raise ValueError("T-matrix sequence must be rectangular")
    return (len(value), *shape), [tm for _, matrices in children for tm in matrices]


def _description(group: h5py.Group, name: str, description: str, keywords: str) -> None:
    for key, value in (
        ("name", name),
        ("description", description),
        ("keywords", keywords),
    ):
        if value:
            group.attrs[key] = str(value)


def save_hdf5(
    h5file: h5py.Group,
    tms: MatrixSet,
    name: str = "",
    description: str = "",
    keywords: str = "",
    embedding_group: h5py.Group | str | None = None,
    embedding_name: str = "",
    embedding_description: str = "",
    embedding_keywords: str = "",
    uuid: bytes | None = None,
    uuid_version: int = 4,
    lunit: str = "nm",
    *,
    scatterers: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    computation: Mapping[str, Any] | None = None,
) -> None:
    """Write spherical T matrices to an open HDF5 group, one matrix at a time.

    Supports one matrix or rectangular parameter sweeps sharing a basis and
    polarization convention. k0 and positions use reciprocal lunit and lunit.
    Optional scatterer and computation dictionaries write tmat.h5 v1 material,
    geometry, mesh and reproducibility-file metadata. The format version records
    the layout; this writer does not certify acceptance by a T-matrix database.
    """
    if lunit not in LENGTHS:
        raise ValueError(f"unrecognized length unit: {lunit}")
    shape, matrices = _collect(tms)
    first = matrices[0]
    basis = first.basis
    for tm in matrices:
        if (
            tm.poltype != first.poltype
            or tm.basis.modes != basis.modes
            or not np.array_equal(tm.basis.positions, basis.positions)
        ):
            raise ValueError(
                "T matrices must share mode labels, origins and polarization type"
            )
    values = h5file.create_dataset(
        "tmatrix", shape=(*shape, len(basis), len(basis)), dtype=np.complex128
    )
    for index, tm in zip(np.ndindex(shape), matrices, strict=True):
        values[index] = tm.array
    h5file.create_dataset(
        "uuid", data=np.void(uuid4().bytes if uuid is None else uuid)
    ).attrs["version"] = 4 if uuid is None else uuid_version
    _description(h5file, name, description, keywords)
    h5file.attrs["created_with"] = (
        f"treams-rs={version('treams-rs')}, h5py={version('h5py')}, numpy={np.__version__}"
    )
    h5file.create_dataset(
        "angular_vacuum_wavenumber",
        data=np.array([tm.k0 for tm in matrices]).reshape(shape),
    ).attrs["unit"] = lunit + "^{-1}"
    modes = h5file.require_group("modes")
    modes["l"], modes["m"] = basis.l, basis.m
    labels = (
        ("negative", "positive")
        if first.poltype == "helicity"
        else ("magnetic", "electric")
    )
    modes["polarization"] = [labels[p] for p in basis.pol]
    if np.any(basis.pidx != 0):
        modes["position_index"] = basis.pidx
        # Hard links preserve legacy names without duplicated mutable data.
        modes["pidx"] = modes["position_index"]
        modes["index"] = modes["position_index"]
    if basis.positions.shape != (1, 3) or np.any(basis.positions != 0):
        modes.create_dataset("positions", data=basis.positions).attrs["unit"] = lunit
    embedding = (
        h5file.require_group(
            "embedding" if embedding_group is None else embedding_group
        )
        if isinstance(embedding_group, (str, type(None)))
        else embedding_group
    )
    if "embedding" not in h5file:
        h5file["embedding"] = embedding
    for key, attribute in (
        ("relative_permittivity", "epsilon"),
        ("relative_permeability", "mu"),
    ):
        embedding[key] = np.array(
            [getattr(tm.material, attribute) for tm in matrices]
        ).reshape(shape)
    if first.poltype == "helicity":
        embedding["chirality_parameter"] = np.array(
            [tm.material.kappa for tm in matrices]
        ).reshape(shape)
        embedding["chirality"] = embedding["chirality_parameter"]
    _description(embedding, embedding_name, embedding_description, embedding_keywords)
    if scatterers is not None:
        _save_scatterers(h5file, scatterers, lunit)
    if computation is not None:
        _save_computation(h5file, computation, lunit)
    if (
        scatterers
        and computation
        and (
            computation.get("mesh") is not None
            or "semi-analytical" in computation.get("keywords", "")
        )
    ):
        h5file.attrs["storage_format_version"] = "v1"


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _read(
    group: h5py.Group, names: Sequence[str], default: object = None
) -> NDArray[np.generic]:
    for name in names:
        if name in group:
            return np.asarray(group[name][()])
    return np.asarray(default)


def _polarizations(values: NDArray[np.generic]) -> tuple[list[int], str]:
    labels = [_text(value) for value in values]
    helicity = {"negative": 0, "minus": 0, "positive": 1, "plus": 1}
    parity = {"magnetic": 0, "te": 0, "M": 0, "electric": 1, "tm": 1, "N": 1}
    for mapping, poltype in ((helicity, "helicity"), (parity, "parity")):
        if labels and all(label in mapping for label in labels):
            return [mapping[label] for label in labels], poltype
    raise ValueError("mode polarizations must consistently name helicity or parity")


def _convert_to_k0(
    values: NDArray[np.generic], kind: str, unit: str, target: str
) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    scale = INVLENGTHS[target]
    try:
        if kind == "vacuum_wavelength":
            return 2 * np.pi / (x * LENGTHS[unit] * scale)
        if kind in ("frequency", "angular_frequency"):
            return (
                x
                * FREQUENCIES[unit]
                / (299792458.0 * scale)
                * (2 * np.pi if kind == "frequency" else 1)
            )
        return (
            x
            * INVLENGTHS[unit]
            / scale
            * (2 * np.pi if kind == "vacuum_wavenumber" else 1)
        )
    except KeyError as error:
        raise ValueError(f"unrecognized {kind} unit: {unit}") from error


def load_hdf5(
    filename: str | Path | h5py.Group, lunit: str = "nm"
) -> TMatrix | NDArray[np.object_]:
    """Read one matrix or a parameter sweep, converting all coordinates to lunit.

    Accepts shared or separate incident/scattered mode lists and legacy treams
    names for embedding chirality and particle indices. Rectangular mode subsets
    are embedded into the union basis with missing coefficients set to zero.
    """
    if lunit not in LENGTHS:
        raise ValueError(f"unrecognized length unit: {lunit}")
    if isinstance(filename, h5py.Group):
        return _load_hdf5(filename, lunit)
    with h5py.File(filename, "r") as handle:
        return _load_hdf5(handle, lunit)


def _load_hdf5(handle: h5py.Group, lunit: str) -> TMatrix | NDArray[np.object_]:
    for kind in (
        "frequency",
        "angular_frequency",
        "vacuum_wavelength",
        "vacuum_wavenumber",
        "angular_vacuum_wavenumber",
    ):
        if kind in handle:
            frequencies = handle[kind]
            ks = _convert_to_k0(
                np.asarray(frequencies[()]),
                kind,
                _text(frequencies.attrs["unit"]),
                lunit + "^{-1}",
            )
            break
    else:
        raise ValueError("no frequency, wavelength or wavenumber definition found")
    dataset = handle["tmatrix"]
    if dataset.ndim < 2:
        raise ValueError("tmatrix data requires at least two matrix dimensions")
    shape = dataset.shape[:-2]
    positions = np.asarray(
        _read(handle, ["modes/positions"], np.zeros((1, 3))), dtype=np.float64
    )
    if "modes/positions" in handle:
        positions *= (
            LENGTHS[_text(handle["modes/positions"].attrs.get("unit", lunit))]
            / LENGTHS[lunit]
        )
    bases = []
    poltypes = []
    for side in ("incident", "scattered"):
        degree = _read(handle, [f"modes/l_{side}", "modes/l"])
        order = _read(handle, [f"modes/m_{side}", "modes/m"])
        polarization = _read(
            handle, [f"modes/polarization_{side}", "modes/polarization"]
        )
        if any(x.ndim != 1 for x in (degree, order, polarization)):
            raise ValueError("mode definition missing or not one-dimensional")
        pol, poltype = _polarizations(polarization)
        indices = _read(
            handle,
            [
                f"modes/positions_index_{side}",
                f"modes/position_index_{side}",
                "modes/position_index",
                "modes/pidx",
                "modes/index",
            ],
            np.zeros_like(degree),
        )
        bases.append(
            SphericalWaveBasis(zip(indices, degree, order, pol, strict=True), positions)
        )
        poltypes.append(poltype)
    incident, scattered = bases
    if poltypes[0] != poltypes[1]:
        raise ValueError(
            "incident and scattered modes must use the same polarization type"
        )
    if dataset.shape[-2:] != (len(scattered), len(incident)):
        raise ValueError("matrix shape does not match incident/scattered modes")
    basis = SphericalWaveBasis(
        dict.fromkeys((*incident.modes, *scattered.modes)), positions
    )
    incoming = [basis.modes.index(mode) for mode in incident.modes]
    outgoing = [basis.modes.index(mode) for mode in scattered.modes]
    embedding = handle.get("embedding", handle.get("materials/embedding", handle))
    n = _read(embedding, ["refractive_index"], 1).astype(np.complex128)
    impedance = _read(embedding, ["relative_impedance"], 1 / n).astype(np.complex128)
    epsilon = _read(embedding, ["relative_permittivity"], n / impedance)
    mu = _read(embedding, ["relative_permeability"], n * impedance)
    kappa = _read(embedding, ["chirality_parameter", "chirality"], 0)
    ks, epsilon, mu, kappa = (
        np.broadcast_to(x, shape) for x in (ks, epsilon, mu, kappa)
    )
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        array = np.zeros((len(basis), len(basis)), complex)
        array[np.ix_(outgoing, incoming)] = dataset[index]
        result[index] = TMatrix(
            array,
            k0=float(ks[index]),
            basis=basis,
            material=Material(
                complex(epsilon[index]), complex(mu[index]), complex(kappa[index])
            ),
            poltype=poltypes[0],
        )
    return result.item() if not shape else result
