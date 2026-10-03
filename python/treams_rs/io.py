"""T-matrix files in the tmat.h5 layout, and Gmsh meshes of sphere clusters.

Mirrors ``treams.io``. Install the optional ``treams-rs[io]`` extra for h5py.

- ``save_hdf5`` and ``load_hdf5`` write and read spherical T-matrices, alone or
  as a parameter sweep, in the HDF5 layout tmat.h5: the matrices, the angular
  vacuum wavenumber, the mode labels ``modes/l``, ``modes/m`` and
  ``modes/polarization``, the expansion positions and the embedding medium.
- ``mesh_spheres`` adds spheres to a Gmsh model, for solvers that need a
  volume mesh. It takes the ``gmsh.model`` object; install gmsh separately.
- ``LENGTHS``, ``INVLENGTHS`` and ``FREQUENCIES`` give the SI factor of each
  unit name, for example ``LENGTHS["nm"] == 1e-9`` and
  ``FREQUENCIES["THz"] == 1e12``.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from importlib.metadata import version as _version
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4 as _uuid4

import h5py
import numpy as np

from ._bases import SphericalBasis as _SphericalBasis
from ._material import Material as _Material
from ._tmatrix import TMatrix as _TMatrix

if TYPE_CHECKING:
    # Annotation-only names; runtime code uses the private aliases above.
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray

    from ._tmatrix import TMatrix

__all__ = [
    "FREQUENCIES",
    "INVLENGTHS",
    "LENGTHS",
    "MatrixSet",
    "load_hdf5",
    "mesh_spheres",
    "save_hdf5",
]

type MatrixSet = TMatrix | Sequence[MatrixSet] | NDArray[np.object_]
"""A ``TMatrix`` or a rectangular sweep: nested sequences or an object array of them."""

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
"""Length units in metres, from ``"ym"`` to ``"Ym"``: ``LENGTHS["nm"] == 1e-9``."""
INVLENGTHS = {unit + "^{-1}": 1 / scale for unit, scale in LENGTHS.items()}
"""Inverse length units in 1/m, the reciprocals of ``LENGTHS``: ``"nm^{-1}"`` is 1e9."""
FREQUENCIES = {prefix + "Hz": 10.0**power for prefix, power in _PREFIXES.items()} | {
    prefix + "s^{-1}": 10.0**-power for prefix, power in _PREFIXES.items()
}
"""Frequency units in Hz, written ``"THz"`` or ``"ps^{-1}"``: both map to 1e12."""


def mesh_spheres(
    radii: ArrayLike,
    positions: ArrayLike,
    model: Any,
    meshsize: float | None = None,
    meshsize_boundary: float | None = None,
) -> Any:
    """Add spheres, physical volume/surface groups and mesh sizes to a Gmsh model.

    Mirrors ``treams.io.mesh_spheres``. Each sphere gets one physical volume
    group and one physical surface group. Gmsh allocates every tag, so geometry
    already in the model stays usable. You initialize Gmsh, generate and write
    the mesh, and finalize Gmsh.

    Args:
        radii: Sphere radii, shape (spheres,); positive.
        positions: Sphere centres, shape (spheres, 3), in the unit of ``radii``.
        model: The Gmsh model to extend, usually ``gmsh.model``.
        meshsize: Mesh size at the points of the model; 0.2 times the largest
            radius by default.
        meshsize_boundary: Mesh size on the sphere surfaces; ``meshsize`` by
            default.

    Returns:
        ``model``, extended.

    Differences from treams:
        treams uses the sphere volume tags as surface tags, which picks the
        wrong surfaces in a model that already holds geometry. treams-rs reads
        the boundary surfaces of each sphere.
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


_DESCRIPTION = ("name", "description", "keywords")
_MATERIAL_KEYS = ("relative_permittivity", "relative_permeability", "chirality")


def _describe(group: h5py.Group, values: Mapping[str, object]) -> None:
    """Store the non-empty string forms of the name/description/keywords entries."""
    for key in _DESCRIPTION:
        text = str(values.get(key, ""))
        if text:
            group.attrs[key] = text


def _scatterer_items(
    scatterers: Mapping[str, Any] | Sequence[Mapping[str, Any]], lunit: str
) -> list[Mapping[str, Any]]:
    """Normalize scatterer metadata and validate it before anything is written."""
    items = [scatterers] if isinstance(scatterers, _Mapping) else list(scatterers)
    for item in items:
        unit = item.get("geometry", {}).get("unit", lunit)
        if unit not in LENGTHS:
            raise ValueError(f"unrecognized geometry length unit: {unit}")
    return items


def _save_scatterers(
    group: h5py.Group, items: Sequence[Mapping[str, Any]], lunit: str
) -> None:
    for index, item in enumerate(items):
        scatterer = group.require_group(
            "scatterer" if len(items) == 1 else f"scatterer_{index}"
        )
        _describe(scatterer, item)
        material = item.get("material", {})
        target = scatterer.require_group("material")
        _describe(target, material)
        for key in _MATERIAL_KEYS:
            if key in material:
                target[key] = material[key]
        geometry = item.get("geometry", {})
        target = scatterer.require_group("geometry")
        _describe(target, geometry)
        unit = geometry.get("unit", lunit)
        target.attrs["unit"] = unit
        if "shape" in geometry:
            target.attrs["shape"] = geometry["shape"]
        data = {
            key: value
            for key, value in geometry.items()
            if key not in (*_DESCRIPTION, "shape", "unit")
        }
        if "position" in item:
            data["position"] = item["position"]
        for key, value in data.items():
            target.create_dataset(key, data=value).attrs["unit"] = unit


type _Mesh = Mapping[str, Any] | tuple[str, str] | None


def _computation_files(
    computation: Mapping[str, Any],
) -> tuple[_Mesh, list[tuple[str, str]]]:
    """Read the mesh and reproducibility files before anything is written."""
    mesh = computation.get("mesh")
    if mesh is not None and not isinstance(mesh, _Mapping):
        path = _Path(mesh)
        mesh = ("mesh" + path.suffix, path.read_text(encoding="utf-8"))
    files: list[tuple[str, str]] = []
    for item in computation.get("files", ()):
        spec = item if isinstance(item, _Mapping) else {"path": item}
        path = _Path(spec["path"])
        files.append((spec.get("name", path.name), path.read_text(encoding="utf-8")))
    return mesh, files


def _save_computation(
    group: h5py.Group,
    computation: Mapping[str, Any],
    mesh: _Mesh,
    files: Sequence[tuple[str, str]],
    lunit: str,
) -> None:
    target = group.require_group("computation")
    _describe(target, computation)
    if "method" in computation:
        target.attrs["method"] = computation["method"]
    target.attrs["software"] = (
        computation.get("software")
        or f"python={platform.python_version()}, treams-rs={_version('treams-rs')}, h5py={_version('h5py')}, numpy={np.__version__}"
    )
    if mesh is not None:
        if isinstance(mesh, _Mapping):
            saved_mesh = target.require_group("mesh")
            for name, content in mesh.items():
                saved_mesh[name] = content
        else:
            saved_mesh = target.create_dataset(mesh[0], data=mesh[1])
        saved_mesh.attrs["unit"] = lunit
        group["mesh"] = h5py.SoftLink(saved_mesh.name)
    if files:
        file_group = target.require_group("files")
        for name, content in files:
            file_group[name] = content


def _collect(value: MatrixSet) -> tuple[tuple[int, ...], list[TMatrix]]:
    if isinstance(value, _TMatrix):
        return (), [value]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _collect(value.item())
    if (
        not isinstance(value, (_Sequence, np.ndarray))
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
    """Write spherical T-matrices to an open HDF5 group in the tmat.h5 layout.

    Mirrors ``treams.io.save_hdf5``. Every matrix of a sweep shares the mode
    labels, the positions and the polarization convention; k0 and the embedding
    medium may vary. ``save_hdf5`` checks every input and reads the mesh and
    reproducibility files before its first write, so an error leaves the group
    unchanged.

    The file gets the attribute ``storage_format_version = "v1"`` only when it
    holds both ``scatterers`` and ``computation`` metadata, and the computation
    stores a mesh or lists ``semi-analytical`` in its keywords. treams writes
    the same marker under the same conditions and also requires the embedding
    medium at ``/embedding``, which ``save_hdf5`` always provides through a link.
    The marker names the layout; it does not mean that a T-matrix database
    accepts the file.

    Args:
        h5file: Open, writable HDF5 group, for example an ``h5py.File``.
        tms: One ``TMatrix`` or a rectangular sweep: nested sequences or an
            object array of them. The sweep shape leads the shape of the
            ``tmatrix`` dataset.
        name: Name attribute of the file; skipped when empty, like every text
            attribute.
        description: Description attribute of the file.
        keywords: Keywords attribute of the file, for example
            ``"czinfinity, mirrorxyz, passive, reciprocal"``.
        embedding_group: Group, or group path, that receives the embedding
            medium; ``"embedding"`` by default. A link ``/embedding`` points
            to another group.
        embedding_name: Name attribute of the embedding medium.
        embedding_description: Description attribute of the embedding medium.
        embedding_keywords: Keywords attribute of the embedding medium.
        uuid: Bytes stored as the ``uuid`` dataset; a random version-4 UUID
            by default.
        uuid_version: Version attribute of a given ``uuid``.
        lunit: Length unit of the positions, a key of ``LENGTHS``; k0 is
            stored in its inverse.
        scatterers: Scatterer metadata: one mapping, written to
            ``/scatterer``, or a sequence, written to ``/scatterer_<i>``. A
            mapping may hold ``name``, ``description``, ``keywords``,
            ``material`` (``relative_permittivity``, ``relative_permeability``,
            ``chirality`` and text entries), ``geometry`` (``shape``, ``unit``
            and size datasets) and ``position``.
        computation: Metadata of the computation, written to
            ``/computation``: ``name``, ``description``, ``keywords``,
            ``method``, ``software`` (the package versions by default),
            ``mesh`` (a file path, or a mapping of file name to content) and
            ``files`` (paths, or ``{"path": ..., "name": ...}`` mappings, of
            files that reproduce the computation).

    Differences from treams:
        ``lunit`` comes before the keyword-only ``scatterers`` and
        ``computation``; treams orders them ``scatterers, computation,
        lunit``. Pass all three by keyword to run the same call in both.
        treams accepts only a sequence of matrices, and raises NameError
        for computation metadata without a mesh or the semi-analytical
        keyword. The file also stores the embedding chirality under both
        ``chirality`` and ``chirality_parameter``, and the position index
        under ``position_index``, ``pidx`` and ``index``, so treams reads it.
    """
    if lunit not in LENGTHS:
        raise ValueError(f"unrecognized length unit: {lunit}")
    shape, matrices = _collect(tms)
    # Validate and read every input before the first write, so errors leave
    # the target group untouched.
    items = None if scatterers is None else _scatterer_items(scatterers, lunit)
    mesh, files = _computation_files(computation or {})
    first = matrices[0]
    basis = first.basis
    for tm in matrices:
        if (
            tm.polarization != first.polarization
            or tm.basis.modes != basis.modes
            or not np.array_equal(tm.basis.positions, basis.positions)
        ):
            raise ValueError(
                "T matrices must share mode labels, positions and polarization type"
            )
    values = h5file.create_dataset(
        "tmatrix", shape=(*shape, len(basis), len(basis)), dtype=np.complex128
    )
    for index, tm in zip(np.ndindex(shape), matrices, strict=True):
        values[index] = tm.array
    h5file.create_dataset(
        "uuid", data=np.void(_uuid4().bytes if uuid is None else uuid)
    ).attrs["version"] = 4 if uuid is None else uuid_version
    _describe(h5file, {"name": name, "description": description, "keywords": keywords})
    h5file.attrs["created_with"] = (
        f"treams-rs={_version('treams-rs')}, h5py={_version('h5py')}, numpy={np.__version__}"
    )
    h5file.create_dataset(
        "angular_vacuum_wavenumber",
        data=np.array([tm.k0 for tm in matrices]).reshape(shape),
    ).attrs["unit"] = lunit + "^{-1}"
    modes = h5file.require_group("modes")
    modes["l"], modes["m"] = basis.l, basis.m
    labels = (
        ("negative", "positive")
        if first.polarization == "helicity"
        else ("magnetic", "electric")
    )
    modes["polarization"] = [labels[p] for p in basis.pol]
    if np.any(basis.pidx != 0):
        modes["position_index"] = basis.pidx
        # Hard links keep the treams key names pidx and index without copying data.
        modes["pidx"] = modes["position_index"]
        modes["index"] = modes["position_index"]
    if basis.positions.shape != (1, 3) or np.any(basis.positions != 0):
        modes.create_dataset("positions", data=basis.positions).attrs["unit"] = lunit
    group = "embedding" if embedding_group is None else embedding_group
    embedding = h5file.require_group(group) if isinstance(group, str) else group
    if "embedding" not in h5file:
        h5file["embedding"] = embedding
    for key, attribute in (
        ("relative_permittivity", "epsilon"),
        ("relative_permeability", "mu"),
    ):
        embedding[key] = np.array(
            [getattr(tm.medium, attribute) for tm in matrices]
        ).reshape(shape)
    if first.polarization == "helicity":
        embedding["chirality_parameter"] = np.array(
            [tm.medium.kappa for tm in matrices]
        ).reshape(shape)
        embedding["chirality"] = embedding["chirality_parameter"]
    _describe(
        embedding,
        {
            "name": embedding_name,
            "description": embedding_description,
            "keywords": embedding_keywords,
        },
    )
    if items is not None:
        _save_scatterers(h5file, items, lunit)
    if computation is not None:
        _save_computation(h5file, computation, mesh, files, lunit)
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
    """Read one T-matrix or a parameter sweep from a tmat.h5 file.

    Mirrors ``treams.io.load_hdf5``. The frequency comes from the first of
    ``frequency``, ``angular_frequency``, ``vacuum_wavelength``,
    ``vacuum_wavenumber`` and ``angular_vacuum_wavenumber``, converted with its
    unit attribute. The file may list one set of modes or separate incident
    and scattered modes, and may use the treams key names ``chirality``,
    ``pidx`` and ``index``. A matrix over a subset of modes becomes a square
    matrix over the union of both mode sets, with zeros for the missing
    coefficients.

    Args:
        filename: Path of an HDF5 file, or an open ``h5py.Group``.
        lunit: Length unit of the returned positions and of 1/k0, a key of
            ``LENGTHS``.

    Returns:
        One ``TMatrix`` for a file without sweep axes, otherwise an object
        array of ``TMatrix`` with the sweep shape.

    Differences from treams:
        treams drops the stored positions when it joins the incident and
        scattered modes, and reads only ``chirality_parameter`` for the
        embedding chirality.
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
            unit = frequencies.attrs.get("unit")
            if unit is None:
                raise ValueError(f"{kind} requires a unit attribute")
            ks = _convert_to_k0(
                np.asarray(frequencies[()]), kind, _text(unit), lunit + "^{-1}"
            )
            break
    else:
        raise ValueError("no frequency, wavelength or wavenumber definition found")
    dataset = handle["tmatrix"]
    if dataset.ndim < 2:
        raise ValueError("tmatrix data requires at least two matrix dimensions")
    shape = dataset.shape[:-2]
    positions = np.zeros((1, 3))
    if "modes/positions" in handle:
        stored = handle["modes/positions"]
        unit = _text(stored.attrs.get("unit", lunit))
        if unit not in LENGTHS:
            raise ValueError(f"unrecognized positions unit: {unit}")
        positions = np.asarray(stored[()], dtype=np.float64) * (
            LENGTHS[unit] / LENGTHS[lunit]
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
            _SphericalBasis(zip(indices, degree, order, pol, strict=True), positions)
        )
        poltypes.append(poltype)
    incident, scattered = bases
    if poltypes[0] != poltypes[1]:
        raise ValueError(
            "incident and scattered modes must use the same polarization type"
        )
    if dataset.shape[-2:] != (len(scattered), len(incident)):
        raise ValueError("matrix shape does not match incident/scattered modes")
    basis = _SphericalBasis(
        dict.fromkeys((*incident.modes, *scattered.modes)), positions
    )
    lookup = {mode: i for i, mode in enumerate(basis.modes)}
    incoming = [lookup[mode] for mode in incident.modes]
    outgoing = [lookup[mode] for mode in scattered.modes]
    # Square files in the union order need no zero fill and scatter.
    square = incoming == outgoing == list(range(len(basis)))
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
        if square:
            array = np.asarray(dataset[index], dtype=complex)
        else:
            array = np.zeros((len(basis), len(basis)), complex)
            array[np.ix_(outgoing, incoming)] = dataset[index]
        result[index] = _TMatrix(
            array,
            k0=float(ks[index]),
            basis=basis,
            material=_Material(
                complex(epsilon[index]), complex(mu[index]), complex(kappa[index])
            ),
            poltype=poltypes[0],
        )
    return result.item() if not shape else result
