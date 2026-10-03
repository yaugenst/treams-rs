"""treams names that treams-rs spells differently, and where each one went.

``UPSTREAM_NAMES`` covers the treams top-level namespace. ``UPSTREAM_MEMBERS``
maps members of treams objects to the treams-rs physics names and marks the
members that exist only under their treams name. The two tables feed three
places:

* errors: ``treams_rs.<name>`` (through ``treams_rs.__getattr__``) and a
  missing member of a physics class (through ``UpstreamMembers``) raise an
  AttributeError that names the treams-rs replacement and its note;
* the support catalog, which publishes both tables;
* the name map ``docs/coming-from-treams/names.md``, whose table
  ``scripts/generate_docs.py`` writes from the catalog.

Each physics class lists its treams members in the order of the
treams-compatible block at the end of the class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, NoReturn

__all__ = [
    "UPSTREAM_MEMBERS",
    "UPSTREAM_NAMES",
    "Renamed",
    "UpstreamMembers",
    "missing_attribute",
]


class Renamed(NamedTuple):
    """Where a treams name lives in treams-rs."""

    target: str | None
    """treams-rs name relative to the owner, or None when nothing corresponds."""
    note: str
    """What differs, in one or two sentences; empty when the rename says it all."""


_OPERATORS = (
    "BField",
    "ChangePoltype",
    "DField",
    "EField",
    "Expand",
    "ExpandLattice",
    "FField",
    "GField",
    "HField",
    "Permute",
    "PhysicsArray",
    "Rotate",
    "Translate",
    "bfield",
    "changepoltype",
    "dfield",
    "efield",
    "expand",
    "expandlattice",
    "ffield",
    "gfield",
    "hfield",
    "permute",
    "rotate",
    "translate",
)

# treams top-level name -> treams-rs name relative to treams_rs.
UPSTREAM_NAMES: dict[str, Renamed] = {
    "SphericalWaveBasis": Renamed("SphericalBasis", ""),
    "CylindricalWaveBasis": Renamed("CylindricalBasis", ""),
    "PlaneWaveBasisByUnitVector": Renamed("PlaneWaveBasis", ""),
    "PlaneWaveBasisByComp": Renamed("PlaneWavePorts", ""),
    "TMatrixC": Renamed("CylindricalTMatrix", ""),
    "SMatrices": Renamed(
        "SMatrix",
        "treams-rs SMatrix is the full two-port network (treams SMatrices); "
        "one block is ScatteringBlock (treams SMatrix).",
    ),
    **{name: Renamed(f"operators.{name}", "") for name in _OPERATORS},
    "config": Renamed(
        None,
        "treams-rs has no global settings; pass poltype= or polarization= "
        "explicitly (default helicity).",
    ),
    "util": Renamed(
        None,
        "treams-rs has no util module: bases are ordered sets, and PhysicsArray "
        "is treams_rs.operators.PhysicsArray.",
    ),
}

_FIELDS = Renamed(
    None,
    "Fields belong to waves: use response.scatter(incident).efield(points); "
    "treams_rs.operators.efield builds a field matrix.",
)
_ANNOTATIONS = Renamed(
    None,
    "treams-rs objects keep basis, k0, medium and polarization as attributes "
    "instead of array annotations.",
)
_CLUSTER = Renamed(
    None,
    "Use Cluster(particles, positions=xyz); Cluster.solve() returns the coupled "
    "T-matrix.",
)
# Members that treams-rs names as treams does.
_TREAMS_ONLY = "treams-rs keeps the treams name."


def _treams_only(name: str, note: str = "") -> Renamed:
    return Renamed(name, f"{_TREAMS_ONLY} {note}" if note else _TREAMS_ONLY)


_SETTING = {
    "material": Renamed("medium", ""),
    "poltype": Renamed("polarization", ""),
}
_TMATRIX_HEAD = {
    **_SETTING,
    "modetype": _treams_only(
        "modetype", "Always ('singular', 'regular'): outgoing rows, incident columns."
    ),
    "changepoltype": Renamed("with_polarization", ""),
    "expand": Renamed("in_basis", ""),
}
_TMATRIX_TAIL = {
    "interaction": _treams_only("interaction"),
    "latticeinteraction": _treams_only("latticeinteraction"),
    "expandlattice": _treams_only("expandlattice"),
    "permute": _treams_only(
        "permute", "It raises TypeError: permutations act on plane-wave ports."
    ),
    "cluster": _CLUSTER,
    "ann": _ANNOTATIONS,
    "relax": _ANNOTATIONS,
    **dict.fromkeys(
        ("efield", "hfield", "dfield", "bfield", "gfield", "ffield"), _FIELDS
    ),
}
_WAVE_HEAD = {**_SETTING, "modetype": Renamed("kind", "")}

# treams-rs class -> treams member -> treams-rs member of that class. Each class
# lists its treams names in the order of its treams-compatible block.
UPSTREAM_MEMBERS: dict[str, dict[str, Renamed]] = {
    "TMatrix": {
        **_TMATRIX_HEAD,
        "xs": Renamed("cross_sections", ""),
        "xs_ext_avg": Renamed("average_cross_sections", "Read its extinction field."),
        "xs_sca_avg": Renamed("average_cross_sections", "Read its scattering field."),
        "cd": Renamed("circular_dichroism", ""),
        "db": Renamed("duality_breaking", ""),
        "chi": Renamed("electromagnetic_chirality", ""),
        "sphere": _treams_only(
            "sphere", "sphere_tmatrix and multilayer_sphere_tmatrix take keywords."
        ),
        **_TMATRIX_TAIL,
    },
    "CylindricalTMatrix": {
        **_TMATRIX_HEAD,
        "xw": Renamed("cross_widths", ""),
        "xw_ext_avg": Renamed("average_cross_widths", "Read its extinction field."),
        "xw_sca_avg": Renamed("average_cross_widths", "Read its scattering field."),
        "cylinder": _treams_only(
            "cylinder",
            "cylinder_tmatrix and multilayer_cylinder_tmatrix take keywords.",
        ),
        **_TMATRIX_TAIL,
        "from_array": Renamed(
            None,
            "Use solve_periodic(...).to_cylindrical(basis) for a z-periodic "
            "array of spheres.",
        ),
    },
    "SMatrix": {
        "material": Renamed(
            None,
            "Returns (positive_medium, negative_medium), the treams order of the pair.",
        ),
        "poltype": Renamed("polarization", ""),
        "changepoltype": Renamed("with_polarization", ""),
        "add": Renamed("cascade", ""),
        "illuminate": Renamed("scatter", ""),
        "tr": Renamed("power", ""),
        "cd": Renamed("circular_dichroism", ""),
        "periodic": Renamed("transfer_matrix", ""),
        "bands_kz": Renamed("bands", ""),
        "from_array": Renamed(
            None, "Use solve_periodic(...).to_smatrix(ports) for a periodic array."
        ),
    },
    "Wave": {
        **_WAVE_HEAD,
        "changepoltype": Renamed("with_polarization", ""),
        "expand": Renamed("in_basis", ""),
    },
    "PlaneWave": {
        **_WAVE_HEAD,
        "expand": Renamed("in_basis", ""),
        "changepoltype": Renamed("with_polarization", ""),
    },
    "Material": {
        "from_n": Renamed("from_refractive_index", ""),
        "from_nmp": Renamed("from_helicity_indices", ""),
    },
}


def missing_attribute(owner: str, name: str) -> AttributeError:
    """The error for a treams name: ``owner`` is ``"treams_rs"`` or a class name."""
    if owner == "treams_rs":
        target, note = UPSTREAM_NAMES[name]
        upstream, prefix = f"treams.{name}", "treams_rs."
    else:
        target, note = UPSTREAM_MEMBERS[owner][name]
        upstream, prefix = f"the treams member {name}", f"{owner}."
    if target is None:
        return AttributeError(f"{owner} has no {name!r}: {note}")
    message = f"{owner} has no {name!r}: {upstream} corresponds to {prefix}{target}."
    return AttributeError(f"{message} {note}" if note else message)


def _member_error(cls: type, name: str, default: str) -> AttributeError:
    """The error for a missing ``name`` on ``cls`` or one of its instances."""
    for owner in cls.__mro__:
        members = UPSTREAM_MEMBERS.get(owner.__name__)
        if members is not None:
            if name in members:
                return missing_attribute(owner.__name__, name)
            break
    return AttributeError(default, name=name)


class _ClassMembers(type):
    """Metaclass that catches treams members looked up on the class, e.g. TMatrix.cluster."""

    # Hidden from type checkers, which would otherwise accept any attribute.
    if not TYPE_CHECKING:

        def __getattr__(cls, name: str) -> NoReturn:
            raise _member_error(
                cls, name, f"type object {cls.__name__!r} has no attribute {name!r}"
            )


class UpstreamMembers(metaclass=_ClassMembers):
    """Base of the physics classes: a missing treams member names its replacement.

    Looking up a name listed in UPSTREAM_MEMBERS that the class lacks raises
    missing_attribute; any other missing name raises the usual AttributeError.
    Both run only after the normal attribute lookup fails. Python also calls
    __getattr__ when a property raises AttributeError, so such an error
    reads as a missing attribute named after the property.
    """

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> NoReturn:
            raise _member_error(
                type(self),
                name,
                f"{type(self).__name__!r} object has no attribute {name!r}",
            )
