"""Planar S-matrices: interfaces, layers, propagation and their composition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from . import _native, diff
from ._array import PhysicsArray
from ._bases import PlaneWavePorts
from ._dispatch import autodiff_method, backend_for, namespace
from ._material import Material, MaterialLike
from ._operators import periodic_channels
from ._polarization import (
    PARITY_CHANGE,
    change_polarization,
    check_poltype_medium,
    resolve_poltype,
    target_poltype,
)
from ._results import BandModes, CircularDichroism, PowerBalance, ScatteredPorts
from ._upstream import UpstreamMembers
from ._validation import check_k0, one_of
from ._waves import PlaneWave, Wave, check_compatible, propagation_side

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import ArrayLike, NDArray

    from ._tmatrix import CylindricalTMatrix, TMatrix

__all__ = [
    "SMatrix",
    "ScatteringBlock",
    "chirality_density",
    "interface",
    "multilayer_slab",
    "poynting_avg_z",
    "propagation",
    "slab",
    "stack",
]


class ScatteringBlock(PhysicsArray):
    """One scattering block with explicit input and output port metadata.

    The treams counterpart is ``treams.SMatrix``, one block of the network
    that treams calls ``SMatrices``. It keeps the treams attribute names of
    PhysicsArray. Blocks from ``SMatrix.block`` store material and modetype as
    pairs (outgoing ports, incoming ports); a block built directly stores what
    it is given. poltype is one polarization convention for both sides.
    """

    def __init__(
        self,
        arr: ArrayLike,
        *,
        k0: float,
        basis: PlaneWavePorts,
        material: MaterialLike | tuple[MaterialLike, MaterialLike] = 1,
        poltype: str | None = None,
        modetype: tuple[str, str] = ("up", "up"),
    ) -> None:
        poltype = resolve_poltype(poltype)
        _check_block(arr, basis, modetype)
        super().__init__(
            arr,
            basis=basis,
            k0=k0,
            material=material,
            poltype=poltype,
            modetype=modetype,
        )

    @classmethod
    def _adopt(
        cls,
        array: NDArray[np.complex128],
        *,
        k0: float,
        basis: PlaneWavePorts,
        material: tuple[MaterialLike, MaterialLike],
        poltype: str,
        modetype: tuple[str, str],
    ) -> Self:
        """Wrap a read-only internal array, or a view of one, without copying it.

        Validation matches the public constructor. The block takes the lattice
        and Bloch vector of its ports.
        """
        poltype = resolve_poltype(poltype)
        _check_block(array, basis, modetype)
        result = cls.__new__(cls)
        result.array = array
        result.basis, result.k0 = basis, k0
        result.poltype, result.modetype = poltype, modetype
        result.material = (Material(material[0]), Material(material[1]))
        result.lattice, result.kpar = basis.lattice, basis.kpar
        result._check()
        return result


def _check_block(
    arr: ArrayLike, basis: PlaneWavePorts, modetype: tuple[str, str]
) -> None:
    if not isinstance(basis, PlaneWavePorts):
        raise TypeError("S-matrix requires a component plane-wave basis")
    if np.shape(arr) != (len(basis), len(basis)) or not np.isfinite(arr).all():
        raise ValueError("S-matrix requires a finite square block matching its basis")
    if len(modetype) != 2 or any(value not in ("up", "down") for value in modetype):
        raise ValueError("S-matrix port directions must be up or down")


class SMatrix(UpstreamMembers):
    """Planar two-port network: four scattering blocks of plane-wave ports.

    The treams counterpart is ``treams.SMatrices``; one block is a
    ScatteringBlock (``treams.SMatrix``). ``array`` has shape (2, 2, n, n):
    the first index is the outgoing direction and the second the incoming
    one, 0 for up (towards the positive side) and 1 for down.
    ``positive_medium`` (above) and ``negative_medium`` (below) are the
    exterior media on either side of the basis normal. ``interface``,
    ``slab``, ``stack`` and the module-level factories take media and layers
    from the negative to the positive side; the treams ``material`` pair is
    (positive side, negative side).

    The treams-rs names come first. The treams names (``add``,
    ``illuminate``, ``tr``, ``material``, ...) follow at the end of the class
    and call them.

    Treat it as immutable: the constructor checks the metadata attributes once
    and never again.
    """

    polarization: str
    """Polarization channel convention."""

    def __new__(cls, smats: Any = None, **kwargs: Any) -> Any:
        backend = backend_for(smats, kwargs)
        if backend is None:
            return super().__new__(cls)
        material = kwargs.pop("material", None)
        positive = kwargs.pop("positive_medium", None)
        negative = kwargs.pop("negative_medium", None)
        if material is not None:
            if positive is not None or negative is not None:
                raise ValueError(
                    "specify positive_medium and negative_medium or material, not both"
                )
            if isinstance(material, tuple):
                if len(material) != 2:
                    raise ValueError(
                        "material requires two media: (positive side, negative side)"
                    )
                positive, negative = material
            else:
                positive = negative = material
        polarization = one_of(
            "polarization",
            kwargs.pop("polarization", None),
            "poltype",
            kwargs.pop("poltype", None),
            None,
        )
        return namespace(backend).smatrix(
            smats,
            positive_medium=1 if positive is None else positive,
            negative_medium=1 if negative is None else negative,
            polarization=resolve_poltype(polarization),
            **kwargs,
        )

    def __init__(
        self,
        smats: ArrayLike,
        *,
        k0: float,
        basis: PlaneWavePorts,
        positive_medium: MaterialLike | None = None,
        negative_medium: MaterialLike | None = None,
        polarization: str | None = None,
        material: MaterialLike | tuple[MaterialLike, MaterialLike] | None = None,
        poltype: str | None = None,
    ):
        """Copy the four blocks of a two-port network.

        Args:
            smats: shape (2, 2, n, n) with n = len(basis). The first index is the
                outgoing direction (0 up, 1 down), the second the incoming one.
            k0: positive vacuum angular wavenumber.
            basis: the plane-wave ports of every block.
            positive_medium: exterior medium on the positive side (above),
                vacuum by default.
            negative_medium: exterior medium on the negative side (below),
                vacuum by default.
            polarization: "helicity" (default) or "parity".
            material: treams name of the media: one medium for both sides, or
                the pair (positive side, negative side).
            poltype: treams name of ``polarization``.

        Give each quantity under one name.
        """
        # treams keywords
        if material is None:
            material = (
                1 if positive_medium is None else positive_medium,
                1 if negative_medium is None else negative_medium,
            )
        elif positive_medium is not None or negative_medium is not None:
            raise ValueError(
                "specify positive_medium and negative_medium or material, not both"
            )
        polarization = one_of("polarization", polarization, "poltype", poltype, None)
        self._setup(
            smats,
            copy=True,
            k0=k0,
            basis=basis,
            material=material,
            poltype=polarization,
        )

    @classmethod
    def _adopt(
        cls,
        array: NDArray[np.complex128],
        *,
        k0: float,
        basis: PlaneWavePorts,
        material: MaterialLike | tuple[MaterialLike, MaterialLike] = 1,
        poltype: str | None = None,
    ) -> Self:
        """Wrap a freshly computed internal array without copying it.

        Nothing else may write to the array; it becomes read-only. Validation
        matches the public constructor.
        """
        result = cls.__new__(cls)
        result._setup(
            array,
            copy=False,
            k0=k0,
            basis=basis,
            material=material,
            poltype=poltype,
        )
        return result

    def _setup(
        self,
        smats: ArrayLike,
        *,
        copy: bool,
        k0: float,
        basis: PlaneWavePorts,
        material: MaterialLike | tuple[MaterialLike, MaterialLike],
        poltype: str | None,
    ) -> None:
        poltype = resolve_poltype(poltype)
        if not isinstance(basis, PlaneWavePorts):
            raise TypeError("S-matrix requires a component plane-wave basis")
        # copy=None keeps a complex128 C-ordered array as it is.
        self.array: NDArray[np.complex128] = np.array(
            smats, dtype=np.complex128, copy=True if copy else None, order="C"
        )
        if (
            self.array.shape != (2, 2, len(basis), len(basis))
            or not np.isfinite(self.array).all()
        ):
            raise ValueError(
                "S matrices require finite shape (2, 2, len(basis), len(basis))"
            )
        k0 = check_k0(k0)
        if isinstance(material, tuple):
            if len(material) != 2:
                raise ValueError(
                    "material requires two media: (positive side, negative side)"
                )
            self._media = (Material(material[0]), Material(material[1]))
        else:
            self._media = (Material(material), Material(material))
        self.polarization = check_poltype_medium(poltype, *self._media)
        self.k0, self.basis = k0, basis
        self.array.flags.writeable = False

    def __getitem__(
        self, key: int | str | tuple[int | str, int | str]
    ) -> NDArray[np.complex128]:
        keys = {0: 0, 1: 1, "up": 0, "down": 1}
        if isinstance(key, tuple):
            return self.array[keys[key[0]], keys[key[1]]]
        return self.array[keys[key]]

    def block(self, outgoing: int | str, incoming: int | str) -> ScatteringBlock:
        """Read-only block view with port metadata, sharing this stack's storage.

        Numeric indexing remains an ndarray view for inexpensive numerical work.
        Name physical sides with positive/negative, for example
        block(outgoing="positive", incoming="negative") is forward transmission.
        Numeric 0/1 and up/down specify propagation directions instead.
        """
        outgoing_keys = {0: 0, 1: 1, "up": 0, "down": 1, "positive": 0, "negative": 1}
        incoming_keys = {0: 0, 1: 1, "up": 0, "down": 1, "negative": 0, "positive": 1}
        i, j = outgoing_keys[outgoing], incoming_keys[incoming]
        return ScatteringBlock._adopt(
            self.array[i, j],
            k0=self.k0,
            basis=self.basis,
            material=(self._media[i], self._media[1 - j]),
            poltype=self.polarization,
            modetype=(("up", "down")[i], ("up", "down")[j]),
        )

    def __len__(self) -> int:
        return 2

    @property
    def positive_medium(self) -> Material:
        """Exterior medium on the positive side of the basis normal."""
        return self._media[0]

    @property
    def negative_medium(self) -> Material:
        """Exterior medium on the negative side of the basis normal."""
        return self._media[1]

    @autodiff_method
    def cascade(self, next_layer: SMatrix) -> SMatrix:
        """Place the next layer on the positive side and compose all reflections.

        Args:
            next_layer: an SMatrix with the same ports, k0 and polarization.
                Its negative medium must equal this network's positive medium.

        Returns the network from this negative medium to the positive medium of
        ``next_layer``.
        """
        self._check_adjacent(next_layer)
        value, _ = diff.smatrix_add(self.array, next_layer.array)
        return type(self)._adopt(
            value,
            basis=self.basis,
            k0=self.k0,
            material=(next_layer._media[0], self._media[1]),
            poltype=self.polarization,
        )

    def with_polarization(self, polarization: str) -> SMatrix:
        """Express the same scattering response in another polarization convention.

        Every port needs its partner of opposite polarization in the basis.
        """
        return self._with_polarization_target(polarization)

    def transfer_matrix(self) -> NDArray[np.complex128]:
        """Transfer matrix for repeating a cell along the basis normal.

        The outer media must match. Strongly evanescent channels can make a
        transfer matrix ill-conditioned; finite-stack composition remains in S form.
        """
        if self._media[0] != self._media[1]:
            raise ValueError("periodic repetition requires matching outer media")
        return diff.smatrix_periodic(self.array)[0]

    @autodiff_method
    def bands(self, *, period: float) -> BandModes[NDArray[np.complex128]]:
        """Bloch wavenumbers and eigenvectors of the repeated cell along the basis normal.

        Args:
            period: positive cell length along the normal, in units inverse
                to k0. The outer media must match.
        """
        return BandModes(*self._bands(period))

    @autodiff_method
    def power(
        self,
        incident: ArrayLike | PlaneWave | Wave,
        *,
        side: str | None = None,
    ) -> PowerBalance[float]:
        """Transmission, reflection and absorption fractions for one illumination.

        Args:
            incident: a PlaneWave, a Wave in these ports, or n coefficients of
                one illumination, ordered like the ports.
            side: the exterior the light comes from, "negative" or "positive".
                Physical waves infer it from their propagation direction; raw
                arrays default to "negative".

        Returns fractions of the incident power, so transmission + reflection
        + absorption = 1.
        """
        return PowerBalance(
            *self._transmission_balance(incident, _side_direction(side))
        )

    def circular_dichroism(
        self,
        incident: ArrayLike | PlaneWave | Wave,
        *,
        side: str | None = None,
    ) -> CircularDichroism:
        """Named transmission and total-outgoing-power polarization contrasts."""
        return CircularDichroism(
            *self._circular_dichroism(incident, _side_direction(side))
        )

    @autodiff_method
    def scatter(
        self,
        *,
        negative: ArrayLike | PlaneWave | Wave | None = None,
        positive: ArrayLike | PlaneWave | Wave | None = None,
    ) -> ScatteredPorts[Wave]:
        """Outgoing waves for incidence from either or both named exterior sides.

        Each supplied source is one illumination; coherent two-sided illumination
        is accepted. Evaluate fields on the corresponding exterior side.
        """
        if negative is not None:
            up, down = self._illuminated(negative, positive, "up", None)
        elif positive is not None:
            up, down = self._illuminated(positive, None, "down", None)
        else:
            raise ValueError("supply an incident wave on at least one side")
        return ScatteredPorts(
            Wave(
                up,
                basis=self.basis,
                k0=self.k0,
                medium=self._media[0],
                kind="up",
                polarization=self.polarization,
            ),
            Wave(
                down,
                basis=self.basis,
                k0=self.k0,
                medium=self._media[1],
                kind="down",
                polarization=self.polarization,
            ),
        )

    def __iter__(self) -> Iterator[NDArray[np.complex128]]:
        return iter(self.array)

    @classmethod
    def interface(
        cls,
        basis: PlaneWavePorts,
        k0: float,
        materials: Sequence[MaterialLike],
        poltype: str | None = None,
    ) -> SMatrix:
        """Planar interface between two media, normal to the basis normal.

        Args:
            basis: the plane-wave ports; ``basis.normal_axis`` is the normal.
            k0: positive vacuum angular wavenumber, in inverse length units.
            materials: two media ordered negative side, positive side (below,
                above); each follows the Material constructor.
            poltype: "helicity" (default) or "parity"; parity needs achiral
                media.

        ``interface(negative_medium=..., positive_medium=...)`` takes keywords
        instead.
        """
        backend = backend_for(k0, materials)
        if backend is not None:
            if len(materials) != 2:
                raise ValueError(
                    "an interface requires two materials, below then above"
                )
            return namespace(backend).interface(
                basis=basis,
                k0=k0,
                negative_medium=materials[0],
                positive_medium=materials[1],
                polarization=resolve_poltype(poltype),
            )
        poltype = resolve_poltype(poltype)
        k0 = check_k0(k0)
        if len(materials) != 2:
            raise ValueError("an interface requires two materials, below then above")
        below, above = (Material(m) for m in materials)
        check_poltype_medium(poltype, below, above)
        groups = _port_groups(basis)
        compact = _layer_blocks(groups, k0, (below, above), [], basis.alignment)
        return cls._adopt(
            _scatter_ports(basis, groups, compact, poltype),
            basis=basis,
            k0=k0,
            material=(above, below),
            poltype=poltype,
        )

    @classmethod
    def _from_response(
        cls,
        local: TMatrix | CylindricalTMatrix,
        basis: PlaneWavePorts,
        lattice: ArrayLike,
        kpar: ArrayLike,
        response: Callable[[], NDArray[np.complex128]],
    ) -> SMatrix:
        """Radiate a periodic response of ``local``'s cell into plane-wave ports.

        Part of the package-internal protocol: PeriodicResponse.to_smatrix calls
        it. ``response()`` supplies the solved array once the ports match the
        lattice.
        """
        channels = periodic_channels(
            local.basis, basis, local.ks, lattice, kpar, local.polarization
        )
        value, _ = diff.smatrix_from_array(response(), channels)
        return cls._adopt(
            value,
            basis=basis,
            k0=local.k0,
            material=local.medium,
            poltype=local.polarization,
        )

    @classmethod
    def propagation(
        cls,
        r: ArrayLike,
        basis: PlaneWavePorts,
        k0: float,
        material: MaterialLike = 1,
        poltype: str | None = None,
    ) -> SMatrix:
        """Homogeneous propagation through one medium.

        Args:
            r: distance along the basis normal, or a Cartesian displacement of
                shape (3,), in units inverse to k0.
            basis: the plane-wave ports.
            k0: positive vacuum angular wavenumber.
            material: the homogeneous medium, vacuum by default.
            poltype: "helicity" (default) or "parity".

        ``propagation(distance=..., medium=...)`` takes keywords instead.
        """
        backend = backend_for(r, k0, material)
        if backend is not None:
            return namespace(backend).propagation(
                distance=r,
                basis=basis,
                k0=k0,
                medium=material,
                polarization=resolve_poltype(poltype),
            )
        poltype = resolve_poltype(poltype)
        k0 = check_k0(k0)
        axis = basis.normal_axis
        distance = np.asarray(r, dtype=np.float64)
        if distance.ndim == 0:
            distance = np.eye(3)[axis] * float(distance)
        if distance.shape != (3,):
            raise ValueError("propagation requires a scalar or Cartesian displacement")
        axes = [(axis + 1) % 3, (axis + 2) % 3, axis]
        vectors = np.column_stack(basis.kvecs(k0, material))
        value, _ = diff.propagation_matrix(vectors[:, axes], distance[axes])
        return cls._adopt(
            value, basis=basis, k0=k0, material=Material(material), poltype=poltype
        )

    def _check_adjacent(self, upper: SMatrix) -> None:
        """Check that ``upper`` can sit on the positive side of this S-matrix."""
        check_compatible(
            (self.k0, self._media[0], self.polarization),
            (upper.k0, upper._media[1], upper.polarization),
            "cascaded S-matrices",
        )
        if (
            self.basis.modes != upper.basis.modes
            or self.basis.alignment != upper.basis.alignment
        ):
            raise ValueError("cascaded S-matrices must have the same ports")

    @classmethod
    def stack(cls, items: Sequence[SMatrix]) -> SMatrix:
        """Cascade networks in order from the negative to the positive side.

        Args:
            items: at least one SMatrix; each sits on the positive side of the
                one before, as in ``cascade``.
        """
        backend = backend_for(items)
        if backend is not None:
            return namespace(backend).stack(items)
        if not items:
            raise ValueError("stack requires at least one S matrix")
        result = items[0]
        for item in items[1:]:
            result = result.cascade(item)
        return result

    @classmethod
    def slab(
        cls,
        thickness: ArrayLike,
        basis: PlaneWavePorts,
        k0: float,
        materials: Sequence[MaterialLike],
        poltype: str | None = None,
    ) -> SMatrix:
        """Planar layers between two exterior media.

        Args:
            thickness: one nonnegative thickness per layer (a scalar for one
                layer), in units inverse to k0.
            basis: the plane-wave ports.
            k0: positive vacuum angular wavenumber.
            materials: len(thickness) + 2 media from the negative to the
                positive side: the negative exterior, each layer, then the
                positive exterior.
            poltype: "helicity" (default) or "parity".

        Both polarizations propagate inside the stack, including channels absent
        from the requested ports. The result keeps the selected input and output
        ports. ``slab`` and ``multilayer_slab`` take keywords instead.
        """
        backend = backend_for(thickness, k0, materials)
        if backend is not None:
            if len(materials) < 2:
                raise ValueError("slabs require two exterior materials")
            return namespace(backend).multilayer_slab(
                thicknesses=thickness,
                basis=basis,
                k0=k0,
                negative_medium=materials[0],
                positive_medium=materials[-1],
                materials=materials[1:-1],
                polarization=resolve_poltype(poltype),
            )
        poltype = resolve_poltype(poltype)
        k0 = check_k0(k0)
        values = np.atleast_1d(np.asarray(thickness, dtype=np.float64))
        if (
            len(materials) != len(values) + 2
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError(
                "slabs require nonnegative thicknesses and two exterior materials"
            )
        groups = _port_groups(basis)
        media = [Material(m) for m in materials]
        check_poltype_medium(poltype, *media)
        compact = _layer_blocks(groups, k0, media, values, basis.alignment)
        return cls._adopt(
            _scatter_ports(basis, groups, compact, poltype),
            k0=k0,
            basis=basis,
            material=(media[-1], media[0]),
            poltype=poltype,
        )

    def double(self, n: int = 1) -> SMatrix:
        """Cascade the network with itself n times, giving 2**n copies.

        Args:
            n: nonnegative number of doublings. The outer media must match.
        """
        if n < 0:
            raise ValueError("doubling count must be nonnegative")
        result = self
        for _ in range(n):
            result = result.cascade(result)
        return result

    def _with_polarization_target(self, target: str | None) -> SMatrix:
        """All four blocks in the named convention, or the other one for None."""
        target = target_poltype(self.polarization, target)
        if target is None:
            return self
        return type(self)._adopt(
            change_polarization(self.array, self.basis, (2, 3)),
            basis=self.basis,
            k0=self.k0,
            material=self._media,
            poltype=target,
        )

    def rotate(self, phi: float, theta: float = 0, psi: float = 0) -> SMatrix:
        """Rotate xy plane-wave labels and their periodic metadata around z."""
        if theta != 0:
            raise ValueError("plane rotations require zero theta")
        return self._with_array(self.array, self.basis.rotate(phi + psi))

    def _with_array(
        self, value: NDArray[np.complex128], basis: PlaneWavePorts
    ) -> SMatrix:
        """The same media and convention with a transformed array and ports."""
        return type(self)._adopt(
            value,
            k0=self.k0,
            basis=basis,
            material=self._media,
            poltype=self.polarization,
        )

    def _port_vectors(
        self, basis: PlaneWavePorts, side: int, direction: int
    ) -> NDArray[np.complex128]:
        """Wavevectors (n, 3) of the ports in exterior ``side`` propagating up/down."""
        return np.column_stack(
            basis.kvecs(self.k0, self._media[side], ("up", "down")[direction])
        )

    def translate(self, r: ArrayLike) -> SMatrix:
        """Shift the reference position of every port by r, shape (3,).

        Each block gains diagonal plane-wave phase factors on both sides.
        """
        r = np.asarray(r, dtype=float)
        if r.shape != (3,):
            raise ValueError("S-matrix translation requires one Cartesian displacement")
        # Outgoing block rows i live in medium i, incoming columns j in 1 - j.
        left = np.stack(
            [
                diff.plane_phases(r[None, :], self._port_vectors(self.basis, i, i))[0][
                    0
                ]
                for i in range(2)
            ]
        )
        right = np.stack(
            [
                diff.plane_phases(
                    -r[None, :], self._port_vectors(self.basis, 1 - j, j)
                )[0][0]
                for j in range(2)
            ]
        )
        # Scale one full-size temporary in place; a second temporary measurably
        # slows repeated calls.
        value = left[:, None, :, None] * self.array
        value *= right[None, :, None, :]
        return self._with_array(value, self.basis)

    def permute(self, n: int = 1) -> SMatrix:
        """Cyclic coordinate change of both ports and their polarization frames.

        Each block becomes F_i S_ij G_j with the forward polarization change F_i
        of its outgoing ports and the inverse change G_j of its incoming ports.
        """
        basis = self.basis.permute(n)
        lookup = self.basis._lookup
        # partners[pol, k]: the port with k's direction and polarization pol.
        partners = np.array(
            [
                [lookup.get((x, y, pol), -1) for x, y, _ in self.basis.modes]
                for pol in (0, 1)
            ],
            dtype=np.intp,
        ).reshape(2, -1)
        columns = [np.flatnonzero(partners[pol] >= 0) for pol in (0, 1)]
        forward = [
            diff.plane_permutation(
                self._port_vectors(self.basis, i, i),
                self.basis.pol,
                n,
                poltype=self.polarization,
            )[0]
            for i in range(2)
        ]
        inverse = [
            diff.plane_permutation(
                self._port_vectors(basis, 1 - j, j),
                basis.pol,
                -n,
                poltype=self.polarization,
            )[0]
            for j in range(2)
        ]
        # Ports of one source polarization have distinct directions, so their
        # partners are distinct rows and a fancy-index += accumulates them all;
        # np.add.at over both polarizations at once is several times slower.
        sources = [
            [cols[self.basis.pol[cols] == source] for source in (0, 1)]
            for cols in columns
        ]
        value = np.zeros_like(self.array)
        for i, j in np.ndindex(2, 2):
            left = np.zeros_like(self.array[i, j])
            for pol, groups in enumerate(sources):
                for cols in groups:
                    left[partners[pol, cols]] += (
                        forward[i][pol, cols, None] * self.array[i, j, cols]
                    )
            for pol, cols in enumerate(columns):
                value[i, j][:, cols] += (
                    left[:, partners[pol, cols]] * inverse[j][pol, cols]
                )
        return self._with_array(value, basis)

    def _incident(
        self, illu: ArrayLike | PlaneWave | Wave, modetype: str
    ) -> NDArray[np.complex128]:
        if isinstance(illu, (PlaneWave, Wave)):
            check_compatible(
                (illu.k0, illu.medium, illu.polarization),
                (self.k0, self._media[1 if modetype == "up" else 0], self.polarization),
                "illumination and the S-matrix side it enters",
            )
            if _direction(illu, None, self.basis.normal_axis) != modetype:
                raise ValueError(f"illumination must propagate {modetype}")
            return (
                illu._expanded(self.basis)
                if isinstance(illu, PlaneWave)
                else illu._expanded(self.basis, modetype)
            )
        return np.asarray(illu, dtype=np.complex128)

    def _illuminated(
        self,
        illu: ArrayLike | PlaneWave | Wave,
        illu2: ArrayLike | PlaneWave | Wave | None,
        modetype: str | None,
        smat: SMatrix | None,
    ) -> tuple[NDArray[np.complex128], ...]:
        """Outgoing fields, and optionally internal fields below an adjacent stack.

        With ``smat``, return outgoing up/down and internal up/down coefficients.
        Incident PlaneWave metadata refers to the outer media of the combined pair.
        """
        upper = self if smat is None else smat
        if smat is not None:
            self._check_adjacent(smat)
        modetype = _direction(illu, modetype, self.basis.normal_axis)
        first = (self if modetype == "up" else upper)._incident(illu, modetype)
        second = (
            np.zeros_like(first)
            if illu2 is None
            else (upper if modetype == "up" else self)._incident(
                illu2, "down" if modetype == "up" else "up"
            )
        )
        if (
            first.shape != (len(self.basis),)
            or second.shape != first.shape
            or not np.isfinite(first).all()
            or not np.isfinite(second).all()
        ):
            raise ValueError(
                "illumination requires one finite amplitude per basis mode"
            )
        up, down = (first, second) if modetype == "up" else (second, first)
        if smat is not None:
            fields = _native.smatrix_illuminate_value(
                self.array, smat.array, up[:, None], down[:, None]
            )
            return tuple(fields[:, :, 0])
        return self[0, 0] @ up + self[0, 1] @ down, self[1, 0] @ up + self[1, 1] @ down

    def _bands(
        self, period: float
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Bloch wavenumbers along the basis normal and their right eigenvectors.

        ``period`` is the positive cell length along that normal.
        """
        if self._media[0] != self._media[1]:
            raise ValueError("periodic repetition requires matching outer media")
        return diff.bands(self.array, period)[0]

    def _transmission(
        self, incident: NDArray[np.complex128], direction: str
    ) -> NDArray[np.float64]:
        groups: dict[tuple[float, float], int] = {}
        modes = [
            (groups.setdefault((x, y), len(groups)), p) for x, y, p in self.basis.modes
        ]
        return _native.smatrix_tr_value(
            self.array,
            incident,
            [medium._plane_ks(self.k0).tolist() for medium in self._media],
            [medium.impedance for medium in self._media],
            list(groups),
            modes,
            self.basis.normal_axis,
            self.polarization == "helicity",
            0 if direction == "up" else 1,  # direction: 0 up, 1 down
        )

    def _transmission_balance(
        self, illu: ArrayLike | PlaneWave | Wave, modetype: str | None
    ) -> tuple[float, float]:
        """Transmitted and reflected power fractions of one illumination."""
        direction = _direction(illu, modetype, self.basis.normal_axis)
        incident = self._incident(illu, direction)
        power = self._transmission(incident[:, None], direction)
        return float(power[0, 0]), float(power[1, 0])

    def _circular_dichroism(
        self, illu: ArrayLike | PlaneWave | Wave, modetype: str | None
    ) -> tuple[float, float]:
        """Transmission and total-outgoing-power contrast against opposite polarization.

        These are upstream's two CD formulas: (T_opposite-T)/(T_opposite+T)
        and ((T+R)_opposite-(T+R))/((T+R)_opposite+(T+R)). The second quantity
        is normalized by outgoing power, although upstream calls it absorption CD.
        Helicity bases must contain both polarizations of each direction.
        """
        direction = _direction(illu, modetype, self.basis.normal_axis)
        incident = self._incident(illu, direction)
        if self.polarization == "helicity":
            indices = self.basis._lookup
            try:
                opposite = incident[
                    [indices[(x, y, 1 - p)] for x, y, p in self.basis.modes]
                ]
            except KeyError as error:
                raise ValueError(
                    "CD requires both helicities of each direction"
                ) from error
        else:
            opposite = incident * (2 * self.basis.pol - 1)
        power = self._transmission(np.column_stack((incident, opposite)), direction)
        transmission, reflection = power[:, 0]
        opposite_t, opposite_r = power[:, 1]
        total = transmission + reflection
        opposite_total = opposite_t + opposite_r
        if transmission + opposite_t == 0 or total + opposite_total == 0:
            raise ValueError(
                "CD is undefined for zero summed transmission or outgoing power"
            )
        return (opposite_t - transmission) / (opposite_t + transmission), (
            opposite_total - total
        ) / (opposite_total + total)

    # treams-compatible names (see treams_rs._upstream.UPSTREAM_MEMBERS)

    @property
    def material(self) -> tuple[Material, Material]:
        """treams order: (positive side, negative side); prefer positive_medium/negative_medium."""
        return self._media

    @property
    def poltype(self) -> str:
        """treams name of ``polarization``."""
        return self.polarization

    def changepoltype(self, poltype: str | None = None) -> SMatrix:
        """treams name of ``with_polarization``; omitting poltype switches convention."""
        return self._with_polarization_target(poltype)

    def add(self, upper: SMatrix) -> SMatrix:
        """treams name of ``cascade``."""
        return self.cascade(upper)

    def illuminate(
        self,
        illu: ArrayLike | PlaneWave | Wave,
        illu2: ArrayLike | PlaneWave | Wave | None = None,
        *,
        modetype: str | None = None,
        smat: SMatrix | None = None,
    ) -> tuple[NDArray[np.complex128], ...]:
        """treams name of ``scatter``, returning coefficient arrays.

        With ``smat``, also returns the internal up/down coefficients below that
        adjacent stack.
        """
        return self._illuminated(illu, illu2, modetype, smat)

    def tr(
        self, illu: ArrayLike | PlaneWave | Wave, *, modetype: str | None = None
    ) -> tuple[float, float]:
        """treams name of ``power``, returning (transmission, reflection)."""
        return self._transmission_balance(illu, modetype)

    def cd(
        self, illu: ArrayLike | PlaneWave | Wave, *, modetype: str | None = None
    ) -> tuple[float, float]:
        """treams name of ``circular_dichroism``, returning the two contrasts.

        These are (T_opposite-T)/(T_opposite+T) and
        ((T+R)_opposite-(T+R))/((T+R)_opposite+(T+R)).
        """
        return self._circular_dichroism(illu, modetype)

    periodic = transfer_matrix

    def bands_kz(
        self, az: float
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """treams name of ``bands(period=az)``, returning (kz, eigenvectors).

        kz lies along the basis normal, which is x or y for yz or zx bases.
        """
        return self._bands(az)


def _port_groups(basis: PlaneWavePorts) -> dict[tuple[float, float], list[int]]:
    """Port indices of each distinct transverse wavevector, in first-seen order."""
    groups: dict[tuple[float, float], list[int]] = {}
    for i, (x, y, _) in enumerate(basis):
        groups.setdefault((x, y), []).append(i)
    return groups


def _layer_blocks(
    groups: dict[tuple[float, float], list[int]],
    k0: float,
    media: Sequence[Material],
    thickness: ArrayLike,
    alignment: str,
) -> NDArray[np.complex128]:
    """Helicity blocks (groups, 2, 2, 2, 2) of layers, from one native call."""
    if not groups:
        return np.zeros((0, 2, 2, 2, 2), complex)
    return diff.layer_stack(
        [m._plane_ks(k0) for m in media],
        [m.impedance for m in media],
        list(groups),
        thickness,
        alignment=alignment,
        fixed_q=True,
    )[0]


def _scatter_ports(
    basis: PlaneWavePorts,
    groups: dict[tuple[float, float], list[int]],
    compact: NDArray[np.complex128],
    poltype: str,
) -> NDArray[np.complex128]:
    """Place per-wavevector helicity blocks (groups, 2, 2, 2, 2) at their ports.

    Ports missing one polarization keep the matching rows and columns.
    """
    if poltype == "parity":
        compact = PARITY_CHANGE @ compact @ PARITY_CHANGE.T
    array = np.zeros((2, 2, len(basis), len(basis)), complex)
    for q, indices in enumerate(groups.values()):
        index = np.array(indices)
        pol = basis.pol[index]
        array[:, :, index[:, None], index] = compact[q][:, :, pol[:, None], pol]
    return array


def _direction(
    illu: ArrayLike | PlaneWave | Wave, modetype: str | None, axis: int = 2
) -> str:
    if modetype is None:
        if isinstance(illu, Wave):
            if (
                not isinstance(illu.basis, PlaneWavePorts)
                or illu.basis.normal_axis != axis
            ):
                raise ValueError(
                    "incident wave must use plane ports with the same normal"
                )
            return illu.kind
        if isinstance(illu, PlaneWave):
            return propagation_side(illu.kvecs[0], axis)
        # Raw coefficients carry no direction; they illuminate from below.
        return "up"
    if modetype not in ("up", "down"):
        raise ValueError("modetype must be up or down")
    return modetype


def _side_direction(side: str | None) -> str | None:
    if side is None:
        return None
    if side not in ("negative", "positive"):
        raise ValueError("incident side must be negative or positive")
    return "up" if side == "negative" else "down"


def chirality_density(
    basis: PlaneWavePorts,
    k0: float,
    material: MaterialLike = 1,
    poltype: str | None = None,
    z: ArrayLike = (0.0, 0.0),
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Up/down/coherent-cross forms of 2 Re(E* . i Z H), along the basis normal.

    Equal endpoints evaluate at that plane. z gives coordinates along the normal.
    For amplitudes u,d the density is
    Re(u* U u + d* D d + d* X u). X can be complex for a shifted interval.
    Differs from treams; see Differences from treams: plane-wave chirality
    interval.
    """
    poltype = resolve_poltype(poltype)
    medium = Material(material)
    check_poltype_medium(poltype, medium)
    normal = basis.kvecs(k0, medium)[basis.normal_axis]
    if basis.alignment != "xy":
        q = basis.components
        if poltype == "helicity":
            values, _ = diff.oriented_chirality(
                q, normal, z, polarizations=basis.pol, axis=basis.normal_axis
            )
            up, down, cross = (np.diag(row) for row in values)
        else:
            values, _ = diff.oriented_chirality(
                np.repeat(q, 2, axis=0),
                np.repeat(normal, 2),
                z,
                polarizations=np.tile([0, 1], len(basis)),
                axis=basis.normal_axis,
            )
            values = values.reshape(3, len(basis), 2)
            same = np.all(q[:, None, :] == q[None, :, :], axis=-1)
            sign = 2 * basis.pol - 1
            pair = sign[:, None] * sign
            up, down, cross = (
                0.5 * same * (pair * row[:, 0] + row[:, 1]) for row in values
            )
        return up, down, cross
    values, _ = diff.chirality_density(
        medium.ks(k0)[basis.pol],
        normal,
        z,
    )
    if poltype == "helicity":
        values = values * (2 * basis.pol - 1)
        up, down, cross = (np.diag(row) for row in values)
        return up, down, cross
    same = np.all(basis.components[:, None, :] == basis.components[None, :, :], axis=-1)
    paired = same & (basis.pol[:, None] != basis.pol)
    up, down, cross = (paired * row for row in values)
    return up, down, cross


def poynting_avg_z(
    basis: PlaneWavePorts,
    k0: float,
    material: MaterialLike = 1,
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Upstream-compatible axial forms, with treams normalization and branches.

    These retain treams' parity normalization and complex-medium conventions;
    they are not physical Poynting-flux matrices for arbitrary media. Use
    ``SMatrix.power`` for transmitted/reflected power or sample E and H for
    the local physical flux ``0.5 * real(E cross conj(H))``.
    """
    poltype = resolve_poltype(poltype)
    if basis.alignment != "xy":
        raise ValueError("axial power forms require xy-aligned plane bases")
    medium = Material(material)
    kx, ky = basis.components.T
    # This compatibility helper keeps upstream branches, including materials
    # unsupported by the physical plane-wave constructors.
    kz = medium.kzs(k0, kx, ky, basis.pol)
    gamma = kz / (medium.ks(k0)[basis.pol] * medium.impedance)
    selection = (kx[:, None] == kx) & (ky[:, None] == ky)
    pol = basis.pol
    if poltype == "parity" and not medium.ischiral:
        selection &= pol[:, None] == pol
        return selection * (
            (1 - pol) * gamma.conj() + pol * gamma
        ) * 0.25, selection * ((1 - pol) * gamma.conj() - pol * gamma) * 0.25
    if poltype == "helicity":
        signs = 2 * pol - 1
        return selection * (
            signs[:, None] * signs * gamma[:, None].conj() + gamma
        ) * 0.25, selection * (
            signs[:, None] * signs * gamma[:, None].conj() - gamma
        ) * 0.25
    raise ValueError("the parity convention requires an achiral medium")


def interface(
    *,
    basis: PlaneWavePorts,
    k0: float,
    negative_medium: MaterialLike,
    positive_medium: MaterialLike,
    polarization: str = "helicity",
) -> SMatrix:
    """Planar interface between the negative and the positive medium.

    Args:
        basis: the plane-wave ports; ``basis.normal_axis`` is the normal.
        k0: positive vacuum angular wavenumber, in inverse length units.
        negative_medium: the medium on the negative side (below).
        positive_medium: the medium on the positive side (above).
        polarization: "helicity" (default) or "parity"; parity needs achiral
            media.
    """
    return SMatrix.interface(
        basis, k0, [negative_medium, positive_medium], polarization
    )


def slab(
    *,
    basis: PlaneWavePorts,
    k0: float,
    thickness: float,
    material: MaterialLike,
    negative_medium: MaterialLike = 1,
    positive_medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> SMatrix:
    """One homogeneous layer between explicitly named exterior media.

    ``basis=PlaneWavePorts.default([0, 0])`` supplies normal-incidence channels.
    k0 is 2*pi/vacuum_wavelength; thickness uses the matching length unit.
    material is the layer permittivity or Material(epsilon, mu, kappa).
    Exterior media default to vacuum. Pass a physical plane wave to
    ``network.power(incident)`` for named transmission, reflection and absorption
    fractions; its direction selects the illuminated side. For derivatives use
    this same function inside your autodiff framework's gradient calculation;
    the thickness, frequency or material selects its adapter automatically.

    A lossy chiral layer distinguishes the two helicities::

        import numpy as np
        import treams_rs as tr

        k0 = 2*np.pi/0.8
        ports = tr.PlaneWavePorts.default([0, 0])
        network = tr.slab(basis=ports, k0=k0, thickness=0.17,
                          material=tr.Material(2.3+0.08j, 1, 0.2+0.01j))
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
        power = network.power(incident)
        # Explicit amplitudes use the actual channel labels, not an assumed order.
        positive = (ports.pol == 1).astype(complex)
        np.testing.assert_allclose(power.transmission,
                                   network.power(positive).transmission)
        negative = tr.plane_wave([0, 0, 1], "negative_helicity", k0=k0)
        assert abs(power.transmission - network.power(negative).transmission) > 0.01
    """
    return multilayer_slab(
        basis=basis,
        k0=k0,
        thicknesses=[thickness],
        materials=[material],
        negative_medium=negative_medium,
        positive_medium=positive_medium,
        polarization=polarization,
    )


def multilayer_slab(
    *,
    basis: PlaneWavePorts,
    k0: float,
    thicknesses: ArrayLike,
    materials: Sequence[MaterialLike],
    negative_medium: MaterialLike = 1,
    positive_medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> SMatrix:
    """Planar layers ordered from the negative to the positive side.

    Args:
        basis: the plane-wave ports; ``basis.normal_axis`` is the normal.
        k0: positive vacuum angular wavenumber, in inverse length units.
        thicknesses: one nonnegative thickness per layer, in units inverse to k0.
        materials: one medium per layer, in the same order.
        negative_medium: exterior medium on the negative side, vacuum by default.
        positive_medium: exterior medium on the positive side, vacuum by default.
        polarization: "helicity" (default) or "parity".
    """
    return SMatrix.slab(
        thicknesses,
        basis,
        k0,
        [negative_medium, *materials, positive_medium],
        polarization,
    )


def propagation(
    *,
    distance: ArrayLike,
    basis: PlaneWavePorts,
    k0: float,
    medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> SMatrix:
    """Homogeneous propagation by a normal distance or Cartesian displacement.

    Args:
        distance: distance along the basis normal, or a Cartesian displacement
            of shape (3,), in units inverse to k0.
        basis: the plane-wave ports.
        k0: positive vacuum angular wavenumber.
        medium: the homogeneous medium, vacuum by default.
        polarization: "helicity" (default) or "parity".
    """
    return SMatrix.propagation(distance, basis, k0, medium, polarization)


def stack(layers: Sequence[SMatrix]) -> SMatrix:
    """Cascade layers in order from negative to positive side of their normal."""
    return SMatrix.stack(layers)
