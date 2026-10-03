"""PhysicsArray: a read-only array with the metadata that the operators read."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from . import _operator_objects as op
from ._bases import (
    CylindricalBasis,
    PlaneWaveBasis,
    PlaneWavePorts,
    SphericalBasis,
)
from ._material import Material
from ._polarization import DEFAULT_POLTYPE, check_poltype_medium, resolve_poltype
from ._validation import check_k0, frozen

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._bases import FieldBasis
    from ._lattice import Lattice, WaveVector
    from ._material import MaterialLike

__all__ = ["PhysicsArray"]


class PhysicsArray(NDArrayOperatorsMixin):
    """Read-only array with the metadata that the operators read.

    Mirrors ``treams.PhysicsArray``. Unlike treams.PhysicsArray this is not an
    ndarray subclass; arithmetic and slicing return plain arrays, so no
    result carries metadata that no longer fits it. Use ``.array`` for NumPy
    work and the bound operators (``.rotate``, ``.expand``, ``.efield``, ...)
    for physics; they also return plain arrays.

    A scalar metadata value describes every axis; a tuple gives one value per
    axis, from left to right. ScatteringBlock builds on this class.

    Args:
        arr: The values; stored as an owned, read-only complex128 copy.
        basis: Basis of the axes.
        k0: Vacuum angular wavenumber.
        material: Medium, vacuum by default.
        poltype: Polarization convention, ``"helicity"`` by default.
        modetype: Kind of the waves: ``"regular"``, ``"singular"``, ``"up"``
            or ``"down"``.
        lattice: Lattice of a periodic arrangement.
        kpar: Bloch vector of a periodic arrangement.
    """

    changepoltype = op.OperatorAttribute(op.ChangePoltype)
    rotate = op.OperatorAttribute(op.Rotate)
    translate = op.OperatorAttribute(op.Translate)
    expand = op.OperatorAttribute(op.Expand)
    expandlattice = op.OperatorAttribute(op.ExpandLattice)
    permute = op.OperatorAttribute(op.Permute)
    efield = op.OperatorAttribute(op.EField)
    hfield = op.OperatorAttribute(op.HField)
    dfield = op.OperatorAttribute(op.DField)
    bfield = op.OperatorAttribute(op.BField)
    gfield = op.OperatorAttribute(op.GField)
    ffield = op.OperatorAttribute(op.FField)

    def __init__(
        self,
        arr: ArrayLike,
        *,
        basis: FieldBasis | tuple[FieldBasis | None, ...] | None = None,
        k0: float | tuple[float | None, ...] | None = None,
        material: MaterialLike | tuple[MaterialLike | None, ...] = 1,
        poltype: str | tuple[str | None, ...] | None = None,
        modetype: str | tuple[str | None, ...] | None = None,
        lattice: Lattice | None = None,
        kpar: WaveVector | None = None,
    ) -> None:
        poltype = DEFAULT_POLTYPE if poltype is None else poltype
        self.array: NDArray[np.complex128] = frozen(arr, np.complex128)
        self.basis, self.k0, self.poltype, self.modetype = basis, k0, poltype, modetype
        self.material = (
            tuple(Material(item) if item is not None else None for item in material)
            if isinstance(material, tuple)
            else Material(material)
        )
        self.lattice, self.kpar = lattice, kpar
        self._check()

    def _check(self) -> None:
        for dim in range(-min(self.ndim, 2), 0):
            values = {}
            for name in ("basis", "k0", "material", "poltype", "modetype"):
                value = getattr(self, name)
                if isinstance(value, tuple):
                    if len(value) != self.ndim:
                        raise ValueError(f"{name} requires one value per array axis")
                    value = value[dim]
                values[name] = value
            basis, k0, medium, poltype = (
                values[name] for name in ("basis", "k0", "material", "poltype")
            )
            if basis is not None and not isinstance(
                basis,
                (
                    SphericalBasis,
                    CylindricalBasis,
                    PlaneWavePorts,
                    PlaneWaveBasis,
                ),
            ):
                raise TypeError("basis must be a wave basis or a tuple of axis bases")
            if basis is not None and len(basis) != self.shape[dim]:
                raise ValueError("basis dimension does not match array axis")
            if k0 is not None:
                check_k0(k0)
            # __init__ only fills in the default; check explicit and per-axis values.
            if poltype is not None:
                resolve_poltype(poltype)
            check_poltype_medium(poltype, medium)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of ``array``."""
        return self.array.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of ``array``."""
        return self.array.ndim

    def __len__(self) -> int:
        return len(self.array)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    def __getitem__(self, key: Any) -> Any:
        return self.array[key]

    @override
    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, op.Operator):
            return NotImplemented
        return self.array @ np.asarray(other)

    @override
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        if any(isinstance(value, op.Operator) for value in inputs):
            return NotImplemented
        inputs = tuple(
            value.array if isinstance(value, PhysicsArray) else value
            for value in inputs
        )
        if "out" in kwargs:
            kwargs["out"] = tuple(
                value.array if isinstance(value, PhysicsArray) else value
                for value in kwargs["out"]
            )
        return getattr(ufunc, method)(*inputs, **kwargs)
