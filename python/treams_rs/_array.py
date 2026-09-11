"""Explicit numerical arrays carrying optional physical metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from . import _operators as op
from ._core import (
    CylindricalWaveBasis,
    Material,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)
from .config import _resolve_poltype

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._core import MaterialLike
    from ._lattice import Lattice, WaveVector
    from ._operators import FieldBasis


class PhysicsArray(NDArrayOperatorsMixin):
    """Owned array and metadata for explicit operator evaluation.

    Scalars describe all channel axes; tuples describe axes from left to right.
    Array arithmetic and slicing return NumPy values, without silently assigning
    physical metadata to a result. Use .array for arbitrary NumPy operations and
    .rotate.eval(...), .expand(...), or the other bound operators for physics.
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
        poltype = _resolve_poltype(None) if poltype is None else poltype
        self.array: NDArray[np.complex128] = np.array(
            arr, dtype=np.complex128, copy=True
        )
        self.basis, self.k0, self.poltype, self.modetype = basis, k0, poltype, modetype
        self.material = (
            tuple(Material(item) if item is not None else None for item in material)
            if isinstance(material, tuple)
            else Material(material)
        )
        self.lattice, self.kpar = lattice, kpar
        self._check()
        self.array.flags.writeable = False

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
                    SphericalWaveBasis,
                    CylindricalWaveBasis,
                    PlaneWaveBasisByComp,
                    PlaneWaveBasisByUnitVector,
                ),
            ):
                raise TypeError("basis must be a wave basis or a tuple of axis bases")
            if basis is not None and len(basis) != self.shape[dim]:
                raise ValueError("basis dimension does not match array axis")
            if k0 is not None and (not np.isfinite(k0) or k0 <= 0):
                raise ValueError("k0 must be finite and positive")
            if poltype is not None and poltype not in ("helicity", "parity"):
                raise ValueError("polarization type must be helicity or parity")
            if poltype == "parity" and medium is not None and medium.ischiral:
                raise ValueError(
                    "parity polarization is not permitted in a chiral medium"
                )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
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
