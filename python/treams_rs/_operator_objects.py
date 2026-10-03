"""Operator classes and OperatorAttribute, built on the functions of ``_operators``."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

import numpy as np

from ._bases import CylindricalBasis, PlaneWavePorts, SphericalBasis
from ._fields import WaveFields, field_samples
from ._operators import (
    bfield,
    changepoltype,
    dfield,
    efield,
    expand,
    expandlattice,
    ffield,
    gfield,
    hfield,
    permute,
    rotate,
    translate,
)
from ._polarization import OPPOSITE_POLTYPE, resolve_poltype

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

    from ._bases import FieldBasis
    from ._lattice import Lattice, WaveVector

__all__ = [
    "BField",
    "ChangePoltype",
    "DField",
    "EField",
    "Expand",
    "ExpandLattice",
    "FField",
    "FieldOperator",
    "GField",
    "HField",
    "Operator",
    "OperatorAttribute",
    "Permute",
    "Rotate",
    "Translate",
]


class Operator:
    """Reusable operator: stores its arguments and builds its matrix on demand.

    Mirrors ``treams.Operator``. Calling the operator with the keyword-only
    arguments of its function returns the matrix as a plain array.
    ``op @ obj`` reads those arguments from the attributes of ``obj`` with the
    same names (``basis``, ``k0``, ``material``, ``poltype``, ``modetype``, ...)
    and multiplies.
    """

    __array_priority__ = 1.0
    _FUNC: ClassVar[Callable[..., NDArray[np.generic]]]
    _metadata: ClassVar[tuple[str, ...]] = ()
    _parameters: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(
        cls, *, func: Callable[..., NDArray[np.generic]] | None = None
    ) -> None:
        super().__init_subclass__()
        if func is not None:
            cls._FUNC = func
        cls._parameters = tuple(
            name
            for name in inspect.signature(cls.__init__).parameters
            if name != "self"
        )
        if hasattr(cls, "_FUNC"):
            cls._metadata = tuple(
                name
                for name, parameter in inspect.signature(cls._FUNC).parameters.items()
                if parameter.kind == inspect.Parameter.KEYWORD_ONLY
            )

    def __init__(self, *args: Any, isinv: bool = False) -> None:
        self._args = args
        self._isinv = bool(isinv)

    @property
    def isinv(self) -> bool:
        """Whether this operator applies the inverse of its function."""
        return self._isinv

    @property
    def inv(self) -> Self:
        """The inverse operator: the same arguments with ``isinv`` toggled."""
        return type(self)(*self._args, isinv=not self.isinv)

    @property
    def FUNC(self) -> Callable[..., NDArray[np.generic]]:  # noqa: N802 - upstream operator interface
        """The function that builds the matrix, for example ``operators.rotate``."""
        return type(self)._FUNC

    def __call__(self, **kwargs: Any) -> NDArray[np.generic]:
        if self.isinv:
            return self._call_inv(**kwargs)
        return self.FUNC(*self._args, **kwargs)

    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        raise NotImplementedError("this operator has no inverse")

    def get_kwargs(self, obj: Any, dim: int = -1) -> dict[str, Any]:
        """Keyword arguments of ``FUNC`` read from the attributes of ``obj``.

        A tuple attribute holds one value per array axis; ``dim`` picks the axis.
        Missing and None attributes are left out.
        """
        # The names are the treams PhysicsArray ones (material, poltype,
        # modetype); physics objects such as TMatrix provide them as aliases.
        values = {}
        for name in self._metadata:
            value = getattr(obj, name, None)
            if isinstance(value, tuple):
                value = value[dim]
            if value is not None:
                values[name] = value
        return values

    def __matmul__(self, other: Any) -> NDArray[np.generic]:
        if isinstance(other, Operator):
            raise NotImplementedError(
                "evaluate operators in explicit bases before composing"
            )
        return self(
            **self.get_kwargs(other, -2 if np.ndim(other) > 1 else -1)
        ) @ np.asarray(other)

    def __rmatmul__(self, other: Any) -> NDArray[np.generic]:
        return np.asarray(other) @ self(**self.get_kwargs(other))

    def __repr__(self) -> str:
        arguments = ", ".join(repr(value) for value in self._args)
        return (
            f"{type(self).__name__}({arguments}{', isinv=True' if self.isinv else ''})"
        )


def _reverse_basis(kwargs: dict[str, Any]) -> None:
    if isinstance(kwargs.get("basis"), tuple):
        kwargs["basis"] = kwargs["basis"][::-1]


class Rotate(Operator, func=rotate):
    """Euler rotation with an exact inverse obtained by reversing the angles."""

    def __init__(
        self, phi: float, theta: float = 0, psi: float = 0, *, isinv: bool = False
    ) -> None:
        super().__init__(phi, theta, psi, isinv=isinv)

    @override
    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        _reverse_basis(kwargs)
        return rotate(*(-value for value in self._args[::-1]), **kwargs)


class Translate(Operator, func=translate):
    """Translation with the inverse displacement in the reversed basis pair."""

    def __init__(self, r: ArrayLike, *, isinv: bool = False) -> None:
        super().__init__(np.array(r, dtype=np.float64, copy=True), isinv=isinv)

    @override
    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        _reverse_basis(kwargs)
        return translate(-self._args[0], **kwargs)


class ChangePoltype(Operator, func=changepoltype):
    """Helicity/parity conversion; omission selects the opposite input convention."""

    def __init__(
        self, poltype: str | tuple[str, str] | None = None, *, isinv: bool = False
    ) -> None:
        super().__init__(*(() if poltype is None else (poltype,)), isinv=isinv)

    @override
    def get_kwargs(self, obj: Any, dim: int = -1) -> dict[str, Any]:
        """As ``Operator.get_kwargs``; without a stored poltype, switch the object's."""
        kwargs = super().get_kwargs(obj, dim)
        if not self._args:
            poltype = getattr(obj, "poltype", None)
            if isinstance(poltype, tuple):
                poltype = poltype[dim]
            if poltype is not None:
                kwargs["poltype"] = OPPOSITE_POLTYPE[resolve_poltype(poltype)]
        return kwargs

    @override
    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        _reverse_basis(kwargs)
        if self._args and isinstance(self._args[0], tuple):
            destination = self._args[0][::-1]
        else:
            poltype = self._args[0] if self._args else kwargs.pop("poltype", None)
            destination = OPPOSITE_POLTYPE[resolve_poltype(poltype)]
        return changepoltype(destination, **kwargs)


class Expand(Operator, func=expand):
    """Expand into a destination basis, obtaining the source from keyword metadata."""

    def __init__(
        self,
        basis: FieldBasis | tuple[FieldBasis, FieldBasis],
        modetype: str | tuple[str, str] | None = None,
        *,
        isinv: bool = False,
    ) -> None:
        super().__init__(basis, modetype, isinv=isinv)

    def __call__(self, **kwargs: Any) -> NDArray[np.generic]:
        basis, modetype = self._args
        source = kwargs.pop("basis", None)
        if source is not None and not isinstance(basis, tuple):
            basis = (basis, source)
        source_type = kwargs.pop("modetype", None)
        if modetype is None:
            modetype = source_type
        elif source_type is not None and not isinstance(modetype, tuple):
            modetype = (modetype, source_type)
        if self.isinv:
            basis = basis[::-1] if isinstance(basis, tuple) else basis
            modetype = modetype[::-1] if isinstance(modetype, tuple) else modetype
        return expand(basis, modetype, **kwargs)

    @override
    def get_kwargs(self, obj: Any, dim: int = -1) -> dict[str, Any]:
        """As ``Operator.get_kwargs``, plus the source ``basis`` and ``modetype``."""
        kwargs = super().get_kwargs(obj, dim)
        for name in ("basis", "modetype"):
            value = getattr(obj, name, None)
            if isinstance(value, tuple):
                value = value[dim]
            if value is not None:
                kwargs[name] = value
        return kwargs


class ExpandLattice(Operator, func=expandlattice):
    """Periodic expansion into an optional destination basis."""

    def __init__(
        self,
        lattice: ArrayLike | Lattice | None = None,
        kpar: ArrayLike | WaveVector | None = None,
        basis: FieldBasis | None = None,
        modetype: str | None = None,
        eta: complex = 0,
    ) -> None:
        super().__init__(lattice, kpar, basis, modetype, eta)

    @property
    @override
    def inv(self) -> Self:
        """Raises NotImplementedError: a periodic image sum has no inverse."""
        raise NotImplementedError("a periodic image sum has no inverse")

    def __call__(self, **kwargs: Any) -> NDArray[np.generic]:
        lattice, kpar, basis, modetype, eta = self._args
        if "basis" in kwargs:
            source = kwargs.pop("basis")
            basis = source if basis is None else (basis, source)
        inherited_lattice, inherited_kpar = (
            kwargs.pop("lattice", None),
            kwargs.pop("kpar", None),
        )
        lattice = inherited_lattice if lattice is None else lattice
        kpar = inherited_kpar if kpar is None else kpar
        source_type = kwargs.pop("modetype", None)
        if source_type is not None and not isinstance(modetype, tuple):
            destination, source = basis if isinstance(basis, tuple) else (basis, basis)
            if modetype is None:
                modetype = (
                    "up"
                    if isinstance(destination, PlaneWavePorts)
                    else "singular"
                    if isinstance(destination, CylindricalBasis)
                    and isinstance(source, SphericalBasis)
                    else "regular"
                )
            modetype = (modetype, source_type)
        eta = kwargs.pop("eta", eta)
        return expandlattice(
            lattice, kpar, basis=basis, modetype=modetype, eta=eta, **kwargs
        )

    @override
    def get_kwargs(self, obj: Any, dim: int = -1) -> dict[str, Any]:
        """As ``Operator.get_kwargs``, plus the object's ``lattice`` and ``kpar``."""
        kwargs = super().get_kwargs(obj, dim)
        for name in ("lattice", "kpar"):
            value = getattr(obj, name, None)
            if value is not None:
                kwargs[name] = value
        return kwargs


class Permute(Operator, func=permute):
    """Cyclic plane-wave coordinate permutation."""

    def __init__(self, n: int = 1, *, isinv: bool = False) -> None:
        super().__init__(n, isinv=isinv)

    @override
    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        kwargs["basis"] = kwargs["basis"].permute(self._args[0])
        return permute(-self._args[0], **kwargs)


class FieldOperator(Operator):
    """Cartesian field evaluation; no inverse from sampled fields is assumed."""

    _KIND: ClassVar[str]

    def __init__(self, r: ArrayLike) -> None:
        super().__init__(np.array(r, dtype=np.float64, copy=True))

    @property
    @override
    def inv(self) -> Self:
        """Raises NotImplementedError: sampled fields have no general inverse."""
        raise NotImplementedError("field evaluation has no general inverse")

    @override
    def __matmul__(self, other: Any) -> NDArray[np.generic]:
        # Existing wave objects evaluate weighted fields directly, without a
        # sample-by-mode matrix. Ordinary coefficient matrices use the operator.
        # PhysicsArray binds these operators, so it is imported at call time.
        from ._array import PhysicsArray

        if isinstance(other, WaveFields):
            return getattr(other, self.FUNC.__name__)(*self._args)
        if isinstance(other, PhysicsArray) and other.ndim == 1:
            return field_samples(
                self._KIND,
                self._args[-1],
                other.array,
                pol=self._args[0] if len(self._args) == 2 else 0,
                **self.get_kwargs(other),
            )
        return super().__matmul__(other)


class EField(FieldOperator, func=efield):
    """Electric field evaluation."""

    _KIND = "E"


class HField(FieldOperator, func=hfield):
    """Magnetic field evaluation."""

    _KIND = "H"


class DField(FieldOperator, func=dfield):
    """Displacement field evaluation."""

    _KIND = "D"


class BField(FieldOperator, func=bfield):
    """Magnetic flux density evaluation."""

    _KIND = "B"


class GField(FieldOperator, func=gfield):
    """Riemann-Silberstein field G, with explicit helicity and sample points."""

    _KIND = "G"

    def __init__(self, pol: int, r: ArrayLike) -> None:
        Operator.__init__(self, pol, np.array(r, dtype=np.float64, copy=True))


class FField(GField, func=ffield):
    """Riemann-Silberstein field F, with explicit helicity and sample points."""

    _KIND = "F"


class OperatorAttribute:
    """Operator bound to the object it is read from, such as ``PhysicsArray.rotate``.

    Mirrors ``treams.OperatorAttribute``. Calling the attribute transforms the
    object: ``a.rotate(phi)`` returns ``R @ a`` for a vector and ``R @ a @ inv(R)``
    for a matrix, as a plain array. Unlike treams' OperatorAttribute, binding
    creates a new object, so a saved bound attribute keeps its object.
    """

    def __init__(self, op: type[Operator], obj: Any = None) -> None:
        self._op, self._obj = op, obj

    @property
    def OP(self) -> type[Operator]:  # noqa: N802 - upstream operator interface
        """The operator class, for example ``Rotate``."""
        return self._op

    def __get__(self, obj: Any, objtype: type | None = None) -> Self:
        return self if obj is None else type(self)(self.OP, obj)

    def _prepare(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], inverse: bool
    ) -> tuple[Operator, dict[str, Any]]:
        constructor = {
            key: kwargs.pop(key) for key in self.OP._parameters if key in kwargs
        }
        op = self.OP(*args, **constructor)
        if inverse:
            op = op.inv
        dim = -1 if inverse or np.ndim(self._obj) == 1 else -2
        inherited = op.get_kwargs(self._obj, dim)
        if "basis" in kwargs and "basis" in inherited:
            kwargs["basis"] = (kwargs["basis"], inherited["basis"])
        return op, inherited | kwargs

    def eval(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        """Matrix of the operator for the bound object, as a plain array.

        The arguments are those of the operator class and of its function; the
        object supplies the remaining keyword arguments.
        """
        op, metadata = self._prepare(args, kwargs, False)
        return op(**metadata)

    def eval_inv(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        """Matrix of the inverse operator for the bound object."""
        op, metadata = self._prepare(args, kwargs, True)
        return op(**metadata)

    def apply_left(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        """The operator matrix times the bound object: ``eval(...) @ obj``."""
        op, metadata = self._prepare(args, kwargs, False)
        if not kwargs and isinstance(op, FieldOperator):
            return op @ self._obj
        return op(**metadata) @ np.asarray(self._obj)

    def apply_right(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        """The bound object times the inverse matrix: ``obj @ eval_inv(...)``."""
        return np.asarray(self._obj) @ self.eval_inv(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        if np.ndim(self._obj) <= 1:
            return self.apply_left(*args, **kwargs)
        try:
            inverse = self.eval_inv(*args, **kwargs)
        except NotImplementedError:
            return self.apply_left(*args, **kwargs)
        return self.eval(*args, **kwargs) @ np.asarray(self._obj) @ inverse

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.OP.__name__})"
