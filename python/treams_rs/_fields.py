"""Field sampling shared by operator builders and wave objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import diff
from ._bases import (
    PlaneWaveBasis,
    PlaneWavePorts,
    SphericalBasis,
)
from ._dispatch import autodiff_method
from ._material import Material, as_material
from ._polarization import check_poltype_medium, resolve_poltype
from ._validation import check_k0

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._bases import CylindricalBasis
    from ._material import MaterialLike

    type FieldBasis = (
        SphericalBasis | CylindricalBasis | PlaneWavePorts | PlaneWaveBasis
    )

__all__ = [
    "BATCH_OPERATOR_ENTRIES",
    "WaveFields",
    "field",
    "field_points",
    "field_samples",
    "riemann",
    "rs_weights",
]


def field_points(r: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(r, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("require Cartesian field points (..., 3)")
    return points


def field(
    quantity: str,
    r: ArrayLike,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike,
    modetype: str | None,
    poltype: str,
    coefficients: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Field ``quantity`` ("E", "H", "D" or "B") as an operator or samples.

    Returns the operator (..., 3, modes), or the samples (..., 3) of
    ``coefficients``.
    """
    medium = as_material(material)
    points = field_points(r)
    check_k0(k0)
    plane = isinstance(basis, (PlaneWavePorts, PlaneWaveBasis))
    modetype = ("up" if plane else "regular") if modetype is None else modetype
    if not isinstance(basis, PlaneWaveBasis) and modetype not in (
        ("up", "down") if plane else ("regular", "singular")
    ):
        raise ValueError("invalid field mode type for this basis")
    if poltype not in ("helicity", "parity"):
        raise ValueError("polarization type must be helicity or parity")
    check_poltype_medium(poltype, medium)
    weights = np.ones(len(basis), dtype=np.complex128)
    if quantity in ("H", "B"):
        weights *= -1j / medium.impedance
        if poltype == "helicity":
            weights *= 2 * basis.pol - 1
        else:
            if isinstance(basis, PlaneWaveBasis):
                basis = type(basis)([(*mode[:3], 1 - mode[3]) for mode in basis.modes])
            else:
                modes = [(*mode[:-1], 1 - mode[-1]) for mode in basis.modes]
                basis = (
                    type(basis)(modes, basis.alignment)
                    if isinstance(basis, PlaneWavePorts)
                    else type(basis)(modes, basis.positions)
                )
    if quantity == "D":
        weights *= (
            medium.epsilon + (2 * basis.pol - 1) * medium.kappa / medium.impedance
            if poltype == "helicity"
            else medium.epsilon
        )
    if quantity == "B":
        weights *= (
            medium.mu + (2 * basis.pol - 1) * medium.kappa * medium.impedance
            if poltype == "helicity"
            else medium.mu
        )
    weighted = None
    if coefficients is not None:
        amplitudes = np.asarray(coefficients, dtype=np.complex128)
        if amplitudes.shape != (len(basis),):
            raise ValueError("field requires one amplitude per basis mode")
        weighted = amplitudes * weights
    if isinstance(basis, (PlaneWavePorts, PlaneWaveBasis)):
        value, _ = diff.plane_field(
            weighted,
            points.reshape(-1, 3),
            np.column_stack(basis.kvecs(k0, medium, modetype)),
            basis.pol,
            poltype=poltype,
            fixed_vectors=True,
        )
    elif weighted is None:
        value, _ = diff.field_operator(
            points.reshape(-1, 3),
            basis,
            medium.ks(k0),
            poltype=poltype,
            singular=modetype == "singular",
        )
    else:
        value, _ = diff.field(
            weighted,
            points.reshape(-1, 3),
            basis,
            medium.ks(k0),
            poltype=poltype,
            singular=modetype == "singular",
        )
    if weighted is not None:
        return value.reshape(points.shape)
    return value.reshape((*points.shape[:-1], 3, len(basis))) * weights


def rs_weights(
    pol: int,
    basis: FieldBasis,
    poltype: str,
    medium: Material | None = None,
    quantity: str = "G",
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], float]:
    """Per-mode electric weights and the magnetic scalar of the G or F field.

    Riemann-Silberstein fields combine ``E * electric + i Z H * magnetic``.
    F additionally weights helicity modes by ``nmp[pol] / n`` of ``medium``.
    """
    # advect.gfield and advect.ffield reuse these weights. _framework_waves._Fields
    # and advect.hfield have their own expressions for framework arrays, because
    # these NumPy expressions must stay bitwise identical.
    if pol not in (-1, 0, 1):
        raise ValueError("Riemann-Silberstein polarization must be -1, 0 or 1")
    pol = max(pol, 0)
    # Preserve upstream's different spherical and cylindrical/plane scalings.
    normalization = np.sqrt(2) if isinstance(basis, SphericalBasis) else 1.0
    if poltype == "helicity":
        electric = normalization * (basis.pol == pol)
        if quantity == "F":
            if medium is None:
                raise ValueError("the F field requires the embedding medium")
            return electric * (medium.nmp[basis.pol] / medium.n), 0.0
        return electric, 0.0
    if poltype == "parity":
        return np.full(len(basis), normalization), normalization * (2 * pol - 1)
    # Callers resolve poltype first; this guards direct calls to this helper.
    raise ValueError("polarization type must be helicity or parity")


def riemann(
    quantity: str,
    r: ArrayLike,
    basis: FieldBasis,
    k0: float,
    medium: Material,
    modetype: str | None,
    poltype: str,
    pol: int,
    coefficients: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128]:
    """G or F operator (..., 3, modes), or samples (..., 3) of one amplitude vector."""
    electric, magnetic = rs_weights(pol, basis, poltype, medium, quantity)
    if coefficients is None:
        value = field("E", r, basis, k0, medium, modetype, poltype) * electric
    else:
        value = field(
            "E", r, basis, k0, medium, modetype, poltype, coefficients * electric
        )
    if magnetic:
        value += (
            1j
            * medium.impedance
            * magnetic
            * field("H", r, basis, k0, medium, modetype, poltype, coefficients)
        )
    return value


class WaveFields:
    """Shared field sampling for plane and multipole amplitudes."""

    if TYPE_CHECKING:

        @property
        def array(self) -> NDArray[np.complex128]: ...
        @property
        def basis(self) -> FieldBasis: ...

        k0: float
        medium: Material
        kind: str
        polarization: str

    def _samples(
        self, quantity: str, r: ArrayLike, pol: int = 0
    ) -> NDArray[np.complex128]:
        return field_samples(
            quantity,
            r,
            self.array,
            basis=self.basis,
            k0=self.k0,
            material=self.medium,
            modetype=self.kind,
            poltype=self.polarization,
            pol=pol,
        )

    @autodiff_method
    def efield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian electric samples (..., 3), or (..., 3, illuminations) for a batch."""
        return self._samples("E", r)

    @autodiff_method
    def hfield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian magnetic samples (..., 3), or (..., 3, illuminations) for a batch."""
        return self._samples("H", r)

    @autodiff_method
    def dfield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian displacement samples (..., 3), or (..., 3, illuminations)."""
        return self._samples("D", r)

    @autodiff_method
    def bfield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian flux-density samples (..., 3), or (..., 3, illuminations)."""
        return self._samples("B", r)

    @autodiff_method
    def gfield(self, pol: int, r: ArrayLike) -> NDArray[np.complex128]:
        """Riemann-Silberstein G samples (..., 3[, illuminations]), upstream scaling."""
        return self._samples("G", r, pol)

    @autodiff_method
    def ffield(self, pol: int, r: ArrayLike) -> NDArray[np.complex128]:
        """Riemann-Silberstein F samples (..., 3[, illuminations]) with chiral weights."""
        return self._samples("F", r, pol)


# Complex entries of one sampled field operator in a coefficient-batch product.
BATCH_OPERATOR_ENTRIES = 2**22


def field_samples(
    quantity: str,
    r: ArrayLike,
    coefficients: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
    pol: int = 0,
) -> NDArray[np.complex128]:
    """Samples (..., 3) of coefficients (modes,) or (..., 3, B) of (modes, B).

    ``quantity`` is the field: "E", "H", "D", "B", or the Riemann-Silberstein
    fields "G" and "F".

    One vector uses the native weighted kernel, without a sample-by-mode
    operator. A batch evaluates the operator once per chunk of points and
    multiplies it by all columns, instead of one weighted kernel per column.
    """
    poltype = resolve_poltype(poltype)
    medium = as_material(material)
    amplitudes = np.asarray(coefficients, dtype=np.complex128)

    def sample(
        points: ArrayLike, values: NDArray[np.complex128] | None
    ) -> NDArray[np.complex128]:
        if quantity in ("G", "F"):
            return riemann(
                quantity, points, basis, k0, medium, modetype, poltype, pol, values
            )
        return field(quantity, points, basis, k0, medium, modetype, poltype, values)

    if amplitudes.ndim != 2:
        return sample(r, amplitudes)
    points = field_points(r)
    flat = points.reshape(-1, 3)
    chunk = max(1, BATCH_OPERATOR_ENTRIES // (3 * max(len(basis), 1)))
    value = np.concatenate(
        [
            sample(flat[start : start + chunk], None) @ amplitudes
            for start in range(0, max(len(flat), 1), chunk)
        ]
    )
    return value.reshape(*points.shape[:-1], 3, amplitudes.shape[1])
