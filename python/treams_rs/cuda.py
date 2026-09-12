"""Optional complex128 CUDA execution with explicit device ownership.

Build with ``maturin develop --release --features cuda``. The default wheel
contains no CUDA code or dependencies. No GPU work or driver import occurs
until constructing :class:`Device`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def compiled() -> bool:
    """Whether this extension was built with the optional CUDA feature."""
    return hasattr(_native, "CudaDevice")


class Device:
    """One persistent CUDA stream; matrices and LU factors retain its lifetime."""

    def __init__(self, ordinal: int = 0) -> None:
        if not compiled():
            raise RuntimeError(
                "CUDA was compiled out; rebuild with maturin develop --release --features cuda"
            )
        self._device = _native.CudaDevice(ordinal)

    @property
    def name(self) -> str:
        """NVIDIA device name."""
        return self._device.name

    def synchronize(self) -> None:
        """Wait for submitted work, including resident operations."""
        self._device.synchronize()

    def upload(self, value: ArrayLike) -> _native.CudaMatrix:
        """Upload a finite matrix; no implicit host downloads happen afterward."""
        return self._device.upload(np.asarray(value, dtype=np.complex128))

    def factor(self, operator: ArrayLike) -> Factor:
        """Upload and factor once; reuse for requested RHS and their pullbacks."""
        return Factor(self, operator)

    def matmul(
        self,
        left: _native.CudaMatrix,
        right: _native.CudaMatrix,
        *,
        adjoint_left: bool = False,
        adjoint_right: bool = False,
    ) -> _native.CudaMatrix:
        """Multiply resident matrices, optionally conjugate-transposing either input.

        Adjoint operands reuse their device storage. For a fixed sampling operator
        ``F``, ``matmul(F, g, adjoint_left=True)`` computes the coefficient pullback
        ``Fᴴ g`` without a transposed copy of ``F`` or a host round trip.
        """
        return self._device.matmul(
            left, right, adjoint_left=adjoint_left, adjoint_right=adjoint_right
        )


class Factor:
    """A reusable complex128 LU factorization on the supplied device."""

    def __init__(self, device: Device, operator: ArrayLike) -> None:
        self._factor = device._device.factor(np.asarray(operator, dtype=np.complex128))

    @property
    def dimension(self) -> int:
        """Square operator dimension."""
        return self._factor.dimension

    @property
    def nbytes(self) -> int:
        """Owned GPU factor and scaling storage, excluding library workspace."""
        return self._factor.nbytes

    def solve(self, rhs: ArrayLike, *, adjoint: bool = False) -> NDArray[np.complex128]:
        """Solve A X=B (or Aᴴ X=B), including the host/device transfers."""
        return self._factor.solve(np.asarray(rhs, dtype=np.complex128), adjoint=adjoint)

    def solve_device(
        self, rhs: _native.CudaMatrix, *, adjoint: bool = False
    ) -> _native.CudaMatrix:
        """Copy and solve a resident RHS; both input and result stay on device."""
        return self._factor.solve_device(rhs, adjoint=adjoint)

    def solve_with_pullback(
        self, rhs: ArrayLike
    ) -> tuple[NDArray[np.complex128], _native.CudaSolveContext]:
        """Return the host solution and a one-use native complex-pairing pullback."""
        return self._factor.solve_with_pullback(np.asarray(rhs, dtype=np.complex128))


class PlaneWaves:
    """Fused complex128 plane-wave superposition through a pure Rust cuTile kernel.

    The expansion and polarization vectors stay on the device. Evaluating real
    points constructs only the requested fields, with no point-by-mode operator.
    The first call JIT-compiles; subsequent calls reuse the cached kernel.
    """

    def __init__(
        self,
        wavevectors: ArrayLike,
        polarizations: ArrayLike,
        coefficients: ArrayLike,
        *,
        device: int = 0,
        poltype: str = "helicity",
    ) -> None:
        if not hasattr(_native, "CudaPlaneWaves"):
            raise RuntimeError(
                "CUDA Tile requires Linux and a build with --features cuda-tile"
            )
        if poltype not in {"helicity", "parity"}:
            raise ValueError("poltype must be helicity or parity")
        pol = np.asarray(polarizations)
        if not np.all((pol == 0) | (pol == 1)):
            raise ValueError("polarizations must be 0 or 1")
        self._expansion = _native.CudaPlaneWaves(
            device,
            np.asarray(wavevectors, dtype=np.complex128),
            np.asarray(pol, dtype=np.uint8),
            np.asarray(coefficients, dtype=np.complex128),
            poltype == "helicity",
        )

    def evaluate(self, points: ArrayLike) -> NDArray[np.complex128]:
        """Return E(points), including point upload and field download."""
        return self._expansion.evaluate(np.asarray(points, dtype=np.float64))
