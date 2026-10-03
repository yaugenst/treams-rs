"""Mie and Fresnel coefficients in the helicity basis.

Mirrors ``treams.coeffs``. ``mie`` and ``mie_cyl`` return one complex128
(2, 2) block per call: rows are the scattered and columns the incident
helicity, with index 0 for negative and 1 for positive helicity. Materials
are (epsilon, mu, kappa): relative permittivity, relative permeability and
chirality parameter, ordered from the innermost layer out to the embedding
medium. ``diff.mie``, ``diff.mie_cyl`` and ``diff.fresnel`` return the same
values together with a context that computes gradients.

Differences from treams:
    - treams' functions are generalized ufuncs that broadcast over leading
      axes, for example ``mie(l[:, None], ...)``. Here ``mie`` takes one
      degree, ``mie_cyl`` one order and ``fresnel`` one interface: loop over
      the degrees, orders or interfaces. An array degree or order raises
      ValueError.
    - Invalid layers raise ValueError, for example a zero or decreasing
      radius; treams raises ZeroDivisionError for a zero radius.

Example::

    import numpy as np
    from treams_rs import coeffs

    # A sphere of size parameter k0 r = 0.5 and permittivity 4 in vacuum.
    dipole = coeffs.mie(1, [0.5], [4, 1], [1, 1], [0, 0])
    assert dipole.shape == (2, 2)
    # Both helicities scatter alike from an achiral sphere.
    assert np.isclose(dipole[0, 0], dipole[1, 1])
    assert np.isclose(dipole[0, 1], dipole[1, 0])
"""

from __future__ import annotations

from functools import partial as _partial
from typing import TYPE_CHECKING

import numpy as np

from . import diff
from ._dispatch import backend_for as _backend_for

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "fresnel",
    "mie",
    "mie_cyl",
]


def fresnel(ks: ArrayLike, kzs: ArrayLike, zs: ArrayLike) -> NDArray[np.complex128]:
    """Fresnel coefficients of one planar interface between two chiral media.

    Mirrors ``treams.coeffs.fresnel``. The interface is the plane z = 0 between
    the negative side (z < 0) and the positive side (z > 0). The result
    ``r[d_out, d_in, p_out, p_in]`` maps an incident wave of direction ``d_in``
    and helicity ``p_in`` to an outgoing wave of direction ``d_out`` and helicity
    ``p_out``; direction 0 travels up (towards +z) and 1 down. So ``r[0, 0]``
    transmits from the negative to the positive side, and ``r[1, 0]`` reflects
    back into the negative side.

    Args:
        ks: Wavenumbers, shape (2, 2): negative then positive side, each as
            (negative, positive) helicity.
        kzs: z components of the wavevectors, same shape and order.
        zs: Relative impedances of the negative and the positive side, shape
            (2,).

    Returns:
        complex128 array of shape (2, 2, 2, 2).

    Differences from treams:
        treams broadcasts over leading axes of the inputs; here the inputs
        describe one interface.
    """
    backend = _backend_for(ks, kzs, zs)
    if backend is not None:
        values = tuple(backend.array(v, complex_=True) for v in (ks, kzs, zs))
        return backend.apply(diff.fresnel, (2, 2, 2, 2), *values)
    return diff.fresnel(ks, kzs, zs)[0]


def mie(
    l: int,
    x: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> NDArray[np.complex128]:
    """Mie coefficients of a multilayer chiral sphere for one degree l.

    Mirrors ``treams.coeffs.mie``. The (2, 2) block maps incident to scattered
    helicities: the T-matrix entries of degree l, which do not depend on the
    order m.

    Args:
        l: Degree, an integer from 1 to 128.
        x: Size parameters k0 * radius of the layers from the inside out, shape
            (layers,), increasing.
        epsilon: Relative permittivities, shape (layers + 1,): the layers from
            the inside out, then the embedding medium.
        mu: Relative permeabilities, same shape and order.
        kappa: Chirality parameters, same shape and order.

    Returns:
        complex128 array of shape (2, 2): rows scattered, columns incident
        helicity.

    Differences from treams:
        treams broadcasts over arrays of degrees and layers; here ``l`` is one
        integer and ``x`` one 1-D array. An array ``l`` raises ValueError, so
        loop over the degrees.
    """
    if np.ndim(l) != 0:
        raise ValueError(
            "mie takes one integer degree; loop over degrees (treams broadcasts)"
        )
    backend = _backend_for(x, epsilon, mu, kappa)
    if backend is not None:
        values = (
            backend.array(x),
            *(backend.array(v, complex_=True) for v in (epsilon, mu, kappa)),
        )
        return backend.apply(_partial(diff.mie, l), (2, 2), *values)
    return diff.mie(l, x, epsilon, mu, kappa)[0]


def mie_cyl(
    kz: float,
    m: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> NDArray[np.complex128]:
    """Scattering coefficients of a multilayer chiral cylinder for one order m.

    Mirrors ``treams.coeffs.mie_cyl``. The cylinder is infinite along z. The
    (2, 2) block maps incident to scattered helicities: the T-matrix entries of
    the axial wavenumber ``kz`` and the order ``m``.

    Args:
        kz: Axial wavenumber, real.
        m: Order, one integer.
        k0: Vacuum wavenumber.
        radii: Radii of the layers from the inside out, shape (layers,),
            increasing.
        epsilon: Relative permittivities, shape (layers + 1,): the layers from
            the inside out, then the embedding medium.
        mu: Relative permeabilities, same shape and order.
        kappa: Chirality parameters, same shape and order.

    Returns:
        complex128 array of shape (2, 2): rows scattered, columns incident
        helicity.

    Differences from treams:
        treams broadcasts over arrays of orders and layers; here ``m`` is one
        integer and ``radii`` one 1-D array. An array ``m`` raises ValueError, so
        loop over the orders.
    """
    if np.ndim(m) != 0:
        raise ValueError(
            "mie_cyl takes one integer order; loop over orders (treams broadcasts)"
        )
    backend = _backend_for(kz, k0, radii, epsilon, mu, kappa)
    if backend is not None:
        values = (
            *(backend.array(v) for v in (kz, k0, radii)),
            *(backend.array(v, complex_=True) for v in (epsilon, mu, kappa)),
        )

        def record(kz: Any, *values: Any) -> Any:
            return diff.mie_cyl(kz, m, *values)

        return backend.apply(record, (2, 2), *values)
    return diff.mie_cyl(kz, m, k0, radii, epsilon, mu, kappa)[0]
