"""Coefficients of spherical waves: translation, rotation and periodic arrays.

Mirrors ``treams.sw``. A spherical mode has a degree l, an order m and a
polarization index pol (0 or 1). Coefficients relate a source mode to a
destination mode: ``lambda_``, ``mu`` and ``pol`` label the destination,
``l``, ``m`` and ``qol`` the source. ``periodic_to_pw`` and ``periodic_to_cw``
take the plane or cylindrical destination first. Label arguments broadcast
like NumPy ufunc arguments, and further positional and keyword arguments
(``out``, ``where``) go to the ufunc.

``poltype`` selects the polarization convention: ``"helicity"`` (the default
when omitted), where pol 0 is negative and 1 positive helicity, or
``"parity"``, where pol 0 is the transverse-electric and 1 the
transverse-magnetic wave. ``singular=True`` selects singular (outgoing) waves
at the source; ``singular=False`` regular ones.

Differences from treams:
    - Invalid modes, such as ``|m| > l``, and nonfinite arguments raise
      ValueError; treams returns 0 or NaN for them.
    - There is no ``config.POLTYPE``: ``poltype=None`` always means
      ``"helicity"``.
    - ``translate_periodic`` drops exact duplicate rows of ``out`` and of
      ``in_`` and returns a matrix with one row or column per distinct mode,
      so it can be smaller than treams' result.
    - A custom lattice sum ``func`` in ``translate_periodic`` has no
      gradients. Compute the table of lattice sums with a differentiable
      function and pass it to ``diff.lattice_expansion_from_table`` instead.

Example::

    import numpy as np
    from treams_rs import sw

    # Regular translation by zero is the identity.
    assert np.isclose(sw.translate(1, 0, 1, 1, 0, 1, 0.0, 0, 0, singular=False), 1)
    # Rotation about z by phi multiplies the mode of order m by exp(-i m phi).
    assert np.isclose(sw.rotate(2, 1, 0, 2, 1, 0, 0.3), np.exp(-0.3j))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _lattice, _modes, _native, diff, lattice
from . import _wave_functions as _autodiff
from ._autodiff_functions import transparent_function as _transparent_function
from ._bases import SphericalBasis as _SphericalBasis
from ._polarization import is_helicity as _is_helicity

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

    # Annotation-only name; runtime code uses the private alias above.
    from ._bases import SphericalBasis

__all__ = [
    "periodic_to_cw",
    "periodic_to_pw",
    "rotate",
    "translate",
    "translate_periodic",
]


def translate(
    lambda_: ArrayLike,
    mu: ArrayLike,
    pol: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    kr: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    poltype: str | None = None,
    singular: bool = True,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Translation coefficient of one spherical mode pair.

    Mirrors ``treams.sw.translate``. With A and B from ``special.tl_vsw_A`` and
    ``special.tl_vsw_B`` (``tl_vsw_rA``, ``tl_vsw_rB`` for regular waves), the
    coefficient is A + (2 pol - 1) B for equal helicities and 0 otherwise, or A
    for equal and B for different parities. ``kr``, ``theta`` and ``phi`` are the
    spherical coordinates of the displacement ``k (r_destination - r_source)``.
    The singular coefficient at ``kr = 0`` is 0, as in treams.

    Args:
        lambda_: Degree of the destination mode.
        mu: Order of the destination mode.
        pol: Polarization index of the destination mode.
        l: Degree of the source mode.
        m: Order of the source mode.
        qol: Polarization index of the source mode.
        kr: Distance times the wavenumber, real or complex.
        theta: Polar angle of the displacement.
        phi: Azimuthal angle of the displacement.
        poltype: ``"helicity"`` (default) or ``"parity"``.
        singular: True expands a singular wave at the source in regular waves at
            the destination; False expands a regular wave in regular waves.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the label and coordinate
        arguments.
    """
    if _is_helicity(poltype):
        function = _native.sw_translate_sh if singular else _native.sw_translate_rh
    else:
        function = _native.sw_translate_sp if singular else _native.sw_translate_rp
    return function(lambda_, mu, pol, l, m, qol, kr, theta, phi, *args, **kwargs)


def rotate(
    lambda_: ArrayLike,
    mu: ArrayLike,
    pol: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    phi: ArrayLike,
    theta: ArrayLike = 0,
    psi: ArrayLike = 0,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Rotation coefficient D^l_{mu m}(phi, theta, psi) of one spherical mode pair.

    Mirrors ``treams.sw.rotate``. D is the Wigner D-matrix ``special.wignerd``,
    and the coefficient is 0 unless ``lambda_ == l`` and ``pol == qol``.
    The Euler angles follow the z-y-z convention. In the object-fixed frame
    the rotations apply ``phi`` first, ``theta`` second and ``psi`` third; in
    the fixed frame ``psi`` first and ``phi`` last. The coefficient is the
    same for helicity and parity.

    Args:
        lambda_: Degree of the destination mode.
        mu: Order of the destination mode.
        pol: Polarization index of the destination mode.
        l: Degree of the source mode.
        m: Order of the source mode.
        qol: Polarization index of the source mode.
        phi: First Euler angle, about z.
        theta: Second Euler angle, about y; 0 by default.
        psi: Third Euler angle, about z; 0 by default.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the label and coordinate
        arguments.
    """
    return _native.sw_rotate(
        lambda_, mu, pol, l, m, qol, phi, theta, psi, *args, **kwargs
    )


def translate_periodic(
    ks: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    rs: ArrayLike,
    out: Sequence[ArrayLike],
    in_: Sequence[ArrayLike] | None = None,
    rsin: ArrayLike | None = None,
    poltype: str | None = None,
    eta: complex = 0,
    func: Callable[..., Any] = lattice.lsumsw,
) -> NDArray[np.complex128]:
    """Coupling matrix of spherical modes in a 1D, 2D or 3D lattice.

    Mirrors ``treams.sw.translate_periodic``. Entry (i, j) sums the singular
    translation coefficients from source mode j, repeated at every lattice point
    with the Bloch phase of ``kpar``, to destination mode i. The lattice
    sums converge fast because they split into a real-space and a
    reciprocal-space series (the Ewald split); ``eta`` sets where they split.

    Args:
        ks: Wavenumber of the medium, or two wavenumbers (negative, positive
            helicity) in a chiral medium. Parity needs one wavenumber, or two
            equal ones.
        kpar: Bloch wavevector along the lattice, shape (D,) with 1 <= D <= 3.
        a: Lattice vectors as rows, shape (D, D).
        rs: Positions of the particles in the unit cell, shape (P, 3).
        out: Destination modes: three label arrays (l, m, pol) for particle 0,
            or four arrays (particle index, l, m, pol).
        in_: Source modes in the same form; ``out`` when omitted.
        rsin: Positions of the source particles; ``rs`` when omitted.
        poltype: ``"helicity"`` (default) or ``"parity"``.
        eta: Ewald split parameter; 0 selects it automatically.
        func: Lattice sum with the signature of ``lattice.lsumsw``. It receives
            the shifts ``r_source - r_destination``.

    Returns:
        complex128 array of shape (destination modes, source modes), with exact
        duplicate modes counted once.

    Differences from treams:
        treams returns one row per entry of ``out`` and one column per entry of
        ``in_``, duplicates included. A custom ``func`` has no gradients: use
        ``diff.lattice_expansion_from_table`` with a differentiable table.
    """
    destination, source = _modes.periodic_bases(_SphericalBasis, rs, out, in_, rsin)
    if func is not lattice.lsumsw:
        return _callback_coupling(destination, source, ks, kpar, a, poltype, eta, func)
    return diff.lattice_expansion(
        destination,
        source,
        np.broadcast_to(np.asarray(ks, dtype=np.complex128), (2,)),
        kpar,
        a,
        poltype=poltype,
        eta=eta,
    )[0]


def _callback_coupling(
    destination: SphericalBasis,
    source: SphericalBasis,
    ks: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    poltype: str | None,
    eta: complex,
    func: Callable[..., Any],
) -> NDArray[np.complex128]:
    """Periodic coupling from the lattice-sum table that ``func`` returns."""
    helicity = _is_helicity(poltype)
    wavenumbers = np.atleast_1d(np.asarray(ks, dtype=np.complex128))
    if wavenumbers.shape not in ((1,), (2,)) or not np.isfinite(wavenumbers).all():
        raise ValueError("require one or two finite medium wavenumbers")
    if wavenumbers.size == 2 and wavenumbers[0] == wavenumbers[1]:
        wavenumbers = wavenumbers[:1]
    if not helicity and wavenumbers.size != 1:
        raise ValueError("parity requires an achiral embedding medium")
    maximum = int(np.max(destination.l) + np.max(source.l))
    modes = np.array(
        [
            (degree, order)
            for degree in range(maximum + 1)
            for order in range(-degree, degree + 1)
        ]
    )
    shifts = (
        source.positions[None, :, None, None, :]
        - destination.positions[:, None, None, None, :]
    )
    cell, bloch = _lattice.periodic_geometry(a, kpar, True)
    dim = len(bloch)
    values = func(
        dim,
        modes[:, 0],
        modes[:, 1],
        wavenumbers[:, None],
        bloch[0] if dim == 1 else bloch,
        cell[0][0] if dim == 1 else cell,
        shifts,
        eta,
    )
    shape = (
        len(destination.positions),
        len(source.positions),
        wavenumbers.size,
        len(modes),
    )
    return diff.lattice_expansion_from_table(
        np.broadcast_to(values, shape),
        destination,
        source,
        poltype="helicity" if helicity else "parity",
    )[0]


def periodic_to_pw(
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    pol: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    area: ArrayLike,
    poltype: str | None = None,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Plane-wave coefficient of a 2D periodic array of spherical waves.

    Mirrors ``treams.sw.periodic_to_pw``. The source mode (l, m, qol) repeats at
    every lattice point with the Bloch phase; the coefficient belongs to the
    destination plane wave (kx, ky, kz, pol). It holds for one particle at the
    origin; other positions need the phase of ``pw.translate``.

    Args:
        kx: x component of the destination wavevector.
        ky: y component of the destination wavevector.
        kz: z component of the destination wavevector, real or complex.
        pol: Polarization index of the destination plane wave.
        l: Degree of the source mode.
        m: Order of the source mode.
        qol: Polarization index of the source mode.
        area: Area of the unit cell.
        poltype: ``"helicity"`` (default) or ``"parity"``.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the label and coordinate
        arguments.
    """
    function = (
        _native.sw_periodic_to_pw_h
        if _is_helicity(poltype)
        else _native.sw_periodic_to_pw_p
    )
    return function(kx, ky, kz, pol, l, m, qol, area, *args, **kwargs)


def periodic_to_cw(
    kz: ArrayLike,
    m: ArrayLike,
    pol: ArrayLike,
    l: ArrayLike,
    mu: ArrayLike,
    qol: ArrayLike,
    k: ArrayLike,
    area: ArrayLike,
    poltype: str | None = None,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Cylindrical-wave coefficient of a periodic chain of spherical waves along z.

    Mirrors ``treams.sw.periodic_to_cw``. ``kz``, ``m`` and ``pol`` label the
    cylindrical destination mode; ``l``, ``mu`` and ``qol`` the spherical source
    mode. The source repeats every ``area`` along z with the Bloch phase.

    Args:
        kz: Axial wavenumber of the destination mode.
        m: Order of the destination mode.
        pol: Polarization index of the destination mode.
        l: Degree of the source mode.
        mu: Order of the source mode.
        qol: Polarization index of the source mode.
        k: Wavenumber of the medium.
        area: Period of the array along z; treams keeps the name ``area``.
        poltype: ``"helicity"`` (default) or ``"parity"``.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the label and coordinate
        arguments.
    """
    function = (
        _native.sw_periodic_to_cw_h
        if _is_helicity(poltype)
        else _native.sw_periodic_to_cw_p
    )
    return function(kz, m, pol, l, mu, qol, k, area, *args, **kwargs)


# Keep native NumPy semantics while selecting a derivative adapter for framework
# inputs. The callbacks share the existing Rust records with the physics API.

translate = _transparent_function(translate, _autodiff.sw_translate, module=__name__)
rotate = _transparent_function(rotate, _autodiff.sw_rotate, module=__name__)
translate_periodic = _transparent_function(
    translate_periodic, _autodiff.sw_translate_periodic, module=__name__
)
periodic_to_cw = _transparent_function(
    periodic_to_cw, _autodiff.sw_periodic_to_cw, module=__name__
)
periodic_to_pw = _transparent_function(
    periodic_to_pw, _autodiff.periodic_to_pw, module=__name__
)
