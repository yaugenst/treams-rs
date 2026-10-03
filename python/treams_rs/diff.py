"""Records: functions that return a value and a context for its gradients.

A record is a function that returns a value and a context. The context stores what is needed to compute gradients later and can be used once: `context.pullback(g)` takes the gradient `g` of a real-valued loss with respect to the value and returns the gradients with respect to the inputs, one for each differentiable input, in the order of the arguments. Gradients follow the convention dL = Re Σ conj(g)·dx.

One record and its pullback::

    import numpy as np
    from treams_rs import diff

    value, context = diff.solve(2 * np.eye(2), np.ones((2, 1)))
    grad_operator, grad_rhs = context.pullback(np.ones((2, 1)))
    assert value.shape == grad_rhs.shape == (2, 1)

Terms:

* record: a function of ``diff``, or a ``record`` method, that returns
  ``(value, context)``.
* context: the object that stores what the gradients need. A second
  ``pullback`` raises ValueError; call the record again for another gradient.
* residual: the Rust name of the data a context stores.
* pullback: ``context.pullback(g)``, the map from the gradient with respect to
  the value to the gradients with respect to the inputs.
* cotangent: the gradient ``g`` of the loss with respect to one value. It has
  the shape of that value and is complex for a complex value.
* dynamic inputs: the inputs that get a gradient, in pullback order.
* static configuration: labels, bases, cutoffs and options. They stay fixed
  and get no gradient. A record without gradients lists all its inputs here.

A pullback returns one gradient per dynamic input, in the order of the
forward arguments; a basis contributes the gradient of its ``positions``.
Two variants change that list:

* ``context.pullback_axial(g)`` of cylindrical expansions, fields and lattice
  expansions appends the gradient of the axial wavenumbers kz.
* ``context.pullback_blocks(g)`` of factors built from separate particle
  blocks returns a list with one gradient per block in place of the local
  matrix.

Objects record too: ``InteractionFactor.record(incident)`` (the factor that
``factor_interaction``, ``factor_interaction_blocks`` and
``sphere_cluster_factor`` return) and ``iterative.SphereCluster.record``.

``advect.X``, ``jax.X`` and ``torch.X`` differentiate ``diff.X`` in their
framework; ``jax`` and ``torch`` provide bessel, illuminate, interaction,
solve and sphere. The physics objects (TMatrix, SMatrix, Wave, ...) call the
records for their values and drop the contexts.

Each ``advect`` function that differentiates a record (``advect.X`` for
``diff.X``) takes the inputs of that record, with the dynamic arrays
positional in pullback order and the static configuration keyword-only. Most
signatures are identical; these differ:

* bessel, angular, incgamma, intkambe and wignerd take their arguments (z,
  z and eta, or the three angles) first and the labels as keywords;
  lattice_sum takes k, kpar, a, r and eta first and dim, degree and order as
  keywords.
* expansion, lattice_expansion, field, field_operator, plane_expansion,
  spherical_channels and periodic_to_cw take the positions as arrays
  (``positions``, or ``destination_positions`` and ``source_positions``)
  and the bases as keywords; plane_expansion, spherical_channels and
  cylindrical_channels also take their polarizations as keywords. The
  cylindrical variants take the axial wavenumbers as an array too: ``kzs``
  (distinct values) for expansions, ``kz`` (one per mode) for fields and
  periodic_to_cw.
* plane_field, plane_permutation, rotation and lattice_expansion_from_table
  take their polarizations, bases or permutation count as keywords.
* cylindrical_channels splits q into static ``kz_labels`` and a dynamic
  ``kx`` and has no fixed_q.
* mie calls its size parameters ``sizes`` instead of ``x``.
* factor_interaction, factor_interaction_blocks and sphere_cluster_factor
  exist only in ``diff``.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, cast

import numpy as np

from . import _native
from ._bases import ALIGNMENT_AXIS, CylindricalBasis, SphericalBasis
from ._lattice import periodic_geometry
from ._modes import ebcm_modes
from ._polarization import resolve_poltype
from ._validation import MAX_DEGREE, MAX_LABEL, MAX_ORDER, one_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from ._modes import Modes

__all__ = [
    "angular",
    "bands",
    "bessel",
    "chirality_density",
    "coordinates",
    "cylinder",
    "cylindrical_channels",
    "cylindrical_translation",
    "ebcm_qmat",
    "eig",
    "expansion",
    "factor_interaction",
    "factor_interaction_blocks",
    "field",
    "field_operator",
    "fresnel",
    "illuminate",
    "incgamma",
    "interaction",
    "interface_coefficients",
    "intkambe",
    "lattice_expansion",
    "lattice_expansion_from_table",
    "lattice_sum",
    "layer_stack",
    "mie",
    "mie_cyl",
    "oriented_chirality",
    "particle_cluster",
    "periodic_to_cw",
    "plane_expansion",
    "plane_field",
    "plane_permutation",
    "plane_phases",
    "propagation_matrix",
    "rotation",
    "smatrix_add",
    "smatrix_from_array",
    "smatrix_illuminate",
    "smatrix_periodic",
    "smatrix_tr",
    "solve",
    "sph_harm",
    "sphere",
    "sphere_cluster",
    "sphere_cluster_factor",
    "spherical_channels",
    "spherical_translation",
    "svdvals",
    "tmatrix_metric",
    "vector_coordinates",
    "vector_wave",
    "wignerd",
]


def _polarizations(values: ArrayLike) -> list[int]:
    pols = np.asarray(values)
    if not np.all((pols == 0) | (pols == 1)):
        raise ValueError("polarizations must be 0 or 1")
    return pols.astype(np.int64).tolist()


def _wavenumber_pair(ks: ArrayLike) -> tuple[complex, complex]:
    values = np.asarray(ks, dtype=np.complex128)
    if values.shape != (2,):
        raise ValueError("ks must contain negative and positive helicity wavenumbers")
    return complex(values[0]), complex(values[1])


def _layers(
    epsilon: ArrayLike, mu: ArrayLike | None, kappa: ArrayLike | None
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Complex layer materials; mu defaults to one and kappa to zero."""
    eps = np.ascontiguousarray(epsilon, dtype=np.complex128)
    return (
        eps,
        np.ones_like(eps)
        if mu is None
        else np.ascontiguousarray(mu, dtype=np.complex128),
        np.zeros_like(eps)
        if kappa is None
        else np.ascontiguousarray(kappa, dtype=np.complex128),
    )


def _shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """np.broadcast_shapes, without NumPy for equal shapes and single elements."""
    # Plain loops without key functions or generators: these helpers run on
    # every recorded call, whose kernels take only a few microseconds.
    shape: tuple[int, ...] = ()
    for s in shapes:
        if s != shape and s:
            # Every shape seen so far broadcasts to `shape`.
            if not shape or (len(s) >= len(shape) and prod(shape) == 1):
                shape = s
            elif len(s) > len(shape) or prod(s) != 1:
                return np.broadcast_shapes(*shapes)
    return shape


def _pack[T: np.generic](
    value: NDArray[T], shape: tuple[int, ...], core: tuple[int, ...] = ()
) -> NDArray[T]:
    """Flatten a broadcast batch to (-1, *core) for the native broadcasting rule.

    A batch of one element stays unbroadcast; Rust broadcasts it and reduces
    its gradient back to the original argument shape.
    """
    if not core:
        if value.shape != shape and value.size != 1:
            value = np.broadcast_to(value, shape)
        return value.ravel()
    outer = value.shape[: value.ndim - len(core)]
    if outer != shape and prod(outer) != 1:
        value = np.broadcast_to(value, shape + core)
    return value.reshape((-1, *core))


def _flat[T: np.generic](
    values: Sequence[NDArray[T]], shape: tuple[int, ...]
) -> list[NDArray[T]]:
    """_pack of every value, with one Python call for the whole batch."""
    return [
        v.ravel()
        if v.shape == shape or v.size == 1
        else np.broadcast_to(v, shape).ravel()
        for v in values
    ]


_INTEGER = (int, np.integer)
_NUMBER = (int, float, complex)


def _numbers(values: Sequence[object]) -> bool:
    """Whether every value is a Python number (bool, int, float or complex)."""
    for v in values:  # noqa: SIM110 - all() over a generator is slower
        if not isinstance(v, _NUMBER):
            return False
    return True


def _required(name: str, value: str | None, alias: str, alias_value: str | None) -> str:
    """A required keyword given under ``name`` or its alias ``alias``."""
    resolved = one_of(name, value, alias, alias_value, None)
    if resolved is None:
        raise TypeError(f"missing required keyword argument {name!r}")
    return resolved


def _label_error(name: str, bound: int) -> ValueError:
    return ValueError(f"{name} must be integers in [-{bound}, {bound}]")


def _mode_tuples(
    labels: Sequence[NDArray[np.float64]], shape: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """One label tuple, or one per broadcast element when any label varies."""
    if all(v.size == 1 for v in labels):
        return [tuple(int(v.item()) for v in labels)]
    columns = (np.broadcast_to(v, shape).ravel().astype(int).tolist() for v in labels)
    return list(zip(*columns, strict=True))


def _label_modes(
    values: Sequence[ArrayLike],
    bound: int,
    name: str,
    shapes: Sequence[tuple[int, ...]],
) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    """Validated integer label tuples and the shape broadcast with ``shapes``.

    Labels must be finite integers within +/-bound, whether Python ints or arrays.
    """
    for v in values:
        if not isinstance(v, _INTEGER) or not -bound <= v <= bound:
            break
    else:
        integers = tuple(map(int, cast("Sequence[int]", values)))
        return [integers], _shape((), *shapes)
    if all(isinstance(v, _INTEGER) for v in values):
        raise _label_error(name, bound)
    labels = tuple(np.asarray(v, dtype=np.float64) for v in values)
    if any(
        np.any(~np.isfinite(v) | (v != np.floor(v)) | (np.abs(v) > bound))
        for v in labels
    ):
        raise _label_error(name, bound)
    shape = _shape(*(v.shape for v in labels), *shapes)
    return _mode_tuples(labels, shape), shape


def bessel(
    order: ArrayLike,
    z: ArrayLike,
    *,
    function: str = "j",
    spherical: bool = False,
    derivative: bool = False,
) -> tuple[NDArray[np.complex128], _native.BesselContext]:
    """Bessel functions or their first derivatives, broadcast over order and z.

    Returns:
        complex128 array with the broadcast shape of ``order`` and ``z``. Two
        Python numbers give a 0-d array.

    Dynamic inputs:
        z: complex argument. Its gradient has the shape of ``z``; broadcast axes
            are summed.

    Static configuration:
        order: real order, broadcast with ``z``.
        function: "j", "y", "h1" or "h2".
        spherical: spherical instead of cylindrical Bessel functions.
        derivative: the first derivative with respect to z instead of the value.
    """
    if isinstance(order, (int, float)) and isinstance(z, (int, float, complex)):
        return _native.bessel_record_scalar(
            order, z, function, spherical, int(derivative)
        )
    orders, arguments, shape, argument_shape = _bessel_inputs(order, z)
    return _native.bessel_record(
        orders, arguments, function, spherical, int(derivative), shape, argument_shape
    )


def _bessel_inputs(
    order: ArrayLike, z: ArrayLike
) -> tuple[
    NDArray[np.float64], NDArray[np.complex128], tuple[int, ...], tuple[int, ...]
]:
    orders = np.asarray(order, dtype=np.float64)
    arguments = np.asarray(z, dtype=np.complex128)
    shape = _shape(orders.shape, arguments.shape)
    return _pack(orders, shape), _pack(arguments, shape), shape, arguments.shape


def lattice_sum(
    dim: int,
    degree: ArrayLike,
    order: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    *,
    spherical: bool = True,
    part: str = "full",
    shell: ArrayLike = 0,
) -> tuple[NDArray[np.complex128], _native.LatticeSumContext]:
    """Lattice sums of one Bloch lattice, broadcast over labels and geometry.

    The Ewald method splits each sum into a real-space and a reciprocal-space
    series; ``eta`` sets where, and 0 picks it automatically.
    Ewald sums require ``Im k >= 0`` and, for spherical waves, ``Re k >= 0``.
    Finite direct shells accept every finite nonzero complex wavenumber.

    Returns:
        complex128 array with the broadcast batch shape of all inputs.

    Dynamic inputs:
        k: complex wavenumber.
        kpar: Bloch vector, last axis of length dim.
        a: lattice vectors as rows, last axes (dim, dim); a scalar in 1D.
        r: shift, last axis 3 (spherical) or 2 (cylindrical).
        eta: complex Ewald split parameter. The full sum does not depend on
            it, so its gradient is zero for part="full"; the real and reciprocal
            parts need an explicit nonzero eta for gradients.

    Static configuration:
        dim: lattice dimension, 1 to 3 for spherical and 1 to 2 for
            cylindrical sums.
        degree: integer degree l, broadcast.
        order: integer order m, broadcast.
        spherical: spherical (lsumsw) or cylindrical (lsumcw) sums.
        part: "full", "real", "reciprocal" or "direct".
        shell: integer cube shell of the direct sum (part="direct").
    """
    if not 1 <= dim <= (3 if spherical else 2):
        raise ValueError("invalid lattice dimension")
    try:
        component = ("full", "real", "reciprocal", "direct").index(part)
    except ValueError:
        raise ValueError("part must be full, real, reciprocal or direct") from None
    arguments = (
        np.asarray(k, dtype=np.complex128),
        np.asarray(kpar, dtype=np.float64),
        np.asarray(a, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(eta, dtype=np.complex128),
    )
    shapes = [argument.shape for argument in arguments]
    wavenumbers, bloch, cells, shifts, etas = arguments
    if dim == 1:
        if bloch.shape[-1:] != (1,):
            bloch = bloch[..., None]
        if cells.shape[-2:] != (1, 1):
            cells = cells[..., None, None]
    coordinates = 3 if spherical else 2
    if (
        bloch.shape[-1:] != (dim,)
        or cells.shape[-2:] != (dim, dim)
        or shifts.shape[-1:] != (coordinates,)
    ):
        raise ValueError(
            "lattice, Bloch vector and Cartesian shift dimensions must agree"
        )
    degrees, orders, shells = (
        np.asarray(value, dtype=np.float64) for value in (degree, order, shell)
    )
    if any(
        not v.item().is_integer()
        if v.ndim == 0
        else (~np.isfinite(v) | (v != np.floor(v))).any()
        for v in (degrees, orders, shells)
    ):
        raise ValueError("degree, order and shell must be finite integers")
    if (
        (np.abs(degrees) > MAX_DEGREE).any()
        or (np.abs(orders) > MAX_DEGREE).any()
        or ((shells < 0) | (shells > np.iinfo(np.int32).max)).any()
    ):
        raise ValueError("lattice labels or shell exceed supported bounds")
    shape = _shape(
        degrees.shape,
        orders.shape,
        shells.shape,
        wavenumbers.shape,
        bloch.shape[:-1],
        cells.shape[:-2],
        shifts.shape[:-1],
        etas.shape,
    )
    return _native.lattice_sum_record(
        spherical,
        dim,
        [(mode[0], mode[1]) for mode in _mode_tuples((degrees, orders), shape)],
        _pack(wavenumbers, shape),
        _pack(bloch, shape, (dim,)),
        _pack(cells, shape, (dim, dim)),
        _pack(shifts, shape, (coordinates,)),
        _pack(etas, shape),
        component,
        _pack(shells, shape).astype(np.int64).tolist(),
        shape,
        shapes,
    )


def incgamma(
    n: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.complex128], _native.IncgammaContext]:
    """Upper incomplete gamma function Gamma(n, z) for half-integer n.

    Returns:
        complex128 array with the broadcast shape of ``n`` and ``z``.

    Dynamic inputs:
        z: complex argument; its gradient has the shape of ``z``.

    Static configuration:
        n: half-integer degree, broadcast with ``z``.
    """
    if isinstance(n, (int, float)) and isinstance(z, (int, float, complex)):
        return _native.incgamma_record_scalar(n, z)
    degrees, arguments, shape, argument_shape = _bessel_inputs(n, z)
    return _native.incgamma_record(degrees, arguments, shape, argument_shape)


def intkambe(
    n: ArrayLike, z: ArrayLike, eta: ArrayLike
) -> tuple[NDArray[np.complex128], _native.IntkambeContext]:
    """Kambe integral of the Ewald lattice sums, broadcast over n, z and eta.

    Derivatives follow the selected complex branch. At z=0, n=-2 has a cusp
    without a derivative, and n<=-3 has a zero z derivative.

    Returns:
        complex128 array with the broadcast shape of ``n``, ``z`` and ``eta``.

    Dynamic inputs:
        z: complex argument.
        eta: complex Ewald split parameter.

    Static configuration:
        n: integer order, broadcast.
    """
    if (
        isinstance(n, int)
        and isinstance(z, (int, float, complex))
        and isinstance(eta, (int, float, complex))
    ):
        if abs(n) > MAX_LABEL:
            raise _label_error("Kambe orders", MAX_LABEL)
        return _native.intkambe_record_scalar(n, z, eta)
    arguments = (
        np.asarray(z, dtype=np.complex128),
        np.asarray(eta, dtype=np.complex128),
    )
    orders, shape = _label_modes(
        (n,), MAX_LABEL, "Kambe orders", (arguments[0].shape, arguments[1].shape)
    )
    return _native.intkambe_record(
        np.array([order for (order,) in orders], dtype=np.int32),
        _pack(arguments[0], shape),
        _pack(arguments[1], shape),
        shape,
        (arguments[0].shape, arguments[1].shape),
    )


def angular(
    degree: ArrayLike,
    order: ArrayLike,
    z: ArrayLike,
    *,
    function: str = "legendre",
) -> tuple[NDArray[np.complex128], _native.AngularContext]:
    """Legendre, pi or tau angular functions of integer degree, broadcast over z.

    A derivative at a branch point raises ValueError unless its cotangent is
    zero.

    Returns:
        complex128 array with the broadcast shape of ``degree``, ``order`` and
        ``z``.

    Dynamic inputs:
        z: complex argument; its gradient has the shape of ``z``.

    Static configuration:
        degree: integer degree l, broadcast.
        order: integer order m, broadcast.
        function: "legendre", "pi" or "tau".
    """
    if (
        isinstance(degree, (int, float))
        and isinstance(order, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.angular_record_scalar(degree, order, z, function)
    if isinstance(degree, (int, float)) and isinstance(order, (int, float)):
        # Rust broadcasts one-element labels; only z needs an array.
        labels = np.array((degree, order), dtype=np.float64)
        arguments = np.asarray(z, dtype=np.complex128)
        return _native.angular_record(
            labels[:1],
            labels[1:],
            arguments.ravel(),
            function,
            arguments.shape,
            arguments.shape,
        )
    degrees = np.asarray(degree, dtype=np.float64)
    orders = np.asarray(order, dtype=np.float64)
    arguments = np.asarray(z, dtype=np.complex128)
    shape = _shape(degrees.shape, orders.shape, arguments.shape)
    return _native.angular_record(
        _pack(degrees, shape),
        _pack(orders, shape),
        _pack(arguments, shape),
        function,
        shape,
        arguments.shape,
    )


def wignerd(
    degree: ArrayLike,
    row: ArrayLike,
    column: ArrayLike,
    phi: ArrayLike,
    theta: ArrayLike,
    psi: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.WignerdContext]:
    """Wigner D matrix elements D^l_{m m'}(phi, theta, psi), broadcast.

    Returns:
        complex128 array with the broadcast shape of the labels and the angles.

    Dynamic inputs:
        phi: first z-y-z Euler angle, complex allowed.
        theta: second Euler angle.
        psi: third Euler angle. Each gradient has the shape of its angle.

    Static configuration:
        degree: integer degree l.
        row: integer order m.
        column: integer order m'.
    """
    if (
        isinstance(degree, int)
        and isinstance(row, int)
        and isinstance(column, int)
        and isinstance(phi, (int, float, complex))
        and isinstance(theta, (int, float, complex))
        and isinstance(psi, (int, float, complex))
    ):
        if max(abs(degree), abs(row), abs(column)) > MAX_LABEL:
            raise _label_error("Wigner labels", MAX_LABEL)
        return _native.wignerd_record_scalar(
            (degree, row, column), (complex(phi), complex(theta), complex(psi))
        )
    angles = tuple(np.asarray(v, dtype=np.complex128) for v in (phi, theta, psi))
    modes, shape = _label_modes(
        (degree, row, column), MAX_LABEL, "Wigner labels", [v.shape for v in angles]
    )
    return _native.wignerd_record(
        [(mode[0], mode[1], mode[2]) for mode in modes],
        _pack(angles[0], shape),
        _pack(angles[1], shape),
        _pack(angles[2], shape),
        shape,
        (angles[0].shape, angles[1].shape, angles[2].shape),
    )


def chirality_density(
    ks: ArrayLike, normal: ArrayLike, z: ArrayLike = (0.0, 0.0)
) -> tuple[NDArray[np.complex128], _native.ChiralityDensityContext]:
    """Chirality density forms of plane waves in an xy basis.

    Multiply by helicity signs, or pair opposite parity modes, to form
    operators. The cross form contracts as Re(down.conj() @ cross @ up). The
    transverse wavevectors are real.

    Returns:
        complex128 array (3, modes): the up, down and cross forms.

    Dynamic inputs:
        ks: wavenumber of each mode, shape (modes,).
        normal: complex normal component of each wavevector, shape (modes,).
        z: interval (start, end) along the normal; equal ends give the density
            on one plane.

    Static configuration:
        none.
    """
    start, stop = np.asarray(z, dtype=np.float64)
    return _native.chirality_density(
        np.ascontiguousarray(ks, dtype=np.complex128),
        np.ascontiguousarray(normal, dtype=np.complex128),
        (float(start), float(stop)),
    )


def oriented_chirality(
    transverse: ArrayLike,
    normal: ArrayLike,
    z: ArrayLike = (0.0, 0.0),
    *,
    polarizations: ArrayLike,
    axis: int = 2,
) -> tuple[NDArray[np.complex128], _native.OrientedChiralityContext]:
    """Signed helicity chirality forms for any Cartesian normal.

    Returns:
        complex128 array (3, modes): the up, down and cross forms.

    Dynamic inputs:
        transverse: real transverse components, shape (modes, 2), in cyclic
            order after ``axis``.
        normal: complex normal component of each wavevector, shape (modes,).
        z: interval (start, end) along the normal.

    Static configuration:
        polarizations: pol index of each mode.
        axis: the normal axis, 0, 1 or 2.
    """
    start, stop = np.asarray(z, dtype=np.float64)
    return _native.oriented_chirality(
        np.asarray(transverse, dtype=np.float64),
        np.asarray(normal, dtype=np.complex128),
        np.asarray(polarizations).tolist(),
        axis,
        (float(start), float(stop)),
    )


def ebcm_qmat(
    radii: ArrayLike,
    slopes: ArrayLike,
    ks: ArrayLike,
    zs: ArrayLike,
    *,
    theta: ArrayLike,
    weights: ArrayLike,
    destination: Modes,
    source: Modes | None = None,
    singular: bool = True,
    radial_area_factor: bool = True,
) -> tuple[NDArray[np.complex128], _native.EbcmQmatContext]:
    """Q matrix of an axisymmetric particle from a sampled surface (EBCM).

    ``destination`` and ``source`` are the modes that ``ebcm.qmat`` calls
    ``out`` and ``in_``. radial_area_factor=False reproduces treams.ebcm.qmat,
    which omits the radial surface-area factor; see Differences from treams.

    Returns:
        complex128 array (len(destination), len(source)).

    Dynamic inputs:
        radii: surface radius at each polar node, shape (nodes,).
        slopes: derivative of the radius with respect to the polar angle,
            shape (nodes,).
        ks: wavenumbers, shape (2, 2): inside then outside, negative then
            positive helicity.
        zs: impedances, shape (2,): inside then outside.

    Static configuration:
        theta: polar quadrature nodes in [0, pi], shape (nodes,).
        weights: quadrature weights, shape (nodes,).
        destination: (l, m, pol) modes of the rows.
        source: (l, m, pol) modes of the columns; defaults to ``destination``.
        singular: singular (True) or regular (False) radial functions, as in
            ``ebcm.qmat``.
        radial_area_factor: keep the factor r of the surface element.
    """
    zm, zp = np.asarray(zs, dtype=np.complex128)
    return _native.ebcm_qmat(
        np.column_stack([theta, weights, radii, slopes]).astype(np.float64),
        ebcm_modes(destination),
        ebcm_modes(destination if source is None else source),
        np.ascontiguousarray(ks, dtype=np.complex128),
        (complex(zm), complex(zp)),
        singular,
        radial_area_factor,
    )


def tmatrix_metric(
    operator: ArrayLike,
    ks: ArrayLike = (1.0, 1.0),
    *,
    polarizations: ArrayLike,
    metric: str | None = None,
    kind: str | None = None,
) -> tuple[float, _native.TMatrixMetricContext]:
    """Global helicity metric of a T-matrix: cd, db or chi.

    A zero scattering norm, or zero total absorption for cd, has no defined
    metric. At zero contrast, chi has a value and a zero gradient.

    Returns:
        float.

    Dynamic inputs:
        operator: helicity T-matrix in a global basis, shape (modes, modes).
        ks: real wavenumbers for pol 0 and 1, shape (2,); only cd depends on
            them.

    Static configuration:
        polarizations: pol index of each mode.
        metric: "cd" (circular dichroism), "db" (duality breaking) or "chi"
            (electromagnetic chirality); ``kind`` is an alias.
    """
    metric = _required("metric", metric, "kind", kind)
    km, kp = np.asarray(ks, dtype=np.float64)
    return _native.tmatrix_metric(
        np.ascontiguousarray(operator, dtype=np.complex128),
        np.asarray(polarizations).tolist(),
        (float(km), float(kp)),
        metric,
    )


def svdvals(operator: ArrayLike) -> tuple[NDArray[np.float64], _native.SvdvalsContext]:
    """Singular values of a complex matrix, in descending order.

    The pullback uses the thin singular vectors. Repeated positive values need
    equal cotangents. A zero singular value needs a zero cotangent: single
    values there have no gradient.

    Returns:
        float64 array (min(rows, columns),).

    Dynamic inputs:
        operator: complex matrix (rows, columns).

    Static configuration:
        none.
    """
    return _native.svdvals(np.ascontiguousarray(operator, dtype=np.complex128))


def solve(
    operator: ArrayLike, rhs: ArrayLike
) -> tuple[NDArray[np.complex128], _native.SolveContext]:
    """Solve A X = B.

    The pullback reuses the LU factors of the forward solve.

    Returns:
        complex128 array X with the shape of ``rhs``.

    Dynamic inputs:
        operator: square matrix A, shape (n, n).
        rhs: right-hand sides B, shape (n, columns).

    Static configuration:
        none.
    """
    return _native.solve(
        np.ascontiguousarray(operator, dtype=np.complex128),
        np.ascontiguousarray(rhs, dtype=np.complex128),
    )


def eig(
    operator: ArrayLike,
) -> tuple[tuple[NDArray[np.complex128], NDArray[np.complex128]], _native.EigContext]:
    """Eigenvalues and unit right eigenvectors of a complex matrix.

    The largest component of each vector is real and positive. Single modes
    at repeated eigenvalues have no gradient; equal value cotangents with zero
    vector cotangents still give the gradient of their spectral sums.

    Returns:
        (values, vectors): complex128 arrays (n,) and (n, n); column i of
        ``vectors`` belongs to ``values[i]``. ``context.pullback`` takes one
        cotangent for each.

    Dynamic inputs:
        operator: square matrix, shape (n, n).

    Static configuration:
        none.
    """
    values, vectors, context = _native.eig(
        np.ascontiguousarray(operator, dtype=np.complex128)
    )
    return (values, vectors), context


def smatrix_from_array(
    response: ArrayLike, channels: ArrayLike
) -> tuple[NDArray[np.complex128], _native.SMatrixFromArrayContext]:
    """S-matrix of a periodic array from its solved response and its channels.

    Returns:
        complex128 array (2, 2, ports, ports) in the block order of SMatrix.

    Dynamic inputs:
        response: solved periodic response in local multipole channels,
            shape (multipoles, multipoles).
        channels: incidence and emission channels of ``spherical_channels`` or
            ``cylindrical_channels``, shape (2, 2, multipoles, ports).

    Static configuration:
        none.
    """
    return _native.smatrix_from_array(
        np.ascontiguousarray(response, dtype=np.complex128),
        np.ascontiguousarray(channels, dtype=np.complex128),
    )


def smatrix_tr(
    matrices: ArrayLike,
    incident: ArrayLike,
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    *,
    modes: Sequence[tuple[int, int]],
    axis: int = 2,
    poltype: str | None = None,
    modetype: str = "up",
    fixed_q: bool = False,
) -> tuple[NDArray[np.float64], _native.SMatrixTrContext]:
    """Transmitted and reflected power fractions of a planar network.

    Each incident column is one illumination. At normal incidence the
    polarization vectors have no derivative with respect to the direction;
    pass fixed_q=True there.

    Returns:
        float64 array (2, illuminations): transmission, then reflection.

    Dynamic inputs:
        matrices: S-matrix blocks, shape (2, 2, n, n).
        incident: amplitudes, shape (n, illuminations).
        ks: wavenumbers, shape (2, 2): the positive then the negative exterior
            medium (the order of SMatrix.material), pol 0 then 1.
        zs: impedances of the positive and the negative exterior medium,
            shape (2,).
        q: transverse wavevector of each diffraction group, shape (groups, 2).
            Groups need distinct q values.

    Static configuration:
        modes: (group index, pol) of each port; all distinct.
        axis: the normal axis, 0, 1 or 2.
        poltype: "helicity" (default) or "parity".
        modetype: "up" for light from the negative side, "down" for light from
            the positive side.
        fixed_q: hold q fixed.
    """
    poltype = resolve_poltype(poltype)
    if modetype not in ("up", "down"):
        raise ValueError("invalid polarization or propagation direction")
    return _native.smatrix_tr(
        np.asarray(matrices, dtype=np.complex128),
        np.asarray(incident, dtype=np.complex128),
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(zs, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        modes,
        axis,
        poltype == "helicity",
        0 if modetype == "up" else 1,  # direction: 0 up, 1 down
        fixed_q,
    )


def smatrix_illuminate(
    lower: ArrayLike, upper: ArrayLike, up: ArrayLike, down: ArrayLike
) -> tuple[NDArray[np.complex128], _native.SMatrixIlluminateContext]:
    """Outgoing and internal fields of two adjacent stacks lit from both sides.

    The solve computes only these right-hand sides, never the combined
    network.

    Returns:
        complex128 array (4, modes, illuminations): outgoing up, outgoing down,
        then the internal up and down coefficients between the two stacks.

    Dynamic inputs:
        lower: S-matrix blocks of the lower stack, shape (2, 2, modes, modes).
        upper: S-matrix blocks of the upper stack, same shape.
        up: amplitudes travelling up into the lower stack, shape
            (modes, illuminations).
        down: amplitudes travelling down into the upper stack, same shape.

    Static configuration:
        none.
    """
    return _native.smatrix_illuminate(
        np.asarray(lower, dtype=np.complex128),
        np.asarray(upper, dtype=np.complex128),
        np.asarray(up, dtype=np.complex128),
        np.asarray(down, dtype=np.complex128),
    )


def smatrix_periodic(
    smats: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.SMatrixPeriodicContext]:
    """Transfer matrix of one S-matrix cell for periodic repetition.

    Returns:
        complex128 array (2 * n, 2 * n).

    Dynamic inputs:
        smats: S-matrix blocks, shape (2, 2, n, n).

    Static configuration:
        none.
    """
    return _native.smatrix_periodic(np.ascontiguousarray(smats, dtype=np.complex128))


def bands(
    smats: ArrayLike, period: float
) -> tuple[tuple[NDArray[np.complex128], NDArray[np.complex128]], _native.BandsContext]:
    """Bloch wavenumbers along the normal and right eigenvectors of a periodic cell.

    The wavenumbers use the principal logarithm. Derivatives hold its branch
    and the eigenvector pivot phase fixed; single modes at repeated eigenvalues
    have no gradient.

    Returns:
        (wavenumbers, vectors): complex128 arrays (2 * n,) and (2 * n, 2 * n);
        column i of ``vectors`` belongs to ``wavenumbers[i]``.
        ``context.pullback`` takes one cotangent for each.

    Dynamic inputs:
        smats: S-matrix blocks of the cell, shape (2, 2, n, n).
        period: cell length along the normal.

    Static configuration:
        none.
    """
    wavenumbers, vectors, context = _native.bands(
        np.ascontiguousarray(smats, dtype=np.complex128), period
    )
    return (wavenumbers, vectors), context


def spherical_channels(
    basis: SphericalBasis,
    ks: ArrayLike,
    q: ArrayLike,
    polarizations: ArrayLike,
    area: float,
    *,
    poltype: str | None = None,
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.SphericalChannelsContext]:
    """Incidence and emission channels between multipoles and plane-wave ports.

    Index the result as ``channels[role, side]``: role 0 for incidence and 1
    for emission, side 0 for up and 1 for down. Emitted arrays are transposed:
    ``channels[1, side].T`` maps multipoles to outgoing plane waves. At
    exactly normal incidence the azimuth is undefined; pass fixed_q=True for
    gradients there.

    Returns:
        complex128 array (2, 2, len(basis), ports).

    Dynamic inputs:
        basis.positions: positions of the multipoles, shape (positions, 3).
        ks: wavenumbers for pol 0 and 1, shape (2,).
        q: transverse wavevector of each port, shape (ports, 2).
        area: area of the unit cell.

    Static configuration:
        basis: SphericalBasis of the multipoles.
        polarizations: pol index of each port.
        poltype: "helicity" (default) or "parity".
        fixed_q: hold q fixed.
    """
    poltype = resolve_poltype(poltype)
    pols = _polarizations(polarizations)
    return _native.spherical_channels(
        list(basis.modes),
        basis.positions.tolist(),
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        pols,
        area,
        poltype == "helicity",
        fixed_q,
    )


def smatrix_add(
    lower: ArrayLike, upper: ArrayLike
) -> tuple[NDArray[np.complex128], _native.SMatrixAddContext]:
    """Cascade two S-matrices: ``upper`` on the positive side of ``lower``.

    Returns:
        complex128 array (2, 2, n, n).

    Dynamic inputs:
        lower: S-matrix blocks, shape (2, 2, n, n).
        upper: S-matrix blocks, same shape.

    Static configuration:
        none.
    """
    return _native.smatrix_add(
        np.ascontiguousarray(lower, dtype=np.complex128),
        np.ascontiguousarray(upper, dtype=np.complex128),
    )


def fresnel(
    ks: ArrayLike, kzs: ArrayLike, zs: ArrayLike
) -> tuple[NDArray[np.complex128], _native.FresnelContext]:
    """Fresnel blocks of one chiral interface, ordered as ``coeffs.fresnel``.

    Returns:
        complex128 array (2, 2, 2, 2).

    Dynamic inputs:
        ks: wavenumbers, shape (2, 2): negative then positive medium, pol 0
            then 1.
        kzs: normal wavevector components, same shape and order.
        zs: impedances of the negative and the positive medium, shape (2,).

    Static configuration:
        none.
    """
    return _native.fresnel(
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(kzs, dtype=np.complex128).tolist(),
        np.asarray(zs, dtype=np.complex128).tolist(),
    )


def propagation_matrix(
    vectors: ArrayLike, distance: ArrayLike
) -> tuple[NDArray[np.complex128], _native.PropagationMatrixContext]:
    """S-matrix of homogeneous propagation by a Cartesian displacement.

    Returns:
        complex128 array (2, 2, modes, modes).

    Dynamic inputs:
        vectors: complex wavevectors of the upgoing modes, shape (modes, 3),
            with the normal component last.
        distance: displacement in the same axis order, shape (3,).

    Static configuration:
        none.
    """
    return _native.propagation_matrix(
        np.asarray(vectors, dtype=np.complex128).tolist(),
        np.asarray(distance, dtype=np.float64).tolist(),
    )


def mie(
    degree: int,
    x: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.MieContext]:
    """Mie coefficients of a multilayer sphere for one degree, ordered as ``coeffs.mie``.

    Returns:
        complex128 array (2, 2).

    Dynamic inputs:
        x: size parameters k0 * radius of the layers, inside out, shape
            (layers,).
        epsilon: relative permittivities, shape (layers + 1,): the layers
            inside out, then the embedding medium.
        mu: relative permeabilities, same shape and order.
        kappa: chirality parameters, same shape and order.

    Static configuration:
        degree: the degree l.
    """
    return _native.mie(
        degree,
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.ascontiguousarray(mu, dtype=np.complex128),
        np.ascontiguousarray(kappa, dtype=np.complex128),
    )


def mie_cyl(
    kz: float,
    order: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.MieCylContext]:
    """Coefficients of a multilayer cylinder for one order, ordered as ``coeffs.mie_cyl``.

    Returns:
        complex128 array (2, 2).

    Dynamic inputs:
        kz: real axial wavenumber.
        k0: vacuum angular wavenumber.
        radii: layer radii, inside out, shape (layers,).
        epsilon: relative permittivities, shape (layers + 1,): the layers
            inside out, then the embedding medium.
        mu: relative permeabilities, same shape and order.
        kappa: chirality parameters, same shape and order.

    Static configuration:
        order: the order m.
    """
    return _native.mie_cyl(
        kz,
        order,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.ascontiguousarray(mu, dtype=np.complex128),
        np.ascontiguousarray(kappa, dtype=np.complex128),
    )


def sphere(
    lmax: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], _native.SphereContext]:
    """Helicity T-matrix of a multilayer sphere at the origin.

    Returns:
        complex128 array (n, n) with n = 2*lmax*(lmax+2), in the order of
        ``SphericalBasis.default(lmax)``.

    Dynamic inputs:
        k0: vacuum angular wavenumber.
        radii: layer radii, inside out, shape (layers,).
        epsilon: relative permittivities, shape (layers + 1,): the layers
            inside out, then the embedding medium.
        mu: relative permeabilities, same shape; ones when omitted.
        kappa: chirality parameters, same shape; zeros when omitted.

    Static configuration:
        lmax: maximum degree.
    """
    return _native.sphere(
        lmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        *_layers(epsilon, mu, kappa),
    )


def sphere_cluster(
    lmax: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    positions: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.SphereClusterContext]:
    """Coupled T-matrix of homogeneous nonmagnetic spheres in vacuum.

    Returns:
        complex128 array (N, N) with N = spheres * 2*lmax*(lmax+2), sphere by
        sphere, each in the order of ``SphericalBasis.default(lmax)``.

    Dynamic inputs:
        k0: vacuum angular wavenumber.
        radii: sphere radii, shape (spheres,).
        epsilon: relative permittivities, shape (spheres,).
        positions: sphere centres, shape (spheres, 3).

    Static configuration:
        lmax: maximum degree.
    """
    return _native.sphere_cluster(
        lmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.ascontiguousarray(positions, dtype=np.float64),
    )


def particle_cluster(
    local: Sequence[ArrayLike],
    positions: ArrayLike,
    ks: ArrayLike,
    *,
    bases: Sequence[SphericalBasis | CylindricalBasis],
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], _native.ParticleClusterContext]:
    """Coupled T-matrix of particles with different T-matrices.

    Local bases can have different cutoffs and mode subsets; each uses one
    position. The enclosing surfaces of the particles must not overlap.

    Returns:
        complex128 array (N, N), N the sum of the basis lengths, particle by
        particle.

    Dynamic inputs:
        local: one T-matrix per particle, each (len(basis), len(basis)).
            The pullback returns one gradient per particle.
        positions: particle positions, shape (particles, 3).
        ks: wavenumbers for pol 0 and 1, shape (2,).

    Static configuration:
        bases: one global basis per particle, all of one wave family.
        poltype: "helicity" (default) or "parity".
    """
    poltype = resolve_poltype(poltype)
    if len(local) != len(bases) or not bases:
        raise ValueError("one local matrix and basis required per particle")
    family = type(bases[0])
    modes: list[tuple[int, float, int, int]] = []
    arrays = []
    for particle, (value, basis) in enumerate(zip(local, bases, strict=True)):
        if type(basis) is not family or not basis.isglobal:
            raise ValueError("local particles require global bases of one wave family")
        array = np.asarray(value, dtype=np.complex128)
        if array.shape != (len(basis), len(basis)):
            raise ValueError("local matrix shape must match its basis")
        arrays.append(array)
        modes.extend((particle, degree, order, pol) for _, degree, order, pol in basis)
    wave_numbers = _wavenumber_pair(ks)
    points = np.asarray(positions, dtype=np.float64).tolist()
    if family is SphericalBasis:
        return _native.particle_cluster(
            arrays,
            [(p, int(degree), m, pol) for p, degree, m, pol in modes],
            points,
            wave_numbers,
            poltype == "helicity",
        )
    return _native.cylindrical_particle_cluster(
        arrays, modes, points, wave_numbers, poltype == "helicity"
    )


def factor_interaction(
    local: ArrayLike, coupling: ArrayLike
) -> _native.InteractionFactor:
    """Factor I - T C once to solve only requested incident columns.

    Returns:
        an InteractionFactor. ``factor.solve(incident)`` gives the scattered
        coefficients; ``factor.record(incident)`` also returns an
        IlluminateContext.

    Dynamic inputs:
        local: T, shape (n, n).
        coupling: C, shape (n, n). The pullback of ``factor.record`` returns
            gradients for local, coupling and the incident columns.

    Static configuration:
        none.
    """
    return _native.InteractionFactor(
        np.asarray(local, dtype=np.complex128),
        np.asarray(coupling, dtype=np.complex128),
    )


def sphere_cluster_factor(
    lmax: int, k0: float, radii: ArrayLike, epsilon: ArrayLike, positions: ArrayLike
) -> _native.InteractionFactor:
    """Factor I - T C of homogeneous nonmagnetic spheres in vacuum, assembled in Rust.

    The spheres are those of ``sphere_cluster``. The factor never forms the
    block-diagonal local matrix or the coupled T-matrix. For gradients with
    respect to radii, materials and positions use ``iterative.SphereCluster``.
    The context of ``factor.record(incident)`` takes ``pullback_blocks``, which
    returns gradients for the local blocks (one per sphere), the coupling and
    the incident columns.

    Returns:
        an InteractionFactor; see ``factor_interaction``.

    Dynamic inputs:
        none (use ``iterative.SphereCluster`` for gradients).

    Static configuration:
        lmax: maximum degree.
        k0: vacuum angular wavenumber.
        radii: sphere radii, shape (spheres,).
        epsilon: relative permittivities, shape (spheres,).
        positions: sphere centres, shape (spheres, 3).
    """
    return _native.sphere_cluster_factor(
        lmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.asarray(positions, dtype=np.float64),
    )


def factor_interaction_blocks(
    local: Sequence[ArrayLike], coupling: ArrayLike
) -> _native.InteractionFactor:
    """Factor I - T C with the local T-matrix kept as separate dense blocks.

    Returns:
        an InteractionFactor; see ``factor_interaction``.

    Dynamic inputs:
        local: one block per particle, each square.
        coupling: C, shape (n, n), n the sum of the block sizes. The context of
            ``factor.record`` takes ``pullback_blocks``, which returns a list of
            block gradients, the coupling gradient and the incident gradient.

    Static configuration:
        none.
    """
    return _native.InteractionFactor.from_blocks(
        [np.asarray(block, dtype=np.complex128) for block in local],
        np.asarray(coupling, dtype=np.complex128),
    )


def illuminate(
    local: ArrayLike, coupling: ArrayLike, incident: ArrayLike
) -> tuple[NDArray[np.complex128], _native.IlluminateContext]:
    """Solve (I - T C) scattered = T incident for the requested columns.

    It never forms the full coupled T-matrix. ``factor_interaction`` keeps the
    LU factors for several calls.

    Returns:
        complex128 array (n, illuminations).

    Dynamic inputs:
        local: T, shape (n, n).
        coupling: C, shape (n, n).
        incident: incident coefficients, shape (n, illuminations).

    Static configuration:
        none.
    """
    return factor_interaction(local, coupling).record(
        np.asarray(incident, dtype=np.complex128)
    )


def interaction(
    local: ArrayLike, coupling: ArrayLike
) -> tuple[NDArray[np.complex128], _native.InteractionContext]:
    """Coupled T-matrix X = (I - T C)^-1 T.

    Returns:
        complex128 array (n, n).

    Dynamic inputs:
        local: T, shape (n, n).
        coupling: C, shape (n, n).

    Static configuration:
        none.
    """
    return _native.interaction(
        np.ascontiguousarray(local, dtype=np.complex128),
        np.ascontiguousarray(coupling, dtype=np.complex128),
    )


def expansion(
    destination: SphericalBasis | CylindricalBasis,
    source: SphericalBasis | CylindricalBasis,
    ks: ArrayLike,
    *,
    poltype: str | None = None,
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.ExpansionContext]:
    """Expansion matrix between two multipole bases (the treams Expand operator).

    Supports spherical to spherical, cylindrical to cylindrical and cylindrical
    to spherical (regular waves only). The parity convention with cylinders
    needs an achiral medium. For two cylindrical bases,
    ``context.pullback_axial`` appends the gradient of kzs, the sorted distinct
    axial wavenumbers of both bases; each moves all modes with that value
    together.

    Returns:
        complex128 array (len(destination), len(source)).

    Dynamic inputs:
        destination.positions: shape (positions, 3).
        source.positions: shape (positions, 3).
        ks: wavenumbers for pol 0 and 1, shape (2,).

    Static configuration:
        destination: SphericalBasis or CylindricalBasis of the rows.
        source: basis of the columns.
        poltype: "helicity" (default) or "parity".
        singular: expand singular source waves into regular destination waves
            (False: regular into regular).
    """
    poltype = resolve_poltype(poltype)
    wavenumbers = _wavenumber_pair(ks)
    if isinstance(destination, CylindricalBasis) and isinstance(
        source, CylindricalBasis
    ):
        if poltype == "parity" and wavenumbers[0] != wavenumbers[1]:
            raise ValueError("parity requires an achiral medium")
        return _native.cylindrical_expansion(
            list(destination.modes),
            list(source.modes),
            destination.positions.tolist(),
            source.positions.tolist(),
            wavenumbers,
            singular,
        )
    if isinstance(destination, SphericalBasis) and isinstance(source, CylindricalBasis):
        if singular:
            raise ValueError(
                "cylindrical-to-spherical conversion requires regular waves"
            )
        return _native.cw_to_sw(
            list(destination.modes),
            list(source.modes),
            destination.positions.tolist(),
            source.positions.tolist(),
            wavenumbers,
            poltype == "helicity",
        )
    if not isinstance(destination, SphericalBasis) or not isinstance(
        source, SphericalBasis
    ):
        raise ValueError("unsupported wave-family conversion")
    return _native.expansion(
        list(destination.modes),
        list(source.modes),
        destination.positions.tolist(),
        source.positions.tolist(),
        wavenumbers,
        poltype == "helicity",
        singular,
    )


def rotation(
    angles: ArrayLike,
    destination: SphericalBasis | CylindricalBasis,
    source: SphericalBasis | CylindricalBasis | None = None,
) -> tuple[NDArray[np.complex128], _native.RotationContext]:
    """Rotation matrix of z-y-z Euler angles between two multipole bases.

    The positions stay where they are. Cylindrical bases allow only theta=0.

    Returns:
        complex128 array (len(destination), len(source)).

    Dynamic inputs:
        angles: (phi, theta, psi) in radians. The pullback returns three
            floats.

    Static configuration:
        destination: SphericalBasis or CylindricalBasis of the rows.
        source: basis of the same family for the columns; defaults to
            ``destination``.
    """
    source = destination if source is None else source
    values = np.asarray(angles, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("rotation requires three Euler angles")
    args = (
        destination.positions.tolist(),
        source.positions.tolist(),
        (float(values[0]), float(values[1]), float(values[2])),
    )
    if isinstance(destination, SphericalBasis) and isinstance(source, SphericalBasis):
        return _native.rotation(list(destination.modes), list(source.modes), *args)
    if isinstance(destination, CylindricalBasis) and isinstance(
        source, CylindricalBasis
    ):
        return _native.cylindrical_rotation(
            list(destination.modes), list(source.modes), *args
        )
    raise ValueError("rotation bases must belong to the same wave family")


def field_operator(
    points: ArrayLike,
    basis: SphericalBasis | CylindricalBasis,
    ks: ArrayLike,
    *,
    poltype: str | None = None,
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.FieldOperatorContext]:
    """Electric field matrix of every basis mode at every point.

    For a cylindrical basis, ``context.pullback_axial`` appends the real
    gradient of kz, one per mode.

    Returns:
        complex128 array (points, 3, len(basis)).

    Dynamic inputs:
        points: Cartesian points, shape (points, 3).
        basis.positions: shape (positions, 3).
        ks: wavenumbers for pol 0 and 1, shape (2,).

    Static configuration:
        basis: SphericalBasis or CylindricalBasis.
        poltype: "helicity" (default) or "parity".
        singular: singular instead of regular waves.
    """
    poltype = resolve_poltype(poltype)
    args = (
        basis.positions.tolist(),
        np.ascontiguousarray(points, dtype=np.float64),
        _wavenumber_pair(ks),
        poltype == "helicity",
        singular,
    )
    if isinstance(basis, CylindricalBasis):
        return _native.cylindrical_field_operator(list(basis.modes), *args)
    return _native.field_operator(list(basis.modes), *args)


def field(
    coefficients: ArrayLike,
    points: ArrayLike,
    basis: SphericalBasis | CylindricalBasis,
    ks: ArrayLike,
    *,
    poltype: str | None = None,
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.FieldContext]:
    """Electric field of multipole coefficients at Cartesian points.

    Use ``singular=True`` for scattered (outgoing) coefficients; the default
    samples regular waves. ``wave.efield(points)`` chooses this from the kind
    of the wave. The context stores no (points, 3, modes) operator. For a
    cylindrical basis, ``context.pullback_axial`` appends the real gradient of
    kz, one per mode.

    Returns:
        complex128 array (points, 3).

    Dynamic inputs:
        coefficients: multipole amplitudes, shape (len(basis),).
        points: Cartesian points, shape (points, 3).
        basis.positions: shape (positions, 3).
        ks: wavenumbers for pol 0 and 1, shape (2,); both k0 in vacuum.

    Static configuration:
        basis: SphericalBasis or CylindricalBasis.
        poltype: "helicity" (default) or "parity".
        singular: singular instead of regular waves.
    """
    poltype = resolve_poltype(poltype)
    args = (
        basis.positions.tolist(),
        np.ascontiguousarray(coefficients, dtype=np.complex128),
        np.ascontiguousarray(points, dtype=np.float64),
        _wavenumber_pair(ks),
        poltype == "helicity",
        singular,
    )
    if isinstance(basis, CylindricalBasis):
        return _native.cylindrical_field(list(basis.modes), *args)
    return _native.field(list(basis.modes), *args)


def cylinder(
    kzs: ArrayLike,
    mmax: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], _native.CylinderContext]:
    """Helicity T-matrix of a multilayer infinite cylinder along z.

    Returns:
        complex128 array (n, n) with n = 2*len(kzs)*(2*mmax+1), ordered by kz
        as given, then m, then pol 1 before pol 0.

    Dynamic inputs:
        kzs: distinct real axial wavenumbers, shape (axial,).
        k0: vacuum angular wavenumber.
        radii: layer radii, inside out, shape (layers,).
        epsilon: relative permittivities, shape (layers + 1,): the layers
            inside out, then the embedding medium.
        mu: relative permeabilities, same shape; ones when omitted.
        kappa: chirality parameters, same shape; zeros when omitted.

    Static configuration:
        mmax: maximum order.
    """
    return _native.cylinder(
        np.ascontiguousarray(kzs, dtype=np.float64),
        mmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        *_layers(epsilon, mu, kappa),
    )


def plane_phases(
    points: ArrayLike, vectors: ArrayLike
) -> tuple[NDArray[np.complex128], _native.PlanePhasesContext]:
    """Phase factors exp(i k.r) of complex wavevectors at real displacements.

    Axial wavevectors are supported. The context stores only the inputs.

    Returns:
        complex128 array (points, vectors).

    Dynamic inputs:
        points: real displacements, shape (points, 3).
        vectors: complex wavevectors, shape (vectors, 3).

    Static configuration:
        none.
    """
    return _native.plane_phases(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(vectors, dtype=np.complex128),
    )


def plane_field(
    coefficients: ArrayLike | None,
    points: ArrayLike,
    vectors: ArrayLike,
    polarizations: ArrayLike,
    *,
    poltype: str | None = None,
    fixed_vectors: bool = False,
) -> tuple[NDArray[np.complex128], _native.PlaneFieldContext]:
    """Electric field of weighted plane waves, or its operator.

    Returns:
        complex128 array (points, 3), or the operator (points, 3, modes) when
        ``coefficients`` is None.

    Dynamic inputs:
        coefficients: amplitudes, shape (modes,); the gradient is empty for an
            operator.
        points: Cartesian points, shape (points, 3).
        vectors: complex wavevectors, shape (modes, 3).

    Static configuration:
        polarizations: pol index of each mode.
        poltype: "helicity" (default) or "parity".
        fixed_vectors: hold the vectors fixed. Along the polarization axis the
            treams polarization vectors have no direction derivative.
    """
    poltype = resolve_poltype(poltype)
    pols = _polarizations(polarizations)
    return _native.plane_field(
        np.ascontiguousarray(vectors, dtype=np.complex128),
        pols,
        np.ascontiguousarray(points, dtype=np.float64),
        None
        if coefficients is None
        else np.ascontiguousarray(coefficients, dtype=np.complex128),
        poltype == "helicity",
        fixed_vectors,
    )


def plane_expansion(
    destination: SphericalBasis | CylindricalBasis,
    vectors: ArrayLike,
    polarizations: ArrayLike,
    *,
    poltype: str | None = None,
    fixed_vectors: bool = False,
) -> tuple[NDArray[np.complex128], _native.PlaneExpansionContext]:
    """Regular multipole coefficients of plane waves.

    Cylindrical axial components are fixed labels with zero gradients. Along
    the axis, fixed_vectors gives position gradients at fixed incidence.

    Returns:
        complex128 array (len(destination), modes).

    Dynamic inputs:
        destination.positions: shape (positions, 3).
        vectors: complex wavevectors, shape (modes, 3).

    Static configuration:
        destination: SphericalBasis or CylindricalBasis.
        polarizations: pol index of each plane wave.
        poltype: "helicity" (default) or "parity".
        fixed_vectors: hold the vectors fixed.
    """
    poltype = resolve_poltype(poltype)
    pols = _polarizations(polarizations)
    if isinstance(destination, CylindricalBasis):
        return _native.cylindrical_plane_expansion(
            list(destination.modes),
            destination.positions.tolist(),
            np.asarray(vectors, dtype=np.complex128).tolist(),
            pols,
            poltype == "helicity",
            fixed_vectors,
        )
    return _native.plane_expansion(
        list(destination.modes),
        destination.positions.tolist(),
        np.asarray(vectors, dtype=np.complex128).tolist(),
        pols,
        poltype == "helicity",
        fixed_vectors,
    )


def cylindrical_channels(
    basis: CylindricalBasis,
    ks: ArrayLike,
    q: ArrayLike,
    polarizations: ArrayLike,
    period: float,
    *,
    poltype: str | None = None,
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.CylindricalChannelsContext]:
    """Incidence and emission channels of a periodic array of cylinders along x.

    Indexed like ``spherical_channels``; up and down refer to +y and -y.

    Returns:
        complex128 array (2, 2, len(basis), ports).

    Dynamic inputs:
        basis.positions: shape (positions, 3).
        ks: wavenumbers for pol 0 and 1, shape (2,).
        q: (kz, kx) of each zx-aligned port, shape (ports, 2). The kz are
            fixed labels, so the gradient of q[:, 0] is zero.
        period: period along x.

    Static configuration:
        basis: CylindricalBasis of the multipoles.
        polarizations: pol index of each port.
        poltype: "helicity" (default) or "parity".
        fixed_q: also hold kx fixed.
    """
    poltype = resolve_poltype(poltype)
    pols = _polarizations(polarizations)
    return _native.cylindrical_channels(
        list(basis.modes),
        basis.positions.tolist(),
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        pols,
        period,
        poltype == "helicity",
        fixed_q,
    )


def interface_coefficients(
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    *,
    alignment: str = "xy",
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.InterfaceCoefficientsContext]:
    """Blocks of one planar interface for one transverse wavevector.

    Returns:
        complex128 array (2, 2, 2, 2): outgoing and incoming direction, then
        pol 0 and 1.

    Dynamic inputs:
        ks: wavenumbers, shape (2, 2): negative (below) then positive (above)
            medium, pol 0 then 1.
        zs: impedances of the two media, shape (2,).
        q: transverse wavevector, shape (2,).

    Static configuration:
        alignment: "xy", "yz" or "zx", the transverse axes; the remaining axis
            is the normal.
        fixed_q: hold q fixed.
    """
    if alignment not in ("xy", "yz", "zx"):
        raise ValueError("interface alignment must be xy, yz or zx")
    return _native.interface_coefficients(
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(zs, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        ALIGNMENT_AXIS[alignment],
        fixed_q,
    )


def layer_stack(
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    thickness: ArrayLike,
    *,
    alignment: str = "xy",
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.LayerStackContext]:
    """Blocks of planar layers for independent transverse wavevectors.

    Returns:
        complex128 array (channels, 2, 2, 2, 2): per channel the outgoing and
        incoming direction, then pol 0 and 1.

    Dynamic inputs:
        ks: wavenumbers, shape (media, 2), from the negative to the positive
            side, pol 0 then 1.
        zs: impedances, shape (media,).
        q: transverse wavevector of each channel, shape (channels, 2).
        thickness: one thickness per interior medium, shape (media - 2,).

    Static configuration:
        alignment: "xy", "yz" or "zx", the transverse axes.
        fixed_q: hold q fixed.
    """
    if alignment not in ("xy", "yz", "zx"):
        raise ValueError("layer alignment must be xy, yz or zx")
    return _native.layer_stack(
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(zs, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        np.atleast_1d(np.asarray(thickness, dtype=np.float64)).tolist(),
        ALIGNMENT_AXIS[alignment],
        fixed_q,
    )


def periodic_to_cw(
    destination: CylindricalBasis,
    source: SphericalBasis,
    ks: ArrayLike,
    period: float,
    *,
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], _native.PeriodicToCwContext]:
    """Spherical waves of a z-periodic chain radiated as singular cylindrical waves.

    Physical diffraction orders satisfy kz = kpar + 2*pi*n/period; the kz of
    each cylindrical mode has its own gradient.

    Returns:
        complex128 array (len(destination), len(source)).

    Dynamic inputs:
        destination.positions: shape (positions, 3).
        source.positions: shape (positions, 3).
        ks: wavenumbers for pol 0 and 1, shape (2,).
        destination.kz: real axial wavenumber of each cylindrical mode. Its
            gradient comes before that of period.
        period: chain period along z.

    Static configuration:
        destination: CylindricalBasis of the rows.
        source: SphericalBasis of the columns.
        poltype: "helicity" (default) or "parity".
    """
    poltype = resolve_poltype(poltype)
    return _native.periodic_to_cw(
        list(destination.modes),
        list(source.modes),
        destination.positions.tolist(),
        source.positions.tolist(),
        np.asarray(ks, dtype=np.complex128).tolist(),
        period,
        poltype == "helicity",
    )


def plane_permutation(
    vectors: ArrayLike,
    polarizations: ArrayLike,
    n: int = 1,
    *,
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], _native.PlanePermutationContext]:
    """Change of the Cartesian axes of plane waves, x to y to z cyclically.

    The permuted wavevectors are ``np.roll(vectors, n, axis=1)``.

    Returns:
        complex128 array (2, modes): the amplitudes of both output pols for each
        input mode.

    Dynamic inputs:
        vectors: complex wavevectors, shape (modes, 3).

    Static configuration:
        polarizations: pol index of each mode.
        n: number of cyclic permutations.
        poltype: "helicity" (default) or "parity".
    """
    poltype = resolve_poltype(poltype)
    if n != int(n):
        raise ValueError("number of permutations must be integer")
    return _native.plane_permutation(
        np.asarray(vectors, dtype=np.complex128),
        np.asarray(polarizations, dtype=np.float64),
        int(n) % 3,
        poltype == "helicity",
    )


def coordinates(
    points: ArrayLike, *, function: str | None = None, kind: str | None = None
) -> tuple[NDArray[np.float64], _native.CoordinatesContext]:
    """Coordinate conversion of points, such as car2sph.

    Angular derivatives at an axis or at the origin raise ValueError unless
    their cotangent is zero.

    Returns:
        float64 array with the shape of ``points``.

    Dynamic inputs:
        points: last axis of two polar or three spatial components.

    Static configuration:
        function: "car2sph", "sph2car", "car2cyl", "cyl2car", "cyl2sph",
            "sph2cyl", "car2pol" or "pol2car"; ``kind`` is an alias.
    """
    function = _required("function", function, "kind", kind)
    return _native.coordinates_record(np.asarray(points, dtype=np.float64), function)


def vector_coordinates(
    vectors: ArrayLike,
    points: ArrayLike,
    *,
    function: str | None = None,
    kind: str | None = None,
) -> tuple[NDArray[np.complex128], _native.VectorCoordinatesContext]:
    """Vector components converted between coordinate systems.

    Returns:
        complex128 array with the broadcast shape of ``vectors`` and
        ``points``.

    Dynamic inputs:
        vectors: vector components, last axis 2 or 3.
        points: the points in the source coordinates, same last axis.
            Each gradient has the shape of its input.

    Static configuration:
        function: a coordinate conversion such as "car2sph", the name of the
            vector function without its v prefix; ``kind`` is an alias.
    """
    function = _required("function", function, "kind", kind)
    vector = np.asarray(vectors, dtype=np.complex128)
    position = np.asarray(points, dtype=np.float64)
    dim = 2 if function in ("car2pol", "pol2car") else 3
    if (
        not vector.shape
        or not position.shape
        or vector.shape[-1] != dim
        or position.shape[-1] != dim
    ):
        raise ValueError("last axes must match coordinate dimension")
    v, p = np.broadcast_arrays(vector, position)
    return _native.vector_coordinates_record(
        vector if vector.size == dim else v,
        position if position.size == dim else cast("NDArray[np.float64]", p),
        function,
        v.shape,
        (vector.shape, position.shape),
    )


def vector_wave(
    *arguments: ArrayLike,
    function: str,
    degree: ArrayLike = 0,
    order: ArrayLike = 0,
    pol: ArrayLike | None = None,
    polarization: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], _native.VectorWaveContext]:
    """Vector waves and harmonics with a gradient for every continuous argument.

    The arguments are those of the public function without the labels:
    (kr, theta, phi) for spherical waves, (theta, phi) for harmonics,
    (kz, krr, phi, z[, k]) for cylindrical waves and (kx, ky, kz, x, y, z) for
    plane waves.

    Returns:
        complex128 array with the broadcast shape of the arguments and labels,
        plus a last axis of 3 for vector functions.

    Dynamic inputs:
        arguments: one gradient each, with the shape of its argument.

    Static configuration:
        function: a special function name, such as "vsw_rA".
        degree: integer degree l.
        order: integer order m.
        pol: pol index 0 or 1 (default 0); ``polarization`` is an alias.
    """
    if polarization is not None:
        pol = one_of("pol", pol, "polarization", polarization, 0)
    labels = (degree, order, 0 if pol is None else pol)
    if _numbers(arguments):
        # One sample: a single conversion yields every one-element argument.
        values = list(np.array(arguments, dtype=np.complex128).reshape(-1, 1))
        shapes: list[tuple[int, ...]] = [()] * len(values)
        modes, shape = _label_modes(labels, MAX_DEGREE, "wave labels", ())
    else:
        arrays = [np.asarray(v, dtype=np.complex128) for v in arguments]
        shapes = [v.shape for v in arrays]
        modes, shape = _label_modes(labels, MAX_DEGREE, "wave labels", shapes)
        values = _flat(arrays, shape)
    # Three labels make each mode a (degree, order, polarization) triple.
    triples = cast("list[tuple[int, int, int]]", modes)
    return _native.vector_wave_record(function, triples, values, shape, shapes)


def sph_harm(
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    degree: ArrayLike,
    order: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.VectorWaveContext]:
    """Normalized spherical harmonics Y_lm(theta, phi), broadcast.

    Returns:
        complex128 array with the broadcast shape of the inputs.

    Dynamic inputs:
        theta: polar angle.
        phi: azimuthal angle.

    Static configuration:
        degree: integer degree l.
        order: integer order m.
    """
    return vector_wave(theta, phi, function="sph_harm", degree=degree, order=order)


def spherical_translation(
    kr: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    destination: Sequence[ArrayLike],
    source: Sequence[ArrayLike],
    poltype: str | None = None,
    singular: bool = True,
) -> tuple[NDArray[np.complex128], _native.SphericalTranslationContext]:
    """Spherical translation coefficient of one mode pair (``sw.translate``), broadcast.

    Returns:
        complex128 array with the broadcast shape of the inputs.

    Dynamic inputs:
        kr: complex wavenumber times distance.
        theta: polar angle of the displacement.
        phi: azimuthal angle of the displacement.

    Static configuration:
        destination: (degree, order, pol) labels of the destination mode.
        source: (degree, order, pol) labels of the source mode.
        poltype: "helicity" (default) or "parity".
        singular: singular instead of regular translation.
    """
    poltype = resolve_poltype(poltype)
    if len(destination) != 3 or len(source) != 3:
        raise ValueError("each mode requires degree, order and polarization")
    arguments = tuple(np.asarray(v, dtype=np.complex128) for v in (kr, theta, phi))
    rows, shape = _label_modes(
        (*destination, *source), MAX_DEGREE, "mode labels", [v.shape for v in arguments]
    )
    return _native.spherical_translation_record(
        [((r[0], r[1], r[2]), (r[3], r[4], r[5])) for r in rows],
        (
            _pack(arguments[0], shape),
            _pack(arguments[1], shape),
            _pack(arguments[2], shape),
        ),
        poltype == "helicity",
        singular,
        shape,
        (arguments[0].shape, arguments[1].shape, arguments[2].shape),
    )


def cylindrical_translation(
    krr: ArrayLike,
    phi: ArrayLike,
    z: ArrayLike,
    kz: ArrayLike,
    *,
    order: ArrayLike,
    singular: bool = True,
) -> tuple[NDArray[np.complex128], _native.CylindricalTranslationContext]:
    """Cylindrical translation kernel of the order difference, broadcast.

    The two matching axial labels move together with kz; changing which labels
    are equal is a discrete step without a gradient.

    Returns:
        complex128 array with the broadcast shape of the inputs.

    Dynamic inputs:
        krr: radial wavenumber times radial distance.
        phi: azimuthal angle of the displacement.
        z: axial distance.
        kz: axial wavenumber.

    Static configuration:
        order: source order minus destination order.
        singular: singular instead of regular translation.
    """
    arguments = tuple(np.asarray(v, dtype=np.complex128) for v in (krr, phi, z, kz))
    orders, shape = _label_modes(
        (order,), MAX_ORDER, "order differences", [v.shape for v in arguments]
    )
    return _native.cylindrical_translation_record(
        [value for (value,) in orders],
        (
            _pack(arguments[0], shape),
            _pack(arguments[1], shape),
            _pack(arguments[2], shape),
            _pack(arguments[3], shape),
        ),
        singular,
        shape,
        (
            arguments[0].shape,
            arguments[1].shape,
            arguments[2].shape,
            arguments[3].shape,
        ),
    )


def lattice_expansion(
    destination: SphericalBasis | CylindricalBasis,
    source: SphericalBasis | CylindricalBasis,
    ks: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    *,
    poltype: str | None = None,
    eta: complex = 0,
) -> tuple[NDArray[np.complex128], _native.LatticeExpansionContext]:
    """Lattice coupling from singular to regular waves, including nonzero self images.

    The Ewald method splits each lattice sum into a real-space and a
    reciprocal-space series; ``eta`` sets where, and 0 picks it automatically.
    For two cylindrical bases, ``context.pullback_axial`` appends the gradient
    of kzs, the sorted distinct axial wavenumbers of both bases, with matching
    groups fixed.

    Returns:
        complex128 array (len(destination), len(source)).

    Dynamic inputs:
        destination.positions: shape (positions, 3).
        source.positions: shape (positions, 3).
        ks: wavenumbers for pol 0 and 1, shape (2,).
        kpar: Bloch vector.
        a: lattice vectors.

    Static configuration:
        destination: basis of the rows.
        source: basis of the same family for the columns.
        poltype: "helicity" (default) or "parity".
        eta: Ewald split parameter.
    """
    poltype = resolve_poltype(poltype)
    wavenumbers = np.asarray(ks, dtype=np.complex128)
    if wavenumbers.shape != (2,):
        raise ValueError(
            "context requires two medium wavenumbers, one per polarization"
        )
    if poltype == "parity" and wavenumbers[0] != wavenumbers[1]:
        raise ValueError("invalid polarization type for embedding medium")
    matrix, bloch = periodic_geometry(a, kpar, isinstance(source, SphericalBasis))
    pair = (complex(wavenumbers[0]), complex(wavenumbers[1]))
    if isinstance(destination, SphericalBasis) and isinstance(source, SphericalBasis):
        return _native.lattice_expansion(
            list(destination.modes),
            list(source.modes),
            destination.positions.tolist(),
            source.positions.tolist(),
            pair,
            poltype == "helicity",
            bloch,
            matrix,
            eta,
        )
    if isinstance(destination, CylindricalBasis) and isinstance(
        source, CylindricalBasis
    ):
        return _native.cylindrical_lattice_expansion(
            list(destination.modes),
            list(source.modes),
            destination.positions.tolist(),
            source.positions.tolist(),
            pair,
            bloch,
            matrix,
            eta,
        )
    raise ValueError("periodic expansion requires matching wave families")


def lattice_expansion_from_table(
    values: ArrayLike,
    destination: SphericalBasis,
    source: SphericalBasis | None = None,
    *,
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], _native.LatticeExpansionFromTableContext]:
    """Lattice expansion from a table of lattice harmonics computed elsewhere.

    The context stores only the sparse angular map; whatever computes the
    table owns its derivatives. Parity needs one wavenumber channel.

    Returns:
        complex128 array (len(destination), len(source)).

    Dynamic inputs:
        values: the table, shape (destination positions, source positions,
            1 or 2 wavenumber channels, harmonics). The last index is l*l+l+m,
            up to the sum of the largest degrees of both bases.

    Static configuration:
        destination: SphericalBasis of the rows.
        source: SphericalBasis of the columns; defaults to ``destination``.
        poltype: "helicity" (default) or "parity".
    """
    poltype = resolve_poltype(poltype)
    source = destination if source is None else source
    table = np.asarray(values, dtype=np.complex128)
    return _native.lattice_expansion_from_table(
        list(destination.modes),
        list(source.modes),
        destination.positions.tolist(),
        source.positions.tolist(),
        poltype == "helicity",
        table,
    )
