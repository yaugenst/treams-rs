"""Advect adapter: physics objects and records differentiated by Advect, on the
CPU and in first-order reverse mode only. Each gradient pass uses the data
stored by its forward pass once, so call the transformed objective again for
every optimization step. Inputs may be float64, complex128, float32 or
complex64; the Rust code computes in double precision, outputs are float64 or
complex128, and each gradient has the dtype of its input.

Build every object of a differentiated objective from this namespace::

    import advect as ad
    import advect.numpy as np
    import treams_rs.advect as tr

    def objective(radius):
        sphere = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=radius, material=3.0)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        return sphere.cross_sections(wave).scattering

    value, gradient = ad.value_and_grad(objective)(np.asarray(0.2))

The Rust core computes each gradient analytically with a pullback: a map from
the gradient with respect to an output to the gradients with respect to the
inputs. ``treams_rs.diff`` defines records, contexts and pullbacks.

Install ``treams-rs[advect]``. Forward mode, higher derivatives, staging and
checkpointing are not available. Mode cutoffs, integer labels and topology are
static. Pass inputs as arrays (scalars as ``np.asarray(x)``) and keep traced
values inside Advect: converting them to float or NumPy, or building objects
from the NumPy-only root ``treams_rs`` namespace, loses derivatives.

Advect has no public ``wrap``. Run a custom record through
``treams_rs.jax.wrap`` or ``treams_rs.torch.wrap``, or compose the expert
functions of ``treams_rs.advect``.

Framework adapters guide: https://yaugenst.github.io/treams-rs/differentiation/frameworks/
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import advect as ad
import numpy as np

# Physical objects share one implementation; this namespace selects execution.
from . import _framework, _framework_backend, diff
from ._bases import CylindricalBasis, PlaneWaveBasis, PlaneWavePorts, SphericalBasis
from ._fields import rs_weights
from ._framework_backend import Material, Recorded, material_defaults
from ._framework_smatrix import SMatrix, stack
from ._framework_tmatrix import Cluster, PeriodicResponse, TMatrix, solve_periodic
from ._framework_waves import PlaneWave, PortWave, Wave
from ._lattice import Lattice
from ._polarization import resolve_poltype
from ._records import apply_pullback, run_record
from ._results import BandModes, CrossSections, PowerBalance, ScatteredPorts

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from numpy.typing import ArrayLike, NDArray

    from ._modes import Modes
    from ._records import Pullback, Record


type _Values = tuple[ArrayLike, ...]
type _Forward = Callable[[_Values], tuple[NDArray[np.complex128], Pullback]]


# The Advect primitive behind _operation: one native forward, and its pullback
# kept as Advect's residual for the transpose.
@ad.primitive(static_argnames=("forward",), residual=True)
def _execute(
    values: _Values, *, forward: _Forward
) -> ad.PrimitiveResult[NDArray[np.complex128]]:
    value, pullback = forward(values)
    # Clearing the holder drops the native context even when a failed trace is kept.
    return ad.PrimitiveResult(value, [pullback], release=list.clear)


@_execute.def_transpose
def _transpose(
    cotangent: ArrayLike,
    primals: _Values,
    _output: ArrayLike,
    residual: list[Pullback],
    *,
    forward: _Forward,
) -> _Values:
    # Advect and the native core both use dL = Re(vdot(gradient, dx)), so
    # cotangents stay unconjugated. reshape=True accepts native gradients that
    # store a scalar parameter as shape (1,), such as the layer_stack thickness.
    return apply_pullback(
        residual[0],
        (
            np.ascontiguousarray(cotangent, dtype=np.complex128).reshape(
                np.shape(cotangent)
            ),
        ),
        tuple(np.asarray(primal) for primal in primals),
        conjugate=False,
        reshape=True,
    )


def _operation(
    record: Record,
    *values: ArrayLike,
    shape: tuple[int, ...] | None = None,
    real: bool = False,
) -> Any:
    """Run ``record(*values) -> (output, context or pullback)`` as one primitive.

    The output is complex128, also for a real-valued record. ``real`` returns
    the real part and passes the real part of the cotangent to the pullback.
    ``shape`` is unused: the output comes from the native forward, and
    Backend.apply checks it for the physical objects.
    """

    def forward(primals: _Values) -> tuple[NDArray[np.complex128], Pullback]:
        outputs, pullback, multiple = run_record(record, primals)
        output = np.asarray(outputs if multiple else outputs[0], dtype=np.complex128)
        if not real:
            return output, pullback

        def real_pullback(g: NDArray[np.complex128]) -> Any:
            return pullback(g.real)

        return output, real_pullback

    # Normalize containers before the primitive flattens its dynamic leaves.
    result = _execute(
        tuple(ad.numpy.asarray(value) for value in values), forward=forward
    )
    return ad.numpy.real(result) if real else result


def _with_static(
    record: Record, values: Sequence[Any], *, static: Collection[int]
) -> Any:
    """Differentiate ``record`` in ``values`` except at the indices in ``static``.

    Static values reach the record unchanged and their gradients are dropped.
    ``functools.partial`` binds keyword options; this drops gradients.
    """
    dynamic = [i for i in range(len(values)) if i not in static]

    def dynamic_record(*primals: Any) -> Recorded:
        arguments = list(values)
        for i, value in zip(dynamic, primals, strict=True):
            arguments[i] = value
        output, context = record(*arguments)

        def pullback(g: NDArray[np.complex128]) -> _Values:
            gradients = context.pullback(g)
            return tuple(gradients[i] for i in dynamic)

        return output, pullback

    return _operation(dynamic_record, *(values[i] for i in dynamic))


def _optional(value: ArrayLike | None) -> _Values:
    return () if value is None else (value,)


def incgamma(z: ArrayLike, *, n: ArrayLike) -> NDArray[np.complex128]:
    """Upper incomplete gamma function, differentiable in z; n stays fixed."""
    return _operation(partial(diff.incgamma, n), z)


def intkambe(z: ArrayLike, eta: ArrayLike, *, n: ArrayLike) -> NDArray[np.complex128]:
    """Kambe integral, differentiable in z and eta; n stays fixed."""
    return _operation(partial(diff.intkambe, n), z, eta)


def lattice_sum(
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    *,
    dim: int,
    degree: ArrayLike = 0,
    order: ArrayLike = 0,
    spherical: bool = True,
    part: str = "full",
    shell: ArrayLike = 0,
) -> NDArray[np.complex128]:
    """Lattice sum, differentiable in k, kpar, a, r and eta.

    The wave labels and the direct-shell index stay fixed.
    """
    return _operation(
        partial(
            diff.lattice_sum,
            dim,
            degree,
            order,
            spherical=spherical,
            part=part,
            shell=shell,
        ),
        k,
        kpar,
        a,
        r,
        eta,
    )


def angular(
    z: ArrayLike, *, degree: ArrayLike, order: ArrayLike, function: str = "legendre"
) -> NDArray[np.complex128]:
    """Integer-degree angular functions, differentiable in z."""
    return _operation(partial(diff.angular, degree, order, function=function), z)


def wignerd(
    phi: ArrayLike,
    theta: ArrayLike,
    psi: ArrayLike,
    *,
    degree: ArrayLike,
    row: ArrayLike,
    column: ArrayLike,
) -> NDArray[np.complex128]:
    """Wigner D elements, differentiable in the three complex Euler angles."""
    return _operation(partial(diff.wignerd, degree, row, column), phi, theta, psi)


def chirality_density(
    ks: ArrayLike, normal: ArrayLike, z: ArrayLike = (0.0, 0.0)
) -> NDArray[np.complex128]:
    """Compact up/down/cross chirality coefficients, differentiable in ks, normal and z."""
    return _operation(diff.chirality_density, ks, normal, z)


def oriented_chirality(
    transverse: ArrayLike,
    normal: ArrayLike,
    z: ArrayLike = (0.0, 0.0),
    *,
    polarizations: ArrayLike,
    axis: int = 2,
) -> NDArray[np.complex128]:
    """Signed helicity chirality forms, differentiable in transverse, normal and z."""
    return _operation(
        partial(diff.oriented_chirality, polarizations=polarizations, axis=axis),
        transverse,
        normal,
        z,
    )


def sphere_cluster(
    lmax: int, k0: ArrayLike, radii: ArrayLike, epsilon: ArrayLike, positions: ArrayLike
) -> NDArray[np.complex128]:
    """Differentiable interacting T-matrix of homogeneous spheres in vacuum."""
    return _operation(partial(diff.sphere_cluster, lmax), k0, radii, epsilon, positions)


def particle_cluster(
    local: Sequence[ArrayLike],
    positions: ArrayLike,
    ks: ArrayLike,
    *,
    bases: Sequence[SphericalBasis | CylindricalBasis],
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Interacting T-matrix of particles with their own local matrices.

    Differentiable in the local matrices, the positions and ks.
    """
    poltype = resolve_poltype(poltype)

    def record(positions: Any, ks: Any, *local: Any) -> Recorded:
        value, context = diff.particle_cluster(
            local, positions, ks, bases=bases, poltype=poltype
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            local, positions, ks = context.pullback(g)
            return (positions, ks, *local)

        return value, pullback

    return _operation(record, positions, ks, *local)


def eig(operator: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Eigenvalues and phase-fixed eigenvectors of a complex matrix, differentiable.

    At repeated eigenvalues only equally weighted sums of those eigenvalues,
    without eigenvector dependence, have a gradient. Individual eigenmodes
    have no gradient there.
    """

    def record(matrix: Any) -> Recorded:
        (eigenvalues, eigenvectors), context = diff.eig(matrix)
        return np.vstack((eigenvalues, eigenvectors)), lambda g: context.pullback(
            g[0], g[1:]
        )

    packed = _operation(record, operator)
    return packed[0], packed[1:]


def smatrix_add(lower: ArrayLike, upper: ArrayLike) -> NDArray[np.complex128]:
    """Differentiable Redheffer composition, with arrays shaped (2, 2, n, n)."""
    return _operation(diff.smatrix_add, lower, upper)


def smatrix_illuminate(
    lower: ArrayLike, upper: ArrayLike, up: ArrayLike, down: ArrayLike
) -> NDArray[np.complex128]:
    """Outgoing/internal up/down fields for mode-by-illumination incident arrays."""
    return _operation(diff.smatrix_illuminate, lower, upper, up, down)


def smatrix_periodic(smats: ArrayLike) -> NDArray[np.complex128]:
    """Periodic transfer matrix; its gradient reuses the forward factorization."""
    return _operation(diff.smatrix_periodic, smats)


def bands(
    smats: ArrayLike, period: ArrayLike
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Normal Bloch wavenumbers and eigenvectors, differentiable in smats and period."""

    def record(matrices: Any, p: Any) -> Recorded:
        (wavenumbers, vectors), context = diff.bands(matrices, float(np.asarray(p)))
        return np.vstack((wavenumbers, vectors)), lambda g: context.pullback(
            g[0], g[1:]
        )

    packed = _operation(record, smats, period)
    return packed[0], packed[1:]


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
) -> NDArray[np.complex128]:
    """Differentiable T/R rows for illumination columns and complete port geometry."""
    record = partial(
        diff.smatrix_tr,
        modes=modes,
        axis=axis,
        poltype=resolve_poltype(poltype),
        modetype=modetype,
        fixed_q=fixed_q,
    )
    if fixed_q:
        q = np.asarray(q, dtype=np.float64)
    return _with_static(
        record, (matrices, incident, ks, zs, q), static={4} if fixed_q else ()
    )


def smatrix_circular_dichroism(
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
) -> NDArray[np.complex128]:
    """Circular dichroism in transmission and in total outgoing power.

    Returns the two contrasts (P1 - P0) / (P0 + P1): first of the transmitted
    power, then of the transmitted plus reflected power. P0 belongs to
    ``incident`` and P1 to the same illumination in the opposite helicity (or
    with the sign of pol 0 flipped in the parity convention). One Rust power
    evaluation covers both illuminations, and every input is differentiable.
    """
    poltype = resolve_poltype(poltype)
    incoming = ad.numpy.asarray(incident)
    if incoming.ndim != 1:
        raise ValueError("CD requires one incident mode vector")
    if poltype == "helicity":
        indices = {mode: i for i, mode in enumerate(modes)}
        try:
            opposite = incoming[[indices[(group, 1 - pol)] for group, pol in modes]]
        except KeyError as error:
            raise ValueError("CD requires both helicities of each direction") from error
    else:
        opposite = incoming * np.array([2 * pol - 1 for _, pol in modes])
    power = smatrix_tr(
        matrices,
        ad.numpy.stack((incoming, opposite), axis=1),
        ks,
        zs,
        q,
        modes=modes,
        axis=axis,
        poltype=poltype,
        modetype=modetype,
        fixed_q=fixed_q,
    )
    total = ad.numpy.sum(power, axis=0)
    denominator = ad.numpy.stack((ad.numpy.sum(power[0]), ad.numpy.sum(total)))
    if ad.numpy.any(denominator == 0):
        raise ValueError(
            "CD is undefined for zero summed transmission or outgoing power"
        )
    return (
        ad.numpy.stack((power[0, 1] - power[0, 0], total[1] - total[0])) / denominator
    )


def smatrix_from_array(
    response: ArrayLike, channels: ArrayLike
) -> NDArray[np.complex128]:
    """Differentiable radiation of an effective multipole response into plane waves."""
    return _operation(diff.smatrix_from_array, response, channels)


def spherical_channels(
    positions: ArrayLike,
    ks: ArrayLike,
    q: ArrayLike,
    area: ArrayLike,
    *,
    basis: SphericalBasis,
    polarizations: ArrayLike,
    poltype: str | None = None,
    fixed_q: bool = False,
) -> NDArray[np.complex128]:
    """Differentiable spherical incidence/radiation channels.

    fixed_q treats q as a static constant; this supports exactly normal incidence.
    Direction gradients otherwise require nonzero transverse wavevectors.
    """
    poltype = resolve_poltype(poltype)

    def record(positions: Any, ks: Any, q: Any, area: Any) -> Recorded:
        return diff.spherical_channels(
            SphericalBasis(basis.modes, positions=positions),
            ks,
            q,
            polarizations,
            float(np.asarray(area)),
            poltype=poltype,
            fixed_q=fixed_q,
        )

    if fixed_q:
        q = np.asarray(q, dtype=np.float64)
    return _with_static(record, (positions, ks, q, area), static={2} if fixed_q else ())


def fresnel(ks: ArrayLike, kzs: ArrayLike, zs: ArrayLike) -> NDArray[np.complex128]:
    """Differentiable chiral planar-interface coefficients."""
    return _operation(diff.fresnel, ks, kzs, zs)


def propagation_matrix(
    vectors: ArrayLike, distance: ArrayLike
) -> NDArray[np.complex128]:
    """Differentiable propagation for upgoing wavevectors and a Cartesian displacement."""
    return _operation(diff.propagation_matrix, vectors, distance)


def mie(
    degree: int, sizes: ArrayLike, epsilon: ArrayLike, mu: ArrayLike, kappa: ArrayLike
) -> NDArray[np.complex128]:
    """Differentiable spherical Mie coefficient matrix."""
    return _operation(partial(diff.mie, degree), sizes, epsilon, mu, kappa)


def mie_cyl(
    kz: ArrayLike,
    order: int,
    k0: ArrayLike,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> NDArray[np.complex128]:
    """Differentiable cylindrical Mie coefficient matrix."""

    def record(axial: Any, k: Any, *layers: Any) -> Recorded:
        return diff.mie_cyl(
            float(np.asarray(axial)), order, float(np.asarray(k)), *layers
        )

    return _operation(record, kz, k0, radii, epsilon, mu, kappa)


def rotation(
    angles: ArrayLike,
    *,
    destination: SphericalBasis | CylindricalBasis,
    source: SphericalBasis | CylindricalBasis | None = None,
) -> NDArray[np.complex128]:
    """Euler-angle derivatives; cylindrical theta must remain fixed at zero."""
    # The native gradient is one list of the three Euler-angle derivatives.
    return _operation(
        partial(diff.rotation, destination=destination, source=source), angles
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
) -> NDArray[np.complex128]:
    """Axisymmetric EBCM surface integral, differentiable in radii, slopes, ks and zs."""
    return _operation(
        partial(
            diff.ebcm_qmat,
            theta=theta,
            weights=weights,
            destination=destination,
            source=source,
            singular=singular,
            radial_area_factor=radial_area_factor,
        ),
        radii,
        slopes,
        ks,
        zs,
    )


def tmatrix_metric(
    operator: ArrayLike,
    ks: ArrayLike = (1.0, 1.0),
    *,
    polarizations: ArrayLike,
    metric: str | None = None,
    kind: str | None = None,
) -> NDArray[np.float64]:
    """Global helicity metric of a T-matrix, differentiable in operator and ks.

    ``metric`` is "cd", "db" or "chi", as in ``diff.tmatrix_metric``; ``kind``
    is an alias.
    """

    def record(matrix: Any, wavenumbers: Any) -> Recorded:
        value, context = diff.tmatrix_metric(
            matrix,
            wavenumbers,
            polarizations=polarizations,
            metric=metric,
            kind=kind,
        )
        return value, lambda g: context.pullback(float(g))

    return _operation(record, operator, ks, real=True)


def svdvals(operator: ArrayLike) -> NDArray[np.float64]:
    """Singular values of a complex matrix in descending order, differentiable."""
    return _operation(diff.svdvals, operator, real=True)


def _real_axial(values: ArrayLike, name: str) -> NDArray[np.float64]:
    axial = np.asarray(values)
    if np.iscomplexobj(axial):
        if np.any(axial.imag != 0):
            raise ValueError(f"{name} must be real")
        axial = axial.real
    return np.asarray(axial, dtype=np.float64)


def _dynamic_basis(
    basis: SphericalBasis | CylindricalBasis,
    positions: ArrayLike,
    kz: ArrayLike | None = None,
) -> SphericalBasis | CylindricalBasis:
    """``basis`` at ``positions``, with one axial wavenumber ``kz`` per mode."""
    if kz is None:
        return type(basis)(basis.modes, positions)
    if not isinstance(basis, CylindricalBasis):
        raise ValueError("axial derivatives require a cylindrical basis")
    axial = _real_axial(kz, "kz")
    if axial.shape != (len(basis),):
        raise ValueError("kz must contain one real axial wavenumber per mode")
    result = CylindricalBasis(
        (
            (p, float(value), m, pol)
            for (p, _, m, pol), value in zip(basis.modes, axial, strict=True)
        ),
        positions,
    )
    if len(result) != len(basis):
        raise ValueError("axial wavenumbers must preserve distinct mode labels")
    return result


def _field_record(
    function: Callable[..., Recorded],
    basis: SphericalBasis | CylindricalBasis,
    *,
    poltype: str,
    singular: bool,
    axial: bool,
) -> Record:
    """Record of ``diff.field`` or ``diff.field_operator`` with moving positions.

    The record takes the function's leading arguments, the positions, ``ks``
    and, if ``axial``, one ``kz`` per mode; ``basis`` follows the positions.
    """

    def record(*values: Any) -> Recorded:
        *leading, positions, ks = values[:-1] if axial else values
        value, context = function(
            *leading,
            _dynamic_basis(basis, positions, *values[-1:] if axial else ()),
            ks,
            poltype=poltype,
            singular=singular,
        )
        return value, context.pullback_axial if axial else context.pullback

    return record


def field_operator(
    points: ArrayLike,
    positions: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalBasis | CylindricalBasis,
    poltype: str | None = None,
    singular: bool = False,
    kz: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Full field matrix at the expansion centres ``positions``.

    For a cylindrical basis, ``kz`` (one value per mode, like ``basis.kz``)
    makes the axial wavenumbers differentiable.
    """
    record = _field_record(
        diff.field_operator,
        basis,
        poltype=resolve_poltype(poltype),
        singular=singular,
        axial=kz is not None,
    )
    return _operation(record, points, positions, ks, *_optional(kz))


def field(
    coefficients: ArrayLike,
    points: ArrayLike,
    positions: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalBasis | CylindricalBasis,
    poltype: str | None = None,
    singular: bool = False,
    kz: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Electric field at points, differentiable in every array argument.

    ``positions`` are the expansion centres. For a cylindrical basis, ``kz``
    (one value per mode, like ``basis.kz``) makes the axial wavenumbers
    differentiable.

    For routine scattering prefer ``Cluster(...).scatter(incident).efield(points)``.
    Raw scattered coefficients are singular (outgoing) waves and require
    ``singular=True``; ``singular=False`` means regular incident waves. ``ks`` contains medium
    wavenumbers for negative/positive helicity, both k0 in vacuum. Helicity
    labels do not mean opposite propagation directions or opposite signs of k.

    The raw path agrees with the physical wave API::

        import numpy as np
        import treams_rs.advect as tr

        k0 = 2.0
        sphere = tr.sphere_tmatrix(k0=k0, lmax=2, radius=0.2, material=3.0)
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
        scattered = sphere.scatter(incident)
        points = [[0.4, 0.1, 0.3]]
        raw = tr.field(scattered.coefficients, points, scattered.basis.positions,
                       [k0, k0], basis=scattered.basis, singular=True)
        np.testing.assert_allclose(raw, scattered.efield(points))
    """
    record = _field_record(
        diff.field,
        basis,
        poltype=resolve_poltype(poltype),
        singular=singular,
        axial=kz is not None,
    )
    return _operation(record, coefficients, points, positions, ks, *_optional(kz))


def hfield(
    coefficients: ArrayLike,
    points: ArrayLike,
    positions: ArrayLike,
    ks: ArrayLike,
    impedance: ArrayLike,
    *,
    basis: SphericalBasis | CylindricalBasis,
    poltype: str | None = None,
    singular: bool = False,
    kz: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Magnetic samples, including impedance derivatives through the linear weights.

    Arguments as for ``field``; ``kz`` holds one value per mode.
    """
    # _framework_waves._Fields.hfield applies the same weights to framework
    # arrays. gfield and ffield reuse _fields.rs_weights; H has no NumPy
    # helper there to share.
    poltype = resolve_poltype(poltype)
    weights = 2 * basis.pol - 1 if poltype == "helicity" else 1
    if poltype == "parity":
        basis = type(basis)(
            [(*mode[:3], 1 - mode[3]) for mode in basis.modes], basis.positions
        )
    coefficients = (
        -1j * ad.numpy.asarray(coefficients) * weights / ad.numpy.asarray(impedance)
    )
    return field(
        coefficients,
        points,
        positions,
        ks,
        basis=basis,
        poltype=poltype,
        singular=singular,
        kz=kz,
    )


def gfield(
    pol: int,
    coefficients: ArrayLike,
    points: ArrayLike,
    positions: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalBasis | CylindricalBasis,
    poltype: str | None = None,
    singular: bool = False,
    kz: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Weighted G field samples with upstream scaling, differentiable like ``field``.

    Arguments as for ``field``; ``kz`` holds one value per mode.
    """
    poltype = resolve_poltype(poltype)
    electric, magnetic = rs_weights(pol, basis, poltype)
    value = field(
        ad.numpy.asarray(coefficients) * electric,
        points,
        positions,
        ks,
        basis=basis,
        poltype=poltype,
        singular=singular,
        kz=kz,
    )
    if magnetic:
        value = value + 1j * magnetic * hfield(
            coefficients,
            points,
            positions,
            ks,
            1.0,
            basis=basis,
            poltype=poltype,
            singular=singular,
            kz=kz,
        )
    return value


def ffield(
    pol: int,
    coefficients: ArrayLike,
    points: ArrayLike,
    positions: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalBasis | CylindricalBasis,
    poltype: str | None = None,
    singular: bool = False,
    kz: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Weighted F samples, differentiating the chiral index weights as well.

    Arguments as for ``field``; ``kz`` holds one value per mode.
    """
    poltype = resolve_poltype(poltype)
    if poltype == "helicity":
        weights = ad.numpy.asarray(ks)
        coefficients = (
            ad.numpy.asarray(coefficients)
            * 2
            * weights[basis.pol]
            / ad.numpy.sum(weights)
        )
        ks = weights
    return gfield(
        pol,
        coefficients,
        points,
        positions,
        ks,
        basis=basis,
        poltype=poltype,
        singular=singular,
        kz=kz,
    )


def _group_axial(
    destination: SphericalBasis | CylindricalBasis,
    source: SphericalBasis | CylindricalBasis,
    kzs: ArrayLike | None = None,
) -> tuple[
    NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.intp] | None
]:
    if kzs is None:
        return None, None, None
    if not isinstance(destination, CylindricalBasis) or not isinstance(
        source, CylindricalBasis
    ):
        raise ValueError("axial expansion derivatives require two cylindrical bases")
    groups = np.unique(np.concatenate((destination.kz, source.kz)))
    axial = _real_axial(kzs, "kzs")
    if axial.shape != groups.shape or not np.all(np.isfinite(axial)):
        raise ValueError("kzs must contain one finite real value per axial group")
    if len(np.unique(axial)) != len(groups):
        raise ValueError("axial groups must remain distinct")
    return (
        axial[np.searchsorted(groups, destination.kz)],
        axial[np.searchsorted(groups, source.kz)],
        np.searchsorted(np.sort(axial), axial),
    )


def _axial_pullback(context: Any, order: NDArray[np.intp] | None) -> Pullback:
    """Pullback of an expansion context, with axial gradients in input order.

    ``order`` maps the native axial gradients (sorted groups) to the order of
    the ``kzs`` input; None means no axial input.
    """
    if order is None:
        return context.pullback

    def pullback(g: NDArray[np.complex128]) -> _Values:
        *gradients, axial = context.pullback_axial(g)
        return (*gradients, axial[order])

    return pullback


def expansion(
    destination_positions: ArrayLike,
    source_positions: ArrayLike,
    ks: ArrayLike,
    *,
    destination: SphericalBasis | CylindricalBasis,
    source: SphericalBasis | CylindricalBasis,
    poltype: str | None = None,
    singular: bool = False,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Expansion matrix, differentiable in the positions, ks and optional axial groups.

    ``kzs`` holds the sorted distinct axial wavenumbers of both bases, one value
    per group, like ``kzs`` of ``TMatrixC.cylinder`` in treams (``kz`` of the
    field functions holds one value per mode). One value moves the entire
    matching group in both bases. Values must remain distinct; changing which
    modes couple is a discrete operation.
    """
    poltype = resolve_poltype(poltype)

    def record(
        destination_positions: Any, source_positions: Any, ks: Any, *axial: Any
    ) -> Recorded:
        destination_kz, source_kz, order = _group_axial(destination, source, *axial)
        value, context = diff.expansion(
            _dynamic_basis(destination, destination_positions, destination_kz),
            _dynamic_basis(source, source_positions, source_kz),
            ks,
            poltype=poltype,
            singular=singular,
        )
        return value, _axial_pullback(context, order)

    return _operation(
        record, destination_positions, source_positions, ks, *_optional(kzs)
    )


def cylinder(
    kzs: ArrayLike,
    mmax: int,
    k0: ArrayLike,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Differentiable multilayer/chiral cylinder T-matrix.

    ``kzs`` holds the sorted distinct axial wavenumbers, as in treams'
    ``TMatrixC.cylinder(kzs, ...)``.
    """

    def record(axial: Any, k: Any, *layers: Any) -> Recorded:
        return diff.cylinder(axial, mmax, float(np.asarray(k)), *layers)

    return _operation(
        record, kzs, k0, radii, epsilon, *material_defaults(epsilon, mu, kappa)
    )


def lattice_expansion(
    destination_positions: ArrayLike,
    source_positions: ArrayLike,
    ks: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    *,
    destination: SphericalBasis | CylindricalBasis,
    source: SphericalBasis | CylindricalBasis,
    poltype: str | None = None,
    eta: complex = 0,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Lattice expansion (periodic coupling), differentiable in positions, ks, kpar and a.

    The Ewald split eta is a numerical constant; its exact physical derivative is zero.
    Optional ``kzs`` holds the sorted distinct axial wavenumbers of both bases,
    one value per group, as in ``expansion``. Groups must remain distinct.
    """
    poltype = resolve_poltype(poltype)

    def record(
        destination_positions: Any,
        source_positions: Any,
        ks: Any,
        bloch: Any,
        vectors: Any,
        *axial: Any,
    ) -> Recorded:
        destination_kz, source_kz, order = _group_axial(destination, source, *axial)
        value, context = diff.lattice_expansion(
            _dynamic_basis(destination, destination_positions, destination_kz),
            _dynamic_basis(source, source_positions, source_kz),
            ks,
            bloch,
            vectors,
            poltype=poltype,
            eta=eta,
        )
        return value, _axial_pullback(context, order)

    return _operation(
        record,
        destination_positions,
        source_positions,
        ks,
        kpar,
        a,
        *_optional(kzs),
    )


def plane_phases(points: ArrayLike, vectors: ArrayLike) -> NDArray[np.complex128]:
    """Plane-wave translation phases, differentiable in the points and wavevectors."""
    return _operation(diff.plane_phases, points, vectors)


def plane_field(
    coefficients: ArrayLike | None,
    points: ArrayLike,
    vectors: ArrayLike,
    *,
    polarizations: ArrayLike,
    poltype: str | None = None,
    fixed_vectors: bool = False,
) -> NDArray[np.complex128]:
    """Plane-wave field or field operator, differentiable in amplitudes, points and wavevectors.

    fixed_vectors removes the wavevectors from the differentiable inputs.
    """
    record = partial(
        diff.plane_field,
        polarizations=polarizations,
        poltype=resolve_poltype(poltype),
        fixed_vectors=fixed_vectors,
    )
    static = {
        i
        for i, is_static in ((0, coefficients is None), (2, fixed_vectors))
        if is_static
    }
    return _with_static(record, (coefficients, points, vectors), static=static)


def plane_expansion(
    positions: ArrayLike,
    vectors: ArrayLike,
    *,
    destination: SphericalBasis | CylindricalBasis,
    polarizations: ArrayLike,
    poltype: str | None = None,
    fixed_vectors: bool = False,
) -> NDArray[np.complex128]:
    """Plane-wave expansion into multipoles, differentiable in positions and wavevectors."""
    poltype = resolve_poltype(poltype)

    def record(positions: Any, vectors: Any) -> Recorded:
        return diff.plane_expansion(
            type(destination)(destination.modes, positions),
            vectors,
            polarizations,
            poltype=poltype,
            fixed_vectors=fixed_vectors,
        )

    return _with_static(
        record, (positions, vectors), static={1} if fixed_vectors else ()
    )


def cylindrical_channels(
    positions: ArrayLike,
    ks: ArrayLike,
    kx: ArrayLike,
    period: ArrayLike,
    *,
    basis: CylindricalBasis,
    kz_labels: ArrayLike,
    polarizations: ArrayLike,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Differentiable cylindrical radiation, holding axial mode labels fixed."""
    poltype = resolve_poltype(poltype)

    def record(positions: Any, ks: Any, kx: Any, period: Any) -> Recorded:
        value, context = diff.cylindrical_channels(
            type(basis)(basis.modes, positions),
            ks,
            np.column_stack([kz_labels, kx]),
            polarizations,
            float(np.asarray(period)),
            poltype=poltype,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            go, gk, gq, ga = context.pullback(g)
            return go, gk, gq[:, 1], np.asarray(ga)

        return value, pullback

    return _operation(record, positions, ks, kx, period)


def interface_coefficients(
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    *,
    alignment: str = "xy",
    fixed_q: bool = False,
) -> NDArray[np.complex128]:
    """Coefficients of one planar interface, differentiable in ks, zs and q."""
    return _with_static(
        partial(diff.interface_coefficients, alignment=alignment, fixed_q=fixed_q),
        (ks, zs, q),
        static={2} if fixed_q else (),
    )


def layer_stack(
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    thickness: ArrayLike,
    *,
    alignment: str = "xy",
    fixed_q: bool = False,
) -> NDArray[np.complex128]:
    """Compact S-matrix of a layer stack, differentiable in ks, zs, q and thickness.

    The output holds one (2, 2, 2, 2) block per transverse wavevector.
    """
    return _with_static(
        partial(diff.layer_stack, alignment=alignment, fixed_q=fixed_q),
        (ks, zs, q, thickness),
        static={2} if fixed_q else (),
    )


def periodic_to_cw(
    destination_positions: ArrayLike,
    source_positions: ArrayLike,
    ks: ArrayLike,
    kz: ArrayLike,
    period: ArrayLike,
    *,
    destination: CylindricalBasis,
    source: SphericalBasis,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Periodic spherical-to-cylindrical radiation, including moving Fourier labels.

    ``kz`` holds one axial wavenumber per destination mode, like
    ``destination.kz``.
    """
    poltype = resolve_poltype(poltype)

    def record(
        destination_positions: Any,
        source_positions: Any,
        ks: Any,
        kz: Any,
        period: Any,
    ) -> Recorded:
        modes = [
            (p, float(value), m, pol)
            for (p, _, m, pol), value in zip(
                destination.modes, np.asarray(kz, dtype=np.float64), strict=True
            )
        ]
        return diff.periodic_to_cw(
            type(destination)(modes, destination_positions),
            type(source)(source.modes, source_positions),
            ks,
            float(np.asarray(period)),
            poltype=poltype,
        )

    return _operation(record, destination_positions, source_positions, ks, kz, period)


def plane_permutation(
    vectors: ArrayLike,
    *,
    polarizations: ArrayLike,
    n: int = 1,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Cyclic-axis polarization coefficients, differentiable in the complex wavevectors."""
    return _operation(
        partial(
            diff.plane_permutation,
            polarizations=polarizations,
            n=n,
            poltype=resolve_poltype(poltype),
        ),
        vectors,
    )


def coordinates(
    points: ArrayLike, *, function: str | None = None, kind: str | None = None
) -> NDArray[np.float64]:
    """Coordinate conversion of points, differentiable in the points.

    ``function`` is a conversion such as "car2sph", as in
    ``diff.coordinates``; ``kind`` is an alias.
    """
    return _operation(
        partial(diff.coordinates, function=function, kind=kind), points, real=True
    )


def vector_coordinates(
    vectors: ArrayLike,
    points: ArrayLike,
    *,
    function: str | None = None,
    kind: str | None = None,
) -> NDArray[np.complex128]:
    """Vector-frame conversion, differentiable in the vectors and the points.

    ``function`` is a conversion such as "car2sph", as in
    ``diff.vector_coordinates``; ``kind`` is an alias.
    """
    return _operation(
        partial(diff.vector_coordinates, function=function, kind=kind),
        vectors,
        points,
    )


def vector_wave(
    *arguments: ArrayLike,
    function: str,
    degree: ArrayLike = 0,
    order: ArrayLike = 0,
    pol: ArrayLike | None = None,
    polarization: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Vector waves and harmonics, differentiable in every continuous argument.

    ``pol`` is the pol index 0 or 1 (default 0); ``polarization`` is an alias.
    """
    return _operation(
        partial(
            diff.vector_wave,
            function=function,
            degree=degree,
            order=order,
            pol=pol,
            polarization=polarization,
        ),
        *arguments,
    )


def sph_harm(
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    degree: ArrayLike,
    order: ArrayLike,
) -> NDArray[np.complex128]:
    """Normalized spherical harmonic, differentiable in theta and phi."""
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
) -> NDArray[np.complex128]:
    """Spherical translation coefficients, differentiable in kr, theta and phi."""
    return _operation(
        partial(
            diff.spherical_translation,
            destination=destination,
            source=source,
            poltype=resolve_poltype(poltype),
            singular=singular,
        ),
        kr,
        theta,
        phi,
    )


def cylindrical_translation(
    krr: ArrayLike,
    phi: ArrayLike,
    z: ArrayLike,
    kz: ArrayLike,
    *,
    order: ArrayLike,
    singular: bool = True,
) -> NDArray[np.complex128]:
    """Cylindrical translation coefficients, differentiable in krr, phi, z and kz."""
    return _operation(
        partial(diff.cylindrical_translation, order=order, singular=singular),
        krr,
        phi,
        z,
        kz,
    )


def lattice_expansion_from_table(
    values: ArrayLike,
    *,
    destination: SphericalBasis,
    source: SphericalBasis | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Couple a custom differentiable lattice table to spherical modes, differentiable in ``values``."""
    return _operation(
        partial(
            diff.lattice_expansion_from_table,
            destination=destination,
            source=source,
            poltype=resolve_poltype(poltype),
        ),
        values,
    )


_backend = _framework_backend.Backend(ad.numpy, _operation)


# One shared implementation of the physical constructors, bound to this backend.
# _api and _ops keep their names: their bound methods pickle as references to them.
_api = _framework.Constructors(_backend, __name__)
plane_wave = _api.plane_wave
smatrix = _api.smatrix
wave = _api.wave
tmatrix = _api.tmatrix
sphere_tmatrix = _api.sphere_tmatrix
multilayer_sphere_tmatrix = _api.multilayer_sphere_tmatrix
cylinder_tmatrix = _api.cylinder_tmatrix
multilayer_cylinder_tmatrix = _api.multilayer_cylinder_tmatrix
slab = _api.slab
interface = _api.interface
multilayer_slab = _api.multilayer_slab
propagation = _api.propagation

# The expert operations shared with the other adapters.
_ops = _framework.Operations(_backend, __name__)
solve = _ops.solve
interaction = _ops.interaction
illuminate = _ops.illuminate
sphere = _ops.sphere
bessel = _ops.bessel

# treams calls circular dichroism cd (TMatrix.cd).
smatrix_cd = smatrix_circular_dichroism


__all__ = [
    "BandModes",
    "Cluster",
    "CrossSections",
    "CylindricalBasis",
    "Lattice",
    "Material",
    "PeriodicResponse",
    "PlaneWave",
    "PlaneWaveBasis",
    "PlaneWavePorts",
    "PortWave",
    "PowerBalance",
    "SMatrix",
    "ScatteredPorts",
    "SphericalBasis",
    "TMatrix",
    "Wave",
    "angular",
    "bands",
    "bessel",
    "chirality_density",
    "coordinates",
    "cylinder",
    "cylinder_tmatrix",
    "cylindrical_channels",
    "cylindrical_translation",
    "ebcm_qmat",
    "eig",
    "expansion",
    "ffield",
    "field",
    "field_operator",
    "fresnel",
    "gfield",
    "hfield",
    "illuminate",
    "incgamma",
    "interaction",
    "interface",
    "interface_coefficients",
    "intkambe",
    "lattice_expansion",
    "lattice_expansion_from_table",
    "lattice_sum",
    "layer_stack",
    "mie",
    "mie_cyl",
    "multilayer_cylinder_tmatrix",
    "multilayer_slab",
    "multilayer_sphere_tmatrix",
    "oriented_chirality",
    "particle_cluster",
    "periodic_to_cw",
    "plane_expansion",
    "plane_field",
    "plane_permutation",
    "plane_phases",
    "plane_wave",
    "propagation",
    "propagation_matrix",
    "rotation",
    "slab",
    "smatrix",
    "smatrix_add",
    "smatrix_cd",
    "smatrix_circular_dichroism",
    "smatrix_from_array",
    "smatrix_illuminate",
    "smatrix_periodic",
    "smatrix_tr",
    "solve",
    "solve_periodic",
    "sph_harm",
    "sphere",
    "sphere_cluster",
    "sphere_tmatrix",
    "spherical_channels",
    "spherical_translation",
    "stack",
    "svdvals",
    "tmatrix",
    "tmatrix_metric",
    "vector_coordinates",
    "vector_wave",
    "wave",
    "wignerd",
]
