"""Exact adjoint identities of native records from their symmetries.

Maxwell's equations have no length scale: multiplying every length by s and
dividing every wavenumber by s leaves these records unchanged. Differentiating
at s = 1 gives, for any cotangent g, the Euler identity

    sum_i w_i Re(vdot(pullback(g)_i, x_i)) = 0

with w_i = +1 for lengths, -1 for wavenumbers and 0 for dimensionless inputs.
It holds exactly for the complete pullback, so it catches a gradient that the
Python layer reorders, drops, rescales or attaches to a rebuilt basis, without
finite differences. Similarity and unitary orbits give the same kind of
identity for eigenvalues and singular values, and broadcasting records must
agree with their elementwise scalar calls and pullbacks.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import diff

from _support import complex_normal, sum_to

pytestmark = [pytest.mark.gradients, pytest.mark.physics]

LENGTH, WAVENUMBER, FIXED = 1, -1, 0


def _wavenumbers(rng, shape=()):
    return rng.uniform(0.7, 2.0, shape) + 1j * rng.uniform(0, 0.2, shape)


def _separated(rng, count, spacing=1.2):
    """Points at least ``spacing`` apart: jittered sites of a line."""
    offsets = rng.uniform(-0.2, 0.2, (count, 3))
    return offsets + np.outer(np.arange(count), [0.3, -0.2, 1.0]) * spacing


def _materials(rng, layers):
    epsilon = rng.uniform(1.5, 4, layers + 1) + 1j * rng.uniform(0, 0.2, layers + 1)
    mu = rng.uniform(0.9, 1.5, layers + 1) + 0j
    kappa = rng.uniform(-0.1, 0.1, layers + 1) + 0j
    return epsilon, mu, kappa


def sphere(rng):
    radii = np.sort(rng.uniform(0.1, 0.4, 2))
    return (
        lambda k0, radii, *materials: diff.sphere(2, float(k0), radii, *materials),
        (np.asarray(rng.uniform(0.7, 2.0)), radii, *_materials(rng, 2)),
        (WAVENUMBER, LENGTH, FIXED, FIXED, FIXED),
    )


def cylinder(rng):
    radii = np.sort(rng.uniform(0.1, 0.4, 2))
    return (
        lambda kz, k0, radii, *materials: diff.cylinder(
            kz, 2, float(k0), radii, *materials
        ),
        (
            rng.uniform(-0.5, 0.5, 2),
            np.asarray(rng.uniform(0.7, 2.0)),
            radii,
            *_materials(rng, 2),
        ),
        (WAVENUMBER, WAVENUMBER, LENGTH, FIXED, FIXED, FIXED),
    )


def cluster(rng):
    epsilon = rng.uniform(1.5, 4, 3) + 1j * rng.uniform(0, 0.2, 3)
    return (
        lambda k0, radii, epsilon, positions: diff.sphere_cluster(
            1, float(k0), radii, epsilon, positions
        ),
        (np.asarray(1.1), rng.uniform(0.1, 0.3, 3), epsilon, _separated(rng, 3)),
        (WAVENUMBER, LENGTH, FIXED, LENGTH),
    )


def expansion(rng):
    return (
        lambda destination, source, ks: diff.expansion(
            tr.SphericalBasis.default(2, 2, destination),
            tr.SphericalBasis.default(1, 1, source),
            ks,
            singular=True,
        ),
        (_separated(rng, 2), _separated(rng, 3)[2:] + 1.0, _wavenumbers(rng, 2)),
        (LENGTH, LENGTH, WAVENUMBER),
    )


def field(rng):
    basis = tr.SphericalBasis.default(2, 2)
    return (
        lambda coefficients, points, positions, ks: diff.field(
            coefficients,
            points,
            tr.SphericalBasis(basis.modes, positions),
            ks,
            singular=True,
        ),
        (
            complex_normal(rng, len(basis)),
            _separated(rng, 3)[1:] + 0.5,
            rng.uniform(-0.3, 0.3, (2, 3)),
            _wavenumbers(rng, 2),
        ),
        (FIXED, LENGTH, LENGTH, WAVENUMBER),
    )


def plane_phases(rng):
    return (
        diff.plane_phases,
        (rng.normal(size=(3, 3)), complex_normal(rng, (4, 3))),
        (LENGTH, WAVENUMBER),
    )


def propagation(rng):
    return (
        diff.propagation_matrix,
        (complex_normal(rng, (3, 3)), rng.normal(size=3)),
        (WAVENUMBER, LENGTH),
    )


def layer_stack(rng):
    # Wavenumbers and impedances of three media and two in-plane wavevectors.
    return (
        diff.layer_stack,
        (
            _wavenumbers(rng, (3, 2)),
            rng.uniform(0.5, 1.5, 3) + 1j * rng.uniform(0, 0.1, 3),
            rng.uniform(-0.4, 0.4, (2, 2)),
            rng.uniform(0.1, 0.8, 1),
        ),
        (WAVENUMBER, FIXED, WAVENUMBER, LENGTH),
    )


def periodic_expansion(rng):
    # Pullback order: destination and source positions, ks, Bloch vector, cell.
    basis = tr.SphericalBasis.default(1)

    def record(destination, source, ks, kpar, cell):
        value, context = diff.lattice_expansion(
            tr.SphericalBasis(basis.modes, destination),
            tr.SphericalBasis(basis.modes, source),
            ks,
            kpar,
            cell,
        )
        return value, context

    return (
        record,
        (
            rng.uniform(-0.2, 0.2, (1, 3)),
            rng.uniform(-0.2, 0.2, (1, 3)),
            _wavenumbers(rng) * np.ones(2),
            rng.uniform(-0.3, 0.3, 2),
            np.diag(rng.uniform(1.5, 2.5, 2))
            + np.triu(rng.uniform(-0.2, 0.2, (2, 2)), 1),
        ),
        (LENGTH, LENGTH, WAVENUMBER, WAVENUMBER, LENGTH),
    )


RECORDS = {
    builder.__name__: builder
    for builder in (
        sphere,
        cylinder,
        cluster,
        expansion,
        field,
        plane_phases,
        propagation,
        layer_stack,
        periodic_expansion,
    )
}


@given(
    name=st.sampled_from(sorted(RECORDS)),
    seed=st.integers(0, 2**16),
    scale=st.floats(0.5, 2.0),
)
def test_scale_invariance_and_euler_identity(name, seed, scale):
    rng = np.random.default_rng(seed)
    record, inputs, weights = RECORDS[name](rng)
    value, context = record(*inputs)
    scaled = tuple(x * scale**w for x, w in zip(inputs, weights, strict=True))
    assert_allclose(record(*scaled)[0], value, rtol=0, atol=1e-11 * abs(value).max())
    gradients = context.pullback(complex_normal(rng, np.shape(value)))
    terms = [
        w * np.vdot(g, x).real
        for g, x, w in zip(gradients, inputs, weights, strict=True)
        if w
    ]
    assert abs(sum(terms)) <= 1e-11 * max(abs(t) for t in terms)


def _well_separated(rng, n):
    return np.diag(np.arange(1.0, n + 1)) + 0.15 * complex_normal(rng, (n, n))


def _anti_hermitian(rng, n):
    x = complex_normal(rng, (n, n))
    return x - x.conj().T


@given(n=st.integers(1, 6), seed=st.integers(0, 2**16))
def test_eigenvalue_sum_is_the_trace_and_similarity_invariant(n, seed):
    rng = np.random.default_rng(seed)
    a = _well_separated(rng, n)
    (_, vectors), context = diff.eig(a)
    # d(sum of eigenvalues) = d trace A, whose gradient is the identity.
    assert_allclose(
        context.pullback(np.ones(n, complex), np.zeros_like(vectors)),
        np.eye(n),
        atol=1e-13,
    )
    # Eigenvalues are invariant under A -> exp(X) A exp(-X): dA = XA - AX.
    weights = complex_normal(rng, n)
    _, context = diff.eig(a)
    gradient = context.pullback(weights, np.zeros_like(vectors))
    x = complex_normal(rng, (n, n))
    change = x @ a - a @ x
    scale = np.linalg.norm(gradient) * np.linalg.norm(change)
    assert abs(np.vdot(gradient, change).real) <= 1e-13 * scale


@given(m=st.integers(1, 5), n=st.integers(1, 5), seed=st.integers(0, 2**16))
def test_singular_values_are_unitarily_invariant(m, n, seed):
    rng = np.random.default_rng(seed)
    a = complex_normal(rng, (m, n))
    values, context = diff.svdvals(a)
    gradient = context.pullback(rng.normal(size=values.shape))
    # U A V for unitary U and V keeps the singular values: dA = X A + A Y.
    change = _anti_hermitian(rng, m) @ a + a @ _anti_hermitian(rng, n)
    scale = np.linalg.norm(gradient) * np.linalg.norm(change)
    assert abs(np.vdot(gradient, change).real) <= 1e-13 * scale


# Broadcasting ---------------------------------------------------------------------


def _complex(low, high, spread):
    """Sampler of complex arrays with real parts in [low, high]."""
    return lambda rng, shape: (
        rng.uniform(low, high, shape) + 1j * rng.uniform(-spread, spread, shape)
    )


def _choice(*values):
    return lambda rng, shape: rng.choice(values, shape)


def _angle(low, high):
    return lambda rng, shape: rng.uniform(low, high, shape)


#: name -> (argument samplers, record, number of trailing inputs with gradients)
BROADCASTS = {
    "bessel": (
        (_choice(0.0, 0.5, 1.0, 2.0, 3.5), _complex(0.3, 4, 1)),
        diff.bessel,
        1,
    ),
    "spherical-hankel-derivative": (
        (_choice(0.0, 1.0, 2.0, 3.0), _complex(0.3, 4, 1)),
        lambda n, z: diff.bessel(n, z, function="h1", spherical=True, derivative=True),
        1,
    ),
    "incgamma": (
        (_choice(-2.5, -0.5, 0.5, 1.0, 2.5), _complex(0.3, 3, 0.5)),
        diff.incgamma,
        1,
    ),
    "intkambe": (
        (
            _choice(-3, -2, -1, 0, 1, 3),
            _complex(0.5, 1.5, 0.1),
            _complex(0.8, 1.4, 0.1),
        ),
        diff.intkambe,
        2,
    ),
    "legendre": (
        (_choice(1, 2, 3, 4), _choice(-1, 0, 1), _complex(-0.9, 0.9, 0.2)),
        diff.angular,
        1,
    ),
    "tau": (
        (_choice(1, 2, 3, 4), _choice(-1, 0, 1), _complex(-0.9, 0.9, 0.2)),
        lambda degree, order, z: diff.angular(degree, order, z, function="tau"),
        1,
    ),
    "wigner": (
        (
            _choice(1, 2, 3),
            _choice(-1, 0, 1),
            _choice(-1, 0, 1),
            _angle(-3, 3),
            _angle(0.1, 3),
            _angle(-3, 3),
        ),
        diff.wignerd,
        3,
    ),
    "sph_harm": (
        (_angle(0.1, 3), _angle(-3, 3)),
        lambda theta, phi: diff.sph_harm(theta, phi, degree=2, order=-1),
        2,
    ),
    "vsw_rA": (
        (_complex(0.5, 4, 0.3), _angle(0.1, 3), _angle(-3, 3)),
        lambda *x: diff.vector_wave(*x, function="vsw_rA", degree=2, order=1),
        3,
    ),
    "spherical_translation": (
        (_complex(0.5, 4, 0.3), _angle(0.1, 3), _angle(-3, 3)),
        lambda *x: diff.spherical_translation(
            *x, destination=(2, 1, 0), source=(1, 0, 1), poltype="parity"
        ),
        3,
    ),
    "cylindrical_translation": (
        (_complex(0.5, 4, 0.3), _angle(-3, 3), _angle(-1, 1), _angle(-0.5, 0.5)),
        lambda *x: diff.cylindrical_translation(*x, order=1),
        4,
    ),
}


def _gradients(context, cotangent):
    # Contexts take arrays, 0-d included; vector waves return a list of
    # per-argument gradients.
    gradients = context.pullback(np.asarray(cotangent))
    return tuple(gradients) if isinstance(gradients, tuple | list) else (gradients,)


@given(data=st.data(), name=st.sampled_from(sorted(BROADCASTS)))
def test_broadcasting_is_elementwise_and_pullbacks_sum_over_it(data, name):
    # Values are the elementwise scalar calls (their own native entry points)
    # and each gradient is the sum of the elementwise gradients over the axes
    # its argument was broadcast along: an exact linear identity.
    samplers, record, count = BROADCASTS[name]
    shapes = data.draw(
        hnp.mutually_broadcastable_shapes(
            num_shapes=len(samplers), max_dims=3, max_side=3, min_side=0
        )
    )
    rng = np.random.default_rng(data.draw(st.integers(0, 2**16)))
    arrays = [
        sample(rng, shape)
        for sample, shape in zip(samplers, shapes.input_shapes, strict=True)
    ]
    value, context = record(*arrays)
    broadcast = shapes.result_shape
    assert value.shape[: len(broadcast)] == broadcast
    cotangent = complex_normal(rng, value.shape)
    gradients = _gradients(context, cotangent)
    assert len(gradients) == count
    expected = [np.zeros(broadcast, complex) for _ in range(count)]
    scale = max(1.0, abs(value).max(initial=0))
    for index in np.ndindex(broadcast):
        scalars = [np.broadcast_to(a, broadcast)[index].item() for a in arrays]
        element, local = record(*scalars)
        assert_allclose(value[index], element, rtol=1e-12, atol=1e-13 * scale)
        for total, gradient in zip(
            expected, _gradients(local, cotangent[index]), strict=True
        ):
            total[index] = gradient
    for gradient, total, shape in zip(
        gradients, expected, shapes.input_shapes[-count:], strict=True
    ):
        assert np.shape(gradient) == shape
        assert_allclose(
            gradient,
            sum_to(total, shape),
            rtol=1e-12,
            atol=1e-12 * max(1.0, abs(total).max(initial=0)),
        )
