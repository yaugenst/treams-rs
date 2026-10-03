"""Translations: the native single-pair jet and the operators of every basis family.

Checks the Cartesian single-pair jet _native.cartesian_translation_jet against
treams.sw.translate, with its position derivative, unit invariance and near-axis
limit, and operators.translate against treams for every basis family, with physical
plane-wave identities, reciprocity and the diff.expansion and plane-phase pullbacks.
The same coefficients are checked through special in test_translation_coefficients
and through sw.translate in test_wave_namespaces.
"""

from itertools import pairwise, product

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr

# _native: the single-pair translation jet is a test hook without a public wrapper.
from treams_rs import _native, diff
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal, reciprocal

positive = st.floats(0.2, 3.0, allow_nan=False, allow_infinity=False)


@pytest.mark.reference
@given(
    degree=st.integers(1, 4),
    order=st.integers(-1, 1),
    # Upstream angular functions lose small transverse components through cos(theta).
    # The independent resolved-angle limit below covers the near-axis regime.
    x=st.one_of(st.just(0.0), st.floats(-1.0, -0.02), st.floats(0.02, 1.0)),
    z=positive,
    singular=st.booleans(),
)
def test_translation_reference(degree, order, x, z, singular):
    displacement = np.array([x, 0.0, z])
    spherical = treams.special.car2sph(displacement)
    to, source = (degree, order, 1), (2, -1, 1)
    actual = _native.cartesian_translation_jet(
        to, source, 1.2 + 0.1j, tuple(displacement), True, singular
    )[0]
    expected = treams.sw.translate(
        *to,
        *source,
        (1.2 + 0.1j) * spherical[0],
        spherical[1],
        spherical[2],
        singular=singular,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-11)


MODES = list(product(range(1, 4), range(-1, 2), (0, 1)))


@pytest.mark.gradients
@pytest.mark.parametrize(
    "position,singular,step,atol,rtol",
    [
        ((0.0, 0.0, 0.0), False, 1e-6, 3e-10, 2e-8),
        ((0.0, 0.0, 1.0), False, 1e-5, 2e-6, 2e-7),
        ((0.0, 0.0, 1.0), True, 1e-5, 2e-6, 2e-7),
        ((0.2, -0.5, 1.0), True, 1e-5, 2e-6, 2e-7),
        ((0.0, 0.0, -1.0), True, 1e-5, 2e-6, 2e-7),
    ],
)
def test_translation_position_derivative(position, singular, step, atol, rtol):
    # Every mode pair with l <= 3, |m| <= 1, both polarizations and both
    # conventions, at the origin (regular only), on and off the z axis.
    k = 1.2 + 0.1j
    for to, source, helicity in product(MODES, MODES, (False, True)):
        _, derivative, _ = _native.cartesian_translation_jet(
            to, source, k, position, helicity, singular
        )
        for axis, shift in enumerate(np.eye(3) * step):
            plus, minus = (
                _native.cartesian_translation_jet(
                    to, source, k, tuple(position + sign * shift), helicity, singular
                )[0]
                for sign in (1, -1)
            )
            np.testing.assert_allclose(
                derivative[axis], (plus - minus) / (2 * step), atol=atol, rtol=rtol
            )


@pytest.mark.physics
@pytest.mark.parametrize("singular", [False, True])
@given(
    exponent=st.integers(-150, 150),
    degrees=st.tuples(st.integers(1, 12), st.integers(1, 12)),
    orders=st.tuples(st.floats(-1, 1), st.floats(-1, 1)),
)
def test_translation_is_invariant_under_a_change_of_units(
    singular, exponent, degrees, orders
):
    # (k, d) -> (k s, d / s) leaves the coefficient unchanged, scales the displacement
    # gradient by s and the wavenumber derivative by 1/s. Dividing r^p P_p^m by r^p
    # would return NaN once r^p under- or overflows, e.g. at SI-scale displacements.
    to, source = ((n, round(f * n), 1) for n, f in zip(degrees, orders, strict=True))
    k, d = 1.2 + 0.1j, (0.3, -0.4, 1.1)
    value, gradient, dk = _native.cartesian_translation_jet(
        to, source, k, d, True, singular
    )
    s = 10.0**exponent
    scaled = _native.cartesian_translation_jet(
        to, source, k * s, tuple(x / s for x in d), True, singular
    )
    scale = abs(value) + np.abs(gradient).sum() + abs(k * dk)
    np.testing.assert_allclose(scaled[0], value, rtol=0, atol=1e-11 * scale)
    np.testing.assert_allclose(
        np.array(scaled[1]) / s, gradient, rtol=0, atol=1e-11 * scale
    )
    np.testing.assert_allclose(k * s * scaled[2], k * dk, rtol=0, atol=1e-11 * scale)


@pytest.mark.reference
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("z", [-0.5, 0.5])
@given(exponent=st.integers(8, 300), y_fraction=st.floats(-1, 1))
def test_near_axis_translation_against_resolved_angle_limit(
    singular, z, exponent, y_fraction
):
    # For this azimuthal difference the translation is (x-i*y)*slope + O(rho**3).
    # Extrapolate the slope from four resolvable upstream angles, where its
    # associated Legendre evaluation has not rounded to the axis.
    k = 1.2 + 0.1j
    estimates = []
    for h in (0.01, 0.005, 0.0025, 0.00125):
        spherical = treams.special.car2sph([h, 0, z])
        estimates.append(
            treams.sw.translate(
                4,
                0,
                1,
                2,
                -1,
                1,
                k * spherical[0],
                spherical[1],
                spherical[2],
                singular=singular,
            )
            / h
        )
    for order in range(1, 4):
        estimates = [
            (4**order * b - a) / (4**order - 1) for a, b in pairwise(estimates)
        ]
    slope = estimates[0]
    scale = 10.0 ** (-exponent)
    value, gradient, _ = _native.cartesian_translation_jet(
        (4, 0, 1), (2, -1, 1), k, (scale, scale * y_fraction, z), True, singular
    )
    np.testing.assert_allclose(
        value / scale, slope * (1 - 1j * y_fraction), rtol=2e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        gradient[:2], [slope, -1j * slope], rtol=2e-9, atol=1e-12
    )


@pytest.mark.reference
@pytest.mark.parametrize("family", ["sw", "cw", "unit", "xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_translation_reference_all_families(family, poltype):
    positions = [[0.1, 0.2, -0.1], [-0.3, 0.1, 0.4]]
    pairs = []
    for package in (tr, treams):
        if family == "sw":
            source = (
                tr.SphericalBasis if package is tr else treams.SphericalWaveBasis
            ).default(2, 2, positions)
            destination = (
                tr.SphericalBasis if package is tr else treams.SphericalWaveBasis
            )(list(source)[::-2], positions)
        elif family == "cw":
            source = (
                tr.CylindricalBasis if package is tr else treams.CylindricalWaveBasis
            ).default([0.2, -0.3], 2, 2, positions)
            destination = (
                tr.CylindricalBasis if package is tr else treams.CylindricalWaveBasis
            )(list(source)[::-2], positions)
        elif family == "unit":
            source = (
                tr.PlaneWaveBasis
                if package is tr
                else treams.PlaneWaveBasisByUnitVector
            ).default([[0.2, 0.3, 1], [0.1j, 0.2, 1]])
            destination = (
                tr.PlaneWaveBasis
                if package is tr
                else treams.PlaneWaveBasisByUnitVector
            )(list(source)[::-2])
        else:
            source = (
                tr.PlaneWavePorts if package is tr else treams.PlaneWaveBasisByComp
            ).default([[0.2, 0.3], [2.1, 0.1]], family)
            destination = (
                tr.PlaneWavePorts if package is tr else treams.PlaneWaveBasisByComp
            )(list(source)[::-2], family)
        pairs.append((destination, source))
    r = np.array([[[0.1, 0.2, -0.3], [-0.2, 0.1, 0.3], [0, 0, 0]]])
    mask = np.random.default_rng(97).random((len(pairs[0][0]), len(pairs[0][1]))) > 0.2
    common = dict(
        k0=1.3,
        material=(1.4 + 0.1j, 1.2, 0.04 if poltype == "helicity" else 0),
        poltype=poltype,
        modetype="down",
        where=mask,
    )
    actual = tr.operators.translate(r, basis=pairs[0], **common)
    expected = treams.translate(r, basis=pairs[1], **common)
    assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.physics
@pytest.mark.parametrize("family", ["sw", "cw"])
@settings(max_examples=20)
@given(
    seed=st.integers(0, 2**32 - 1),
    poltype=st.sampled_from(["helicity", "parity"]),
    points=st.integers(1, 3),
)
def test_multipole_translation_displaces_each_particle_alike(
    family, seed, poltype, points
):
    # translate(r) pairs equal particle indices and moves every destination
    # origin to r relative to its source: the pidx-masked expansion between the
    # bases with all origins at r and at zero, for rectangular multi-particle pairs.
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(3, 3))
    full = (
        tr.SphericalBasis.default(2, 3, positions)
        if family == "sw"
        else tr.CylindricalBasis.default([0.1, -0.3], 2, 3, positions)
    )
    destination, source = (
        full[np.sort(rng.choice(len(full), rng.integers(1, len(full)), replace=False))]
        for _ in range(2)
    )
    offsets = rng.normal(size=(points, 3))
    options = dict(k0=1.3, material=(1.4 + 0.1j, 1.2), poltype=poltype)
    actual = tr.operators.translate(offsets, basis=(destination, source), **options)
    origin = type(source)(source.modes, np.zeros_like(source.positions))
    for value, offset in zip(actual, offsets, strict=True):
        shifted = type(destination)(
            destination.modes, np.broadcast_to(offset, destination.positions.shape)
        )
        expected = tr.operators.expand((shifted, origin), **options) * (
            destination.pidx[:, None] == source.pidx
        )
        assert_allclose(value, expected, rtol=0, atol=0)


@pytest.mark.physics
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("side", ["up", "down"])
@given(scale=st.floats(0.01, 100), phase=st.floats(-2, 2))
def test_plane_basis_conversion_and_translation_preserve_fields(
    poltype, side, scale, phase
):
    k0, material = 1.3 / scale, (1.4, 1.2, 0.04 if poltype == "helicity" else 0)
    source = tr.PlaneWavePorts.default(np.array([[0.2, 0.3], [2.1, -0.1]]) / scale)
    full = source.byunitvector(k0, material, side)
    destination = tr.PlaneWaveBasis(full.modes[::-1])
    assert_allclose(
        tr.operators.expand(source, side, k0=k0, material=material, poltype=poltype),
        np.eye(len(source)),
        atol=0,
    )
    mapping = tr.operators.expand(
        (destination, source), side, k0=k0, material=material, poltype=poltype
    )
    assert_allclose(mapping, np.eye(len(source))[::-1], atol=0)
    assert_allclose(
        tr.operators.expand(
            (source, destination), side, k0=k0, material=material, poltype=poltype
        )
        @ mapping,
        np.eye(len(source)),
        atol=0,
    )
    amplitudes = np.array([0.3, 0.1j, -0.2, 0.4 + 0.1j]) * np.exp(1j * phase)
    points = np.array([[0.2, 0.1, 0.3], [-0.1, 0.3, -0.2]]) * scale
    shift = np.array([0.1, -0.2, 0.3]) * scale
    options = dict(k0=k0, material=material, poltype=poltype, modetype=side)
    moved = (
        tr.operators.translate(shift, basis=(destination, source), **options)
        @ amplitudes
    )
    expected = tr.operators.efield(points + shift, basis=source, **options) @ amplitudes
    assert_allclose(
        tr.operators.efield(points, basis=destination, **options) @ moved,
        expected,
        rtol=2e-12,
        atol=2e-12,
    )
    zero = tr.operators.expand(
        (source, source), ("up", "down"), k0=k0, material=material, poltype=poltype
    )
    assert_allclose(zero, 0, atol=0)


@pytest.mark.physics
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(
    position=st.tuples(st.floats(0.5, 1.5), st.floats(-1, 1), st.floats(-1, 1)),
    loss=st.floats(0, 0.3),
    singular=st.booleans(),
)
def test_expansion_and_translation_reciprocity(
    cylindrical, poltype, position, loss, singular
):
    # Lorentz reciprocity, E = P E^T P, holds at every truncation, in lossy and
    # chiral media, between origins and for regular and singular waves alike.
    positions = [[0, 0, 0], position, [-0.3, 0.6, 0.4]]
    family, labels = (
        (tr.CylindricalBasis, ([0.2, -0.2], 2))
        if cylindrical
        else (tr.SphericalBasis, (2,))
    )
    basis = family.default(*labels, 3, positions)
    material = (2 + loss * 1j, 1.1, 0.05 if poltype == "helicity" else 0)
    ks = tr.Material(material).ks(1.3)
    value = tr.diff.expansion(basis, basis, ks, singular=singular, poltype=poltype)[0]
    expected = reciprocal(value, basis.modes, cylindrical=cylindrical)
    assert_allclose(value, expected, rtol=0, atol=1e-13 * np.abs(value).max())
    # Reversing a translation of one origin gives its reciprocal, and in the
    # parity basis spatial inversion multiplies each wave by (-1)^(l + [TE]).
    basis = family.default(*labels)
    options = dict(basis=basis, k0=1.3, material=material, poltype=poltype)
    forward = tr.operators.translate(position, **options)
    backward = tr.operators.translate(-np.array(position), **options)
    tolerance = 1e-13 * np.abs(forward).max()
    expected = reciprocal(forward, basis.modes, cylindrical=cylindrical)
    assert_allclose(backward, expected, rtol=0, atol=tolerance)
    if poltype == "parity" and not cylindrical:
        parity = np.array(
            [(-1.0) ** (degree + (pol == 0)) for _, degree, _, pol in basis]
        )
        expected = parity[:, None] * forward * parity
        assert_allclose(backward, expected, rtol=0, atol=tolerance)


@pytest.mark.gradients
@pytest.mark.parametrize("singular", [False, True])
def test_expansion_pullback(singular):
    def record(destination, source, ks):
        return diff.expansion(
            tr.SphericalBasis.default(1, 2, destination),
            tr.SphericalBasis.default(2, positions=source),
            ks,
            singular=singular,
        )

    values = (
        np.array([[0.2, -0.1, 0.3], [0.2, 1.1, 0.4]]),
        np.array([[-0.1, 0.2, 1.1]]),
        np.array([1.1 + 0.02j, 1.2 + 0.1j]),
    )
    check_pullback(record, *values, rtol=3e-7, atol=3e-6)
    # Moving every origin together leaves the expansion unchanged.
    value, context = record(*values)
    cotangent = complex_normal(np.random.default_rng(21), value.shape)
    destination, source, _ = context.pullback(cotangent)
    np.testing.assert_allclose(
        destination.sum(axis=0) + source.sum(axis=0), 0, atol=1e-9
    )


# Components down to tiny magnitudes (test_tiny_displacement_translation_is_continuous).
coordinates = st.one_of(
    st.just(0.0), st.floats(-0.2, 0.2), st.floats(-1e-3, 1e-3), st.floats(-1e-40, 1e-40)
)
small_displacements = st.tuples(coordinates, coordinates, coordinates).map(np.array)


@pytest.mark.physics
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(
    a=small_displacements,
    b=small_displacements,
    k0=st.floats(0.5, 1.5),
    epsilon=st.floats(1.0, 3.0),
)
@settings(max_examples=15)
def test_spherical_regular_translation_composition(poltype, a, b, k0, epsilon):
    # Regular translations form a group: translating by b and then by a through
    # an intermediate basis of degree 12 equals translating by a + b. The
    # truncation error is of order j_13(k |a|) < 1e-15 for k |a| < 1.
    small, big = tr.SphericalBasis.default(2), tr.SphericalBasis.default(12)
    material = (epsilon + 0.1j, 1, 0.1 if poltype == "helicity" else 0)
    options = dict(k0=k0, material=material, poltype=poltype)
    composed = tr.operators.translate(
        a, basis=(small, big), **options
    ) @ tr.operators.translate(b, basis=(big, small), **options)
    expected = tr.operators.translate(a + b, basis=small, **options)
    assert_allclose(composed, expected, rtol=0, atol=2e-14)


@pytest.mark.physics
@pytest.mark.parametrize("displacement", [[0, 0, 6.5e-53], [1e-20, 0, 1e-20]])
def test_tiny_displacement_translation_is_continuous(displacement):
    # |d|^p underflows from degree 6 (6.5e-53) or 10 (1e-20) on; the angular
    # factor on the unit sphere keeps the coefficients near the identity.
    basis = tr.SphericalBasis([(10, m, 1) for m in range(-10, 11)])
    options = dict(k0=1.0, material=1, poltype="helicity")
    actual = tr.operators.translate(displacement, basis=basis, **options)
    assert_allclose(actual, np.eye(len(basis)), rtol=0, atol=1e-13)


@pytest.mark.gradients
@given(scale=st.floats(0.1, 3), imaginary=st.floats(-0.2, 0.2))
def test_plane_phase_values_and_all_pullbacks(scale, imaginary):
    points = np.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]]) * scale
    vectors = (
        np.array(
            [[1.2 + imaginary * 1j, 0.1j, 0.3], [0, 0, 1.3 + imaginary * 1j], [0, 0, 0]]
        )
        / scale
    )
    value = tr.diff.plane_phases(points, vectors)[0]
    assert_allclose(value, np.exp(1j * points @ vectors.T), rtol=1e-13, atol=1e-13)
    g = np.arange(6).reshape(3, 2).T.astype(complex) + 0.2j
    check_pullback(
        tr.diff.plane_phases,
        points,
        vectors,
        directions=(np.full(points.shape, 0.1), np.full(vectors.shape, 0.03 + 0.02j)),
        cotangents=g,
        step=1e-5,
        rtol=3e-8,
        atol=3e-9,
    )


@pytest.mark.gradients
@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
def test_parallel_plane_phase_contractions(layout):
    rng = np.random.default_rng(817)
    points = rng.normal(size=(257, 3)) * 0.2
    vectors = rng.normal(size=(17, 3)) + 0.1j * rng.normal(size=(17, 3))
    expected = np.exp(1j * points @ vectors.T)
    value, context = tr.diff.plane_phases(points, vectors)
    raw = complex_normal(rng, (257, 34))
    g = raw[:, ::2] if layout == "strided" else raw[:, :17]
    g = (
        g[::-1]
        if layout == "reversed"
        else np.array(g, order=layout)
        if layout in ("C", "F")
        else g
    )
    gp, gk = context.pullback(g)
    assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
    assert_allclose(
        gp, ((g.conj() * 1j * expected) @ vectors).real, rtol=2e-12, atol=2e-12
    )
    assert_allclose(gk, (g * (1j * expected).conj()).T @ points, rtol=2e-12, atol=2e-12)


@pytest.mark.gradients
def test_advect_plane_translation_and_amplitudes():
    points = np.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]])
    vectors = np.array([[1.2 + 0.1j, 0, 0.3], [0, 0, 1.3 - 0.1j]])
    coefficients = np.array([0.2 + 0.1j, -0.3 + 0.4j])

    def objective(points, vectors, coefficients):
        value = ad.plane_phases(points, vectors) @ coefficients
        return anp.sum(anp.real(value * anp.conj(value)))

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2)),
        points,
        vectors,
        coefficients,
        directions=(
            np.full(points.shape, 0.1),
            np.full(vectors.shape, 0.02 + 0.03j),
            np.array([0.04, 0.03j]),
        ),
        step=1e-5,
        rtol=3e-8,
        atol=3e-9,
    )


@pytest.mark.gradients
@pytest.mark.interface
def test_plane_phase_gradients_at_the_origin_and_for_no_points():
    value, context = tr.diff.plane_phases([[0, 0, 0]], [[0, 0, 1]])
    gp, gk = context.pullback(np.ones_like(value))
    assert_allclose(gp, 0, atol=0)
    assert_allclose(gk, 0, atol=0)
    empty, context = tr.diff.plane_phases(np.empty((0, 3)), [[0, 0, 1]])
    assert empty.shape == (0, 1)
    assert context.pullback(empty)[0].shape == (0, 3)
