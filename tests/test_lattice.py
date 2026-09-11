# ruff: noqa: E741 - conventional multipole degree l
"""Ewald sums: independent direct sums, split independence and Bloch symmetry."""

import itertools

import numpy as np
import pytest
import scipy.special as sp
import treams.lattice as oracle
import treams.special as osc
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import _native

pytestmark = pytest.mark.filterwarnings(
    "ignore:`scipy.special.sph_harm` is deprecated:DeprecationWarning"
)


@pytest.mark.parametrize("n", [-10, -3, -1.5, -0.5, 0, 0.5, 1, 4, 10])
@pytest.mark.parametrize(
    "z", [0.3, 1.5, 2 + 4j, 14 + 2j, complex(-3, 0.0), complex(-3, -0.0)]
)
def test_gamma_reference(n, z):
    assert_allclose(_native.incgamma(n, z), osc.incgamma(n, z), rtol=2e-10, atol=1e-14)


@pytest.mark.parametrize("n", [-10, -9, -4, -3, -2, -1, 0, 1, 9, 10])
@pytest.mark.parametrize(
    "z,eta", [(0.8, 1.2), (1 + 1j, 2 - 1.1j), (1.4 + 0.01j, 3 - 0.01j), (3, 2)]
)
def test_kambe_reference(n, z, eta):
    assert_allclose(
        _native.intkambe(n, z, eta), osc.intkambe(n, z, eta), rtol=2e-8, atol=1e-13
    )


def evaluate(spherical, dim, l, m, k, q, a, r, eta=0):
    return _native.lattice_sum(
        spherical,
        [(l, m)],
        k,
        np.atleast_1d(q).tolist(),
        np.atleast_2d(a).tolist(),
        tuple(r),
        eta,
    )[0]


def reference(spherical, dim, l, m, k, q, a, r, eta=0):
    if spherical:
        if dim == 1:
            return oracle.lsumsw1d_shift(l, m, k, q[0], a[0, 0], r, eta)
        if dim == 2:
            return oracle.lsumsw2d_shift(l, m, k, q, a, r, eta)
        return oracle.lsumsw3d(l, m, k, q, a, r, eta)
    if dim == 1:
        return oracle.lsumcw1d_shift(m, k, q[0], a[0, 0], r[:2], eta)
    return oracle.lsumcw2d(m, k, q, a, r[:2], eta)


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
@pytest.mark.parametrize("l,m", [(0, 0), (1, -1), (2, 0), (3, 1), (4, -2)])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("k", [2.1, 2.1 + 0.2j])
def test_lattice_reference(spherical, dim, l, m, shift, k):
    a = np.diag(np.linspace(1.4, 1.8, dim))
    if dim > 1:
        a[1, 0] = 0.2
    q = np.linspace(0.1, 0.3, dim)
    r = np.array([0.16, -0.11, 0.08] if shift else [0.0, 0.0, 0.0])
    if not spherical:
        r[2] = 0
    actual = evaluate(spherical, dim, l, m, k, q, a, r)
    expected = reference(spherical, dim, l, m, k, q, a, r)
    assert_allclose(actual, expected, rtol=3e-8, atol=3e-9)


@settings(max_examples=24, deadline=None)
@given(
    spherical=st.booleans(),
    dim=st.integers(1, 2),
    order=st.integers(0, 3),
    pitch=st.floats(1.3, 2.0),
    bloch=st.floats(0.05, 0.35),
    scale=st.floats(0.8, 1.25),
)
def test_bloch_scale_split_and_inversion(spherical, dim, order, pitch, bloch, scale):
    a = np.diag([pitch] * dim)
    q = np.array([bloch] * dim)
    r = np.array([0.19, 0.13, 0.07 if spherical else 0.0])
    k = 2.2 + 0.3j
    l, m = order, order
    axes = [2] if spherical and dim == 1 else list(range(dim))
    offset = np.zeros(3)
    offset[axes] = a[0]
    value = evaluate(spherical, dim, l, m, k, q, a, r, 1.2)
    assert_allclose(
        evaluate(spherical, dim, l, m, k, q, a, r, 1.6), value, rtol=3e-8, atol=3e-9
    )
    assert_allclose(
        evaluate(spherical, dim, l, m, k, q, a, r + offset, 1.2),
        np.exp(-1j * np.dot(q, a[0])) * value,
        rtol=3e-10,
        atol=3e-10,
    )
    assert_allclose(
        evaluate(spherical, dim, l, m, k / scale, q / scale, a * scale, r * scale, 1.2),
        value,
        rtol=3e-10,
        atol=3e-10,
    )
    assert_allclose(
        evaluate(spherical, dim, l, m, k, -q, a, -r, 1.2),
        (-1) ** order * value,
        rtol=3e-10,
        atol=3e-10,
    )


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
def test_absolutely_convergent_direct_sum(spherical, dim):
    a = np.diag([1.7] * dim)
    q = np.array([0.12] * dim)
    r = np.array([0.21, 0.13, -0.12 if spherical else 0.0])
    k = 2.0 + 1.2j
    axes = [2] if spherical and dim == 1 else list(range(dim))
    total = 0j
    l, m = 2, -1
    for point in itertools.product(range(-13, 14), repeat=dim):
        vector = np.array(point) @ a
        shift = -r.copy()
        shift[axes] -= vector
        radius = np.linalg.norm(shift)
        phi = np.arctan2(shift[1], shift[0])
        if spherical:
            theta = np.arctan2(np.hypot(*shift[:2]), shift[2])
            term = (
                np.sqrt(np.pi / (2 * k * radius)) * sp.hankel1(l + 0.5, k * radius)
            ) * sp.sph_harm_y(l, m, theta, phi)
        else:
            term = sp.hankel1(m, k * radius) * np.exp(1j * m * phi)
        total += term * np.exp(1j * np.dot(q, vector))
    assert_allclose(
        evaluate(spherical, dim, l, m, k, q, a, r), total, rtol=2e-9, atol=2e-10
    )


def test_threshold_and_invalid_lattice():
    with pytest.raises(ValueError, match="threshold"):
        evaluate(False, 1, 0, 0, 1, [1], [[1.5]], [0.1, 0.1, 0])
    with pytest.raises(ValueError, match="independent"):
        evaluate(True, 2, 1, 0, 2, [0, 0], [[1, 1], [1, 1]], [0, 0, 0])


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_periodic_spherical_coupling_and_solve(dim, poltype):
    import treams

    from treams_rs import SphericalWaveBasis, TMatrix, lattice

    positions = np.array([[0, 0, 0], [0.31, 0.12, 0.21]])
    basis = SphericalWaveBasis.default(2, positions=positions)
    ks = (
        np.array([1.9 + 0.08j, 2.1 + 0.09j])
        if poltype == "helicity"
        else np.array([2.0 + 0.08j] * 2)
    )
    a = np.diag([1.5] * dim)
    q = np.array([0.11] * dim)
    expected = treams.sw.translate_periodic(
        ks, q, a, positions, np.array(basis.modes).T, poltype=poltype
    )
    actual = lattice.expansion(basis, basis, ks, a, q, poltype=poltype)
    assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)

    local = TMatrix.cluster(
        [
            TMatrix.sphere(1, 2, [0.1], [2, 1], poltype),
            TMatrix.sphere(2, 2, [0.12], [3, 1], poltype),
        ],
        positions,
    )
    upstream = treams.TMatrix.cluster(
        [
            treams.TMatrix.sphere(1, 2, [0.1], [2, 1], poltype),
            treams.TMatrix.sphere(2, 2, [0.12], [3, 1], poltype),
        ],
        positions,
    )
    response = local.latticeinteraction.solve(a, q)
    assert_allclose(
        response, upstream.latticeinteraction.solve(a, q), rtol=3e-8, atol=1e-11
    )
    assert_allclose(
        local.latticeinteraction(a, q) @ response, local.array, rtol=3e-10, atol=1e-14
    )


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_periodic_cylindrical_coupling_and_solve(dim, poltype):
    import treams

    from treams_rs import CylindricalWaveBasis, TMatrixC, lattice

    positions = np.array([[0, 0, 0], [0.31, 0.12, 0.21]])
    basis = CylindricalWaveBasis.default([0, 0.4], 2, positions=positions)
    ks = [1.9 + 0.08j, 2.1 + 0.09j] if poltype == "helicity" else [2.0 + 0.08j] * 2
    a = np.diag([1.5] * dim)
    q = np.array([0.11] * dim)
    oracle_a, oracle_q = (float(a[0, 0]), float(q[0])) if dim == 1 else (a, q)
    actual = lattice.expansion(basis, basis, ks, a, q, poltype=poltype)
    expected = treams.cw.translate_periodic(
        ks, oracle_q, oracle_a, positions, (basis.pidx, basis.kz, basis.m, basis.pol)
    )
    assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)

    local = TMatrixC.cylinder([0.4], 2, 2, [0.1], [2, 1], poltype)
    upstream = treams.TMatrixC.cylinder([0.4], 2, 2, [0.1], [2, 1])
    if poltype == "parity":
        upstream = upstream.changepoltype("parity")
    response = local.latticeinteraction.solve(a, q)
    assert_allclose(
        response,
        upstream.latticeinteraction.solve(oracle_a, oracle_q),
        rtol=3e-8,
        atol=1e-11,
    )
    assert_allclose(
        local.latticeinteraction(a, q) @ response, local.array, rtol=3e-10, atol=1e-14
    )


def test_public_lattice_mode_broadcasting():
    from treams_rs import lattice

    l = np.array([1, 2, 3])
    m = np.array([[-1], [0], [1]])
    actual = lattice.lsumsw2d(l, m, 2, [0.1, 0.2], [[1.5, 0], [0.1, 1.6]], [0.1, -0.2])
    assert_allclose(
        actual,
        oracle.lsumsw2d(l, m, 2, [0.1, 0.2], [[1.5, 0], [0.1, 1.6]], [0.1, -0.2], 0),
        rtol=2e-8,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
@pytest.mark.parametrize("l,m", [(0, 0), (1, -1), (2, 0), (3, 1)])
@pytest.mark.parametrize("origin", [True, False])
def test_all_continuous_lattice_derivatives(spherical, dim, l, m, origin):
    a = np.diag(np.linspace(1.5, 1.8, dim))
    if dim > 1:
        a[1, 0] = 0.17
    q = np.zeros(dim) if origin else np.linspace(0.1, 0.3, dim)
    r = np.zeros(3) if origin else np.array([0.21, -0.14, 0.17 if spherical else 0])
    k = 2.1 + 0.2j
    value, dk, dr, dq, da = _native.lattice_derivatives(
        spherical, (l, m), k, q.tolist(), a.tolist(), tuple(r), 1.2
    )
    assert_allclose(
        value, evaluate(spherical, dim, l, m, k, q, a, r, 1.2), rtol=2e-9, atol=2e-9
    )
    qdir = np.linspace(0.1, -0.2, dim)
    adir = np.arange(1, 1 + dim * dim).reshape(dim, dim) * 0.05
    rdir = np.array([0.12, 0.09, -0.1 if spherical else 0])
    directions = [
        (0.3 + 0.1j, np.zeros(dim), np.zeros_like(a), np.zeros(3), dk * (0.3 + 0.1j)),
        (0, qdir, np.zeros_like(a), np.zeros(3), np.dot(dq[:dim], qdir)),
        (
            0,
            np.zeros(dim),
            adir,
            np.zeros(3),
            np.sum(np.asarray(da)[:dim, :dim] * adir),
        ),
    ]
    if not origin:
        directions.append((0, np.zeros(dim), np.zeros_like(a), rdir, np.dot(dr, rdir)))
    for kdir, qdir, adir, rdir, analytical in directions:
        h = 2e-5
        plus = evaluate(
            spherical,
            dim,
            l,
            m,
            k + h * kdir,
            q + h * qdir,
            a + h * adir,
            r + h * rdir,
            1.2,
        )
        minus = evaluate(
            spherical,
            dim,
            l,
            m,
            k - h * kdir,
            q - h * qdir,
            a - h * adir,
            r - h * rdir,
            1.2,
        )
        assert_allclose(analytical, (plus - minus) / (2 * h), rtol=2e-7, atol=2e-8)
    euler = (
        -k * dk
        - np.dot(q, dq[:dim])
        + np.dot(r, dr)
        + np.sum(a * np.asarray(da)[:dim, :dim])
    )
    assert_allclose(euler, 0, atol=2e-9 * (1 + abs(value)))


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
@pytest.mark.parametrize("equal_wavenumbers", [False, True])
def test_periodic_matrix_pullback(spherical, dim, equal_wavenumbers):
    from treams_rs import CylindricalWaveBasis, SphericalWaveBasis, lattice

    basis = (
        SphericalWaveBasis.default(1)
        if spherical
        else CylindricalWaveBasis.default([0.3], 1)
    )
    destination = type(basis)(basis.modes, [[0.21, 0.12, 0.15]])
    source = type(basis)(basis.modes, [[0, 0, 0]])
    ks = np.array([2.0 + 0.1j, (2.0 if equal_wavenumbers else 2.1) + 0.1j])
    q = np.linspace(0.1, 0.2, dim)
    a = np.diag([1.6] * dim)
    value, context = lattice.expansion_with_context(
        destination, source, ks, a, q, eta=1.2
    )
    rng = np.random.default_rng(21)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = context.pullback(g)
    parameters = [destination.positions, source.positions, ks, q, a]
    directions = [rng.normal(size=x.shape) * 0.1 for x in parameters]
    directions[2] = directions[2] + 0.12j
    h = 2e-5

    def function(values):
        return lattice.expansion(
            type(basis)(basis.modes, values[0]),
            type(basis)(basis.modes, values[1]),
            values[2],
            values[4],
            values[3],
            eta=1.2,
        )

    for axis, (gradient, direction) in enumerate(
        zip(gradients, directions, strict=True)
    ):
        plus, minus = list(parameters), list(parameters)
        plus[axis] = plus[axis] + h * direction
        minus[axis] = minus[axis] - h * direction
        numerical = np.vdot(g, (function(plus) - function(minus)) / (2 * h)).real
        assert_allclose(
            np.vdot(gradient, direction).real, numerical, rtol=2e-7, atol=2e-7
        )
    assert_allclose(gradients[0].sum(axis=0) + gradients[1].sum(axis=0), 0, atol=1e-12)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@settings(max_examples=8, deadline=None)
@given(
    spherical=st.booleans(),
    pitch=st.floats(1.4, 1.9),
    radius=st.floats(0.1, 0.22),
    bloch=st.floats(0.0, 0.3),
)
def test_advect_periodic_complete_solve(spherical, pitch, radius, bloch):
    import advect
    import advect.numpy as anp

    from treams_rs import CylindricalWaveBasis, SphericalWaveBasis
    from treams_rs import advect as ad

    basis = (
        SphericalWaveBasis.default(1)
        if spherical
        else CylindricalWaveBasis.default([0.3], 1)
    )
    positions = np.array([[0.1, 0.03, 0.02]])

    def objective(radii, vectors, q):
        local = (
            ad.sphere(1, 2.0, radii, np.array([3.0 + 0.1j, 1.0]))
            if spherical
            else ad.cylinder(
                np.array([0.3]), 1, 2.0, radii, np.array([3.0 + 0.1j, 1.0])
            )
        )
        coupling = ad.lattice_expansion(
            positions,
            positions,
            np.array([2.0, 2.0]),
            q,
            vectors,
            destination=basis,
            source=basis,
        )
        effective = ad.interaction(local, coupling)
        return anp.sum(anp.real(effective * anp.conj(effective))) + 0.07 * anp.sum(
            anp.real(effective)
        )

    values = [np.array([radius]), np.diag([pitch, pitch * 1.1]), np.array([bloch, 0.1])]
    gradients = advect.grad(objective, argnums=(0, 1, 2))(*values)
    directions = [
        np.array([0.2]),
        np.array([[0.1, 0.03], [-0.02, -0.1]]),
        np.array([0.07, -0.1]),
    ]
    h = 1e-5
    plus = [v + h * d for v, d in zip(values, directions, strict=True)]
    minus = [v - h * d for v, d in zip(values, directions, strict=True)]
    actual = sum(np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True))
    assert_allclose(
        actual, (objective(*plus) - objective(*minus)) / (2 * h), rtol=2e-6, atol=1e-10
    )


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
@pytest.mark.parametrize("m", [-1, 0, 1])
def test_coincident_origin_regular_image_derivative(spherical, dim, m):
    a = np.diag([1.7] * dim)
    q = np.linspace(0.1, 0.2, dim)
    k = 2.0 + 1.2j
    indices = np.array(list(itertools.product(range(-13, 14), repeat=dim)))
    indices = indices[np.any(indices != 0, axis=1)]
    vectors = indices @ a
    embedded = np.zeros((len(vectors), 3))
    axes = [2] if spherical and dim == 1 else list(range(dim))
    embedded[:, axes] = vectors
    phase = np.exp(1j * (vectors @ q))

    def direct(r):
        shift = -embedded - r
        radius = np.linalg.norm(shift, axis=1)
        phi = np.arctan2(shift[:, 1], shift[:, 0])
        if spherical:
            theta = np.arctan2(np.hypot(shift[:, 0], shift[:, 1]), shift[:, 2])
            wave = (
                np.sqrt(np.pi / (2 * k * radius))
                * sp.hankel1(1.5, k * radius)
                * sp.sph_harm_y(1, m, theta, phi)
            )
        else:
            wave = sp.hankel1(m, k * radius) * np.exp(1j * m * phi)
        return np.sum(wave * phase)

    result = _native.lattice_derivatives(
        spherical, (1, m), k, q.tolist(), a.tolist(), (0, 0, 0), 1.2
    )
    assert_allclose(result[0], direct(np.zeros(3)), rtol=2e-8, atol=1e-10)
    for axis in range(3 if spherical else 2):
        step = np.eye(3)[axis] * 1e-5
        assert_allclose(
            result[2][axis],
            (direct(step) - direct(-step)) / (2e-5),
            rtol=2e-7,
            atol=2e-9,
        )


@pytest.mark.parametrize("period", [7.2, 12.8])
def test_large_cylindrical_cell_ewald_split_invariance(period):
    # Upstream's automatic split loses 0.00676 at this order/displacement.
    # The converged reference and our automatic split agree independently.
    import treams_rs as tr

    k = np.sqrt(1.3**2 - 0.2**2)
    automatic = tr.lattice.lsumcw(1, -6, k, 0.1, period, [0.8, 0])
    expected = oracle.lsumcw1d(-6, k, 0.1, period, 0.8, 0.7)
    for eta in (0.4, 0.7, 1.0):
        actual = tr.lattice.lsumcw(1, -6, k, 0.1, period, [0.8, 0], eta=eta)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        assert_allclose(actual, automatic, rtol=1e-12, atol=1e-10)
