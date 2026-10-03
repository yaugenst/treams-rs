"""Diffraction orders and spherical radiation channels of periodic arrays: upstream
references, completeness and reciprocity, and the diff.spherical_channels pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import (
    Cluster,
    Material,
    PlaneWavePorts,
    SphericalBasis,
    TMatrix,
    diff,
    solve_periodic,
)
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import (
    assert_unitary_ports,
    complex_normal,
    oracle_smatrix_array,
    varying,
)

#: Rectangular cell of the sphere arrays and their five lowest diffraction orders.
CELL = np.diag([1.7, 1.8])
ORDERS = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])


@pytest.mark.reference
@given(pitch=st.floats(1, 4), cutoff=st.floats(0, 6))
def test_diffraction_basis_reference(pitch, cutoff):
    a = np.diag([pitch, 1.7])
    ours = PlaneWavePorts.diffr_orders([0.1, 0.2], a, cutoff)
    oracle = treams.PlaneWaveBasisByComp.diffr_orders([0.1, 0.2], a, cutoff)
    assert_allclose(ours.kx, oracle.kx)
    assert_allclose(ours.ky, oracle.ky)
    assert_allclose(ours.pol, oracle.pol)


@pytest.mark.physics
@given(
    shear=st.floats(-3, 3),
    cutoff=st.floats(0, 5),
    kpar=st.tuples(st.floats(-1, 1), st.floats(-1, 1)),
)
def test_diffraction_basis_skew_completeness(shear, cutoff, kpar):
    a = np.array([[2.0, shear], [0.0, 1.7]])
    reciprocal = 2 * np.pi * np.linalg.inv(a).T
    basis = PlaneWavePorts.diffr_orders(kpar, a, cutoff)
    # The cutoff bounds the reciprocal vector before the Bloch vector is added.
    labels = (basis.components - kpar) @ a.T / (2 * np.pi)
    orders = np.rint(labels).astype(int)
    assert_allclose(labels, orders, rtol=0, atol=1e-9)
    # Each order carries both polarizations.
    assert sorted(zip(map(tuple, orders), basis.pol, strict=True)) == sorted(
        (order, pol) for order in set(map(tuple, orders)) for pol in (0, 1)
    )
    # |(m, n) @ reciprocal| >= |(m, n)| * (smallest singular value) bounds the
    # integer search for every order within the cutoff.
    bound = int(np.ceil(cutoff / np.linalg.svd(reciprocal, compute_uv=False)[-1]))
    expected = {
        (m, n)
        for m in range(-bound, bound + 1)
        for n in range(-bound, bound + 1)
        if np.linalg.norm(np.array([m, n]) @ reciprocal) <= cutoff
    }
    assert set(map(tuple, orders)) == expected


@pytest.mark.reference
@pytest.mark.parametrize(
    "poltype,medium",
    [(p, m) for m in [1, (2.1 + 0.2j, 1.2)] for p in ["helicity", "parity"]]
    + [("helicity", (2.3 + 0.1j, 1.1, 0.07))],
)
@pytest.mark.parametrize("q", [[[0, 0]], [[0.2, 0.1], [3.7, 0.2], [-0.1, -0.4]]])
def test_channels_reference(poltype, medium, q):
    positions = [[0.1, 0.2, -0.3], [-0.2, 0.1, 0.4]]
    basis = SphericalBasis.default(3, 2, positions)
    sw = treams.SphericalWaveBasis.default(3, 2, positions)
    pw = treams.PlaneWaveBasisByComp.default(q)
    plane = PlaneWavePorts.default(q)
    qs = np.column_stack([plane.kx, plane.ky])
    value, _ = diff.spherical_channels(
        basis, Material(medium).ks(1.3), qs, plane.pol, 3.0, poltype=poltype
    )
    for side, direction in enumerate(("up", "down")):
        if not np.any(qs):
            # As in test_plane_to_spherical_reference, avoid rounding complex
            # kz/k away from +/-1 in the oracle. Angular coefficients are scale
            # invariant; restore the physical phase and radiation normalization.
            sign = 1 if direction == "up" else -1
            ks = treams.Material(medium).ks(1.3)[plane.pol]
            phase = np.exp(1j * sign * basis.positions[basis.pidx, 2, None] * ks)
            incident = (
                treams.pw.to_sw(
                    basis.l[:, None],
                    basis.m[:, None],
                    basis.pol[:, None],
                    0,
                    0,
                    sign,
                    plane.pol,
                    poltype=poltype,
                )
                * phase
            )
            outgoing = treams.sw.periodic_to_pw(
                0,
                0,
                sign,
                plane.pol[:, None],
                basis.l,
                basis.m,
                basis.pol,
                3.0,
                poltype=poltype,
            ) / (ks[:, None] ** 2 * phase.T)
            np.testing.assert_array_equal(value[:, side, abs(basis.m) != 1], 0)
        else:
            incident = treams.expand(
                basis=(sw, pw),
                modetype=("regular", direction),
                k0=1.3,
                material=treams.Material(medium),
                poltype=poltype,
            )
            outgoing = treams.expandlattice(
                np.diag([1.5, 2.0]),
                [0, 0],
                basis=(pw, sw),
                modetype=direction,
                k0=1.3,
                material=treams.Material(medium),
                poltype=poltype,
            )
        assert_allclose(value[0, side], incident, rtol=2e-12, atol=2e-12)
        assert_allclose(value[1, side].T, outgoing, rtol=2e-12, atol=2e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("normal", [True, False])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_channel_pullbacks_all_parameters(normal, poltype):
    rng = np.random.default_rng(32)
    basis = SphericalBasis.default(2, 2, [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]])
    ks = np.array([1.3 + 0.1j, 1.3 + 0.1j])
    if poltype == "helicity":
        ks[1] += 0.1
    q = np.zeros((2, 2)) if normal else np.array([[0.2, 0.1], [2.3, -0.3]])
    pols = np.array([0, 1])
    values = [basis.positions, ks, q, np.array(2.8)]

    def record(positions, ks, q, area):
        return diff.spherical_channels(
            SphericalBasis(basis.modes, positions),
            ks,
            q,
            pols,
            float(area),
            poltype=poltype,
            fixed_q=normal,
        )

    value, ctx = record(*values)
    g = complex_normal(rng, value.shape)
    gradients = ctx.pullback(g)
    directions = [rng.normal(size=v.shape) * 0.1 for v in values]
    directions[1] = directions[1] + 0.1j
    if poltype == "parity":
        # The parity basis requires equal wavenumbers.
        directions[1][:] = directions[1][0]
    # At normal incidence q is fixed and not differentiated.
    free = [0, 1, 3] if normal else [0, 1, 2, 3]
    if normal:
        assert_allclose(gradients[2], 0, atol=0)
    check_pullback(
        varying(record, values, *free),
        *(values[i] for i in free),
        directions=tuple(directions[i] for i in free),
        cotangents=g,
        step=2e-6,
        rtol=1e-7,
        atol=1e-6,
    )


@pytest.mark.gradients
@pytest.mark.interface
def test_normal_incidence_direction_derivative_is_explicit():
    basis = SphericalBasis.default(1)
    value, ctx = diff.spherical_channels(basis, [1.3, 1.3], [[0, 0]], [1], 2.8)
    with pytest.raises(ValueError, match="azimuth"):
        ctx.pullback(np.ones_like(value))


@pytest.mark.physics
@pytest.mark.parametrize("qx", [1e-12, 1e-100, 1e-160, 1e-320])
def test_nearly_normal_channel_scaling(qx):
    basis = SphericalBasis.default(3)
    value, _ = diff.spherical_channels(
        basis, [1.3, 1.3], [[qx, 0]], [1], 2.8, fixed_q=True
    )
    origin, _ = diff.spherical_channels(
        basis, [1.3, 1.3], [[0, 0]], [1], 2.8, fixed_q=True
    )
    assert_allclose(value, origin, rtol=1e-11, atol=1e-10)


@pytest.mark.gradients
@pytest.mark.parametrize("size,ports", [(3, 2), (2, 5), (5, 5)])
def test_native_radiation_pullback(size, ports):
    rng = np.random.default_rng(15)
    response = complex_normal(rng, (size, size))
    channels = complex_normal(rng, (2, 2, size, ports))
    expected = np.array(
        [
            [
                channels[1, i].T @ response @ channels[0, j]
                + (np.eye(ports) if i == j else 0)
                for j in range(2)
            ]
            for i in range(2)
        ]
    )
    value, ctx = diff.smatrix_from_array(response, channels)
    assert_allclose(value, expected, rtol=1e-13, atol=1e-13)
    g = complex_normal(rng, value.shape)
    gradients = ctx.pullback(g)
    # S - I is linear in the response and in each channel group, so every
    # directional derivative is itself a forward evaluation (no finite step).
    identity = diff.smatrix_from_array(np.zeros_like(response), channels)[0]

    def pairing(r, c):
        return np.vdot(g, diff.smatrix_from_array(r, c)[0] - identity).real

    d_response = complex_normal(rng, response.shape)
    d_channels = complex_normal(rng, channels.shape)
    assert_allclose(
        np.vdot(gradients[0], d_response).real,
        pairing(d_response, channels),
        rtol=1e-12,
        atol=1e-12,
    )
    incoming, outgoing = channels.copy(), channels.copy()
    incoming[0], outgoing[1] = d_channels[0], d_channels[1]
    assert_allclose(
        np.vdot(gradients[1], d_channels).real,
        pairing(response, incoming) + pairing(response, outgoing),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.physics
@pytest.mark.reference
@pytest.mark.parametrize("q", [[0, 0], [0.1, 0.2]])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("epsilon", [3.0, 3.0 + 0.2j])
def test_periodic_particle_reflection_transmission(q, poltype, epsilon):
    a = CELL
    vectors = np.asarray(q) + ORDERS @ (2 * np.pi * np.linalg.inv(a).T)
    basis = PlaneWavePorts.default(vectors)
    ob = treams.PlaneWaveBasisByComp.default(vectors)
    positions = [[0, 0, 0], [0.6, 0.2, 0.1]]
    cell = Cluster(
        [TMatrix.sphere(2, 2.1, r, [epsilon, 1], poltype) for r in (0.18, 0.22)],
        positions=positions,
    )
    ot = treams.TMatrix.cluster(
        [treams.TMatrix.sphere(2, 2.1, r, [epsilon, 1], poltype) for r in (0.18, 0.22)],
        positions,
    )
    # On macOS the oracle leaves a divide-by-zero status flag while evaluating
    # zero reciprocal vectors. Its returned sum is finite and remains checked.
    with np.errstate(divide="ignore"):
        expected = treams.SMatrices.from_array(ot.latticeinteraction.solve(a, q), ob)
    value = solve_periodic(cell, lattice=a, kpar=q).to_smatrix(basis)
    assert_allclose(
        value.array,
        oracle_smatrix_array(expected),
        rtol=2e-10,
        atol=1e-11,
    )
    incident = np.zeros(len(basis), complex)
    incident[0] = 1
    assert_allclose(value.tr(incident), expected.tr(incident), rtol=2e-10, atol=1e-11)
    if np.imag(epsilon) == 0:
        assert_allclose(sum(value.tr(incident)), 1, atol=2e-10)
        # Only the zero order, the first two modes, propagates.
        assert_unitary_ports(value.array, slice(0, 2), atol=2e-10)


@pytest.mark.physics
@settings(max_examples=15)
@given(
    q=st.tuples(st.floats(-0.4, 0.4), st.floats(-0.4, 0.4)),
    loss=st.floats(0, 0.5),
    # Offsets include tiny and subnormal ones, where a naive reciprocal sum cancels
    # (see test_tiny_out_of_plane_shift_is_continuous).
    dz=st.one_of(st.just(0.0), st.floats(-0.3, 0.3), st.floats(-1e-8, 1e-8)),
    poltype=st.sampled_from(["helicity", "parity"]),
)
def test_periodic_array_transmittance_is_reciprocal(q, loss, dz, poltype):
    # Reciprocity relates transmission from below at q to transmission from
    # above at -q with input and output polarizations interchanged, so the
    # unpolarized transmittances agree, with loss and without any symmetry of
    # the two-sphere cell. Without flipping q the two sides differ. Only the
    # zero order, the first two ports, propagates for |q| <= 0.4 at k = 2.1.
    def array(kpar):
        vectors = np.asarray(kpar) + ORDERS @ (2 * np.pi * np.linalg.inv(CELL).T)
        spheres = [
            TMatrix.sphere(2, 2.1, radius, [3 + loss * 1j, 1], poltype)
            for radius in (0.18, 0.22)
        ]
        cluster = Cluster(spheres, positions=[[0, 0, 0], [0.6, 0.2, dz]])
        return solve_periodic(cluster, lattice=CELL, kpar=kpar).to_smatrix(
            PlaneWavePorts.default(vectors)
        )

    below, above = array(q), array((-q[0], -q[1]))
    zero_order = np.eye(len(ORDERS) * 2)[:2]
    assert_allclose(
        sum(below.tr(e, modetype="up")[0] for e in zero_order),
        sum(above.tr(e, modetype="down")[0] for e in zero_order),
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.gradients
@pytest.mark.parametrize("normal", [False, True])
# Each example checks every input separately (up to 1 + 2 x 6 objectives).
@settings(max_examples=3)
@given(radius=st.floats(0.12, 0.22), pitch=st.floats(1.6, 1.9))
def test_advect_complete_array_reflectance(normal, radius, pitch):
    basis = SphericalBasis.default(1, 2)
    orders = np.array([[0, 0]]) if normal else np.array([[0, 0], [1, 0], [0, -1]])
    pols = np.tile([1, 0], len(orders))

    def objective(k0, radii, positions, cell, eps, bloch):
        zero = anp.zeros((6, 6), dtype=complex)
        left = ad.sphere(1, k0, radii[:1], anp.array([eps[0], 1]))
        right = ad.sphere(1, k0, radii[1:], anp.array([eps[1], 1]))
        local = anp.concatenate(
            [
                anp.concatenate([left, zero], axis=1),
                anp.concatenate([zero, right], axis=1),
            ],
            axis=0,
        )
        ks = anp.array([k0, k0])
        coupling = ad.lattice_expansion(
            positions, positions, ks, bloch, cell, destination=basis, source=basis
        )
        response = ad.interaction(local, coupling)
        area = cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0]
        reciprocal = (2 * np.pi / area) * anp.array(
            [[cell[1, 1], -cell[1, 0]], [-cell[0, 1], cell[0, 0]]]
        )
        q = (
            np.zeros((2, 2))
            if normal
            else anp.repeat(bloch + orders @ reciprocal, 2, axis=0)
        )
        channels = ad.spherical_channels(
            positions, ks, q, area, basis=basis, polarizations=pols, fixed_q=normal
        )
        smat = ad.smatrix_from_array(response, channels)
        weights = anp.real(anp.sqrt(k0**2 - anp.sum(q**2, axis=1) + 0j)) / k0
        reflected = smat[1, 0, :, 0]
        return anp.sum(weights * anp.real(reflected * anp.conj(reflected))) / weights[0]

    values = [
        np.array(2.1),
        np.array([radius, 0.19]),
        np.array([[0.1, 0.03, -0.04], [0.7, 0.1, 0.08]]),
        np.array([[pitch, 0.05], [-0.03, 1.8]]),
        np.array([3.0 + 0.1j, 2.7 + 0.2j]),
    ]
    directions = [
        np.array(0.1),
        np.array([0.04, -0.03]),
        np.array([[0.1, 0.01, -0.04], [-0.03, 0.02, 0.04]]),
        np.array([[0.1, 0.02], [0.01, -0.03]]),
        np.array([0.1 + 0.03j, -0.1 + 0.01j]),
    ]
    if normal:
        # The Bloch vector stays zero at normal incidence: it is not an input.
        def checked(*args):
            return objective(*args, np.zeros(2))

    else:
        checked = objective
        values.append(np.array([0.1, 0.2]))
        directions.append(np.array([0.04, -0.01]))
    check_gradient(
        checked,
        advect.grad(checked, argnums=tuple(range(len(values)))),
        *values,
        directions=tuple(directions),
        step=1e-5,
        rtol=3e-6,
        atol=1e-10,
    )
