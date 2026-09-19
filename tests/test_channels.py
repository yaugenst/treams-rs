import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import (
    Material,
    TMatrix,
    diff,
)
from treams_rs import (
    PlaneWavePorts as PlaneWaveBasisByComp,
)
from treams_rs import (
    SMatrix as SMatrices,
)
from treams_rs import (
    SphericalBasis as SphericalWaveBasis,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:'where' used without 'out'.*:UserWarning"
)


@given(pitch=st.floats(1, 4), cutoff=st.floats(0, 6))
def test_diffraction_basis_reference(pitch, cutoff):
    a = np.diag([pitch, 1.7])
    ours = PlaneWaveBasisByComp.diffr_orders([0.1, 0.2], a, cutoff)
    oracle = treams.PlaneWaveBasisByComp.diffr_orders([0.1, 0.2], a, cutoff)
    assert_allclose(ours.kx, oracle.kx)
    assert_allclose(ours.ky, oracle.ky)
    assert_allclose(ours.pol, oracle.pol)


@given(shear=st.floats(-3, 3), cutoff=st.floats(0, 5))
def test_diffraction_basis_skew_completeness(shear, cutoff):
    a = np.array([[2.0, shear], [0.0, 1.7]])
    reciprocal = 2 * np.pi * np.linalg.inv(a).T
    basis = PlaneWaveBasisByComp.diffr_orders([0, 0], a, cutoff)
    actual = set(
        map(
            tuple,
            np.rint(np.column_stack([basis.kx, basis.ky]) @ a.T / (2 * np.pi)).astype(
                int
            ),
        )
    )
    expected = {
        (m, n)
        for m in range(-5, 6)
        for n in range(-5, 6)
        if np.linalg.norm(np.array([m, n]) @ reciprocal) <= cutoff
    }
    assert actual == expected
    assert len(basis) == 2 * len(expected)


@pytest.mark.parametrize(
    "poltype,medium",
    [(p, m) for m in [1, (2.1 + 0.2j, 1.2)] for p in ["helicity", "parity"]]
    + [("helicity", (2.3 + 0.1j, 1.1, 0.07))],
)
@pytest.mark.parametrize("q", [[[0, 0]], [[0.2, 0.1], [3.7, 0.2], [-0.1, -0.4]]])
def test_channels_reference(poltype, medium, q):
    positions = [[0.1, 0.2, -0.3], [-0.2, 0.1, 0.4]]
    basis = SphericalWaveBasis.default(3, 2, positions)
    sw = treams.SphericalWaveBasis.default(3, 2, positions)
    pw = treams.PlaneWaveBasisByComp.default(q)
    plane = PlaneWaveBasisByComp.default(q)
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


@pytest.mark.parametrize("normal", [True, False])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_channel_pullbacks_all_parameters(normal, poltype):
    rng = np.random.default_rng(32)
    basis = SphericalWaveBasis.default(2, 2, [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]])
    ks = np.array([1.3 + 0.1j, 1.3 + 0.1j])
    if poltype == "helicity":
        ks[1] += 0.1
    q = np.zeros((2, 2)) if normal else np.array([[0.2, 0.1], [2.3, -0.3]])
    pols = np.array([0, 1])
    values = [basis.positions, ks, q, np.array(2.8)]

    def function(values):
        b = SphericalWaveBasis(basis.modes, values[0])
        return diff.spherical_channels(
            b,
            values[1],
            values[2],
            pols,
            float(values[3]),
            poltype=poltype,
            fixed_q=normal,
        )

    value, ctx = function(values)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = ctx.pullback(g)
    directions = [rng.normal(size=v.shape) * 0.1 for v in values]
    directions[1] = directions[1] + 0.1j
    if poltype == "parity":
        directions[1][:] = directions[1][0]
    if normal:
        directions[2][:] = 0
        assert_allclose(gradients[2], 0, atol=0)
    h = 2e-6
    plus = [v + h * d for v, d in zip(values, directions, strict=True)]
    minus = [v - h * d for v, d in zip(values, directions, strict=True)]
    numerical = np.vdot(g, (function(plus)[0] - function(minus)[0]) / (2 * h)).real
    actual = sum(np.vdot(a, b).real for a, b in zip(gradients, directions, strict=True))
    assert_allclose(actual, numerical, rtol=1e-7, atol=1e-6)
    with pytest.raises(ValueError, match="consumed"):
        ctx.pullback(g)


def test_normal_incidence_direction_derivative_is_explicit():
    basis = SphericalWaveBasis.default(1)
    value, ctx = diff.spherical_channels(basis, [1.3, 1.3], [[0, 0]], [1], 2.8)
    with pytest.raises(ValueError, match="azimuth"):
        ctx.pullback(np.ones_like(value))


@pytest.mark.parametrize("qx", [1e-12, 1e-100, 1e-160, 1e-320])
def test_nearly_normal_channel_scaling(qx):
    basis = SphericalWaveBasis.default(3)
    value, _ = diff.spherical_channels(
        basis, [1.3, 1.3], [[qx, 0]], [1], 2.8, fixed_q=True
    )
    origin, _ = diff.spherical_channels(
        basis, [1.3, 1.3], [[0, 0]], [1], 2.8, fixed_q=True
    )
    assert_allclose(value, origin, rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize("size,ports", [(3, 2), (2, 5), (5, 5)])
def test_native_radiation_pullback(size, ports):
    rng = np.random.default_rng(15)
    response = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    channels = rng.normal(size=(2, 2, size, ports)) + 1j * rng.normal(
        size=(2, 2, size, ports)
    )
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
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = ctx.pullback(g)
    directions = [
        rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
        for v in (response, channels)
    ]
    h = 1e-6
    plus = diff.smatrix_from_array(
        response + h * directions[0], channels + h * directions[1]
    )[0]
    minus = diff.smatrix_from_array(
        response - h * directions[0], channels - h * directions[1]
    )[0]
    assert_allclose(
        sum(np.vdot(a, b).real for a, b in zip(gradients, directions, strict=True)),
        np.vdot(g, (plus - minus) / (2 * h)).real,
        rtol=1e-7,
        atol=1e-6,
    )


@pytest.mark.parametrize("q", [[0, 0], [0.1, 0.2]])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("epsilon", [3.0, 3.0 + 0.2j])
@pytest.mark.filterwarnings("ignore:.*scipy.special.sph_harm.*:DeprecationWarning")
def test_periodic_particle_reflection_transmission(q, poltype, epsilon):
    a = np.diag([1.7, 1.8])
    orders = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    vectors = np.asarray(q) + orders @ (2 * np.pi * np.linalg.inv(a).T)
    basis = PlaneWaveBasisByComp.default(vectors)
    ob = treams.PlaneWaveBasisByComp.default(vectors)
    tm = TMatrix._assemble(
        [TMatrix.sphere(2, 2.1, r, [epsilon, 1], poltype) for r in (0.18, 0.22)],
        [[0, 0, 0], [0.6, 0.2, 0.1]],
    )
    ot = treams.TMatrix.cluster(
        [treams.TMatrix.sphere(2, 2.1, r, [epsilon, 1], poltype) for r in (0.18, 0.22)],
        tm.basis.positions,
    )
    # On macOS the oracle leaves a divide-by-zero status flag while evaluating
    # zero reciprocal vectors. Its returned sum is finite and remains checked.
    with np.errstate(divide="ignore"):
        expected = treams.SMatrices.from_array(ot.latticeinteraction.solve(a, q), ob)
    value = SMatrices._from_array(tm, basis, lattice=a, kpar=q)
    assert_allclose(
        value.array,
        np.array([[np.asarray(expected[i, j]) for j in range(2)] for i in range(2)]),
        rtol=2e-10,
        atol=1e-11,
    )
    incident = np.zeros(len(basis), complex)
    incident[0] = 1
    assert_allclose(value.tr(incident), expected.tr(incident), rtol=2e-10, atol=1e-11)
    if np.imag(epsilon) == 0:
        assert_allclose(sum(value.tr(incident)), 1, atol=2e-10)


@settings(max_examples=8, deadline=None)
@given(radius=st.floats(0.12, 0.22), pitch=st.floats(1.6, 1.9), normal=st.booleans())
def test_advect_complete_array_reflectance(radius, pitch, normal):
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad

    basis = SphericalWaveBasis.default(1, 2)
    orders = np.array([[0, 0]]) if normal else np.array([[0, 0], [1, 0], [0, -1]])
    pols = np.tile([1, 0], len(orders))

    def objective(k0, radii, positions, cell, bloch, eps):
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
        np.array([0.0, 0.0]) if normal else np.array([0.1, 0.2]),
        np.array([3.0 + 0.1j, 2.7 + 0.2j]),
    ]
    directions = [
        0.1,
        np.array([0.04, -0.03]),
        np.array([[0.1, 0.01, -0.04], [-0.03, 0.02, 0.04]]),
        np.array([[0.1, 0.02], [0.01, -0.03]]),
        np.zeros(2) if normal else np.array([0.04, -0.01]),
        np.array([0.1 + 0.03j, -0.1 + 0.01j]),
    ]
    gradients = advect.grad(objective, argnums=tuple(range(6)))(*values)
    h = 1e-5
    plus = [v + h * d for v, d in zip(values, directions, strict=True)]
    minus = [v - h * d for v, d in zip(values, directions, strict=True)]
    assert_allclose(
        sum(np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True)),
        (objective(*plus) - objective(*minus)) / (2 * h),
        rtol=3e-6,
        atol=1e-10,
    )
