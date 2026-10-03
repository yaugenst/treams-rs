"""Periodic arrays across wave families: spherical-to-cylindrical conversion,
expandlattice and array constructors against upstream, image sums, power, pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_array_constructor_reference(poltype):
    period, bloch, k0 = 1.7, 0.2, 1.3
    material = (1.4 + 0.1j, 1.2, 0.07 if poltype == "helicity" else 0)
    kzs = bloch + 2 * np.pi / period * np.arange(-1, 2)
    basis = tr.CylindricalBasis.default(kzs, 3)
    particle = tr.TMatrix.sphere(3, k0, 0.2, [3.1, material], poltype=poltype)
    actual = tr.solve_periodic(particle, lattice=period, kpar=bloch).to_cylindrical(
        basis
    )
    oracle = treams.TMatrix.sphere(3, k0, 0.2, [3.1, material], poltype=poltype)
    response = oracle.latticeinteraction.solve(period, bloch)
    expected = treams.TMatrixC.from_array(
        response, treams.CylindricalWaveBasis.default(kzs, 3)
    )
    assert_allclose(actual.array, expected, rtol=3e-10, atol=2e-11)
    assert actual.k0 == particle.k0
    assert actual.material == particle.material
    assert actual.poltype == poltype


def _two_axis_chain():
    """treams' chain_tmatrixc example: two spheres per cell, one axis through each."""
    k0, period, kz = 2 * np.pi / 700, 300, 0.005
    lmax = mmax = 3
    radii, positions = [75, 65], [[-30, 0, -75], [30, 0, 75]]
    epsilon, bmax = -16.5 + 1j, 3.1 * 2 * np.pi / period
    spheres = [
        tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=r, material=tr.Material(epsilon))
        for r in radii
    ]
    cell = tr.Cluster(spheres, positions=positions)
    chain = tr.solve_periodic(cell, lattice=period, kpar=kz)
    basis = tr.CylindricalBasis.diffr_orders(kz, mmax, period, bmax, 2, positions)
    lattice = treams.Lattice(period)
    oracle_spheres = [
        treams.TMatrix.sphere(lmax, k0, r, [treams.Material(epsilon), 1]) for r in radii
    ]
    oracle_chain = treams.TMatrix.cluster(
        oracle_spheres, positions
    ).latticeinteraction.solve(lattice, kz)
    oracle_basis = treams.CylindricalWaveBasis.diffr_orders(
        kz, mmax, lattice, bmax, 2, positions
    )
    return chain, basis, oracle_chain, oracle_basis, (lattice, kz)


@pytest.mark.reference
def test_two_axis_cylindrical_response_reference():
    chain, basis, oracle_chain, oracle_basis, _ = _two_axis_chain()
    actual = chain.to_cylindrical(basis).array
    expected = np.asarray(treams.TMatrixC.from_array(oracle_chain, oracle_basis))
    assert basis.modes == tuple(map(tuple, oracle_basis))
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-9 * np.abs(expected).max())


@pytest.mark.reference
def test_two_axis_cylindrical_wave_reference():
    chain, basis, oracle_chain, oracle_basis, (lattice, kz) = _two_axis_chain()
    incident = complex_normal(np.random.default_rng(7), len(chain.basis))
    actual = chain.scatter(incident).in_basis(basis, kind="singular").coefficients
    conversion = treams.expandlattice(
        lattice,
        kz,
        basis=(oracle_basis, oracle_chain.basis),
        k0=oracle_chain.k0,
        material=oracle_chain.material,
        poltype=oracle_chain.poltype,
    )
    expected = np.asarray(conversion) @ (np.asarray(oracle_chain) @ incident)
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-9 * np.abs(expected).max())


@pytest.mark.interface
def test_several_axes_need_one_axis_per_particle():
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=1, radius=0.2, material=3.1)
    chain = tr.solve_periodic(sphere, lattice=1.7, kpar=0.2)
    basis = tr.CylindricalBasis.default([0.2], 1, 2, [[0, 0, 0], [0.5, 0, 0]])
    with pytest.raises(ValueError, match="one axis per particle position"):
        chain.to_cylindrical(basis)
    with pytest.raises(ValueError, match="one axis per particle position"):
        chain.scatter(np.ones(len(chain.basis))).in_basis(basis, kind="singular")


@pytest.mark.physics
@given(radius=st.floats(0.1, 0.3), scale=st.floats(0.6, 1.8))
def test_array_lossless_power_and_scale(radius, scale):
    period, bloch, k0 = 1.7, 0.2, 1.3
    particle = tr.TMatrix.sphere(3, k0, radius, [3, 1])
    basis = tr.CylindricalBasis.default([bloch], 3)
    value = (
        tr.solve_periodic(particle, lattice=period, kpar=bloch)
        .to_cylindrical(basis)
        .array
    )
    scattering = np.eye(len(basis)) + 2 * value
    assert_allclose(scattering.conj().T @ scattering, np.eye(len(basis)), atol=2e-10)
    scaled = tr.solve_periodic(
        tr.TMatrix.sphere(3, k0 / scale, radius * scale, [3, 1]),
        lattice=period * scale,
        kpar=bloch / scale,
    ).to_cylindrical(tr.CylindricalBasis.default([bloch / scale], 3))
    assert_allclose(scaled.array, value, rtol=2e-10, atol=2e-12)


@pytest.mark.physics
@pytest.mark.parametrize(
    "poltype,kappa", [("helicity", 0), ("parity", 0), ("helicity", 0.12)]
)
@settings(max_examples=20)
@given(radius=st.floats(0.1, 0.3), loss=st.floats(0, 0.2))
def test_array_cross_width_with_evanescent_orders(poltype, kappa, radius, loss):
    period, bloch, k0 = 5.2, 0.15, 1.3
    medium = (1, 1, kappa)
    particle = tr.TMatrix.sphere(
        3, k0, radius, [(3.1 + loss * 1j, 1, 0.04), medium], poltype=poltype
    )
    basis = tr.CylindricalBasis.default(
        bloch + 2 * np.pi / period * np.arange(-1, 2), 3
    )
    tm = tr.solve_periodic(particle, lattice=period, kpar=bloch).to_cylindrical(basis)
    radiating = abs(basis.kz) < tm.ks[basis.pol].real
    assert radiating.any() and not radiating.all()
    widths = []
    # Seven angles integrate all Fourier differences through m=3 exactly.
    for kz, pol in np.unique(np.column_stack((basis.kz, basis.pol))[radiating], axis=0):
        k = tm.ks[int(pol)].real
        rho = np.sqrt(k**2 - kz**2)
        for phi in np.arange(7) * 2 * np.pi / 7:
            wave = tr.plane_wave(
                [rho * np.cos(phi), rho * np.sin(phi), kz],
                int(pol),
                k0=k0,
                material=medium,
                poltype=poltype,
            )
            incident = wave.expand(basis)
            assert_allclose(np.vdot(incident, incident).real, 7, atol=2e-13)
            widths.append(tm.xw(wave))
    widths = np.asarray(widths)
    assert np.all(widths[:, 0] <= widths[:, 1] + 2e-12)
    if loss == 0:
        assert_allclose(widths[:, 0], widths[:, 1], atol=2e-12)
    assert_allclose(
        [tm.xw_sca_avg, tm.xw_ext_avg], widths.mean(axis=0), rtol=2e-12, atol=2e-13
    )
    # Adding closed incoming/outgoing channels cannot change a far-field average.
    truncated = tr.CylindricalTMatrix(
        tm.array[np.ix_(radiating, radiating)],
        k0=k0,
        basis=tr.CylindricalBasis(np.asarray(basis.modes)[radiating]),
        material=medium,
        poltype=poltype,
    )
    assert_allclose(
        [tm.xw_sca_avg, tm.xw_ext_avg],
        [truncated.xw_sca_avg, truncated.xw_ext_avg],
        atol=2e-13,
    )


@pytest.mark.interface
@pytest.mark.parametrize("kz", [1.3, 2.0])
def test_cross_width_rejects_evanescent_illumination_and_cutoff(kz):
    basis = tr.CylindricalBasis.default([kz], 0)
    tm = tr.CylindricalTMatrix(np.eye(2), k0=1.3, basis=basis)
    for operation in (
        lambda: tm.xw([1, 0]),
        lambda: tm.xw_sca_avg,
        lambda: tm.xw_ext_avg,
    ):
        with pytest.raises(ValueError, match=r"cutoff|propagating incident"):
            operation()


@pytest.mark.reference
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_expandlattice_same_family_and_plane_radiation(cylindrical, poltype):
    k0, material = 1.3, (1.4 + 0.1j, 1.2)
    if cylindrical:
        source = tr.CylindricalBasis.default([0.2], 2)
        oracle_source = treams.CylindricalWaveBasis.default([0.2], 2)
        lattice, bloch = 1.7, 0.1
        q = [[0.2, bloch + n * 2 * np.pi / lattice] for n in [-1, 0, 1]]
        ports = tr.PlaneWavePorts.default(q, "zx")
        oracle_ports = treams.PlaneWaveBasisByComp.default(q, "zx")
    else:
        source = tr.SphericalBasis.default(2)
        oracle_source = treams.SphericalWaveBasis.default(2)
        lattice, bloch = np.diag([1.7, 1.8]), [0.1, 0.2]
        ports = tr.PlaneWavePorts.diffr_orders(bloch, lattice, 4)
        oracle_ports = treams.PlaneWaveBasisByComp.default(ports.components[::2])
    common = dict(k0=k0, material=material, poltype=poltype)
    actual = tr.operators.expandlattice(lattice, bloch, basis=source, **common)
    expected = treams.expandlattice(lattice, bloch, basis=oracle_source, **common)
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    for side in ("up", "down"):
        actual = tr.operators.expandlattice(
            lattice, bloch, basis=(ports, source), modetype=(side, "singular"), **common
        )
        expected = treams.expandlattice(
            lattice, bloch, basis=(oracle_ports, oracle_source), modetype=side, **common
        )
        assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.interface
def test_expandlattice_cylindrical_orders_and_wave_types():
    source = tr.SphericalBasis.default(2)
    destination = tr.CylindricalBasis.default([0.2, 0.2 + 2 * np.pi / 1.7], 2)
    for modetype in (None, "singular", ("singular", "singular")):
        actual = tr.operators.expandlattice(
            1.7, 0.2, basis=(destination, source), k0=1.3, modetype=modetype
        )
        assert_allclose(
            actual, tr.diff.periodic_to_cw(destination, source, [1.3, 1.3], 1.7)[0]
        )
    with pytest.raises(ValueError, match="diffraction orders"):
        tr.operators.expandlattice(1.7, 0.3, basis=(destination, source), k0=1.3)
    with pytest.raises(ValueError, match="outgoing waves"):
        tr.operators.expandlattice(
            1.7, 0.2, basis=(destination, source), k0=1.3, modetype="regular"
        )


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_periodic_conversion_reference(poltype):
    source = tr.SphericalBasis.default(4)
    destination = tr.CylindricalBasis.default([-0.3, 0.2, 2.7], 4)
    ks = (
        np.array([1.3 + 0.1j, 1.5 + 0.2j])
        if poltype == "helicity"
        else np.full(2, 1.3 + 0.1j)
    )
    actual = tr.diff.periodic_to_cw(destination, source, ks, 1.7, poltype=poltype)[0]
    expected = treams.sw.periodic_to_cw(
        destination.kz[:, None],
        destination.m[:, None],
        destination.pol[:, None],
        source.l,
        source.m,
        source.pol,
        ks[source.pol],
        1.7,
        poltype=poltype,
    )
    assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.gradients
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("coincident", [False, True])
def test_periodic_conversion_all_pullbacks(poltype, coincident):
    source = tr.SphericalBasis.default(3, 2)
    destination = tr.CylindricalBasis.default([-0.3, 0.2], 2, 2)
    rng = np.random.default_rng(89)
    to = np.zeros((2, 3)) if coincident else rng.normal(size=(2, 3)) * 0.1
    origin = np.zeros((2, 3)) if coincident else rng.normal(size=(2, 3)) * 0.1
    ks = (
        np.array([1.3 + 0.1j, 1.5 + 0.2j])
        if poltype == "helicity"
        else np.full(2, 1.3 + 0.1j)
    )
    values = [to, origin, ks, destination.kz, np.array(1.7)]
    directions = [
        rng.normal(size=v.shape) * 0.1 + (0.02j if np.iscomplexobj(v) else 0)
        for v in values
    ]
    if poltype == "parity":
        # The parity basis requires equal wavenumbers.
        directions[2][:] = directions[2][0]

    def record(to, origin, ks, kz, period):
        modes = [
            (p, label, m, pol)
            for (p, _, m, pol), label in zip(destination.modes, kz, strict=True)
        ]
        return tr.diff.periodic_to_cw(
            type(destination)(modes, to),
            type(source)(source.modes, origin),
            ks,
            float(period),
            poltype=poltype,
        )

    value, context = record(*values)
    cotangent = complex_normal(rng, value.shape)
    gradients = context.pullback(cotangent)
    assert_allclose(gradients[0].sum(axis=0) + gradients[1].sum(axis=0), 0, atol=2e-12)
    assert_allclose(
        gradients[4] * values[4], -np.vdot(cotangent, value).real, atol=2e-12
    )
    check_pullback(
        record,
        *values,
        directions=tuple(directions),
        cotangents=cotangent,
        step=1e-5,
        rtol=3e-7,
        atol=2e-8,
    )


@pytest.mark.physics
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_periodic_conversion_reconstructs_image_sum(poltype):
    period, bloch = 1.7, 0.2
    source = tr.SphericalBasis.default(2, positions=[[-0.1, 0.2, -0.2]])
    orders = np.arange(-8, 9)
    destination = tr.CylindricalBasis.default(
        bloch + 2 * np.pi / period * orders, 20, positions=[[0.2, -0.1, 0.3]]
    )
    ks = (
        np.array([1.3 + 0.2j, 1.5 + 0.25j])
        if poltype == "helicity"
        else np.full(2, 1.3 + 0.2j)
    )
    coefficients = np.arange(len(source)) * (0.03 + 0.02j)
    converted = (
        tr.diff.periodic_to_cw(destination, source, ks, period, poltype=poltype)[0]
        @ coefficients
    )
    points = np.array([[2.4, 0.4, 0.1], [-2.3, -0.4, -0.2]])
    actual = tr.diff.field(
        converted, points, destination, ks, poltype=poltype, singular=True
    )[0]
    images = np.arange(-90, 91)
    positions = source.positions + np.column_stack(
        [np.zeros((len(images), 2)), images * period]
    )
    repeated = tr.SphericalBasis.default(2, len(images), positions)
    amplitudes = (np.exp(1j * bloch * period * images)[:, None] * coefficients).ravel()
    expected = tr.diff.field(
        amplitudes, points, repeated, ks, poltype=poltype, singular=True
    )[0]
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


@pytest.mark.gradients
def test_advect_periodic_sphere_to_moving_cylindrical_orders():
    source = tr.SphericalBasis.default(2)
    orders = np.repeat(np.arange(-2, 3), 14)
    destination = tr.CylindricalBasis.default(
        0.2 + 2 * np.pi / 1.7 * np.arange(-2, 3), 3
    )
    propagating = orders == 0

    def objective(radius, k0, period, bloch, origins):
        ks = anp.stack([k0, k0])
        local = ad.sphere(2, k0, anp.reshape(radius, (1,)), [3 + 0.1j, 1])
        coupling = ad.lattice_expansion(
            origins,
            origins,
            ks,
            anp.reshape(bloch, (1,)),
            anp.reshape(period, (1, 1)),
            destination=source,
            source=source,
        )
        response = ad.interaction(local, coupling)
        vector = anp.stack([anp.sqrt(k0**2 - bloch**2), 0.0, bloch])[None, :]
        incident = ad.plane_expansion(
            origins, vector, destination=source, polarizations=[1]
        )[:, 0]
        conversion = ad.periodic_to_cw(
            destination.positions,
            origins,
            ks,
            bloch + 2 * np.pi / period * orders,
            period,
            destination=destination,
            source=source,
        )
        radiation = (conversion @ response @ incident)[propagating]
        return anp.sum(anp.real(radiation * anp.conj(radiation)))

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2, 3, 4)),
        np.array(0.2),
        np.array(1.3),
        np.array(1.7),
        np.array(0.2),
        np.array([[0.1, 0.05, -0.1]]),
        directions=(
            np.array(0.03),
            np.array(0.1),
            np.array(0.04),
            np.array(-0.02),
            np.array([[0.01, -0.02, 0.03]]),
        ),
        step=1e-5,
        rtol=3e-6,
        atol=2e-9,
    )
