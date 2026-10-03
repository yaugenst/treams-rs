"""Particle clusters: diff.sphere_cluster (homogeneous spheres), diff.particle_cluster
(heterogeneous spherical or cylindrical particles) and the upstream-compatible
TMatrix cluster workflow against upstream, rigid motions, reciprocity, energy
balance and pullbacks, including those of diff.interaction and diff.solve; the
conditioning of the solve under unit changes; and the absorption of an
ill-conditioned small-wavenumber, degree-10 sphere chain against a recorded
reference."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs import diff
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal, reciprocal, selecting, to_oracle


def _cluster_basis(bases, positions):
    """Return the multi-origin basis of particles with local ``bases`` at ``positions``.

    The modes of particle ``p`` are those of ``bases[p]``, relabelled to origin
    ``p``, in order; this is the ordering of ``tr.diff.particle_cluster``.
    """
    modes = [(p, *mode[1:]) for p, basis in enumerate(bases) for mode in basis.modes]
    return type(bases[0])(modes, positions)


def _case(poltype, cylindrical=False):
    family = tr.CylindricalBasis if cylindrical else tr.SphericalBasis
    bases = [
        family.default([0.2, 0.4], i) if cylindrical else family.default(i)
        for i in (1, 2)
    ]
    bases.append(family([bases[1].modes[i] for i in [8, 3, 6, 1]]))
    rng = np.random.default_rng(63)
    local = [
        0.002 / len(basis) * complex_normal(rng, (len(basis), len(basis)))
        for basis in bases
    ]
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 0.3, -0.2], [0.4, 1.3, 0.5]])
    material = tr.Material((1.3 + 0.05j, 1.1, 0.06 if poltype == "helicity" else 0))
    return bases, local, positions, material


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("cylindrical", [False, True])
def test_heterogeneous_native_reference_and_all_pullbacks(poltype, cylindrical):
    bases, local, positions, material = _case(poltype, cylindrical)
    matrix_type = treams.TMatrixC if cylindrical else treams.TMatrix
    ks = material.ks(1.3)
    metadata = {"bases": bases, "poltype": poltype}
    value, context = tr.diff.particle_cluster(local, positions, ks, **metadata)
    particles = [
        matrix_type(
            a,
            basis=to_oracle(b),
            k0=1.3,
            material=treams.Material(tuple(material)),
            poltype=poltype,
        )
        for a, b in zip(local, bases, strict=True)
    ]
    expected = matrix_type.cluster(particles, positions).interaction.solve()
    native_type = tr.CylindricalTMatrix if cylindrical else tr.TMatrix
    native_particles = [
        native_type(a, basis=b, k0=1.3, material=material, poltype=poltype)
        for a, b in zip(local, bases, strict=True)
    ]
    assert_allclose(
        tr.Cluster(native_particles, positions=positions).solve().array,
        expected,
        rtol=2e-12,
        atol=1e-15,
    )
    assert_allclose(value, expected, rtol=2e-12, atol=1e-15)
    rng = np.random.default_rng(51)
    g = complex_normal(rng, value.shape)
    glocal, gpositions, gks = context.pullback(g)
    # The local gradients equal the dense interaction pullback below, whose
    # derivative the Rust interaction_adjoint_identity property checks.
    position_direction = rng.normal(size=positions.shape) * 0.03
    ks_direction = rng.normal(size=ks.shape) * 0.03 + 0.01j
    if poltype == "parity":
        ks_direction[:] = ks_direction[0]
    check_pullback(
        selecting(lambda p, k: tr.diff.particle_cluster(local, p, k, **metadata), 1, 2),
        positions,
        ks,
        directions=(position_direction, ks_direction),
        cotangents=g,
        step=1e-5,
        rtol=2e-7,
        atol=2e-10,
    )
    assert_allclose(gpositions.sum(axis=0), 0, atol=1e-14)
    # Scaling cylindrical geometry also changes its fixed kz labels.
    if not cylindrical:
        assert_allclose(
            np.vdot(gpositions, positions).real - np.vdot(gks, ks).real, 0, atol=1e-13
        )
    basis = _cluster_basis(bases, positions)
    coupling = tr.diff.expansion(basis, basis, ks, singular=True, poltype=poltype)[0]
    dense_value, dense_context = tr.diff.interaction(block_diag(*local), coupling)
    assert_allclose(value, dense_value, atol=1e-15)
    dense_gradient = dense_context.pullback(g)[0]
    ends = np.cumsum([len(a) for a in local])
    for local_gradient, end in zip(glocal, ends, strict=True):
        start = end - len(local_gradient)
        assert_allclose(
            local_gradient, dense_gradient[start:end, start:end], rtol=1e-13, atol=1e-13
        )


@pytest.mark.physics
@pytest.mark.parametrize("cylindrical", [False, True])
@given(scale=st.floats(0.5, 3), shift=st.tuples(*(st.floats(-3, 3) for _ in range(3))))
@settings(max_examples=25)
def test_heterogeneous_rigid_translation_and_scale(cylindrical, scale, shift):
    bases, local, positions, material = _case("helicity", cylindrical)
    ks = material.ks(1.3)
    expected = tr.diff.particle_cluster(local, positions, ks, bases=bases)[0]
    if cylindrical:
        bases = [
            tr.CylindricalBasis([(p, kz / scale, m, pol) for p, kz, m, pol in b])
            for b in bases
        ]
    actual = tr.diff.particle_cluster(
        local, positions * scale + shift, ks / scale, bases=bases
    )[0]
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def _full_bases(cylindrical):
    """Complete degree-1 and degree-2 bases, closed under rotation and m -> -m."""
    if cylindrical:
        return [tr.CylindricalBasis.default([0.2, -0.2], i) for i in (1, 2)]
    return [tr.SphericalBasis.default(i) for i in (1, 2)]


def _polarization_weights(basis, ks, cylindrical):
    """Per-mode k (cylindrical) or k^2 (spherical) of the mode's helicity."""
    power = 1 if cylindrical else 2
    return np.asarray(ks)[[mode[-1] for mode in basis.modes]] ** power


position_offsets = st.tuples(
    st.floats(0.6, 1.5), st.floats(-0.5, 0.5), st.floats(-0.5, 0.5)
)


@pytest.mark.physics
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(position=position_offsets, kappa=st.floats(0, 0.1))
def test_heterogeneous_cluster_reciprocity(cylindrical, poltype, position, kappa):
    # Reciprocal particles form a reciprocal cluster. With the columns weighted by
    # k (cylindrical) or k^2 (spherical) of their helicity, T K = P (T K)^T P.
    bases = _full_bases(cylindrical)
    medium = (1.3 + 0.05j, 1.1, kappa if poltype == "helicity" else 0)
    ks = tr.Material(medium).ks(1.3)
    rng = np.random.default_rng(29)
    local = []
    for basis in bases:
        a = 0.002 / len(basis) * complex_normal(rng, (len(basis), len(basis)))
        weights = _polarization_weights(basis, ks, cylindrical)
        mirrored = reciprocal(a * weights, basis.modes, cylindrical=cylindrical)
        local.append((a + mirrored / weights) / 2)
    positions = np.array([[0, 0, 0], position])
    value = tr.diff.particle_cluster(local, positions, ks, bases=bases, poltype=poltype)
    basis = _cluster_basis(bases, positions)
    weighted = value[0] * _polarization_weights(basis, ks, cylindrical)
    assert_allclose(
        weighted,
        reciprocal(weighted, basis.modes, cylindrical=cylindrical),
        rtol=0,
        atol=1e-13 * np.abs(weighted).max(),
    )


@pytest.mark.physics
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(
    position=position_offsets,
    radius=st.floats(0.1, 0.25),
    epsilon=st.floats(1.5, 4.0),
    loss=st.one_of(st.just(0.0), st.floats(0.01, 0.3)),
    kappa=st.floats(0, 0.05),
)
def test_cluster_energy_balance_and_passivity(
    cylindrical, poltype, position, radius, epsilon, loss, kappa
):
    # Power balance of the complete multiple-scattering response. With W = K^-1
    # (the weights of the reciprocity test) and R the regular re-expansion between
    # the particle origins, the absorbed power -(T^H W + W T + 2 T^H W R T) / 2
    # vanishes for lossless particles and is positive semidefinite for lossy ones.
    bases = _full_bases(cylindrical)
    kappa = kappa if poltype == "helicity" else 0
    embedding = (1.0, 1.0, kappa)

    def particle(degree, material):
        materials = [material, embedding]
        if cylindrical:
            return tr.CylindricalTMatrix.cylinder(
                [0.2, -0.2], degree, 1.3, [radius], materials, poltype=poltype
            )
        return tr.TMatrix.sphere(degree, 1.3, radius, materials, poltype=poltype)

    local = [
        particle(1, (epsilon + 1j * loss, 1.1, 2 * kappa)).array,
        particle(2, (2.2, 1.1, 0.5 * kappa)).array,
    ]
    ks = tr.Material(embedding).ks(1.3)
    positions = np.array([[0, 0, 0], position])
    t = tr.diff.particle_cluster(local, positions, ks, bases=bases, poltype=poltype)[0]
    basis = _cluster_basis(bases, positions)
    regular = tr.diff.expansion(basis, basis, ks, singular=False, poltype=poltype)[0]
    w = 1 / _polarization_weights(basis, ks, cylindrical)
    wt = w[:, None] * t
    absorbed = -(wt.conj().T + wt + 2 * t.conj().T @ (w[:, None] * regular) @ t) / 2
    tolerance = 1e-14 * np.abs(t).max()
    if loss == 0:
        assert_allclose(absorbed, 0, atol=tolerance)
    else:
        assert_allclose(absorbed, absorbed.conj().T, atol=tolerance)
        assert np.linalg.eigvalsh(absorbed).min() >= -tolerance


@pytest.mark.physics
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(angles=st.tuples(*(st.floats(-3, 3) for _ in range(3))))
def test_heterogeneous_cluster_rotation_covariance(cylindrical, poltype, angles):
    # Rotating every particle about its origin and every position about the
    # global origin rotates the response: T(D_i L_i D_i^H, R r) = D T(L, r) D^H.
    # This ties the Euler convention of the rotations to that of the translation
    # coefficients; cylinders rotate about the z axis only.
    if cylindrical:
        angles = (angles[0], 0, 0)
    bases = _full_bases(cylindrical)
    rng = np.random.default_rng(71)
    local = [
        0.002 / len(basis) * complex_normal(rng, (len(basis), len(basis)))
        for basis in bases
    ]
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 0.3, -0.2]])
    medium = (1.3 + 0.05j, 1.1, 0.06 if poltype == "helicity" else 0)
    ks = tr.Material(medium).ks(1.3)
    options = dict(bases=bases, poltype=poltype)
    value = tr.diff.particle_cluster(local, positions, ks, **options)[0]
    rotations = [tr.operators.rotate(*angles, basis=basis) for basis in bases]
    rotation = Rotation.from_euler("ZYZ", angles).as_matrix()
    rotated = tr.diff.particle_cluster(
        [d @ a @ d.conj().T for d, a in zip(rotations, local, strict=True)],
        positions @ rotation.T,
        ks,
        **options,
    )[0]
    d = tr.operators.rotate(*angles, basis=_cluster_basis(bases, positions))
    assert_allclose(
        rotated, d @ value @ d.conj().T, rtol=0, atol=1e-13 * np.abs(value).max()
    )


@pytest.mark.gradients
@pytest.mark.parametrize("cylindrical", [False, True])
def test_complete_heterogeneous_material_and_geometry_advect(cylindrical):
    bases = [
        tr.CylindricalBasis.default([0.2, 0.4], i)
        if cylindrical
        else tr.SphericalBasis.default(i)
        for i in (1, 2)
    ]

    def objective(k0, radii, epsilon, mu, kappa, positions):
        materials = [anp.stack([epsilon[i], epsilon[2]]) for i in range(2)]
        permeabilities = [anp.stack([mu[i], mu[2]]) for i in range(2)]
        chiralities = [anp.stack([kappa[i], kappa[2]]) for i in range(2)]
        local = [
            (
                ad.cylinder(
                    [0.2, 0.4],
                    i + 1,
                    k0,
                    anp.reshape(radii[i], (1,)),
                    materials[i],
                    permeabilities[i],
                    chiralities[i],
                )
                if cylindrical
                else ad.sphere(
                    i + 1,
                    k0,
                    anp.reshape(radii[i], (1,)),
                    materials[i],
                    permeabilities[i],
                    chiralities[i],
                )
            )
            for i in range(2)
        ]
        index = anp.sqrt(epsilon[2] * mu[2])
        ks = k0 * anp.stack([index - kappa[2], index + kappa[2]])
        response = ad.particle_cluster(local, positions, ks, bases=bases)
        return anp.sum(anp.abs(response) ** 2)

    values = [
        np.array(1.3),
        np.array([0.13, 0.18]),
        np.array([2.3 + 0.1j, 3.2 + 0.2j, 1.1 + 0.04j]),
        np.array([1.2, 1.3, 1.0]),
        np.array([0.05, -0.04, 0.03]),
        np.array([[0.1, 0.2, 0.3], [1.1, 0.3, -0.2]]),
    ]
    rng = np.random.default_rng(165)
    directions = [rng.normal(size=primal.shape) * 0.1 for primal in values]
    directions[2] = directions[2] + 0.05j
    check_gradient(
        objective,
        advect.grad(objective, argnums=tuple(range(6))),
        *values,
        directions=tuple(directions),
        step=1e-5,
        rtol=3e-6,
        atol=1e-12,
    )


@pytest.mark.gradients
@pytest.mark.interface
def test_heterogeneous_invalid_inputs_and_single_particle():
    basis = tr.SphericalBasis.default(1)
    a = np.eye(len(basis), dtype=complex) * 0.03
    with pytest.raises(ValueError, match="one local"):
        tr.diff.particle_cluster([], [], [1, 1], bases=[])
    with pytest.raises(ValueError, match="shape"):
        tr.diff.particle_cluster([a[:1]], [[0, 0, 0]], [1, 1], bases=[basis])
    with pytest.raises(ValueError, match="distinct"):
        tr.diff.particle_cluster([a, a], [[0, 0, 0]] * 2, [1, 1], bases=[basis] * 2)
    with pytest.raises(ValueError, match="achiral"):
        tr.diff.particle_cluster(
            [a], [[0, 0, 0]], [1, 2], bases=[basis], poltype="parity"
        )
    with pytest.raises(ValueError, match="positive helicity wavenumbers"):
        tr.diff.particle_cluster([a], [[0, 0, 0]], [1, 1, 1], bases=[basis])
    value, context = tr.diff.particle_cluster([a], [[0, 0, 0]], [1, 1], bases=[basis])
    assert_allclose(value, a)
    glocal, gpositions, gks = context.pullback(np.ones_like(a))
    assert_allclose(glocal[0], 1)
    assert_allclose(gpositions, 0)
    assert_allclose(gks, 0)


@pytest.mark.interface
@pytest.mark.parametrize("cylindrical", [False, True])
def test_physical_cluster_rejects_coincident_particles_before_any_solve(cylindrical):
    particle = (
        tr.cylinder_tmatrix(k0=1.3, kz=0, mmax=1, radius=0.2, material=3)
        if cylindrical
        else tr.sphere_tmatrix(k0=1.3, lmax=1, radius=0.2, material=3)
    )
    # Moving along z cannot separate two infinite cylinders.
    positions = [[0, 0, 0], [0, 0, 1 if cylindrical else 0]]
    with pytest.raises(ValueError, match="distinct"):
        tr.Cluster([particle, particle], positions=positions)


@pytest.mark.interface
def test_cylindrical_cluster_excludes_coaxial_particles():
    basis = tr.CylindricalBasis.default([0.2], 1)
    a = np.eye(len(basis), dtype=complex)
    with pytest.raises(ValueError, match="transverse"):
        tr.diff.particle_cluster(
            [a, a], [[0, 0, 0], [0, 0, 1]], [1, 1], bases=[basis] * 2
        )


@pytest.mark.workflows
@pytest.mark.reference
def test_cluster_native_against_upstream():
    radii = [0.2, 0.3]
    epsilon = [3 + 0.1j, 2 + 0.2j]
    positions = [[0.0, 0.0, 0.0], [0.4, -0.2, 1.4]]
    actual, _ = diff.sphere_cluster(2, 1.1, radii, epsilon, positions)
    spheres = [
        treams.TMatrix.sphere(2, 1.1, radius, [eps, 1])
        for radius, eps in zip(radii, epsilon, strict=True)
    ]
    expected = treams.TMatrix.cluster(spheres, positions).interaction.solve()
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-10)


@pytest.mark.gradients
@given(offset=st.tuples(st.floats(-5, 5), st.floats(-5, 5), st.floats(-5, 5)))
def test_cluster_global_translation_invariance(offset):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    args = (1, 1.2, [0.2, 0.3], [2.0, 3.0])
    base, context = diff.sphere_cluster(*args, positions)
    shifted, _ = diff.sphere_cluster(*args, positions + offset)
    np.testing.assert_allclose(shifted, base, atol=3e-13, rtol=3e-12)
    gradient = context.pullback(np.ones_like(base))[3]
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=2e-13)


@pytest.mark.gradients
def test_complete_cluster_directional_derivative():
    check_pullback(
        lambda k0, radii, epsilon, positions: diff.sphere_cluster(
            2, float(k0), radii, epsilon, positions
        ),
        np.asarray(1.1),
        np.array([0.2, 0.3]),
        np.array([3 + 0.1j, 2 + 0.2j]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        step=2e-6,
        rtol=2e-6,
        atol=2e-8,
    )


@pytest.mark.gradients
def test_interaction_directional_derivative():
    check_pullback(
        diff.interaction,
        np.array([[0.2 + 0.1j, 0.03j], [0.05, 0.3 - 0.1j]]),
        np.array([[0.0, 0.1 + 0.2j], [-0.03j, 0.0]]),
        step=2e-6,
        rtol=8e-6,
        atol=3e-8,
    )


def _block_diagonal(spheres, positions):
    """Uncoupled block-diagonal T-matrix of the spheres, for comparing with treams."""
    modes = [
        (index, degree, order, pol)
        for index, sphere in enumerate(spheres)
        for _, degree, order, pol in sphere.basis
    ]
    first = spheres[0]
    return tr.TMatrix(
        block_diag(*(sphere.array for sphere in spheres)),
        basis=tr.SphericalBasis(modes, positions),
        k0=first.k0,
        material=first.medium,
        poltype=first.polarization,
    )


@pytest.mark.workflows
@pytest.mark.reference
@pytest.mark.parametrize(
    "poltype,embedding", [("helicity", (1.2, 1.1, 0.02)), ("parity", (1.2, 1.1, 0))]
)
def test_heterogeneous_cluster_global_expansion_and_cross_sections(poltype, embedding):
    positions = [[0, 0, 0], [0.3, 0.2, 1.4]]
    matrices = []
    for package in (tr, treams):
        spheres = [
            package.TMatrix.sphere(
                lmax,
                1.1,
                [0.1, radius],
                [
                    (2.1, 1.2, 0.03 if poltype == "helicity" else 0),
                    (3 + 0.1j, 1, 0.01 if poltype == "helicity" else 0),
                    embedding,
                ],
                poltype,
            )
            for lmax, radius in [(1, 0.2), (2, 0.3)]
        ]
        if package is tr:
            cluster = _block_diagonal(spheres, positions)
            solved = tr.Cluster(spheres, positions=positions).solve()
        else:
            cluster = treams.TMatrix.cluster(spheres, positions)
            solved = cluster.interaction.solve()
        global_matrix = solved.expand(
            (tr.SphericalBasis if package is tr else treams.SphericalWaveBasis).default(
                4
            )
        )
        incident = np.random.default_rng(1).normal(size=len(solved)) + 0.3j
        if package is treams:
            incident = treams.PhysicsArray(
                incident,
                basis=solved.basis,
                k0=solved.k0,
                material=solved.material,
                poltype=solved.poltype,
                modetype="regular",
            )
        matrices.append((cluster, solved, global_matrix, solved.xs(incident)))
    for actual, expected in zip(*matrices, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=3e-12)
    for name in ("xs_ext_avg", "xs_sca_avg"):
        np.testing.assert_allclose(
            getattr(matrices[0][2], name),
            getattr(matrices[1][2], name),
            rtol=2e-10,
            atol=1e-12,
        )


@pytest.mark.physics
@pytest.mark.gradients
@given(exponent=st.integers(30, 399), real=st.floats(-0.5, 0.5))
def test_solve_and_pullback_preserve_independent_equation_and_unknown_units(
    exponent, real
):
    h = np.array([[2 + 1j, 0.5 - 0.25j], [-0.25 + 0.5j, 3 - 1j]])
    rows = np.exp2(np.array([-exponent, exponent], dtype=float))
    columns = np.exp2(np.array([exponent // 2, -(exponent // 2)], dtype=float))
    y = np.array([[1 + real + 0.2j, 1j], [-0.5j, 1 - real]])
    z = np.array([[0.1j, 0.4], [1 + real, 0.2 + 0.3j]])
    a = h / rows[:, None] / columns[None, :]
    b = (h @ y) / rows[:, None]
    value, context = diff.solve(a, b)
    assert_allclose(value / columns[:, None], y, atol=2e-14)
    ga, gb = context.pullback((h.conj().T @ z) / columns[:, None])
    assert_allclose(gb / rows[:, None], z, atol=2e-14)
    assert_allclose(ga / rows[:, None] / columns[None, :], -z @ y.conj().T, atol=2e-14)


@pytest.mark.reference
def test_small_wavenumber_high_order_chain_absorption_is_stable():
    # PRB 112, 054307 (2025), Fig. 2: four 250 nm SiC spheres with 20 nm gaps,
    # at 3 THz, the lowest frequency of its sweep. There I-TC has unit diagonal
    # but entries exceeding 1e22 at degree 10.
    # Oracle: independent two-sided equilibration, LAPACK balancing and sqrt(T)
    # basis normalization agree on this absorption; unscaled LU differs by 0.3%.
    frequency = 3e12
    k0 = 2 * np.pi * frequency / 299792458.0
    sphere = tr.TMatrix.sphere(
        10,
        k0,
        250e-9,
        [tr.Material(12.87707323 + 0.19225647000000007j), tr.Material()],
    )
    positions = [[0, 0, z * 520e-9] for z in (-1.5, -0.5, 0.5, 1.5)]
    cluster = tr.Cluster([sphere] * 4, positions=positions).solve()
    matrix = np.asarray(cluster.expand(tr.SphericalBasis.default(10)))
    absorption = (
        -2 * np.pi * (np.trace(matrix).real + np.vdot(matrix, matrix).real) / k0**2
    )
    assert_allclose(absorption * 1e12, 0.0001754433586134972, rtol=2e-11, atol=0)
    # Axial symmetry forbids coupling between distinct azimuthal orders.
    orders = np.array([mode[2] for mode in tr.SphericalBasis.default(10)])
    assert_allclose(matrix[orders[:, None] != orders[None, :]], 0, atol=2e-20)
