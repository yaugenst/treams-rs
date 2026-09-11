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


def _case(poltype, cylindrical=False):
    family = tr.CylindricalWaveBasis if cylindrical else tr.SphericalWaveBasis
    bases = [
        family.default([0.2, 0.4], i) if cylindrical else family.default(i)
        for i in (1, 2)
    ]
    bases.append(family([bases[1].modes[i] for i in [8, 3, 6, 1]]))
    rng = np.random.default_rng(63)
    local = [
        0.002
        / len(basis)
        * (
            rng.normal(size=(len(basis), len(basis)))
            + 1j * rng.normal(size=(len(basis), len(basis)))
        )
        for basis in bases
    ]
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 0.3, -0.2], [0.4, 1.3, 0.5]])
    material = tr.Material((1.3 + 0.05j, 1.1, 0.06 if poltype == "helicity" else 0))
    return bases, local, positions, material


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("cylindrical", [False, True])
def test_heterogeneous_native_reference_and_all_pullbacks(poltype, cylindrical):
    bases, local, positions, material = _case(poltype, cylindrical)
    matrix_type = treams.TMatrixC if cylindrical else treams.TMatrix
    basis_type = (
        treams.CylindricalWaveBasis if cylindrical else treams.SphericalWaveBasis
    )
    ks = material.ks(1.3)
    metadata = {"bases": bases, "poltype": poltype}
    value, context = tr.diff.particle_cluster(local, positions, ks, **metadata)
    particles = [
        matrix_type(
            a,
            basis=basis_type(b.modes),
            k0=1.3,
            material=treams.Material(tuple(material)),
            poltype=poltype,
        )
        for a, b in zip(local, bases, strict=True)
    ]
    expected = matrix_type.cluster(particles, positions).interaction.solve()
    native_type = tr.TMatrixC if cylindrical else tr.TMatrix
    native_particles = [
        native_type(a, basis=b, k0=1.3, material=material, poltype=poltype)
        for a, b in zip(local, bases, strict=True)
    ]
    assert_allclose(
        native_type.cluster(native_particles, positions).interaction.solve().array,
        expected,
        rtol=2e-12,
        atol=1e-15,
    )
    assert_allclose(value, expected, rtol=2e-12, atol=1e-15)
    rng = np.random.default_rng(51)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    glocal, gpositions, gks = context.pullback(g)
    primals = [positions, ks, *local]
    gradients = [gpositions, gks, *glocal]
    for i, primal in enumerate(primals):
        direction = rng.normal(size=primal.shape) * 0.03
        if i != 0:
            direction = direction + 0.01j
        if i == 1 and poltype == "parity":
            direction[:] = direction[0]
        plus, minus = list(primals), list(primals)
        h = 1e-5
        plus[i], minus[i] = primal + h * direction, primal - h * direction
        numerical = (
            tr.diff.particle_cluster(plus[2:], *plus[:2], **metadata)[0]
            - tr.diff.particle_cluster(minus[2:], *minus[:2], **metadata)[0]
        ) / (2 * h)
        assert_allclose(
            np.vdot(gradients[i], direction).real,
            np.vdot(g, numerical).real,
            rtol=2e-7,
            atol=2e-10,
        )
    assert_allclose(gpositions.sum(axis=0), 0, atol=1e-14)
    if (
        not cylindrical
    ):  # Scaling cylindrical geometry also changes its fixed kz labels.
        assert_allclose(
            np.vdot(gpositions, positions).real - np.vdot(gks, ks).real, 0, atol=1e-13
        )
    dense = np.zeros_like(value)
    offset = 0
    modes = []
    for p, (a, basis) in enumerate(zip(local, bases, strict=True)):
        end = offset + len(a)
        dense[offset:end, offset:end] = a
        modes.extend((p, *mode[1:]) for mode in basis.modes)
        offset = end
    cluster_basis = type(bases[0])(modes, positions)
    coupling, _ = tr.diff.expansion(
        cluster_basis, cluster_basis, ks, singular=True, poltype=poltype
    )
    dense_value, dense_context = tr.diff.interaction(dense, coupling)
    assert_allclose(value, dense_value, atol=1e-15)
    dense_gradient, _ = dense_context.pullback(g)
    offset = 0
    for local_gradient in glocal:
        end = offset + len(local_gradient)
        assert_allclose(
            local_gradient,
            dense_gradient[offset:end, offset:end],
            rtol=1e-13,
            atol=1e-13,
        )
        offset = end


@pytest.mark.parametrize("cylindrical", [False, True])
def test_heterogeneous_context_ownership_strides_and_invalid_retry(cylindrical):
    bases, local, positions, material = _case("helicity", cylindrical)
    ks = material.ks(1.3)
    local = [np.asfortranarray(a).T for a in local]
    value, context = tr.diff.particle_cluster(local, positions, ks, bases=bases)
    g = np.full_like(value, 1 + 0.3j)[::-1, ::-1]
    expected = context.pullback(g)
    _, context = tr.diff.particle_cluster(local, positions, ks, bases=bases)
    for a in local:
        a[:] = 0
    positions[:] = 0
    ks[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback(g[:1])
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full_like(g, np.nan))
    actual = context.pullback(g)
    for a, b in zip(
        [*actual[0], *actual[1:]], [*expected[0], *expected[1:]], strict=True
    ):
        assert_allclose(a, b)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@pytest.mark.parametrize("cylindrical", [False, True])
@given(scale=st.floats(0.5, 3), shift=st.tuples(*(st.floats(-3, 3) for _ in range(3))))
@settings(max_examples=25)
def test_heterogeneous_rigid_translation_and_scale(cylindrical, scale, shift):
    bases, local, positions, material = _case("helicity", cylindrical)
    ks = material.ks(1.3)
    expected = tr.diff.particle_cluster(local, positions, ks, bases=bases)[0]
    if cylindrical:
        bases = [
            tr.CylindricalWaveBasis([(p, kz / scale, m, pol) for p, kz, m, pol in b])
            for b in bases
        ]
    actual = tr.diff.particle_cluster(
        local, positions * scale + shift, ks / scale, bases=bases
    )[0]
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("cylindrical", [False, True])
def test_complete_heterogeneous_material_and_geometry_advect(cylindrical):
    bases = [
        tr.CylindricalWaveBasis.default([0.2, 0.4], i)
        if cylindrical
        else tr.SphericalWaveBasis.default(i)
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
    gradients = advect.grad(objective, argnums=tuple(range(6)))(*values)
    rng = np.random.default_rng(165)
    for i, primal in enumerate(values):
        direction = rng.normal(size=primal.shape) * 0.1
        if i == 2:
            direction = direction + 0.05j
        plus, minus = list(values), list(values)
        h = 1e-5
        plus[i], minus[i] = primal + h * direction, primal - h * direction
        assert_allclose(
            np.vdot(gradients[i], direction).real,
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=3e-6,
            atol=1e-12,
        )


def test_heterogeneous_invalid_inputs_and_single_particle():
    basis = tr.SphericalWaveBasis.default(1)
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
    value, context = tr.diff.particle_cluster([a], [[0, 0, 0]], [1, 1], bases=[basis])
    assert_allclose(value, a)
    glocal, gpositions, gks = context.pullback(np.ones_like(a))
    assert_allclose(glocal[0], 1)
    assert_allclose(gpositions, 0)
    assert_allclose(gks, 0)


def test_cylindrical_cluster_excludes_coaxial_particles():
    basis = tr.CylindricalWaveBasis.default([0.2], 1)
    a = np.eye(len(basis), dtype=complex)
    with pytest.raises(ValueError, match="transverse"):
        tr.diff.particle_cluster(
            [a, a], [[0, 0, 0], [0, 0, 1]], [1, 1], bases=[basis] * 2
        )
