# /// script
# requires-python = ">=3.12,<3.14"
# ///
"""Run with the clean wheel environment, without the upstream scientific stack."""

import importlib.util

import advect
import advect.numpy as anp
import numpy as np

import treams_rs as tr
from treams_rs import advect as ad

for dependency in ("treams", "scipy", "autograd", "h5py"):
    assert importlib.util.find_spec(dependency) is None, dependency

cell = np.diag([1.7, 1.8])
basis = tr.PlaneWaveBasisByComp.diffr_orders([0, 0], cell, 4)
sphere = tr.TMatrix.sphere(2, 2.1, 0.2, [3, 1])
array = tr.SMatrices.from_array(sphere, basis, lattice=cell, kpar=[0, 0])
np.testing.assert_allclose(
    sum(array.tr(tr.plane_wave([0, 0, 1], 1, k0=2.1))), 1, atol=1e-10
)

modes = tr.SphericalWaveBasis.default(2)
positions = np.zeros((1, 3))


def reflectance(radius):
    particle = ad.sphere(2, 2.1, anp.reshape(radius, (1,)), [3.0, 1.0])
    coupling = ad.lattice_expansion(
        positions, positions, [2.1, 2.1], [0, 0], cell, destination=modes, source=modes
    )
    response = ad.interaction(particle, coupling)
    channels = ad.spherical_channels(
        positions,
        [2.1, 2.1],
        [[0, 0], [0, 0]],
        np.linalg.det(cell),
        basis=modes,
        polarizations=[1, 0],
        fixed_q=True,
    )
    value = ad.smatrix_from_array(response, channels)[1, 0, :, 0]
    return anp.sum(anp.real(value * anp.conj(value)))


gradient = advect.grad(reflectance)(np.array(0.2))
h = 1e-5
np.testing.assert_allclose(
    gradient, (reflectance(0.2 + h) - reflectance(0.2 - h)) / (2 * h), rtol=1e-7
)
print("Clean wheel: periodic power conservation and complete Advect gradient passed")

cb = tr.CylindricalWaveBasis.default([0.2], 1)
points = np.array([[0.8, 0.3, 0.1]])
coefficients = np.ones(len(cb), complex)
electric = tr.efield(points, basis=cb, k0=1.3)
np.testing.assert_allclose(
    electric @ coefficients,
    tr.diff.field(coefficients, points, cb, [1.3, 1.3])[0],
    rtol=1e-12,
    atol=1e-12,
)


def magnetic_energy(impedance):
    value = ad.hfield(
        coefficients, points, cb.positions, [1.3, 1.3], impedance, basis=cb
    )
    return anp.sum(anp.real(value * anp.conj(value)))


np.testing.assert_allclose(
    advect.grad(magnetic_energy)(np.array(0.8)),
    -2 * magnetic_energy(0.8) / 0.8,
    rtol=1e-12,
)
print("Clean wheel: cylindrical field operator and magnetic impedance gradient passed")


plane = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [1.5, -0.1]])
vectors = np.column_stack(plane.kvecs(1.3))
operator = tr.efield(points, basis=plane, k0=1.3)
np.testing.assert_allclose(
    operator, tr.diff.plane_field(None, points, vectors, plane.pol)[0], rtol=1e-12
)


def plane_energy(scale):
    field = ad.plane_field(
        np.ones(len(plane)), points / scale, vectors * scale, polarizations=plane.pol
    )
    return anp.sum(anp.real(field * anp.conj(field)))


np.testing.assert_allclose(advect.grad(plane_energy)(np.array(1.0)), 0, atol=1e-12)
print(
    "Clean wheel: plane operator and Advect coordinate/wavevector scale identity passed"
)


cylinder = tr.TMatrixC.cylinder([0.2], 3, 1.3, [0.2], [4, 1])
cports = tr.PlaneWaveBasisByComp.default([[0.2, 0.1]], "zx")
carray = tr.SMatrices.from_array(cylinder, cports, lattice=1.7, kpar=0.1)
np.testing.assert_allclose(sum(carray.tr([1, 0])), 1, atol=2e-10)


def cylindrical_radiation_energy(period):
    channels = ad.cylindrical_channels(
        cylinder.basis.positions,
        [1.3, 1.3],
        [0.1, 0.1],
        period,
        basis=cylinder.basis,
        kz_labels=[0.2, 0.2],
        polarizations=[1, 0],
    )
    radiated = channels[1]
    return anp.sum(anp.real(radiated * anp.conj(radiated)))


np.testing.assert_allclose(
    advect.grad(cylindrical_radiation_energy)(np.array(1.7)),
    -2 * cylindrical_radiation_energy(1.7) / 1.7,
    rtol=1e-12,
)
print(
    "Clean wheel: cylindrical array power and native radiation period gradient passed"
)


cincident = tr.expand((cylinder.basis, cports), k0=1.3)
cvectors = np.column_stack(cports.kvecs(1.3))
np.testing.assert_allclose(
    cincident,
    tr.diff.plane_expansion(cylinder.basis, cvectors, cports.pol)[0],
    atol=1e-14,
)
print("Clean wheel: cylindrical plane-illumination operator passed")


oriented_slab = tr.SMatrices.slab(0.4, cports, 1.3, [1, 2.3, 1])
np.testing.assert_allclose(sum(carray.add(oriented_slab).tr([1, 0])), 1, atol=2e-10)
print("Clean wheel: cylindrical array plus oriented slab conserves power")


def compact_reflectance(thickness):
    value = ad.layer_stack(
        [[1.3, 1.3], [2.0, 2.0], [1.3, 1.3]],
        [1.0, 0.65, 1.0],
        [[0.2, 0.3], [0.0, 0.0]],
        anp.reshape(thickness, (1,)),
        alignment="zx",
    )
    reflected = value[:, 1, 0, :, 0]
    return anp.sum(anp.real(reflected * anp.conj(reflected)))


np.testing.assert_allclose(
    advect.grad(compact_reflectance)(np.array(0.4)),
    (compact_reflectance(0.4 + h) - compact_reflectance(0.4 - h)) / (2 * h),
    rtol=1e-7,
    atol=1e-9,
)
print("Clean wheel: compact multilayer thickness adjoint passed")


array_basis = tr.CylindricalWaveBasis.default([0.2], 2)
array_cylinder = tr.TMatrixC.from_array(
    tr.TMatrix.sphere(2, 1.3, 0.2, [3, 1]), array_basis, lattice=1.7, kpar=0.2
)
array_scattering = np.eye(len(array_basis)) + 2 * array_cylinder.array
np.testing.assert_allclose(
    array_scattering.conj().T @ array_scattering, np.eye(len(array_basis)), atol=1e-10
)


def periodic_conversion_norm(period):
    value = ad.periodic_conversion(
        array_basis.positions,
        modes.positions,
        [1.3, 1.3],
        array_basis.kz,
        period,
        destination=array_basis,
        source=modes,
    )
    return anp.sum(anp.real(value * anp.conj(value)))


np.testing.assert_allclose(
    advect.grad(periodic_conversion_norm)(np.array(1.7)),
    -2 * periodic_conversion_norm(1.7) / 1.7,
    rtol=1e-12,
)
print("Clean wheel: spherical array to cylindrical power and period adjoint passed")


lower_interface = tr.SMatrices.interface(cports, 1.3, [1, 2.3])
upper_interface = tr.SMatrices.interface(cports, 1.3, [2.3, 1])
internal = lower_interface.illuminate([1, 0], smat=upper_interface)
np.testing.assert_allclose(internal[:2], [[1, 0], [0, 0]], atol=1e-12)


def internal_norm(scale):
    value = ad.smatrix_illuminate(
        lower_interface.array,
        upper_interface.array,
        scale * np.array([[1.0], [0.0]]),
        np.zeros((2, 1)),
    )[2:]
    return anp.sum(anp.real(value * anp.conj(value)))


np.testing.assert_allclose(
    advect.grad(internal_norm)(np.array(1.0)), 2 * internal_norm(1.0), atol=1e-12
)


def uniform_band_norm(k0, period):
    normal = anp.sqrt(k0**2 - 0.2**2 - 0.3**2)
    vectors = anp.stack([anp.stack([0.2, 0.3, normal])] * 2)
    smats = ad.propagation(vectors, anp.stack([0.0, 0.0, period]))
    k, _ = ad.bands(smats, period)
    return anp.sum(anp.real(k * anp.conj(k)))


gk, gp = advect.grad(uniform_band_norm, argnums=(0, 1))(np.array(1.3), np.array(0.4))
np.testing.assert_allclose(gk, 8 * 1.3, atol=2e-11)
np.testing.assert_allclose(gp, 0, atol=2e-11)
print("Clean wheel: internal fields and degenerate uniform-band adjoints passed")


np.testing.assert_allclose(
    tr.gfield(0, points, basis=cb, k0=1.3) + tr.gfield(1, points, basis=cb, k0=1.3),
    tr.efield(points, basis=cb, k0=1.3),
    atol=1e-12,
)


def singular_energy(matrix):
    return anp.sum(ad.svdvals(matrix) ** 2)


matrix = np.array([[1.2, 0.1j], [0.3, 2.1], [0.4j, 0.5]])
np.testing.assert_allclose(advect.grad(singular_energy)(matrix), 2 * matrix, atol=1e-12)


def chirality(radius):
    particle = ad.sphere(
        2,
        1.3,
        anp.reshape(radius, (1,)),
        [3.0 + 0.2j, 1.0],
        [1.4 + 0.1j, 1.0],
        [0.12 + 0.02j, 0.0],
    )
    return ad.tmatrix_metric(particle, polarizations=modes.pol, kind="chi")


np.testing.assert_allclose(
    advect.grad(chirality)(np.array(0.3)),
    (chirality(0.3 + h) - chirality(0.3 - h)) / (2 * h),
    rtol=1e-7,
    atol=1e-10,
)
print(
    "Clean wheel: Riemann-Silberstein fields, native SVD and chirality gradient passed"
)

theta, weights = tr.ebcm._quadrature(48)
surface_radii = 0.3 * (1 + 0.23 * np.cos(theta) ** 2)
surface_slopes = -0.138 * np.cos(theta) * np.sin(theta)
regular, _ = tr.diff.ebcm_qmat(
    surface_radii,
    surface_slopes,
    np.full((2, 2), 1.3),
    np.ones(2),
    theta=theta,
    weights=weights,
    out=modes,
    singular=False,
)
np.testing.assert_allclose(regular, 0, atol=1e-15)


def surface_scale_norm(scale):
    q = ad.ebcm_qmat(
        surface_radii * scale,
        surface_slopes * scale,
        np.array([[2.0, 2.1], [1.3, 1.3]]) / scale,
        [0.7, 1.0],
        theta=theta,
        weights=weights,
        out=modes,
    )
    return anp.sum(anp.real(q * anp.conj(q)))


np.testing.assert_allclose(
    advect.grad(surface_scale_norm)(np.array(1.0)),
    4 * surface_scale_norm(1.0),
    rtol=1e-12,
)
print("Clean wheel: EBCM zero contrast and native surface-scale adjoint passed")

np.testing.assert_allclose(
    tr.chirality_density(plane, 2.1, z=(0, 1))[0],
    np.diag(2 * (2 * plane.pol - 1)),
    atol=1e-12,
)


def density_scale(scale):
    value = ad.chirality_density(
        scale * np.array([1.3 + 0.1j]),
        scale * np.array([1.1 + 0.2j]),
        np.array([-0.2, 0.7]) / scale,
    )
    return anp.real(anp.sum(value))


np.testing.assert_allclose(advect.grad(density_scale)(np.array(1.0)), 0, atol=1e-12)
print("Clean wheel: interval chirality density and native scaling adjoint passed")

source = tr.spherical_wave(1, 0, 1, k0=1.3)
particle = tr.TMatrix.sphere(3, 1.3, 0.2, [3, 1])
scattered = tr.MultipoleWave(
    particle @ source, basis=particle.basis, k0=1.3, modetype="singular"
)
np.testing.assert_allclose(
    scattered.hfield(points),
    tr.hfield(points, basis=particle.basis, k0=1.3, modetype="singular")
    @ scattered.array,
    atol=1e-12,
)
print("Clean wheel: multipole-source scattering and weighted magnetic samples passed")

np.testing.assert_allclose(carray.cd([1, 0])[1], 0, atol=1e-10)
print("Clean wheel: lossless S-matrix outgoing-power contrast passed")

np.testing.assert_allclose(
    source.changepoltype().efield(points), source.efield(points), atol=1e-12
)
print("Clean wheel: shared polarization conversion preserves source fields")

np.testing.assert_array_equal(tr.expand(plane, k0=1.3), np.eye(len(plane)))
np.testing.assert_allclose(
    tr.translate([0, 0, 0], basis=plane, k0=1.3), np.eye(len(plane)), atol=0
)


def phase_scale(scale):
    phases = ad.plane_phases(
        np.array([[0.2, 0.1, -0.3]]) * scale, np.array([[0.2, 0.3, 1.3 + 0.1j]]) / scale
    )
    return anp.real(anp.sum(phases))


np.testing.assert_allclose(advect.grad(phase_scale)(np.array(1.0)), 0, atol=1e-12)
print(
    "Clean wheel: plane translation, identity expansion and native phase adjoint passed"
)

permuted = plane.permute(1)
transform = tr.permute(1, basis=plane, k0=1.3)
inverse = tr.permute(-1, basis=permuted, k0=1.3)
np.testing.assert_allclose(inverse @ transform, np.eye(len(plane)), atol=1e-12)
print("Clean wheel: plane-wave coordinate transformation passed")


# Direct scalar dispatch, NumPy ufunc output semantics and the owned adjoint all
# use the installed Rust extension in an environment without SciPy or treams.
z = np.array([1.3 + 0.2j, 2.1 - 0.1j])
value = tr.special.hankel1(3, z)
out = np.zeros_like(z)
tr.special.hankel1(3, z, out=out, where=[True, False])
np.testing.assert_allclose(out, [tr.special.hankel1(3, complex(z[0])), 0], rtol=1e-13)
derivative, context = tr.diff.bessel(3, z, kind="h1")
np.testing.assert_allclose(derivative, value, rtol=1e-13)
np.testing.assert_allclose(
    context.pullback(np.ones_like(z)), tr.special.hankel1_d(3, z).conj(), rtol=1e-13
)
print(
    "Clean wheel: scalar and ufunc special functions, output masks and native adjoint passed"
)

scalar, context = tr.diff.bessel(3, 1.3 + 0.2j, kind="h1")
np.testing.assert_allclose(scalar, value[0], rtol=1e-13)
np.testing.assert_allclose(
    context.pullback(np.array(1 + 0j)),
    tr.special.hankel1_d(3, 1.3 + 0.2j).conjugate(),
    rtol=1e-13,
)

z = np.array([-1.0, -0.4, 0.7, 1.0])
value, context = tr.diff.angular(3, 0, z)
np.testing.assert_allclose(value, (5 * z**3 - 3 * z) / 2, atol=2e-15)
np.testing.assert_allclose(context.pullback(np.ones(4, complex)), (15 * z**2 - 3) / 2)
out = np.empty(4, complex)
tr.special.pi_fun(3, 1, z, out=out)
np.testing.assert_allclose(out, -(15 * z**2 - 3) / 2)
np.testing.assert_allclose(
    advect.grad(lambda x: anp.real(anp.sum(ad.angular(x, degree=3, order=0))))(z),
    (15 * z**2 - 3) / 2,
)
print("Clean wheel: angular ufuncs, finite polar derivatives and Advect passed")


beta = np.linspace(0, 0.7, 2048)
value, context = tr.diff.wigner(1, -1, 0, 0.0, beta, 0.0)
np.testing.assert_allclose(value, np.sin(beta) / np.sqrt(2), atol=1e-15)
np.testing.assert_allclose(
    context.pullback(np.ones_like(beta, dtype=complex))[1],
    np.cos(beta) / np.sqrt(2),
    atol=1e-15,
)
np.testing.assert_allclose(tr.special.wignersmalld(1, -1, 0, beta), value, atol=1e-15)
np.testing.assert_allclose(tr.special.wigner3j(1, 1, 0, 0, 0, 0), -1 / np.sqrt(3))
np.testing.assert_allclose(
    tr.special.incgamma(1, [0.7, 1.3]), np.exp(-np.array([0.7, 1.3]))
)
assert np.isfinite(tr.special.intkambe(-2, 0.7 + 0.1j, 0.8))
print("Clean wheel: Wigner symbols and Euler adjoints, gamma and Kambe ufuncs passed")


# A regular m=0 M wave on the axis vanishes; use an off-axis sample and the
# complete geometric scale identity to check axial, medium and position VJPs.
cb = tr.CylindricalWaveBasis.default([0.2], 1)
cp = np.array([[0.4, -0.3, 0.2]])
ck = np.array([1.3 + 0.05j, 1.5 + 0.07j])
ca = np.full(len(cb), 0.2 + 0.3j)
cv, cc = tr.diff.field(ca, cp, cb, ck)
cg = cc.pullback_axial(np.full_like(cv, 0.3 + 0.2j))
np.testing.assert_allclose(
    np.sum(cg[1] * cp), np.vdot(cg[3], ck).real + np.dot(cg[4], cb.kz), atol=1e-13
)
np.testing.assert_allclose(
    advect.grad(
        lambda kz: anp.real(
            anp.sum(ad.field(ca, cp, cb.positions, ck, basis=cb, kzs=kz))
        )
    )(cb.kz),
    tr.diff.field(ca, cp, cb, ck)[1].pullback_axial(np.ones_like(cv))[4],
    atol=1e-13,
)
print("Clean wheel: cylindrical axial field adjoints and Advect composition passed")


cd = tr.CylindricalWaveBasis.default([0.2], 1, positions=[[0.3, 0.2, -0.1]])
cv, cc = tr.diff.expansion(cd, cb, ck)
cg = cc.pullback_axial(np.ones_like(cv))
np.testing.assert_allclose(
    np.sum(cg[0] * cd.positions), np.vdot(cg[2], ck).real + cg[3][0] * 0.2, atol=1e-13
)
cv, cc = tr.lattice.expansion_with_context(cb, cb, ck, [[1.7]], [0.1])
cg = cc.pullback_axial(np.ones_like(cv))
np.testing.assert_allclose(
    np.sum(cg[4] * 1.7),
    np.vdot(cg[2], ck).real + cg[3][0] * 0.1 + cg[5][0] * 0.2,
    atol=1e-11,
)
print("Clean wheel: finite and periodic shared axial-group pullbacks passed")


xyz = np.array([[0.3, 0.4, 0.5]])
rvec = np.array([0.2 + 0.1j, -0.3, 0.4j])
sph = tr.special.car2sph(xyz)
np.testing.assert_allclose(tr.special.sph2car(sph), xyz, atol=1e-15)
v = tr.special.vcar2sph(rvec, xyz)
np.testing.assert_allclose(
    tr.special.vsph2car(v, sph), np.broadcast_to(rvec, v.shape), atol=1e-15
)
_, cx = tr.diff.coordinates(xyz, kind="car2sph")
np.testing.assert_allclose(
    cx.pullback(np.array([[1.0, 0.0, 0.0]])), xyz / np.linalg.norm(xyz), atol=1e-15
)
np.testing.assert_allclose(
    advect.grad(
        lambda x: anp.sum(anp.abs(ad.vector_coordinates(rvec, x, kind="car2sph")) ** 2)
    )(xyz),
    0,
    atol=1e-14,
)
print(
    "Clean wheel: coordinate gufuncs, owned pullbacks and vector norm gradient passed"
)


orders = np.arange(-20, 21)
np.testing.assert_allclose(
    np.sum(np.abs(tr.special.sph_harm(orders, 20, 0.3, 0.7)) ** 2),
    41 / (4 * np.pi),
    rtol=1e-12,
)


def local_wave_energy(scale):
    value = ad.vector_wave(
        0.3 * scale,
        0.4 * scale,
        1.2 * scale,
        0.5 / scale,
        0.7 / scale,
        0.9 / scale,
        kind="vpw_A",
        polarization=1,
    )
    return anp.real(anp.sum(anp.conj(value) * value))


np.testing.assert_allclose(local_wave_energy(1.0), 1.0, atol=1e-13)
np.testing.assert_allclose(
    advect.grad(local_wave_energy)(np.array(1.0)), 0.0, atol=1e-13
)
print("Clean wheel: normalized harmonics and complete local-wave scale adjoint passed")


np.testing.assert_allclose(tr.special.tl_vsw_rA(2, 1, 2, 1, 0, 0.7, 0.3), 1, atol=1e-13)


def polar_energy(scale):
    value = ad.cylindrical_translation(
        1.3, 0.4, 0.3 / scale, 0.2 * scale, order=2, singular=False
    )
    return anp.real(anp.conj(value) * value)


np.testing.assert_allclose(advect.grad(polar_energy)(np.array(1.0)), 0, atol=1e-13)
print(
    "Clean wheel: polar translation identity and common-axial-label scale adjoint passed"
)


def gamma_identity(z):
    return anp.real(
        ad.incgamma(z, n=2.5) - 1.5 * ad.incgamma(z, n=1.5) - z**1.5 * anp.exp(-z)
    )


np.testing.assert_allclose(
    advect.grad(gamma_identity)(np.asarray(1.2 + 0.1j)), 0, atol=2e-12
)
print("Clean wheel: incomplete-gamma recurrence and native argument adjoint passed")

cell = tr.Lattice([[1.7, 0.2], [0.0, 1.8]])
np.testing.assert_allclose(
    np.asarray(cell) @ cell.reciprocal.T, 2 * np.pi * np.eye(2), atol=2e-15
)
ports = tr.PlaneWaveBasisByComp.diffr_orders([0.1, 0.2], cell, 4)
assert ports.lattice == cell
assert ports.kpar == tr.WaveVector([0.1, 0.2], alignment="xy")
np.testing.assert_array_equal(tr.lattice.cubeedge(3, 0), [[0, 0, 0]])
np.testing.assert_allclose(
    tr.misc.refractive_index(3 + 0.2j, 1.1, 0.1), tr.Material(3 + 0.2j, 1.1, 0.1).nmp
)
print("Clean wheel: reciprocal geometry, basis metadata and material branches passed")


# Ewald components and geometry pullbacks require no external numerical package.
k = np.asarray(2.1 + 0.2j)
q = np.array([0.1, 0.2])
a = np.diag([1.5, 1.7])
r = np.array([0.19, 0.11, 0.07])
value, context = tr.diff.lattice_sum(2, 2, -1, k, q, a, r, 0.9)
parts = sum(
    tr.diff.lattice_sum(2, 2, -1, k, q, a, r, 0.9, part=part)[0]
    for part in ("real", "reciprocal")
)
np.testing.assert_allclose(value, parts, rtol=3e-12, atol=3e-12)
gk, gq, ga, gr, ge = context.pullback(np.asarray(1, complex))
np.testing.assert_allclose(
    (np.vdot(gk, k) + np.vdot(gq, q) - np.vdot(ga, a) - np.vdot(gr, r)).real,
    0,
    atol=3e-10,
)
np.testing.assert_allclose(ge, 0)
print("Clean wheel: Ewald decomposition and native geometry/scale adjoint passed")


fractional, context = tr.diff.angular(32.3, 12, -0.7)
np.testing.assert_allclose(fractional, 1.931539353538286e16, rtol=3e-12)
np.testing.assert_allclose(
    context.pullback(np.asarray(1, complex)), -8.220038417299478e18, rtol=3e-12
)
table = np.arange(25, dtype=complex).reshape(1, 1, 1, 25) * (0.01 + 0.02j)
value, context = tr.diff.periodic_from_table(table, modes, poltype="parity")
cotangent = np.full_like(value, 0.2 + 0.3j)
np.testing.assert_allclose(
    np.vdot(cotangent, value).real,
    np.vdot(context.pullback(cotangent), table).real,
    rtol=2e-13,
)
print("Clean wheel: fractional Legendre and custom periodic-table adjoints passed")


def interface_power(impedance):
    ks = anp.array([[1.3, 1.3], [2.0, 2.0]])
    zs = anp.stack([1.0, impedance])
    directions = anp.array([[0.2, 0.3]])
    blocks = ad.interface(ks, zs, directions[0])
    power = ad.smatrix_tr(
        blocks,
        anp.array([[1.0], [0.2j]]),
        ks[::-1],
        zs[::-1],
        directions,
        modes=[(0, 0), (0, 1)],
    )
    return anp.real(anp.sum(power))


np.testing.assert_allclose(interface_power(0.7), 1, atol=2e-13)
np.testing.assert_allclose(advect.grad(interface_power)(np.array(0.7)), 0, atol=2e-12)
np.testing.assert_allclose(
    oriented_slab.permute().permute(-1).array, oriented_slab.array, atol=2e-13
)
np.testing.assert_allclose(
    ports.rotate(0.3).rotate(-0.3).components, ports.components, atol=2e-14
)
print("Clean wheel: native power adjoint and matrix coordinate workflows passed")

foreign_basis = tr.SphericalWaveBasis(list(sphere.basis), positions=[[1, 2, 3]])
np.testing.assert_array_equal(
    sphere[foreign_basis].basis.positions, sphere.basis.positions
)
print("Clean wheel: channel selection preserves physical expansion origins")
