# /// script
# requires-python = ">=3.12,<3.16"
# ///
"""Smoke-test one installed treams-rs wheel under its dependency profiles.

Run with the interpreter of a clean environment containing the wheel and only
the dependencies of the selected profiles; neither the upstream ``treams``
oracle nor SciPy may be installed:

* ``base``: ``treams-rs`` with NumPy only;
* ``advect``: ``treams-rs[advect]``, adding complete Advect objectives;
* ``io``: ``treams-rs[io]``, adding the HDF5 interchange round trip.

``advect`` and ``io`` may be combined when both extras are installed together,
for example ``smoke_wheel_install.py advect io``; ``base`` stands alone.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import platform
import pydoc
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import float_environment
import numpy as np

import treams_rs as tr
from treams_rs import _native
from treams_rs.testing import check_pullback

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

# Profile name -> the module its optional dependency provides.
_PROFILE_MODULES = {"advect": "advect", "io": "h5py"}
# Development oracles and optional frameworks outside every smoke profile.
_ABSENT_MODULES = ("treams", "scipy", "autograd", "jax", "torch")
_STEP = 1e-5
# Machines whose native guard manages the flushing bits (x86-64 and AArch64).
_FLUSHING_MACHINES = {"x86_64", "amd64", "aarch64", "arm64"}


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(
    actual: ArrayLike, desired: ArrayLike, *, rtol: float = 1e-7, atol: float = 0
) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol
    )


def _parse_args(argv: list[str] | None) -> frozenset[str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profiles",
        nargs="+",
        choices=("base", *_PROFILE_MODULES),
        help="Dependency profiles installed beside the treams-rs wheel.",
    )
    profiles = frozenset(cast("list[str]", parser.parse_args(argv).profiles))
    if "base" in profiles and len(profiles) > 1:
        parser.error("the base profile cannot be combined with optional profiles")
    return profiles


def _require_module(name: str, *, present: bool) -> None:
    found = importlib.util.find_spec(name) is not None
    _require(
        condition=found is present,
        message=f"Expected module {name!r} to be {'installed' if present else 'absent'}",
    )


def _check_distribution() -> None:
    version = metadata.version("treams-rs")
    _require(
        condition=tr.__version__ == version,
        message="Package and installed distribution versions disagree",
    )
    package = Path(tr.__file__ or "").resolve().parent
    _require(
        condition=package.is_relative_to(Path(sys.prefix).resolve()),
        message=f"treams_rs imported from outside the smoke environment: {package}",
    )
    _require(
        condition=Path(_native.__file__ or "").resolve().parent == package,
        message="The native extension is not part of the installed treams_rs package",
    )
    _require(
        condition=_native.build_profile() == "release",
        message=f"Expected a release native extension, got {_native.build_profile()!r}",
    )
    for name in ("py.typed", "_native.pyi"):
        _require(
            condition=(package / name).is_file(),
            message=f"Installed package is missing {name}",
        )
    print(f"Clean wheel: treams-rs {version} release extension installed")

    # Optimized code must keep native work inside the floating-point guard.
    if float_environment.supported():
        print(f"Clean wheel: {float_environment.check()}")
    else:
        _require(
            condition=platform.machine().lower() not in _FLUSHING_MACHINES,
            message=f"Floating-point mode check unavailable on {platform.machine()}",
        )
        print("Clean wheel: this platform cannot flush subnormals; mode check skipped")


def _check_dependencies(profiles: frozenset[str]) -> None:
    for profile, module in _PROFILE_MODULES.items():
        _require_module(module, present=profile in profiles)
    for module in _ABSENT_MODULES:
        _require_module(module, present=False)
    for name, module in (*_PROFILE_MODULES.items(), ("jax", "jax"), ("torch", "torch")):
        if name in profiles:
            continue
        try:
            getattr(tr, name)
        except ModuleNotFoundError as error:
            _require(
                condition=f"treams-rs[{name}]" in str(error),
                message=f"Missing {module} lacks install guidance: {error}",
            )
        else:
            message = f"Clean wheel unexpectedly imported optional {name}"
            raise AssertionError(message)
    catalog = tr.support_catalog()
    versions = cast("dict[str, object]", catalog["optional_dependencies"])
    installed = {_PROFILE_MODULES[profile] for profile in profiles - {"base"}}
    for module in (*_PROFILE_MODULES.values(), "jax", "torch"):
        _require(
            condition=(versions[module] is not None) is (module in installed),
            message=f"Catalog misreports optional {module}: {versions[module]!r}",
        )
    print("Clean wheel: optional dependencies match the profile with install guidance")


def _check_numpy_workflows() -> None:
    # NumPy 2 uses int64 by default on Windows too, where C long is only 32 bits.
    for dtype in (np.int32, np.int64):
        orders = np.arange(-2, 3, dtype=dtype)
        _close(
            tr.cw.rotate(0.2, orders, 1, 0.2, orders, 1, 0.3), np.exp(-0.3j * orders)
        )
        volumes = tr.lattice.volume(np.array([[[2**30, 0], [0, 8]]], dtype=dtype))
        _require(
            condition=isinstance(volumes, np.ndarray)
            and volumes.dtype == np.dtype("int64"),
            message="Integer volume lost its dtype",
        )
        np.testing.assert_array_equal(volumes, [2**33])
    for order in (2**32 + 1, np.array([2**32 + 1], dtype=np.int64)):
        try:
            tr.cw.rotate(0.2, order, 1, 0.2, order, 1, 0.3)
        except ValueError:
            pass
        else:
            raise AssertionError("An oversized integer order was silently truncated")

    # Primary physical workflow works in the isolated wheel, without an oracle.
    particle = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    source = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
    wave = particle.scatter(source)
    _close(
        particle.cross_sections(source).scattering,
        particle.cross_sections(source).extinction,
        rtol=1e-11,
    )
    _require(
        condition=wave.efield([[0.1, 0.2, 0.8]]).shape == (1, 3),
        message="Scattered field has the wrong shape",
    )
    _require(
        condition=not hasattr(particle, "efield"),
        message="A T-matrix exposes fields directly",
    )

    cell = np.diag([1.7, 1.8])
    basis = tr.PlaneWavePorts.diffr_orders([0, 0], cell, 4)
    sphere = tr.TMatrix.sphere(2, 2.1, 0.2, [3, 1])
    array = tr.solve_periodic(sphere, lattice=cell, kpar=[0, 0]).to_smatrix(basis)
    _close(sum(array.tr(tr.plane_wave([0, 0, 1], 1, k0=2.1))), 1, atol=1e-10)
    print("Clean wheel: periodic power conservation passed")

    cb = tr.CylindricalBasis.default([0.2], 1)
    points = np.array([[0.8, 0.3, 0.1]])
    coefficients = np.ones(len(cb), complex)
    electric = tr.operators.efield(points, basis=cb, k0=1.3)
    _close(
        electric @ coefficients,
        tr.diff.field(coefficients, points, cb, [1.3, 1.3])[0],
        rtol=1e-12,
        atol=1e-12,
    )
    print("Clean wheel: cylindrical field operator passed")

    plane = tr.PlaneWavePorts.default([[0.2, 0.3], [1.5, -0.1]])
    vectors = np.column_stack(plane.kvecs(1.3))
    operator = tr.operators.efield(points, basis=plane, k0=1.3)
    _close(
        operator, tr.diff.plane_field(None, points, vectors, plane.pol)[0], rtol=1e-12
    )
    print("Clean wheel: plane operator passed")

    cylinder = tr.CylindricalTMatrix.cylinder([0.2], 3, 1.3, [0.2], [4, 1])
    cports = tr.PlaneWavePorts.default([[0.2, 0.1]], "zx")
    carray = tr.solve_periodic(cylinder, lattice=1.7, kpar=0.1).to_smatrix(cports)
    _close(sum(carray.tr([1, 0])), 1, atol=2e-10)
    print("Clean wheel: cylindrical array power passed")

    cincident = tr.operators.expand((cylinder.basis, cports), k0=1.3)
    cvectors = np.column_stack(cports.kvecs(1.3))
    _close(
        cincident,
        tr.diff.plane_expansion(cylinder.basis, cvectors, cports.pol)[0],
        atol=1e-14,
    )
    print("Clean wheel: cylindrical plane-illumination operator passed")

    oriented_slab = tr.SMatrix.slab(0.4, cports, 1.3, [1, 2.3, 1])
    _close(sum(carray.add(oriented_slab).tr([1, 0])), 1, atol=2e-10)
    print("Clean wheel: cylindrical array plus oriented slab conserves power")

    array_basis = tr.CylindricalBasis.default([0.2], 2)
    array_cylinder = tr.solve_periodic(
        tr.TMatrix.sphere(2, 1.3, 0.2, [3, 1]), lattice=1.7, kpar=0.2
    ).to_cylindrical(array_basis)
    array_scattering = np.eye(len(array_basis)) + 2 * array_cylinder.array
    _close(
        array_scattering.conj().T @ array_scattering,
        np.eye(len(array_basis)),
        atol=1e-10,
    )
    print("Clean wheel: spherical array to cylindrical power passed")

    lower_interface = tr.SMatrix.interface(cports, 1.3, [1, 2.3])
    upper_interface = tr.SMatrix.interface(cports, 1.3, [2.3, 1])
    internal = lower_interface.illuminate([1, 0], smat=upper_interface)
    _close(internal[:2], [[1, 0], [0, 0]], atol=1e-12)
    print("Clean wheel: internal fields passed")

    _close(
        tr.operators.gfield(0, points, basis=cb, k0=1.3)
        + tr.operators.gfield(1, points, basis=cb, k0=1.3),
        tr.operators.efield(points, basis=cb, k0=1.3),
        atol=1e-12,
    )
    print("Clean wheel: Riemann-Silberstein fields passed")

    modes = tr.SphericalBasis.default(2)
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
        destination=modes,
        singular=False,
    )
    _close(regular, 0, atol=1e-15)
    print("Clean wheel: EBCM zero contrast passed")

    _close(
        tr.chirality_density(plane, 2.1, z=(0, 1))[0],
        np.diag(2 * (2 * plane.pol - 1)),
        atol=1e-12,
    )
    print("Clean wheel: interval chirality density passed")

    multipole = tr.spherical_wave(1, 0, 1, k0=1.3)
    particle = tr.TMatrix.sphere(3, 1.3, 0.2, [3, 1])
    scattered = tr.Wave(
        particle @ multipole, basis=particle.basis, k0=1.3, modetype="singular"
    )
    _close(
        scattered.hfield(points),
        tr.operators.hfield(points, basis=particle.basis, k0=1.3, modetype="singular")
        @ scattered.array,
        atol=1e-12,
    )
    print(
        "Clean wheel: multipole-source scattering and weighted magnetic samples passed"
    )

    _close(carray.cd([1, 0])[1], 0, atol=1e-10)
    print("Clean wheel: lossless S-matrix outgoing-power contrast passed")

    _close(
        multipole.changepoltype().efield(points), multipole.efield(points), atol=1e-12
    )
    print("Clean wheel: shared polarization conversion preserves source fields")

    np.testing.assert_array_equal(
        tr.operators.expand(plane, k0=1.3), np.eye(len(plane))
    )
    _close(
        tr.operators.translate([0, 0, 0], basis=plane, k0=1.3),
        np.eye(len(plane)),
        atol=0,
    )
    print("Clean wheel: plane translation and identity expansion passed")

    permuted = plane.permute(1)
    transform = tr.operators.permute(1, basis=plane, k0=1.3)
    inverse = tr.operators.permute(-1, basis=permuted, k0=1.3)
    _close(inverse @ transform, np.eye(len(plane)), atol=1e-12)
    print("Clean wheel: plane-wave coordinate transformation passed")

    _check_numpy_special_functions()
    _check_numpy_geometry(modes)

    ports = tr.PlaneWavePorts.diffr_orders(
        [0.1, 0.2], tr.Lattice([[1.7, 0.2], [0.0, 1.8]]), 4
    )
    _close(oriented_slab.permute().permute(-1).array, oriented_slab.array, atol=2e-13)
    _close(ports.rotate(0.3).rotate(-0.3).components, ports.components, atol=2e-14)
    print("Clean wheel: matrix coordinate workflows passed")

    foreign_basis = tr.SphericalBasis(list(sphere.basis), positions=[[1, 2, 3]])
    np.testing.assert_array_equal(
        sphere[foreign_basis].basis.positions, sphere.basis.positions
    )
    print("Clean wheel: channel selection preserves physical expansion origins")

    _require(
        condition="support_catalog" in pydoc.render_doc(tr),
        message="Offline help omits the support catalog",
    )
    catalog = tr.support_catalog()
    backends = cast("dict[str, dict[str, object]]", catalog["backends"])
    rows = cast("list[dict[str, object]]", catalog["api"])
    _require(condition=bool(backends["cpu"]["compiled"]), message="CPU backend missing")
    _require(
        condition=any(row["path"] == "treams_rs.jax.wrap" for row in rows),
        message="Catalog omits optional framework entries",
    )
    check_pullback(
        tr.diff.solve, np.eye(2, dtype=complex), np.ones((2, 1), dtype=complex)
    )
    print("Clean wheel: offline catalog, local help and native pullback check passed")


def _check_numpy_special_functions() -> None:
    # Direct scalar dispatch, NumPy ufunc output semantics and the owned adjoint
    # all use the installed Rust extension without SciPy or treams.
    z = np.array([1.3 + 0.2j, 2.1 - 0.1j])
    value = np.asarray(tr.special.hankel1(3, z))
    out = np.zeros_like(z)
    tr.special.hankel1(3, z, out=out, where=[True, False])
    _close(out, np.array([tr.special.hankel1(3, complex(z[0])), 0]), rtol=1e-13)
    derivative, context = tr.diff.bessel(3, z, function="h1")
    _close(derivative, value, rtol=1e-13)
    _close(
        context.pullback(np.ones_like(z)), tr.special.hankel1_d(3, z).conj(), rtol=1e-13
    )
    scalar, context = tr.diff.bessel(3, 1.3 + 0.2j, function="h1")
    _close(scalar, value[0], rtol=1e-13)
    _close(
        context.pullback(np.array(1 + 0j)),
        np.conjugate(tr.special.hankel1_d(3, 1.3 + 0.2j)),
        rtol=1e-13,
    )
    print(
        "Clean wheel: scalar and ufunc special functions, output masks and adjoint passed"
    )

    x = np.array([-1.0, -0.4, 0.7, 1.0])
    legendre, context = tr.diff.angular(3, 0, x)
    _close(legendre, (5 * x**3 - 3 * x) / 2, atol=2e-15)
    _close(context.pullback(np.ones(4, complex)), (15 * x**2 - 3) / 2)
    angular = np.empty(4, complex)
    tr.special.pi_fun(3, 1, x, out=angular)
    _close(angular, -(15 * x**2 - 3) / 2)
    print("Clean wheel: angular ufuncs and finite polar derivatives passed")

    beta = np.linspace(0, 0.7, 2048)
    wigner, context = tr.diff.wignerd(1, -1, 0, 0.0, beta, 0.0)
    _close(wigner, np.sin(beta) / np.sqrt(2), atol=1e-15)
    _close(
        context.pullback(np.ones_like(beta, dtype=complex))[1],
        np.cos(beta) / np.sqrt(2),
        atol=1e-15,
    )
    _close(tr.special.wignersmalld(1, -1, 0, beta), wigner, atol=1e-15)
    _close(tr.special.wigner3j(1, 1, 0, 0, 0, 0), -1 / np.sqrt(3))
    _close(tr.special.incgamma(1, [0.7, 1.3]), np.exp(-np.array([0.7, 1.3])))
    _require(
        condition=bool(np.isfinite(tr.special.intkambe(-2, 0.7 + 0.1j, 0.8))),
        message="Kambe integral is not finite",
    )
    print(
        "Clean wheel: Wigner symbols and Euler adjoints, gamma and Kambe ufuncs passed"
    )

    # A regular m=0 M wave on the axis vanishes; use an off-axis sample and the
    # complete geometric scale identity to check axial, medium and position VJPs.
    cb = tr.CylindricalBasis.default([0.2], 1)
    cp = np.array([[0.4, -0.3, 0.2]])
    ck = np.array([1.3 + 0.05j, 1.5 + 0.07j])
    ca = np.full(len(cb), 0.2 + 0.3j)
    cv, cc = tr.diff.field(ca, cp, cb, ck)
    cg = cc.pullback_axial(np.full_like(cv, 0.3 + 0.2j))
    _close(
        np.sum(cg[1] * cp), np.vdot(cg[3], ck).real + np.dot(cg[4], cb.kz), atol=1e-13
    )
    print("Clean wheel: cylindrical axial field adjoints passed")

    cd = tr.CylindricalBasis.default([0.2], 1, positions=[[0.3, 0.2, -0.1]])
    cv, cc = tr.diff.expansion(cd, cb, ck)
    cg = cc.pullback_axial(np.ones_like(cv))
    _close(
        np.sum(cg[0] * cd.positions),
        np.vdot(cg[2], ck).real + cg[3][0] * 0.2,
        atol=1e-13,
    )
    cv, cc = tr.diff.lattice_expansion(cb, cb, ck, [0.1], [[1.7]])
    cg = cc.pullback_axial(np.ones_like(cv))
    _close(
        np.sum(cg[4] * 1.7),
        np.vdot(cg[2], ck).real + cg[3][0] * 0.1 + cg[5][0] * 0.2,
        atol=1e-11,
    )
    print("Clean wheel: finite and periodic shared axial-group pullbacks passed")

    xyz = np.array([[0.3, 0.4, 0.5]])
    rvec = np.array([0.2 + 0.1j, -0.3, 0.4j])
    sph = tr.special.car2sph(xyz)
    _close(tr.special.sph2car(sph), xyz, atol=1e-15)
    vector = tr.special.vcar2sph(rvec, xyz)
    _close(
        tr.special.vsph2car(vector, sph),
        np.broadcast_to(rvec, vector.shape),
        atol=1e-15,
    )
    _, cx = tr.diff.coordinates(xyz, kind="car2sph")
    _close(
        cx.pullback(np.array([[1.0, 0.0, 0.0]])),
        xyz / np.linalg.norm(xyz),
        atol=1e-15,
    )
    print("Clean wheel: coordinate gufuncs and owned pullbacks passed")

    orders = np.arange(-20, 21)
    _close(
        np.sum(np.abs(tr.special.sph_harm(orders, 20, 0.3, 0.7)) ** 2),
        41 / (4 * np.pi),
        rtol=1e-12,
    )
    _close(tr.special.tl_vsw_rA(2, 1, 2, 1, 0, 0.7, 0.3), 1, atol=1e-13)
    print("Clean wheel: normalized harmonics and polar translation identity passed")


def _check_numpy_geometry(modes: tr.SphericalBasis) -> None:
    cell = tr.Lattice([[1.7, 0.2], [0.0, 1.8]])
    _close(
        np.asarray(cell, dtype=float) @ cell.reciprocal.T,
        2 * np.pi * np.eye(2),
        atol=2e-15,
    )
    ports = tr.PlaneWavePorts.diffr_orders([0.1, 0.2], cell, 4)
    _require(condition=ports.lattice == cell, message="Ports lost their lattice")
    _require(
        condition=ports.kpar == tr.WaveVector([0.1, 0.2], alignment="xy"),
        message="Ports lost their parallel wave vector",
    )
    np.testing.assert_array_equal(tr.lattice.cubeedge(3, 0), [[0, 0, 0]])
    _close(
        tr.misc.refractive_index(3 + 0.2j, 1.1, 0.1),
        tr.Material(3 + 0.2j, 1.1, 0.1).nmp,
    )
    print(
        "Clean wheel: reciprocal geometry, basis metadata and material branches passed"
    )

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
    _close(value, parts, rtol=3e-12, atol=3e-12)
    gk, gq, ga, gr, ge = context.pullback(np.asarray(1, complex))
    _close(
        (np.vdot(gk, k) + np.vdot(gq, q) - np.vdot(ga, a) - np.vdot(gr, r)).real,
        0,
        atol=3e-10,
    )
    _close(ge, 0)
    print("Clean wheel: Ewald decomposition and native geometry/scale adjoint passed")

    fractional, context = tr.diff.angular(32.3, 12, -0.7)
    _close(fractional, 1.931539353538286e16, rtol=3e-12)
    _close(context.pullback(np.asarray(1, complex)), -8.220038417299478e18, rtol=3e-12)
    table = np.arange(25, dtype=complex).reshape(1, 1, 1, 25) * (0.01 + 0.02j)
    value, context = tr.diff.lattice_expansion_from_table(
        table, modes, poltype="parity"
    )
    cotangent = np.full_like(value, 0.2 + 0.3j)
    _close(
        np.vdot(cotangent, value).real,
        np.vdot(context.pullback(cotangent), table).real,
        rtol=2e-13,
    )
    print("Clean wheel: fractional Legendre and custom periodic-table adjoints passed")


def _check_advect_workflows() -> None:
    advect = importlib.import_module("advect")
    anp = cast("Any", importlib.import_module("advect.numpy"))
    ad = cast("Any", importlib.import_module("treams_rs.advect"))
    grad = cast("Callable[..., Callable[..., Any]]", advect.grad)

    for name in (
        "PlaneWavePorts",
        "PlaneWaveBasis",
        "SphericalBasis",
        "CylindricalBasis",
        "Lattice",
    ):
        _require(
            condition=getattr(ad, name) is getattr(tr, name),
            message=f"treams_rs.advect.{name} is not the shared metadata type",
        )

    def physical_loss(radius: Any) -> Any:
        sphere = ad.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
        incoming = ad.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
        return sphere.cross_sections(incoming).scattering

    _require(
        condition=bool(grad(physical_loss)(0.2) > 0),
        message="Scattering cross-section gradient is not positive",
    )

    cell = np.diag([1.7, 1.8])
    modes = tr.SphericalBasis.default(2)
    positions = np.zeros((1, 3))

    def reflectance(radius: Any) -> Any:
        particle = ad.sphere(2, 2.1, anp.reshape(radius, (1,)), [3.0, 1.0])
        coupling = ad.lattice_expansion(
            positions,
            positions,
            [2.1, 2.1],
            [0, 0],
            cell,
            destination=modes,
            source=modes,
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

    _close(
        grad(reflectance)(np.array(0.2)),
        (reflectance(0.2 + _STEP) - reflectance(0.2 - _STEP)) / (2 * _STEP),
        rtol=1e-7,
    )
    print("Clean wheel: complete periodic Advect gradient passed")

    cb = tr.CylindricalBasis.default([0.2], 1)
    points = np.array([[0.8, 0.3, 0.1]])
    coefficients = np.ones(len(cb), complex)

    def magnetic_energy(impedance: Any) -> Any:
        value = ad.hfield(
            coefficients, points, cb.positions, [1.3, 1.3], impedance, basis=cb
        )
        return anp.sum(anp.real(value * anp.conj(value)))

    _close(
        grad(magnetic_energy)(np.array(0.8)),
        -2 * magnetic_energy(0.8) / 0.8,
        rtol=1e-12,
    )
    print("Clean wheel: magnetic impedance gradient passed")

    plane = tr.PlaneWavePorts.default([[0.2, 0.3], [1.5, -0.1]])
    vectors = np.column_stack(plane.kvecs(1.3))

    def plane_energy(scale: Any) -> Any:
        field = ad.plane_field(
            np.ones(len(plane)),
            points / scale,
            vectors * scale,
            polarizations=plane.pol,
        )
        return anp.sum(anp.real(field * anp.conj(field)))

    _close(grad(plane_energy)(np.array(1.0)), 0, atol=1e-12)
    print("Clean wheel: Advect coordinate/wavevector scale identity passed")

    cylinder = tr.CylindricalTMatrix.cylinder([0.2], 3, 1.3, [0.2], [4, 1])

    def cylindrical_radiation_energy(period: Any) -> Any:
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

    _close(
        grad(cylindrical_radiation_energy)(np.array(1.7)),
        -2 * cylindrical_radiation_energy(1.7) / 1.7,
        rtol=1e-12,
    )
    print("Clean wheel: native radiation period gradient passed")

    def compact_reflectance(thickness: Any) -> Any:
        value = ad.layer_stack(
            [[1.3, 1.3], [2.0, 2.0], [1.3, 1.3]],
            [1.0, 0.65, 1.0],
            [[0.2, 0.3], [0.0, 0.0]],
            anp.reshape(thickness, (1,)),
            alignment="zx",
        )
        reflected = value[:, 1, 0, :, 0]
        return anp.sum(anp.real(reflected * anp.conj(reflected)))

    _close(
        grad(compact_reflectance)(np.array(0.4)),
        (compact_reflectance(0.4 + _STEP) - compact_reflectance(0.4 - _STEP))
        / (2 * _STEP),
        rtol=1e-7,
        atol=1e-9,
    )
    print("Clean wheel: compact multilayer thickness adjoint passed")

    array_basis = tr.CylindricalBasis.default([0.2], 2)

    def periodic_to_cw_norm(period: Any) -> Any:
        value = ad.periodic_to_cw(
            array_basis.positions,
            modes.positions,
            [1.3, 1.3],
            array_basis.kz,
            period,
            destination=array_basis,
            source=modes,
        )
        return anp.sum(anp.real(value * anp.conj(value)))

    _close(
        grad(periodic_to_cw_norm)(np.array(1.7)),
        -2 * periodic_to_cw_norm(1.7) / 1.7,
        rtol=1e-12,
    )
    print("Clean wheel: spherical array to cylindrical period adjoint passed")

    cports = tr.PlaneWavePorts.default([[0.2, 0.1]], "zx")
    lower_interface = tr.SMatrix.interface(cports, 1.3, [1, 2.3])
    upper_interface = tr.SMatrix.interface(cports, 1.3, [2.3, 1])

    def internal_norm(scale: Any) -> Any:
        value = ad.smatrix_illuminate(
            lower_interface.array,
            upper_interface.array,
            scale * np.array([[1.0], [0.0]]),
            np.zeros((2, 1)),
        )[2:]
        return anp.sum(anp.real(value * anp.conj(value)))

    _close(grad(internal_norm)(np.array(1.0)), 2 * internal_norm(1.0), atol=1e-12)

    def uniform_band_norm(k0: Any, period: Any) -> Any:
        normal = anp.sqrt(k0**2 - 0.2**2 - 0.3**2)
        wavevectors = anp.stack([anp.stack([0.2, 0.3, normal])] * 2)
        smats = ad.propagation_matrix(wavevectors, anp.stack([0.0, 0.0, period]))
        k, _ = ad.bands(smats, period)
        return anp.sum(anp.real(k * anp.conj(k)))

    gk, gp = cast(
        "tuple[Any, Any]",
        grad(uniform_band_norm, argnums=(0, 1))(np.array(1.3), np.array(0.4)),
    )
    _close(gk, 8 * 1.3, atol=2e-11)
    _close(gp, 0, atol=2e-11)
    print("Clean wheel: internal fields and degenerate uniform-band adjoints passed")

    def singular_energy(matrix: Any) -> Any:
        return anp.sum(ad.svdvals(matrix) ** 2)

    matrix = np.array([[1.2, 0.1j], [0.3, 2.1], [0.4j, 0.5]])
    _close(grad(singular_energy)(matrix), 2 * matrix, atol=1e-12)

    def chirality(radius: Any) -> Any:
        particle = ad.sphere(
            2,
            1.3,
            anp.reshape(radius, (1,)),
            [3.0 + 0.2j, 1.0],
            [1.4 + 0.1j, 1.0],
            [0.12 + 0.02j, 0.0],
        )
        return ad.tmatrix_metric(particle, polarizations=modes.pol, kind="chi")

    _close(
        grad(chirality)(np.array(0.3)),
        (chirality(0.3 + _STEP) - chirality(0.3 - _STEP)) / (2 * _STEP),
        rtol=1e-7,
        atol=1e-10,
    )
    print("Clean wheel: native SVD and chirality gradient passed")

    theta, weights = tr.ebcm._quadrature(48)
    surface_radii = 0.3 * (1 + 0.23 * np.cos(theta) ** 2)
    surface_slopes = -0.138 * np.cos(theta) * np.sin(theta)

    def surface_scale_norm(scale: Any) -> Any:
        q = ad.ebcm_qmat(
            surface_radii * scale,
            surface_slopes * scale,
            np.array([[2.0, 2.1], [1.3, 1.3]]) / scale,
            [0.7, 1.0],
            theta=theta,
            weights=weights,
            destination=modes,
        )
        return anp.sum(anp.real(q * anp.conj(q)))

    _close(
        grad(surface_scale_norm)(np.array(1.0)), 4 * surface_scale_norm(1.0), rtol=1e-12
    )
    print("Clean wheel: native EBCM surface-scale adjoint passed")

    def density_scale(scale: Any) -> Any:
        value = ad.chirality_density(
            scale * np.array([1.3 + 0.1j]),
            scale * np.array([1.1 + 0.2j]),
            np.array([-0.2, 0.7]) / scale,
        )
        return anp.real(anp.sum(value))

    _close(grad(density_scale)(np.array(1.0)), 0, atol=1e-12)
    print("Clean wheel: native chirality-density scaling adjoint passed")

    def phase_scale(scale: Any) -> Any:
        phases = ad.plane_phases(
            np.array([[0.2, 0.1, -0.3]]) * scale,
            np.array([[0.2, 0.3, 1.3 + 0.1j]]) / scale,
        )
        return anp.real(anp.sum(phases))

    _close(grad(phase_scale)(np.array(1.0)), 0, atol=1e-12)
    print("Clean wheel: native phase adjoint passed")

    x = np.array([-1.0, -0.4, 0.7, 1.0])

    def angular_sum(value: Any) -> Any:
        return anp.real(anp.sum(ad.angular(value, degree=3, order=0)))

    _close(grad(angular_sum)(x), (15 * x**2 - 3) / 2)
    print("Clean wheel: angular Advect composition passed")

    cp = np.array([[0.4, -0.3, 0.2]])
    ck = np.array([1.3 + 0.05j, 1.5 + 0.07j])
    ca = np.full(len(cb), 0.2 + 0.3j)
    cv, _ = tr.diff.field(ca, cp, cb, ck)

    def axial_field(kz: Any) -> Any:
        return anp.real(anp.sum(ad.field(ca, cp, cb.positions, ck, basis=cb, kz=kz)))

    _close(
        grad(axial_field)(cb.kz),
        tr.diff.field(ca, cp, cb, ck)[1].pullback_axial(np.ones_like(cv))[4],
        atol=1e-13,
    )
    print("Clean wheel: cylindrical axial Advect composition passed")

    xyz = np.array([[0.3, 0.4, 0.5]])
    rvec = np.array([0.2 + 0.1j, -0.3, 0.4j])

    def vector_norm(value: Any) -> Any:
        return anp.sum(anp.abs(ad.vector_coordinates(rvec, value, kind="car2sph")) ** 2)

    _close(grad(vector_norm)(xyz), 0, atol=1e-14)
    print("Clean wheel: vector norm gradient passed")

    def local_wave_energy(scale: Any) -> Any:
        value = ad.vector_wave(
            0.3 * scale,
            0.4 * scale,
            1.2 * scale,
            0.5 / scale,
            0.7 / scale,
            0.9 / scale,
            function="vpw_A",
            polarization=1,
        )
        return anp.real(anp.sum(anp.conj(value) * value))

    _close(local_wave_energy(1.0), 1.0, atol=1e-13)
    _close(grad(local_wave_energy)(np.array(1.0)), 0.0, atol=1e-13)
    print("Clean wheel: complete local-wave scale adjoint passed")

    def polar_energy(scale: Any) -> Any:
        value = ad.cylindrical_translation(
            1.3, 0.4, 0.3 / scale, 0.2 * scale, order=2, singular=False
        )
        return anp.real(anp.conj(value) * value)

    _close(grad(polar_energy)(np.array(1.0)), 0, atol=1e-13)
    print("Clean wheel: common-axial-label scale adjoint passed")

    def gamma_identity(z: Any) -> Any:
        return anp.real(
            ad.incgamma(z, n=2.5) - 1.5 * ad.incgamma(z, n=1.5) - z**1.5 * anp.exp(-z)
        )

    _close(grad(gamma_identity)(np.asarray(1.2 + 0.1j)), 0, atol=2e-12)
    print("Clean wheel: incomplete-gamma recurrence and native argument adjoint passed")

    def interface_power(impedance: Any) -> Any:
        ks = anp.array([[1.3, 1.3], [2.0, 2.0]])
        zs = anp.stack([1.0, impedance])
        directions = anp.array([[0.2, 0.3]])
        blocks = ad.interface_coefficients(ks, zs, directions[0])
        power = ad.smatrix_tr(
            blocks,
            anp.array([[1.0], [0.2j]]),
            ks[::-1],
            zs[::-1],
            directions,
            modes=[(0, 0), (0, 1)],
        )
        return anp.real(anp.sum(power))

    _close(interface_power(0.7), 1, atol=2e-13)
    _close(grad(interface_power)(np.array(0.7)), 0, atol=2e-12)
    print("Clean wheel: native power adjoint passed")


def _check_io_workflows() -> None:
    h5py = cast("Any", importlib.import_module("h5py"))
    from treams_rs import io

    sphere = tr.TMatrix.sphere(1, 1.3, 0.2, [3, (1.3, 1.1, 0.08)])
    cluster = cast(
        "tr.TMatrix",
        tr.Cluster([sphere, sphere], positions=[[0, 0, 0], [0.7, 0.2, 0.1]]).solve(),
    )
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        io.save_hdf5(handle, cluster)
        loaded = io.load_hdf5(handle, lunit="um")
    if not isinstance(loaded, tr.TMatrix):
        message = f"HDF5 round trip returned {type(loaded).__name__}, not a T-matrix"
        raise TypeError(message)
    _close(loaded.array, cluster.array)
    _close(loaded.basis.positions, cluster.basis.positions * 1e-3)
    _close(loaded.k0, cluster.k0 * 1e3)
    _require(condition=loaded.material == cluster.material, message="Material changed")
    print("Clean wheel: optional HDF5 chirality, origins and units round trip passed")


def main(argv: list[str] | None = None) -> int:
    """Exercise the installed wheel under the selected dependency profiles."""
    profiles = _parse_args(argv)
    _check_distribution()
    _check_dependencies(profiles)
    _check_numpy_workflows()
    if "advect" in profiles:
        _check_advect_workflows()
    if "io" in profiles:
        _check_io_workflows()
    print(f"Clean wheel: smoke profiles {', '.join(sorted(profiles))} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
