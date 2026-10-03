"""Riemann-Silberstein fields (operators.gfield and ffield) of every basis family:
upstream conventions, reconstruction of E and H, and Advect derivatives."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs.testing import check_gradient

from _support import to_oracle


def _basis(family):
    if family in ("sw", "cw"):
        positions = [[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4]]
        full = (
            tr.SphericalBasis.default(2, 2, positions)
            if family == "sw"
            else tr.CylindricalBasis.default([0.2, -0.3], 2, 2, positions)
        )
        basis = type(full)(full.modes[::3], positions)
    elif family == "unit":
        basis = tr.PlaneWaveBasis.default([[0.2, 0.3, 1], [1.1, -0.2, -0.3]])
    else:
        basis = tr.PlaneWavePorts.default([[0.2, 0.3], [3.2, -0.1]], family)
    return basis, to_oracle(basis)


@pytest.mark.reference
@pytest.mark.parametrize("family", ["sw", "cw", "xy", "yz", "zx", "unit"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("pol", [-1, 1])
@pytest.mark.parametrize("kind", ["gfield", "ffield"])
def test_riemann_field_upstream_conventions(family, poltype, pol, kind):
    basis, oracle = _basis(family)
    points = np.array([[[0.7, 0.8, 0.2], [-0.4, 0.7, 0.1]]])
    args = dict(
        k0=1.3,
        material=(2.3 + 0.1j, 1.2 + 0.05j, 0.04 if poltype == "helicity" else 0),
        poltype=poltype,
        modetype="singular" if family in ("sw", "cw") else "down",
    )
    actual = getattr(tr.operators, kind)(pol, points, basis=basis, **args)
    expected = getattr(treams, kind)(pol, points, basis=oracle, **args)
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    if pol == -1:
        assert_allclose(
            getattr(tr.operators, kind)(0, points, basis=basis, **args), actual
        )


@pytest.mark.physics
@pytest.mark.parametrize("family", ["sw", "cw", "xy"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(k0=st.floats(0.8, 1.7), x=st.floats(0.7, 1.3))
def test_riemann_reconstructs_electric_and_magnetic(family, poltype, k0, x):
    basis, _ = _basis(family)
    args = dict(basis=basis, k0=k0, material=(2.3, 1.2), poltype=poltype)
    point = [x, 0.6, 0.2]
    minus, plus = [tr.operators.gfield(p, point, **args) for p in (0, 1)]
    factor = (np.sqrt(2) if family == "sw" else 1) * (2 if poltype == "parity" else 1)
    assert_allclose(
        (plus + minus) / factor, tr.operators.efield(point, **args), atol=1e-12
    )
    assert_allclose(
        (plus - minus) / factor,
        1j
        * tr.Material(args["material"]).impedance
        * tr.operators.hfield(point, **args),
        atol=1e-12,
    )
    assert_allclose(tr.operators.ffield(1, point, **args), plus, atol=1e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("family", ["sw", "cw"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_weighted_riemann_all_advect_inputs(family, poltype):
    basis, _ = _basis(family)
    rng = np.random.default_rng(83)
    coefficients = rng.normal(size=len(basis)) + 0.1j
    points = np.array([[0.7, 0.8, 0.2], [-0.4, 0.7, 0.1]])
    medium = tr.Material(
        (2.3 + 0.1j, 1.2 + 0.05j, 0.04 if poltype == "helicity" else 0)
    )
    ks = medium.ks(1.3)
    options = {"basis": basis, "poltype": poltype, "singular": True}
    for kind in ("gfield", "ffield"):
        for pol in (0, 1):
            assert_allclose(
                getattr(ad, kind)(
                    pol, coefficients, points, basis.positions, ks, **options
                ),
                getattr(tr.operators, kind)(
                    pol,
                    points,
                    basis=basis,
                    k0=1.3,
                    material=medium,
                    poltype=poltype,
                    modetype="singular",
                )
                @ coefficients,
                atol=1e-11,
                rtol=1e-11,
            )

    def objective(*values):
        field = ad.gfield(0, *values, **options) + 0.3j * ad.ffield(
            1, *values, **options
        )
        return anp.sum(anp.real(field * anp.conj(field)))

    values = [coefficients, points, basis.positions, ks]
    directions = [rng.normal(size=v.shape) * 0.1 for v in values]
    directions[0] = directions[0] + 0.03j
    directions[3] = (
        np.full_like(ks, 0.04 + 0.02j) if poltype == "parity" else directions[3] + 0.02j
    )
    gradients = advect.grad(objective, argnums=(0, 1, 2, 3))(*values)
    check_gradient(
        objective,
        lambda *_: gradients,
        *values,
        directions=tuple(directions),
        step=1e-5,
        rtol=2e-7,
        atol=2e-7,
    )
    assert_allclose(
        np.vdot(gradients[0], coefficients).real, 2 * objective(*values), atol=1e-10
    )
    assert_allclose(gradients[1].sum(axis=0) + gradients[2].sum(axis=0), 0, atol=1e-10)


@pytest.mark.interface
def test_riemann_invalid_polarization():
    basis = tr.SphericalBasis.default(1)
    with pytest.raises(ValueError, match="polarization"):
        tr.operators.gfield(2, [0.2, 0.3, 0.4], basis=basis, k0=1)
