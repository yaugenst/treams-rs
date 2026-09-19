"""Owned lattice pullbacks through broadcast geometry and complete Advect traces."""

import advect as ad
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import advect, diff, lattice


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
@pytest.mark.parametrize("part", ["full", "real", "reciprocal", "direct"])
def test_broadcast_geometry_pullbacks_and_owned_inputs(spherical, dim, part):
    coordinates = 3 if spherical else 2
    k = np.array([[2.1 + 0.2j], [2.3 + 0.1j], [2.25 + 0.2j], [2.4 + 0.15j]])
    q = np.linspace(0.1, 0.2, dim)
    a = np.diag(np.linspace(1.5, 1.7, dim))
    r = np.array([[0.19, 0.11, 0.07], [0.21, -0.09, 0.05], [0.13, 0.15, -0.11]])[
        :, :coordinates
    ]
    eta = np.array([[0.9 + 0.03j, 0.95 - 0.02j, 1.0 + 0.01j]])
    original = tuple(v.copy() for v in (k, q, a, r, eta))
    options = {"spherical": spherical, "part": part, "shell": 2}
    value, context = diff.lattice_sum(dim, [2, 3, 4], -1, k, q, a, r, eta, **options)
    prefix = {
        "full": "lsum",
        "real": "realsum",
        "reciprocal": "recsum",
        "direct": "dsum",
    }[part]
    labels = ([2, 3, 4], -1) if spherical else (-1,)
    expected = getattr(lattice, prefix + ("sw" if spherical else "cw"))(
        dim, *labels, k, q, a, r, 2 if part == "direct" else eta
    )
    assert_allclose(value, expected, rtol=3e-12, atol=3e-12)
    for argument in (k, q, a, r, eta):
        argument[:] = 7
    g = (np.arange(12).reshape(4, 3) * (0.1 + 0.04j) + 0.2)[:, ::-1]
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.ones((3, 2), complex))
    gradients = context.pullback(g)
    rng = np.random.default_rng(8)
    for j, gradient in enumerate(gradients):
        assert gradient.shape == original[j].shape
        direction = rng.normal(size=original[j].shape) * 0.1
        if np.iscomplexobj(original[j]):
            direction = direction + 0.07j
        plus, minus = list(original), list(original)
        h = 2e-6
        plus[j] = original[j] + h * direction
        minus[j] = original[j] - h * direction
        fp = diff.lattice_sum(dim, [2, 3, 4], -1, *plus, **options)[0]
        fm = diff.lattice_sum(dim, [2, 3, 4], -1, *minus, **options)[0]
        assert_allclose(
            np.vdot(gradient, direction).real,
            np.vdot(g, (fp - fm) / (2 * h)).real,
            rtol=4e-6,
            atol=3e-7,
        )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@settings(max_examples=16, deadline=None)
@given(
    spherical=st.booleans(),
    dim=st.integers(1, 2),
    order=st.integers(0, 3),
    pitch=st.floats(1.4, 1.8),
)
def test_advect_split_and_scale_invariants(spherical, dim, order, pitch):
    k = np.asarray(2.1 + 0.2j)
    q = np.full(dim, 0.13)
    a = np.eye(dim) * pitch
    r = np.array([0.19, 0.11, 0.07])[: 3 if spherical else 2]
    options = {"dim": dim, "degree": order, "order": order, "spherical": spherical}

    def split(eta):
        args = (k, q, a, r, eta)
        return ad.numpy.real(
            advect.lattice_sum(*args, part="full", **options)
            - advect.lattice_sum(*args, part="real", **options)
            - advect.lattice_sum(*args, part="reciprocal", **options)
        )

    assert_allclose(split(0.9), 0, atol=3e-12)
    assert_allclose(ad.grad(split)(np.asarray(0.9)), 0, atol=3e-12)
    for part in ("full", "real", "reciprocal", "direct"):

        def scale_identity(scale, part=part):
            return ad.numpy.real(
                advect.lattice_sum(
                    k / scale,
                    q / scale,
                    a * scale,
                    r * scale,
                    0.9,
                    part=part,
                    shell=2,
                    **options,
                )
            )

        assert_allclose(ad.grad(scale_identity)(np.asarray(1.0)), 0, atol=3e-9)


def test_empty_scalar_geometry_and_split_contract():
    value, context = diff.lattice_sum(
        1, 2, 1, np.empty((0,), complex), 0.1, 1.7, [0.2, 0.1, 0.1]
    )
    assert value.shape == (0,)
    gradients = context.pullback(np.empty((0,), complex))
    for actual, shape in zip(gradients, [(0,), (), (), (3,), ()], strict=True):
        assert actual.shape == shape
        assert_allclose(actual, 0)
    with pytest.raises(ValueError, match="explicit nonzero"):
        diff.lattice_sum(1, 2, 1, 2.1, 0.1, 1.7, [0.2, 0.1, 0.1], part="real")


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
def test_self_correction_split_gradient(spherical, dim):
    args = (
        2.1 + 0.2j,
        np.full(dim, 0.13),
        np.eye(dim) * 1.7,
        np.zeros(3 if spherical else 2),
    )
    for part in ("real", "reciprocal"):
        _, context = diff.lattice_sum(
            dim, 0, 0, *args, 0.9, spherical=spherical, part=part
        )
        derivative = context.pullback(np.asarray(0.2 + 0.3j))[-1]
        h = 2e-6
        fp = diff.lattice_sum(
            dim, 0, 0, *args, 0.9 + h, spherical=spherical, part=part
        )[0]
        fm = diff.lattice_sum(
            dim, 0, 0, *args, 0.9 - h, spherical=spherical, part=part
        )[0]
        assert_allclose(
            derivative.real,
            np.vdot(0.2 + 0.3j, (fp - fm) / (2 * h)).real,
            rtol=3e-7,
            atol=2e-9,
        )


def test_direct_half_cell_adjoint_reports_nondifferentiable_grouping():
    _, context = diff.lattice_sum(
        1, 2, 0, 2.1 + 0.2j, 0.13, 1.7, [0, 0, 0.85], part="direct", shell=1
    )
    with pytest.raises(ValueError, match="half-cell"):
        context.pullback(np.asarray(1, complex))


@pytest.mark.parametrize("label", ["degree", "order", "shell"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.5])
def test_scalar_lattice_labels_require_finite_integers(label, value):
    labels = {"degree": 1, "order": 0, "shell": 2}
    labels[label] = value
    with pytest.raises(ValueError, match="finite integers"):
        diff.lattice_sum(
            1,
            labels["degree"],
            labels["order"],
            1.2 + 0.1j,
            0.1,
            1.7,
            [0.1, 0.2, 0.3],
            part="direct",
            shell=labels["shell"],
        )
