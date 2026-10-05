"""Owned lattice pullbacks through broadcast geometry and complete Advect traces."""

import advect
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import advect as ad
from treams_rs import diff, lattice
from treams_rs.testing import check_pullback

from _support import assert_reusable_context, degree_order, selecting

pytestmark = pytest.mark.gradients


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
@pytest.mark.parametrize(
    "part,k0",
    [(part, 2.1 + 0.2j) for part in ("full", "real", "reciprocal", "direct")]
    + [("direct", k) for k in (2.1 - 0.2j, -2.1 + 0.2j, -2.1 - 0.2j)],
)
def test_broadcast_geometry_pullbacks_and_owned_inputs(spherical, dim, part, k0):
    coordinates = 3 if spherical else 2
    # Degrees (2,), k (2, 1), r (2, coordinates) and eta (1, 2) broadcast to
    # the (2, 2) output along different axes.
    k = np.array([[k0], [k0 + 0.2 - 0.1j]])
    q = np.linspace(0.1, 0.2, dim)
    a = np.diag(np.linspace(1.5, 1.7, dim))
    r = np.array([[0.19, 0.11, 0.07], [0.21, -0.09, 0.05]])[:, :coordinates]
    eta = np.array([[0.9 + 0.03j, 0.95 - 0.02j]])
    original = tuple(v.copy() for v in (k, q, a, r, eta))
    options = {"spherical": spherical, "part": part, "shell": 2}

    def record(*values):
        return diff.lattice_sum(dim, [2, 3], -1, *values, **options)

    value, context = record(k, q, a, r, eta)
    prefix = {
        "full": "lsum",
        "real": "realsum",
        "reciprocal": "recsum",
        "direct": "dsum",
    }[part]
    labels = ([2, 3], -1) if spherical else (-1,)
    expected = getattr(lattice, prefix + ("sw" if spherical else "cw"))(
        dim, *labels, k, q, a, r, 2 if part == "direct" else eta
    )
    assert_allclose(value, expected, rtol=3e-12, atol=3e-12)
    for argument in (k, q, a, r, eta):
        argument[:] = 7
    g = (np.arange(4).reshape(2, 2) * (0.1 + 0.04j) + 0.2)[:, ::-1]
    # The context owns its inputs: after they are overwritten it pulls back
    # what a fresh record of the original values, checked below, does. The
    # extra (1, 4) probe has the right size and would pass a size-only check.
    assert_reusable_context(
        context,
        g,
        record(*original)[1].pullback(g),
        wrong_shape=np.ones((1, 4), complex),
        rtol=1e-13,
        atol=1e-15,
    )
    rng = np.random.default_rng(8)
    check_pullback(
        record,
        *original,
        directions=tuple(
            rng.normal(size=v.shape) * 0.1 + (0.07j if np.iscomplexobj(v) else 0)
            for v in original
        ),
        cotangents=g,
        step=2e-6,
        rtol=4e-6,
        atol=3e-7,
    )


@pytest.mark.physics
@settings(max_examples=16)
@given(
    spherical=st.booleans(),
    dim=st.integers(1, 2),
    data=st.data(),
    pitch=st.floats(1.4, 1.8),
)
def test_advect_split_and_scale_invariants(spherical, dim, data, pitch):
    # Cylindrical sums ignore the degree; draw orders beyond it there.
    degree, order = (
        data.draw(degree_order(0, 3), label="degree, order")
        if spherical
        else (0, data.draw(st.integers(-3, 3), label="order"))
    )
    k = np.asarray(2.1 + 0.2j)
    q = np.full(dim, 0.13)
    a = np.eye(dim) * pitch
    r = np.array([0.19, 0.11, 0.07])[: 3 if spherical else 2]
    options = {"dim": dim, "degree": degree, "order": order, "spherical": spherical}

    def split(eta):
        # The parts at any split add up to the complete sum at a fixed split, so
        # their split cotangents cancel.
        args = (k, q, a, r, eta)
        return advect.numpy.real(
            ad.lattice_sum(k, q, a, r, 0.9, part="full", **options)
            - ad.lattice_sum(*args, part="real", **options)
            - ad.lattice_sum(*args, part="reciprocal", **options)
        )

    assert_allclose(split(1.2), 0, atol=3e-12)
    assert_allclose(advect.grad(split)(np.asarray(1.2)), 0, atol=3e-12)
    for part in ("full", "real", "reciprocal", "direct"):

        def scale_identity(scale, part=part):
            return advect.numpy.real(
                ad.lattice_sum(
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

        assert_allclose(advect.grad(scale_identity)(np.asarray(1.0)), 0, atol=3e-9)


@pytest.mark.interface
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


@pytest.mark.physics
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

        def record(eta, part=part):
            return diff.lattice_sum(
                dim, 0, 0, *args, eta, spherical=spherical, part=part
            )

        check_pullback(
            selecting(record, 4),
            np.asarray(0.9),
            directions=(np.asarray(1.0),),
            cotangents=np.asarray(0.2 + 0.3j),
            step=2e-6,
            rtol=3e-7,
            atol=2e-9,
        )


@pytest.mark.interface
def test_direct_half_cell_adjoint_reports_nondifferentiable_grouping():
    _, context = diff.lattice_sum(
        1, 2, 0, 2.1 + 0.2j, 0.13, 1.7, [0, 0, 0.85], part="direct", shell=1
    )
    with pytest.raises(ValueError, match="half-cell"):
        context.pullback(np.asarray(1, complex))


@pytest.mark.interface
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
