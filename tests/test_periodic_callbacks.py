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

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*scipy.special.sph_harm.*:DeprecationWarning"
)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("split", [False, True])
def test_custom_spherical_lattice_callback_reference_and_single_broadcast(
    dim, poltype, split
):
    destination = tr.SphericalBasis.default(
        1, 2, positions=[[0.1, 0.2, 0.3], [-0.3, 0.1, 0.2]]
    )
    source = tr.SphericalBasis.default(2, positions=[[0.03, -0.05, 0.02]])
    ks = [1.3 + 0.1j, 1.5 + 0.1j] if poltype == "helicity" else [1.3 + 0.1j] * 2
    a = 1.7 if dim == 1 else np.eye(dim) * 1.7
    q = 0.13 if dim == 1 else np.linspace(0.1, 0.2, dim)
    calls = []

    def callback(*args):
        calls.append(args)
        return (
            tr.lattice.realsumsw(*args) + tr.lattice.recsumsw(*args)
            if split
            else tr.lattice.lsumsw(*args)
        )

    arguments = (
        ks,
        q,
        a,
        destination.positions,
        tuple(destination[()]),
        tuple(source[()]),
        source.positions,
    )
    result = tr.sw.translate_periodic(
        *arguments, poltype=poltype, eta=0.9, func=callback
    )
    expected = treams.sw.translate_periodic(*arguments, poltype=poltype, eta=0.9)
    assert_allclose(result, expected, atol=3e-11, rtol=3e-11)
    assert_allclose(
        result,
        tr.sw.translate_periodic(*arguments, poltype=poltype, eta=0.9),
        atol=3e-11,
        rtol=3e-11,
    )
    assert len(calls) == 1
    assert calls[0][-2].shape == (2, 1, 1, 1, 3)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@settings(max_examples=20)
@given(
    scale=st.complex_numbers(
        min_magnitude=0.2, max_magnitude=2, allow_nan=False, allow_infinity=False
    )
)
def test_table_linear_pullback_owned_inputs_and_broadcast_values(poltype, scale):
    destination = tr.SphericalBasis(
        [(0, 1, -1, 0), (1, 2, 1, 1)], positions=[[0, 0, 0], [0.2, 0.3, 0.4]]
    )
    source = tr.SphericalBasis([(0, 2, -2, 0), (0, 1, 0, 1)])
    channels = 2 if poltype == "helicity" else 1
    rng = np.random.default_rng(110)
    table = (
        rng.normal(size=(2, 1, channels, 25))
        + 1j * rng.normal(size=(2, 1, channels, 25))
    ) * scale
    value, context = tr.diff.periodic_from_table(
        table, destination, source, poltype=poltype
    )
    g = np.array([[0.2 + 0.3j, 0.4], [0.1j, -0.7 + 0.1j]])[:, ::-1]
    saved = table.copy()
    table[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.zeros((3, 2), complex))
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full((2, 2), np.nan, complex))
    gradient = context.pullback(g)
    assert gradient.shape == saved.shape
    assert_allclose(np.vdot(g, value), np.vdot(gradient, saved), atol=2e-12, rtol=2e-12)
    assert_allclose(
        tr.diff.periodic_from_table(
            saved / scale, destination, source, poltype=poltype
        )[0]
        * scale,
        value,
        atol=2e-12,
        rtol=2e-12,
    )
    # A noncontiguous table has the same logical contraction and pullback.
    strided = saved[..., ::-1].copy()[..., ::-1]
    assert_allclose(
        tr.diff.periodic_from_table(strided, destination, source, poltype=poltype)[0],
        value,
        atol=0,
    )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)

    def objective(x):
        result = ad.periodic_from_table(
            x, destination=destination, source=source, poltype=poltype
        )
        return anp.real(anp.sum(anp.conj(g) * result))

    assert_allclose(advect.grad(objective)(saved), gradient, atol=0)


def test_table_pullback_composes_with_lattice_geometry_adjoint():
    destination = tr.SphericalBasis(
        [(0, 1, -1, 0), (1, 1, 0, 1)], positions=[[0, 0, 0], [0.2, 0.3, 0.4]]
    )
    source = tr.SphericalBasis.default(1)
    modes = np.array(
        [(degree, order) for degree in range(3) for order in range(-degree, degree + 1)]
    )
    shifts = np.array([[0.1, 0.2, 0.3], [-0.3, 0.1, 0.2]])[:, None, None, None, :]

    def objective(k, q, a, r):
        table = ad.lattice_sum(
            k, q, a, r, 0.9, dim=1, degree=modes[:, 0], order=modes[:, 1]
        )
        value = ad.periodic_from_table(table, destination=destination, source=source)
        return anp.sum(anp.real(value * anp.conj(value)))

    values = (np.array([[1.3 + 0.1j]]), np.array(0.13), np.array(1.7), shifts)
    gradients = advect.grad(objective, argnums=(0, 1, 2, 3))(*values)
    base = objective(*values)
    scaled = (values[0] / 1.3, values[1] / 1.3, values[2] * 1.3, values[3] * 1.3)
    assert_allclose(objective(*scaled), base, rtol=1e-12)
    ward = (
        -np.vdot(gradients[0], values[0]).real
        - np.vdot(gradients[1], values[1]).real
        + np.vdot(gradients[2], values[2]).real
        + np.vdot(gradients[3], values[3]).real
    )
    assert_allclose(ward, 0, atol=1e-9 * (1 + base))
    directions = (
        np.array([[0.2 + 0.1j]]),
        np.array(-0.1),
        np.array(0.3),
        np.ones_like(shifts) * 0.02,
    )
    h = 1e-5
    numerical = (
        objective(*(v + h * d for v, d in zip(values, directions, strict=True)))
        - objective(*(v - h * d for v, d in zip(values, directions, strict=True)))
    ) / (2 * h)
    analytic = sum(
        np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True)
    )
    assert_allclose(analytic, numerical, rtol=2e-8, atol=2e-8)
