"""Broadcast Ewald components and direct shells, independently checked."""

import itertools

import numpy as np
import pytest
import scipy.special as sp
import treams.lattice as oracle
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import lattice

pytestmark = pytest.mark.filterwarnings(
    "ignore:`scipy.special.sph_harm` is deprecated:DeprecationWarning"
)

FAMILIES = (
    "sw1d",
    "sw1d_shift",
    "sw2d",
    "sw2d_shift",
    "sw3d",
    "cw1d",
    "cw1d_shift",
    "cw2d",
)


def operands(family, parameter):
    spherical = family.startswith("sw")
    dim = int(family[2])
    shifted = family.endswith("shift")
    labels = (2, -2) if spherical and (dim > 1 or shifted) else (2,)
    q = 0.13 if dim == 1 else np.linspace(0.1, 0.2, dim)
    a = 1.7 if dim == 1 else np.diag(np.linspace(1.5, 1.7, dim))
    r = (
        0.2
        if dim == 1 and not shifted
        else np.array([0.19, 0.11, 0.07])[
            : 3 if spherical and (dim == 3 or shifted) else 2
        ]
    )
    return (*labels, 2.1 + 0.2j, q, a, r, parameter)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "prefix,parameter",
    [
        ("lsum", 0),
        ("realsum", 0.9),
        ("recsum", 0.9),
        ("dsum", 0),
        ("dsum", 1),
        ("dsum", 4),
    ],
)
def test_components_and_shells_reference(family, prefix, parameter):
    args = operands(family, parameter)
    assert_allclose(
        getattr(lattice, prefix + family)(*args),
        getattr(oracle, prefix + family)(*args),
        rtol=3e-8,
        atol=3e-9,
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "prefix,parameter", [("lsum", 0.9), ("realsum", 0.9), ("recsum", 0.9), ("dsum", 2)]
)
def test_broadcast_lattice_geometry_and_strided_outputs(family, prefix, parameter):
    args = list(operands(family, parameter))
    args[0] = np.array([2, 3, 4])
    args[-5] = np.array([[2.1 + 0.2j], [2.3 + 0.15j]])
    # Independent geometry for each wavelength, broadcast against all orders.
    args[-4] = np.asarray(args[-4]) * np.array([1.0, 1.2]).reshape(
        (2, 1) + (1,) * np.ndim(args[-4])
    )
    args[-3] = np.asarray(args[-3]) * np.array([1.0, 1.1]).reshape(
        (2, 1) + (1,) * np.ndim(args[-3])
    )
    expected = getattr(oracle, prefix + family)(*args)
    storage = np.empty((3, 4), complex)
    out = storage[:, ::2].T[:, ::-1]
    result = getattr(lattice, prefix + family)(*args, out=out)
    assert result is out
    assert_allclose(result, expected, rtol=3e-8, atol=3e-9)
    # Output aliases the wavenumber input; NumPy must buffer this overlap.
    args[-5] = np.broadcast_to(args[-5], (2, 3)).copy()
    result = getattr(lattice, prefix + family)(*args, out=args[-5])
    assert_allclose(result, expected, rtol=3e-8, atol=3e-9)


@settings(max_examples=16, deadline=None)
@given(family=st.sampled_from(FAMILIES), eta=st.floats(0.8, 1.1), shift=st.booleans())
def test_split_identity_including_self_correction(family, eta, shift):
    args = list(operands(family, eta))
    if not shift:
        args[-2] = np.zeros_like(args[-2])
    full = getattr(lattice, "lsum" + family)(*args)
    real = getattr(lattice, "realsum" + family)(*args)
    reciprocal = getattr(lattice, "recsum" + family)(*args)
    assert_allclose(full, real + reciprocal, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
def test_direct_shell_against_cartesian_sum(spherical, dim):
    a = np.diag(np.linspace(1.5, 1.7, dim))
    q = np.linspace(0.1, 0.2, dim)
    # x-only spherical displacement catches upstream's incorrect 1d axis shortcut.
    r = np.array([0.2, 0.0, 0.0]) if spherical else np.array([0.2, 0.0])
    axes = [2] if spherical and dim == 1 else list(range(dim))
    shell, k, degree, order = 2, 2.1 + 0.2j, 2, -1
    expected = 0j
    for point in itertools.product(range(-shell, shell + 1), repeat=dim):
        if max(map(abs, point)) != shell:
            continue
        vector = np.asarray(point) @ a
        shift = -r.copy()
        shift[axes] -= vector
        radius = np.linalg.norm(shift)
        phi = np.arctan2(shift[1], shift[0])
        if spherical:
            theta = np.arctan2(np.hypot(*shift[:2]), shift[2])
            term = (
                sp.spherical_jn(degree, k * radius)
                + 1j * sp.spherical_yn(degree, k * radius)
            ) * sp.sph_harm_y(degree, order, theta, phi)
        else:
            term = sp.hankel1(order, k * radius) * np.exp(1j * order * phi)
        expected += term * np.exp(1j * np.dot(q, vector))
    actual = (
        lattice.dsumsw(dim, degree, order, k, q, a, r, shell)
        if spherical
        else lattice.dsumcw(dim, order, k, q, a, r, shell)
    )
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("shell", [0, 1, 3])
def test_half_cell_shell_pairing_and_reflection(spherical, shell):
    name = "dsumsw1d" if spherical else "dsumcw1d"
    for order in (0, 1, 2):
        args = (order, 2.1 + 0.2j, 0.13, 1.7, 0.85, shell)
        positive = getattr(lattice, name)(*args)
        assert_allclose(positive, getattr(oracle, name)(*args), rtol=3e-12, atol=3e-12)
        negative = getattr(lattice, name)(order, 2.1 + 0.2j, -0.13, 1.7, -0.85, shell)
        assert_allclose(negative, (-1) ** order * positive, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize(
    "name",
    ["lsumsw1d", "realsumsw1d", "recsumsw1d", "lsumcw1d", "realsumcw1d", "recsumcw1d"],
)
def test_axial_scalar_ufunc_mask_does_not_evaluate_invalid_values(name):
    k = np.array([2.1 + 0.2j, complex(np.nan), 2.3 + 0.1j])
    out = np.full(3, 9 + 3j)
    mask = np.array([True, False, True])
    args = (2, k, 0.13, 1.7, 0.2, 0.9)
    actual = getattr(lattice, name)(*args, out=out, where=mask)
    assert actual is out
    assert out[1] == 9 + 3j
    expected = getattr(oracle, name)(2, k[mask], 0.13, 1.7, 0.2, 0.9)
    assert_allclose(out[mask], expected, rtol=3e-9, atol=3e-10)


@settings(max_examples=50, deadline=None)
@given(
    order=st.integers(-12, 12),
    k=st.floats(1.1, 2.7),
    eta=st.floats(0.7, 1.2),
    q=st.one_of(st.just(0.0), st.floats(-1.0, -0.05), st.floats(0.05, 1.0)),
)
def test_axial_reciprocal_recurrence_reference_and_bloch_identity(order, k, eta, q):
    k, eta = k + 0.15j, eta + 0.03j
    value = lattice.recsumcw1d(order, k, q, 1.7, 0.2, eta)
    assert_allclose(
        value, oracle.recsumcw1d(order, k, q, 1.7, 0.2, eta), rtol=3e-9, atol=3e-10
    )
    shifted = lattice.recsumcw1d(order, k, q, 1.7, 1.9, eta)
    inverted = lattice.recsumcw1d(order, k, -q, 1.7, -0.2, eta)
    assert_allclose(shifted, np.exp(-1j * q * 1.7) * value, rtol=3e-11, atol=3e-11)
    assert_allclose(inverted, (-1) ** order * value, rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize("order", [0, 1, 2, 5, 12])
def test_axial_reciprocal_tiny_bloch_has_the_zero_bloch_limit(order):
    args = (1.7, 0.2, 1.0 + 0.03j)
    value = lattice.recsumcw1d(order, 2.0 + 0.15j, 1e-240, *args)
    limit = lattice.recsumcw1d(order, 2.0 + 0.15j, 0, *args)
    assert_allclose(value, limit, rtol=2e-13, atol=2e-13)
    with pytest.raises(ZeroDivisionError):
        oracle.recsumcw1d(order, 2.0 + 0.15j, 1e-240, *args)


@pytest.mark.parametrize("family", ["cw1d", "cw1d_shift", "cw2d"])
@settings(max_examples=25, deadline=None)
@given(order=st.integers(-8, 8), shell=st.integers(0, 5), scale=st.floats(0.7, 1.4))
def test_direct_cylindrical_scalar_array_and_strided_geometry(
    family, order, shell, scale
):
    args = list(operands(family, shell))
    args[0] = order
    args[1] *= scale
    actual = getattr(lattice, "dsum" + family)(*args)
    assert_allclose(
        actual,
        getattr(oracle, "dsum" + family)(*args),
        rtol=3e-11,
        atol=3e-11 * (1 + abs(actual)),
    )
    # Supplying out exercises the native ufunc, including the integer-label loop.
    out = np.empty((), complex)
    assert getattr(lattice, "dsum" + family)(*args, out=out) is out
    assert_allclose(out, actual, rtol=2e-13, atol=2e-13)
    if family != "cw1d":
        r = args[-2]
        args[-2] = r[::-1].copy()[::-1]
        assert_allclose(
            getattr(lattice, "dsum" + family)(*args), actual, rtol=0, atol=0
        )
