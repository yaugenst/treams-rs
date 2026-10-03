"""Shared helpers for the Python test suite.

pytest's default ``prepend`` import mode puts ``tests/`` on ``sys.path``, so test
modules import this file as ``_support``. Keep only contracts, input builders and
oracle adapters here that several test files share; physics stays in the tests.
"""

import contextlib
import functools
from pathlib import Path

import numpy as np
import pytest
import treams
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs._upstream import UPSTREAM_NAMES

#: The repository root, resolved from this file so that it does not depend on
#: the directory of the test module that imports it.
ROOT = Path(__file__).resolve().parents[1]

# One-use native contexts ---------------------------------------------------------


def assert_tree_allclose(actual, expected, *, rtol=1e-7, atol=0.0):
    """Apply ``assert_allclose`` leafwise to matching tuples or lists of arrays."""
    if isinstance(expected, tuple | list):
        assert isinstance(actual, tuple | list), type(actual)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_tree_allclose(a, e, rtol=rtol, atol=atol)
    else:
        assert_allclose(actual, expected, rtol=rtol, atol=atol)


def _wrong_shapes(cotangent):
    """Wrong shapes of one cotangent's dimension (``(2,)`` for 0-d cotangents)."""
    shape = np.shape(cotangent)
    if not shape:
        yield (2,)
        return
    yield (shape[0] + 1, *shape[1:])
    for axis, size in enumerate(shape):
        if size != 1:
            yield (*shape[:axis], 1, *shape[axis + 1 :])
    if shape[::-1] != shape:
        yield shape[::-1]


def _with_nan(cotangent):
    """Copy ``cotangent`` with a NaN in its last entry, or ``None`` when empty."""
    cotangent = np.array(cotangent)
    if cotangent.size == 0:
        return None
    cotangent.flat[-1] = np.nan
    return cotangent


def assert_one_use_context(
    context,
    cotangent,
    expected=None,
    *,
    wrong_shape=None,
    shape_match="shape",
    finite_match="finite",
    rtol=1e-7,
    atol=0.0,
):
    """Check the one-use contract of a native pullback context; return its gradients.

    ``cotangent`` is one array, or a tuple with one array per output. Corrupting
    one cotangent at a time, wrong shapes raise ``ValueError`` matching
    ``shape_match`` and a NaN entry one matching ``finite_match``; neither
    consumes the residual, so the valid cotangents then succeed (and match
    ``expected``, a tree of arrays, when given), and a second use raises
    ``ValueError`` mentioning "consumed". The wrong shapes keep dtype and
    dimension: the first axis grown by one, each non-unit axis set to 1 (NumPy
    would broadcast these) and the axes reversed (same size, which a size-only
    check accepts). ``wrong_shape`` adds a site-specific probe with the structure
    of ``cotangent``, for example of another dimension. Overwrite the recorded
    arrays before calling this to check that the context owns its inputs.
    """
    multiple = isinstance(cotangent, tuple)
    cotangents = tuple(np.asarray(c) for c in (cotangent if multiple else (cotangent,)))

    def replacing(index, replacement):
        return (*cotangents[:index], replacement, *cotangents[index + 1 :])

    probes = [
        replacing(index, np.zeros(shape, c.dtype))
        for index, c in enumerate(cotangents)
        for shape in _wrong_shapes(c)
    ]
    if wrong_shape is not None:
        probes.append(wrong_shape if multiple else (wrong_shape,))
    for probe in probes:
        with pytest.raises(ValueError, match=shape_match):
            context.pullback(*probe)
    for index, c in enumerate(cotangents):
        nan = _with_nan(c)
        if nan is not None:
            with pytest.raises(ValueError, match=finite_match):
                context.pullback(*replacing(index, nan))
    gradients = context.pullback(*cotangents)
    if expected is not None:
        assert_tree_allclose(gradients, expected, rtol=rtol, atol=atol)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(*cotangents)
    return gradients


def selecting(record, *indices):
    """Adapt a native ``record`` to pull back only the gradients at ``indices``.

    ``check_pullback`` needs one gradient per dynamic parameter: fix the other
    inputs in the ``record`` closure. Selecting every index turns a list of
    gradients into the tuple ``check_pullback`` requires.
    """

    def adapted(*args):
        value, context = record(*args)

        def pullback(*cotangents):
            gradients = context.pullback(*cotangents)
            return tuple(gradients[i] for i in indices)

        return value, pullback

    return adapted


def varying(record, values, *indices):
    """Adapt ``record(*values)`` to take and pull back only the inputs at ``indices``;
    the others stay fixed at ``values``."""

    def partial(*varied):
        arguments = list(values)
        for index, value in zip(indices, varied, strict=True):
            arguments[index] = value
        return record(*arguments)

    return selecting(partial, *indices)


# Memory layouts ------------------------------------------------------------------

#: Every layout ``arrange`` produces; all hold the same logical values.
LAYOUTS = (
    "C",
    "F",
    "block-F",
    "strided",
    "reversed",
    "flipped",
    "F-flipped",
    "C-flip-first",
    "C-flip-last",
    "F-flip-first",
    "F-flip-last",
)

# Storage order and flipped axis (None: every axis) of the negative-stride layouts.
_FLIPS = {
    "flipped": ("C", None),
    "F-flipped": ("F", None),
    "C-flip-first": ("C", 0),
    "C-flip-last": ("C", -1),
    "F-flip-first": ("F", 0),
    "F-flip-last": ("F", -1),
}


def arrange(a, layout):
    """Return a new array equal to ``a`` (at least 1-d) in ``layout``.

    ``"C"`` and ``"F"`` are contiguous copies; ``"block-F"`` is column-major in
    the last two axes and C order across the others; ``"strided"`` and
    ``"reversed"`` read zero-padded storage with steps 2 and -2 along the last
    axis. The flipped layouts read C- or Fortran-contiguous storage backwards
    along every axis, the first or the last axis only, so their strides may mix
    signs; NumPy flags none of them as contiguous.
    """
    a = np.asarray(a)
    if layout == "C":
        return np.array(a, order="C")
    if layout == "F":
        return np.array(a, order="F")
    if layout == "block-F":
        if a.ndim < 2:
            return a.copy()
        return np.array(a.swapaxes(-1, -2), order="C").swapaxes(-1, -2)
    if layout in ("strided", "reversed"):
        storage = np.zeros((*a.shape[:-1], 2 * a.shape[-1]), a.dtype)
        view = storage[..., ::2] if layout == "strided" else storage[..., ::-2]
        view[...] = a
        return view
    if layout in _FLIPS:
        order, axis = _FLIPS[layout]
        return np.flip(np.array(np.flip(a, axis), order=order), axis)
    raise ValueError(f"unknown layout {layout!r}")


def layouts():
    """Hypothesis strategy over the names in ``LAYOUTS``; shrinks towards "C"."""
    return st.sampled_from(LAYOUTS)


@st.composite
def strided_copies(draw, a):
    """Hypothesis strategy for fresh copies of ``a`` (at least 1-d) in any layout:
    a drawn axis order of zero-padded storage read with steps of 1, -1, 2 or -2
    per axis; shrinks towards a C-contiguous copy."""
    a = np.asarray(a)
    order = draw(st.permutations(range(a.ndim)))
    steps = [draw(st.sampled_from((1, -1, 2, -2))) for _ in a.shape]
    storage = np.zeros([abs(steps[axis]) * a.shape[axis] for axis in order], a.dtype)
    logical = storage.transpose(np.argsort(order))
    view = logical[tuple(slice(None, None, step) for step in steps)]
    view[...] = a
    return view


def sum_to(a, shape):
    """Sum ``a`` over the axes that broadcasting added to ``shape``: the adjoint of
    ``np.broadcast_to(x, a.shape)`` for ``x`` of ``shape``."""
    a = np.sum(a, axis=tuple(range(np.ndim(a) - len(shape))))
    return np.sum(
        a, axis=tuple(i for i, n in enumerate(shape) if n == 1), keepdims=True
    )


# Random complex inputs -----------------------------------------------------------


def complex_normal(rng, shape):
    """Complex standard-normal samples: all real parts, then all imaginary parts."""
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def finite_complex(*, min_magnitude=0.0, max_magnitude=None):
    """Hypothesis strategy for finite complex scalars in a magnitude range (bounds
    hold up to a relative rounding of about epsilon)."""
    return st.complex_numbers(
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        allow_nan=False,
        allow_infinity=False,
    )


def complex_arrays(shape, *, max_magnitude=1.0, fill=None):
    """Hypothesis strategy for complex128 arrays of ``finite_complex`` entries.

    ``fill`` goes to ``hnp.arrays``: by default most entries share one background
    value; ``st.nothing()`` draws every entry independently.
    """
    return hnp.arrays(
        np.complex128,
        shape,
        elements=finite_complex(max_magnitude=max_magnitude),
        fill=fill,
    )


# Angular-momentum labels ---------------------------------------------------------


@st.composite
def degree_order(draw, low=0, high=10, *, margin=0):
    """Hypothesis strategy for ``(degree, order)`` with ``|order| <= degree - margin``;
    the order shrinks towards zero."""
    degree = draw(st.integers(low, high))
    limit = degree - margin
    return degree, draw(st.integers(-limit, limit))


# Energy conservation -------------------------------------------------------------


def assert_unitary_ports(array, index=slice(None), *, atol):
    """Assert that an S-matrix (2, 2, n, n) restricted to the modes at ``index`` is
    unitary: ``M^H M = I`` for ``M = array.transpose(0, 2, 1, 3).reshape(2n, 2n)``.

    This is energy conservation for every input, orthogonality included, when
    the outer media are identical and lossless, the selected modes propagate,
    and coupled modes carry equal flux per unit amplitude.
    """
    block = np.asarray(array)[:, :, index][:, :, :, index]
    n = block.shape[-1]
    matrix = block.transpose(0, 2, 1, 3).reshape(2 * n, 2 * n)
    assert_allclose(matrix.conj().T @ matrix, np.eye(2 * n), rtol=0, atol=atol)


# Reciprocity ---------------------------------------------------------------------


def reciprocal(matrix, modes, *, cylindrical):
    """Return ``P matrix.T P`` for the signed mode permutation ``P`` of reciprocity.

    ``P`` maps ``(origin, l, m, pol)`` to ``(origin, l, -m, pol)`` (spherical) or
    ``(origin, kz, m, pol)`` to ``(origin, -kz, -m, pol)`` (cylindrical) with sign
    ``(-1)**m``. ``modes`` label both axes of ``matrix`` and must be closed under
    ``P``. A reciprocal operator equals its ``reciprocal`` once its columns are
    weighted by the wavenumber powers of their polarizations.
    """
    modes = [tuple(mode) for mode in modes]
    index = {mode: i for i, mode in enumerate(modes)}
    partner = [
        index[(origin, -label if cylindrical else label, -m, pol)]
        for origin, label, m, pol in modes
    ]
    sign = np.array([(-1.0) ** mode[2] for mode in modes])
    return sign[:, None] * np.asarray(matrix).T[np.ix_(partner, partner)] * sign


# Upstream treams oracle ----------------------------------------------------------


#: treams class of each treams-rs class that treams spells differently.
_TREAMS_CLASS = {
    renamed.target: name for name, renamed in UPSTREAM_NAMES.items() if renamed.target
}


def to_oracle(basis, modes=None):
    """The upstream basis with the geometry of a native ``basis`` and optionally
    other ``modes``."""
    cls = getattr(treams, _TREAMS_CLASS[type(basis).__name__])
    modes = basis.modes if modes is None else modes
    if isinstance(basis, tr.SphericalBasis | tr.CylindricalBasis):
        return cls(modes, basis.positions)
    if isinstance(basis, tr.PlaneWavePorts):
        return cls(modes, alignment=basis.alignment)
    return cls(modes)


def oracle_smatrix_array(smatrices):
    """Stack the blocks of an upstream ``SMatrices`` into the native (2, 2, n, n)."""
    return np.array([[np.asarray(smatrices[i, j]) for j in range(2)] for i in range(2)])


# Installed API catalog -------------------------------------------------------------


@functools.cache
def catalog():
    """The installed ``support_catalog()``, built once per process; do not mutate."""
    return tr.support_catalog()


# JAX precision ---------------------------------------------------------------------


@contextlib.contextmanager
def jax_x64(enabled=True):
    """Set JAX's global ``jax_enable_x64`` flag, restore it on exit; yield ``jax``.

    The scoped ``jax.enable_x64`` does not reach the threads that run native
    callbacks. ``enabled=None`` only restores the current setting.
    """
    jax = pytest.importorskip("jax")
    previous = jax.config.read("jax_enable_x64")
    if enabled is not None:
        jax.config.update("jax_enable_x64", enabled)
    try:
        yield jax
    finally:
        jax.config.update("jax_enable_x64", previous)
