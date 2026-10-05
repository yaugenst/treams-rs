"""Native object and context bindings at the NumPy boundary.

The stub declares every native signature. Every context of the object bindings
has a CASES entry and satisfies one contract: records are deterministic,
derivative contexts are reusable and rejected directions leave them unchanged,
residuals own their inputs and share no memory with returned values, and records
and pullbacks read every NumPy memory layout.
"""

import ast
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import treams_rs as tr

# The native bindings are the layer under test.
from treams_rs import _native

from _support import (
    LAYOUTS,
    arrange,
    assert_reusable_context,
    assert_tree_allclose,
    complex_normal,
)

# Stub signatures -----------------------------------------------------------------

_STUB = ast.parse(Path(_native.__file__).with_name("_native.pyi").read_text())
_P = inspect.Parameter


def _declared(function, *, bound):
    """(name, kind, default) of a stub function; ``...`` is any default."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults = [None] * (len(positional) - len(arguments.defaults))
    parameters = [
        (
            argument.arg,
            _P.POSITIONAL_ONLY
            if argument in arguments.posonlyargs
            else _P.POSITIONAL_OR_KEYWORD,
            default,
        )
        for argument, default in zip(
            positional, [*defaults, *arguments.defaults], strict=True
        )
    ]
    if arguments.vararg:
        parameters.append((arguments.vararg.arg, _P.VAR_POSITIONAL, None))
    parameters += [
        (argument.arg, _P.KEYWORD_ONLY, default)
        for argument, default in zip(
            arguments.kwonlyargs, arguments.kw_defaults, strict=True
        )
    ]
    if arguments.kwarg:
        parameters.append((arguments.kwarg.arg, _P.VAR_KEYWORD, None))
    return [
        (name, kind, None if default is None else ast.literal_eval(default))
        for name, kind, default in parameters[bound:]
    ]


def _runtime(value, *, bound):
    """(name, kind, default) of a native callable, as ``_declared`` reports it."""
    try:
        signature = inspect.signature(value)
    except ValueError as error:  # For example a parameter named like a keyword.
        raise AssertionError(f"{value!r}: {error}") from error
    parameters = list(signature.parameters.values())[bound:]
    return [
        (p.name, p.kind, None if p.default is _P.empty else p.default)
        for p in parameters
    ]


def _assert_same_parameters(declared, runtime, name):
    assert [(n, k) for n, k, _ in declared] == [(n, k) for n, k, _ in runtime], name
    for (_, _, stub), (_, _, value) in zip(declared, runtime, strict=True):
        assert (stub is None) == (value is None), name
        if stub not in (None, Ellipsis):
            assert stub == value, name


def _decorators(function):
    return {ast.unparse(decorator) for decorator in function.decorator_list}


@pytest.mark.interface
@pytest.mark.parametrize(
    "stub",
    [node for node in _STUB.body if isinstance(node, ast.FunctionDef)],
    ids=lambda node: node.name,
)
def test_stub_functions_declare_the_native_parameters(stub):
    """Names, kinds and defaults of every native function match its stub."""
    _assert_same_parameters(
        _declared(stub, bound=0),
        _runtime(getattr(_native, stub.name), bound=0),
        stub.name,
    )


@pytest.mark.interface
@pytest.mark.parametrize(
    "stub",
    [node for node in _STUB.body if isinstance(node, ast.ClassDef)],
    ids=lambda node: node.name,
)
def test_stub_classes_declare_the_native_members(stub):
    """Constructors, methods, static methods and properties match the stub."""
    cls = getattr(_native, stub.name)
    members = {
        node.name: node for node in stub.body if isinstance(node, ast.FunctionDef)
    }
    constructor = members.pop("__init__", None)
    runtime = _runtime(cls, bound=0)
    if constructor is None:
        assert runtime == [], stub.name
    else:
        _assert_same_parameters(_declared(constructor, bound=1), runtime, stub.name)
    exported = {name for name in vars(cls) if not name.startswith("__")}
    assert set(members) == exported, stub.name
    for name, member in members.items():
        value = inspect.getattr_static(cls, name)
        decorators = _decorators(member)
        if "property" in decorators:
            assert inspect.isgetsetdescriptor(value), f"{stub.name}.{name}"
            continue
        static = "staticmethod" in decorators
        assert isinstance(value, staticmethod) == static, f"{stub.name}.{name}"
        _assert_same_parameters(
            _declared(member, bound=0 if static else 1),
            _runtime(getattr(cls, name), bound=0 if static else 1),
            f"{stub.name}.{name}",
        )


# Reusable contexts -----------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One native record and pullback method of a context.

    ``record(*arrays)`` returns the native result; ``arrays`` are fresh NumPy
    inputs that reach the binding exactly as given (never normalized by a Python
    wrapper), so their layouts and later mutation are seen by the binding.
    """

    record: Callable[..., Any]
    arrays: tuple[np.ndarray, ...] = ()
    method: str = "pullback"


def _rng():
    return np.random.default_rng(2024)


def _square(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.eye(n) + 0.2 * complex_normal(rng, (n, n))


def _smatrix(n=3, seed=0):
    rng = np.random.default_rng(seed)
    blocks = 0.2 * complex_normal(rng, (2, 2, n, n))
    blocks[0, 0] += np.eye(n)
    blocks[1, 1] += np.eye(n)
    return blocks


_RADII = np.array([0.3, 0.5])
_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 1.5]])
_LAYERS = {
    "epsilon": np.array([2.0 + 0.1j, 3.0, 1.0]),
    "mu": np.array([1.0, 1.1 + 0.02j, 1.0]),
    "kappa": np.array([0.0, 0.05, 0.0], complex),
}
_INCIDENT = np.linspace(0.2, 1.4, 24).reshape(12, 2) * (1 + 0.3j)
_POINTS = np.array([[0.4, 0.3, -0.7], [1.2, -0.1, 0.6]])
_VECTORS = np.array([[0.3, 0.4, 1.2], [0.1, -0.2, 1.3 + 0.1j]])
_SW = tr.SphericalBasis.default(1)
_SW2 = tr.SphericalBasis.default(1, nmax=2, positions=_POSITIONS)
_CW = tr.CylindricalBasis.default([0.1, 0.3], 1)
_KS = (1.3 + 0.01j, 1.5 + 0.02j)


def _modes(basis):
    """Native mode labels and positions of a public basis."""
    return list(basis.modes), basis.positions.tolist()


def _ebcm_samples():
    nodes, weights = np.polynomial.legendre.leggauss(16)
    theta = (nodes + 1) * np.pi / 2
    radii = 0.3 * (1 + 0.2 * np.cos(theta) ** 2)
    slopes = -0.12 * np.cos(theta) * np.sin(theta)
    return np.column_stack([theta, weights * np.pi / 2, radii, slopes])


def case(record, *arrays, method="pullback"):
    """A factory of ``Case`` with fresh copies of ``arrays``."""
    return lambda: Case(record, tuple(np.array(a) for a in arrays), method)


_ORIGIN = [[0.0] * 3]
_SW_PAIR = (list(_SW.modes), list(_SW.modes))
_CW_PAIR = (list(_CW.modes), list(_CW.modes))

CASES = {
    "mie": case(lambda *a: _native.mie(2, *a), 2 * _RADII, *_LAYERS.values()),
    "sphere": case(lambda *a: _native.sphere(1, 2.0, *a), _RADII, *_LAYERS.values()),
    "mie_cyl": case(
        lambda *a: _native.mie_cyl(0.2, 1, 2.0, *a), _RADII, *_LAYERS.values()
    ),
    "cylinder": case(
        lambda kz, *a: _native.cylinder(kz, 1, 2.0, *a),
        np.array([0.1, -0.3]),
        _RADII,
        *_LAYERS.values(),
    ),
    "sphere_cluster": case(
        lambda *a: _native.sphere_cluster(1, 2.0, *a),
        _RADII,
        _LAYERS["epsilon"][:2],
        _POSITIONS,
    ),
    "interaction": case(_native.interaction, 0.3 * _square(4, 1), 0.2 * _square(4, 2)),
    "particle_cluster": case(
        lambda *local: _native.particle_cluster(list(local), *_modes(_SW2), _KS, True),
        0.3 * _square(6, 1),
        0.2 * _square(6, 2),
    ),
    "cylindrical_particle_cluster": case(
        lambda local: _native.cylindrical_particle_cluster(
            [local], *_modes(_CW), _KS, True
        ),
        0.3 * _square(len(_CW), 1),
    ),
    "tmatrix_metric": case(
        lambda a: _native.tmatrix_metric(a, list(_SW.pol), (1.2, 1.4), "cd"),
        0.2 * _square(6, 3) - 0.1,
    ),
    "expansion": case(
        lambda: _native.expansion(
            list(_SW.modes),
            list(_SW2.modes),
            [[0.1, 0.2, -0.3]],
            _POSITIONS.tolist(),
            _KS,
            True,
            True,
        )
    ),
    "cylindrical_expansion": case(
        lambda: _native.cylindrical_expansion(
            *_CW_PAIR, [[0.1, 0.2, -0.3]], [[0.4, -0.1, 0.2]], _KS, True
        )
    ),
    "cw_to_sw": case(
        lambda: _native.cw_to_sw(
            list(_SW.modes), list(_CW.modes), [[0.1, 0.2, -0.3]], _ORIGIN, _KS, True
        )
    ),
    "rotation": case(
        lambda: _native.rotation(*_SW_PAIR, _ORIGIN, _ORIGIN, (0.3, 0.5, -0.2))
    ),
    "cylindrical_rotation": case(
        lambda: _native.cylindrical_rotation(
            *_CW_PAIR, _ORIGIN, _ORIGIN, (0.3, 0.0, 0.0)
        )
    ),
    "plane_expansion": case(
        lambda: _native.plane_expansion(
            *_modes(_SW2),
            [[0.3, 0.4, 1.2], [0.1, -0.2, 1.3 + 0.1j]],
            [0, 1],
            True,
            False,
        )
    ),
    "cylindrical_plane_expansion": case(
        lambda: _native.cylindrical_plane_expansion(
            *_modes(_CW), [[0.3, 0.4, 0.1], [0.1, -0.2, 0.3]], [0, 1], True, False
        )
    ),
    "periodic_to_cw": case(
        lambda: _native.periodic_to_cw(
            list(_CW.modes),
            list(_SW.modes),
            _ORIGIN,
            [[0.0, 0.1, 0.0]],
            list(_KS),
            1.7,
            True,
        )
    ),
    "lattice_expansion_from_table": case(
        lambda table: _native.lattice_expansion_from_table(
            *_SW_PAIR, _ORIGIN, _ORIGIN, True, table
        ),
        complex_normal(_rng(), (1, 1, 2, 9)),
    ),
    "lattice_expansion": case(
        lambda: _native.lattice_expansion(
            *_SW_PAIR,
            _ORIGIN,
            [[0.1, 0.0, 0.0]],
            _KS,
            True,
            [0.1, 0.2],
            [[1.5, 0.0], [0.3, 1.4]],
            0.9 + 0.1j,
        )
    ),
    "cylindrical_lattice_expansion": case(
        lambda: _native.cylindrical_lattice_expansion(
            *_CW_PAIR, _ORIGIN, [[0.1, 0.0, 0.0]], _KS, [0.1], [[1.5]], 0.9 + 0.1j
        )
    ),
    "lattice_sum_record": case(
        lambda k, q, a, r, eta: _native.lattice_sum_record(
            True,
            2,
            [(2, -1), (3, -1)],
            k,
            q,
            a,
            r,
            eta,
            0,
            [0],
            [2],
            [[2], [2], [2, 2], [2, 3], []],
        ),
        np.array([2.1 + 0.2j, 2.3 + 0.1j]),
        np.array([[0.1, 0.2]]),
        np.array([[[1.5, 0.0], [0.2, 1.4]]]),
        np.array([[0.19, 0.11, 0.07], [0.21, -0.09, 0.05]]),
        np.array([0.9 + 0.02j]),
    ),
    "field": case(
        lambda c, p: _native.field(*_modes(_SW2), c, p, _KS, True, True),
        complex_normal(_rng(), len(_SW2)),
        _POINTS,
    ),
    "cylindrical_field": case(
        lambda c, p: _native.cylindrical_field(*_modes(_CW), c, p, _KS, True, True),
        complex_normal(_rng(), len(_CW)),
        _POINTS,
    ),
    "field_operator": case(
        lambda p: _native.field_operator(*_modes(_SW2), p, _KS, True, False), _POINTS
    ),
    "cylindrical_field_operator": case(
        lambda p: _native.cylindrical_field_operator(*_modes(_CW), p, _KS, True, True),
        _POINTS,
    ),
    "plane_field": case(
        lambda v, p, c: _native.plane_field(v, [0, 1], p, c, True, False),
        _VECTORS,
        _POINTS,
        np.array([0.4 - 0.1j, 1.2 + 0.3j]),
    ),
    "plane_field.operator": case(
        lambda v, p: _native.plane_field(v, [0, 1], p, None, True, False),
        _VECTORS,
        _POINTS,
    ),
    "plane_phases": case(_native.plane_phases, _POINTS, _VECTORS),
    "plane_permutation": case(
        lambda v, pol: _native.plane_permutation(v, pol, 1, True),
        _VECTORS,
        np.array([0.0, 1.0]),
    ),
    "spherical_channels": case(
        lambda: _native.spherical_channels(
            *_modes(_SW), list(_KS), [[0.2, 0.1]] * 2, [0, 1], 2.8, True, False
        )
    ),
    "cylindrical_channels": case(
        lambda: _native.cylindrical_channels(
            *_modes(_CW), [1.3, 1.3], [[0.3, 0.1]] * 2, [0, 1], 1.7, True, False
        )
    ),
    "smatrix_from_array": case(
        _native.smatrix_from_array,
        0.2 * _square(3, 4),
        complex_normal(_rng(), (2, 2, 3, 2)),
    ),
    "smatrix_add": case(_native.smatrix_add, _smatrix(3, 1), _smatrix(3, 2)),
    "smatrix_illuminate": case(
        _native.smatrix_illuminate,
        _smatrix(3, 1),
        _smatrix(3, 2),
        *complex_normal(_rng(), (2, 3, 2)),
    ),
    "smatrix_periodic": case(_native.smatrix_periodic, _smatrix(3, 5)),
    "bands": case(lambda s: _native.bands(s, 1.3), _smatrix(2, 6)),
    **{
        f"smatrix_tr{suffix}": case(
            lambda m, i, d=direction: _native.smatrix_tr(
                m,
                i,
                [[1.3, 1.5], [1.2, 1.4]],
                [0.9, 1.1],
                [[0.2, 0.13]],
                [(0, 0), (0, 1)],
                2,
                True,
                d,
                False,
            ),
            _smatrix(2, 7),
            complex_normal(_rng(), (2, 3)),
        )
        # Upward transmission borrows blocks [0, 0] and [1, 0], downward [1, 1]
        # and [0, 1].
        for direction, suffix in ((0, ""), (1, ".down"))
    },
    "chirality_density": case(
        lambda ks, normal: _native.chirality_density(ks, normal, (-0.2, 0.4)),
        np.array([1.3 + 0.1j, 1.1]),
        np.array([1.2 + 0.1j, 0.3 + 0.5j]),
    ),
    "oriented_chirality": case(
        lambda q, normal: _native.oriented_chirality(q, normal, [0, 1], 2, (-0.2, 0.4)),
        np.array([[0.2, 0.3], [0.4, 0.5]]),
        np.array([1.2 + 0.1j, 0.3 + 0.5j]),
    ),
    "fresnel": case(
        lambda: _native.fresnel(
            [[1.3, 1.5], [1.1, 1.4]], [[1.2, 1.4], [1.0, 1.3]], [1.2, 0.9]
        )
    ),
    "interface_coefficients": case(
        lambda: _native.interface_coefficients(
            [[1.3, 1.5], [1.1, 1.4]], [1.2, 0.9], [0.2, 0.1], 2, False
        )
    ),
    "propagation_matrix": case(
        lambda: _native.propagation_matrix(
            [[0.2, 0.1, 1.2 + 0.01j], [0.2, 0.1, 1.3 + 0.02j]], [0.0, 0.1, 0.4]
        )
    ),
    "layer_stack": case(
        lambda: _native.layer_stack(
            [[1.0, 1.0], [1.5 + 0.1j, 1.6 + 0.1j], [1.2, 1.2]],
            [1.0, 0.7, 0.9],
            [[0.2, 0.1]],
            [0.4],
            2,
            False,
        )
    ),
    "ebcm_qmat": case(
        lambda samples, ks: _native.ebcm_qmat(
            samples,
            [(1, 0, 0), (1, 1, 1)],
            [(1, 0, 1), (1, 1, 0)],
            ks,
            (1.2, 0.9),
            True,
            True,
        ),
        _ebcm_samples(),
        np.array([[2.1 + 0.2j, 2.3 + 0.1j], [1.3, 1.4]]),
    ),
    "solve": case(_native.solve, _square(4, 1), complex_normal(_rng(), (4, 2))),
    "eig": case(_native.eig, _square(4, 3) + np.diag([0, 1, 2, 3])),
    "eigvals": case(_native.eigvals, _square(4, 3) + np.diag([0, 1, 2, 3])),
    "svdvals": case(_native.svdvals, _square(4, 5)),
    "InteractionFactor.record": case(
        lambda local, coupling, incident: _native.InteractionFactor(
            local, coupling
        ).record(incident),
        0.3 * _square(4, 1),
        0.2 * _square(4, 2),
        complex_normal(_rng(), (4, 2)),
    ),
    "InteractionFactor.from_blocks.record": case(
        lambda a, b, coupling, incident: _native.InteractionFactor.from_blocks(
            [a, b], coupling
        ).record(incident),
        0.3 * _square(2, 1),
        0.3 * _square(2, 2),
        0.2 * _square(4, 3),
        complex_normal(_rng(), (4, 2)),
        method="pullback_blocks",
    ),
    "sphere_cluster_factor.record": case(
        lambda radii, epsilon, positions, incident: _native.sphere_cluster_factor(
            1, 2.0, radii, epsilon, positions
        ).record(incident),
        _RADII,
        _LAYERS["epsilon"][:2],
        _POSITIONS,
        _INCIDENT,
        method="pullback_blocks",
    ),
    "IterativeSphereCluster.record": case(
        lambda radii, epsilon, positions, incident: _native.IterativeSphereCluster(
            1, 2.0, radii, epsilon, positions
        ).record(incident),
        _RADII,
        _LAYERS["epsilon"][:2],
        _POSITIONS,
        _INCIDENT,
    ),
}

CASES |= {
    f"{name}.pullback_axial": lambda name=name: replace(
        CASES[name](), method="pullback_axial"
    )
    for name in (
        "cylindrical_expansion",
        "cylindrical_lattice_expansion",
        "cylindrical_field",
        "cylindrical_field_operator",
    )
}


def _split(result):
    """Recorded values and the context of a native result (reports dropped)."""
    items = result if isinstance(result, tuple) else (result,)
    index = next(i for i, item in enumerate(items) if hasattr(item, "pullback"))
    return items[:index], items[index]


def _cotangents(values):
    """Random cotangents with each recorded value's shape and real/complex kind."""
    rng = np.random.default_rng(7)
    return tuple(
        complex_normal(rng, np.shape(v))
        if np.iscomplexobj(v)
        else (rng.normal(size=np.shape(v)) if np.ndim(v) else rng.normal())
        for v in values
    )


class _Method:
    """Expose a case's pullback method as ``pullback`` for the shared helper."""

    def __init__(self, context, case):
        self.pullback = getattr(context, case.method)


def _assert_reusable(context, cotangents, expected):
    """The shared reuse contract, plus cotangents with an extra leading axis."""
    if all(np.ndim(c) == 0 for c in cotangents):  # Python scalar cotangents
        with pytest.raises(ValueError, match="finite"):
            context.pullback(*(np.nan for _ in cotangents))
        assert_tree_allclose(context.pullback(*cotangents), expected, rtol=0, atol=0)
        assert_tree_allclose(context.pullback(*cotangents), expected, rtol=0, atol=0)
        return
    single = len(cotangents) == 1
    extra_axis = tuple(np.asarray(c)[None] for c in cotangents)
    assert_reusable_context(
        context,
        cotangents[0] if single else cotangents,
        expected,
        wrong_shape=extra_axis[0] if single else extra_axis,
        rtol=0,
        atol=0,
    )


@pytest.mark.gradients
@pytest.mark.parametrize("name", CASES)
def test_context_is_deterministic_reusable_and_owns_its_inputs(name):
    """Rejected cotangents keep the residual; overwriting the recorded inputs
    or the returned values does not change the pullback."""
    case = CASES[name]()
    values, context = _split(case.record(*case.arrays))
    twin_values, twin = _split(case.record(*case.arrays))
    assert_tree_allclose(values, twin_values, rtol=0, atol=0)
    cotangents = _cotangents(values)
    expected = _Method(twin, case).pullback(*cotangents)
    pushforward_name = case.method.replace("pullback", "pushforward")
    twin_pushforward = getattr(twin, pushforward_name)
    directions = expected if isinstance(expected, tuple) else (expected,)
    # Iterative pullbacks append convergence certificates after their gradients.
    directions = directions[: len(inspect.signature(twin_pushforward).parameters)]
    expected_tangent = twin_pushforward(*directions)
    expected_plain = (
        twin.pullback(*cotangents) if case.method == "pullback_axial" else None
    )
    outputs = [v for v in values if isinstance(v, np.ndarray) and v.dtype.kind in "fc"]
    for a in (*case.arrays, *outputs):
        a[...] = np.nan
    _assert_reusable(_Method(context, case), cotangents, expected)
    pushforward = getattr(context, pushforward_name)
    assert_tree_allclose(pushforward(*directions), expected_tangent, rtol=0, atol=0)
    assert_tree_allclose(
        _Method(context, case).pullback(*cotangents), expected, rtol=0, atol=0
    )
    assert_tree_allclose(pushforward(*directions), expected_tangent, rtol=0, atol=0)
    if expected_plain is not None:  # axial and ordinary methods share the residual
        assert_tree_allclose(
            context.pullback(*cotangents), expected_plain, rtol=0, atol=0
        )


@pytest.mark.gradients
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name", CASES)
def test_context_inputs_and_cotangents_in_every_layout(name, layout):
    """Records and pullbacks read every NumPy layout of their arrays."""
    case = CASES[name]()
    values, context = _split(case.record(*case.arrays))
    cotangents = _cotangents(values)
    expected = _Method(context, case).pullback(*cotangents)
    arranged = [arrange(a, layout) for a in case.arrays]
    actual_values, context = _split(case.record(*arranged))
    assert_tree_allclose(actual_values, values, rtol=1e-12, atol=1e-12)
    cotangents = [arrange(c, layout) if np.ndim(c) else c for c in cotangents]
    actual = _Method(context, case).pullback(*cotangents)
    assert_tree_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("name", CASES)
def test_context_takes_real_arrays_and_nested_lists(name):
    """A real cotangent equals its complex cast; nested sequences are arrays.

    Real arrays convert as a whole: NumPy before 2.5 turns a size-1 row into a
    complex scalar with a DeprecationWarning, so an element-wise conversion
    would flatten a trailing unit axis and accept it. Ignoring the warning
    checks that such a cotangent is rejected on those versions too.
    """
    case = CASES[name]()
    values, context = _split(case.record(*case.arrays))
    _, twin = _split(case.record(*case.arrays))
    real = [np.real(c) if np.ndim(c) else c for c in _cotangents(values)]
    lists = [(c + 0j).tolist() if np.ndim(c) else c for c in real]
    unit_axis = [c[..., None] if np.ndim(c) else c for c in real]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if any(np.ndim(c) for c in real):
            with pytest.raises(ValueError, match="shape"):
                _Method(context, case).pullback(*unit_axis)
        gradients = _Method(context, case).pullback(*real)
    assert_tree_allclose(
        gradients, _Method(twin, case).pullback(*lists), rtol=0, atol=0
    )


#: Contexts of the broadcast special-function, coordinate and wave bindings;
#: their test modules check the same contract with assert_reusable_context.
BROADCAST_CONTEXTS = {
    "AngularContext",
    "BesselContext",
    "CoordinatesContext",
    "CylindricalTranslationContext",
    "IncgammaContext",
    "IntkambeContext",
    "SphericalTranslationContext",
    "VectorCoordinatesContext",
    "VectorWaveContext",
    "WignerdContext",
}


@pytest.mark.interface
def test_every_object_context_method_has_a_case():
    """A new context or pullback method fails here until CASES covers it."""
    methods = {
        (name, method)
        for name in dir(_native)
        if isinstance(cls := getattr(_native, name), type)
        and hasattr(cls, "pullback")
        and name not in BROADCAST_CONTEXTS
        for method in vars(cls)
        if method.startswith("pullback")
    }
    covered = set()
    for factory in CASES.values():
        case = factory()
        covered.add((type(_split(case.record(*case.arrays))[1]).__name__, case.method))
    assert covered == methods


@pytest.mark.interface
def test_every_native_class_names_the_native_module():
    """Reprs, pickling errors and the reference show ``treams_rs._native.<Name>``."""
    classes = {
        name: cls.__module__
        for name in dir(_native)
        if isinstance(cls := getattr(_native, name), type)
    }
    assert "SphereContext" in classes
    assert classes == dict.fromkeys(classes, "treams_rs._native")
