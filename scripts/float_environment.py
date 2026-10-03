"""Native results for callers that flush subnormals, compared with IEEE callers.

XLA flushes subnormals to zero (FTZ/DAZ) on the thread that runs a computation,
also while a ``jax.pure_callback`` runs, and ``torch.set_flush_denormal(True)``
does the same. Every native entry point runs its body in
``treams_core::fpenv::ieee``, so its outcome (value, pullback of a returned
context, or error) must not depend on the caller's mode. ``check`` runs every
case once as usual and once inside ``_native._run_flushing_for_tests``, which
flushes as XLA does, and requires equal outcomes bit for bit.

Drawn cases call every scalar binding that the stub declares and every dtype
loop of every native ufunc, elementwise and generalized, with subnormal, tiny
and ordinary operands, until each returns a value for some call and, with a
float or complex operand, has a call that ``sensitive`` predicts flushing would
change. ``INSENSITIVE`` names the loops that the rule cannot see, each with a
witness call chosen by hand. With seed 0, as the tests and the wheel check
draw, a build whose guard only runs its work returns other results on the
flushing thread for every binding and loop with a float or complex operand and
for every family. Hand-picked families add array bindings: Bessel values
through ufunc wrappers, literal scalars with subnormal results, wrapper fast
paths, records with their pullbacks, band-pivot solves and a lossless slab.
Other array bindings, such as those of T-matrices, fields, EBCM, cylinders,
illuminations and channels, are not compared here.

``tests/bindings/test_float_environment.py`` runs the check on the build under test and
``scripts/check_wheel.py`` on the release wheel; it needs only NumPy and the
extension. Operands are built before the flushing call: arithmetic on the
flushing thread would flush them.
"""

from __future__ import annotations

import ast
import math
import random
import re
import sys
from dataclasses import dataclass, replace
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import treams_rs as tr
from treams_rs import _native, diff, special

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    type Cases = Mapping[str, Call]
    type Calls = dict[str, Call]

# Operands by stub annotation or ufunc type code. Flushing zeroes subnormal
# operands (denormals-are-zero) and results (flush-to-zero), also products of
# two operands near 1e-160; the ordinary values keep many draws inside the valid
# domains. Ufunc loops take mode labels as floats too, so two thirds of their
# real operands are integral. Integers in tuples and in ufunc loops are mode
# labels: degrees, orders and polarizations.
FLOATS = (5e-324, 3e-310, 2.2250738585072014e-308, 1e-300, 1e-160, 0.5, -0.7, 2.5)
INTEGRAL = (0.0, 1.0, 2.0, 3.0, -1.0)
COMPLEXES = (
    3e-310 + 0j,
    2e-310 + 3e-310j,
    1 + 3e-310j,
    1e-300j,
    0.5 + 0.1j,
    2 - 0.5j,
    -0.3 - 1e-310j,
    1 + 0j,
    740 + 0j,
)
# Lattice sums take a number of terms that grows with the wavenumber, so the
# generalized loops draw complex values of moderate magnitude only.
MODERATE = tuple(z for z in COMPLEXES if abs(z) < 10)
LABELS = (0, 1, 1, 2)
POOLS = {
    "float": FLOATS + INTEGRAL,
    "complex": COMPLEXES,
    "int": (0, 1, 1, 2, -1, 3),
    "bool": (False, True),
    # The special functions that the `function` parameters of scalar bindings select.
    "str": ("j", "y", "h1", "h2", "legendre", "pi", "tau"),
}

SCALAR = r"(?:float|complex|int|bool|str)"
SCALAR_ANNOTATION = re.compile(
    rf"{SCALAR}|tuple\[{SCALAR}(?:, {SCALAR})*(?:, \.\.\.)?\]"
)

# At least this many bindings and loops; fewer means that the enumeration broke.
SCALAR_BINDINGS = 22
UFUNC_LOOPS = 145

# Draw at most this many calls of a binding or loop while none returns a value
# or none is flush-sensitive. The rarest, cw_rotate_scalar, needs
# equal wavenumbers, orders and polarizations: about 1200 draws on average.
DRAW_LIMIT = 10_000

# Loops with float operands that flushing changes through subnormal
# intermediates only, which `sensitive` cannot see, and the operands of a
# witness call that a kernel that flushes would change. None of their drawn
# calls may count as sensitive; the witness joins the draws.
INSENSITIVE = {
    # sqrt(k**2 - kx**2 - ky**2) of real operands is never subnormal, and
    # subnormal operands square to zero anyway; the square of k = 1e-160 is
    # subnormal, so the witness returns 9.99994e-161 with gradual underflow and
    # 0 in a kernel that flushes.
    "wave_vector_z[ddd->D]": (0.0, 0.0, 1e-160),
}

# Pivots inside the unit square but outside the unit disc, whose reciprocal faer
# forms through the subnormal MIN_POSITIVE / |z|**2.
PIVOTS = np.array([1 + 0.1j, 0.9 + 0.6j, -0.3 - 0.99j])
OPERATOR = np.diag(PIVOTS) + 0.05j * np.ones((3, 3))

run_flushing = _native._run_flushing_for_tests


class Call(partial):
    """A deterministic call whose outcome for IEEE callers is computed once."""

    @cached_property
    def ieee(self) -> object:
        return outcome(self)


@dataclass(frozen=True)
class Failure:
    """The outcome of a call that raised."""

    kind: str
    message: str


@dataclass(frozen=True)
class Bits:
    """The bit pattern of a float, complex or array value."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes


def flushes() -> bool:
    """Whether this thread flushes the subnormal ``MIN_POSITIVE / 2`` to zero."""
    tiny, two = sys.float_info.min, 2.0
    return tiny / two == 0.0


def supported() -> bool:
    """Whether the test hook flushes on this platform (x86-64 and AArch64)."""
    return run_flushing(flushes)


def scalar_bindings() -> dict[str, list[str]]:
    """Parameter annotations of the native functions that the stub declares with
    Python scalars and tuples of them only: in Rust, they take register values."""
    stub = Path(_native.__file__).with_name("_native.pyi")
    bindings = {}
    for node in ast.parse(stub.read_text()).body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        arguments = node.args
        annotations = [
            ast.unparse(a.annotation or ast.Constant(None)) for a in arguments.args
        ]
        extra = arguments.posonlyargs or arguments.vararg or arguments.kwonlyargs
        if (
            annotations
            and not (extra or arguments.kwarg)
            and all(SCALAR_ANNOTATION.fullmatch(a) for a in annotations)
        ):
            bindings[node.name] = annotations
    assert len(bindings) >= SCALAR_BINDINGS, f"only {sorted(bindings)} found"
    return bindings


def _draw(annotation: str, rng: random.Random, *, label: bool = False) -> object:
    if annotation.startswith("tuple["):
        items = annotation.removeprefix("tuple[").removesuffix("]").split(", ")
        if items[-1] == "...":
            items = items[:1] * 3
        return tuple(_draw(item, rng, label=True) for item in items)
    return rng.choice(LABELS if label and annotation == "int" else POOLS[annotation])


def _drawn(
    name: str,
    function: Callable[..., object],
    draw: Callable[[random.Random], tuple[object, ...]],
    operands: Callable[[tuple[object, ...]], tuple[object, ...]],
    seed: int,
    draws: int,
) -> Calls:
    """At least ``draws`` calls of ``function``, a binding or loop, with the
    operands of values from ``draw``, and more, up to ``DRAW_LIMIT``, until one
    returns a value and, unless ``INSENSITIVE`` names it, one is flush-sensitive
    or has no float or complex operand. An exempt loop gets its witness call."""
    rng = random.Random(f"{name}:{seed}")
    calls = {}
    returns, found = False, name in INSENSITIVE
    if found:
        witness = INSENSITIVE[name]
        calls[f"{name}{witness!r}"] = Call(function, *operands(witness))
    for count in range(DRAW_LIMIT):
        if count >= draws and returns and found:
            break
        values = draw(rng)
        calls[f"{name}{values!r}"] = call = Call(function, *operands(values))
        if not (returns and found):
            returns = returns or not isinstance(call.ieee, Failure)
            found = found or not _floating(call.args) or sensitive(call)
    assert returns, f"no call of {name} returns a value"
    assert found, f"no call of {name} is flush-sensitive"
    return calls


def scalar_cases(seed: int, draws: int) -> dict[str, Calls]:
    """Drawn calls of every scalar binding, by binding."""

    def arguments(parameters: list[str], rng: random.Random) -> tuple[object, ...]:
        return tuple(_draw(parameter, rng) for parameter in parameters)

    return {
        name: _drawn(
            name,
            getattr(_native, name),
            partial(arguments, parameters),
            tuple,
            seed,
            draws,
        )
        for name, parameters in scalar_bindings().items()
    }


def _values(
    codes: str,
    shapes: list[tuple[int, ...]],
    complexes: tuple[complex, ...],
    rng: random.Random,
) -> tuple[object, ...]:
    """One element of each operand, core dimensions as nested lists."""

    def element(code: str) -> object:
        if code == "d":
            return rng.choice(INTEGRAL if rng.random() < 2 / 3 else FLOATS)
        return rng.choice(LABELS if code == "l" else complexes)

    return tuple(
        np.reshape([element(code) for _ in range(math.prod(shape))], shape).tolist()
        for code, shape in zip(codes, shapes, strict=True)
    )


def _operands(codes: str, values: tuple[object, ...]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.array([value], dtype=code) for code, value in zip(codes, values, strict=True)
    )


def ufunc_cases(seed: int, draws: int) -> dict[str, Calls]:
    """Drawn one-element calls of every dtype loop of every native ufunc,
    elementwise and generalized, by loop."""
    cases = {}
    for name in dir(_native):
        function = getattr(_native, name)
        if not isinstance(function, np.ufunc):
            continue
        # The fixed core shape of each input.
        shapes = [()] * function.nin
        if function.signature is not None:
            inputs = function.signature.split("->")[0]
            shapes = [
                tuple(int(size) for size in dimensions.split(",") if size)
                for dimensions in re.findall(r"\(([^)]*)\)", inputs)
            ]
        complexes = COMPLEXES if function.signature is None else MODERATE
        for types in function.types:
            codes = types.split("->")[0]
            loop = f"{name}[{types}]"
            cases[loop] = _drawn(
                loop,
                function,
                partial(_values, codes, shapes, complexes),
                partial(_operands, codes),
                seed,
                draws,
            )
    assert len(cases) >= UFUNC_LOOPS, f"only {sorted(cases)} found"
    assert INSENSITIVE.keys() <= cases.keys(), "every exemption names a loop"
    return cases


def family_cases() -> dict[str, Calls]:
    """Subnormal results of scalar bindings through the public API, and some
    array bindings, with results that flushing would change where they admit
    them."""
    tiny = np.array([1e-31, 3e-31])
    parallel = np.tile(tiny, 2048)  # above the parallel threshold of the loop
    points = np.array([[1.0, 3e-310, 0.5], [2e-310, 1.0, -0.5]] * 32)
    ports = tr.PlaneWavePorts.default([[0.1, 0.05]])
    material = tr.Material(5.0, 1.0, 0.0)
    k, q, a = np.asarray(2.1 + 0.2j), np.array([0.1, 0.2]), np.diag([1.5, 1.7])
    r = np.array([0.19, 0.11, 0.07])
    return {
        # Subnormal values through a 0-d, a 1-d and a parallel ufunc loop.
        "bessel": {
            "0-d": Call(special.spherical_jn, 10, 1e-31),
            "1-d": Call(special.spherical_jn, 10, tiny),
            "parallel": Call(special.spherical_jn, 10, parallel),
        },
        # Literal arguments whose results are subnormal.
        "scalars": {
            "incgamma_scalar": Call(special.incgamma, 0.5, 740.0),
            "lpmv_real_scalar": Call(special.lpmv, 0, 1, 3e-310),
            "angular_scalar": Call(special.lpmv, 0, 1, 3e-310 + 0j),
            "bessel_record_scalar": Call(diff.bessel, 1, 3e-310, spherical=True),
            "incgamma_record_scalar": Call(diff.incgamma, 0.5, 740.0),
        },
        # Scalar fast paths of ufunc wrappers, and a generalized ufunc loop.
        "wrappers": {
            "pw_translate": Call(
                _native.pw_translate, 1.0, 0.5, 0.2, 3e-310, 1e-310, 2e-310
            ),
            "vpw_A": Call(_native.vpw_A, 1.0, 0.5, 0.2, 3e-310, 1e-310, 2e-310, 1),
            "car2sph": Call(_native.car2sph, [1.0, 3e-310, 0.5]),
            "vcar2sph": Call(_native.vcar2sph, [1.0, 0.0, 0.0], [1.0, 3e-310, 0.5]),
            "cell_reciprocal": Call(_native.cell_reciprocal, [[1.0, 3e-310], [0, 2.0]]),
            "car2sph loop": Call(_native.car2sph, points),
        },
        # Records with pullback contexts, with and without the GIL.
        "records": {
            "bessel": Call(diff.bessel, 1, np.array([3e-310, 1e-309]), spherical=True),
            "incgamma": Call(diff.incgamma, 0.5, np.array([740.0])),
            "lattice_sum": Call(diff.lattice_sum, 2, 2, -1, k, q, a, r, 0.9),
        },
        # LU solves whose pivots faer inverts through subnormals, with pullbacks.
        "solve": {
            **{f"pivot {z}": Call(diff.solve, [[z]], [[1.0]]) for z in PIVOTS},
            "system": Call(diff.solve, OPERATOR, np.eye(3, dtype=complex)),
        },
        # A lossless slab whose multiple-reflection pivot lies in the band.
        "slab": {
            "slab": Call(
                lambda: (
                    tr.slab(
                        k0=1.4375, basis=ports, thickness=0.234375, material=material
                    ).array
                )
            )
        },
    }


def outcome(call: Callable[[], object]) -> object:
    """A bit-exact description of ``call()``, or its ``Failure``. Records are
    pulled back once with unit cotangents."""
    try:
        return _bits(call())
    except Exception as error:  # errors are outcomes as well
        return Failure(type(error).__name__, str(error))


def _bits(value: object) -> object:
    if isinstance(value, float | complex | np.ndarray | np.generic):
        array = np.asarray(value)
        return Bits(array.dtype.str, array.shape, array.tobytes())
    if isinstance(value, tuple | list):
        if (
            value
            and hasattr(value[-1], "pullback")
            and isinstance(value[0], np.ndarray)
        ):
            # A record: pull its context back once, with a unit cotangent.
            *values, context = value
            pullback = outcome(partial(context.pullback, np.ones_like(values[0])))
            return (*map(_bits, values), pullback)
        return tuple(map(_bits, value))
    if value is None or isinstance(value, int | str):
        return value
    raise TypeError(f"no bit pattern for {type(value).__name__}")


def _zero_subnormals(array: np.ndarray) -> np.ndarray:
    """Replace the subnormal parts of a float64 or complex128 ``array`` in place
    by zeros of their sign, as flushing reads operands and writes results."""
    parts = array.reshape(-1).view(np.float64)
    parts[np.abs(parts) < sys.float_info.min] *= 0.0
    return array


def _flushed(outcome: object) -> object:
    """``outcome`` with the subnormal parts of its values flushed to zero."""
    if isinstance(outcome, Bits):
        if np.dtype(outcome.dtype).char not in "dD":
            return outcome
        array = _zero_subnormals(np.frombuffer(outcome.data, outcome.dtype).copy())
        return replace(outcome, data=array.tobytes())
    if isinstance(outcome, tuple):
        return tuple(map(_flushed, outcome))
    return outcome


def _floating(value: object) -> bool:
    """Whether ``value`` is, or holds, a float or complex operand."""
    if isinstance(value, tuple | list):
        return any(map(_floating, value))
    return isinstance(value, float | complex) or (
        isinstance(value, np.ndarray) and value.dtype.char in "dD"
    )


def _read_as_zero(value: object) -> object:
    """``value`` with its subnormal parts replaced by zeros of their sign, as
    denormals-are-zero reads them."""
    if isinstance(value, tuple | list):
        return type(value)(map(_read_as_zero, value))
    if not _floating(value):
        return value
    array = _zero_subnormals(np.array(value))
    return array if isinstance(value, np.ndarray) else array.item()


def sensitive(call: Call) -> bool:
    """Whether flushing is predicted to change the IEEE outcome of ``call``: it
    has a subnormal part, which flush-to-zero writes as zero, or ``call``
    returns another value when its subnormal operands read as zero, as
    denormals-are-zero reads them, and its subnormal results flush.

    Checks for zero may test bit patterns, which flushing leaves alone, so this
    is a prediction, not a proof. Errors of zeroed operands do not count: such
    a check lets the flushing call pass. A call that raises for IEEE callers
    and returns a value for zeroed operands counts, although such a check can
    make the flushing call raise as well (realsumsw1d[dDdddD->D] with a
    subnormal split parameter): label loops such as wigner3j[dddddd->d], where
    a subnormal label is not an integer but zero is, and some lattice sums have
    no other sensitive calls in thousands of draws. Subnormal intermediates
    stay unseen (``INSENSITIVE``)."""
    expected = call.ieee
    if _flushed(expected) != expected:
        return True
    arguments = map(_read_as_zero, call.args)
    zeroed = outcome(partial(call.func, *arguments, **call.keywords))
    return not isinstance(zeroed, Failure) and _flushed(zeroed) != expected


def compare(cases: Cases) -> list[str]:
    """The names of the cases whose outcome differs on a thread that flushes
    subnormals from their outcome for IEEE callers.

    Fails unless that thread flushes before and after the native calls, and
    until the test hook returns."""
    expected = {name: call.ieee for name, call in cases.items()}

    def flushing() -> tuple[bool, dict[str, object], bool]:
        before = flushes()
        actual = {name: outcome(call) for name, call in cases.items()}
        return before, actual, flushes()

    before, actual, after = run_flushing(flushing)
    assert before, "the test hook flushes subnormals"
    assert after, "native calls restore the caller's mode"
    assert not flushes(), "the test hook restores the caller's mode"
    return [name for name in cases if actual[name] != expected[name]]


def check_family(families: Mapping[str, Cases]) -> int:
    """Compare every case of every family bit for bit; every family must return
    a value for some case. Returns the number of cases."""
    cases = {case: call for calls in families.values() for case, call in calls.items()}
    mismatches = compare(cases)
    assert mismatches == [], f"flushing callers get other results: {mismatches}"
    silent = [
        family
        for family, calls in families.items()
        if all(isinstance(call.ieee, Failure) for call in calls.values())
    ]
    assert silent == [], f"no case returns a value: {silent}"
    return len(cases)


def check_drawn(groups: dict[str, Calls]) -> int:
    """``check_family`` for the drawn calls of bindings or loops, whose exempt
    loops must still have no flush-sensitive call. Returns the number of calls."""
    stale = [
        group
        for group in INSENSITIVE.keys() & groups.keys()
        if any(map(sensitive, groups[group].values()))
    ]
    assert stale == [], f"stale exemptions: {stale}"
    return check_family(groups)


def check(seed: int = 0, draws: int = 48) -> str:
    """Run every case on a flushing thread; a summary, or an assertion error."""
    scalars = check_drawn(scalar_cases(seed, draws))
    loops = check_drawn(ufunc_cases(seed, draws))
    families = check_family(family_cases())
    return (
        f"{scalars} scalar-binding calls, {loops} ufunc-loop calls and {families} "
        "calls of binding families on a flushing thread match IEEE callers bit "
        "for bit"
    )
