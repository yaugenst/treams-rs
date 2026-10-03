# ruff: noqa: E741 - conventional multipole degree l
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["mpmath==1.3.0", "numpy"]
# ///
"""Generate or check the mpmath reference tables in crates/treams-core/references/.

Each table has a subcommand:

    python scripts/generate_references.py TABLE [--check] [--output PATH] [--workers N]

Without --check the command computes every row of TABLE and writes the header and
the rows to PATH (default: the committed table); lattice-sums recomputes the cases
of the committed table and keeps its comments in place. With --check it reads the
rows of PATH, recomputes each one from its key, prints the worst relative error
against the table's tolerance and exits with status 1 when a row is outside it; it
writes nothing. Run it from the repository root in the development environment:

    uv run python scripts/generate_references.py incgamma --check

crates/treams-core/references/README.md lists the tables, their recipes, run times
and the tests that read them.
"""

import argparse
import cmath
import itertools
import math
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

# mpmath creates its functions when it is imported, so a type checker sees them as
# None; typing the module as Any keeps the numerical code readable.
mp: Any = mpmath

REFERENCES = Path("crates/treams-core/references")
# A key holds the fields left of the colon of a row, a row's values those right of it.
Key = tuple[Any, ...]
Values = list[complex]


# Upper incomplete gamma function -------------------------------------------------

INCGAMMA_DEGREES = [-8, -7.5, -6.5, -5, -4.5, -3, -2.5, -1.5, -1, -0.5, 0, 0.5, 1]
INCGAMMA_DEGREES += [1.5, 3, 4.5, 7.5, 8]
INCGAMMA_REAL = [-60, -36, -20, -9, -3, -0.5, 0, 1, 4, 4.5, 12, 36, 60]
INCGAMMA_IMAG = [0.0, 1e-06, 0.4, 3, 10, 20]


def incgamma_rows() -> list[Key]:
    """The grid of the header without z = 0, where Gamma(n, z) has a pole for n <= 0;
    im(z) = -0.0 adds the lower side of the cut for re(z) < 0."""
    return [
        (n, float(re), im)
        for n in INCGAMMA_DEGREES
        for re in INCGAMMA_REAL
        for im in [*INCGAMMA_IMAG, *([-0.0] if re < 0 else [])]
        if re != 0 or im != 0
    ]


def incgamma(key: Key) -> Values:
    n, re, im = key
    with mp.workdps(40):
        value = mp.gammainc(n, mp.mpc(re, abs(im)))
        if math.copysign(1.0, im) < 0:
            value = mp.conj(value)
        return [complex(value)]


# Kambe integral by quadrature ----------------------------------------------------

KAMBE_ORDERS = [-14, -11, -8, -5, -4, -3, -2, -1, 0, 1, 4, 7]
KAMBE_Z = [(0.3, 0.1), (1.2, 0.0), (2.5, 0.4), (4.0, -0.5)]
KAMBE_ETA = [0.5, 1.0, 1.7, 3.0]
# The table leaves out values below this modulus.
KAMBE_SMALLEST = 1e-30


def kambe_rows() -> list[Key]:
    return [
        (n, re, im, eta)
        for n in KAMBE_ORDERS
        for re, im in KAMBE_Z
        for eta in KAMBE_ETA
    ]


def kambe_quadrature(key: Key) -> Values:
    """int_eta^inf t^n exp(-z^2 t^2 / 2 + 1 / (2 t^2)) dt by tanh-sinh quadrature,
    confirmed by Gauss-Legendre quadrature where the table keeps the value."""
    n, re, im, eta_float = key
    with mp.workdps(40):
        z, eta = mp.mpc(re, im), mp.mpf(eta_float)

        def integrand(t: Any) -> Any:
            return t**n * mp.exp(-(z**2) * t**2 / 2 + 1 / (2 * t**2))

        # Intervals that double from the decay scale of the integrand at eta, up to
        # where it has fallen by 60 orders of magnitude.
        scale = 1 / max(abs(z) ** 2 * eta, eta**-3)
        points, step = [eta], scale
        peak = abs(integrand(eta))
        while abs(integrand(eta + step)) > mp.mpf(10) ** -60 * peak:
            points.append(eta + step)
            peak = max(peak, abs(integrand(eta + step)))
            step *= 2
        points.append(mp.inf)
        value = mp.quad(integrand, points)
        check = mp.quad(integrand, points, method="gauss-legendre")
        if abs(value) >= KAMBE_SMALLEST and abs(check - value) > 1e-15 * abs(value):
            raise ArithmeticError(f"I_{n}({z}, {eta}): quadratures disagree")
        return [complex(value)]


def kambe_omit(values: Values) -> bool:
    return abs(values[0]) < KAMBE_SMALLEST


# Kambe integrals and Ewald chain sums in high precision ---------------------------

CHAIN_DIGITS = 30
TURN = 2 * math.pi
# (l, m, k, kpar, period, shift): chains 2 to 5 periods off the axis, where the
# double-precision Ewald sum with the automatic split cancels.
CHAIN_ROWS = [
    *[
        (l, m, 1.0, 0.3, 1.7, (4.0, 0.0, 0.3))
        for l, m in [(0, 0), (8, 0), (8, 5), (16, 0), (20, -13)]
    ],
    *[
        (l, m, 1.0, 0.3, 1.7, (4.25, 0.0, 0.3))
        for l, m in [(0, 0), (12, 4), (16, -16), (20, 0)]
    ],
    *[(l, m, 6.0, 0.3, 1.7, (6.0, 0.0, 0.3)) for l, m in [(0, 0), (3, -1), (12, 6)]],
    *[(l, m, 2.0, 0.3, 1.7, (8.0, 0.0, 0.3)) for l, m in [(0, 0), (2, 1)]],
    *[(l, m, 12.0, 0.3, 1.7, (4.0, 0.0, 0.3)) for l, m in [(0, 0), (3, 2)]],
    (1, 1, 3.3, 0.3, 1.7, (3.4, 1.0, 0.0)),
    *[(l, m, 4.36, 1.1, 0.88, (-2.472, -1.38, 0.437)) for l, m in [(3, 1), (10, -7)]],
    *[(l, m, 0.8, 0.3, 1.7, (5.5, 0.0, 0.3)) for l, m in [(0, 0), (6, 2)]],
    *[
        (l, m, complex(2.3, 0.4), 0.3, 1.7, (5.0, 1.0, 0.2))
        for l, m in [(2, -1), (16, 9)]
    ],
    (5, 3, complex(1.0, 0.9), -2.0, 1.0, (3.0, -2.0, 0.45)),
    (4, -2, 1.5, 0.4 + 2 * TURN / 1.2, 1.2, (3.5, 2.0, -0.5)),
    (24, 11, 1.0, 0.0, 1.7, (5.0, 0.0, -0.6)),
    (24, 0, 3.0, 0.2, 1.0, (3.0, 0.5, 0.1)),
]


def solid(l: int, m: int, point: Any) -> Any:
    """r^l P_l^m(z / r) e^(i m phi) with the Condon-Shortley phase."""
    x, y, z = point
    square = x * x + y * y + z * z
    order = abs(m)
    value = mp.mpc(1)
    for k in range(1, order + 1):
        value *= -(2 * k - 1) * mp.mpc(x, y)
    previous = mp.mpc(0)
    for degree in range(order + 1, l + 1):
        a = mp.mpf(2 * degree - 1) / (degree - order)
        b = mp.mpf(degree + order - 1) / (degree - order)
        value, previous = a * z * value - b * square * previous, value
    if m < 0:
        ratio = mp.factorial(l - order) / mp.factorial(l + order)
        value = (-1) ** order * ratio * mp.conj(value)
    return value


def scaled_seed(d: Any, a: Any) -> Any:
    """Gamma(d, a) / a^d, by the continued fraction of DLMF 8.9.2 at large Re a,
    where mpmath's gammainc is slow at high precision."""
    if abs(a) < 20 or mp.re(a) <= 0.3 * abs(a):
        return mp.gammainc(d, a) / mp.power(a, d)
    tiny = mp.mpf(10) ** (-(mp.mp.dps + 50))
    b = a + 1 - d
    c, dd = 1 / tiny, 1 / b
    h = dd
    for j in range(1, 20000):
        coefficient = -j * (j - d)
        b += 2
        dd = b + coefficient * dd
        c = b + coefficient / c
        dd = 1 / (dd if abs(dd) >= tiny else tiny)
        c = c if abs(c) >= tiny else tiny
        delta = c * dd
        h *= delta
        if abs(delta - 1) < mp.mpf(10) ** (-(mp.mp.dps + 5)):
            return mp.exp(-a) * h
    return mp.gammainc(d, a) / mp.power(a, d)


def scaled_gammas(top: Any, a: Any, count: int) -> list[Any]:
    """Gamma(d, a) / a^d for d = top, top - 1, ...: the downward recurrence in the
    extra precision its unstable steps (|d| < |a|) need."""
    size = abs(complex(a))
    extra = sum(
        max(0.0, math.log10(size / abs(float(top) - j)))
        for j in range(1, count)
        if float(top) != j
    )
    with mp.workdps(mp.mp.dps + int(extra) + 10):
        a = mp.mpmathify(a)
        value = scaled_seed(mp.mpf(top), a)
        exponential = mp.exp(-a)
        values = [value]
        degree = mp.mpf(top)
        for _ in range(count - 1):
            # The step to d = 0 would divide by zero: Gamma(0, a) = E_1(a) restarts it.
            if degree == 1:
                value = mp.e1(a)
            else:
                value = (a * value - exponential) / (degree - 1)
            degree -= 1
            values.append(value)
    return [+value for value in values]


def kambe(n: int, z: Any, eta: Any) -> Any:
    """int_eta^inf t^n exp(-z^2 t^2 / 2 + 1 / (2 t^2)) dt by its gamma series, for a
    real split eta or the complex eta of the lattice-sum arguments."""
    c = 1 / (2 * eta**2)
    count = int(3 * abs(c) + 2 * mp.mp.dps + 40)
    ladder = scaled_gammas(mp.mpf(n + 1) / 2, (z * eta) ** 2 / 2, count)
    total, term = mp.mpc(0), mp.mpf(1)
    for j in range(count):
        if j:
            term *= c / j
        total += term * ladder[j]
    return eta ** (n + 1) / 2 * total


def reduced(n: int, v: Any, t: Any, below: bool) -> Any:
    """(-1)^n int_1^inf u^(-n-1) exp(-v u - t / (2u)) du, continued in v; `below`
    takes a real v from below the cut, as the outgoing sums do."""
    if below:
        v = mp.mpc(mp.re(v), -(mp.mpf(10) ** (-(mp.mp.dps + 40))))
    count = int(3 * abs(t) / 2 + 2 * mp.mp.dps + 40)
    ladder = scaled_gammas(-n, v, count)
    total, term = mp.mpc(0), mp.mpf(1)
    for j in range(count):
        if j:
            term *= -t / (2 * j)
        total += term * ladder[j]
    return (-1) ** n * total


def polynomial(l: int, m: int, n: int, rho2: Any, beta: Any, azimuth: Any) -> Any:
    order = abs(m)
    total = mp.mpc(0)
    for s in range(n, min(2 * n - order, l) + 1):
        if (order - s) % 2:
            continue
        total += (
            rho2 ** ((2 * n - s - order) // 2)
            * beta ** (l - s)
            / (
                mp.factorial(n - (s + order) // 2)
                * mp.factorial(n - (s - order) // 2)
                * mp.factorial(l - s)
                * mp.factorial(s - n)
            )
        )
    return total * azimuth**order


def ewald(
    l: int, m: int, k: Any, kpar: Any, period: Any, shift: Any, digits: int
) -> Any:
    """The chain sum sum_n e^(i kpar n a) h_l(k |X_n|) Y_lm(X_n), X_n = -r - n a z,
    by the Ewald method at a real split chosen to balance its two cancellations."""
    x, y, z = shift
    size = abs(complex(k))
    rho = max(float(mp.hypot(x, y)), 0.2 * float(period))
    eta = mp.mpf(min((2 * size**2 * rho**2) ** -0.25, 1.5 / size))
    t_size = size**2 * float(x * x + y * y) * float(eta) ** 2
    loss = (t_size + 1 / (2 * float(eta) ** 2)) / math.log(10)
    with mp.workdps(int(digits + 15 + loss)):
        target = mp.mpf(10) ** (-(digits + 8))
        order = abs(m)
        norm = mp.sqrt(
            (2 * l + 1)
            / (4 * mp.pi)
            * mp.factorial(l - order)
            / mp.factorial(l + order)
        )
        if m < 0:
            norm *= mp.factorial(l + order) / mp.factorial(l - order)
        real = mp.mpc(0)
        for side in (1, -1):
            quiet, n = 0, 0 if side == 1 else -1
            while quiet < 3:
                point = (-x, -y, -z - n * period)
                distance = mp.sqrt(sum(c * c for c in point))
                term = (
                    -1j
                    * mp.sqrt(2 / mp.pi)
                    * k**l
                    * norm
                    * solid(l, m, point)
                    * kambe(2 * l, k * distance, eta)
                    * mp.expjpi(kpar * n * period / mp.pi)
                )
                real += term
                quiet = (
                    quiet + 1
                    if abs(term) <= target and abs(k * distance * eta) > 3
                    else 0
                )
                n += side
        t = k**2 * (x * x + y * y) * eta**2
        rho2 = k**2 * (x * x + y * y)
        azimuth = -k * mp.mpc(x, y if m >= 0 else -y)
        prefactor = (
            -1j
            * mp.sqrt((2 * l + 1) / mp.pi)
            * (-1j) ** (l - m)
            * mp.sqrt(mp.factorial(l + m) * mp.factorial(l - m))
            / (2 * period * k)
        )
        below = mp.im(k) == 0
        reciprocal = mp.mpc(0)
        for side in (1, -1):
            quiet, g = 0, 0 if side == 1 else -1
            while quiet < 3:
                q = kpar + g * 2 * mp.pi / period
                beta = q / k
                v = (beta**2 - 1) / (2 * eta**2)
                total = mp.mpc(0)
                for n in range(order, l + 1):
                    weight = polynomial(l, m, n, rho2, beta, azimuth)
                    if weight != 0:
                        total += (eta**2 / 2) ** n * reduced(n, v, t, below) * weight
                term = prefactor * total * mp.expjpi(-q * z / mp.pi)
                reciprocal += term
                quiet = quiet + 1 if abs(term) <= target and mp.re(v) > 5 else 0
                g += side
        return +(real + reciprocal)


# Kambe integral at the arguments of the lattice sums -----------------------------

LATTICE_V = [1e-12, 1e-8, 1e-4, 0.01, 0.1, 0.5, 1, 3, 10]
# Each phase in double precision; -e^(-1e-8 i) as e^(i (pi - 1e-8)), whose rounding
# the keys of the table carry.
LATTICE_V_PHASES = [
    1,
    -1,
    cmath.exp(1j * (math.pi - 1e-8)),
    cmath.exp(0.5j),
    cmath.exp(-2j),
]
LATTICE_W = [0.5, 1, 1.5, 2, 3, 4, 5, 6]
LATTICE_W_PHASES = [1, cmath.exp(1e-8j), cmath.exp(0.2j), cmath.exp(-0.3j)]
# Real w that add order -1 at |v| <= 1e-8, and the v they leave out.
LATTICE_EXTRA_W = (0.5, 2, 5)
LATTICE_EXTRA_SKIP = 1e-8 * cmath.exp(0.5j)


def kambe_lattice_rows() -> list[Key]:
    """One order per (v, w), cycling through -16..-1, plus order -1 at small v."""
    rows: list[Key] = []
    pair = 0
    for w_size in LATTICE_W:
        for w_phase in LATTICE_W_PHASES:
            w = w_size * w_phase
            # numpy's complex division, whose last bits the keys carry.
            eta = complex(np.complex128(-1j) / np.complex128(w))
            for v_size in LATTICE_V:
                for v_phase in LATTICE_V_PHASES:
                    if v_phase == -1 and w_phase != 1:
                        continue
                    v = v_size * v_phase
                    x = cmath.sqrt(-2 * v * w * w)
                    if x.imag == 0:
                        x = complex(x.real, 1e-100)
                    n = -16 + pair % 16
                    pair += 1
                    key = (x.real, x.imag, eta.real, eta.imag)
                    rows.append((n, *key))
                    extra = (
                        w_phase == 1
                        and w_size in LATTICE_EXTRA_W
                        and v_size <= 1e-8
                        and v != LATTICE_EXTRA_SKIP
                    )
                    if extra and n != -1:
                        rows.append((-1, *key))
    return rows


def kambe_on_sheet(n: int, x: Any, eta: Any) -> Any:
    """I_n(x, eta) by the gamma series on the sheet the lattice sums take: odd orders
    on the principal branch of ln a, even orders, which are even in x, on the branch
    sqrt(a) = x eta / sqrt 2 with Re x >= 0."""
    if n % 2 == 0 and mp.re(x) < 0:
        x = -x
    value = kambe(n, x, eta)
    a = (x * eta) ** 2 / 2
    if n % 2 or mp.re(mp.sqrt(2 * a) * mp.conj(x * eta)) >= 0:
        return value
    # On the other branch of sqrt(a), each Gamma(d - j, a) / a^(d - j) of the series
    # drops 2 Gamma(d - j) / a^(d - j) (d is a half-integer); the loop sums these.
    d, c = mp.mpf(n + 1) / 2, 1 / (2 * eta**2)
    term = 2 * mp.gamma(d) * mp.power(a, -d)
    total, peak, j = term, abs(term), 0
    while j <= abs(c * a) or abs(term) > mp.mpf(10) ** -(mp.mp.dps + 10) * peak:
        j += 1
        term *= c * a / (j * (d - j))
        total += term
        peak = max(peak, abs(term))
    return value - eta ** (n + 1) / 2 * total


def kambe_lattice(key: Key) -> Values:
    """The gamma series at 70 digits, confirmed at 100 digits."""
    n, xr, xi, er, ei = key
    values = []
    for digits in (70, 100):
        with mp.workdps(digits):
            values.append(kambe_on_sheet(int(n), mp.mpc(xr, xi), mp.mpc(er, ei)))
    with mp.workdps(100):
        if abs(values[0] - values[1]) > mp.mpf(10) ** -48 * abs(values[1]):
            raise ArithmeticError(f"I_{n}: 70 and 100 digits disagree")
    return [complex(values[0])]


# Off-axis chain sums --------------------------------------------------------------


def chain_rows() -> list[Key]:
    return [
        (l, m, complex(k).real, complex(k).imag, kpar, period, *shift)
        for l, m, k, kpar, period, shift in CHAIN_ROWS
    ]


def chain(key: Key) -> Values:
    l, m, kr, ki, kpar, period, x, y, z = key
    k = mp.mpc(kr, ki) if ki != 0 else mp.mpf(kr)
    l, m = int(l), int(m)
    kpar, period = mp.mpf(kpar), mp.mpf(period)
    shift = [mp.mpf(c) for c in (x, y, z)]
    value = ewald(l, m, k, kpar, period, shift, CHAIN_DIGITS)
    with mp.workdps(2 * CHAIN_DIGITS + 20):
        h = mp.mpf(10) ** (-(CHAIN_DIGITS // 2 + 8))
        steps = [(h, 0, (0, 0, 0), 0), (0, h, (0, 0, 0), 0)]
        steps += [
            (0, 0, tuple(h if j == i else 0 for j in range(3)), 0) for i in range(3)
        ]
        steps.append((0, 0, (0, 0, 0), h))

        def at(sign: int, step: Any) -> Any:
            dk, dkpar, dr, da = step
            moved = [shift[i] + sign * dr[i] for i in range(3)]
            return ewald(
                l,
                m,
                k + sign * dk,
                kpar + sign * dkpar,
                period + sign * da,
                moved,
                CHAIN_DIGITS + 25,
            )

        derivatives = [(at(1, step) - at(-1, step)) / (2 * h) for step in steps]
    return [complex(value), *(complex(d) for d in derivatives)]


# Ewald lattice sums -------------------------------------------------------------
# A sum S = sum_R e^(i kpar.R) w(-r - R) over the lattice points R of outgoing waves
# w = h_l(k |X|) Y_lm(X) (spherical) or H_m(k |X|) e^(i m phi) (cylindrical), in the
# lattice frame of lattice::sum: 1D spherical lattices along z, 1D cylindrical ones
# along x, 2D ones in the xy plane.


@dataclass(frozen=True)
class LatticeSum:
    """One row of lattice_sums.txt: `wave dim re(k) im(k) x y z re(eta) im(eta) rows
    bloch part jet tolerance`, whose Bloch vector `bloch` is the field `kpar`."""

    spherical: bool
    l: int
    m: int
    dim: int
    k: complex
    shift: tuple[float, float, float]
    eta: complex
    vectors: list[list[float]]
    kpar: list[float]
    part: str
    jet: bool


def lattice_sum(key: Key) -> LatticeSum:
    spherical = key[0] == "s"
    l, m = (int(key[1]), int(key[2])) if spherical else (0, int(key[1]))
    fields = key[3:] if spherical else key[2:]
    dim = int(fields[0])
    x = [float(field) for field in fields[1:-3]]
    rows = [x[7 + dim * i : 7 + dim * (i + 1)] for i in range(dim)]
    return LatticeSum(
        spherical,
        l,
        m,
        dim,
        complex(x[0], x[1]),
        (x[2], x[3], x[4]),
        complex(x[5], x[6]),
        rows,
        x[7 + dim * dim : 7 + dim * dim + dim],
        fields[-3],
        fields[-2] == "1",
    )


def lattice_frame(case: LatticeSum, vector: Sequence[Any]) -> list[Any]:
    """A vector of the lattice's own components in Cartesian coordinates."""
    padded = [*vector, *[mp.mpf(0)] * (3 - len(vector))]
    return (
        [padded[1], padded[2], padded[0]]
        if case.spherical and case.dim == 1
        else padded
    )


def harmonic_norm(l: int, m: int) -> Any:
    """The factor of Y_lm over the solid harmonic r^l P_l^m e^(i m phi) of `solid`."""
    order = abs(m)
    norm = mp.sqrt(
        (2 * l + 1) / (4 * mp.pi) * mp.factorial(l - order) / mp.factorial(l + order)
    )
    if m < 0:
        norm *= mp.factorial(l + order) / mp.factorial(l - order)
    return norm


def spherical_hankel(l: int, x: Any) -> Any:
    """h_l^(1)(x) by its finite sum."""
    total = mp.mpc(0)
    for j in range(l + 1):
        weight = mp.factorial(l + j) / (mp.factorial(j) * mp.factorial(l - j))
        total += (1j / (2 * x)) ** j * weight
    return (-1j) ** (l + 1) * mp.exp(1j * x) / x * total


def complex_solid(l: int, m: int, x: Any, y: Any, z: Any) -> Any:
    """`solid` at complex coordinates, which plane waves of complex direction need."""
    square = x * x + y * y + z * z
    order = abs(m)
    value = mp.mpc(1)
    for j in range(1, order + 1):
        value *= -(2 * j - 1) * (x + 1j * y if m >= 0 else x - 1j * y)
    previous = mp.mpc(0)
    for degree in range(order + 1, l + 1):
        a = mp.mpf(2 * degree - 1) / (degree - order)
        b = mp.mpf(degree + order - 1) / (degree - order)
        value, previous = a * z * value - b * square * previous, value
    if m < 0:
        value *= (-1) ** order * mp.factorial(l - order) / mp.factorial(l + order)
    return value


def outgoing(k: Any, square: Any) -> Any:
    """sqrt(k^2 - q^2) with Im >= 0, and Re > 0 where it is real."""
    root = mp.sqrt(k * k - square)
    return -root if mp.im(root) < 0 or (mp.im(root) == 0 and mp.re(root) < 0) else root


def points(dim: int, shell: int) -> list[tuple[int, ...]]:
    """The integer points of Chebyshev norm `shell`."""
    return [
        n
        for n in itertools.product(range(-shell, shell + 1), repeat=dim)
        if max(map(abs, n)) == shell
    ]


def shell_sum(dim: int, term: Callable[[tuple[int, ...]], Any], digits: int) -> Any:
    """Sum `term` over integer shells until three in a row add below 10^-digits."""
    total, shell, quiet = mp.mpc(0), 0, 0
    while quiet < 3:
        part = mp.fsum(term(n) for n in points(dim, shell))
        total += part
        small = abs(part) < mp.mpf(10) ** -digits * max(abs(total), 1)
        quiet = quiet + 1 if small else 0
        shell += 1
    return total


def direct_sum(case: LatticeSum, k: Any, kpar: Sequence[Any], real: Any = None) -> Any:
    """The image sum, or with a split `real` the real-space Ewald part."""
    vectors = [lattice_frame(case, [mp.mpf(c) for c in row]) for row in case.vectors]
    kpar_xyz = lattice_frame(case, kpar)
    r = [mp.mpf(c) for c in case.shift]
    l, m = case.l, case.m

    def term(n: tuple[int, ...]) -> Any:
        point = [sum(n[i] * vectors[i][c] for i in range(case.dim)) for c in range(3)]
        x = [-r[c] - point[c] for c in range(3)]
        phase = mp.expj(sum(kpar_xyz[c] * point[c] for c in range(3)))
        if case.spherical:
            distance = mp.sqrt(sum(c * c for c in x))
            harmonic = harmonic_norm(l, m) * solid(l, m, x)
            if real is None:
                return (
                    phase * spherical_hankel(l, k * distance) * harmonic / distance**l
                )
            factor = -1j * mp.sqrt(2 / mp.pi) * k**l * harmonic
            return phase * factor * kambe_on_sheet(2 * l, k * distance, real)
        radius = mp.sqrt(x[0] ** 2 + x[1] ** 2)
        if real is None:
            azimuth = (x[0] + 1j * x[1]) / radius
            return phase * mp.hankel1(m, k * radius) * azimuth**m
        planar = x[0] + 1j * (x[1] if m >= 0 else -x[1])
        sign = -1 if m < 0 and m % 2 else 1
        factor = -2j / mp.pi * sign * (k * planar) ** abs(m)
        return phase * factor * kambe_on_sheet(2 * abs(m) - 1, k * radius, real)

    return shell_sum(case.dim, term, mp.mp.dps + 3)


def spectral_sum(case: LatticeSum, k: Any, kpar: Sequence[Any]) -> Any:
    """The plane-wave (Weyl) series of a 2D spherical or 1D cylindrical sum off the
    lattice plane or axis, with reciprocal vectors from the exact lattice rows."""
    x, y, z = (mp.mpf(c) for c in case.shift)
    if not case.spherical:
        period, side = mp.mpf(case.vectors[0][0]), 1 if y < 0 else -1

        def order(n: tuple[int, ...]) -> Any:
            q = kpar[0] + 2 * mp.pi * n[0] / period
            gamma = outgoing(k, q * q)
            direction = ((q + 1j * side * gamma) / k) ** case.m
            return mp.expj(-q * x + gamma * abs(y)) / gamma * direction

        total = shell_sum(1, order, mp.mp.dps + 5)
        return 2 / period * (-1j) ** case.m * total
    (a, b), (c, d) = ([mp.mpf(e) for e in row] for row in case.vectors)
    area = a * d - b * c
    turn = 2 * mp.pi / area
    reciprocal = [[turn * d, -turn * c], [-turn * b, turn * a]]
    side = 1 if z < 0 else -1
    norm = harmonic_norm(case.l, case.m)

    def wave(n: tuple[int, ...]) -> Any:
        qx = kpar[0] + n[0] * reciprocal[0][0] + n[1] * reciprocal[1][0]
        qy = kpar[1] + n[0] * reciprocal[0][1] + n[1] * reciprocal[1][1]
        gamma = outgoing(k, qx * qx + qy * qy)
        harmonic = (
            norm * complex_solid(case.l, case.m, qx, qy, side * gamma) / k**case.l
        )
        return mp.expj(-(qx * x + qy * y) + gamma * abs(z)) / gamma * harmonic

    total = shell_sum(2, wave, mp.mp.dps + 5)
    return 2 * mp.pi / (area * k * 1j**case.l) * total


def axis_sum(case: LatticeSum) -> Any:
    """A 1D spherical sum of order 0 on the lattice axis by Lerch transcendents: the
    images on either side of the shift add e^(i k d) / d^s along the axis."""
    k, z = mp.mpmathify(case.k), mp.mpf(case.shift[2])
    period, kpar, l = mp.mpf(case.vectors[0][0]), mp.mpf(case.kpar[0]), case.l
    first = int(mp.floor(-z / period)) + 1
    above = z / period + first
    below = 1 - above
    total = mp.mpc(0)
    for j in range(l + 1):
        weight = mp.factorial(l + j) / (mp.factorial(j) * mp.factorial(l - j))
        power = j + 1
        ahead = mp.lerchphi(mp.expj((kpar + k) * period), power, above)
        ahead *= mp.expj(kpar * period * first + k * period * above)
        behind = mp.lerchphi(mp.expj((k - kpar) * period), power, below)
        behind *= mp.expj(kpar * period * (first - 1) + k * period * below)
        scale = (1j / 2) ** j * weight / (k * period) ** power
        total += scale * ((-1) ** l * ahead + behind)
    return (-1j) ** (l + 1) * mp.sqrt((2 * l + 1) / (4 * mp.pi)) * total


def lattice_sum_row(key: Key) -> Values:
    """The reference of a row by the recipe of its section in the table's header."""
    case = lattice_sum(key)
    k = case.k
    kpar = [mp.mpf(c) for c in case.kpar]
    if case.part == "real":
        with mp.workdps(60 if case.dim == 1 else 40):
            eta = mp.mpmathify(case.eta)
            return [complex(direct_sum(case, mp.mpmathify(k), kpar, eta))]
    if case.part == "reciprocal":
        with mp.workdps(40):
            k_mp, eta = mp.mpmathify(k), mp.mpmathify(case.eta)
            full = spectral_sum(case, k_mp, kpar)
            return [complex(full - direct_sum(case, k_mp, kpar, eta))]
    if case.spherical and case.dim == 1:
        with mp.workdps(40):
            return [complex(axis_sum(case))]
    if k.imag > 0:
        with mp.workdps(30):
            return [complex(direct_sum(case, mp.mpmathify(k), kpar))]
    with mp.workdps(30 if case.spherical else 40):
        k_mp = mp.mpmathify(k)
        value = spectral_sum(case, k_mp, kpar)
        if not (case.jet and case.spherical):
            return [complex(value)]

        def in_k(t: Any) -> Any:
            return spectral_sum(case, t, kpar)

        def in_kpar(t: Any) -> Any:
            return spectral_sum(case, k_mp, [t, *kpar[1:]])

        dk, dq = mp.diff(in_k, k_mp), mp.diff(in_kpar, kpar[0])
        return [complex(value), complex(dk), complex(dq)]


def lattice_sum_rows(path: Path) -> list[Key]:
    """The cases of the table at `path`, which are chosen by hand."""
    lines = path.read_text().splitlines()
    return [tuple(line.split(":")[0].split()) for line in lines if line[:1] != "#"]


# Tables --------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    """A reference table: its file, header, rows and how to compute one row."""

    path: Path
    # Comment lines above the rows; None for a table that keeps its comments between
    # its rows, whose generator rewrites the values of the committed rows in place.
    header: str | None
    # The keys of the rows, given the table that holds the cases chosen by hand.
    rows: Callable[[Path], list[Key]]
    compute: Callable[[Key], Values]
    # Integer key fields, written without a decimal point.
    integers: tuple[int, ...]
    # Largest relative error of a recomputed value that --check accepts.
    tolerance: float
    omit: Callable[[Values], bool] = lambda _: False


# Doubles that round the same high-precision value differ by at most one unit in
# the last place of each part.
ROUNDING = 2.0**-52

TABLES = {
    "incgamma": Table(
        REFERENCES / "incgamma.txt",
        """\
# Upper incomplete gamma Gamma(n, z): `n re(z) im(z): re im`, a signed zero im(z)
# selects the side of the negative real axis. Generated with mpmath 1.3.0 at 40
# digits: mp.gammainc(n, z) on the upper side, conjugated for im(z) = -0.0.
# degrees [-8, -7.5, -6.5, -5, -4.5, -3, -2.5, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 3, 4.5, 7.5, 8]
# re(z) [-60, -36, -20, -9, -3, -0.5, 0, 1, 4, 4.5, 12, 36, 60]; im(z) [0.0, 1e-06, 0.4, 3, 10, 20] and -0.0 for re(z) < 0
""",
        lambda _: incgamma_rows(),
        incgamma,
        (),
        ROUNDING,
    ),
    "kambe": Table(
        REFERENCES / "kambe.txt",
        """\
# Kambe integral I_n(z, eta) = int_eta^inf t^n exp(-z^2 t^2 / 2 + 1 / (2 t^2)) dt:
# `n re(z) im(z) eta: re im`. Generated with mpmath 1.3.0 at 40 digits by tanh-sinh
# quadrature, confirmed by Gauss-Legendre quadrature to 1e-15;
# values below 1e-30 are omitted. Subdivided on the decay scale 1 / max(|z|^2 eta, eta^-3) of the integrand at eta.
# orders [-14, -11, -8, -5, -4, -3, -2, -1, 0, 1, 4, 7]; z [(0.3, 0.1), (1.2, 0.0), (2.5, 0.4), (4.0, -0.5)]; eta [0.5, 1.0, 1.7, 3.0]
""",
        lambda _: kambe_rows(),
        kambe_quadrature,
        (0,),
        ROUNDING,
        kambe_omit,
    ),
    "kambe-lattice": Table(
        REFERENCES / "kambe_lattice.txt",
        """\
# Kambe integral I_n(x, eta) at the arguments that the reduced Kambe integral F_n of
# the lattice sums passes to it: x = sqrt(-2 v w^2), with x.im = 1e-100 where it is
# real, and eta = -i/w, for v in {1e-12, 1e-8, 1e-4, 0.01, 0.1, 0.5, 1, 3, 10} times
# {1, -1, -e^(-1e-8 i), e^(0.5 i), e^(-2i)} and w in {0.5, 1, 1.5, 2, 3, 4, 5, 6} times
# {1, e^(1e-8 i), e^(0.2 i), e^(-0.3 i)}; v on the negative real axis with complex w is
# left out (a = v then lies on the branch cut). One order per (v, w), cycling through
# -16..-1, and order -1 also at |v| <= 1e-8 for real w in {0.5, 2, 5}, except at
# v = 1e-8 e^(0.5 i). `n re(x) im(x) re(eta) im(eta): re im`.
# Generated with mpmath 1.3.0 from the gamma series at 70 digits (its terms cancel by
# up to e^(w^2), 16 digits at w = 6), confirmed at 100 digits to 1e-48 and, for the
# 264 rows with Re a > 0 on the principal branch, by 30-digit quadrature of
# eta^(n+1)/2 int_1^inf s^((n-1)/2) e^(-a s + c/s) ds, a = (x eta)^2/2, c = 1/(2 eta^2).
# Even orders take the branch sqrt(a) = x eta / sqrt 2 with Re x >= 0.
""",
        lambda _: kambe_lattice_rows(),
        kambe_lattice,
        (0,),
        ROUNDING,
    ),
    "lattice-chain": Table(
        REFERENCES / "lattice_chain.txt",
        """\
# 1D spherical lattice sums S_lm (a chain of period a along z) off the axis and their
# first derivatives, `l m re(k) im(k) kpar a x y z: S dS/dk dS/dkpar dS/dx dS/dy dS/dz dS/da`
# as re im pairs (dS/dk holomorphic), kpar the Bloch wave number. Generated by
# scripts/generate_references.py lattice-chain with mpmath 1.3.0: the Ewald method at
# 30 digits with a real split chosen for the precision it needs, independent of the
# double-precision split and of the spectral series; derivatives by central
# differences at 55 digits. Agrees with the spectral series in mpmath to 1e-17 and
# with absolutely convergent image sums for Im k > 0.
""",
        lambda _: chain_rows(),
        chain,
        (0, 1),
        ROUNDING,
    ),
    "lattice-sums": Table(
        REFERENCES / "lattice_sums.txt",
        None,
        lattice_sum_rows,
        lattice_sum_row,
        (),
        ROUNDING,
    ),
}


def format_key(table: Table, key: Key) -> str:
    return " ".join(
        x if isinstance(x, str) else str(int(x)) if i in table.integers else repr(x)
        for i, x in enumerate(key)
    )


def parse_key(table: Table, text: str) -> Key:
    fields = text.split()
    return tuple(fields) if table.header is None else tuple(map(float, fields))


def format_row(table: Table, key: Key, values: Values) -> str:
    numbers = " ".join(f"{c.real!r} {c.imag!r}" for c in values)
    return f"{format_key(table, key)}: {numbers}"


def compute_row(name: str, key: Key) -> Values:
    return TABLES[name].compute(key)


def compute(name: str, keys: Sequence[Key], workers: int) -> list[Values]:
    """The values of every key, with progress on stderr."""
    results: list[Values] = []
    with ProcessPoolExecutor(workers) as pool:
        for values in pool.map(partial(compute_row, name), keys):
            results.append(values)
            if len(results) % max(1, len(keys) // 10) == 0:
                print(f"{name}: {len(results)}/{len(keys)} rows", file=sys.stderr)
    return results


def relative_error(actual: Values, expected: Values) -> float:
    """The largest error of a value relative to its modulus (absolute at zero)."""
    return max(
        abs(a - e) / abs(e) if e != 0 else abs(a)
        for a, e in zip(actual, expected, strict=True)
    )


def generate(name: str, output: Path, workers: int) -> None:
    table = TABLES[name]
    keys = table.rows(table.path)
    rows = [
        format_row(table, key, values)
        for key, values in zip(keys, compute(name, keys, workers), strict=True)
        if not table.omit(values)
    ]
    if table.header is None:
        lines = table.path.read_text().splitlines()
        new = iter(rows)
        rows = [line if line.startswith("#") else next(new) for line in lines]
    output.write_text((table.header or "") + "".join(row + "\n" for row in rows))


def check(name: str, path: Path, workers: int) -> bool:
    """Recompute the rows of a committed table; True when all are within tolerance."""
    table = TABLES[name]
    lines = path.read_text().splitlines()
    header = "".join(line + "\n" for line in lines if line.startswith("#"))
    rows = [line.split(":") for line in lines if not line.startswith("#")]
    texts = [key.strip() for key, _ in rows]
    keys = [parse_key(table, key) for key in texts]
    expected = [
        [complex(float(a), float(b)) for a, b in zip(v[::2], v[1::2], strict=True)]
        for v in (values.split() for _, values in rows)
    ]
    grid = table.rows(path)
    known = set(texts)
    left_out = [key for key in grid if format_key(table, key) not in known]
    order = [format_key(table, key) for key in grid if format_key(table, key) in known]
    actual = compute(name, keys + left_out, workers)
    ok = True
    if table.header is not None and header != table.header:
        print(f"{name}: the header differs from the generator's")
        ok = False
    if order != texts:
        print(f"{name}: the keys differ from the generator's rows")
        ok = False
    omitted = actual[len(keys) :]
    if not all(table.omit(values) for values in omitted):
        print(f"{name}: a row the table leaves out is above its threshold")
        ok = False
    errors = [
        relative_error(a, e) for a, e in zip(actual[: len(keys)], expected, strict=True)
    ]
    worst = max(range(len(errors)), key=errors.__getitem__)
    within = errors[worst] <= table.tolerance
    outside = sum(error > table.tolerance for error in errors)
    print(
        f"{name}: {len(keys)} rows ({len(left_out)} left out by rule), "
        f"{sum(error == 0 for error in errors)} identical, {outside} outside the "
        f"tolerance {table.tolerance:.2e}; worst relative error {errors[worst]:.2e} "
        f"at `{texts[worst]}`: {'ok' if within else 'FAILED'}"
    )
    return ok and within


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table", choices=sorted(TABLES))
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute the rows of the table and compare; write nothing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="table to write, or to read with --check (default: the committed one)",
    )
    parser.add_argument("--workers", type=int, default=2, help="worker processes")
    args = parser.parse_args()
    path = args.output or TABLES[args.table].path
    if args.check:
        sys.exit(0 if check(args.table, path, args.workers) else 1)
    generate(args.table, path, args.workers)


if __name__ == "__main__":
    main()
