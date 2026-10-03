"""Table-driven NumPy boundary contract for every native ufunc.

Every case draws valid, bounded operands with a leading loop axis. The inner
loops evaluate deterministic elementwise kernels, so each property is exact:
results must not depend on the parallel partition, memory layout, stride-0
broadcasting, alignment, output aliasing, or which dtype loop served a
value-preserving cast.
"""

# ruff: noqa: E741 - upstream degree argument l
from __future__ import annotations

import ast
import functools
import os
import pickle
import re
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

# The native ufuncs are the layer under test.
from treams_rs import _native, cw, lattice, misc, pw, special, sw

if TYPE_CHECKING:
    from collections.abc import Callable

    Factory = Callable[[np.random.Generator, int], list[np.ndarray]]

pytestmark = pytest.mark.interface

# Above every elementwise parallel threshold of the native loops.
LARGE = 2048


@dataclass(frozen=True)
class Case:
    make: Factory
    size: int = LARGE
    # Operands before this index are broadcast together (mode labels stay valid).
    split: int | None = None
    # Operand index sets whose stride-0 broadcast selects a precomputed plan.
    fixed: tuple[tuple[int, ...], ...] = ()
    # Domain valid for every registered loop (defaults to real parts of make).
    real: Factory | None = None
    loops: dict[str, str] = field(default_factory=dict)


def _polar(rng, n, low, high):
    return rng.uniform(low, high, n) * np.exp(1j * rng.uniform(-1.3, 1.3, n))


def _complex(rng, n, real, imag=(0.0, 0.0)):
    return rng.uniform(*real, n) + 1j * rng.uniform(*imag, n)


def _mode(rng, n, low=1, high=6):
    l = rng.integers(low, high + 1, n)
    return l, rng.integers(-l, l + 1)


def _pol(rng, n):
    return rng.integers(0, 2, n)


def _angles(rng, n, count):
    return [rng.uniform(-3, 3, n) for _ in range(count)]


def _bessel(spherical):
    def make(rng, n):
        order = rng.integers(0, 9, n) if spherical else rng.uniform(-4, 6, n)
        return [order.astype(float), _polar(rng, n, 0.05 if spherical else 0.3, 8)]

    return Case(make)


def _angular(legendre):
    def make(rng, n):
        l, m = _mode(rng, n, 0, 8)
        z = _complex(rng, n, (-0.95, 0.95), (-0.4, 0.4))
        labels = [m, l] if legendre else [l, m]
        return [labels[0].astype(float), labels[1].astype(float), z]

    return Case(make)


def _wigner(euler):
    def make(rng, n):
        l = rng.integers(0, 9, n)
        m, k = rng.integers(-l, l + 1), rng.integers(-l - 1, l + 2)
        angles = [
            _complex(rng, n, (-3, 3), (-0.3, 0.3)) for _ in range(3 if euler else 1)
        ]
        return [l.astype(float), m.astype(float), k.astype(float), *angles]

    return Case(make)


def _wigner3j(rng, n):
    j1, j2 = rng.integers(0, 7, n), rng.integers(0, 7, n)
    j3 = rng.integers(np.abs(j1 - j2), j1 + j2 + 1)
    m1, m2 = rng.integers(-j1, j1 + 1), rng.integers(-j2, j2 + 1)
    m3 = np.where(rng.random(n) < 0.8, -m1 - m2, rng.integers(-j3, j3 + 1))
    return [v.astype(float) for v in (j1, j2, j3, m1, m2, m3)]


def _lattice(prefix, family):
    spherical = family.startswith("sw")
    dim = int(family[2])
    shifted = family.endswith("shift")
    ordered = spherical and (dim > 1 or shifted)
    labels = 2 if ordered else 1

    def make(rng, n):
        l, m = _mode(rng, n, 1, 2) if spherical else (rng.integers(-2, 3, n), None)
        scale = rng.uniform(0.95, 1.05, n)
        q = 0.13 * scale if dim == 1 else np.linspace(0.1, 0.2, dim) * scale[:, None]
        a = (
            1.7 * scale
            if dim == 1
            else np.diag(np.linspace(1.5, 1.7, dim)) * scale[:, None, None]
        )
        width = 3 if spherical and (dim == 3 or shifted) else 2
        r = (
            rng.uniform(0.05, 0.2, n)
            if dim == 1 and not shifted
            else rng.uniform(0.05, 0.2, (n, width))
        )
        last = (
            rng.integers(0, 3, n)
            if prefix == "dsum"
            else _complex(rng, n, (0.8, 1.1), (-0.05, 0.05))
        )
        k = _complex(rng, n, (2.0, 2.3), (0.15, 0.25))
        modes = [l, m] if ordered else [l]
        return [*(v.astype(float) for v in modes), k, q, a, r, last]

    return Case(make, size=16, fixed=((labels + 1, labels + 2),))


def _vsh(rng, n):
    l, m = _mode(rng, n)
    return [l, m, _complex(rng, n, (-3, 3), (-0.3, 0.3)), rng.uniform(-3, 3, n)]


def _vsw(polarized):
    def make(rng, n):
        l, m = _mode(rng, n)
        theta = _complex(rng, n, (-3, 3), (-0.3, 0.3))
        args = [l, m, _polar(rng, n, 0.1, 5), theta, rng.uniform(-3, 3, n)]
        return [*args, _pol(rng, n)] if polarized else args

    return Case(make, fixed=((0, 1, 2),))


def _vcw(normalized, polarized):
    def make(rng, n):
        args = [
            rng.uniform(-1, 1, n),
            rng.integers(-4, 5, n),
            _polar(rng, n, 0.1, 5),
            rng.uniform(-3, 3, n),
            rng.uniform(-2, 2, n),
        ]
        if normalized:
            args.append(_complex(rng, n, (1.2, 2.0), (0, 0.2)))
        if polarized:
            args.append(_pol(rng, n))
        return args

    return Case(make)


def _plane_vector(rng, n):
    return [_complex(rng, n, (-1, 1), (-0.1, 0.1)) for _ in range(3)]


def _vpw(polarized):
    def make(rng, n):
        args = [*_plane_vector(rng, n), *(rng.uniform(-2, 2, n) for _ in range(3))]
        return [*args, _pol(rng, n)] if polarized else args

    fixed = (0, 1, 2, 6) if polarized else (0, 1, 2)
    return Case(make, fixed=(fixed,))


def _axial(rng, n):
    kz = rng.uniform(-1, 1, n)
    return kz, np.where(rng.random(n) < 0.8, kz, rng.uniform(-1, 1, n))


def _tl_vcw(rng, n):
    kz, qz = _axial(rng, n)
    return [
        kz,
        rng.integers(-4, 5, n),
        qz,
        rng.integers(-4, 5, n),
        _polar(rng, n, 0.3, 5),
        rng.uniform(-3, 3, n),
        rng.uniform(-2, 2, n),
    ]


def _sw_rotate(rng, n):
    l, m = _mode(rng, n)
    return [
        np.where(rng.random(n) < 0.8, l, l + 1),
        rng.integers(-l, l + 1),
        _pol(rng, n),
        l,
        m,
        _pol(rng, n),
        *_angles(rng, n, 3),
    ]


def _cw_rotate(rng, n):
    kz, qz = _axial(rng, n)
    m = rng.integers(-4, 5, n)
    mu = np.where(rng.random(n) < 0.8, m, rng.integers(-4, 5, n))
    return [kz, mu, _pol(rng, n), qz, m, _pol(rng, n), rng.uniform(-3, 3, n)]


def _sw_translate(rng, n):
    l, m = _mode(rng, n, 1, 5)
    lam, mu = _mode(rng, n, 1, 5)
    return [
        lam,
        mu,
        _pol(rng, n),
        l,
        m,
        _pol(rng, n),
        _polar(rng, n, 0.3, 5),
        *_angles(rng, n, 2),
    ]


def _cw_translate(rng, n):
    kz, qz = _axial(rng, n)
    return [
        kz,
        rng.integers(-4, 5, n),
        _pol(rng, n),
        qz,
        rng.integers(-4, 5, n),
        _pol(rng, n),
        _polar(rng, n, 0.3, 5),
        rng.uniform(-3, 3, n),
        rng.uniform(-2, 2, n),
    ]


def _tl_vsw(rng, n):
    l, m = _mode(rng, n, 1, 5)
    lam, mu = _mode(rng, n, 1, 5)
    return [
        lam,
        mu,
        l,
        m,
        _polar(rng, n, 0.3, 5),
        _complex(rng, n, (-3, 3), (-0.2, 0.2)),
        rng.uniform(-3, 3, n),
    ]


def _sw_periodic_to_pw(rng, n):
    l, m = _mode(rng, n)
    kx, ky = rng.uniform(-0.6, 0.6, n), rng.uniform(-0.6, 0.6, n)
    kz = _complex(rng, n, (0.5, 1.5), (0.0, 0.2)) * rng.choice([-1, 1], n)
    return [kx, ky, kz, _pol(rng, n), l, m, _pol(rng, n), rng.uniform(1, 3, n)]


def _cw_periodic_to_pw(rng, n):
    kx, kz = rng.uniform(-0.6, 0.6, n), rng.uniform(-0.6, 0.6, n)
    ky = _complex(rng, n, (0.5, 1.5), (0.0, 0.2)) * rng.choice([-1, 1], n)
    qz = np.where(rng.random(n) < 0.8, kz, rng.uniform(-0.6, 0.6, n))
    m = rng.integers(-4, 5, n)
    return [kx, ky, kz, _pol(rng, n), qz, m, _pol(rng, n), rng.uniform(1, 3, n)]


def _sw_periodic_to_cw(rng, n):
    l, m = _mode(rng, n)
    return [
        rng.uniform(-0.8, 0.8, n),
        rng.integers(-4, 5, n),
        _pol(rng, n),
        l,
        m,
        _pol(rng, n),
        _complex(rng, n, (1.2, 2.0), (0, 0.2)),
        rng.uniform(4, 6, n),
    ]


def _pw_to_sw(rng, n):
    l, m = _mode(rng, n)
    return [l, m, _pol(rng, n), *_plane_vector(rng, n), _pol(rng, n)]


def _pw_to_cw(rng, n):
    kz = rng.uniform(-0.8, 0.8, n)
    qz = np.where(rng.random(n) < 0.8, kz, rng.uniform(-0.8, 0.8, n))
    return [
        kz,
        rng.integers(-4, 5, n),
        _pol(rng, n),
        rng.uniform(-1, 1, n),
        _complex(rng, n, (-1, 1), (-0.1, 0.1)),
        qz,
        _pol(rng, n),
    ]


def _cw_to_sw(rng, n):
    l, m = _mode(rng, n)
    return [
        l,
        m,
        _pol(rng, n),
        rng.uniform(-0.8, 0.8, n),
        m + rng.integers(-1, 2, n),
        _pol(rng, n),
        _complex(rng, n, (1.2, 2.0), (0, 0.2)),
    ]


def _pw_permute_xyz(rng, n):
    return [*_plane_vector(rng, n), _pol(rng, n), _pol(rng, n)]


def _pw_translate(rng, n):
    return [*_plane_vector(rng, n), *(rng.uniform(-2, 2, n) for _ in range(3))]


def _points(dim):
    def make(rng, n):
        return [rng.uniform(0.1, 2, (n, dim))]

    return Case(make)


def _vectors(dim):
    def make(rng, n):
        vectors = rng.normal(size=(n, dim)) + 1j * rng.normal(size=(n, dim))
        return [vectors, rng.uniform(0.1, 2, (n, dim))]

    return Case(make)


def _cells(dim):
    def make(rng, n):
        return [np.eye(dim) * 1.3 + rng.uniform(-0.3, 0.3, (n, dim, dim))]

    def integer(rng, n):
        return [rng.integers(-3, 4, (n, dim, dim)) + 4 * np.eye(dim, dtype=int)]

    return make, integer


def _refractive(rng, n):
    return [
        _complex(rng, n, (1, 4), (0, 0.5)),
        _complex(rng, n, (0.8, 1.5), (0, 0.2)),
        _complex(rng, n, (-0.3, 0.3), (0, 0.05)),
    ]


BESSEL = [
    *(f"{kind}{suffix}" for suffix in ("", "_d") for kind in ("jv", "yv")),
    *(f"hankel{kind}{suffix}" for suffix in ("", "_d") for kind in (1, 2)),
]
SPHERICAL = [
    f"spherical_{kind}{suffix}"
    for suffix in ("", "_d")
    for kind in ("jn", "yn", "hankel1", "hankel2")
]
LATTICE = [
    (prefix, family)
    for prefix in ("lsum", "realsum", "recsum", "dsum")
    for family in (
        "sw1d",
        "sw1d_shift",
        "sw2d",
        "sw2d_shift",
        "sw3d",
        "cw1d",
        "cw1d_shift",
        "cw2d",
    )
]
TRANSFORMS = [
    "car2cyl",
    "car2sph",
    "cyl2car",
    "cyl2sph",
    "sph2car",
    "sph2cyl",
    "car2pol",
    "pol2car",
]

CASES: dict[str, Case] = {
    **{name: _bessel(False) for name in BESSEL},
    **{name: _bessel(True) for name in SPHERICAL},
    "lpmv": _angular(True),
    "pi_fun": _angular(False),
    "tau_fun": _angular(False),
    "wignersmalld": _wigner(False),
    "wignerd": _wigner(True),
    "wigner3j": Case(_wigner3j),
    "incgamma": Case(
        lambda rng, n: [
            rng.integers(-8, 9, n) / 2,
            _complex(rng, n, (0.8, 2.5), (-0.3, 0.3)),
        ]
    ),
    "intkambe": Case(
        lambda rng, n: [
            rng.integers(-5, 4, n).astype(float),
            _complex(rng, n, (0.8, 1.8)),
            _complex(rng, n, (0.9, 1.6)),
        ]
    ),
    "refractive_indices": Case(_refractive),
    "wave_vector_z": Case(
        lambda rng, n: [*_plane_vector(rng, n)[:2], _complex(rng, n, (1, 2), (0, 0.2))]
    ),
    "first_brillouin_1d": Case(
        lambda rng, n: [rng.uniform(-10, 10, n), rng.uniform(0.5, 5, n)]
    ),
    **{prefix + family: _lattice(prefix, family) for prefix, family in LATTICE},
    "sph_harm": Case(
        lambda rng, n: [
            *(v.astype(float) for v in _mode(rng, n, 0, 6)[::-1]),
            rng.uniform(-3, 3, n),
            _complex(rng, n, (-3, 3), (-0.3, 0.3)),
        ]
    ),
    **{name: Case(_vsh) for name in ("vsh_X", "vsh_Y", "vsh_Z")},
    **{f"vsw_{r}{kind}": _vsw(kind == "A") for r in ("", "r") for kind in "MNA"},
    **{
        f"vcw_{r}{kind}": _vcw(kind != "M", kind == "A")
        for r in ("", "r")
        for kind in "MNA"
    },
    **{f"vpw_{kind}": _vpw(kind == "A") for kind in "MNA"},
    "tl_vcw": Case(_tl_vcw),
    "tl_vcw_r": Case(_tl_vcw),
    "sw_rotate": Case(_sw_rotate, split=3),
    "cw_rotate": Case(_cw_rotate),
    **{f"sw_periodic_to_pw_{p}": Case(_sw_periodic_to_pw) for p in "hp"},
    "cw_periodic_to_pw": Case(_cw_periodic_to_pw),
    **{f"sw_periodic_to_cw_{p}": Case(_sw_periodic_to_cw, split=3) for p in "hp"},
    **{
        f"sw_translate_{r}{p}": Case(
            _sw_translate, split=3, fixed=((0, 1, 2, 3, 4, 5),)
        )
        for r in "sr"
        for p in "hp"
    },
    **{f"cw_translate_{r}": Case(_cw_translate) for r in "sr"},
    "pw_translate": Case(_pw_translate),
    **{f"pw_to_sw_{p}": Case(_pw_to_sw) for p in "hp"},
    "pw_to_cw": Case(_pw_to_cw),
    **{f"cw_to_sw_{p}": Case(_cw_to_sw) for p in "hp"},
    **{
        f"pw_permute_xyz{kind}_{p}": Case(_pw_permute_xyz)
        for kind in ("", "_inverse")
        for p in "hp"
    },
    **{
        f"tl_vsw_{r}{kind}": Case(
            _tl_vsw, split=2, fixed=((0, 1, 2, 3), (0, 1, 2, 3, 4))
        )
        for r in ("", "r")
        for kind in "AB"
    },
    **{name: _points(2 if "pol" in name else 3) for name in TRANSFORMS},
    **{f"v{name}": _vectors(2 if "pol" in name else 3) for name in TRANSFORMS},
    "cell_volume": Case(_cells(3)[0], real=_cells(3)[1]),
    "cell_reciprocal": Case(_cells(3)[0]),
}

# Ufuncs NumPy only reaches through a Rust wrapper with a Python-scalar path.
HIDDEN = {
    "pw_translate",
    "vpw_M",
    "vpw_N",
    "vpw_A",
    "cell_volume",
    "cell_reciprocal",
    *TRANSFORMS,
    *(f"v{name}" for name in TRANSFORMS),
}


# Loop dtype rows and gufunc signatures are part of the native ABI: every row
# must match the Rust kernel's operand types, so changes must be deliberate.
# A row holds the comma-separated loop types, the gufunc signature ("-" for an
# elementwise ufunc) and ufuncs with exactly these loops; a long list of names
# continues on the next row with the same types and signature.
REGISTRY_ROWS = """
dD->D - jv jv_d yv yv_d hankel1 hankel1_d hankel2 hankel2_d incgamma
dD->D - spherical_jn spherical_jn_d spherical_yn spherical_yn_d
dD->D - spherical_hankel1 spherical_hankel1_d spherical_hankel2 spherical_hankel2_d
dDD->D - intkambe
ddd->d,ddD->D - lpmv
ddD->D - pi_fun tau_fun
dddD->D - wignersmalld
dddDDD->D - wignerd
dddddd->d - wigner3j
dddd->D,dddD->D - sph_harm
ddd->d,DDD->D (),(),()->(2) refractive_indices
ddd->D,DDD->D - wave_vector_z
dd->d - first_brillouin_1d
dDdddD->D - lsumcw1d lsumsw1d realsumcw1d realsumsw1d recsumcw1d recsumsw1d
dDdddD->D (),(),(),(),(2),()->() lsumcw1d_shift realsumcw1d_shift recsumcw1d_shift
dDdddD->D (),(),(2),(2,2),(2),()->() lsumcw2d realsumcw2d recsumcw2d
ddDdddD->D (),(),(),(),(),(3),()->() lsumsw1d_shift realsumsw1d_shift
ddDdddD->D (),(),(),(),(),(3),()->() recsumsw1d_shift
ddDdddD->D (),(),(),(2),(2,2),(2),()->() lsumsw2d realsumsw2d recsumsw2d
ddDdddD->D (),(),(),(2),(2,2),(3),()->() lsumsw2d_shift realsumsw2d_shift
ddDdddD->D (),(),(),(2),(2,2),(3),()->() recsumsw2d_shift
ddDdddD->D (),(),(),(3),(3,3),(3),()->() lsumsw3d realsumsw3d recsumsw3d
lDdddl->D,dDdddl->D - dsumcw1d dsumsw1d
lDdddl->D,dDdddl->D (),(),(),(),(2),()->() dsumcw1d_shift
lDdddl->D,dDdddl->D (),(),(2),(2,2),(2),()->() dsumcw2d
llDdddl->D,ddDdddl->D (),(),(),(),(),(3),()->() dsumsw1d_shift
llDdddl->D,ddDdddl->D (),(),(),(2),(2,2),(2),()->() dsumsw2d
llDdddl->D,ddDdddl->D (),(),(),(2),(2,2),(3),()->() dsumsw2d_shift
llDdddl->D,ddDdddl->D (),(),(),(3),(3,3),(3),()->() dsumsw3d
lldd->D,llDd->D (),(),(),()->(3) vsh_X vsh_Y vsh_Z
llDdd->D,llDDd->D (),(),(),(),()->(3) vsw_M vsw_N vsw_rM vsw_rN
llDddl->D,llDDdl->D (),(),(),(),(),()->(3) vsw_A vsw_rA
dlDdd->D (),(),(),(),()->(3) vcw_M vcw_rM
dlDddD->D (),(),(),(),(),()->(3) vcw_N vcw_rN
dlDddDl->D (),(),(),(),(),(),()->(3) vcw_A vcw_rA
dddddd->D,DDDddd->D (),(),(),(),(),()->(3) vpw_M vpw_N
ddddddl->D,DDDdddl->D (),(),(),(),(),(),()->(3) vpw_A
dldlDdd->D - tl_vcw
dldlddd->D,dldlDdd->D - tl_vcw_r
llllDdd->D,llllDDd->D - tl_vsw_A tl_vsw_B tl_vsw_rA tl_vsw_rB
llllllddd->D - sw_rotate
dlldlld->D - cw_rotate
llllllddd->D,llllllDdd->D - sw_translate_rh sw_translate_rp sw_translate_sh
llllllddd->D,llllllDdd->D - sw_translate_sp
dlldllddd->D,dlldllDdd->D - cw_translate_r cw_translate_s
dddddd->D,DDDddd->D - pw_translate
dddlllld->D,ddDlllld->D - sw_periodic_to_pw_h sw_periodic_to_pw_p
dlllllDd->D - sw_periodic_to_cw_h sw_periodic_to_cw_p
dddldlld->D,dDdldlld->D - cw_periodic_to_pw
llldllD->D - cw_to_sw_h cw_to_sw_p
llldddl->D,lllDDDl->D - pw_to_sw_h pw_to_sw_p
dlldddl->D,dlldDdl->D - pw_to_cw
dddll->D,DDDll->D - pw_permute_xyz_h pw_permute_xyz_p
dddll->D,DDDll->D - pw_permute_xyz_inverse_h pw_permute_xyz_inverse_p
d->d (3)->(3) car2cyl car2sph cyl2car cyl2sph sph2car sph2cyl
d->d (2)->(2) car2pol pol2car
dd->d,Dd->D (3),(3)->(3) vcar2cyl vcar2sph vcyl2car vcyl2sph vsph2car vsph2cyl
dd->d,Dd->D (2),(2)->(2) vcar2pol vpol2car
l->l,d->d (i,i)->() cell_volume
d->d (i,i)->(i,i) cell_reciprocal
"""
REGISTRY: dict[str, tuple[tuple[str, ...], str | None]] = {}
for types, signature, *names in map(str.split, REGISTRY_ROWS.strip().splitlines()):
    for name in names:
        assert name not in REGISTRY, name
        REGISTRY[name] = (
            tuple(types.split(",")),
            None if signature == "-" else signature,
        )


class _Captured(Exception):  # noqa: N818 - control flow, not an error
    pass


class _Capture(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise _Captured(ufunc)


def _hidden(name: str) -> np.ufunc:
    """Return the ufunc a Rust wrapper forwards NumPy dispatch to."""
    probe = np.zeros(3).view(_Capture)
    arguments = {
        "pw_translate": (probe,) * 6,
        "vpw_M": (probe,) * 6,
        "vpw_N": (probe,) * 6,
        "vpw_A": (probe,) * 7,
        "cell_volume": (np.eye(3).view(_Capture),),
        "cell_reciprocal": (np.eye(3).view(_Capture),),
    }.get(name, (probe,) * (2 if name.startswith("v") else 1))
    with pytest.raises(_Captured) as captured:
        getattr(_native, name)(*arguments)
    return captured.value.args[0]


def ufunc(name: str) -> np.ufunc:
    return _hidden(name) if name in HIDDEN else getattr(_native, name)


def _exposed(value: object) -> set[str]:
    """Names under which the public namespaces expose ``value`` itself."""
    return {
        export
        for namespace in (cw, lattice, misc, pw, special, sw)
        for export in namespace.__all__
        if getattr(namespace, export) is value
    }


@functools.cache
def data(name: str) -> tuple[np.ufunc, tuple[np.ndarray, ...], np.ndarray]:
    """Deterministic operands and their per-element (serial) reference."""
    case = CASES[name]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    args = tuple(case.make(rng, case.size))
    function = ufunc(name)
    reference = np.stack([function(*(a[i] for a in args)) for i in range(case.size)])
    for value in args:
        value.setflags(write=False)
    reference.setflags(write=False)
    return function, args, reference


def _sentinel(dtype):
    return {"c": complex(np.nan, np.nan), "f": np.nan}.get(dtype.kind, -7)


def _mask(size):
    # Unmasked runs of 70 exceed the vectorized loop thresholds.
    return (np.arange(size) // 70) % 2 == 0


def test_every_native_ufunc_has_a_contract_case():
    exported = {n for n in dir(_native) if isinstance(getattr(_native, n), np.ufunc)}
    assert exported | HIDDEN == set(CASES) == set(REGISTRY)
    assert not exported & HIDDEN


@pytest.mark.parametrize("name", REGISTRY)
def test_registered_loops_match_the_golden_registry(name):
    types, signature = REGISTRY[name]
    function = ufunc(name)
    # A ufunc or wrapper that a namespace exposes directly reports the public
    # name (pw.to_cw is 'to_cw'), and a hidden ufunc that of its wrapper, so
    # NumPy error messages show the name the caller used.
    native = getattr(_native, name)
    assert function.__name__ == native.__name__ in (_exposed(native) or {name})
    if name in HIDDEN:
        # pickle finds a wrapper through its __module__ and __name__.
        assert pickle.loads(pickle.dumps(native)) is native
    assert (function.nin, function.nout) == (len(types[0].split("->")[0]), 1)
    assert tuple(function.types) == types
    assert function.signature == signature


def test_stub_declares_exactly_the_native_module():
    stub = Path(_native.__file__).with_name("_native.pyi")
    declared = {}
    for node in ast.parse(stub.read_text()).body:
        if isinstance(node, ast.ClassDef):
            declared[node.name] = type
        elif isinstance(node, ast.FunctionDef):
            declared[node.name] = "function"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert ast.unparse(node.annotation) == "np.ufunc"
            declared[node.target.id] = np.ufunc
    runtime = {n for n in dir(_native) if not n.startswith("__")}
    assert set(declared) == runtime
    for name, kind in declared.items():
        value = getattr(_native, name)
        if kind == "function":
            assert callable(value) and not isinstance(value, (type, np.ufunc)), name
        else:
            assert isinstance(value, kind), name


@pytest.mark.parametrize("name", CASES)
def test_vectorized_values_equal_scalar_calls(name):
    function, args, reference = data(name)
    value = function(*args)
    assert value.dtype == reference.dtype
    assert_array_equal(value, reference)
    assert np.all(np.isfinite(reference))


@pytest.mark.parametrize("name", CASES)
def test_broadcast_operands_fill_every_output(name):
    """Stride-0 operands (plans, fast paths) agree with materialized copies."""
    function, args, reference = data(name)
    count = len(args)
    split = CASES[name].split or count // 2
    patterns = {
        tuple(range(count)),
        tuple(range(split)),
        tuple(range(split, count)),
        *CASES[name].fixed,
    }
    for pattern in sorted(patterns):
        views = [
            np.broadcast_to(a[:1], a.shape) if j in pattern else a
            for j, a in enumerate(args)
        ]
        expected = function(*(np.ascontiguousarray(v) for v in views))
        assert_array_equal(function(*views), expected)
        out = np.full_like(expected, _sentinel(expected.dtype))
        assert function(*views, out=out) is out
        assert_array_equal(out, expected)
        if function.signature is None:
            mask = _mask(len(out))
            out = np.full_like(expected, _sentinel(expected.dtype))
            function(*views, out=out, where=mask)
            assert_array_equal(out[mask], expected[mask])
            assert_array_equal(out[~mask], _sentinel(expected.dtype))
    # Every operand broadcast from the first element: every row is the first.
    views = [np.broadcast_to(a[:1], a.shape) for a in args]
    assert_array_equal(
        function(*views), np.broadcast_to(reference[:1], reference.shape)
    )


@pytest.mark.parametrize("name", CASES)
def test_reversed_and_strided_operands(name):
    function, args, reference = data(name)
    assert_array_equal(function(*(a[::-1] for a in args)), reference[::-1])
    assert_array_equal(
        function(*(np.repeat(a, 2, axis=0)[1::2] for a in args)), reference
    )
    # Negative and Fortran-ordered core strides, for operands and output.
    flipped = [a[..., ::-1].copy()[..., ::-1] for a in args]
    out = np.empty(reference.shape[::-1], reference.dtype).T
    assert function(*flipped, out=out) is out
    assert_array_equal(out, reference)
    out = np.empty_like(reference)[..., ::-1]
    assert function(*(np.asfortranarray(a) for a in args), out=out) is out
    assert_array_equal(out, reference)


def _unaligned(value):
    """Copy into a buffer that is misaligned by one byte and every other row."""
    shape = (2 * len(value), *value.shape[1:])
    storage = np.ndarray(
        shape,
        dtype=value.dtype,
        buffer=bytearray(value.dtype.itemsize * int(np.prod(shape)) + 1),
        offset=1,
    )
    storage[::2] = value
    return storage[::2]


@pytest.mark.parametrize("name", CASES)
def test_unaligned_strided_operands_and_masked_output(name):
    function, args, reference = data(name)
    operands = [_unaligned(a) for a in args]
    assert not any(a.flags.aligned for a in operands if a.dtype.itemsize > 1)
    out = _unaligned(np.full_like(reference, _sentinel(reference.dtype)))
    assert function(*operands, out=out) is out
    assert_array_equal(out, reference)
    if function.signature is None:
        out = _unaligned(np.full_like(reference, _sentinel(reference.dtype)))
        mask = _mask(len(out))
        function(*operands, out=out, where=mask)
        assert_array_equal(out[mask], reference[mask])
        assert_array_equal(out[~mask], _sentinel(reference.dtype))


@pytest.mark.parametrize("name", CASES)
def test_output_aliasing_an_input(name):
    function, args, reference = data(name)
    matches = [
        j
        for j, a in enumerate(args)
        if a.dtype == reference.dtype and a.shape == reference.shape
    ]
    if not matches:
        pytest.skip("no input shares the output dtype and shape")
    for j in matches:
        operands = list(args)
        operands[j] = alias = args[j].copy()
        assert function(*operands, out=alias) is alias
        assert_array_equal(alias, reference)
        # Exact overlap through a negative stride is not buffered by NumPy.
        operands = [a[::-1] for a in args]
        operands[j] = alias = args[j].copy()[::-1]
        function(*operands, out=alias)
        assert_array_equal(alias, reference[::-1])


# Public wrappers of native ufuncs, keyed by the case whose operands they take:
# the special functions named after their case that are not the ufunc itself,
# and the renamed and namespace wrappers.
WRAPPERS = {
    **{
        name: wrapper
        for name in CASES
        if callable(wrapper := getattr(special, name, None))
        and not isinstance(wrapper, np.ufunc)
    },
    "pw_translate": pw.translate,
    "cell_volume": lattice.volume,
    "cell_reciprocal": lattice.reciprocal,
    "cw_rotate": cw.rotate,
    "sw_rotate": sw.rotate,
    "cw_translate_s": cw.translate,
    "cw_translate_r": functools.partial(cw.translate, singular=False),
    **{
        prefix + p: functools.partial(wrapper, poltype=poltype, **options)
        for prefix, wrapper, options in (
            ("sw_translate_s", sw.translate, {}),
            ("sw_translate_r", sw.translate, {"singular": False}),
            ("pw_permute_xyz_", pw.permute_xyz, {}),
            ("pw_permute_xyz_inverse_", pw.permute_xyz, {"inverse": True}),
            ("pw_to_sw_", pw.to_sw, {}),
            ("cw_to_sw_", cw.to_sw, {}),
            ("sw_periodic_to_pw_", sw.periodic_to_pw, {}),
            ("sw_periodic_to_cw_", sw.periodic_to_cw, {}),
        )
        for p, poltype in (("h", "helicity"), ("p", "parity"))
    },
}


@pytest.mark.parametrize("name", WRAPPERS)
def test_public_wrappers_forward_out_where_and_axes(name):
    wrapper = WRAPPERS[name]
    function, args, reference = data(name)
    sentinel = _sentinel(reference.dtype)
    out = _unaligned(np.full_like(reference, sentinel))
    assert wrapper(*args, out=out) is out
    assert_array_equal(out, reference)
    if function.signature is None:
        mask = _mask(len(out))
        out = np.full_like(reference, sentinel)
        assert wrapper(*args, out=out, where=mask) is out
        assert_array_equal(out[mask], reference[mask])
        assert_array_equal(out[~mask], sentinel)
    else:
        # Default core axes for the operands, the output's moved to the front.
        *cores, core = [
            len(dims.split(",")) if dims else 0
            for dims in re.findall(r"\((.*?)\)", function.signature)
        ]
        axes = [tuple(range(-k, 0)) for k in cores] + [tuple(range(core))]
        assert_array_equal(wrapper(*args, axes=axes), np.moveaxis(reference, 0, -1))


def _loop_operands(types, args):
    codes = types.split("->")[0]
    return [
        np.asarray(a.real if code != "D" else a, dtype=np.dtype(code))
        for code, a in zip(codes, args, strict=True)
    ]


@pytest.mark.parametrize("name", [name for name in CASES if len(ufunc(name).types) > 1])
def test_every_dtype_loop_agrees_on_shared_values(name):
    function, args, _ = data(name)
    case = CASES[name]
    rng = np.random.default_rng(7)
    shared = case.real(rng, 64) if case.real else [a[:64] for a in args]
    shared = _loop_operands(function.types[0], shared)
    results = [
        function(*_loop_operands(types, shared), signature=types)
        for types in function.types
    ]
    first = results[0]
    for types, result in zip(function.types[1:], results[1:], strict=True):
        assert result.dtype == np.dtype(types[-1])
        assert np.all(np.isfinite(result))
        assert_array_equal(
            np.asarray(result, dtype=complex), np.asarray(first, complex)
        )


# Wrapper-internal ufuncs whose loops report element errors, with the operand
# index and a value that fails there; cell_volume reports none.
POISON = {
    "pw_translate": (3, np.nan),
    "vpw_M": (3, np.nan),
    "vpw_N": (3, np.nan),
    "vpw_A": (3, np.nan),
    "cell_reciprocal": (0, 0.0),
    **dict.fromkeys(TRANSFORMS, (0, np.nan)),
    **{f"v{name}": (1, np.nan) for name in TRANSFORMS},
}

REPORT_ERRORS = """
import sys
import numpy as np
from treams_rs import _native

with np.load(sys.argv[1]) as saved:
    for name in sys.argv[2:]:
        args = [saved[f"{name}.{j}"] for j in range(int(saved[name]))]
        try:
            getattr(_native, name)(*args)
        except ValueError as error:
            print(name, error, sep=": ")
        else:
            print(name, "no error", sep=": ")
"""


def _poisoned(name):
    index, value = POISON[name]
    args = [a.copy() for a in data(name)[1]]
    args[index][-1] = value
    return args


def test_poisoned_wrappers_are_every_wrapper_with_element_errors():
    assert set(POISON) == HIDDEN - {"cell_volume"}


@pytest.mark.parametrize("threads", ["1", "4"])
def test_wrapper_batch_errors_raise_while_numpy_releases_the_gil(threads, tmp_path):
    """A failing element raises the ValueError it raises alone, in every batch.

    NumPy runs large loops with the GIL released, while the Rust wrapper that
    called the ufunc still counts the thread as attached to Python; reporting
    the error must take the GIL back. The subprocess turns a crash into a
    failure of this test.
    """
    saved: dict[str, np.ndarray | int] = {}
    expected = []
    for name in POISON:
        args = _poisoned(name)
        with pytest.raises(ValueError) as alone:
            getattr(_native, name)(*(a[-1:] for a in args))
        expected.append(f"{name}: {alone.value}")
        saved[name] = len(args)
        saved |= {f"{name}.{j}": a for j, a in enumerate(args)}
    path = tmp_path / "batches.npz"
    np.savez(path, **saved)
    result = subprocess.run(
        [sys.executable, "-c", REPORT_ERRORS, str(path), *POISON],
        capture_output=True,
        text=True,
        timeout=600,
        env=dict(os.environ, RAYON_NUM_THREADS=threads),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected


REPORT_FIRST_ERRORS = """
import numpy as np
from treams_rs import _native

n = 400_000
for first in (0, 1):
    # Across the halving split, so different workers reach each element first.
    order, x = np.ones(n), np.full(n, 0.5)
    order[n // 2 - 5 + 10 * first], x[n // 2 + 5 - 10 * first] = 0.5, 1.5
    errors = set()
    for _ in range(20):
        try:
            _native.lpmv(order, 3.0, x)
        except ValueError as error:
            errors.add(str(error))
    print(*sorted(errors), sep=" | ")
"""


def test_parallel_batches_raise_their_first_failing_element():
    """Of several failing elements, the first raises, as in a serial loop."""
    errors = []
    for order, x in ((0.5, 0.5), (1.0, 1.5)):
        with pytest.raises(ValueError) as alone:
            _native.lpmv(np.array([order]), 3.0, np.array([x]))
        errors.append(str(alone.value))
    result = subprocess.run(
        [sys.executable, "-c", REPORT_FIRST_ERRORS],
        capture_output=True,
        text=True,
        timeout=600,
        env=dict(os.environ, RAYON_NUM_THREADS="4"),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == errors


# Python-scalar fast paths (Python wrappers and Rust pyfunctions) bypass NumPy
# dispatch. They must return what the ufunc returns for the same values as
# 0-d or batch-of-one arrays: equal values and dtype, or the same ValueError.
REAL = st.one_of(st.integers(-3, 3), st.floats(-3, 3))
NUMBER = st.one_of(REAL, st.builds(complex, st.floats(-3, 3), st.floats(-1, 1)))
ORDER = st.one_of(st.integers(-4, 4), st.floats(-4, 4))
LABEL = st.one_of(st.integers(-2, 9), st.just(300))
POL = st.integers(-1, 2)
AXIAL = st.sampled_from([0.2, -0.5, 1])


def _zero_d(function, **keywords):
    return (
        lambda *a: function(*a, **keywords),
        lambda *a: function(*map(np.asarray, a), **keywords),
    )


def _batch(function):
    return (
        lambda *a: function(*(list(v) for v in a)),
        lambda *a: function(*(np.array([v]) for v in a))[0],
    )


def _cell(function):
    return (
        lambda c: function(np.array(c, dtype=float)),
        lambda c: function(np.array([c], dtype=float))[0],
    )


def _vector(dim, values):
    return st.lists(values, min_size=dim, max_size=dim)


FAST_PATHS = {
    "hankel1": (_zero_d(special.hankel1), st.tuples(ORDER, NUMBER)),
    "hankel2": (_zero_d(special.hankel2), st.tuples(ORDER, NUMBER)),
    "lpmv": (
        _zero_d(special.lpmv),
        st.tuples(ORDER, st.one_of(st.integers(0, 9), st.floats(0, 9)), NUMBER),
    ),
    "pi_fun": (
        _zero_d(special.pi_fun),
        st.tuples(st.integers(0, 9), ORDER, NUMBER),
    ),
    "tau_fun": (
        _zero_d(special.tau_fun),
        st.tuples(st.integers(0, 9), ORDER, NUMBER),
    ),
    "incgamma": (
        _zero_d(special.incgamma),
        st.tuples(
            st.one_of(st.integers(-8, 8).map(lambda k: k / 2), st.floats(-4, 4)),
            NUMBER,
        ),
    ),
    "intkambe": (
        _zero_d(special.intkambe),
        st.tuples(st.one_of(st.integers(-6, 6), st.just(300)), NUMBER, NUMBER),
    ),
    "wigner3j": (_zero_d(special.wigner3j), st.tuples(*[LABEL] * 6)),
    **{
        name: (
            _zero_d(getattr(special, name)),
            st.tuples(AXIAL, LABEL, AXIAL, LABEL, NUMBER, REAL, REAL),
        )
        for name in ("tl_vcw", "tl_vcw_r")
    },
    "cw.rotate": (
        _zero_d(cw.rotate),
        st.tuples(AXIAL, LABEL, POL, AXIAL, LABEL, POL, REAL),
    ),
    **{
        f"cw.translate-{singular}": (
            _zero_d(cw.translate, singular=singular),
            st.tuples(
                AXIAL,
                LABEL,
                POL,
                AXIAL,
                LABEL,
                POL,
                st.one_of(NUMBER, st.just(0)),
                REAL,
                st.one_of(REAL, st.just(0.0)),
            ),
        )
        for singular in (False, True, 0, 1)
    },
    **{
        f"pw.permute_xyz-{poltype}-{inverse}": (
            _zero_d(pw.permute_xyz, poltype=poltype, inverse=inverse),
            st.tuples(NUMBER, NUMBER, NUMBER, POL, POL),
        )
        for poltype in ("helicity", "parity")
        for inverse in (False, True, 0, 1)
    },
    "pw.translate": (
        _zero_d(pw.translate),
        st.tuples(NUMBER, NUMBER, NUMBER, REAL, REAL, REAL),
    ),
    **{
        name: (
            _zero_d(getattr(special, name)),
            st.tuples(NUMBER, NUMBER, NUMBER, REAL, REAL, REAL),
        )
        for name in ("vpw_M", "vpw_N")
    },
    "vpw_A": (
        _zero_d(special.vpw_A),
        st.tuples(NUMBER, NUMBER, NUMBER, REAL, REAL, REAL, POL),
    ),
    **{
        name: (
            _batch(getattr(special, name)),
            st.tuples(_vector(2 if "pol" in name else 3, REAL)),
        )
        for name in TRANSFORMS
    },
    **{
        f"v{name}": (
            _batch(getattr(special, f"v{name}")),
            st.tuples(
                _vector(2 if "pol" in name else 3, NUMBER),
                _vector(2 if "pol" in name else 3, REAL),
            ),
        )
        for name in TRANSFORMS
    },
    **{
        name: (
            _cell(getattr(_native, name)),
            st.integers(1, 3).flatmap(
                lambda dim: st.tuples(_vector(dim, _vector(dim, REAL)))
            ),
        )
        for name in ("cell_volume", "cell_reciprocal")
    },
}


def _outcome(call, args):
    try:
        return call(*args), None
    except ValueError:
        return None, ValueError


@pytest.mark.parametrize("name", FAST_PATHS)
@settings(max_examples=40)
@given(data=st.data())
def test_python_scalar_fast_paths_match_the_ufunc(name, data):
    (fast, array), strategy = FAST_PATHS[name]
    args = data.draw(strategy)
    value, error = _outcome(fast, args)
    expected, expected_error = _outcome(array, args)
    assert error is expected_error
    if error is None:
        assert np.asarray(value).dtype == np.asarray(expected).dtype
        assert_array_equal(value, expected)


@pytest.mark.parametrize("function", [pw.translate, special.vpw_M])
def test_complex_positions_are_a_scalar_path_convenience(function):
    """Python complex positions with zero imaginary part are accepted as scalars.

    NumPy's type resolution rejects complex positions, which have no loop; the
    scalar path keeps accepting the equivalent Python and NumPy complex scalars.
    """
    k = (0.3, 1, 2 + 0.1j)
    assert_array_equal(function(*k, 1 + 0j, 0, 0), function(*k, 1.0, 0, 0))
    assert_array_equal(function(*k, np.complex128(1), 0, 0), function(*k, 1, 0, 0))
    with pytest.raises(TypeError, match=f"ufunc '{function.__name__}'"):
        function(*k, np.asarray(1 + 0j), 0, 0)
