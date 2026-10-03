"""The treams-rs thread pool: every parallel region runs on it, and results repeat
bit for bit at every budget."""

import re
from itertools import accumulate

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_native_contexts import (
    _KS,
    _ORIGIN,
    _cotangents,
    _Method,
    _modes,
    _split,
    case,
)

import treams_rs as tr
from treams_rs import _native, misc

from _support import ROOT

# Budgets above the CPUs of small hosts are deliberate here.
OVERSUBSCRIBED = pytest.mark.filterwarnings(
    "ignore:.*oversubscription slows:RuntimeWarning"
)
BUDGETS = (1, 2, 3)


def _bits(tree):
    """The bytes of every array and number in a nested result."""
    if isinstance(tree, (list, tuple)):
        return [_bits(item) for item in tree]
    if isinstance(tree, dict):
        return {key: _bits(value) for key, value in tree.items()}
    return np.asarray(tree).tobytes()


def _pulled_back(make_case):
    """The recorded values and the gradients of one case."""
    current = make_case()
    values, context = _split(current.record(*current.arrays))
    return values, _Method(context, current).pullback(*_cotangents(values))


_RNG = np.random.default_rng(11)
_SW3 = tr.SphericalBasis.default(3, nmax=2, positions=[[0, 0, 0], [0.2, -0.1, 0.9]])
_CW_MANY = tr.CylindricalBasis.default(np.linspace(-0.6, 0.6, 40), 2)
_POINTS = _RNG.uniform(-1.5, 1.5, (300, 3)) + np.array([0.0, 0.0, 3.0])
_Q = np.column_stack([np.linspace(-0.4, 0.4, 150), np.linspace(0.3, -0.2, 150)])
_POLS = np.arange(150) % 2
_VECTORS = np.column_stack(
    [np.linspace(-0.5, 0.5, 96), np.linspace(0.4, -0.4, 96), np.full(96, 1.3 + 0.01j)]
)

# Each pullback reduces over more items than the 64 chunks of a deterministic
# reduction, and pw::field over more than 4096 entries.
REDUCTIONS = {
    "spherical_channels": case(
        lambda: _native.spherical_channels(
            *_modes(_SW3), list(_KS), _Q.tolist(), _POLS.tolist(), 2.8, True, False
        )
    ),
    "cylindrical_channels": case(
        lambda: _native.cylindrical_channels(
            *_modes(tr.CylindricalBasis.default([0.1, 0.3], 2)),
            [1.3, 1.3],
            np.column_stack([_Q[:, 0], np.full(150, 0.1)]).tolist(),
            _POLS.tolist(),
            1.7,
            True,
            False,
        )
    ),
    "cw_to_sw": case(
        lambda: _native.cw_to_sw(
            list(tr.SphericalBasis.default(2).modes),
            list(_CW_MANY.modes),
            [[0.1, 0.2, -0.3]],
            _ORIGIN,
            _KS,
            True,
        )
    ),
    "plane_expansion": case(
        lambda: _native.plane_expansion(
            *_modes(_SW3), _VECTORS.tolist(), (np.arange(96) % 2).tolist(), True, False
        )
    ),
    "periodic_to_cw": case(
        lambda: _native.periodic_to_cw(
            list(tr.CylindricalBasis.default([0.1, 0.3], 2).modes),
            list(tr.SphericalBasis.default(6).modes),
            _ORIGIN,
            [[0.0, 0.1, 0.0]],
            list(_KS),
            1.7,
            True,
        )
    ),
    "lattice_expansion": case(
        lambda: _native.lattice_expansion(
            list(tr.SphericalBasis.default(4).modes),
            list(tr.SphericalBasis.default(4).modes),
            _ORIGIN,
            [[0.1, 0.0, 0.0]],
            _KS,
            True,
            [0.1, 0.2],
            [[1.5, 0.0], [0.3, 1.4]],
            0.9 + 0.1j,
        )
    ),
    "field": case(
        lambda c, p: _native.field(*_modes(_SW3), c, p, _KS, True, True),
        np.linspace(0.3, 1.2, len(_SW3)) * (1 + 0.2j),
        _POINTS,
    ),
    "field_operator": case(
        lambda p: _native.field_operator(*_modes(_SW3), p, _KS, True, False), _POINTS
    ),
    "plane_field": case(
        lambda v, p, c: _native.plane_field(
            v, (np.arange(96) % 2).tolist(), p, c, True, False
        ),
        _VECTORS,
        _POINTS[:100],
        np.linspace(0.4, 1.2, 96) * (1 - 0.1j),
    ),
    "layer_stack": case(
        lambda: _native.layer_stack(
            [[1.0, 1.0], [1.5 + 0.1j, 1.6 + 0.1j], [1.2, 1.2]],
            [1.0, 0.7, 0.9],
            _Q.tolist(),
            [0.4],
            2,
            False,
        )
    ),
    "IterativeSphereCluster.record": case(
        lambda radii, epsilon, positions, incident: _native.IterativeSphereCluster(
            1, 1.3, radii, epsilon, positions
        ).record(incident),
        np.full(70, 0.15),
        np.full(70, 2.0 + 0.05j),
        np.column_stack([np.arange(70) % 7, np.arange(70) // 7, np.zeros(70)]) * 0.6,
        np.linspace(0.2, 1.4, 70 * 6 * 2).reshape(70 * 6, 2) * (1 + 0.3j),
    ),
}


@pytest.mark.gradients
@OVERSUBSCRIBED
@pytest.mark.parametrize("name", REDUCTIONS)
def test_pullback_reductions_repeat_bit_for_bit_at_every_budget(name):
    # Gradients that add partial sums over many items use fixed chunks, combined
    # in order, so neither the thread count nor the run changes their bits.
    results = []
    for threads in BUDGETS:
        with tr.threads(threads):
            results.append(_bits(_pulled_back(REDUCTIONS[name])))
            results.append(_bits(_pulled_back(REDUCTIONS[name])))
    assert all(result == results[0] for result in results[1:])


@pytest.mark.interface
@OVERSUBSCRIBED
def test_forward_values_and_dense_algebra_repeat_bit_for_bit_at_every_budget():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(200, 200)) + 1j * rng.normal(size=(200, 200))
    b = rng.normal(size=(200, 64)) + 0j
    spheres = [tr.sphere_tmatrix(k0=1.3, lmax=3, radius=0.2, material=4 + 0.1j)] * 3
    cluster = tr.Cluster(spheres, positions=[[0, 0, 0], [0, 0, 0.8], [0.7, 0, 0.3]])
    z = np.linspace(0.1, 30, 20_000) + 0.2j
    results = []
    for threads in BUDGETS:
        with tr.threads(threads):
            results.append(
                _bits(
                    [
                        tr.diff.solve(a, b)[0],
                        tr.diff.solve(a, b[:, :1])[0],
                        tr.diff.eig(a[:150, :150])[0],
                        tr.diff.svdvals(a)[0],
                        cluster.solve().array,
                        tr.special.spherical_hankel1(5, z),
                    ]
                )
            )
    assert all(result == results[0] for result in results[1:])


@pytest.mark.interface
@OVERSUBSCRIBED
@pytest.mark.parametrize("threads", [1, 2])
@given(seed=st.integers(0, 2**32 - 1), size=st.integers(1025, 3000))
@settings(max_examples=10)
def test_two_input_ufunc_reductions_match_a_sequential_fold(threads, seed, size):
    # reduce/accumulate pass the accumulator as an input and the output; the
    # parallel path reads every input first and must not run for them.
    rng = np.random.default_rng(seed)
    k = rng.uniform(-3.0, 3.0, size)
    pitch = rng.uniform(0.5, 3.0, size)
    chain = np.concatenate([k[:1], pitch[1:]])  # the accumulator starts at k[0]
    with tr.threads(threads):
        reduced = misc.firstbrillouin1d.reduce(chain)
        accumulated = misc.firstbrillouin1d.accumulate(chain)
        elementwise = misc.firstbrillouin1d(k, pitch)
    expected = list(accumulate(chain, lambda a, b: misc.firstbrillouin1d(a, b)))
    assert np.array_equal(accumulated, expected)
    assert reduced == expected[-1]
    assert np.array_equal(
        elementwise,
        [misc.firstbrillouin1d(a, b) for a, b in zip(k, pitch, strict=True)],
    )


@pytest.mark.interface
def test_in_place_elementwise_ufuncs_match_out_of_place():
    pitch = np.linspace(0.5, 3.0, 4096)
    k = np.linspace(-6.0, 6.0, 4096)
    expected = misc.firstbrillouin1d(k, pitch)
    misc.firstbrillouin1d(k, pitch, out=k)
    assert np.array_equal(k, expected)


# Rayon parallel iterators and pool entry points, faer's global parallelism,
# and the faer solver types that read it.
PARALLEL = re.compile(
    r"\.(par_\w+|into_par_iter)\("
    r"|rayon::(join|scope\w*|in_place_scope\w*|spawn\w*|broadcast|current_num_threads)\b"
    r"|get_global_parallelism|Par::rayon"
    r"|\b(Svd|Eigen|SelfAdjointEigen|GeneralizedEigen|PartialPivLu|FullPivLu|Qr"
    r"|ColPivQr|Llt|Ldlt|Lblt)::new\w*\(|\bsolvers::"
)
POOLED = re.compile(r"\bthreads::(install|join|dense|product)\(")


@pytest.mark.interface
def test_rust_parallel_regions_run_on_the_treams_pool():
    # Rayon's global pool hangs forked children and ignores set_num_threads.
    # Every known parallel construct sits inside a call that installs the
    # treams-rs pool; the pool module and the benches are exempt. Operators on
    # faer matrices also read faer's global parallelism and stay unused.
    def spans(text):
        for match in POOLED.finditer(text):
            depth = 0
            for end in range(match.end() - 1, len(text)):
                depth += {"(": 1, ")": -1}.get(text[end], 0)
                if depth == 0:
                    yield match.start(), end
                    break

    unpooled = []
    for path in sorted((ROOT / "crates").rglob("*.rs")):
        if (
            path == ROOT / "crates/treams-core/src/threads.rs"
            or "benches" in path.parts
        ):
            continue
        text = path.read_text()
        covered = list(spans(text))
        unpooled.extend(
            f"{path.relative_to(ROOT)}:{text.count(chr(10), 0, m.start()) + 1}: {m[0]}"
            for m in PARALLEL.finditer(text)
            if not any(a <= m.start() <= b for a, b in covered)
        )
    assert unpooled == []
