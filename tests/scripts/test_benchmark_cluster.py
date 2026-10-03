"""The benchmark's chunked upstream gate is the whole-array assertion."""

import json

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from _scripts import ROOT, load

pytestmark = pytest.mark.interface

bench = load("benchmark_cluster")


@pytest.mark.parametrize("transpose", [False, True])
def test_large_parity_check_is_bounded_and_checks_last_element(transpose, monkeypatch):
    check = bench._assert_allclose
    expected = np.arange(131076, dtype=np.float64).reshape(3, -1).astype(complex)
    actual = expected.copy()
    if transpose:
        actual, expected = actual.T[::-1], expected.T[::-1]
    original = np.testing.assert_allclose
    sizes = []

    def bounded(a, b, **kwargs):
        sizes.append(a.size)
        assert a.size <= 65536
        original(a, b, **kwargs)

    monkeypatch.setattr(np.testing, "assert_allclose", bounded)
    check(actual, expected)
    assert sum(sizes) == actual.size
    actual.flat[-1] += 1
    with pytest.raises(AssertionError, match="flat indices 131072:131076"):
        check(actual, expected)


def test_bounded_parity_preserves_nonfinite_and_shape_checks():
    check = bench._assert_allclose
    check([np.nan, np.inf, -np.inf], [np.nan, np.inf, -np.inf])
    with pytest.raises(AssertionError):
        check([np.inf], [-np.inf])
    with pytest.raises(AssertionError, match="shape mismatch"):
        check(np.ones((2, 1)), np.ones((1, 2)))


MUTATIONS = {
    "none": lambda a, b: None,
    "fails": lambda a, b: a * (1 + 1e-8),
    "passes": lambda a, b: a * (1 + 1e-10),
    "matched-nan": lambda a, b: np.nan,
    "unmatched-nan": lambda a, b: a,
    "matched-inf": lambda a, b: np.inf,
    "opposite-inf": lambda a, b: -np.inf,
}


@settings(max_examples=150)
@given(
    size=st.sampled_from([1, 6, 7, 8, 13, 14, 15, 21, 22, 36]),
    layout=st.sampled_from(["flat", "rows", "transposed", "reversed"]),
    mutation=st.sampled_from(sorted(MUTATIONS)),
    position=st.floats(0, 1, exclude_max=True),
    seed=st.integers(0, 2**32 - 1),
)
# Mutate the last element of a 7-element chunk in flat, reversed and strided order.
@example(size=7, layout="flat", mutation="fails", position=0.9, seed=0)
@example(size=14, layout="reversed", mutation="fails", position=0.0, seed=0)
@example(size=21, layout="transposed", mutation="unmatched-nan", position=0.86, seed=0)
def test_chunked_gate_agrees_with_whole_array_assertion(
    size, layout, mutation, position, seed
):
    """Chunk boundaries, strides and NaN/inf placement never change the verdict."""
    generator = np.random.default_rng(seed)
    expected = generator.normal(size=size) + 1j * generator.normal(size=size)
    actual = expected.copy()
    index = int(position * size)
    value = MUTATIONS[mutation](actual[index], expected[index])
    if value is not None:
        actual[index] = value
        if mutation == "unmatched-nan":
            expected[index] = np.nan
        elif mutation in ("matched-nan", "matched-inf"):
            expected[index] = value
        elif mutation == "opposite-inf":
            expected[index] = np.inf
    columns = 3 if size % 3 == 0 else 1
    shaped = {
        "flat": lambda x: x,
        "rows": lambda x: x.reshape(-1, columns),
        "transposed": lambda x: x.reshape(-1, columns).T,
        "reversed": lambda x: x[::-1],
    }[layout]
    actual, expected = shaped(actual), shaped(expected)
    try:
        np.testing.assert_allclose(actual, expected, rtol=bench.RTOL, atol=bench.ATOL)
    except AssertionError:
        whole = False
    else:
        whole = True
    try:
        bench._assert_allclose(actual, expected, chunk=7)
    except AssertionError:
        chunked = False
    else:
        chunked = True
    assert chunked == whole


def recorded_commands(node):
    if isinstance(node, dict):
        if isinstance(node.get("command"), list):
            yield node["command"]
        for child in node.values():
            yield from recorded_commands(child)
    elif isinstance(node, list):
        for child in node:
            yield from recorded_commands(child)


def test_every_planned_workload_is_a_harness_choice():
    """Plans and manifests cannot name a workload the harness would reject."""
    assert len(set(bench.WORKLOADS)) == len(bench.WORKLOADS) == 203
    planned = {
        command[command.index("--workload") + 1]
        for path in sorted((ROOT / "benchmarks").glob("*.json"))
        for command in recorded_commands(json.loads(path.read_text()))
        if "--workload" in command
        and (
            not command[0].endswith(".py")
            or command[0].endswith(("benchmark_cluster.py", "qualify_upstream.py"))
        )
    }
    assert len(planned) > 150
    assert planned <= set(bench.WORKLOADS)
