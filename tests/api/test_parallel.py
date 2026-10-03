"""The thread budget of treams_rs.parallel: controls, environment and process pools."""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import treams_rs as tr

pytestmark = pytest.mark.interface

ENVIRONMENT = ("TREAMS_RS_NUM_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS")
# Tests that ask for two threads must also pass on one-CPU hosts.
OVERSUBSCRIBED = pytest.mark.filterwarnings(
    "ignore:.*oversubscription slows:RuntimeWarning"
)


def run(code, env=None, timeout=120):
    """Run ``code`` in a fresh interpreter whose environment sets no budget."""
    base = {k: v for k, v in os.environ.items() if k not in ENVIRONMENT}
    base.update(env or {})
    process = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # A hung fork leaves worker processes behind; end the whole group.
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    assert process.returncode == 0, stderr
    return stdout


def info_in(env):
    return json.loads(
        run("import json, treams_rs as tr; print(json.dumps(tr.thread_info()))", env)
    )


def test_thread_info_describes_the_budget_without_starting_a_pool():
    info = info_in({})
    assert info["threads"] == info["available"] >= 1
    assert info["source"] == "available_parallelism"
    assert info["pool_threads"] is None
    assert info["forked"] is False
    assert info["diagnostics"] == []
    assert info["environment"] == list(ENVIRONMENT)


@pytest.mark.parametrize(
    ("env", "threads", "source"),
    [
        (
            {"TREAMS_RS_NUM_THREADS": "2", "RAYON_NUM_THREADS": "3"},
            2,
            "TREAMS_RS_NUM_THREADS",
        ),
        ({"RAYON_NUM_THREADS": " 3 ", "OMP_NUM_THREADS": "2"}, 3, "RAYON_NUM_THREADS"),
        ({"RAYON_NUM_THREADS": "0", "OMP_NUM_THREADS": "2,1"}, 2, "OMP_NUM_THREADS"),
        ({"TREAMS_RS_NUM_THREADS": ""}, None, "available_parallelism"),
    ],
)
def test_the_environment_sets_the_budget_in_precedence_order(env, threads, source):
    info = info_in(env)
    assert info["source"] == source
    assert info["threads"] == (info["available"] if threads is None else threads)


def test_only_unparsable_settings_warn_at_import():
    # Oversubscription can be deliberate, and warnings at import break suites
    # that turn warnings into errors; diagnostics report it instead.
    out = run(
        """
        import json, warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import treams_rs as tr
        print(json.dumps({
            "warnings": [str(w.message) for w in caught
                         if w.category is tr.parallel.ThreadingWarning],
            "info": tr.thread_info(),
        }))
        """,
        {"TREAMS_RS_NUM_THREADS": "two", "RAYON_NUM_THREADS": "100000"},
    )
    result = json.loads(out)
    assert result["warnings"] == [
        'ignored TREAMS_RS_NUM_THREADS="two": expected a positive integer'
    ]
    info = result["info"]
    assert info["source"] == "RAYON_NUM_THREADS"
    assert info["threads"] == 65535  # Rayon's largest pool
    assert any("oversubscription" in d for d in info["diagnostics"])


def test_an_omp_limit_is_reported():
    info = info_in({"OMP_NUM_THREADS": "1"})
    assert info["threads"] == 1
    if info["available"] > 1:
        assert any(d.startswith("OMP_NUM_THREADS limits") for d in info["diagnostics"])


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_set_num_threads_takes_positive_integers(value):
    with pytest.raises(ValueError, match="positive integer"):
        tr.set_num_threads(value)


def test_oversubscription_warns_before_the_budget_changes():
    before = tr.thread_info()
    available = before["available"]
    with (
        pytest.warns(tr.parallel.ThreadingWarning, match="exceeds"),
        tr.threads(available + 1),
    ):
        assert tr.get_num_threads() == available + 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", tr.parallel.ThreadingWarning)
        with pytest.raises(tr.parallel.ThreadingWarning):
            tr.set_num_threads(available + 1)
        with pytest.raises(tr.parallel.ThreadingWarning), tr.threads(available + 1):
            pass
    after = tr.thread_info()
    assert (after["threads"], after["source"]) == (before["threads"], before["source"])


@OVERSUBSCRIBED
def test_threads_sizes_the_pool_and_restores_the_budget():
    before = tr.thread_info()
    z = np.linspace(0.5, 4.0, 4096) + 0.1j
    expected = tr.special.spherical_jn(3, z)
    with tr.threads(2):
        assert tr.get_num_threads() == 2
        assert tr.thread_info()["source"] == "set_num_threads"
        assert np.array_equal(tr.special.spherical_jn(3, z), expected)
        assert tr.thread_info()["pool_threads"] == 2
        with tr.threads(1):
            assert np.array_equal(tr.special.spherical_jn(3, z), expected)
        assert tr.get_num_threads() == 2
    after = tr.thread_info()
    assert (after["threads"], after["source"]) == (before["threads"], before["source"])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc thread names")
def test_parallel_work_runs_on_named_treams_threads():
    names = json.loads(
        run(
            """
            import json, os, time, numpy as np
            import treams_rs as tr
            tr.special.spherical_jn(2, np.linspace(0.1, 5, 5000))
            # A worker names itself when it first runs, which on a busy host
            # can come after the call that started the pool returned.
            deadline = time.monotonic() + 60
            while True:
                names = [open(f"/proc/self/task/{t}/comm").read().strip()
                         for t in os.listdir("/proc/self/task")]
                if sum(n.startswith("treams-") for n in names) >= 3:
                    break
                if time.monotonic() > deadline:
                    break
                time.sleep(0.01)
            print(json.dumps(sorted(names)))
            """,
            {"TREAMS_RS_NUM_THREADS": "3"},
        )
    )
    assert sorted(n for n in names if n.startswith("treams-")) == [
        "treams-0",
        "treams-1",
        "treams-2",
    ]


def test_one_thread_and_small_dense_algebra_stay_on_the_calling_thread():
    # A one-thread budget, and dense algebra below faer's own parallel sizes,
    # gain nothing from a hand-off to the pool.
    info = json.loads(
        run(
            """
            import json, numpy as np
            import treams_rs as tr
            rng = np.random.default_rng(0)
            a = rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))
            tr.diff.solve(a, a[:, :3])
            tr.diff.svdvals(a)
            tr.diff.eig(a)
            small = tr.thread_info()["pool_threads"]
            with tr.threads(1):
                tr.special.spherical_jn(2, np.linspace(0.1, 5, 5000))
            print(json.dumps([small, tr.thread_info()["pool_threads"]]))
            """,
            {"TREAMS_RS_NUM_THREADS": "4"},
        )
    )
    assert info == [None, None]


FORK = """
import json, multiprocessing as mp, sys
import numpy as np
import treams_rs as tr

def work(order):
    value = tr.special.spherical_jn(order, np.linspace(0.1, 5, 5000) + 0.05j)
    info = tr.thread_info()
    return value.tolist(), info["forked"], info["pool_threads"]

if __name__ == "__main__":
    parent = work(2)
    spheres = [tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=4)] * 2
    tr.Cluster(spheres, positions=[[0, 0, 0], [0, 0, 0.8]]).solve()
    with mp.get_context(sys.argv[1]).Pool(2) as pool:
        children = pool.map(work, [2, 2])
    print(json.dumps({
        "equal": all(c[0] == parent[0] for c in children),
        "forked": [c[1] for c in children],
        "pool": [c[2] for c in children],
        "budget": tr.get_num_threads(),
    }))
"""


@pytest.mark.parametrize("budget", [None, "1", "3"])
@pytest.mark.parametrize("method", ["fork", "forkserver", "spawn"])
def test_process_pools_work_after_parallel_work(budget, method, tmp_path):
    # Before the pool was owned, a child forked after a parallel call waited
    # forever for the parent's Rayon workers, which do not exist after fork.
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not available on this platform")
    script = tmp_path / "pool.py"
    script.write_text(FORK)
    base = {k: v for k, v in os.environ.items() if k not in ENVIRONMENT}
    if budget is not None:
        base["TREAMS_RS_NUM_THREADS"] = budget
    process = subprocess.Popen(
        [sys.executable, str(script), method],
        env=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    assert process.returncode == 0, stderr
    result = json.loads(stdout)
    assert result["equal"]
    # Every child runs a pool of the parent's budget; with one thread, the Bessel
    # functions stay on the calling thread.
    budget = result["budget"]
    assert result["pool"] == [None if budget == 1 else budget] * 2
    if method != "forkserver":
        assert result["forked"] == [method == "fork"] * 2


def test_threadpoolctl_limits_treams_rs():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    [entry] = [
        info for info in threadpoolctl.threadpool_info() if info["user_api"] == "treams"
    ]
    assert entry["internal_api"] == "treams_rs"
    assert entry["num_threads"] == tr.get_num_threads()
    before = tr.get_num_threads()
    with threadpoolctl.threadpool_limits(limits=1, user_api="treams"):
        assert tr.get_num_threads() == 1
    assert tr.get_num_threads() == before
