"""Suite-wide pytest configuration.

pytest imports this initial conftest before any test module, so the Hypothesis
profile below applies to every property no matter which files are selected or in
which order they are collected. Hypothesis binds its settings when ``@given``
decorates a test, which is why the profile must be loaded here and never as a
side effect of importing a test module.

Every test also carries at least one category marker. ``CATEGORIES`` is the
only list of their names and descriptions: ``pytest_configure`` registers them,
and tests/test_suite_rules.py enforces the rule for everything collected.
"""

import os
import sys

import pytest
from hypothesis import settings

# Every profile derives from the settings active at import: Hypothesis's own "ci"
# profile (derandomized, no example database) when it detects CI, otherwise its
# stock default. Debug-build timings vary too much for a per-example deadline.
# Hosted CI selects "ci", which replaces Hypothesis's profile of that name and
# keeps the stock budget of 100 examples.
settings.register_profile("dev", max_examples=30, deadline=None, print_blob=True)
settings.register_profile("ci", settings.get_profile("dev"), max_examples=100)
settings.register_profile("thorough", settings.get_profile("dev"), max_examples=300)
# `--hypothesis-profile=<name>` on the command line takes precedence.
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

#: Category marker names and descriptions; docs/development/testing.md explains them.
CATEGORIES = {
    "physics": "analytic and conservation-law validation",
    "gradients": "native pullback correctness",
    "interface": "Python interface behavior",
    "workflows": "complete scattering workflows",
    "reference": "comparison to independent numerical references",
}
_UNCATEGORIZED = pytest.StashKey[list[str]]()


def pytest_configure(config):
    """Register ``CATEGORIES`` before collection, which ``--strict-markers`` checks."""
    for name, description in CATEGORIES.items():
        config.addinivalue_line("markers", f"{name}: {description}")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Record collected tests without a category before `-m` deselects any."""
    config.stash[_UNCATEGORIZED] = [
        item.nodeid
        for item in items
        if not any(item.get_closest_marker(name) for name in CATEGORIES)
    ]


@pytest.fixture
def category_markers():
    """Names of the category markers every test must carry at least one of."""
    return tuple(CATEGORIES)


@pytest.fixture
def uncategorized_tests(pytestconfig):
    """Node ids of collected tests that carry none of ``CATEGORIES``."""
    return pytestconfig.stash[_UNCATEGORIZED]


_GLOBAL_POOL_UNUSED: list[bool] = []


def pytest_sessionfinish(session, exitstatus):
    """Fail the session if native code started Rayon's global pool.

    Every parallel region must run on the treams-rs pool
    (``treams_core::threads``): the global pool cannot be resized, and a child
    forked after it started hangs. The probe starts that pool, so it runs once
    per interpreter, and only when the extension was loaded.
    """
    native = sys.modules.get("treams_rs._native")
    if native is None or not hasattr(native, "rayon_global_pool_unused"):
        return
    if not _GLOBAL_POOL_UNUSED:
        _GLOBAL_POOL_UNUSED.append(native.rayon_global_pool_unused())
    if not _GLOBAL_POOL_UNUSED[0]:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "FAILED: native code started Rayon's global pool; run parallel "
                "work inside treams_core::threads::install",
                red=True,
            )
