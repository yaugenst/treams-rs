"""Saved lattice contexts preserve derivatives without reevaluating their values."""

import numpy as np
import pytest

import treams_rs as tr

from _support import assert_saved_context

pytestmark = pytest.mark.interface


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
def test_lattice_sum_saved_state(spherical, broadcast):
    coordinates = 3 if spherical else 2
    parameters = (
        np.array([2.1 + 0.2j, 2.3 + 0.1j]) if broadcast else np.asarray(2.1 + 0.2j),
        np.array([0.1, 0.15]),
        np.array([[1.6, 0.1], [3.25, 1.7]]),
        np.array([0.19, 0.11, 0.07])[:coordinates],
        np.asarray(0.9 + 0.03j),
    )
    value, context = tr.diff.lattice_sum(2, 2, -1, *parameters, spherical=spherical)
    size = type(context)._state_spec(
        list(value.shape), [list(p.shape) for p in parameters], 2, coordinates
    )
    assert_saved_context(
        context,
        tuple(np.full_like(p, 0.03) for p in parameters),
        np.full_like(value, 0.3 + 0.2j),
        size=size,
        rtol=1e-13,
    )


@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_periodic_expansion_saved_state(cylindrical, shared):
    destination_positions = np.array([[0.12, 0.08, 0.15]])
    source_positions = np.array([[0.03, -0.09, -0.04]])
    if cylindrical:
        destination = tr.CylindricalBasis(
            [(0, 0.2, -1, 0), (0, 0.2, 1, 1)], positions=destination_positions
        )
        source = tr.CylindricalBasis(
            [(0, 0.2, 0, 0), (0, 0.2, 0, 1)], positions=source_positions
        )
    else:
        destination = tr.SphericalBasis(
            [(0, 1, -1, 0), (0, 1, 0, 1)], positions=destination_positions
        )
        source = tr.SphericalBasis(
            [(0, 1, 0, 0), (0, 1, 1, 1)], positions=source_positions
        )
    parameters = (
        destination_positions,
        source_positions,
        np.array([1.3 + 0.05j, 1.3 + 0.05j if shared else 1.5 + 0.07j]),
        np.array([0.13, 0.07]),
        np.array([[1.7, 0.08], [3.38, 1.6]]),
    )
    value, context = tr.diff.lattice_expansion(
        destination, source, *parameters[2:], eta=0.9
    )
    directions = tuple(np.full_like(p, 0.03) for p in parameters)
    if cylindrical:
        directions += (np.array([0.04]),)
    size = type(context)._state_spec(2, 2, 1, 1, cylindrical)
    assert_saved_context(
        context,
        directions,
        np.full_like(value, 0.3 + 0.2j),
        size=size,
        suffix="_axial" if cylindrical else "",
        rtol=1e-13,
    )


@pytest.mark.parametrize("poltype,channels", [("helicity", 2), ("parity", 1)])
def test_lattice_table_saved_state(poltype, channels):
    destination = tr.SphericalBasis(
        [(0, 1, -1, 0), (1, 2, 1, 1)], positions=[[0, 0, 0], [0.2, 0.3, 0.4]]
    )
    source = tr.SphericalBasis([(0, 2, -2, 0), (0, 1, 0, 1)])
    table = np.full((2, 1, channels, 25), 0.2 + 0.1j)
    value, context = tr.diff.lattice_expansion_from_table(
        table, destination, source, poltype=poltype
    )
    size = type(context)._state_spec(2, 2, 2, 1)
    assert_saved_context(
        context,
        (np.full_like(table, 0.03 + 0.02j),),
        np.full_like(value, 0.3 + 0.2j),
        size=size,
        rtol=1e-13,
    )
