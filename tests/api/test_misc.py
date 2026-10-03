"""treams_rs.misc and the Material helpers against treams.misc.

Checks refractive_index and wave_vec_z (with Material.nmp and Material.kzs),
basischange and pickmodes, and the firstbrillouin1d/2d/3d reductions with their
lattice-coordinate and Voronoi-cell properties.
"""

import numpy as np
import pytest
import treams.misc as upstream_misc
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from treams_rs import Material, lattice, misc

from _support import finite_complex


@pytest.mark.interface
@pytest.mark.reference
@pytest.mark.parametrize(
    "epsilon,mu,kappa",
    [(3.0, 1.2, 0.1), (3 + 0.2j, 1 - 0.1j, 0.2 + 0.3j), (-2 + 0.1j, 1, -0.3j)],
)
def test_material_helpers_and_shared_material_path(epsilon, mu, kappa):
    expected = upstream_misc.refractive_index(epsilon, mu, kappa)
    value = misc.refractive_index(epsilon, mu, kappa)
    assert_allclose(value, expected, rtol=2e-15)
    assert value.dtype == expected.dtype
    assert_allclose(Material(epsilon, mu, kappa).nmp, value)
    kx = np.linspace(-2, 2, 27)
    ky = 0.3
    k = np.asarray(1.3 * value[0], complex)
    assert_allclose(
        misc.wave_vec_z(kx, ky, k),
        upstream_misc.wave_vec_z(kx, ky, k),
        rtol=3e-15,
        atol=1e-14,
    )
    assert_allclose(
        Material(epsilon, mu, kappa).kzs(1.3, kx, ky, 0), misc.wave_vec_z(kx, ky, k)
    )


@pytest.mark.interface
@given(
    epsilon=st.floats(0.1, 5),
    mu=st.floats(0.1, 5),
    kappa=st.floats(-0.5, 0.5),
    sign=st.sampled_from([1.0, -1.0]),
    transverse=st.tuples(st.floats(-3, 3), st.floats(-3, 3)),
)
def test_real_material_loops_match_complex_loops(epsilon, mu, kappa, sign, transverse):
    # float64 arguments select separately written real loops; they must agree
    # with the complex loops wherever those are real (epsilon mu > 0).
    epsilon, mu = sign * epsilon, sign * mu
    real = misc.refractive_index(epsilon, mu, kappa)
    assert real.dtype == np.float64
    expected = misc.refractive_index(complex(epsilon), complex(mu), complex(kappa))
    assert_allclose(real, expected, rtol=1e-15, atol=0)
    k = abs(real[1]) + 0.1
    kz = misc.wave_vec_z(*transverse, k)
    assert_allclose(kz, misc.wave_vec_z(*map(complex, (*transverse, k))), rtol=1e-15)
    assert kz.imag >= 0
    assert_allclose(sum(np.square([*transverse, kz])), k * k, rtol=1e-13, atol=1e-13)


@pytest.mark.reference
def test_misc_scalar_axis_limits_and_broadcast():
    assert misc.wave_vec_z(0, 0, 0) == 0
    assert misc.wave_vec_z(1, 0, 1) == 0
    assert misc.wave_vec_z(2, 0, 1).imag > 0
    assert_allclose(
        misc.refractive_index(np.array([[2], [3]]), 1, np.array([0.1, 0.2, 0.3])),
        upstream_misc.refractive_index(
            np.array([[2], [3]]), 1, np.array([0.1, 0.2, 0.3])
        ),
    )


@pytest.mark.physics
@pytest.mark.reference
@given(count=st.integers(0, 20))
def test_mode_change_involution_and_selection(count):
    columns = (
        np.array([(i, 0, pol) for i in range(count) for pol in (0, 1)]).reshape(-1, 3).T
    )
    change = misc.basischange(columns)
    assert_allclose(change @ change, np.eye(2 * count), atol=3e-16)
    assert_array_equal(misc.pickmodes(columns, columns), np.eye(2 * count, dtype=bool))
    if count:
        subset = columns[:, ::3]
        assert_allclose(
            misc.basischange(subset, columns),
            upstream_misc.basischange(subset, columns),
        )
        assert_array_equal(
            misc.pickmodes(subset, columns), upstream_misc.pickmodes(subset, columns)
        )
    for invalid in (columns[:2], columns[0]):
        with pytest.raises(ValueError, match="three or four equally sized"):
            misc.pickmodes(invalid, columns)


@pytest.mark.physics
@pytest.mark.reference
@given(k=st.floats(-100, 100), b=st.floats(0.5, 5), shift=st.integers(-20, 20))
def test_brillouin_scalar_periodicity(k, b, shift):
    result = misc.firstbrillouin1d(k, b)
    assert -0.5 * b < result <= 0.5 * b
    shifted = misc.firstbrillouin1d(k + shift * b, b)
    # Equivalent representatives may lie on opposite sides of a floating-point cell boundary.
    assert abs(result - shifted) < 5e-13 or abs(abs(result - shifted) - b) < 5e-13
    assert_allclose(result, upstream_misc.firstbrillouin1d(k, b), atol=3e-14)


@pytest.mark.interface
@pytest.mark.reference
@pytest.mark.parametrize("dim", [2, 3])
def test_brillouin_native_reference_and_input_ownership(dim):
    b = np.eye(dim) * np.arange(1, dim + 1)
    if dim == 2:
        b[1, 0] = 0.2
    k = np.array([7.1, -6.2, 4.5])[:dim]
    original = k.copy()
    function = getattr(misc, f"firstbrillouin{dim}d")
    expected = getattr(upstream_misc, f"firstbrillouin{dim}d")(k.copy(), b.copy())
    result = function(k, b)
    assert_allclose(result, expected, atol=3e-14)
    assert_array_equal(k, original)
    assert_allclose(function(result, b), result, atol=3e-14)


@st.composite
def reduced_reciprocal_bases(draw):
    """Lagrange-Gauss reduced 2D bases: |b1| <= |b2|, |b1 . b2| <= |b1|^2 / 2."""
    short = draw(st.floats(0.5, 3))
    long = short * draw(st.floats(1, 3))
    cosine = draw(st.floats(-1, 1)) * short / (2 * long)
    angle = draw(st.sampled_from([-1, 1])) * np.arccos(cosine)
    turn = draw(st.floats(-np.pi, np.pi))
    return np.array(
        [
            [short * np.cos(turn), short * np.sin(turn)],
            [long * np.cos(turn + angle), long * np.sin(turn + angle)],
        ]
    )


def _lattice_coordinates(b, displacement):
    """Coordinates n of a displacement n @ b, rows of b being the basis."""
    return np.linalg.solve(np.transpose(b), displacement)


@pytest.mark.physics
@given(
    b=reduced_reciprocal_bases(),
    k=st.tuples(st.floats(-20, 20), st.floats(-20, 20)),
)
def test_brillouin_2d_returns_the_shortest_equivalent_vector(b, k):
    folded = misc.firstbrillouin2d(k, b)
    shift = _lattice_coordinates(b, np.subtract(k, folded))
    assert_allclose(shift, np.round(shift), rtol=0, atol=1e-9)
    # A point of the Voronoi cell of the origin (the first Brillouin zone).
    neighbours = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 2), -1).reshape(-1, 2)
    distances = np.linalg.norm(folded + neighbours @ b, axis=-1)
    assert np.linalg.norm(folded) <= distances.min() + 1e-12 * np.abs(b).max()
    assert_allclose(misc.firstbrillouin2d(folded, b), folded, rtol=0, atol=3e-14)


@pytest.mark.physics
@given(
    seed=st.integers(0, 2**16),
    k=st.tuples(*(st.floats(-20, 20) for _ in range(3))),
)
def test_brillouin_3d_folds_to_an_equivalent_shorter_vector(seed, k):
    # Like upstream, the 3D reduction may stop outside the first zone of a
    # skewed cell, so only equivalence and a non-increasing length hold.
    b = np.random.default_rng(seed).normal(size=(3, 3)) + 2 * np.eye(3)
    folded = misc.firstbrillouin3d(k, b)
    shift = _lattice_coordinates(b, np.subtract(k, folded))
    assert_allclose(shift, np.round(shift), rtol=0, atol=1e-9)
    assert np.linalg.norm(folded) <= np.linalg.norm(k) * (1 + 1e-14)


@pytest.mark.physics
@given(
    k=finite_complex(min_magnitude=0.1, max_magnitude=5).map(
        lambda k: complex(k.real, abs(k.imag))
    ),
    kx=st.floats(-5, 5),
    ky=st.floats(-5, 5),
)
def test_normal_wavenumber_branch(k, kx, ky):
    # kz^2 = k^2 - kx^2 - ky^2 on the decaying or outgoing branch.
    kz = misc.wave_vec_z(kx, ky, k)
    assert_allclose(
        kz**2, k**2 - kx**2 - ky**2, rtol=0, atol=1e-14 * (abs(k) ** 2 + kx**2 + ky**2)
    )
    assert kz.imag >= 0
    assert kz.imag > 0 or kz.real >= 0


@pytest.mark.physics
@given(
    epsilon=finite_complex(max_magnitude=5).map(lambda z: complex(z.real, abs(z.imag))),
    mu=finite_complex(max_magnitude=5).map(lambda z: complex(z.real, abs(z.imag))),
    kappa=st.floats(-1, 1),
    loss=st.floats(-0.9, 0.9),
)
def test_helicity_indices_of_passive_media(epsilon, mu, kappa, loss):
    # n_-/+ = s (sqrt(eps mu) -/+ kappa), with the sign s that makes the mean
    # index s sqrt(eps mu) passive; a chirality loss below that of the mean
    # index keeps both helicity indices strictly passive. (At equality, one
    # index is real and rounding decides its sign.)
    root = np.sqrt(epsilon * mu)
    assume(abs(root) > 1e-3)
    assume(root.imag == 0 or abs(root.imag) > 1e-6 * abs(root))
    kappa = kappa + 1j * loss * abs(root.imag)
    negative, positive = misc.refractive_index(epsilon, mu, kappa)
    mean = (negative + positive) / 2
    assert_allclose(mean**2, epsilon * mu, rtol=1e-13, atol=0)
    # The difference of the indices is exact up to their rounding.
    assert_allclose(
        (positive - negative) * root,
        2 * kappa * mean,
        rtol=1e-12,
        atol=1e-15 * abs(epsilon * mu),
    )
    assert mean.imag >= 0
    assert negative.imag >= -1e-15 * abs(negative)
    assert positive.imag >= -1e-15 * abs(positive)


@pytest.mark.physics
@pytest.mark.parametrize("pitch", [0.3, 0.5, 1.0, 1e-7])
def test_brillouin_accepts_hexagonal_lattices_at_any_pitch(pitch):
    b = lattice.reciprocal(pitch * np.array([[1, 0], [0.5, np.sqrt(3) / 2]]))
    k = np.array([7.1, -6.2]) / pitch
    result = misc.firstbrillouin2d(k, b)
    cells = (k - result) @ lattice.reciprocal(b).T / (2 * np.pi)
    assert_allclose(cells, np.round(cells), atol=1e-9)
    neighbours = np.array([(m, n) for m in (-1, 0, 1) for n in (-1, 0, 1)]) @ b
    lengths = np.linalg.norm(result + neighbours, axis=1)
    assert np.linalg.norm(result) <= lengths.min() * (1 + 1e-12)
