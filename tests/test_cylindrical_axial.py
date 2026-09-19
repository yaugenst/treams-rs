import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs import diff


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
def test_cylindrical_field_axial_pullbacks_and_operator(poltype, singular):
    origins = np.array([[0.1, -0.2, 0.0], [0.3, 0.4, -0.1]])
    basis = tr.CylindricalBasis.default([-0.3, 0.25], 2, 2, origins)
    ks = np.array(
        [1.3 + 0.05j, 1.5 + 0.07j] if poltype == "helicity" else [1.3 + 0.05j] * 2
    )
    points = np.array([[0.5, -0.3, 0.2], [0.8, 0.5, 0.3], [0.1, -0.2, 0.4]])
    if singular:
        points[-1, 0] += 0.5
    rng = np.random.default_rng(55)
    amplitudes = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    kwargs = {"poltype": poltype, "singular": singular}
    field, context = diff.field(amplitudes, points, basis, ks, **kwargs)
    g = rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)
    gradients = context.pullback_axial(g)
    _, context = diff.field(amplitudes, points, basis, ks, **kwargs)
    for actual, expected in zip(gradients[:4], context.pullback(g), strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    operator, context = diff.field_operator(points, basis, ks, **kwargs)
    assert_allclose(operator @ amplitudes, field, rtol=1e-13, atol=1e-12)
    operator_gradient = context.pullback_axial(g[:, :, None] * amplitudes.conj())
    for actual, expected in zip(operator_gradient, gradients[1:], strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
    direction = rng.uniform(-0.1, 0.1, size=len(basis))
    h = 1e-6
    perturbed = [
        tr.CylindricalBasis(
            [
                (p, kz + sign * h * step, m, pol)
                for (p, kz, m, pol), step in zip(basis.modes, direction, strict=True)
            ],
            origins,
        )
        for sign in (1, -1)
    ]
    difference = (
        diff.field(amplitudes, points, perturbed[0], ks, **kwargs)[0]
        - diff.field(amplitudes, points, perturbed[1], ks, **kwargs)[0]
    ) / (2 * h)
    assert_allclose(
        np.dot(gradients[-1], direction),
        np.vdot(g, difference).real,
        rtol=2e-7,
        atol=1e-7,
    )
    _, context = diff.field(amplitudes, points, basis, ks, **kwargs)
    amplitudes[:] = 0
    points[:] = 0
    ks[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback_axial(g[:1])
    for actual, expected in zip(
        context.pullback_axial(np.asfortranarray(g)), gradients, strict=True
    ):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@pytest.mark.parametrize(
    "name", ["field", "hfield", "gfield", "ffield", "field_operator"]
)
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_advect_field_family_axial(name, poltype):
    basis = tr.CylindricalBasis.default([-0.2, 0.3], 1)
    amplitudes = np.arange(len(basis)) * 0.1 + 0.2j
    points = np.array([[0.4, -0.3, 0.2], [0, 0, 0.3]])
    ks = np.array(
        [1.3 + 0.05j, 1.5 + 0.07j] if poltype == "helicity" else [1.3 + 0.05j] * 2
    )

    def objective(kzs):
        kwargs = {"basis": basis, "poltype": poltype, "kzs": kzs}
        if name == "field_operator":
            value = (
                ad.field_operator(points, basis.positions, ks, **kwargs) @ amplitudes
            )
        else:
            args = (amplitudes, points, basis.positions, ks)
            if name == "hfield":
                args = (*args, 0.8)
            if name in ("gfield", "ffield"):
                args = (1, *args)
            value = getattr(ad, name)(*args, **kwargs)
        return anp.sum(anp.abs(value) ** 2)

    kzs = basis.kz
    gradient = advect.grad(objective)(kzs)
    direction = np.linspace(-0.2, 0.3, len(basis))
    h = 1e-6
    assert_allclose(
        np.dot(gradient, direction),
        (objective(kzs + h * direction) - objective(kzs - h * direction)) / (2 * h),
        rtol=2e-7,
        atol=1e-8,
    )


@given(scale=st.floats(0.6, 1.8), kz=st.floats(-0.4, 0.4))
@settings(max_examples=25)
def test_cylindrical_axial_geometric_scale_invariance(scale, kz):
    basis = tr.CylindricalBasis.default([kz], 2)
    points = np.array([[0.4, -0.3, 0.2], [0, 0, 0.3]])
    ks = np.array([1.3 + 0.05j, 1.5 + 0.07j])
    amplitudes = np.full(len(basis), 0.2 + 0.3j)
    expected, _ = diff.field(amplitudes, points, basis, ks)
    scaled = tr.CylindricalBasis.default([kz / scale], 2)
    actual, context = diff.field(amplitudes, points * scale, scaled, ks / scale)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    gradient = context.pullback_axial(np.full_like(actual, 0.3 + 0.2j))
    assert_allclose(
        np.sum(gradient[1] * points * scale),
        np.vdot(gradient[3], ks / scale).real + np.dot(gradient[4], scaled.kz),
        rtol=1e-11,
        atol=1e-12,
    )


def test_complete_cylinder_scattered_field_axial_gradient():
    kzs = np.array([-0.2, 0.3])
    basis = tr.CylindricalBasis.default(kzs, 1)
    points = np.array([[0.6, 0.3, 0.2], [0.4, -0.5, 0.7]])
    illumination = np.linspace(0.1, 0.5, len(basis)) + 0.2j

    def objective(kzs, radius):
        response = ad.cylinder(
            kzs,
            1,
            1.3,
            anp.reshape(radius, (1,)),
            [3.1 + 0.1j, 1.1 + 0.04j],
            [1.2, 1.0],
            [0.07, 0.02],
        )
        medium = tr.Material(1.1 + 0.04j, 1, 0.02)
        value = ad.field(
            response @ illumination,
            points,
            basis.positions,
            medium.ks(1.3),
            basis=basis,
            singular=True,
            kzs=anp.repeat(kzs, 6),
        )
        return anp.sum(anp.abs(value) ** 2)

    gradient = advect.grad(objective, argnums=(0, 1))(kzs, np.array(0.2))
    h = 1e-6
    direction = np.array([0.1, -0.2])
    expected = (
        objective(kzs + h * direction, 0.2 + h * 0.1)
        - objective(kzs - h * direction, 0.2 - h * 0.1)
    ) / (2 * h)
    assert_allclose(
        np.dot(gradient[0], direction) + gradient[1] * 0.1,
        expected,
        rtol=2e-7,
        atol=1e-10,
    )


def test_axial_field_empty_samples_and_parameter_contract():
    basis = tr.CylindricalBasis.default([0.1, 0.3], 1)
    coefficients = np.ones(len(basis), complex)
    points = np.empty((0, 3))
    _, context = diff.field(coefficients, points, basis, [1.3, 1.3])
    for g in context.pullback_axial(np.empty((0, 3), complex)):
        assert_allclose(g, 0)
    _, context = diff.field_operator(points, basis, [1.3, 1.3])
    for g in context.pullback_axial(np.empty((0, 3, len(basis)), complex)):
        assert_allclose(g, 0)
    for kzs, message in [
        (np.ones(len(basis) - 1), "one real"),
        (np.zeros(len(basis)), "distinct"),
        (basis.kz + 0.1j, "real"),
    ]:
        with pytest.raises(ValueError, match=message):
            ad.field_operator(points, basis.positions, [1.3, 1.3], basis=basis, kzs=kzs)
    spherical = tr.SphericalBasis.default(1)
    with pytest.raises(ValueError, match="cylindrical"):
        ad.field_operator(
            points,
            spherical.positions,
            [1.3, 1.3],
            basis=spherical,
            kzs=np.ones(len(spherical)),
        )


def _move_groups(basis, labels, values, positions=None):
    return tr.CylindricalBasis(
        [
            (p, values[np.searchsorted(labels, kz)], m, pol)
            for p, kz, m, pol in basis.modes
        ],
        basis.positions if positions is None else positions,
    )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
def test_expansion_shared_axial_groups(poltype, singular):
    destination = tr.CylindricalBasis.default(
        [-0.2, 0.3, 1.7], 2, positions=[[0.4, 0.3, -0.2]]
    )[1:-1]
    source = tr.CylindricalBasis.default(
        [-0.2, 0.3, 1.7, 0.1], 1, positions=[[0.1, -0.2, 0.1]]
    )
    labels = np.unique(np.concatenate((destination.kz, source.kz)))
    ks = np.array(
        [1.3 + 0.05j, 1.5 + 0.07j] if poltype == "helicity" else [1.3 + 0.05j] * 2
    )
    kwargs = {"poltype": poltype, "singular": singular}
    value, context = diff.expansion(destination, source, ks, **kwargs)
    rng = np.random.default_rng(61)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = context.pullback_axial(g)
    fixed = diff.expansion(destination, source, ks, **kwargs)[1].pullback(g)
    for actual, expected in zip(gradients[:3], fixed, strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
    direction = rng.uniform(-0.1, 0.1, len(labels))
    h = 1e-6

    def objective(values):
        return np.vdot(
            g,
            diff.expansion(
                _move_groups(destination, labels, values),
                _move_groups(source, labels, values),
                ks,
                **kwargs,
            )[0],
        ).real

    assert_allclose(
        np.dot(gradients[3], direction),
        (objective(labels + h * direction) - objective(labels - h * direction))
        / (2 * h),
        rtol=2e-7,
        atol=1e-7,
    )
    assert gradients[3][np.searchsorted(labels, 0.1)] == 0
    assert_allclose(
        np.sum(gradients[0] * destination.positions)
        + np.sum(gradients[1] * source.positions),
        np.vdot(gradients[2], ks).real + np.dot(gradients[3], labels),
        rtol=2e-12,
        atol=1e-10,
    )
    context = diff.expansion(destination, source, ks, **kwargs)[1]
    ks[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback_axial(g[:-1])
    with pytest.raises(ValueError, match="finite"):
        context.pullback_axial(np.full_like(g, np.nan))
    for actual, expected in zip(
        context.pullback_axial(np.asfortranarray(g)), gradients, strict=True
    ):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback_axial(g)


@given(scale=st.floats(0.6, 1.8), kz=st.floats(-0.4, 0.4))
@settings(max_examples=25)
def test_expansion_axial_scale_and_partition_invariance(scale, kz):
    destination = tr.CylindricalBasis.default(
        [kz, 0.7], 1, positions=[[0.3, 0.2, -0.1]]
    )
    source = tr.CylindricalBasis.default([kz, 0.7], 1, positions=[[-0.1, -0.2, 0.3]])
    labels = np.array([kz, 0.7])
    ks = np.array([1.3 + 0.05j, 1.5 + 0.07j])
    value = diff.expansion(destination, source, ks)[0]
    scaled, context = diff.expansion(
        _move_groups(
            destination, labels, labels / scale, destination.positions * scale
        ),
        _move_groups(source, labels, labels / scale, source.positions * scale),
        ks / scale,
    )
    assert_allclose(scaled, value, rtol=1e-12, atol=1e-12)
    g = context.pullback_axial(np.full_like(value, 0.3 + 0.2j))
    assert_allclose(
        np.sum(g[0] * destination.positions * scale)
        + np.sum(g[1] * source.positions * scale),
        np.vdot(g[2], ks / scale).real + np.dot(g[3], labels / scale),
        atol=1e-12,
    )
    # Exchanging the numeric order of groups preserves their identity in Advect.
    values = np.array([0.7, kz])

    def loss(v):
        return anp.real(
            anp.sum(
                ad.expansion(
                    destination.positions,
                    source.positions,
                    ks,
                    destination=destination,
                    source=source,
                    kzs=v,
                )
            )
        )

    gradient = advect.grad(loss)(values)
    finite = (
        loss(values + 1e-6 * np.array([0.2, -0.1]))
        - loss(values - 1e-6 * np.array([0.2, -0.1]))
    ) / 2e-6
    assert_allclose(np.dot(gradient, [0.2, -0.1]), finite, rtol=1e-6, atol=2e-9)


def test_expansion_axial_parallel_matches_column_contractions():
    destination = tr.CylindricalBasis.default(
        [-0.2, 0.3], 3, positions=[[0.4, 0.3, 0.2]]
    )
    source = tr.CylindricalBasis.default([-0.2, 0.3], 8)
    ks = [1.3 + 0.05j, 1.5 + 0.07j]
    value, context = diff.expansion(destination, source, ks)
    g = np.arange(value.size).reshape(value.shape) / value.size + 0.3j
    parallel = context.pullback_axial(g)
    serial = [
        diff.expansion(destination, source[a:b], ks)[1].pullback_axial(g[:, a:b])
        for a, b in [(0, 34), (34, 68)]
    ]
    # Destination includes both groups, so each chunk has the same gradient shape.
    for actual, left, right in zip(parallel, *serial, strict=True):
        assert_allclose(actual, left + right, rtol=1e-12, atol=1e-12)


def test_complete_cylinder_interaction_field_axial_gradient():
    labels = np.array([-0.2, 0.3])
    positions = np.array([[0, 0, 0], [0.8, 0.1, 0.2]])
    basis = tr.CylindricalBasis.default(labels, 1, 2, positions)
    coefficients = np.linspace(0.1, 0.3, len(basis)) + 0.2j

    def objective(kzs):
        t = ad.cylinder(kzs, 1, 1.3, [0.13], [2.3 + 0.02j, 1])
        zero = anp.zeros_like(t)
        local = anp.concatenate(
            (anp.concatenate((t, zero), axis=1), anp.concatenate((zero, t), axis=1)),
            axis=0,
        )
        coupling = ad.expansion(
            positions,
            positions,
            [1.3, 1.3],
            destination=basis,
            source=basis,
            singular=True,
            kzs=kzs,
        )
        scattered = ad.interaction(local, coupling) @ coefficients
        field = ad.field(
            scattered,
            [[0.4, 0.5, 0.3]],
            positions,
            [1.3, 1.3],
            basis=basis,
            singular=True,
            kzs=anp.tile(anp.repeat(kzs, 6), 2),
        )
        return anp.sum(anp.abs(field) ** 2)

    gradient = advect.grad(objective)(labels)
    h = 1e-6
    direction = np.array([0.2, -0.1])
    assert_allclose(
        np.dot(gradient, direction),
        (objective(labels + h * direction) - objective(labels - h * direction))
        / (2 * h),
        rtol=2e-6,
        atol=1e-10,
    )


def test_expansion_axial_requires_fixed_distinct_groups():
    basis = tr.CylindricalBasis.default([-0.2, 0.3], 1)
    kwargs = {"destination": basis, "source": basis}
    for kzs, match in [
        ([0.1], "per axial group"),
        ([0.1, 0.1], "distinct"),
        ([0.1j, 0.3], "real"),
        ([np.nan, 0.3], "finite"),
    ]:
        with pytest.raises(ValueError, match=match):
            ad.expansion([[0.2, 0, 0]], [[0, 0, 0]], [1.3, 1.3], kzs=kzs, **kwargs)
    # Positive and negative zero denote one physical matching group.
    destination = tr.CylindricalBasis([(0, -0.0, 1, 0)], [[0.2, 0.1, 0.3]])
    source = tr.CylindricalBasis([(0, 0.0, -1, 0)])
    value, context = diff.expansion(destination, source, [1.3, 1.3])
    assert context.pullback_axial(np.ones_like(value))[3].shape == (1,)
    sphere = tr.SphericalBasis.default(1)
    value, context = diff.expansion(sphere, sphere, [1.3, 1.3])
    with pytest.raises(ValueError, match="two cylindrical"):
        context.pullback_axial(np.ones_like(value))


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("chiral", [False, True])
def test_periodic_expansion_axial_groups_and_geometric_scale(dim, chiral):
    destination = tr.CylindricalBasis.default(
        [-0.2, 0.3], 1, positions=[[0.3, 0.2, -0.1]]
    )
    source = tr.CylindricalBasis.default([-0.2, 0.3], 1, positions=[[-0.1, -0.2, 0.3]])
    labels = np.array([-0.2, 0.3])
    ks = np.array([1.3 + 0.05j, 1.5 + 0.07j] if chiral else [1.3 + 0.05j] * 2)
    vectors = np.array([[1.7]]) if dim == 1 else np.array([[1.7, 0.2], [0.1, 1.9]])
    bloch = np.array([0.1]) if dim == 1 else np.array([0.1, -0.15])
    value, context = tr.lattice.expansion_with_context(
        destination, source, ks, vectors, bloch
    )
    rng = np.random.default_rng(99)
    cotangent = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    g = context.pullback_axial(cotangent)
    fixed = tr.lattice.expansion_with_context(destination, source, ks, vectors, bloch)[
        1
    ].pullback(cotangent)
    for actual, expected in zip(g[:5], fixed, strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def objective(kzs):
        value = tr.lattice.expansion_with_context(
            _move_groups(destination, labels, kzs),
            _move_groups(source, labels, kzs),
            ks,
            vectors,
            bloch,
        )[0]
        return np.vdot(cotangent, value).real

    direction = np.array([0.2, -0.1])
    h = 1e-6
    assert_allclose(
        np.dot(g[5], direction),
        (objective(labels + h * direction) - objective(labels - h * direction))
        / (2 * h),
        rtol=3e-7,
        atol=1e-8,
    )
    assert_allclose(
        np.sum(g[0] * destination.positions)
        + np.sum(g[1] * source.positions)
        + np.sum(g[4] * vectors),
        np.vdot(g[2], ks).real + np.dot(g[3], bloch) + np.dot(g[5], labels),
        rtol=2e-10,
        atol=1e-10,
    )

    def loss(kzs):
        value = ad.lattice_expansion(
            destination.positions,
            source.positions,
            ks,
            bloch,
            vectors,
            destination=destination,
            source=source,
            kzs=kzs,
        )
        return anp.real(anp.sum(anp.conj(cotangent) * value))

    assert_allclose(advect.grad(loss)(labels), g[5], rtol=1e-11, atol=1e-11)
    exchanged = labels[::-1]
    assert_allclose(
        np.dot(advect.grad(loss)(exchanged), direction),
        (loss(exchanged + h * direction) - loss(exchanged - h * direction)) / (2 * h),
        rtol=3e-7,
        atol=1e-8,
    )
    context = tr.lattice.expansion_with_context(
        destination, source, ks, vectors, bloch
    )[1]
    ks[:] = 0
    vectors[:] = 0
    bloch[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback_axial(cotangent[:-1])
    for actual, expected in zip(context.pullback_axial(cotangent), g, strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(cotangent)


@given(kz=st.floats(-0.3, 0.3), scale=st.floats(0.7, 1.4))
@settings(max_examples=15, deadline=None)
def test_periodic_axial_self_images_scale_and_split(kz, scale):
    ks = np.array([1.3 + 0.05j, 1.5 + 0.07j])

    def evaluate(factor, eta):
        return tr.lattice.expansion_with_context(
            tr.CylindricalBasis.default([kz / factor], 1),
            tr.CylindricalBasis.default([kz / factor], 1),
            ks / factor,
            [[1.7 * factor]],
            [0.1 / factor],
            eta=eta,
        )

    value, context = evaluate(1, 0.9)
    scaled, scaled_context = evaluate(scale, 1.1)
    assert_allclose(scaled, value, rtol=1e-11, atol=1e-11)
    g = context.pullback_axial(np.ones_like(value))
    gs = scaled_context.pullback_axial(np.ones_like(value))
    assert_allclose(gs[5] / scale, g[5], rtol=1e-10, atol=1e-11)
    assert_allclose(
        np.sum(g[4] * 1.7),
        np.vdot(g[2], ks).real + np.dot(g[3], [0.1]) + g[5][0] * kz,
        atol=1e-10,
    )
