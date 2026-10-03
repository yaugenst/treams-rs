"""Framework bridges route native pullbacks exactly.

A bridge owns argument order, the complex convention, projection onto real
inputs, output structure, shapes and dtypes; the native pullbacks themselves
are checked against finite differences at the native layer. Each case therefore
compares a framework's gradient of L = sum_i Re(vdot(w_i, output_i)) with the
native pullback of the same cotangents, to rounding. Advect, JAX and PyTorch
all use Re(vdot(gradient, dx)) for real objectives, except that JAX reports
the conjugate of that gradient for complex inputs.
"""

import functools
import importlib

import advect
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import SphericalBasis, diff

from _support import complex_normal, jax_x64

pytestmark = pytest.mark.gradients

FRAMEWORKS = ("advect", "jax", "torch")


def wrap(record):
    """Framework calls through ``wrap``; JAX fixes shapes from example values."""
    return {
        "jax": lambda ns, values: ns.wrap(record, *values),
        "torch": lambda ns, values: ns.wrap(record),
    }


def call(name, *args, frameworks=FRAMEWORKS, **kwargs):
    """Framework calls of the public function ``name`` with static arguments."""
    return {
        framework: lambda ns, _: functools.partial(getattr(ns, name), *args, **kwargs)
        for framework in frameworks
    }


def sphere_cluster(k0, radii, epsilon, positions):
    return diff.sphere_cluster(1, float(k0), radii, epsilon, positions)


def sphere(k0, radii, epsilon, mu, kappa):
    return diff.sphere(1, float(k0), radii, epsilon, mu, kappa)


def sphere_defaults(k0, radii, epsilon):
    output, context = diff.sphere(1, float(k0), radii, epsilon)
    return output, lambda g: context.pullback(g)[:3]


def plane_field(coefficients, points, vectors):
    return diff.plane_field(coefficients, points, vectors, [0, 1])


def plane_operator(points, vectors):
    output, context = diff.plane_field(None, points, vectors, [0, 1])
    return output, lambda g: context.pullback(g)[1:]


def lattice_sum(k, kpar, a, r, eta):
    return diff.lattice_sum(2, [2, 3], -1, k, kpar, a, r, eta)


def lattice_sum_real(k, kpar, a, r, eta):
    return diff.lattice_sum(1, [2, 3], -1, k, kpar, a, r, eta, part="real")


ORDERS = np.array([0.0, 1.0, 2.0])
BASIS = SphericalBasis.default(1)


def bessel(z):
    return diff.bessel(ORDERS, z)


def rotation(angles):
    # The native pullback returns the three Euler derivatives as one list.
    return diff.rotation(angles, BASIS)


def sph_harm(theta, phi):
    # The native pullback returns one gradient per argument as a tuple.
    return diff.sph_harm(theta, phi, degree=5, order=-2)


def mie(sizes, epsilon, mu, kappa):
    return diff.mie(2, sizes, epsilon, mu, kappa)


def mie_cyl(kz, k0, radii, epsilon, mu, kappa):
    return diff.mie_cyl(float(kz), -1, float(k0), radii, epsilon, mu, kappa)


RNG = np.random.default_rng(28)
LOCAL = np.array([[0.2 + 0.1j, 0.03j], [0.05, 0.3 - 0.1j]])
COUPLING = np.array([[0.0, 0.1 + 0.2j], [-0.03j, 0.0]])
OPERATOR = np.array([[2.3, 0.2], [-0.1, 1.7]])
RHS = np.array([[0.3, 0.1, 0.2], [0.5, -0.2, 0.7]])
EIGEN = np.diag(np.arange(1.0, 5.0)) + 0.15 * complex_normal(RNG, (4, 4))
STACKS = [0.05 * complex_normal(RNG, (2, 2, 2, 2)) + np.eye(2) for _ in range(2)]
POINTS = np.array([[0.2, 0.3, -0.1], [0.3, -0.2, 0.1]])
VECTORS = np.array([[0.3, 0.2, 1.3 + 0.1j], [1.5, -0.1, 0.2j]])
SITES = np.array([[0.19, 0.11, 0.07], [0.21, -0.09, 0.05]])
WAVENUMBERS = np.array([[2.1 + 0.2j], [2.3 + 0.1j]])
LAYERS = (
    np.array([0.18, 0.3]),
    np.array([3.2 + 0.1j, 2.1 + 0.05j, 1.0 + 0j]),
    np.array([1.1 + 0.02j, 1.2 + 0.01j, 1.0 + 0j]),
    np.array([0.1 + 0.01j, 0.05 + 0.02j, 0.0 + 0j]),
)

#: name -> (native record, primal values, framework calls, used outputs or None)
BRIDGES = {
    "solve": (diff.solve, (OPERATOR + 0.1j, RHS + 0.13j), call("solve"), None),
    "solve-real": (
        diff.solve,
        (OPERATOR, RHS),
        wrap(diff.solve) | call("solve", frameworks=("advect",)),
        None,
    ),
    "solve-real-public": (
        diff.solve,
        (OPERATOR, RHS),
        call("solve", frameworks=("jax",)),
        None,
    ),
    "interaction": (diff.interaction, (LOCAL, COUPLING), call("interaction"), None),
    "illuminate": (
        diff.illuminate,
        (LOCAL, COUPLING, np.array([[0.4 + 0.1j, 0.2, 0.3j], [-0.2j, 0.1j, -0.1]])),
        call("illuminate"),
        None,
    ),
    "sphere": (sphere, (np.asarray(1.2), *LAYERS), call("sphere", 1), None),
    "sphere-default-materials": (
        sphere_defaults,
        (np.asarray(1.2), np.array([0.2]), np.array([3.0, 1.0])),
        call("sphere", 1),
        None,
    ),
    "sphere-cluster": (
        sphere_cluster,
        (
            np.asarray(1.2),
            np.array([0.2, 0.25]),
            np.array([3.0 + 0.1j, 2.3 + 0.2j]),
            np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 1.5]]),
        ),
        wrap(sphere_cluster) | call("sphere_cluster", 1, frameworks=("advect",)),
        None,
    ),
    **{
        f"eig-{used}": (
            diff.eig,
            (EIGEN,),
            wrap(diff.eig) | call("eig", frameworks=("advect",)),
            {"values": (0,), "vectors": (1,), "both": (0, 1)}[used],
        )
        for used in ("values", "vectors", "both")
    },
    "smatrix-illuminate": (
        diff.smatrix_illuminate,
        (*STACKS, RNG.normal(size=(2, 3)) + 0.2j, RNG.normal(size=(2, 3)) + 0.1j),
        wrap(diff.smatrix_illuminate)
        | call("smatrix_illuminate", frameworks=("advect",)),
        None,
    ),
    "plane-field": (
        plane_field,
        (np.array([0.7 + 0.1j, -0.3 + 0.2j]), POINTS, VECTORS),
        wrap(plane_field)
        | call("plane_field", frameworks=("advect",), polarizations=[0, 1]),
        None,
    ),
    "plane-field-operator": (
        plane_operator,
        (POINTS, VECTORS),
        wrap(plane_operator)
        | {
            "advect": lambda ns, _: functools.partial(
                ns.plane_field, None, polarizations=[0, 1]
            )
        },
        None,
    ),
    "lattice-sum-broadcast": (
        lattice_sum,
        (
            WAVENUMBERS,
            np.array([0.1, 0.2]),
            np.eye(2) * 1.5,
            SITES,
            np.asarray(0.9 + 0.02j),
        ),
        wrap(lattice_sum)
        | call("lattice_sum", frameworks=("advect",), dim=2, degree=[2, 3], order=-1),
        None,
    ),
    "lattice-sum-real-part": (
        lattice_sum_real,
        (
            WAVENUMBERS,
            np.asarray(0.13),
            np.asarray(1.7),
            SITES,
            np.asarray(0.9 + 0.03j),
        ),
        wrap(lattice_sum_real)
        | call(
            "lattice_sum",
            frameworks=("advect",),
            dim=1,
            degree=[2, 3],
            order=-1,
            part="real",
        ),
        None,
    ),
    "rotation-list-gradient": (
        rotation,
        (np.array([0.2, 0.3, -0.1]),),
        wrap(rotation) | call("rotation", frameworks=("advect",), destination=BASIS),
        None,
    ),
    "sph-harm-tuple-gradients": (
        sph_harm,
        (np.array([0.7, 1.3]), np.array([0.2, 0.4])),
        wrap(sph_harm),
        None,
    ),
    "bessel-broadcast": (
        bessel,
        (np.array([[1.2 + 0.2j], [2.1 - 0.1j]]),),
        {
            framework: lambda ns, _: functools.partial(ns.bessel, order=ORDERS)
            for framework in FRAMEWORKS
        },
        None,
    ),
    "mie": (
        mie,
        (np.array([0.3]), np.array([3.0 + 0.1j, 1.0]), np.ones(2), np.zeros(2)),
        call("mie", 2, frameworks=("advect",)),
        None,
    ),
    "mie-cylinder": (
        mie_cyl,
        (
            np.asarray(0.2),
            np.asarray(1.1),
            np.array([0.3]),
            *(v[1:] for v in LAYERS[1:]),
        ),
        {"advect": lambda ns, _: lambda kz, k0, *v: ns.mie_cyl(kz, -1, k0, *v)},
        None,
    ),
}

CASES = [
    pytest.param(name, framework, id=f"{framework}-{name}")
    for name, (_, _, calls, _) in BRIDGES.items()
    for framework in FRAMEWORKS
    if framework in calls
]


def _tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _native(record, values, weights):
    outputs, context = record(*values)
    pullback = context if callable(context) else context.pullback
    gradients = pullback(*weights)
    # A list is one gradient (the Euler angles), as in the adapters.
    gradients = gradients if isinstance(gradients, tuple) else (gradients,)
    return _tuple(outputs), tuple(
        np.asarray(g) if np.iscomplexobj(v) else np.real(g)
        for g, v in zip(gradients, values, strict=True)
    )


def _framework(name, function, values, weights, used):
    """Outputs and input gradients of sum over used outputs of Re(vdot(w, y))."""
    if name == "torch":
        torch = importlib.import_module("torch")
        inputs = tuple(torch.tensor(v, requires_grad=True) for v in values)
        outputs = _tuple(function(*inputs))
        loss = sum(
            (torch.tensor(weights[i]).conj() * outputs[i]).real.sum() for i in used
        )
        gradients = torch.autograd.grad(loss, inputs)
        return (
            tuple(y.detach().numpy() for y in outputs),
            tuple(g.numpy() for g in gradients),
        )
    if name == "jax":
        jax = importlib.import_module("jax")
        jnp = jax.numpy

        def loss(*inputs):
            outputs = _tuple(function(*inputs))
            value = sum(jnp.real(jnp.vdot(weights[i], outputs[i])) for i in used)
            return value, outputs

        argnums = tuple(range(len(values)))
        (_, outputs), gradients = jax.jit(
            jax.value_and_grad(loss, argnums=argnums, has_aux=True)
        )(*values)
        # For complex inputs JAX returns the conjugate of the Re(vdot) gradient.
        return (
            tuple(np.asarray(y) for y in outputs),
            tuple(np.conj(g) for g in gradients),
        )
    anp = importlib.import_module("advect.numpy")

    def objective(*inputs):
        outputs = _tuple(function(*inputs))
        return sum(anp.sum(anp.real(np.conj(weights[i]) * outputs[i])) for i in used)

    gradients = advect.grad(objective, argnums=tuple(range(len(values))))(*values)
    outputs = _tuple(function(*values))
    return tuple(np.asarray(y) for y in outputs), _tuple(gradients)


@pytest.fixture(scope="module")
def x64():
    with jax_x64():
        yield


@pytest.mark.parametrize("bridge,framework", CASES)
def test_bridge_gradient_is_the_native_pullback(request, bridge, framework):
    ns = importlib.import_module(f"treams_rs.{pytest.importorskip(framework).__name__}")
    if framework == "jax":
        request.getfixturevalue("x64")
    record, values, calls, used = BRIDGES[bridge]
    outputs, _ = record(*values)
    rng = np.random.default_rng(len(bridge))
    weights = tuple(complex_normal(rng, np.shape(y)) for y in _tuple(outputs))
    used = tuple(range(len(weights))) if used is None else used
    weights = tuple(w if i in used else np.zeros_like(w) for i, w in enumerate(weights))
    expected_outputs, expected = _native(record, values, weights)
    function = calls[framework](ns, values)
    outputs, gradients = _framework(framework, function, values, weights, used)
    for actual, reference in zip(outputs, expected_outputs, strict=True):
        assert_allclose(actual, reference, rtol=1e-14, atol=1e-15)
    for gradient, reference, primal in zip(gradients, expected, values, strict=True):
        assert gradient.shape == np.shape(primal)
        assert gradient.dtype == np.asarray(primal).dtype
        assert_allclose(gradient, reference, rtol=1e-13, atol=1e-15)


@pytest.fixture(scope="module", params=["jax", "torch"])
def autodiff(request):
    module = pytest.importorskip(request.param)
    if request.param == "jax":
        request.getfixturevalue("x64")
    return module


@settings(max_examples=10)  # an eager JAX callback costs about 0.1 s
@given(
    m=st.integers(1, 4),
    n=st.integers(1, 4),
    complex_input=st.booleans(),
    real_output=st.booleans(),
    seed=st.integers(0, 2**16),
)
def test_wrap_matches_the_framework_own_derivative(
    autodiff, m, n, complex_input, real_output, seed
):
    # An independent anchor for the conventions: the framework differentiates
    # y = A x (or Re(A x)) itself, and through wrap with the pullback A^H g.
    rng = np.random.default_rng(seed)
    a = complex_normal(rng, (m, n))
    x = complex_normal(rng, n) if complex_input else rng.normal(size=n)
    w = rng.normal(size=m) if real_output else complex_normal(rng, m)

    def record(value):
        y = a @ value
        return (y.real if real_output else y), lambda g: a.conj().T @ g

    if autodiff.__name__ == "jax":
        jnp = autodiff.numpy
        wrapped = importlib.import_module("treams_rs.jax").wrap(record, x)

        def own(value):
            y = jnp.asarray(a) @ value
            return jnp.real(y) if real_output else y

        def gradient(function):
            return autodiff.grad(lambda v: jnp.real(jnp.vdot(w, function(v))))(x)

    else:
        wrapped = importlib.import_module("treams_rs.torch").wrap(record)

        def own(value):
            y = autodiff.as_tensor(a) @ value.to(autodiff.complex128)
            return y.real if real_output else y

        def gradient(function):
            value = autodiff.tensor(x, requires_grad=True)
            loss = (autodiff.as_tensor(w).conj() * function(value)).real.sum()
            return autodiff.autograd.grad(loss, value)[0].numpy()

    actual, expected = gradient(wrapped), gradient(own)
    assert np.asarray(actual).dtype == x.dtype
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
