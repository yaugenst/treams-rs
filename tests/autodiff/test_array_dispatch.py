"""Public array functions retain NumPy behavior and existing analytic gradients."""

import contextlib
import functools
import importlib
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from treams_rs import _native, coeffs, misc, special

from _support import jax_x64

pytestmark = pytest.mark.gradients


@pytest.fixture(scope="module", params=["advect", "jax", "torch", "autograd"])
def engine(request):
    framework = pytest.importorskip(request.param)
    name = request.param
    scope = jax_x64() if name == "jax" else contextlib.nullcontext()
    xp = framework if name == "torch" else importlib.import_module(f"{name}.numpy")
    with scope:
        yield name, framework, xp


def value_gradient(engine, function, value):
    name, framework, xp = engine

    def objective(x):
        y = function(x)
        if name == "torch" and not y.is_complex():
            return xp.sum(y)
        return xp.sum(xp.real(y)) + 0.37 * xp.sum(xp.imag(y))

    if name == "torch":
        x = framework.tensor(value, dtype=framework.float64, requires_grad=True)
        result = objective(x)
        (gradient,) = framework.autograd.grad(result, x)
        return result.detach().numpy(), gradient.numpy()
    transform = framework.value_and_grad(objective)
    if name == "jax":
        transform = framework.jit(transform)
    return transform(np.asarray(value))


CASES = {}
for name in (
    "jv",
    "yv",
    "hankel1",
    "hankel2",
    "jv_d",
    "yv_d",
    "hankel1_d",
    "hankel2_d",
    "spherical_jn",
    "spherical_yn",
    "spherical_hankel1",
    "spherical_hankel2",
    "spherical_jn_d",
    "spherical_yn_d",
    "spherical_hankel1_d",
    "spherical_hankel2_d",
):
    CASES[name] = (functools.partial(getattr(special, name), np.array([0, 1, 2])), 1.2)

CASES.update(
    {
        "hankel_keywords": (lambda x: special.hankel1(v=1, z=x), 1.2),
        "vector_coordinate_keywords": (
            lambda x: special.vcar2sph(vector=[1.0, 2.0, 3.0], points=[x, 0.4, 0.5]),
            1.2,
        ),
        "plane_wave_keywords": (
            lambda x: special.vpw_A(kx=0.3, ky=0.4, kz=x, x=0.1, y=0.2, z=0.3, pol=1),
            1.2,
        ),
        "spherical_jn_derivative": (lambda x: special.spherical_jn(2, x, True), 1.2),
        "spherical_yn_derivative": (lambda x: special.spherical_yn(2, x, True), 1.2),
        "lpmv": (lambda x: special.lpmv(1, 2, x), 0.3),
        "pi_fun": (lambda x: special.pi_fun(2, 1, x), 0.3),
        "tau_fun": (lambda x: special.tau_fun(2, 1, x), 0.3),
        "incgamma": (lambda x: special.incgamma(0.5, x), 1.2),
        "intkambe": (lambda x: special.intkambe(1, x, 0.8), 1.2),
        "wignerd": (lambda x: special.wignerd(2, 1, -1, 0.2, x, 0.4), 1.2),
        "wignersmalld": (lambda x: special.wignersmalld(2, 1, -1, x), 1.2),
        "sph_harm": (lambda x: special.sph_harm(1, 2, 0.4, x), 1.2),
        "mie": (
            lambda x: coeffs.mie(1, [x * 0.3], [2 + x * 0.1, 1], [1, 1], [0, 0]),
            1.2,
        ),
        "mie_cyl": (
            lambda x: coeffs.mie_cyl(0.1, 1, x, [0.3], [2.5, 1], [1, 1], [0, 0]),
            1.2,
        ),
        "fresnel": (
            lambda x: coeffs.fresnel(
                [[2 + x, 2 + x], [1, 1]], [[1.8, 1.8], [0.8, 0.8]], [0.5, 1]
            ),
            0.2,
        ),
        "refractive_index": (lambda x: misc.refractive_index(x, 1, 0.1), 1.2),
        "wave_vec_z": (lambda x: misc.wave_vec_z(0.3, 0.4, x), 1.2),
    }
)


def coordinate(name, x):
    dimension = 2 if "pol" in name else 3
    points = [x, 0.4, 0.7][:dimension]
    if name.startswith("v"):
        return getattr(special, name)([0.2 + 0.3j, 0.5, 0.4][:dimension], points)
    return getattr(special, name)(points)


for name in (
    "car2cyl",
    "car2sph",
    "cyl2car",
    "cyl2sph",
    "sph2car",
    "sph2cyl",
    "car2pol",
    "pol2car",
):
    CASES[name] = functools.partial(coordinate, name), 1.2
    CASES["v" + name] = functools.partial(coordinate, "v" + name), 1.2


def wave(name, x):
    function = getattr(special, name)
    if name.startswith("vsh"):
        return function(2, 1, x, 0.4)
    if name.startswith("vsw"):
        return function(2, 1, x, 0.5, 0.4, *([1] if name.endswith("A") else []))
    if name.startswith("vcw"):
        return function(
            0.3,
            1,
            x,
            0.4,
            0.2,
            *([2.0] if not name.endswith("M") else []),
            *([1] if name.endswith("A") else []),
        )
    return function(0.3, 0.4, x, 0.1, 0.2, 0.3, *([1] if name.endswith("A") else []))


for name in (
    "vsh_X",
    "vsh_Y",
    "vsh_Z",
    "vsw_M",
    "vsw_N",
    "vsw_A",
    "vsw_rM",
    "vsw_rN",
    "vsw_rA",
    "vcw_M",
    "vcw_N",
    "vcw_A",
    "vcw_rM",
    "vcw_rN",
    "vcw_rA",
    "vpw_M",
    "vpw_N",
    "vpw_A",
):
    CASES[name] = functools.partial(wave, name), 1.2

for name in ("tl_vsw_A", "tl_vsw_B", "tl_vsw_rA", "tl_vsw_rB"):
    CASES[name] = (
        (lambda x, name=name: getattr(special, name)(2, 1, 1, 0, x, 0.5, 0.4)),
        1.2,
    )
for name in ("tl_vcw", "tl_vcw_r"):
    CASES[name] = (
        (lambda x, name=name: getattr(special, name)(0.3, 1, 0.3, 0, x, 0.5, 0.4)),
        1.2,
    )


@pytest.mark.parametrize("name", CASES)
def test_ordinary_functions_preserve_values_and_gradients(engine, name):
    function, x = CASES[name]

    def reference(value):
        result = np.asarray(function(value))
        return result.real.sum() + 0.37 * result.imag.sum()

    value, gradient = value_gradient(engine, function, x)
    h = 1e-6
    expected = (reference(x + h) - reference(x - h)) / (2 * h)
    assert_allclose(value, reference(x), rtol=2e-12, atol=1e-12)
    assert_allclose(gradient, expected, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("name", ["car2sph", "sph2car", "vcar2sph", "vsph2car"])
def test_complex_coordinates_raise_like_numpy(engine, name):
    # The float64 cast would otherwise keep only the real part.
    with pytest.raises(TypeError):
        coordinate(name, 1.2 + 0.1j)
    with pytest.raises(TypeError, match="real points"):
        value_gradient(engine, lambda x: coordinate(name, x * (1 + 0.1j)), 1.2)


def test_numpy_ufunc_attributes_methods_and_output_masks_are_retained():
    assert special.jv.nin == _native.jv.nin
    assert special.jv.types == _native.jv.types
    assert special.jv.signature == _native.jv.signature
    assert_array_equal(
        special.jv.outer([0, 1], [0.2, 0.3]), _native.jv.outer([0, 1], [0.2, 0.3])
    )
    out = np.full(2, 7.0 + 0j)
    assert special.jv(1, [0.2, 0.3], out, where=[True, False]) is out
    assert out[1] == 7
    assert out[0] == _native.jv(1, 0.2)
    assert pickle.loads(pickle.dumps(special.jv)) is special.jv


def test_framework_out_is_rejected_before_mutation(engine):
    output = np.full((), 7.0 + 0j)
    with pytest.raises(TypeError, match="out or where"):
        value_gradient(engine, lambda x: special.jv(1, x, out=output), 1.2)
    assert output == 7


def test_complex_argument_preserves_each_framework_gradient_convention(engine):
    name, _, _ = engine
    point = 0.7 + 0.2j
    if name == "torch":
        torch = importlib.import_module("torch")
        z = torch.tensor(point, dtype=torch.complex128, requires_grad=True)
        result = special.jv(1, z)
        loss = result.real + 0.37 * result.imag
        (gradient,) = torch.autograd.grad(loss, z)
        actual = gradient.numpy()
    else:
        _, actual = value_gradient(engine, functools.partial(special.jv, 1), point)

    def reference(z):
        value = special.jv(1, z)
        return value.real + 0.37 * value.imag

    h = 1e-6
    real = (reference(point + h) - reference(point - h)) / (2 * h)
    imag = (reference(point + 1j * h) - reference(point - 1j * h)) / (2 * h)
    expected = real + (1j if name in ("advect", "torch") else -1j) * imag
    assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("point", [0.4, 0.4 + 0.2j])
def test_nested_legendre_inputs_keep_primal_dtype_and_gradient(
    engine, container, point
):
    name, framework, xp = engine
    seen_dtypes = []
    constant = np.asarray(0.25 + 0.1j if np.iscomplexobj(point) else 0.25)

    def function(z):
        result = special.lpmv(1, 2, container([container([z, z + 0.1, constant])]))
        seen_dtypes.append(str(result.dtype))
        return result

    def objective(z):
        result = function(z)
        if name == "torch" and not result.is_complex():
            return result.sum()
        return xp.sum(xp.real(result)) + 0.37 * xp.sum(xp.imag(result))

    if name == "torch":
        dtype = framework.complex128 if np.iscomplexobj(point) else framework.float64
        z = framework.tensor(point, dtype=dtype, requires_grad=True)
        actual_primal = function(z).detach().numpy()
        value = objective(z)
        (gradient,) = framework.autograd.grad(value, z)
        value, gradient = value.detach().numpy(), gradient.numpy()
    else:
        transform = framework.value_and_grad(objective)
        if name == "jax":
            transform = framework.jit(transform)
            actual_primal = np.asarray(framework.jit(function)(point))
        elif name == "autograd":
            _, actual_primal = framework.make_vjp(function)(point)
        else:
            actual_primal = function(point)
        value, gradient = transform(np.asarray(point))

    expected_primal = special.lpmv(1, 2, [[point, point + 0.1, constant]])
    assert_allclose(actual_primal, expected_primal, rtol=2e-12, atol=1e-12)
    assert all(("complex" in dtype) == np.iscomplexobj(point) for dtype in seen_dtypes)

    def reference(z):
        result = special.lpmv(1, 2, [[z, z + 0.1, constant]])
        return result.real.sum() + 0.37 * result.imag.sum()

    h = 1e-6
    expected = (reference(point + h) - reference(point - h)) / (2 * h)
    if np.iscomplexobj(point):
        imag = (reference(point + 1j * h) - reference(point - 1j * h)) / (2 * h)
        expected += (1j if name in ("advect", "torch") else -1j) * imag
    assert_allclose(value, reference(point), rtol=2e-12, atol=1e-12)
    assert_allclose(gradient, expected, rtol=2e-7, atol=2e-9)


def test_real_legendre_rejects_imaginary_continuation(engine):
    name, _, _ = engine
    # JAX reports exceptions raised in its native callback as runtime errors.
    error = Exception if name == "jax" else ValueError
    with pytest.raises(error, match=r"odd orders.*complex z"):
        value_gradient(engine, lambda x: special.lpmv(1, 2, [x]), 1.5)


@pytest.mark.parametrize(
    "order,degree,point,complex_argument",
    [
        (2, 3, 1.5, False),
        (3, 2, 1.5, False),
        (-3, 2, -2.0, False),
        (1, 2, 1.5, True),
        (1, 2.3, 0.4, False),
    ],
)
def test_legendre_retains_allowed_real_and_complex_branches(
    engine, order, degree, point, complex_argument
):
    def function(x):
        return special.lpmv(order, degree, [x + 0j if complex_argument else x])

    value, gradient = value_gradient(engine, function, point)

    def reference(x):
        result = function(x)
        return result.real.sum() + 0.37 * result.imag.sum()

    h = 1e-6
    expected = (reference(point + h) - reference(point - h)) / (2 * h)
    assert_allclose(value, reference(point), rtol=2e-12, atol=1e-12)
    assert_allclose(gradient, expected, rtol=2e-7, atol=2e-9)
