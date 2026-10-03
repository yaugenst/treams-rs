//! Python-scalar fast paths of the ufuncs.
//!
//! The wrappers take a ufunc's `*args, **kwargs`. A call with Python scalars (or
//! one coordinate or cell) and no ufunc options runs the Rust kernel directly;
//! any other call goes to the hidden ufunc in `FALLBACKS`, which handles
//! broadcasting, masks and output arrays. The typed functions evaluate one
//! element; the Python namespaces call them when every argument is a Python
//! scalar or a single vector or cell, and no ufunc option is set.
//!
//! A wrapper and its hidden ufunc report the public name as `__name__`, so
//! error messages name the function the caller used. Three wrappers sit under
//! another attribute: `pw_translate` is `translate` (`pw.translate`), and
//! `cell_volume` and `cell_reciprocal` are `volume` and `reciprocal`
//! (`lattice.volume`, `lattice.reciprocal`).
//!
//! A typed function takes the name of the ufunc whose one-element call it
//! replaces, with the suffix `_scalar`: `cw_rotate_scalar` replaces `cw_rotate`,
//! and `cw_translate_scalar` replaces both variants, `cw_translate_s` and
//! `cw_translate_r`. Two serve a family of ufuncs and take its name:
//! `hankel_scalar` evaluates `hankel1` and `hankel2`, and `angular_scalar`
//! evaluates `lpmv`, `pi_fun` and `tau_fun`. `lpmv_real_scalar` evaluates `lpmv`
//! for real arguments. None of them records a context, unlike the
//! `_record_scalar` functions.

use std::sync::OnceLock;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods, ndarray::Array2,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyComplex, PyDict, PyFloat, PyInt, PyList, PyTuple},
};
use treams_core::{
    Complex, cw,
    fpenv::ieee,
    lattice, rotation,
    special::{self, Bessel, MAX_LABEL},
};

use super::kinds::pol;
use crate::{
    args::angular_function,
    context::{error, radial},
};

/// Wrapper-internal ufuncs behind the Rust scalar fast paths.
pub(super) struct Fallbacks {
    pub(super) pw_translate: Py<PyAny>,
    /// `vpw_M`, `vpw_N` and `vpw_A`, indexed by the `POL` loop parameter.
    pub(super) plane: [Py<PyAny>; 3],
    /// `cell_volume` and `cell_reciprocal`.
    pub(super) cells: [Py<PyAny>; 2],
    /// Point and vector ufunc of each transform, in `TRANSFORMS` order.
    pub(super) coordinates: [[Py<PyAny>; 2]; 8],
}
pub(super) static FALLBACKS: OnceLock<Fallbacks> = OnceLock::new();

/// Whether a wrapper call passes no positional `out` or ufunc keyword options.
fn plain(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> bool {
    args.is_empty() && kwargs.is_none_or(PyDictMethods::is_empty)
}

/// Forward a wrapper call to its ufunc, which owns broadcasting and options.
fn call_ufunc<'py>(
    py: Python<'py>,
    select: impl FnOnce(&Fallbacks) -> &Py<PyAny>,
    operands: impl IntoIterator<Item = Bound<'py, PyAny>>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let fallbacks = FALLBACKS
        .get()
        .ok_or_else(|| PyRuntimeError::new_err("native ufuncs are not registered"))?;
    let arguments = PyTuple::new(
        py,
        operands.into_iter().chain(args.iter()).collect::<Vec<_>>(),
    )?;
    select(fallbacks).bind(py).call(arguments, kwargs)
}

// Scalar Python numbers need no NumPy shape/type dispatch. Arrays and ufunc
// options still use the registered loop, including masks and overlapping output.
fn scalar_number(value: &Bound<'_, PyAny>) -> Option<Complex> {
    if let Ok(value) = value.cast::<PyFloat>() {
        Some(value.value().into())
    } else if let Ok(value) = value.cast::<PyComplex>() {
        Some(Complex::new(value.real(), value.imag()))
    } else if value.is_instance_of::<PyInt>() {
        value.extract::<f64>().ok().map(Into::into)
    } else {
        None
    }
}

// A single coordinate needs no generalized-ufunc shape resolution. Preserve the
// original loop for broadcasting, output options and ndarray subclass dispatch.
fn coordinate_input<T>(value: &Bound<'_, PyAny>, dim: usize) -> Option<[T; 3]>
where
    T: numpy::Element + Copy + Default + for<'a, 'py> FromPyObject<'a, 'py>,
{
    if let Ok(array) = value.cast_exact::<PyArray1<T>>() {
        let array = array.readonly();
        let values = array.as_slice().ok()?;
        if values.len() == dim {
            let mut result = [T::default(); 3];
            result.get_mut(..dim)?.copy_from_slice(values);
            return Some(result);
        }
    } else if value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>() {
        return if dim == 2 {
            value
                .extract::<[T; 2]>()
                .ok()
                .map(|[a, b]| [a, b, T::default()])
        } else {
            value.extract::<[T; 3]>().ok()
        };
    }
    None
}

fn coordinate_call<'py, const KIND: u8>(
    py: Python<'py>,
    points: &Bound<'py, PyAny>,
    vector: Option<&Bound<'py, PyAny>>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let transform = crate::coordinates::TRANSFORMS[usize::from(KIND)].2;
    let dim = transform.dimension();
    if plain(args, kwargs)
        && let Some(position) = coordinate_input::<f64>(points, dim)
    {
        use treams_core::special::coordinates::{point, vector as transform_vector};
        if let Some(vector) = vector {
            if let Some(value) = coordinate_input::<f64>(vector, dim) {
                let result = transform_vector(value.map(Complex::from), position, transform)
                    .map_err(error)?
                    .map(|x| x.re);
                return Ok(PyArray1::from_slice(py, &result[..dim]).into_any());
            }
            if let Some(value) = coordinate_input::<Complex>(vector, dim) {
                let result = transform_vector(value, position, transform).map_err(error)?;
                return Ok(PyArray1::from_slice(py, &result[..dim]).into_any());
            }
        } else {
            let result = point(position, transform).map_err(error)?;
            return Ok(PyArray1::from_slice(py, &result[..dim]).into_any());
        }
    }
    call_ufunc(
        py,
        |f| &f.coordinates[usize::from(KIND)][usize::from(vector.is_some())],
        vector.into_iter().chain([points]).cloned(),
        args,
        kwargs,
    )
}

macro_rules! coordinate_functions {
    ($($kind:literal: $point:ident, $vector:ident, $from:literal, $to:literal;)+) => {
        $(
            #[doc = concat!("Transform ", $from, " points to ", $to, " coordinates.")]
            ///
            /// The last axis holds the components; ufunc options are supported.
            ///
            #[doc = concat!("Mirrors ``treams.special.", stringify!($point), "``.")]
            ///
            /// Differences from treams: ``ValueError`` for non-finite input.
            #[pyfunction]
            #[pyo3(signature=(points, *args, **kwargs))]
            pub(crate) fn $point<'py>(
                py: Python<'py>,
                points: &Bound<'py, PyAny>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<&Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                ieee(|| coordinate_call::<$kind>(py, points, None, args, kwargs))
            }

            #[doc = concat!("Transform vector components from the ", $from, " to the ", $to, " basis.")]
            ///
            /// Positions are given in the input coordinate system; the last axis
            /// holds the components and ufunc options are supported.
            ///
            #[doc = concat!("Mirrors ``treams.special.", stringify!($vector), "``.")]
            ///
            /// Differences from treams: ``ValueError`` for non-finite input.
            #[pyfunction]
            #[pyo3(signature=(vector, points, *args, **kwargs))]
            pub(crate) fn $vector<'py>(
                py: Python<'py>,
                vector: &Bound<'py, PyAny>,
                points: &Bound<'py, PyAny>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<&Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                ieee(|| coordinate_call::<$kind>(py, points, Some(vector), args, kwargs))
            }
        )+
    };
}

coordinate_functions! {
    0: car2cyl, vcar2cyl, "Cartesian", "cylindrical";
    1: car2sph, vcar2sph, "Cartesian", "spherical";
    2: cyl2car, vcyl2car, "cylindrical", "Cartesian";
    3: cyl2sph, vcyl2sph, "cylindrical", "spherical";
    4: sph2car, vsph2car, "spherical", "Cartesian";
    5: sph2cyl, vsph2cyl, "spherical", "cylindrical";
    6: car2pol, vcar2pol, "Cartesian", "polar";
    7: pol2car, vpol2car, "polar", "Cartesian";
}

fn plane_wave_call<'py, const POL: u8>(
    py: Python<'py>,
    values: [&Bound<'py, PyAny>; 6],
    label: Option<&Bound<'py, PyAny>>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    if plain(args, kwargs)
        && label.is_none_or(PyAnyMethods::is_instance_of::<PyInt>)
        && let [Some(kx), Some(ky), Some(kz), Some(x), Some(y), Some(z)] = values.map(scalar_number)
        && x.im == 0.0
        && y.im == 0.0
        && z.im == 0.0
    {
        let pol = label.map_or(Ok(POL), |p| {
            special::pol_index(p.extract::<i64>()?).map_err(error)
        })?;
        let polarization =
            treams_core::pw::polarization([kx, ky, kz], pol, POL == pol::A).map_err(error)?;
        let result =
            treams_core::pw::field_value(polarization, [kx, ky, kz], [x, y, z]).map_err(error)?;
        return Ok(PyArray1::from_slice(py, &result).into_any());
    }
    call_ufunc(
        py,
        |f| &f.plane[usize::from(POL)],
        values.into_iter().chain(label).cloned(),
        args,
        kwargs,
    )
}

macro_rules! plane_wave_function {
    ($name:ident, $public:literal, $pol:path, $wave:literal $(, $label:ident)?) => {
        #[doc = $wave]
        ///
        /// The last axis holds x, y and z. Python scalars take a direct native path
        /// that also accepts complex positions with zero imaginary part. Arrays and
        /// ufunc options use the ufunc of the same name, which requires real
        /// positions.
        ///
        #[doc = concat!("Mirrors ``treams.special.", $public, "``.")]
        ///
        /// Differences from treams: ``ValueError`` for ``kx**2 + ky**2 + kz**2 = 0`` and for
        /// non-finite input.
        #[pyfunction(name = $public)]
        #[pyo3(signature=(kx, ky, kz, x, y, z $(, $label)?, *args, **kwargs))]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            kx: &Bound<'py, PyAny>,
            ky: &Bound<'py, PyAny>,
            kz: &Bound<'py, PyAny>,
            x: &Bound<'py, PyAny>,
            y: &Bound<'py, PyAny>,
            z: &Bound<'py, PyAny>,
            $($label: &Bound<'py, PyAny>,)?
            args: &Bound<'py, PyTuple>,
            kwargs: Option<&Bound<'py, PyDict>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            ieee(|| {
                let label: Option<&Bound<'py, PyAny>> = None;
                $(let label = label.or(Some($label));)?
                plane_wave_call::<{ $pol }>(py, [kx, ky, kz, x, y, z], label, args, kwargs)
            })
        }
    };
}
plane_wave_function!(
    vpw_m,
    "vpw_M",
    pol::M,
    "``M_k(r) = -i phi_hat(k) exp(i k.r)``, the vector plane wave ``M``."
);
plane_wave_function!(
    vpw_n,
    "vpw_N",
    pol::N,
    "``N_k(r) = -theta_hat(k) exp(i k.r)``, the vector plane wave ``N``."
);
plane_wave_function!(
    vpw_a,
    "vpw_A",
    pol::A,
    "``A_k(r) = (N_k(r) +- M_k(r)) / sqrt(2)``, the vector plane wave of helicity ``pol`` \
     (1 positive, 0 negative).",
    pol
);

fn cell_call<'py, const RECIPROCAL_CELL: bool>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    // as_slice also accepts Fortran order, which would read the transpose.
    if plain(args, kwargs)
        && let Ok(array) = cell.cast_exact::<PyArray2<f64>>()
        && array.is_c_contiguous()
        && let [dim, columns] = *array.shape()
        && dim == columns
        && (1..=3).contains(&dim)
        && let Ok(values) = array.readonly().as_slice()
    {
        let matrix = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i < dim && j < dim {
                    values[i * dim + j]
                } else {
                    0.0
                }
            })
        });
        return if RECIPROCAL_CELL {
            let result = lattice::reciprocal(matrix, dim).map_err(error)?;
            Ok(Array2::from_shape_fn((dim, dim), |(i, j)| result[i][j])
                .into_pyarray(py)
                .into_any())
        } else {
            Ok(lattice::volume(matrix, dim)
                .map_err(error)?
                .into_pyobject(py)?
                .into_any())
        };
    }
    call_ufunc(
        py,
        |f| &f.cells[usize::from(RECIPROCAL_CELL)],
        [cell.clone()],
        args,
        kwargs,
    )
}

/// Signed volume (area, length) of cells whose rows are lattice vectors.
///
/// Cells have shape (..., d, d) with d = 1, 2 or 3; integer cells stay integer.
///
/// Mirrors ``treams.lattice.volume``.
///
/// Differences from treams: also one-dimensional cells.
#[pyfunction(name = "volume")]
#[pyo3(signature=(cell, *args, **kwargs))]
pub(crate) fn cell_volume<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    ieee(|| cell_call::<false>(py, cell, args, kwargs))
}

/// Reciprocal lattice vectors as rows, with `a_i . b_j = 2 pi delta_ij`.
///
/// Cells have shape (..., d, d) with d = 1, 2 or 3 and rows as lattice vectors.
///
/// Mirrors ``treams.lattice.reciprocal``.
///
/// Differences from treams: also one-dimensional cells.
#[pyfunction(name = "reciprocal")]
#[pyo3(signature=(cell, *args, **kwargs))]
pub(crate) fn cell_reciprocal<'py>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    ieee(|| cell_call::<true>(py, cell, args, kwargs))
}

/// Plane-wave translation phase exp(i k.r) with array broadcasting.
///
/// Python scalars take a direct native path that also accepts complex
/// positions with zero imaginary part. Arrays and ufunc options use the
/// ufunc of the same name, which requires real positions.
///
/// Mirrors ``treams.pw.translate``.
///
/// Differences from treams: real positions; ``ValueError`` for non-finite input.
#[pyfunction(name = "translate")]
#[pyo3(signature=(kx, ky, kz, x, y, z, *args, **kwargs))]
pub(crate) fn pw_translate<'py>(
    py: Python<'py>,
    kx: &Bound<'py, PyAny>,
    ky: &Bound<'py, PyAny>,
    kz: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    z: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    ieee(|| {
        let operands = [kx, ky, kz, x, y, z];
        if plain(args, kwargs)
            && let [Some(kx), Some(ky), Some(kz), Some(x), Some(y), Some(z)] =
                operands.map(scalar_number)
            && x.im == 0.0
            && y.im == 0.0
            && z.im == 0.0
        {
            return Ok(treams_core::pw::translate([kx, ky, kz], [x.re, y.re, z.re])
                .map_err(error)?
                .into_pyobject(py)?
                .into_any());
        }
        call_ufunc(
            py,
            |f| &f.pw_translate,
            operands.into_iter().cloned(),
            args,
            kwargs,
        )
    })
}

/// One element of `cw_rotate`: `rotation::cw_rotate`.
#[pyfunction]
pub(crate) fn cw_rotate_scalar(
    kz: f64,
    mu: i64,
    p: i64,
    qz: f64,
    m: i64,
    q: i64,
    phi: f64,
) -> PyResult<Complex> {
    ieee(|| rotation::cw_rotate(kz, mu, p, qz, m, q, phi).map_err(error))
}

/// One element of `tl_vcw` (`singular`) or `tl_vcw_r`: `cw::tl_vcw`.
#[pyfunction]
pub(crate) fn tl_vcw_scalar(
    kz: f64,
    mu: i64,
    qz: f64,
    m: i64,
    kr: Complex,
    phi: f64,
    z: f64,
    singular: bool,
) -> PyResult<Complex> {
    ieee(|| {
        let args = [kr, phi.into(), z.into()];
        cw::tl_vcw(kz, mu, qz, m, args, radial(singular)).map_err(error)
    })
}

/// One element of `cw_translate_s` (`singular`) or `cw_translate_r`: `cw::translate`.
#[pyfunction]
pub(crate) fn cw_translate_scalar(
    kz: f64,
    mu: i64,
    p: i64,
    qz: f64,
    m: i64,
    q: i64,
    kr: Complex,
    phi: f64,
    z: f64,
    singular: bool,
) -> PyResult<Complex> {
    ieee(|| {
        let args = [kr, phi.into(), z.into()];
        cw::translate(kz, mu, p, qz, m, q, args, radial(singular)).map_err(error)
    })
}

/// One element of the `pw_permute_xyz_*` ufuncs: `pw::permute_xyz`.
#[pyfunction]
pub(crate) fn pw_permute_xyz_scalar(
    kx: Complex,
    ky: Complex,
    kz: Complex,
    p: i64,
    q: i64,
    helicity: bool,
    inverse: bool,
) -> PyResult<Complex> {
    ieee(|| {
        treams_core::pw::permute_xyz(
            [kx, ky, kz],
            special::pol_index(p).map_err(error)?,
            special::pol_index(q).map_err(error)?,
            if inverse { 2 } else { 1 },
            helicity,
        )
        .map_err(error)
    })
}

/// One element of `lpmv` with real arguments: `special::lpmv_real`.
#[pyfunction]
pub(crate) fn lpmv_real_scalar(degree: f64, order: f64, x: f64) -> PyResult<f64> {
    ieee(|| special::lpmv_real(degree, order, x).map_err(error))
}

/// One element of `hankel1` (`first`) or `hankel2`: `special::bessel`.
#[pyfunction]
pub(crate) fn hankel_scalar(order: f64, z: Complex, first: bool) -> PyResult<Complex> {
    ieee(|| {
        let kind = if first { Bessel::H1 } else { Bessel::H2 };
        special::bessel(order, z, kind, false, 0).map_err(error)
    })
}

/// One element of `lpmv`, `pi_fun` or `tau_fun`: `special::angular_value`.
#[pyfunction]
pub(crate) fn angular_scalar(
    degree: f64,
    order: f64,
    z: Complex,
    function: &str,
) -> PyResult<Complex> {
    ieee(|| special::angular_value(degree, order, z, angular_function(function)?).map_err(error))
}

/// One element of `wigner3j`: `special::wigner3j`.
#[pyfunction]
pub(crate) fn wigner3j_scalar(
    j1: i32,
    j2: i32,
    j3: i32,
    m1: i32,
    m2: i32,
    m3: i32,
) -> PyResult<f64> {
    ieee(|| {
        if [j1, j2, j3, m1, m2, m3]
            .iter()
            .any(|n| n.unsigned_abs() > MAX_LABEL.unsigned_abs())
        {
            return Err(PyValueError::new_err(
                // 260 is MAX_LABEL.
                "Wigner labels must be integers in [-260, 260]",
            ));
        }
        Ok(special::wigner3j(j1, j2, j3, m1, m2, m3))
    })
}

/// One element of `incgamma`: `special::incgamma`.
#[pyfunction]
pub(crate) fn incgamma_scalar(n: f64, z: Complex) -> PyResult<Complex> {
    ieee(|| special::incgamma(n, z).map_err(error))
}

/// One element of `intkambe`: `special::intkambe`.
#[pyfunction]
pub(crate) fn intkambe_scalar(n: i32, z: Complex, eta: Complex) -> PyResult<Complex> {
    ieee(|| special::intkambe(n, z, eta).map_err(error))
}

/// One direct shell of the cylindrical lattice sum of a line (`dim` 1) or plane
/// (`dim` 2) lattice with in-plane Bloch vector `q`, cell rows `cell` and shift `r`.
fn direct_cylindrical(
    m: i32,
    k: Complex,
    q: [f64; 2],
    cell: [[f64; 2]; 2],
    r: [f64; 2],
    shell: i64,
    dim: usize,
) -> PyResult<Complex> {
    let pad = |[x, y]: [f64; 2]| [x, y, 0.0];
    let rows = [pad(cell[0]), pad(cell[1]), [0.0; 3]];
    let geometry = lattice::BlochLattice::from_array(rows, pad(q), dim).map_err(error)?;
    let wave = lattice::Family::Cylindrical { m };
    let part = lattice::SumPart::Direct(shell);
    lattice::sum_part(wave, k, &geometry, pad(r), Complex::default(), part).map_err(error)
}
/// One element of `dsumcw1d`: `lattice::sum_part`.
#[pyfunction]
pub(crate) fn dsumcw1d_scalar(
    m: i32,
    k: Complex,
    q: f64,
    a: f64,
    r: f64,
    shell: i64,
) -> PyResult<Complex> {
    ieee(|| direct_cylindrical(m, k, [q, 0.0], [[a, 0.0], [0.0; 2]], [r, 0.0], shell, 1))
}
/// One element of `dsumcw1d_shift`: `lattice::sum_part`.
#[pyfunction]
pub(crate) fn dsumcw1d_shift_scalar(
    m: i32,
    k: Complex,
    q: f64,
    a: f64,
    r: PyReadonlyArray1<'_, f64>,
    shell: i64,
) -> PyResult<Complex> {
    ieee(|| {
        let r = r.as_array();
        if r.len() != 2 {
            return Err(PyValueError::new_err("shift must have two components"));
        }
        direct_cylindrical(m, k, [q, 0.0], [[a, 0.0], [0.0; 2]], [r[0], r[1]], shell, 1)
    })
}
/// One element of `dsumcw2d`: `lattice::sum_part`.
#[pyfunction]
pub(crate) fn dsumcw2d_scalar(
    m: i32,
    k: Complex,
    q: PyReadonlyArray1<'_, f64>,
    a: PyReadonlyArray2<'_, f64>,
    r: PyReadonlyArray1<'_, f64>,
    shell: i64,
) -> PyResult<Complex> {
    ieee(|| {
        let (q, a, r) = (q.as_array(), a.as_array(), r.as_array());
        if q.len() != 2 || a.shape() != [2, 2] || r.len() != 2 {
            return Err(PyValueError::new_err(
                "require two-component vectors and a 2x2 cell",
            ));
        }
        let cell = [[a[(0, 0)], a[(0, 1)]], [a[(1, 0)], a[(1, 1)]]];
        direct_cylindrical(m, k, [q[0], q[1]], cell, [r[0], r[1]], shell, 2)
    })
}
