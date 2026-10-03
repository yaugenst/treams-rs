//! `NumPy` array conversions shared by the bindings: cotangents and their checks,
//! inputs lent or copied to the core, and outputs.

use std::ops::Deref;

use faer::MatRef;
use nalgebra::{DMatrix, DMatrixView, Dyn, Scalar};
use num_complex::Complex64;
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray2,
    PyReadonlyArrayDyn,
    ndarray::{
        Array2, ArrayView, ArrayView2, Dimension, Ix1, Ix2, Order, ShapeBuilder, ShapeError,
    },
};
use pyo3::{exceptions::PyValueError, intern, prelude::*};
use rayon::prelude::*;
use treams_core::{Complex, coeffs::Matrix2};

use crate::context::cotangent_error;

/// Python-owned `NumPy` arrays of complex (`C`) or real (`R`) values by dimension.
pub(crate) type C1<'py> = Bound<'py, PyArray1<Complex64>>;
pub(crate) type C2<'py> = Bound<'py, PyArray2<Complex64>>;
pub(crate) type C3<'py> = Bound<'py, numpy::PyArray3<Complex64>>;
pub(crate) type C4<'py> = Bound<'py, numpy::PyArray4<Complex64>>;
pub(crate) type C5<'py> = Bound<'py, numpy::PyArray5<Complex64>>;
pub(crate) type CDyn<'py> = Bound<'py, PyArrayDyn<Complex64>>;
pub(crate) type R1<'py> = Bound<'py, PyArray1<f64>>;
pub(crate) type R2<'py> = Bound<'py, PyArray2<f64>>;
pub(crate) type RDyn<'py> = Bound<'py, PyArrayDyn<f64>>;

// Cotangents and their checks.

/// A cotangent of any real or complex dtype, dimension and memory layout.
///
/// A complex128 array of any layout is borrowed. Anything else converts as a
/// whole with `numpy.asarray(cotangent, complex128)`, never element by element:
/// before `NumPy` 2.5 an element-wise conversion would turn size-1 rows into
/// scalars and flatten an `(n, 1)` cotangent to `(n,)`. Pullbacks check the
/// dimension together with the shape, so a cotangent of the wrong dimension
/// raises their `ValueError` as well.
#[derive(Debug)]
pub(crate) struct Cotangent<'py>(PyReadonlyArrayDyn<'py, Complex>);

impl<'py> Deref for Cotangent<'py> {
    type Target = PyReadonlyArrayDyn<'py, Complex>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'py> FromPyObject<'_, 'py> for Cotangent<'py> {
    type Error = PyErr;

    fn extract(cotangent: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let py = cotangent.py();
        let array = if let Ok(array) = cotangent.cast::<PyArrayDyn<Complex>>() {
            array.to_owned()
        } else {
            py.import(intern!(py, "numpy"))?
                .call_method1(intern!(py, "asarray"), (cotangent, Complex::get_dtype(py)))?
                .cast_into::<PyArrayDyn<Complex>>()?
        };
        Ok(Self(array.try_into_readonly()?))
    }
}

/// A cotangent of a real output, of any real or complex dtype, dimension and
/// memory layout.
///
/// A float64 array of any layout is borrowed. Anything else converts as a whole,
/// as [`Cotangent`] does, with `numpy.asarray(numpy.real(cotangent), float64)`:
/// the pairing `Re sum(conj(g) * dx)` of a real output `dx` reads only the real
/// part of `g`.
#[derive(Debug)]
pub(crate) struct RealCotangent<'py>(PyReadonlyArrayDyn<'py, f64>);

impl<'py> Deref for RealCotangent<'py> {
    type Target = PyReadonlyArrayDyn<'py, f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'py> FromPyObject<'_, 'py> for RealCotangent<'py> {
    type Error = PyErr;

    fn extract(cotangent: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let py = cotangent.py();
        let array = if let Ok(array) = cotangent.cast::<PyArrayDyn<f64>>() {
            array.to_owned()
        } else {
            let numpy = py.import(intern!(py, "numpy"))?;
            let real = numpy.call_method1(intern!(py, "real"), (cotangent,))?;
            numpy
                .call_method1(intern!(py, "asarray"), (real, f64::get_dtype(py)))?
                .cast_into::<PyArrayDyn<f64>>()?
        };
        Ok(Self(array.try_into_readonly()?))
    }
}

/// Real and complex scalars, which a finiteness check reads.
pub(crate) trait Finite: Element + Copy {
    fn finite(self) -> bool;
}
impl Finite for f64 {
    fn finite(self) -> bool {
        self.is_finite()
    }
}
impl Finite for Complex64 {
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

/// Whether all values are finite, in one pass without an early exit, which vectorizes.
pub(crate) fn all_finite<'a, T: Finite + 'a>(values: impl IntoIterator<Item = &'a T>) -> bool {
    !values
        .into_iter()
        .fold(false, |bad, &value| bad | !value.finite())
}

/// The view of a cotangent of the `expected` shape; another shape raises the
/// cotangent error.
pub(crate) fn cotangent_view<'a, T: Element, D: Dimension>(
    cotangent: &'a PyReadonlyArrayDyn<'_, T>,
    expected: &[usize],
) -> PyResult<ArrayView<'a, T, D>> {
    let view = cotangent.as_array();
    if view.shape() != expected {
        return Err(cotangent_error(expected));
    }
    view.into_dimensionality()
        .map_err(|_| cotangent_error(expected))
}

/// The view of a finite cotangent of the `expected` shape; another shape or a
/// non-finite entry raises the cotangent error.
pub(crate) fn finite_cotangent<'a, T: Finite, D: Dimension>(
    cotangent: &'a PyReadonlyArrayDyn<'_, T>,
    expected: &[usize],
) -> PyResult<ArrayView<'a, T, D>> {
    let view = cotangent_view(cotangent, expected)?;
    if all_finite(&view) {
        Ok(view)
    } else {
        Err(cotangent_error(expected))
    }
}

/// A finite vector cotangent of `length` entries.
pub(crate) fn vector_cotangent<T: Finite>(
    cotangent: &PyReadonlyArrayDyn<'_, T>,
    length: usize,
) -> PyResult<Vec<T>> {
    Ok(finite_cotangent::<_, Ix1>(cotangent, &[length])?.to_vec())
}

/// A finite matrix cotangent of shape `(rows, columns)`, as a column-major copy.
pub(crate) fn matrix_cotangent(
    cotangent: &Cotangent<'_>,
    shape: (usize, usize),
) -> PyResult<DMatrix<Complex>> {
    let expected: [usize; 2] = shape.into();
    let matrix = matrix_from_view(cotangent_view(cotangent, &expected)?);
    if all_finite(matrix.as_slice()) {
        Ok(matrix)
    } else {
        Err(cotangent_error(&expected))
    }
}

/// A finite cotangent of the `expected` shape as a column-major `(rows, columns)`
/// matrix whose rows run over the leading axes in C order. Merging the axes is a
/// view for C and native layouts and a copy otherwise.
pub(crate) fn merged_cotangent<D: Dimension>(
    cotangent: &Cotangent<'_>,
    expected: &[usize],
    (rows, columns): (usize, usize),
) -> PyResult<DMatrix<Complex>> {
    let view = cotangent_view::<_, D>(cotangent, expected)?;
    let merged = view
        .to_shape(((rows, columns), Order::RowMajor))
        .map_err(layout_error)?;
    let matrix = matrix_from_view(merged.view());
    if all_finite(matrix.as_slice()) {
        Ok(matrix)
    } else {
        Err(cotangent_error(expected))
    }
}

/// A finite 2x2 cotangent of a coefficient matrix.
pub(crate) fn matrix2_cotangent(cotangent: &Cotangent<'_>) -> PyResult<Matrix2> {
    let g = finite_cotangent::<_, Ix2>(cotangent, &[2, 2])?;
    Ok(Matrix2::from_fn(|i, j| g[(i, j)]))
}

// Inputs.

/// A finite column-major copy of the 2-D input argument `name`, in any layout.
pub(crate) fn from_array(
    value: PyReadonlyArray2<'_, Complex>,
    name: &str,
) -> PyResult<DMatrix<Complex>> {
    let matrix = matrix_from_view(value.as_array());
    if all_finite(matrix.as_slice()) {
        Ok(matrix)
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite")))
    }
}

/// Rows of the `(N, K)` argument `name`, in any layout; another width raises an error
/// that names the argument.
pub(crate) fn rows<T: Copy, const K: usize>(
    a: ArrayView2<'_, T>,
    name: &str,
) -> PyResult<Vec<[T; K]>> {
    if a.ncols() != K {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {K})"
        )));
    }
    Ok(a.rows()
        .into_iter()
        .map(|row| std::array::from_fn(|j| row[j]))
        .collect())
}

/// A `NumPy` matrix lent to the core: borrowed when it is C- or F-contiguous with
/// positive strides, otherwise packed once into column-major storage.
///
/// ndarray calls reversed (negative-stride) storage contiguous, so the layout
/// comes from `to_slice`, which accepts positive standard layouts only.
#[derive(Debug)]
pub(crate) struct LentMatrix<'a> {
    rows: usize,
    columns: usize,
    storage: Storage<'a>,
}

#[derive(Debug)]
enum Storage<'a> {
    /// The borrowed elements in column-major (F) order.
    ColumnMajor(&'a [Complex]),
    /// The borrowed elements in row-major (C) order.
    RowMajor(&'a [Complex]),
    /// A column-major copy of strided or reversed elements.
    Packed(DMatrix<Complex>),
}

impl<'a> LentMatrix<'a> {
    pub(crate) fn new(view: ArrayView2<'a, Complex>) -> Self {
        let (rows, columns) = view.dim();
        let storage = if let Some(data) = view.reversed_axes().to_slice() {
            Storage::ColumnMajor(data)
        } else if let Some(data) = view.to_slice() {
            Storage::RowMajor(data)
        } else {
            Storage::Packed(matrix_from_view(view))
        };
        Self {
            rows,
            columns,
            storage,
        }
    }

    /// The matrix as a faer view.
    pub(crate) fn faer(&self) -> MatRef<'_, Complex> {
        match &self.storage {
            Storage::ColumnMajor(data) => {
                MatRef::from_column_major_slice(data, self.rows, self.columns)
            }
            Storage::RowMajor(data) => MatRef::from_row_major_slice(data, self.rows, self.columns),
            Storage::Packed(matrix) => {
                MatRef::from_column_major_slice(matrix.as_slice(), self.rows, self.columns)
            }
        }
    }

    /// The matrix as a nalgebra view.
    pub(crate) fn nalgebra(&self) -> DMatrixView<'_, Complex, Dyn, Dyn> {
        let (data, row_stride, column_stride) = match &self.storage {
            Storage::ColumnMajor(data) => (*data, 1, self.rows),
            Storage::RowMajor(data) => (*data, self.columns, 1),
            Storage::Packed(matrix) => (matrix.as_slice(), 1, self.rows),
        };
        DMatrixView::from_slice_with_strides(
            data,
            self.rows,
            self.columns,
            row_stride,
            column_stride,
        )
    }
}

/// A column-major copy of a 2-D array of any layout.
pub(crate) fn matrix_from_view(a: ArrayView2<'_, Complex>) -> DMatrix<Complex> {
    // Tile the NumPy-to-column-major copy so large C-order inputs do not walk
    // one cache line per element. Both tiles fit in the CPU's L1 data cache.
    if a.t().is_standard_layout()
        && let Some(data) = a.as_slice_memory_order()
    {
        return DMatrix::from_column_slice(a.nrows(), a.ncols(), data);
    }
    let mut result = DMatrix::zeros(a.nrows(), a.ncols());
    if a.is_empty() {
        return result;
    }
    let fill = |(block, columns): (usize, &mut [Complex])| {
        for row in (0..a.nrows()).step_by(32) {
            for (j, column) in columns.chunks_mut(a.nrows()).enumerate() {
                for i in row..(row + 32).min(a.nrows()) {
                    column[i] = a[(i, 32 * block + j)];
                }
            }
        }
    };
    if a.len() >= 65_536 {
        result
            .as_mut_slice()
            .par_chunks_mut(32 * a.nrows())
            .enumerate()
            .for_each(fill);
    } else {
        result
            .as_mut_slice()
            .chunks_mut(32 * a.nrows())
            .enumerate()
            .for_each(fill);
    }
    result
}

// Outputs.

/// A C-order copy of a matrix the residual keeps.
pub(crate) fn matrix<'py>(
    py: Python<'py>,
    value: &DMatrix<Complex>,
) -> Bound<'py, PyArray2<Complex>> {
    Array2::from_shape_fn(value.shape(), |(i, j)| value[(i, j)]).into_pyarray(py)
}

/// A column-major matrix as an F-order array, without copying.
pub(crate) fn owned_matrix<T: Element + Scalar>(
    py: Python<'_>,
    value: DMatrix<T>,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    let shape = value.shape();
    Ok(Array2::from_shape_vec(shape.f(), Vec::from(value.data))
        .map_err(layout_error)?
        .into_pyarray(py))
}

/// Fixed-size rows as an `(N, K)` C-order array, without copying.
pub(crate) fn rows_array<T: Element, const K: usize>(
    py: Python<'_>,
    rows: Vec<[T; K]>,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    let count = rows.len();
    Ok(Array2::from_shape_vec((count, K), rows.into_flattened())
        .map_err(layout_error)?
        .into_pyarray(py))
}

/// A 2x2 coefficient matrix as a C-order array.
pub(crate) fn matrix2_array<'py>(
    py: Python<'py>,
    value: &Matrix2,
) -> Bound<'py, PyArray2<Complex>> {
    Array2::from_shape_fn((2, 2), |(i, j)| value[(i, j)]).into_pyarray(py)
}

/// An ndarray shape error as `ValueError`. The bindings build every array from
/// storage of the matching length, so this error marks a bug, not a bad input.
pub(crate) fn layout_error(error: ShapeError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
