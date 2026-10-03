//! `plane_polarization`, `plane_expansion`, `cylindrical_plane_expansion`,
//! `plane_field`, `plane_phases` and `plane_permutation` with their contexts: plane
//! waves and their expansions into multipole bases (`treams_core::pw`).
use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Array2, Ix2, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, fpenv::ieee};

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{context, detached, error},
    convert::{
        C1, C2, CDyn, Cotangent, LentMatrix, R2, finite_cotangent, layout_error, merged_cotangent,
        owned_matrix, rows, rows_array,
    },
    fields::operator_array,
};

/// The polarization vector of one plane wave: `pw::polarization`.
#[pyfunction]
pub(crate) fn plane_polarization(
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> PyResult<[Complex; 3]> {
    ieee(|| treams_core::pw::polarization(vector, pol, helicity).map_err(error))
}

context!(PlaneExpansionContext(treams_core::pw::ExpansionResidual));
#[pymethods]
impl PlaneExpansionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, C2<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, treams_core::pw::ExpansionResidual::shape)?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                rows_array(py, gradient.positions)?,
                rows_array(py, gradient.vectors)?,
            ))
        })
    }
}
/// Record the expansion of plane waves in a spherical basis: `pw::expansion`.
#[pyfunction]
pub(crate) fn plane_expansion(
    py: Python<'_>,
    modes: Vec<(usize, i32, i32, u8)>,
    positions: Vec<[f64; 3]>,
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    helicity: bool,
    fixed_vectors: bool,
) -> PyResult<(C2<'_>, PlaneExpansionContext)> {
    ieee(|| {
        let basis = make_basis(modes, positions);
        let (value, residual) = detached(py, move || {
            treams_core::pw::expansion(basis, vectors, polarizations, helicity, fixed_vectors)
        })?;
        finish_plane_expansion(py, value, residual)
    })
}

/// Record the expansion of plane waves in a cylindrical basis: `pw::expansion`.
#[pyfunction]
pub(crate) fn cylindrical_plane_expansion(
    py: Python<'_>,
    modes: Vec<(usize, f64, i32, u8)>,
    positions: Vec<[f64; 3]>,
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    helicity: bool,
    fixed_vectors: bool,
) -> PyResult<(C2<'_>, PlaneExpansionContext)> {
    ieee(|| {
        let basis = make_cyl_basis(modes, positions);
        let (value, residual) = detached(py, move || {
            treams_core::pw::expansion(basis, vectors, polarizations, helicity, fixed_vectors)
        })?;
        finish_plane_expansion(py, value, residual)
    })
}

/// A plane-wave expansion matrix, moved into an F-ordered array, and its context.
fn finish_plane_expansion(
    py: Python<'_>,
    value: DMatrix<Complex>,
    residual: treams_core::pw::ExpansionResidual,
) -> PyResult<(C2<'_>, PlaneExpansionContext)> {
    Ok((
        owned_matrix(py, value)?,
        PlaneExpansionContext::new(residual),
    ))
}

context!(PlaneFieldContext(
    treams_core::pw::FieldResidual,
    /// The shape of the output array: `(N, 3)` for weighted fields, `(N, 3, modes)`
    /// for the operator.
    shape: Vec<usize>,
));
#[pymethods]
impl PlaneFieldContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C1<'py>, R2<'py>, C2<'py>)> {
        ieee(|| {
            let (residual, g) = self.residual.take_if(|residual| {
                merged_cotangent::<IxDyn>(&cotangent, &self.shape, residual.shape())
            })?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                result.coefficients.into_pyarray(py),
                rows_array(py, result.points)?,
                rows_array(py, result.vectors)?,
            ))
        })
    }
}
/// Record weighted plane-wave fields at points, or their operator: `pw::field`.
#[pyfunction]
#[pyo3(signature=(vectors,polarizations,points,coefficients,helicity,fixed_vectors))]
pub(crate) fn plane_field<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, Complex>,
    polarizations: Vec<u8>,
    points: PyReadonlyArray2<'py, f64>,
    coefficients: Option<PyReadonlyArray1<'py, Complex>>,
    helicity: bool,
    fixed_vectors: bool,
) -> PyResult<(CDyn<'py>, PlaneFieldContext)> {
    ieee(|| {
        let vectors = rows(vectors.as_array(), "vectors")?;
        let points = rows(points.as_array(), "points")?;
        let coefficients = coefficients.map(|c| c.as_array().to_vec());
        let shape = if coefficients.is_some() {
            vec![points.len(), 3]
        } else {
            vec![points.len(), 3, vectors.len()]
        };
        let (value, residual) = detached(py, move || {
            treams_core::pw::field(
                vectors,
                polarizations,
                points,
                coefficients,
                helicity,
                fixed_vectors,
            )
        })?;
        // Transfer Rust storage directly; expose (samples, Cartesian, modes) by strides.
        let output = if shape.len() == 2 {
            Array2::from_shape_vec((shape[0], 3), Vec::from(value.data))
                .map_err(layout_error)?
                .into_dyn()
        } else {
            operator_array(value)?.into_dyn()
        }
        .into_pyarray(py);
        Ok((output, PlaneFieldContext::new(residual, shape)))
    })
}

context!(PlanePhasesContext(treams_core::pw::PhasesResidual));
#[pymethods]
impl PlanePhasesContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, C2<'py>)> {
        ieee(|| {
            let (residual, g) = self.residual.take_if(|residual| {
                finite_cotangent::<_, Ix2>(&cotangent, &<[usize; 2]>::from(residual.shape()))
            })?;
            // A contiguous cotangent stays borrowed while the read-only Python borrow
            // is alive.
            let lent = LentMatrix::new(g);
            let view = lent.nalgebra();
            let gradient = detached(py, move || residual.pullback(&view))?;
            Ok((
                rows_array(py, gradient.points)?,
                rows_array(py, gradient.vectors)?,
            ))
        })
    }
}

/// Record the phases `exp(i k.r)` of plane waves at points: `pw::phases`.
#[pyfunction]
pub(crate) fn plane_phases<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    vectors: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(C2<'py>, PlanePhasesContext)> {
    ieee(|| {
        let points = rows(points.as_array(), "points")?;
        let vectors = rows(vectors.as_array(), "vectors")?;
        let (value, residual) = detached(py, move || treams_core::pw::phases(points, vectors))?;
        Ok((owned_matrix(py, value)?, PlanePhasesContext::new(residual)))
    })
}

context!(PlanePermutationContext(
    treams_core::pw::PermutationResidual
));

#[pymethods]
impl PlanePermutationContext {
    fn pullback<'py>(&mut self, py: Python<'py>, cotangent: Cotangent<'py>) -> PyResult<C2<'py>> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, treams_core::pw::PermutationResidual::shape)?;
            rows_array(py, detached(py, move || residual.pullback(&g))?)
        })
    }
}

/// Record the coefficients of plane waves under a cyclic change of the axes: `pw::permutation`.
#[pyfunction]
#[allow(clippy::float_cmp)] // Polarizations are exact discrete labels, not measured floats.
pub(crate) fn plane_permutation<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, Complex>,
    polarizations: PyReadonlyArray1<'py, f64>,
    turns: usize,
    helicity: bool,
) -> PyResult<(C2<'py>, PlanePermutationContext)> {
    ieee(|| {
        let vectors = rows(vectors.as_array(), "vectors")?;
        let polarizations = polarizations
            .as_array()
            .iter()
            .map(|&p| {
                if p == 0.0 || p == 1.0 {
                    Ok(u8::from(p == 1.0))
                } else {
                    Err(PyValueError::new_err("polarizations must be 0 or 1"))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let (value, residual) = detached(py, move || {
            treams_core::pw::permutation(vectors, polarizations, turns, helicity)
        })?;
        Ok((
            owned_matrix(py, value)?,
            PlanePermutationContext::new(residual),
        ))
    })
}
