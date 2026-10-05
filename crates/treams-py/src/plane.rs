//! `plane_polarization`, `plane_expansion`, `cylindrical_plane_expansion`,
//! `plane_field`, `plane_phases` and `plane_permutation` with their contexts: plane
//! waves and their expansions into multipole bases (`treams_core::pw`).
use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Array2, Ix2, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, fpenv::ieee};

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{context, detached, error, restore_state, state_array},
    convert::{
        C1, C2, CDyn, Cotangent, LentMatrix, R2, RealTangent, Tangent, finite_cotangent,
        layout_error, matrix_cotangent, merged_cotangent, owned_matrix, rows, rows_array,
        rows_from_dyn, vector_tangent,
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
    #[staticmethod]
    #[pyo3(signature = (multipoles, positions, modes, cylindrical=false))]
    fn _state_spec(
        multipoles: usize,
        positions: usize,
        modes: usize,
        cylindrical: bool,
    ) -> PyResult<usize> {
        ieee(|| {
            treams_core::pw::ExpansionResidual::state_size(
                multipoles,
                positions,
                modes,
                cylindrical,
            )
            .map_err(error)
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| Ok(Self::new(restore_state(&state)?)))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        positions: RealTangent<'py>,
        vectors: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let positions = rows_from_dyn(&positions, "position tangent")?;
            let vectors = rows_from_dyn(&vectors, "wavevector tangent")?;
            let residual = &self.residual;
            owned_matrix(
                py,
                detached(py, move || residual.pushforward(&positions, &vectors))?,
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, C2<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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
    #[staticmethod]
    fn _state_spec(points: usize, modes: usize, weighted: bool) -> PyResult<usize> {
        ieee(|| treams_core::pw::FieldResidual::state_size(points, modes, weighted).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| {
            let residual: treams_core::pw::FieldResidual = restore_state(&state)?;
            let (rows, columns) = residual.shape();
            let shape = if residual.coefficient_count() == 0 {
                vec![rows / 3, 3, columns]
            } else {
                vec![rows / 3, 3]
            };
            Ok(Self::new(residual, shape))
        })
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        coefficients: Tangent<'py>,
        points: RealTangent<'py>,
        vectors: Tangent<'py>,
    ) -> PyResult<CDyn<'py>> {
        ieee(|| {
            let coefficients = vector_tangent(&coefficients, self.residual.coefficient_count())?;
            let points = rows_from_dyn(&points, "point tangent")?;
            let vectors = rows_from_dyn(&vectors, "wavevector tangent")?;
            let residual = &self.residual;
            let value = detached(py, move || {
                residual.pushforward(&coefficients, &points, &vectors)
            })?;
            field_array(py, value, &self.shape)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C1<'py>, R2<'py>, C2<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = merged_cotangent::<IxDyn>(&cotangent, &self.shape, residual.shape())?;
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
        let output = field_array(py, value, &shape)?;
        Ok((output, PlaneFieldContext::new(residual, shape)))
    })
}

/// Transfer a field or field tangent with its recorded public layout.
fn field_array<'py>(
    py: Python<'py>,
    value: DMatrix<Complex>,
    shape: &[usize],
) -> PyResult<CDyn<'py>> {
    let output = if shape.len() == 2 {
        Array2::from_shape_vec((shape[0], 3), Vec::from(value.data))
            .map_err(layout_error)?
            .into_dyn()
    } else {
        operator_array(value)?.into_dyn()
    };
    Ok(output.into_pyarray(py))
}

context!(PlanePhasesContext(treams_core::pw::PhasesResidual));
#[pymethods]
impl PlanePhasesContext {
    #[staticmethod]
    fn _state_spec(points: usize, modes: usize) -> PyResult<usize> {
        ieee(|| treams_core::pw::PhasesResidual::state_size(points, modes).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| Ok(Self::new(restore_state(&state)?)))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        points: RealTangent<'py>,
        vectors: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let points = rows_from_dyn(&points, "point tangent")?;
            let vectors = rows_from_dyn(&vectors, "wavevector tangent")?;
            let residual = &self.residual;
            owned_matrix(
                py,
                detached(py, move || residual.pushforward(&points, &vectors))?,
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, C2<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = finite_cotangent::<_, Ix2>(&cotangent, &<[usize; 2]>::from(residual.shape()))?;
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
    #[staticmethod]
    fn _state_spec(modes: usize) -> PyResult<usize> {
        ieee(|| treams_core::pw::PermutationResidual::state_size(modes).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| Ok(Self::new(restore_state(&state)?)))
    }

    fn pushforward<'py>(&self, py: Python<'py>, vectors: Tangent<'py>) -> PyResult<C2<'py>> {
        ieee(|| {
            let vectors = rows_from_dyn(&vectors, "wavevector tangent")?;
            let residual = &self.residual;
            owned_matrix(py, detached(py, move || residual.pushforward(&vectors))?)
        })
    }

    fn pullback<'py>(&self, py: Python<'py>, cotangent: Cotangent<'py>) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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
