//! `solve`, `eig` and `svdvals` with their contexts (`treams_core::linalg`).
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use treams_core::{Complex, fpenv::ieee, linalg};

use crate::{
    context::{context, detached},
    convert::{
        C1, C2, Cotangent, R1, RealCotangent, from_array, matrix, matrix_cotangent, owned_matrix,
        vector_cotangent,
    },
};

context!(SolveContext(linalg::SolveResidual));

#[pymethods]
impl SolveContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, C2<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, linalg::SolveResidual::shape)?;
            let gradient = detached(py, move || residual.pullback(g))?;
            Ok((
                owned_matrix(py, gradient.operator)?,
                owned_matrix(py, gradient.rhs)?,
            ))
        })
    }
}

/// Record the solution X of `operator X = rhs`: `linalg::solve_owned`.
#[pyfunction]
pub(crate) fn solve<'py>(
    py: Python<'py>,
    operator: PyReadonlyArray2<'py, Complex>,
    rhs: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(C2<'py>, SolveContext)> {
    ieee(|| {
        let a = from_array(operator, "operator")?;
        let b = from_array(rhs, "rhs")?;
        let residual = detached(py, move || linalg::solve_owned(a, b))?;
        // A C-ordered copy: the residual keeps the solution for the pullback.
        Ok((matrix(py, residual.value())?, SolveContext::new(residual)))
    })
}

context!(EigContext(linalg::EigResidual));

#[pymethods]
impl EigContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        eigenvalues: Cotangent<'py>,
        eigenvectors: Cotangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let (residual, (values, vectors)) = self.residual.take_if(|residual| {
                let values = vector_cotangent(&eigenvalues, residual.values().len())?;
                let vectors = matrix_cotangent(&eigenvectors, residual.vectors().shape())?;
                Ok((values, vectors))
            })?;
            owned_matrix(
                py,
                detached(py, move || residual.pullback(&values, vectors))?,
            )
        })
    }
}

/// Record the eigenvalues and unit right eigenvectors of a matrix: `linalg::eig`.
#[pyfunction]
pub(crate) fn eig<'py>(
    py: Python<'py>,
    operator: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(C1<'py>, C2<'py>, EigContext)> {
    ieee(|| {
        let a = from_array(operator, "operator")?;
        let residual = detached(py, move || linalg::eig(&a))?;
        // Copies, the eigenvectors in C order: the residual keeps both for the pullback.
        Ok((
            residual.values().to_vec().into_pyarray(py),
            matrix(py, residual.vectors())?,
            EigContext::new(residual),
        ))
    })
}

context!(SvdvalsContext(linalg::SvdvalsResidual));

#[pymethods]
impl SvdvalsContext {
    /// The singular values are real, so only the real part of a complex
    /// cotangent enters the real pairing.
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: RealCotangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_if(|residual| vector_cotangent(&cotangent, residual.values().len()))?;
            owned_matrix(py, detached(py, move || residual.pullback(&g))?)
        })
    }
}

/// Record the singular values of a matrix: `linalg::svdvals`.
#[pyfunction]
pub(crate) fn svdvals<'py>(
    py: Python<'py>,
    operator: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(R1<'py>, SvdvalsContext)> {
    ieee(|| {
        let operator = from_array(operator, "operator")?;
        let residual = detached(py, move || linalg::svdvals(&operator))?;
        // A copy: the residual keeps the singular values for the pullback.
        Ok((
            residual.values().to_vec().into_pyarray(py),
            SvdvalsContext::new(residual),
        ))
    })
}
