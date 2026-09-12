//! Requested incident fields and reusable factorizations of multiple scattering.

use std::sync::Arc;

use faer::{Accum, linalg::matmul::matmul};
use nalgebra::DMatrix;

use crate::{
    Complex, Error, Result, finite,
    interaction::{self, LocalMatrix, product, product_adjoint_right, view, view_mut},
    linalg::Lu,
};

/// Assemble `I - T C` for another linear-algebra backend, using the CPU convention.
pub fn operator(local: DMatrix<Complex>, coupling: &DMatrix<Complex>) -> Result<DMatrix<Complex>> {
    interaction::operator(&interaction::local_dense(local)?, coupling)
}

/// An immutable factorization shared by any number of incident-field solves.
/// Local particle blocks are retained without a dense block-diagonal expansion.
#[derive(Debug)]
pub struct Factor {
    local: LocalMatrix,
    coupling: DMatrix<Complex>,
    lu: Lu,
}

impl Factor {
    /// Factor `I - T C` once. All supplied data is owned by the factorization.
    pub fn new(local: DMatrix<Complex>, coupling: DMatrix<Complex>) -> Result<Self> {
        Self::prepare(interaction::local_dense(local)?, coupling)
    }

    /// Factor a block-diagonal collection of local particle T-matrices.
    pub fn from_blocks(local: Vec<DMatrix<Complex>>, coupling: DMatrix<Complex>) -> Result<Self> {
        Self::prepare(interaction::local_blocks(local)?, coupling)
    }

    fn prepare(local: LocalMatrix, coupling: DMatrix<Complex>) -> Result<Self> {
        let lu = Lu::new(interaction::operator(&local, &coupling)?)?;
        Ok(Self {
            local,
            coupling,
            lu,
        })
    }

    /// Number of multipole channels.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.local.dimension()
    }

    /// Solve `(I - T C) scattered = T incident` for only the supplied columns.
    pub fn solve(&self, incident: &DMatrix<Complex>) -> Result<DMatrix<Complex>> {
        self.validate(incident)?;
        let mut value = self.local.apply(incident, false);
        self.lu.solve_in_place(view_mut(&mut value));
        if value.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Ok(value)
    }

    fn validate(&self, values: &DMatrix<Complex>) -> Result<()> {
        if values.nrows() != self.dimension()
            || values.ncols() == 0
            || values.iter().any(|&z| !finite(z))
        {
            return Err(Error::InvalidInput(
                "require finite channel-by-illumination columns matching the factorization".into(),
            ));
        }
        Ok(())
    }

    /// Record a solve. Residuals share this immutable factorization and own thin fields.
    pub fn record(self: &Arc<Self>, incident: DMatrix<Complex>) -> Result<Residual> {
        let value = self.solve(&incident)?;
        Ok(Residual {
            factor: Arc::clone(self),
            incident,
            value,
        })
    }
}

/// Thin incident/scattered fields and a shared factorization for one pullback.
#[derive(Debug)]
pub struct Residual {
    factor: Arc<Factor>,
    incident: DMatrix<Complex>,
    /// Scattered coefficients, one column per incident field.
    pub value: DMatrix<Complex>,
}

/// Gradients with respect to local T matrices, coupling, and incident fields.
#[derive(Debug)]
pub struct Gradient {
    /// One cotangent per supplied local block (one for a dense local matrix).
    pub local: Vec<DMatrix<Complex>>,
    /// Coupling-matrix cotangent.
    pub coupling: DMatrix<Complex>,
    /// Incident-field cotangent.
    pub incident: DMatrix<Complex>,
}

impl Residual {
    /// Native implicit adjoint; factorization work is reused from forward.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<Gradient> {
        self.factor.validate(cotangent)?;
        if cotangent.shape() != self.value.shape() {
            return Err(Error::InvalidInput(
                "cotangent shape must match scattered fields".into(),
            ));
        }
        let mut adjoint = cotangent.clone();
        self.factor
            .lu
            .solve_adjoint_in_place(view_mut(&mut adjoint));
        let incident = self.factor.local.apply(&adjoint, true);
        let response = &self.incident + product(&self.factor.coupling, &self.value);
        let blocks = match &self.factor.local {
            LocalMatrix::Dense(matrix) => std::slice::from_ref(matrix),
            LocalMatrix::Blocks(blocks) => blocks.as_slice(),
        };
        let mut offset = 0;
        let local = blocks
            .iter()
            .map(|block| {
                let n = block.nrows();
                let mut gradient = DMatrix::zeros(n, n);
                matmul(
                    view_mut(&mut gradient),
                    Accum::Replace,
                    view(&adjoint).subrows(offset, n),
                    view(&response).subrows(offset, n).adjoint(),
                    Complex::new(1.0, 0.0),
                    faer::get_global_parallelism(),
                );
                offset += n;
                gradient
            })
            .collect();
        let coupling = product_adjoint_right(&incident, &self.value);
        Ok(Gradient {
            local,
            coupling,
            incident,
        })
    }
}

/// Factor and record one requested-illumination solve.
pub fn forward(
    local: DMatrix<Complex>,
    coupling: DMatrix<Complex>,
    incident: DMatrix<Complex>,
) -> Result<Residual> {
    Arc::new(Factor::new(local, coupling)?).record(incident)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]
        #[test]
        fn requested_columns_and_complete_adjoint(scale in 0.2_f64..0.8, phase in -1.0_f64..1.0, n in 2_usize..7, p in 1_usize..4) {
            let sample=|i:usize,j:usize|Complex::new(f64::from(u32::try_from(i+1).unwrap())*0.03, f64::from(u32::try_from(j+1).unwrap())*0.02);
            let local=DMatrix::from_fn(n,n,|i,j|sample(i,j)*scale);
            let coupling=DMatrix::from_fn(n,n,|i,j|if i==j {Complex::default()} else {sample(j,i)*Complex::new(0.2,phase)});
            let incident=DMatrix::from_fn(n,p,sample);
            let weight=DMatrix::from_fn(n,p,|i,j|sample(j,i).conj());
            let full=interaction::forward(local.clone(),coupling.clone()).unwrap();
            let factor=Arc::new(Factor::new(local.clone(),coupling.clone()).unwrap());
            let residual=factor.record(incident.clone()).unwrap();
            prop_assert!((&residual.value-product(&full.value,&incident)).norm()<1e-12);
            let twice=factor.solve(&(&incident*Complex::new(2.0,-0.5))).unwrap();
            prop_assert!((twice-&residual.value*Complex::new(2.0,-0.5)).norm()<1e-12);
            let gradient=residual.pullback(&weight).unwrap();
            let d=Complex::new(0.17,-0.23);
            for parameter in 0..3 {
                let objective=|step:f64| {
                    let mut t=local.clone(); let mut c=coupling.clone(); let mut a=incident.clone();
                    for z in match parameter {0=>t.iter_mut(),1=>c.iter_mut(),_=>a.iter_mut()} {*z+=step*d;}
                    let value=Factor::new(t,c).unwrap().solve(&a).unwrap();
                    value.iter().zip(weight.iter()).map(|(v,g)|(g.conj()*v).re).sum::<f64>()
                };
                let finite=(objective(1e-5)-objective(-1e-5))/(2e-5);
                let entries=match parameter {0=>gradient.local[0].iter(),1=>gradient.coupling.iter(),_=>gradient.incident.iter()};
                let analytic=entries.map(|z|(z.conj()*d).re).sum::<f64>();
                prop_assert!((finite-analytic).abs()<2e-9*(1.0+analytic.abs()));
            }
        }
    }
}
