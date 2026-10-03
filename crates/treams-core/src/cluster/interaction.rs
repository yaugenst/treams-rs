//! Dense solves of `(I - T C) X = T B` through one LU factorization of `I - T C`.
//!
//! [`interaction`] returns the full interacting T-matrix `X = (I - T C)⁻¹ T`.
//! [`InteractionFactor`] keeps the factorization and solves only requested incident
//! fields `B`, any number of times.
//!
//! Upstream: `treams.TMatrix.interaction.solve()` for [`interaction`].
//! [`InteractionFactor`] is a treams-rs extension.

use std::sync::Arc;

use nalgebra::DMatrix;

use crate::{
    Complex, Error, Result,
    linalg::{
        Lu, product, product_adjoint_right, product_adjoint_right_into, product_into,
        product_views, view, view_mut,
    },
    numerics::{self, finite},
};

/// Block-diagonal local T-matrix; a dense local matrix is a single block.
#[derive(Clone, Debug)]
pub(crate) struct LocalMatrix {
    blocks: Vec<DMatrix<Complex>>,
}

impl LocalMatrix {
    /// One dense local matrix.
    pub(crate) fn from_dense(local: DMatrix<Complex>) -> Result<Self> {
        if !local.is_square() || local.nrows() == 0 || local.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "local matrix must be finite, square and nonempty".into(),
            ));
        }
        Ok(Self {
            blocks: vec![local],
        })
    }

    /// The diagonal blocks of a block-diagonal local matrix, one per particle.
    pub(crate) fn from_blocks(blocks: Vec<DMatrix<Complex>>) -> Result<Self> {
        if blocks.is_empty()
            || blocks
                .iter()
                .any(|m| !m.is_square() || m.nrows() == 0 || m.iter().any(|z| !finite(*z)))
        {
            return Err(Error::InvalidInput(
                "local blocks must be finite nonempty square matrices".into(),
            ));
        }
        Ok(Self { blocks })
    }

    pub(crate) fn dimension(&self) -> usize {
        self.blocks.iter().map(DMatrix::nrows).sum()
    }

    /// The dense matrix, with zeros off the diagonal blocks.
    fn to_dense(&self) -> Result<DMatrix<Complex>> {
        let mut result = numerics::zeros(self.dimension(), self.dimension())?;
        let mut offset = 0;
        for block in &self.blocks {
            result
                .view_mut((offset, offset), block.shape())
                .copy_from(block);
            offset += block.nrows();
        }
        Ok(result)
    }

    /// `T right`, or `Tᴴ right` if `adjoint`.
    pub(crate) fn apply(
        &self,
        right: &DMatrix<Complex>,
        adjoint: bool,
    ) -> Result<DMatrix<Complex>> {
        let mut result = numerics::zeros(self.dimension(), right.ncols())?;
        self.apply_into(&mut result, right, adjoint);
        Ok(result)
    }

    fn apply_into(&self, result: &mut DMatrix<Complex>, right: &DMatrix<Complex>, adjoint: bool) {
        let mut offset = 0;
        for block in &self.blocks {
            let target = view_mut(result).subrows_mut(offset, block.nrows());
            let rhs = view(right).subrows(offset, block.nrows());
            if adjoint {
                product_into(target, view(block).adjoint(), rhs);
            } else {
                product_into(target, view(block), rhs);
            }
            offset += block.nrows();
        }
    }

    /// Diagonal blocks of `adjoint responseᴴ`: the cotangent of each local block when
    /// the solve's cotangent is `adjoint` and its local matrix multiplies `response`.
    pub(crate) fn block_gradients(
        &self,
        adjoint: &DMatrix<Complex>,
        response: &DMatrix<Complex>,
    ) -> Vec<DMatrix<Complex>> {
        let mut offset = 0;
        self.blocks
            .iter()
            .map(|block| {
                let n = block.nrows();
                let gradient = product_views(
                    view(adjoint).subrows(offset, n),
                    view(response).subrows(offset, n).adjoint(),
                );
                offset += n;
                gradient
            })
            .collect()
    }
}

/// `I - T C` for a finite coupling of matching size.
fn operator(local: &LocalMatrix, coupling: &DMatrix<Complex>) -> Result<DMatrix<Complex>> {
    if coupling.shape() != (local.dimension(), local.dimension())
        || coupling.iter().any(|z| !finite(*z))
    {
        return Err(Error::InvalidInput(
            "invalid cluster coupling matrix".into(),
        ));
    }
    let mut operator = -local.apply(coupling, false)?;
    operator.set_diagonal(&(operator.diagonal().add_scalar(Complex::new(1.0, 0.0))));
    Ok(operator)
}

/// The LU factorization of `I - T C`, kept with the local T-matrix `T` and the
/// coupling `C`, for any number of incident-field solves.
///
/// A block-diagonal `T` stays a list of particle blocks; it is never expanded to a
/// dense matrix. [`record`](Self::record) shares the factor through an [`Arc`], so
/// every recorded solve keeps only its own incident and scattered fields.
///
/// treams-rs extension. treams solves the full system once with
/// `TMatrix.interaction.solve()`; [`interaction`] computes that result.
#[derive(Clone, Debug)]
pub struct InteractionFactor {
    local: LocalMatrix,
    coupling: DMatrix<Complex>,
    lu: Lu,
}

impl InteractionFactor {
    /// Factor `I - T C` for a dense local matrix `T`.
    pub fn new(local: DMatrix<Complex>, coupling: DMatrix<Complex>) -> Result<Self> {
        Self::factorize(LocalMatrix::from_dense(local)?, coupling)
    }

    /// Factor `I - T C` for a block-diagonal `T` given by its particle blocks.
    pub fn from_blocks(local: Vec<DMatrix<Complex>>, coupling: DMatrix<Complex>) -> Result<Self> {
        Self::factorize(LocalMatrix::from_blocks(local)?, coupling)
    }

    fn factorize(local: LocalMatrix, coupling: DMatrix<Complex>) -> Result<Self> {
        let lu = Lu::new(operator(&local, &coupling)?)?;
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
        self.solve_system(self.local.apply(incident, false)?)
    }

    /// Solve like [`solve`](Self::solve) and keep what the pullback needs. The
    /// residual shares this factor.
    pub fn record(self: &Arc<Self>, incident: DMatrix<Complex>) -> Result<IlluminateResidual> {
        let value = self.solve(&incident)?;
        Ok(IlluminateResidual {
            factor: Arc::clone(self),
            incident,
            value,
        })
    }

    /// `(I - T C)⁻¹ rhs`; a non-finite solution means a numerically singular operator.
    fn solve_system(&self, mut rhs: DMatrix<Complex>) -> Result<DMatrix<Complex>> {
        self.lu.solve_in_place(view_mut(&mut rhs))?;
        if rhs.iter().any(|z| !finite(*z)) {
            return Err(Error::Singular);
        }
        Ok(rhs)
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
}

/// What [`interaction`] saves for its pullback: the factorization of `I - T C` and
/// the interacting T-matrix `X`.
#[derive(Clone, Debug)]
pub struct InteractionResidual {
    factor: InteractionFactor,
    value: DMatrix<Complex>,
}

/// Solve a multiple-scattering system for the interacting T-matrix
/// `X = (I - T C)⁻¹ T`, keeping its LU factorization for the pullback.
///
/// `local` is the block-diagonal T-matrix `T` of all particles, and `coupling` the
/// singular expansion `C` between them, zero on the diagonal blocks.
///
/// Upstream: `treams.TMatrix.interaction.solve`.
pub fn interaction(
    local: DMatrix<Complex>,
    coupling: DMatrix<Complex>,
) -> Result<InteractionResidual> {
    solve_interacting(LocalMatrix::from_dense(local)?, coupling)
}

/// [`interaction`] for a block-diagonal `T` given by its particle blocks, which are
/// never expanded to a dense matrix for the factorization.
pub(crate) fn interaction_blocks(
    blocks: Vec<DMatrix<Complex>>,
    coupling: DMatrix<Complex>,
) -> Result<InteractionResidual> {
    solve_interacting(LocalMatrix::from_blocks(blocks)?, coupling)
}

fn solve_interacting(
    local: LocalMatrix,
    coupling: DMatrix<Complex>,
) -> Result<InteractionResidual> {
    let factor = InteractionFactor::factorize(local, coupling)?;
    let value = factor.solve_system(factor.local.to_dense()?)?;
    Ok(InteractionResidual { factor, value })
}

/// Gradients of the inputs of [`interaction`], in the order of its arguments.
///
/// `L` is the type of the local-matrix gradient: one matrix from
/// [`InteractionResidual::pullback`], or one per diagonal block from
/// [`InteractionResidual::pullback_blocks`].
#[derive(Clone, Debug)]
pub struct InteractionGradient<L = DMatrix<Complex>> {
    /// Gradient of the local T-matrix `T`.
    pub local: L,
    /// Gradient of the coupling `C`.
    pub coupling: DMatrix<Complex>,
}

impl InteractionResidual {
    /// The interacting T-matrix `X`.
    #[must_use]
    pub const fn value(&self) -> &DMatrix<Complex> {
        &self.value
    }

    /// The shape of the interacting T-matrix.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.value.shape()
    }

    /// Gradients of `T` and `C` from `cotangent`, the gradient of a real loss with
    /// respect to `X`.
    ///
    /// The pullback solves one adjoint system with the forward LU. It has no Rayon
    /// reduction; the LU solve and the matrix products run in faer with the worker
    /// count of [`linalg`](crate::linalg).
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<InteractionGradient> {
        self.pullback_with(cotangent, |_, adjoint, response| {
            product_adjoint_right(adjoint, response)
        })
    }

    /// Like [`pullback`](Self::pullback), with one gradient per diagonal block of a
    /// block-diagonal `T`, or a single one for a dense `T`.
    pub fn pullback_blocks(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<InteractionGradient<Vec<DMatrix<Complex>>>> {
        self.pullback_with(cotangent, LocalMatrix::block_gradients)
    }

    fn pullback_with<T>(
        self,
        cotangent: &DMatrix<Complex>,
        local_gradient: impl FnOnce(&LocalMatrix, &DMatrix<Complex>, &DMatrix<Complex>) -> T,
    ) -> Result<InteractionGradient<T>> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|z| !finite(*z)) {
            return Err(Error::InvalidInput(
                "invalid interacting-matrix cotangent".into(),
            ));
        }
        let Self {
            factor:
                InteractionFactor {
                    local,
                    coupling: mut adjoint,
                    lu,
                },
            value,
        } = self;
        // C is dead after forming I + C X. Reuse its allocation for A^-H G,
        // then release the factors before constructing the input cotangents.
        let mut response = product(&adjoint, &value);
        response.set_diagonal(&(response.diagonal().add_scalar(Complex::new(1.0, 0.0))));
        adjoint.copy_from(cotangent);
        lu.solve_adjoint_in_place(view_mut(&mut adjoint))?;
        drop(lu);
        let local_gradient = local_gradient(&local, &adjoint, &response);
        // Both remaining square buffers can be reused: response becomes T^H Y,
        // and the old adjoint becomes (T^H Y) X^H.
        local.apply_into(&mut response, &adjoint, true);
        product_adjoint_right_into(&mut adjoint, &response, &value);
        Ok(InteractionGradient {
            local: local_gradient,
            coupling: adjoint,
        })
    }
}

/// What [`InteractionFactor::record`] saves for its pullback: the shared factor and
/// the incident and scattered fields of the requested columns.
#[derive(Debug)]
pub struct IlluminateResidual {
    factor: Arc<InteractionFactor>,
    incident: DMatrix<Complex>,
    value: DMatrix<Complex>,
}

/// Gradients of the local T-matrix, the coupling and the incident fields of an
/// [`IlluminateResidual`].
#[derive(Debug)]
pub struct IlluminateGradient {
    /// One gradient per local block (one for a dense local matrix).
    pub local: Vec<DMatrix<Complex>>,
    /// Gradient of the coupling `C`.
    pub coupling: DMatrix<Complex>,
    /// Gradient of the incident fields `B`.
    pub incident: DMatrix<Complex>,
}

impl IlluminateResidual {
    /// The scattered coefficients, one column per incident field, which the pullback
    /// reads.
    #[must_use]
    pub const fn value(&self) -> &DMatrix<Complex> {
        &self.value
    }

    /// The shape of the scattered coefficients: channels and incident fields.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.value.shape()
    }

    /// Gradients of `T`, `C` and `B` from `cotangent`, the gradient of a real loss with
    /// respect to the scattered fields.
    ///
    /// With `Y = (I - T C)⁻ᴴ G` from the shared LU, the gradients are `Y (B + C X)ᴴ` on
    /// each local block, `Tᴴ Y Xᴴ` and `Tᴴ Y`. The pullback has no Rayon reduction; the
    /// LU solve and the matrix products run in faer with the worker count of
    /// [`linalg`](crate::linalg).
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<IlluminateGradient> {
        self.factor.validate(cotangent)?;
        if cotangent.shape() != self.shape() {
            return Err(Error::InvalidInput(
                "cotangent shape must match scattered fields".into(),
            ));
        }
        let factor = &*self.factor;
        let mut adjoint = cotangent.clone();
        factor.lu.solve_adjoint_in_place(view_mut(&mut adjoint))?;
        let incident = factor.local.apply(&adjoint, true)?;
        let response = &self.incident + product(&factor.coupling, &self.value);
        let local = factor.local.block_gradients(&adjoint, &response);
        let coupling = product_adjoint_right(&incident, &self.value);
        Ok(IlluminateGradient {
            local,
            coupling,
            incident,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nalgebra::DMatrix;
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{InteractionFactor, InteractionGradient, interaction};
    use crate::{
        Complex,
        test_support::{DEFAULT_CASES, complex, complex_matrix, prop_assert_close},
    };

    /// Local blocks of sizes 1 to 3, a coupling, incident fields and cotangents for
    /// one to four illuminations.
    type Inputs = (
        Vec<DMatrix<Complex>>,
        DMatrix<Complex>,
        DMatrix<Complex>,
        DMatrix<Complex>,
    );

    fn inputs() -> impl Strategy<Value = Inputs> {
        (prop::collection::vec(1_usize..=3, 1..=3), 1_usize..=4).prop_flat_map(|(sizes, p)| {
            let n = sizes.iter().sum();
            (
                sizes
                    .iter()
                    .map(|&size| complex_matrix(size, size, 1.0))
                    .collect::<Vec<_>>(),
                complex_matrix(n, n, 1.0),
                complex_matrix(n, p, 1.0),
                complex_matrix(n, p, 1.0),
            )
        })
    }

    proptest! {
        // Each case factors three systems of at most 9 channels.
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        #[test]
        fn requested_columns_match_the_implicit_adjoint_closed_forms(
            (blocks, coupling, incident, weight) in inputs(),
            amplitude in complex(2.0),
        ) {
            check_requested_columns(blocks, &coupling, &incident, &weight, amplitude)?;
        }
    }

    /// With `X = (I - T C)⁻¹ T A` and `Y = (I - T C)⁻ᴴ G` from an independent nalgebra
    /// LU, recorded solves equal `X` and are linear in `A`, and the pullback returns
    /// the closed forms of the implicit adjoint (formal/Formal/ImplicitAdjoint.lean):
    /// `T̄ = Y (A + C X)ᴴ`, restricted to the local blocks, `C̄ = Tᴴ Y Xᴴ` and
    /// `Ā = Tᴴ Y`. The dense factorization returns the full `T̄`, and the interacting
    /// T-matrix applied to `A`, with the cotangent `G Aᴴ`, gives the same gradients.
    fn check_requested_columns(
        mut blocks: Vec<DMatrix<Complex>>,
        coupling: &DMatrix<Complex>,
        incident: &DMatrix<Complex>,
        weight: &DMatrix<Complex>,
        amplitude: Complex,
    ) -> Result<(), TestCaseError> {
        let n = coupling.nrows();
        let mut local = DMatrix::zeros(n, n);
        let mut offset = 0;
        for block in &blocks {
            local
                .view_mut((offset, offset), block.shape())
                .copy_from(block);
            offset += block.nrows();
        }
        // ‖T‖ ‖C‖ < 1/2 keeps I - T C well conditioned: ‖(I - T C)⁻¹‖ < 2.
        let shrink = Complex::from(0.5 / (1.0 + local.norm() * coupling.norm()));
        local *= shrink;
        for block in &mut blocks {
            *block *= shrink;
        }
        let identity = DMatrix::identity(n, n);
        let system = &identity - &local * coupling;
        let x = system.clone().lu().solve(&(&local * incident)).unwrap();
        let y = system.adjoint().lu().solve(weight).unwrap();
        let full_local = &y * (incident + coupling * &x).adjoint();
        let expected_coupling = local.adjoint() * &y * x.adjoint();
        let expected_incident = local.adjoint() * &y;
        let value_tolerance = 1e-12 * (1.0 + incident.norm());
        let tolerance = value_tolerance * (1.0 + weight.norm());

        let factor =
            Arc::new(InteractionFactor::from_blocks(blocks.clone(), coupling.clone()).unwrap());
        let residual = factor.record(incident.clone()).unwrap();
        prop_assert_close!(residual.value(), &x, value_tolerance);
        let scaled = factor.solve(&(incident * amplitude)).unwrap();
        let expected = residual.value() * amplitude;
        prop_assert_close!(scaled, expected, value_tolerance * (1.0 + amplitude.norm()));
        let gradient = residual.pullback(weight).unwrap();
        prop_assert_eq!(gradient.local.len(), blocks.len());
        let mut offset = 0;
        for (block, actual) in blocks.iter().zip(&gradient.local) {
            let expected = full_local
                .view((offset, offset), block.shape())
                .into_owned();
            prop_assert_close!(actual, &expected, tolerance, "block at {}", offset);
            offset += block.nrows();
        }
        prop_assert_close!(&gradient.coupling, &expected_coupling, tolerance);
        prop_assert_close!(&gradient.incident, &expected_incident, tolerance);

        let dense = Arc::new(InteractionFactor::new(local.clone(), coupling.clone()).unwrap());
        let dense_gradient = dense
            .record(incident.clone())
            .unwrap()
            .pullback(weight)
            .unwrap();
        prop_assert_eq!(dense_gradient.local.len(), 1);
        prop_assert_close!(&dense_gradient.local[0], &full_local, tolerance);

        let interacting = interaction(local, coupling.clone()).unwrap();
        prop_assert_close!(interacting.value() * incident, x, value_tolerance);
        let InteractionGradient {
            local: local_gradient,
            coupling: coupling_gradient,
        } = interacting
            .pullback(&(weight * incident.adjoint()))
            .unwrap();
        prop_assert_close!(local_gradient, full_local, tolerance);
        prop_assert_close!(coupling_gradient, expected_coupling, tolerance);
        Ok(())
    }
}
