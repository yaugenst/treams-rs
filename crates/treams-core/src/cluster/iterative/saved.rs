//! Exact-size callback state for a matrix-free sphere illumination.
//!
//! Mie interfaces and the converged solution are copied; restoring rebuilds only
//! the angular translation plan, without evaluating Mie coefficients or GMRES.
//! treams-rs extension.

use std::sync::Arc;

use nalgebra::DMatrix;

use super::{IterativeResidual, IterativeSolution, IterativeSphereCluster};
use crate::{
    Complex, MAX_DEGREE, Result,
    coeffs::MieResidual,
    linalg::{Convergence, GmresOptions},
    saved::{Reader, SavedState, Writer, invalid},
    sw::{self, TranslationPlan},
};

impl IterativeResidual {
    /// Callback-state bytes for an illumination with these static dimensions.
    /// The layout keeps geometry, Mie residuals, solver options, incident and
    /// scattered columns, and the primal convergence certificates.
    pub fn state_size(lmax: u32, particles: usize, columns: usize) -> Result<usize> {
        if !(1..=MAX_DEGREE.unsigned_abs()).contains(&lmax) || particles == 0 || columns == 0 {
            return Err(invalid());
        }
        let lmax = usize::try_from(lmax).map_err(|_| invalid())?;
        let mie = lmax
            .checked_mul(MieResidual::state_size(1)?)
            .ok_or_else(invalid)?;
        let rows = lmax
            .checked_add(2)
            .and_then(|n| n.checked_mul(lmax))
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(particles))
            .ok_or_else(invalid)?;
        let coefficients = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(32))
            .ok_or_else(invalid)?;
        // Three dimensions, k0 and four GMRES options occupy 64 bytes. Geometry
        // needs four real scalars per particle; every report needs three scalars.
        particles
            .checked_mul(mie.checked_add(32).ok_or_else(invalid)?)
            .and_then(|n| n.checked_add(coefficients))
            .and_then(|n| columns.checked_mul(24).and_then(|c| n.checked_add(c)))
            .and_then(|n| n.checked_add(64))
            .ok_or_else(invalid)
    }
}

impl SavedState for IterativeResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let operator = &self.operator;
        let lmax = operator.degrees.first().map_or(0, Vec::len);
        let particles = operator.radii.len();
        let columns = self.incident.ncols();
        let size = Self::state_size(
            u32::try_from(lmax).map_err(|_| invalid())?,
            particles,
            columns,
        )?;
        let mut writer = Writer::new(size);
        writer.usize(lmax);
        writer.usize(particles);
        writer.usize(columns);
        writer.f64(operator.k0);
        writer.f64(self.options.rtol);
        writer.f64(self.options.atol);
        writer.usize(self.options.restart);
        writer.usize(self.options.max_iterations);
        for &radius in &operator.radii {
            writer.f64(radius);
        }
        for &position in operator.positions.iter().flatten() {
            writer.f64(position);
        }
        for residual in operator.degrees.iter().flatten() {
            residual.write_state(&mut writer);
        }
        for &value in self.incident.iter().chain(self.solution.value.iter()) {
            writer.complex(value);
        }
        for report in &self.solution.convergence {
            writer.usize(report.iterations);
            writer.f64(report.residual_norm);
            writer.f64(report.rhs_norm);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let lmax = u32::try_from(reader.usize()?).map_err(|_| invalid())?;
        let particles = reader.usize()?;
        let columns = reader.usize()?;
        if bytes.len() != Self::state_size(lmax, particles, columns)? {
            return Err(invalid());
        }
        let k0 = reader.f64()?;
        let options = GmresOptions {
            rtol: reader.f64()?,
            atol: reader.f64()?,
            restart: reader.usize()?,
            max_iterations: reader.usize()?,
        };
        options.validate()?;
        let radii = (0..particles)
            .map(|_| reader.f64())
            .collect::<Result<Vec<_>>>()?;
        let positions = (0..particles)
            .map(|_| Ok([reader.f64()?, reader.f64()?, reader.f64()?]))
            .collect::<Result<Vec<_>>>()?;
        let mie_size = MieResidual::state_size(1)?;
        let degrees = (0..particles)
            .map(|_| {
                (0..lmax)
                    .map(|_| MieResidual::from_state(reader.raw(mie_size)?))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let modes = sw::modes(lmax)?;
        // Exact state-size validation already checked these products for overflow.
        let dimension = modes.len() * particles;
        let incident = read_matrix(&mut reader, dimension, columns)?;
        let value = read_matrix(&mut reader, dimension, columns)?;
        let convergence = (0..columns)
            .map(|_| {
                Ok(Convergence {
                    iterations: reader.usize()?,
                    residual_norm: reader.f64()?,
                    rhs_norm: reader.f64()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        reader.finish()?;
        let count = usize::try_from(lmax).map_err(|_| invalid())?;
        Ok(Self {
            operator: Arc::new(IterativeSphereCluster {
                k0,
                radii,
                positions,
                degrees,
                mode_degrees: crate::tmatrix::block_degrees(count).collect(),
                plan: TranslationPlan::between(&modes, &modes, true)?,
                dimension,
            }),
            incident,
            options,
            solution: IterativeSolution { value, convergence },
        })
    }
}

fn read_matrix(reader: &mut Reader<'_>, rows: usize, columns: usize) -> Result<DMatrix<Complex>> {
    let data = (0..rows * columns)
        .map(|_| reader.complex())
        .collect::<Result<Vec<_>>>()?;
    Ok(DMatrix::from_vec(rows, columns, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::patterned;

    fn residual() -> IterativeResidual {
        Arc::new(
            IterativeSphereCluster::new(
                2,
                1.3,
                &[0.2, 0.24],
                &[Complex::new(2.3, 0.04), Complex::new(3.1, 0.02)],
                &[[0.1, 0.0, -0.1], [1.2, 0.2, 0.4]],
            )
            .unwrap(),
        )
        .record(
            patterned(32, 3, 0.3),
            GmresOptions {
                rtol: 2e-12,
                ..GmresOptions::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn saved_state_preserves_solution_certificates_and_reusable_derivatives() {
        let residual = residual();
        let state = residual.save_state().unwrap();
        assert_eq!(state.len(), IterativeResidual::state_size(2, 2, 3).unwrap());
        let restored = IterativeResidual::from_state(&state).unwrap();
        assert_eq!(restored.solution.value, residual.solution.value);
        for (actual, expected) in restored
            .solution
            .convergence
            .iter()
            .zip(&residual.solution.convergence)
        {
            assert_eq!(actual.iterations, expected.iterations);
            assert_eq!(
                actual.residual_norm.to_bits(),
                expected.residual_norm.to_bits()
            );
            assert_eq!(actual.rhs_norm.to_bits(), expected.rhs_norm.to_bits());
        }
        let dr = [0.03, -0.02];
        let de = [Complex::new(0.2, -0.03), Complex::new(-0.1, 0.04)];
        let dp = [[0.03, 0.02, -0.04], [-0.02, 0.01, 0.03]];
        let db = patterned(32, 3, -0.2);
        let tangent = residual.pushforward(0.07, &dr, &de, &dp, &db).unwrap();
        assert_eq!(
            restored
                .pushforward(0.07, &dr, &de, &dp, &db)
                .unwrap()
                .value,
            tangent.value
        );
        let cotangent = patterned(32, 3, 0.7);
        let actual = restored.pullback(&cotangent).unwrap();
        let expected = residual.pullback(&cotangent).unwrap();
        assert_eq!(actual.cluster.k0.to_bits(), expected.cluster.k0.to_bits());
        assert_eq!(actual.cluster.radii, expected.cluster.radii);
        assert_eq!(actual.cluster.epsilon, expected.cluster.epsilon);
        assert_eq!(actual.cluster.positions, expected.cluster.positions);
        assert_eq!(actual.incident, expected.incident);
        assert_eq!(
            restored
                .pushforward(0.07, &dr, &de, &dp, &db)
                .unwrap()
                .value,
            tangent.value
        );
        assert_eq!(restored.save_state().unwrap(), state);
    }

    #[test]
    fn saved_state_checks_exact_dimensions_before_allocating() {
        let state = residual().save_state().unwrap();
        for end in [0, 8, 16, state.len() - 1] {
            assert!(IterativeResidual::from_state(&state[..end]).is_err());
        }
        let mut extra = state.clone();
        extra.push(0);
        assert!(IterativeResidual::from_state(&extra).is_err());
        for index in 0..3 {
            let mut malformed = state.clone();
            malformed[index * 8..(index + 1) * 8].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(IterativeResidual::from_state(&malformed).is_err());
        }
        let mut nested = state;
        // Three headers and the scalar solver settings, then two particles' geometry.
        nested[128..136].copy_from_slice(&2_u64.to_le_bytes());
        assert!(IterativeResidual::from_state(&nested).is_err());
        assert!(IterativeResidual::state_size(2, usize::MAX, 1).is_err());
        assert!(IterativeResidual::state_size(2, 1, 0).is_err());
    }
}
