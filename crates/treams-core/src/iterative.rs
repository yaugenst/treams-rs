//! Requested sphere-cluster illuminations without a dense interaction matrix.
// Mode, particle and Krylov indices are constructed and checked in this module.
#![allow(clippy::indexing_slicing)]

use std::sync::Arc;

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result,
    coeffs::{Material, Matrix2, MieResidual, mie_forward},
    finite,
    tmatrix::ClusterGradient,
    translation_plan::TranslationPlan,
    waves,
};

/// Restarted GMRES accuracy and memory budget. Both passes check the true residual.
#[derive(Clone, Copy, Debug)]
pub struct GmresOptions {
    /// Relative residual tolerance with respect to the right-hand side norm.
    pub rtol: f64,
    /// Absolute residual tolerance, useful for very small right-hand sides.
    pub atol: f64,
    /// Maximum number of Krylov vectors retained before a restart.
    pub restart: usize,
    /// Maximum total operator applications in the Arnoldi iterations.
    pub max_iterations: usize,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-10,
            atol: 0.0,
            restart: 30,
            max_iterations: 300,
        }
    }
}

impl GmresOptions {
    fn validate(self) -> Result<()> {
        if !self.rtol.is_finite()
            || !self.atol.is_finite()
            || self.rtol < 0.0
            || self.atol < 0.0
            || self.rtol + self.atol == 0.0
            || self.restart == 0
            || self.max_iterations == 0
        {
            return Err(Error::InvalidInput(
                "GMRES needs finite nonnegative tolerances, a positive tolerance, restart and iteration limit".into(),
            ));
        }
        Ok(())
    }
}

/// Independently recomputed residual of one converged illumination.
#[derive(Clone, Copy, Debug)]
pub struct Convergence {
    /// Arnoldi iterations performed, including all restarts.
    pub iterations: usize,
    /// Euclidean norm of the true residual.
    pub residual_norm: f64,
    /// Euclidean norm of the right-hand side.
    pub rhs_norm: f64,
}

/// Scattered local multipoles for only the requested incident columns.
#[derive(Debug)]
pub struct IterativeSolution {
    /// Column-major matrix with one scattered-multipole column per illumination.
    pub value: DMatrix<Complex>,
    /// Convergence certificate for each column.
    pub reports: Vec<Convergence>,
}

/// Geometry and local Mie coefficients; no particle-pair matrix is retained.
#[derive(Debug)]
pub struct SphereCluster {
    k0: f64,
    radii: Vec<f64>,
    positions: Vec<[f64; 3]>,
    orders: Vec<Vec<MieResidual>>,
    mode_orders: Vec<usize>,
    plan: TranslationPlan,
    dimension: usize,
}

impl SphereCluster {
    /// Homogeneous nonmagnetic spheres in vacuum, in the usual local helicity order.
    pub fn new(
        lmax: u32,
        k0: f64,
        radii: &[f64],
        epsilon: &[Complex],
        positions: &[[f64; 3]],
    ) -> Result<Self> {
        if !k0.is_finite()
            || k0 <= 0.0
            || radii.is_empty()
            || radii.len() != epsilon.len()
            || radii.len() != positions.len()
            || radii.iter().any(|r| !r.is_finite() || *r <= 0.0)
            || epsilon.iter().any(|&z| !finite(z))
            || positions.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidInput(
                "require positive finite k0 and radii, and matching finite permittivities and positions".into(),
            ));
        }
        for i in 0..radii.len() {
            for j in 0..i {
                let distance = positions[i]
                    .iter()
                    .zip(positions[j])
                    .fold(0.0_f64, |norm, (a, b)| norm.hypot(a - b));
                if distance <= radii[i] + radii[j] || !distance.is_finite() {
                    return Err(Error::InvalidInput(
                        "spherical particles must have finite separation and must not touch or overlap".into(),
                    ));
                }
            }
        }
        let modes = waves::modes(lmax)?;
        let dimension = modes
            .len()
            .checked_mul(radii.len())
            .ok_or_else(|| Error::InvalidInput("cluster is too large".into()))?;
        let orders = radii
            .iter()
            .zip(epsilon)
            .map(|(&radius, &epsilon)| {
                let materials = [
                    Material {
                        epsilon,
                        ..Material::default()
                    },
                    Material::default(),
                ];
                (1..=lmax)
                    .map(|l| mie_forward(l, &[k0 * radius], &materials))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            k0,
            radii: radii.to_vec(),
            positions: positions.to_vec(),
            orders,
            mode_orders: modes
                .iter()
                .step_by(2)
                .map(|mode| {
                    usize::try_from(mode.l - 1)
                        .map_err(|_| Error::InvalidInput("invalid multipole order".into()))
                })
                .collect::<Result<_>>()?,
            plan: TranslationPlan::new(&modes)?,
            dimension,
        })
    }

    /// Number of local multipoles across all particles.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    fn local(&self, input: &[Complex], adjoint: bool) -> Vec<Complex> {
        let mut output = vec![Complex::default(); self.dimension];
        let modes = self.mode_orders.len() * 2;
        for (particle, orders) in self.orders.iter().enumerate() {
            for (block, &order) in self.mode_orders.iter().enumerate() {
                let matrix = &orders[order].value;
                let offset = particle * modes + block * 2;
                for i in 0..2 {
                    for j in 0..2 {
                        let t = if adjoint {
                            matrix[(1 - j, 1 - i)].conj()
                        } else {
                            matrix[(1 - i, 1 - j)]
                        };
                        output[offset + i] += t * input[offset + j];
                    }
                }
            }
        }
        output
    }

    fn coupling(&self, input: &[Complex], adjoint: bool) -> Result<Vec<Complex>> {
        let modes = self.mode_orders.len() * 2;
        let mut output = vec![Complex::default(); self.dimension];
        output
            .par_chunks_mut(modes)
            .enumerate()
            .try_for_each(|(i, output)| -> Result<()> {
                for j in 0..self.positions.len() {
                    if i == j {
                        continue;
                    }
                    let (to, from) = if adjoint { (j, i) } else { (i, j) };
                    let displacement = std::array::from_fn(|axis| {
                        self.positions[to][axis] - self.positions[from][axis]
                    });
                    let block = self
                        .plan
                        .evaluate(Complex::new(self.k0, 0.0), displacement)?;
                    for col in 0..modes {
                        for row in 0..modes {
                            if adjoint {
                                output[col] +=
                                    block[col * modes + row].conj() * input[j * modes + row];
                            } else {
                                output[row] += block[col * modes + row] * input[j * modes + col];
                            }
                        }
                    }
                }
                Ok(())
            })?;
        Ok(output)
    }

    fn apply(&self, input: &[Complex], adjoint: bool) -> Result<Vec<Complex>> {
        let product = if adjoint {
            self.coupling(&self.local(input, true), true)?
        } else {
            self.local(&self.coupling(input, false)?, false)
        };
        Ok(input.iter().zip(product).map(|(x, y)| x - y).collect())
    }

    fn solve_rhs(
        &self,
        rhs: &DMatrix<Complex>,
        options: GmresOptions,
        adjoint: bool,
    ) -> Result<IterativeSolution> {
        let mut value = DMatrix::zeros(self.dimension, rhs.ncols());
        let mut reports = Vec::with_capacity(rhs.ncols());
        // Columns are sequential so the Krylov memory budget is independent of P;
        // the expensive particle-pair applications themselves use Rayon.
        for (column, right) in rhs.column_iter().enumerate() {
            let (answer, report) = gmres(right.as_slice(), options, |x| self.apply(x, adjoint))?;
            value.column_mut(column).copy_from_slice(&answer);
            reports.push(report);
        }
        Ok(IterativeSolution { value, reports })
    }

    /// Solve `(I - T C) X = T B` for only the columns supplied in `incident`.
    /// An unconverged solve returns an error containing the measured residual.
    pub fn solve(
        &self,
        incident: &DMatrix<Complex>,
        options: GmresOptions,
    ) -> Result<IterativeSolution> {
        options.validate()?;
        if incident.nrows() != self.dimension
            || incident.ncols() == 0
            || incident.iter().any(|&z| !finite(z))
        {
            return Err(Error::InvalidInput(
                "incident coefficients must be finite, with one row per local multipole and at least one column".into(),
            ));
        }
        let mut rhs = DMatrix::zeros(self.dimension, incident.ncols());
        for (mut output, input) in rhs.column_iter_mut().zip(incident.column_iter()) {
            output.copy_from_slice(&self.local(input.as_slice(), false));
        }
        self.solve_rhs(&rhs, options, false)
    }

    /// Share this operator through an `Arc` and retain requested columns for an implicit VJP.
    /// Krylov histories are discarded; independent residuals share all geometry and Mie data.
    pub fn record(
        self: Arc<Self>,
        incident: DMatrix<Complex>,
        options: GmresOptions,
    ) -> Result<IterativeResidual> {
        let solution = self.solve(&incident, options)?;
        Ok(IterativeResidual {
            operator: self,
            incident,
            options,
            solution,
        })
    }
}

/// Opaque first-order residual; Krylov histories are discarded after convergence.
#[derive(Debug)]
pub struct IterativeResidual {
    operator: Arc<SphereCluster>,
    incident: DMatrix<Complex>,
    options: GmresOptions,
    /// Forward values and independently checked convergence reports.
    pub solution: IterativeSolution,
}

/// Analytic implicit VJP without a dense coupling or coupling cotangent.
#[derive(Debug)]
pub struct IterativeGradient {
    /// Radius, position, permittivity and vacuum-wavenumber cotangents.
    pub cluster: ClusterGradient,
    /// Incident-coefficient cotangents under the real Hermitian pairing.
    pub incident: DMatrix<Complex>,
    /// Independent convergence certificates of the adjoint solves.
    pub reports: Vec<Convergence>,
}

impl IterativeResidual {
    /// Solve the conjugate-transpose system and contract each particle pair at once.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<IterativeGradient> {
        if cotangent.shape() != self.solution.value.shape() || cotangent.iter().any(|&z| !finite(z))
        {
            return Err(Error::InvalidInput("invalid illumination cotangent".into()));
        }
        let operator = self.operator;
        let adjoint = operator.solve_rhs(cotangent, self.options, true)?;
        let modes = operator.mode_orders.len() * 2;
        let particles = operator.positions.len();
        let columns = self.incident.ncols();
        let mut response = self.incident;
        let mut incident = DMatrix::zeros(operator.dimension, columns);
        for column in 0..columns {
            let scattered =
                operator.coupling(self.solution.value.column(column).as_slice(), false)?;
            for (value, coupling) in response.column_mut(column).iter_mut().zip(scattered) {
                *value += coupling;
            }
            incident
                .column_mut(column)
                .copy_from_slice(&operator.local(adjoint.value.column(column).as_slice(), true));
        }
        let pair_gradients = (0..particles)
            .into_par_iter()
            .map(|i| -> Result<(Vec<[f64; 3]>, f64)> {
                let mut positions = vec![[0.0; 3]; particles];
                let mut k0 = 0.0;
                let mut cotangent = vec![Complex::default(); modes * modes];
                for j in 0..particles {
                    if i == j {
                        continue;
                    }
                    cotangent.fill(Complex::default());
                    for col in 0..modes {
                        for row in 0..modes {
                            for p in 0..columns {
                                cotangent[col * modes + row] += incident[(i * modes + row, p)]
                                    * self.solution.value[(j * modes + col, p)].conj();
                            }
                        }
                    }
                    let displacement = std::array::from_fn(|axis| {
                        operator.positions[i][axis] - operator.positions[j][axis]
                    });
                    let (position, frequency) = operator.plan.pullback(
                        Complex::new(operator.k0, 0.0),
                        displacement,
                        &cotangent,
                    )?;
                    k0 += frequency;
                    for (axis, derivative) in position.into_iter().enumerate() {
                        positions[i][axis] += derivative;
                        positions[j][axis] -= derivative;
                    }
                }
                Ok((positions, k0))
            })
            .try_reduce(
                || (vec![[0.0; 3]; particles], 0.0),
                |(mut a, ak), (b, bk)| {
                    for (a, b) in a.iter_mut().zip(b) {
                        for (a, b) in a.iter_mut().zip(b) {
                            *a += b;
                        }
                    }
                    Ok((a, ak + bk))
                },
            )?;
        let mut cluster = ClusterGradient {
            radii: vec![0.0; particles],
            positions: pair_gradients.0,
            epsilon: vec![Complex::default(); particles],
            k0: pair_gradients.1,
        };
        for (particle, orders) in operator.orders.iter().enumerate() {
            let mut order_cotangents = vec![Matrix2::zeros(); orders.len()];
            for (block, &order) in operator.mode_orders.iter().enumerate() {
                let offset = particle * modes + block * 2;
                for i in 0..2 {
                    for j in 0..2 {
                        for p in 0..columns {
                            order_cotangents[order][(1 - i, 1 - j)] +=
                                adjoint.value[(offset + i, p)] * response[(offset + j, p)].conj();
                        }
                    }
                }
            }
            for (order, cotangent) in orders.iter().zip(order_cotangents) {
                let gradient = order.clone().pullback(&cotangent)?;
                cluster.radii[particle] += operator.k0 * gradient.sizes[0];
                cluster.k0 += operator.radii[particle] * gradient.sizes[0];
                cluster.epsilon[particle] += gradient.epsilon[0];
            }
        }
        Ok(IterativeGradient {
            cluster,
            incident,
            reports: adjoint.reports,
        })
    }
}

fn norm(vector: &[Complex]) -> f64 {
    vector.iter().fold(0.0_f64, |sum, z| sum.hypot(z.norm()))
}

fn gmres(
    rhs: &[Complex],
    options: GmresOptions,
    apply: impl Fn(&[Complex]) -> Result<Vec<Complex>>,
) -> Result<(Vec<Complex>, Convergence)> {
    let dimension = rhs.len();
    let restart = options.restart.min(dimension).min(options.max_iterations);
    let rhs_norm = norm(rhs);
    let tolerance = options.atol.max(options.rtol * rhs_norm);
    let mut answer = vec![Complex::default(); dimension];
    let mut residual = rhs.to_vec();
    let mut iterations = 0;
    loop {
        let residual_norm = norm(&residual);
        if residual_norm.is_finite() && residual_norm <= tolerance {
            return Ok((
                answer,
                Convergence {
                    iterations,
                    residual_norm,
                    rhs_norm,
                },
            ));
        }
        if !residual_norm.is_finite() || iterations >= options.max_iterations {
            return Err(Error::InvalidInput(format!(
                "GMRES did not converge after {iterations} iterations: residual {residual_norm:e}, tolerance {tolerance:e}"
            )));
        }
        let mut basis = Vec::with_capacity(restart + 1);
        basis.push(
            residual
                .iter()
                .map(|z| z / residual_norm)
                .collect::<Vec<_>>(),
        );
        let mut h = vec![Complex::default(); (restart + 1) * restart];
        let mut rotations = Vec::<(f64, Complex)>::with_capacity(restart);
        let mut g = vec![Complex::default(); restart + 1];
        g[0] = Complex::new(residual_norm, 0.0);
        let mut used = 0;
        for j in 0..restart {
            let mut w = apply(&basis[j])?;
            // Two modified Gram-Schmidt passes keep orthogonality near resonance.
            for _ in 0..2 {
                for i in 0..=j {
                    let dot: Complex = basis[i].iter().zip(&w).map(|(v, w)| v.conj() * w).sum();
                    h[j * (restart + 1) + i] += dot;
                    for (w, v) in w.iter_mut().zip(&basis[i]) {
                        *w -= dot * v;
                    }
                }
            }
            let next = norm(&w);
            h[j * (restart + 1) + j + 1] = Complex::new(next, 0.0);
            for (i, &(c, s)) in rotations.iter().enumerate() {
                let offset = j * (restart + 1) + i;
                let a = h[offset];
                let b = h[offset + 1];
                h[offset] = c * a + s * b;
                h[offset + 1] = -s.conj() * a + c * b;
            }
            let diagonal = j * (restart + 1) + j;
            let a = h[diagonal];
            let b = h[diagonal + 1];
            let length = a.norm().hypot(b.norm());
            if !length.is_finite() || length == 0.0 {
                return Err(Error::InvalidInput(
                    "GMRES Arnoldi breakdown before convergence".into(),
                ));
            }
            let phase = if a.norm() == 0.0 {
                Complex::new(1.0, 0.0)
            } else {
                a / a.norm()
            };
            let c = a.norm() / length;
            let s = phase * b.conj() / length;
            h[diagonal] = phase * length;
            h[diagonal + 1] = Complex::default();
            rotations.push((c, s));
            g[j + 1] = -s.conj() * g[j];
            g[j] *= c;
            used = j + 1;
            iterations += 1;
            if g[j + 1].norm() <= tolerance || next == 0.0 || iterations == options.max_iterations {
                break;
            }
            basis.push(w.into_iter().map(|z| z / next).collect());
        }
        for i in (0..used).rev() {
            let tail: Complex = ((i + 1)..used)
                .map(|j| h[j * (restart + 1) + i] * g[j])
                .sum();
            g[i] = (g[i] - tail) / h[i * (restart + 1) + i];
            for (x, v) in answer.iter_mut().zip(&basis[i]) {
                *x += g[i] * v;
            }
        }
        // Never certify convergence using the Hessenberg estimate alone.
        residual = rhs
            .iter()
            .zip(apply(&answer)?)
            .map(|(b, a)| b - a)
            .collect();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use super::*;
    use proptest::prelude::*;

    fn options() -> GmresOptions {
        GmresOptions {
            rtol: 2e-12,
            restart: 8,
            max_iterations: 100,
            ..GmresOptions::default()
        }
    }

    fn inputs(dimension: usize, columns: usize, seed: f64) -> DMatrix<Complex> {
        DMatrix::from_fn(dimension, columns, |i, j| {
            let phase = f64::from(u32::try_from(i + dimension * j).unwrap()) * 0.37 + seed;
            Complex::new(phase.sin(), (phase * 0.7).cos())
        })
    }

    fn pairing(a: &DMatrix<Complex>, b: &DMatrix<Complex>) -> f64 {
        a.iter().zip(b).map(|(a, b)| (a.conj() * b).re).sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(12))]

        #[test]
        fn requested_columns_match_dense_and_obey_symmetries(
            radius in 0.1_f64..0.35,
            epsilon in 1.2_f64..4.0,
            seed in -2.0_f64..2.0,
            scale in 0.7_f64..1.4,
            lmax in 1_u32..=2,
        ) {
            let radii = [radius, 1.1 * radius];
            let eps = [Complex::new(epsilon, 0.0), Complex::new(2.2, 0.03)];
            let positions = [[0.0, 0.0, 0.0], [1.3, 0.2, -0.1]];
            let operator = SphereCluster::new(lmax, 1.2, &radii, &eps, &positions).unwrap();
            let incident = inputs(operator.dimension(), 2, seed);
            let result = operator.solve(&incident, options()).unwrap();
            let dense = crate::tmatrix::cluster(lmax, 1.2, &radii, &eps, &positions).unwrap();
            let expected = dense.value() * &incident;
            prop_assert!((&result.value - &expected).norm() < 2e-10 * expected.norm());
            for report in &result.reports {
                prop_assert!(report.residual_norm <= options().rtol * report.rhs_norm);
            }
            let translated = positions.map(|p| p.map(|x| x + 0.17));
            let other = SphereCluster::new(lmax, 1.2, &radii, &eps, &translated).unwrap();
            prop_assert!((other.solve(&incident, options()).unwrap().value - &result.value).norm() < 2e-12);
            let scaled = SphereCluster::new(lmax, 1.2 / scale, &radii.map(|r| r * scale), &eps,
                &positions.map(|p| p.map(|x| x * scale))).unwrap();
            prop_assert!((scaled.solve(&incident, options()).unwrap().value - &result.value).norm() < 2e-11);
            let g = inputs(operator.dimension(), 2, seed + 0.3);
            let grad = Arc::new(operator).record(incident.clone(), options()).unwrap().pullback(&g).unwrap();
            let dense_grad = dense.pullback(&(&g * incident.adjoint())).unwrap();
            for i in 0..2 {
                prop_assert!((grad.cluster.radii[i] - dense_grad.radii[i]).abs() < 3e-10);
                prop_assert!((grad.cluster.epsilon[i] - dense_grad.epsilon[i]).norm() < 3e-10);
                for axis in 0..3 {
                    prop_assert!((grad.cluster.positions[i][axis] - dense_grad.positions[i][axis]).abs() < 3e-10);
                    prop_assert!((grad.cluster.positions[0][axis] + grad.cluster.positions[1][axis]).abs() < 1e-12);
                }
            }
            prop_assert!((grad.cluster.k0 - dense_grad.k0).abs() < 3e-10);
            prop_assert!((pairing(&g, &result.value) - pairing(&grad.incident, &incident)).abs() < 3e-10);
            let scale_gradient = radii.iter().zip(&grad.cluster.radii).map(|(r,g)| r*g).sum::<f64>()
                + positions.iter().flatten().zip(grad.cluster.positions.iter().flatten()).map(|(r,g)| r*g).sum::<f64>()
                - 1.2 * grad.cluster.k0;
            prop_assert!(scale_gradient.abs() < 3e-10);
        }
    }

    #[test]
    fn matrix_free_adjoint_and_physical_derivatives() {
        let radii = [0.28, 0.32, 0.19];
        let epsilon = [
            Complex::new(2.3, 0.0),
            Complex::new(3.1, 0.03),
            Complex::new(1.7, 0.0),
        ];
        let positions = [[0.0, 0.0, 0.0], [0.9, 0.2, 0.1], [0.2, 0.8, -0.2]];
        let operator = SphereCluster::new(2, 1.4, &radii, &epsilon, &positions).unwrap();
        let incident = inputs(operator.dimension(), 2, 0.1);
        let g = inputs(operator.dimension(), 2, 0.8);
        let u = incident.column(0);
        let v = g.column(0);
        let (u, v) = (u.as_slice(), v.as_slice());
        let left: Complex = v
            .iter()
            .zip(operator.apply(u, false).unwrap())
            .map(|(v, x)| v.conj() * x)
            .sum();
        let right: Complex = operator
            .apply(v, true)
            .unwrap()
            .iter()
            .zip(u)
            .map(|(v, x)| v.conj() * x)
            .sum();
        assert!((left - right).norm() < 1e-12);
        let gradient = Arc::new(operator)
            .record(incident.clone(), options())
            .unwrap()
            .pullback(&g)
            .unwrap();
        let h = 1e-5;
        for parameter in 0..5 {
            let objective = |step: f64| {
                let mut radii = radii;
                let mut epsilon = epsilon;
                let mut positions = positions;
                let mut k0 = 1.4;
                let mut incident = incident.clone();
                match parameter {
                    0 => radii[0] += step,
                    1 => epsilon[1] += step * Complex::new(0.3, -0.2),
                    2 => positions[2][1] += step,
                    3 => k0 += step,
                    _ => incident[(3, 1)] += step * Complex::new(-0.1, 0.4),
                }
                let op = SphereCluster::new(2, k0, &radii, &epsilon, &positions).unwrap();
                pairing(&g, &op.solve(&incident, options()).unwrap().value)
            };
            let fd = (objective(h) - objective(-h)) / (2.0 * h);
            let analytic = match parameter {
                0 => gradient.cluster.radii[0],
                1 => (gradient.cluster.epsilon[1].conj() * Complex::new(0.3, -0.2)).re,
                2 => gradient.cluster.positions[2][1],
                3 => gradient.cluster.k0,
                _ => (gradient.incident[(3, 1)].conj() * Complex::new(-0.1, 0.4)).re,
            };
            assert!(
                (fd - analytic).abs() < 2e-8 * (1.0 + analytic.abs()),
                "parameter {parameter}: {analytic} != {fd}"
            );
        }
    }

    #[test]
    fn gmres_zero_happy_breakdown_restarts_and_failure() {
        let identity = |x: &[Complex]| Ok(x.to_vec());
        let (zero, report) = gmres(&[Complex::default(); 3], options(), identity).unwrap();
        assert_eq!(zero, vec![Complex::default(); 3]);
        assert_eq!(report.iterations, 0);
        let rhs = vec![Complex::new(0.2, -0.3); 3];
        let (answer, report) = gmres(&rhs, options(), identity).unwrap();
        assert!((norm(&answer) - norm(&rhs)).abs() < 1e-14);
        assert_eq!(report.iterations, 1);
        let diagonal = |x: &[Complex]| {
            Ok(x.iter()
                .enumerate()
                .map(|(i, x)| {
                    Complex::new(1.0 + f64::from(u32::try_from(i).unwrap()) * 0.3, 0.1) * x
                })
                .collect())
        };
        let limited = GmresOptions {
            restart: 1,
            max_iterations: 1,
            ..options()
        };
        assert!(gmres(&rhs, limited, diagonal).is_err());
        let restarted = GmresOptions {
            restart: 1,
            ..options()
        };
        let (_, report) = gmres(&rhs, restarted, diagonal).unwrap();
        assert!(report.iterations > 1);
        assert!(report.residual_norm <= restarted.rtol * report.rhs_norm);
        assert!(gmres(&rhs, options(), |_| Ok(vec![Complex::default(); 3])).is_err());
    }

    #[test]
    fn single_sphere_and_particle_permutation() {
        let eps = [Complex::new(2.1, 0.02), Complex::new(3.2, 0.0)];
        let radii = [0.2, 0.3];
        let positions = [[0.0; 3], [1.0, 0.2, -0.3]];
        let single = SphereCluster::new(2, 1.2, &radii[..1], &eps[..1], &positions[..1]).unwrap();
        let input = inputs(single.dimension(), 2, 0.1);
        let value = single.solve(&input, options()).unwrap();
        let dense = crate::tmatrix::sphere(
            2,
            1.2,
            &radii[..1],
            &[
                Material {
                    epsilon: eps[0],
                    ..Material::default()
                },
                Material::default(),
            ],
        )
        .unwrap();
        assert!((&value.value - dense.value * &input).norm() < 1e-13);
        let original = SphereCluster::new(2, 1.2, &radii, &eps, &positions).unwrap();
        let shuffled = SphereCluster::new(
            2,
            1.2,
            &[radii[1], radii[0]],
            &[eps[1], eps[0]],
            &[positions[1], positions[0]],
        )
        .unwrap();
        let input = inputs(original.dimension(), 2, 0.5);
        let m = input.nrows() / 2;
        let permute = |x: &DMatrix<Complex>| {
            DMatrix::from_fn(x.nrows(), x.ncols(), |i, j| x[((i + m) % (2 * m), j)])
        };
        let actual = shuffled.solve(&permute(&input), options()).unwrap().value;
        let expected = permute(&original.solve(&input, options()).unwrap().value);
        assert!((actual - expected).norm() < 1e-12);
    }

    #[test]
    fn domain_errors_are_explicit() {
        let eps = [Complex::new(2.0, 0.0)];
        assert!(SphereCluster::new(1, 0.0, &[0.2], &eps, &[[0.0; 3]]).is_err());
        assert!(SphereCluster::new(1, 1.0, &[0.2], &eps, &[]).is_err());
        assert!(
            SphereCluster::new(
                1,
                1.0,
                &[0.2, 0.3],
                &[eps[0]; 2],
                &[[0.0; 3], [0.4, 0.0, 0.0]]
            )
            .is_err()
        );
        let op = SphereCluster::new(1, 1.0, &[0.2], &eps, &[[0.0; 3]]).unwrap();
        let input = DMatrix::zeros(6, 1);
        assert!(
            op.solve(
                &input,
                GmresOptions {
                    rtol: -1.0,
                    ..options()
                }
            )
            .is_err()
        );
        assert!(op.solve(&DMatrix::zeros(5, 1), options()).is_err());
        assert!(op.solve(&DMatrix::zeros(6, 0), options()).is_err());
    }
}
