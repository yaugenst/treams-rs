//! Matrix-free sphere clusters: GMRES solves for requested incident fields that never
//! form the coupling matrix. treams-rs extension.
#![allow(clippy::indexing_slicing)] // Mode, particle and Krylov indices built and checked here.

use std::sync::Arc;

use faer::{Accum, MatMut, Par, linalg::matmul::matmul};
use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result,
    cluster::SphereClusterGradient,
    coeffs::{Matrix2, MieResidual, mie, to_mode_order},
    linalg::{Convergence, GmresOptions, gmres, view},
    numerics::{finite, parallel::try_fold_ordered},
    special::Radial,
    sw::{self, TranslationPlan},
};

/// Scattered local multipoles for only the requested incident columns.
#[derive(Debug)]
pub struct IterativeSolution {
    /// Column-major matrix with one scattered-multipole column per illumination.
    pub value: DMatrix<Complex>,
    /// Convergence certificate for each column.
    pub convergence: Vec<Convergence>,
}

/// The matrix-free counterpart of [`sphere_cluster`](super::sphere_cluster): the same
/// spheres, solved with GMRES for the requested incident columns only.
///
/// It keeps the geometry and the local Mie coefficients, never a particle-pair matrix.
///
/// treams-rs extension; treams has no matrix-free cluster solver.
#[derive(Debug)]
pub struct IterativeSphereCluster {
    k0: f64,
    radii: Vec<f64>,
    positions: Vec<[f64; 3]>,
    degrees: Vec<Vec<MieResidual>>,
    mode_degrees: Vec<usize>,
    plan: TranslationPlan,
    dimension: usize,
}

impl IterativeSphereCluster {
    /// Homogeneous nonmagnetic spheres in vacuum, with the arguments of
    /// [`sphere_cluster`](super::sphere_cluster). Each particle's modes follow
    /// [`sw::modes`] in helicity convention.
    pub fn new(
        lmax: u32,
        k0: f64,
        radii: &[f64],
        epsilon: &[Complex],
        positions: &[[f64; 3]],
    ) -> Result<Self> {
        crate::cluster::validate_spheres(k0, radii, epsilon, positions)?;
        let modes = sw::modes(lmax)?;
        let dimension = modes
            .len()
            .checked_mul(radii.len())
            .ok_or_else(|| Error::InvalidInput("cluster is too large".into()))?;
        let degrees = radii
            .iter()
            .zip(epsilon)
            .map(|(&radius, &epsilon)| {
                let materials = crate::cluster::vacuum_sphere(epsilon);
                (1..=lmax)
                    .map(|l| mie(l, &[k0 * radius], &materials))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let count = degrees.first().map_or(0, Vec::len);
        Ok(Self {
            k0,
            radii: radii.to_vec(),
            positions: positions.to_vec(),
            degrees,
            mode_degrees: crate::tmatrix::block_degrees(count).collect(),
            plan: TranslationPlan::between(&modes, &modes, true)?,
            dimension,
        })
    }

    /// Number of local multipoles across all particles.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// `T input`, or `Tᴴ input` if `adjoint`, from the 2 x 2 Mie block of each degree
    /// and order.
    ///
    /// [`sphere`](crate::tmatrix::sphere) places the same blocks in a dense T-matrix
    /// for [`sphere_cluster`](super::sphere_cluster); this applies them in place.
    fn local(&self, input: &[Complex], adjoint: bool) -> Vec<Complex> {
        let mut output = vec![Complex::default(); self.dimension];
        let modes = self.modes_per_particle();
        for (particle, degrees) in self.degrees.iter().enumerate() {
            for (block, &degree) in self.mode_degrees.iter().enumerate() {
                let matrix = to_mode_order(degrees[degree].value());
                let offset = particle * modes + block * 2;
                for i in 0..2 {
                    for j in 0..2 {
                        let t = if adjoint {
                            matrix[(j, i)].conj()
                        } else {
                            matrix[(i, j)]
                        };
                        output[offset + i] += t * input[offset + j];
                    }
                }
            }
        }
        output
    }

    /// Multipoles per particle.
    const fn modes_per_particle(&self) -> usize {
        self.mode_degrees.len() * 2
    }

    /// The particle-pair coupling `C`, or `Cᴴ`, applied to `columns` column-major
    /// vectors. Each translation block is evaluated once for all columns.
    fn coupling(&self, input: &[Complex], columns: usize, adjoint: bool) -> Result<Vec<Complex>> {
        let modes = self.modes_per_particle();
        let particles = self.positions.len();
        let rows: Vec<Vec<Complex>> = crate::threads::install(|| {
            (0..particles)
                .into_par_iter()
                .map(|i| -> Result<Vec<Complex>> {
                    // Rows of particle `i`, column-major over the input columns.
                    let mut output = vec![Complex::default(); modes * columns];
                    for j in (0..particles).filter(|&j| j != i) {
                        let (to, from) = if adjoint { (j, i) } else { (i, j) };
                        let displacement = std::array::from_fn(|axis| {
                            self.positions[to][axis] - self.positions[from][axis]
                        });
                        let block = self.plan.evaluate(
                            Complex::new(self.k0, 0.0),
                            displacement,
                            Radial::Singular,
                        )?;
                        for (column, output) in output.chunks_exact_mut(modes).enumerate() {
                            let source = &input[column * self.dimension + j * modes..][..modes];
                            apply_block(&block, source, output, adjoint);
                        }
                    }
                    Ok(output)
                })
                .collect::<Result<_>>()
        })?;
        let mut output = vec![Complex::default(); self.dimension * columns];
        for (i, rows) in rows.iter().enumerate() {
            for (column, values) in rows.chunks_exact(modes).enumerate() {
                output[column * self.dimension + i * modes..][..modes].copy_from_slice(values);
            }
        }
        Ok(output)
    }

    /// `(I - T C) input`, or `(I - T C)ᴴ input = (I - Cᴴ Tᴴ) input` if `adjoint`: the
    /// operator that GMRES inverts.
    fn apply(&self, input: &[Complex], adjoint: bool) -> Result<Vec<Complex>> {
        let product = if adjoint {
            self.coupling(&self.local(input, true), 1, true)?
        } else {
            self.local(&self.coupling(input, 1, false)?, false)
        };
        Ok(input.iter().zip(product).map(|(x, y)| x - y).collect())
    }

    /// Solve `(I - T C) X = rhs`, or `(I - T C)ᴴ X = rhs` if `adjoint`, one column of
    /// `rhs` at a time.
    fn solve_rhs(
        &self,
        rhs: &DMatrix<Complex>,
        options: GmresOptions,
        adjoint: bool,
    ) -> Result<IterativeSolution> {
        let mut value = DMatrix::zeros(self.dimension, rhs.ncols());
        let mut convergence = Vec::with_capacity(rhs.ncols());
        // Columns are sequential so the Krylov memory budget is independent of P;
        // the expensive particle-pair applications themselves use Rayon.
        for (column, right) in rhs.column_iter().enumerate() {
            let (answer, report) = gmres(right.as_slice(), options, |x| self.apply(x, adjoint))?;
            value.column_mut(column).copy_from_slice(&answer);
            convergence.push(report);
        }
        Ok(IterativeSolution { value, convergence })
    }

    /// Solve `(I - T C) X = T B` for only the columns `B` supplied in `incident`.
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

    /// Solve like [`solve`](Self::solve) and keep what the pullback needs. The
    /// residual shares this cluster through the [`Arc`]; the Krylov vectors are dropped.
    pub fn record(
        self: &Arc<Self>,
        incident: DMatrix<Complex>,
        options: GmresOptions,
    ) -> Result<IterativeResidual> {
        let solution = self.solve(&incident, options)?;
        Ok(IterativeResidual {
            operator: Arc::clone(self),
            incident,
            options,
            solution,
        })
    }
}

/// What [`IterativeSphereCluster::record`] saves for its pullback: the shared
/// cluster, the incident fields, the GMRES options and the solution.
#[derive(Debug)]
pub struct IterativeResidual {
    operator: Arc<IterativeSphereCluster>,
    incident: DMatrix<Complex>,
    options: GmresOptions,
    solution: IterativeSolution,
}

/// Gradients of an [`IterativeResidual`], without a dense coupling matrix or its
/// gradient.
///
/// The fields follow the inputs: `cluster` holds the gradients of the arguments of
/// [`IterativeSphereCluster::new`] in their order (`k0`, `radii`, `epsilon`,
/// `positions`), then come those of the incident fields of
/// [`record`](IterativeSphereCluster::record) and the convergence of the adjoint
/// solves.
#[derive(Debug)]
pub struct IterativeGradient {
    /// Gradients of the vacuum wavenumber, radii, permittivities and positions.
    pub cluster: SphereClusterGradient,
    /// Gradient of the incident fields.
    pub incident: DMatrix<Complex>,
    /// Independent convergence certificates of the adjoint solves.
    pub convergence: Vec<Convergence>,
}

impl IterativeResidual {
    /// The forward solution and its convergence certificates; the pullback reads the
    /// solution.
    #[must_use]
    pub const fn solution(&self) -> &IterativeSolution {
        &self.solution
    }

    /// The shape of the solution: local multipoles and incident columns.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.solution.value.shape()
    }

    /// Gradients of the cluster and the incident fields from `cotangent`, the gradient
    /// of a real loss with respect to the solution.
    ///
    /// One adjoint GMRES solve per column gives `Y = (I - T C)⁻ᴴ G`; each particle pair
    /// then contributes to the position and `k0` gradients. Those contributions add in
    /// chunks of particles fixed by the particle count, and the chunk sums in chunk
    /// order; the radius and permittivity gradients add in particle order. The thread
    /// count changes none of the gradients.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<IterativeGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid illumination cotangent".into()));
        }
        let operator = self.operator;
        let adjoint = operator.solve_rhs(cotangent, self.options, true)?;
        let columns = self.incident.ncols();
        let value = &self.solution.value;
        // The local matrices multiply the response B + C X; the coupling sees T^H Y.
        let mut response = self.incident;
        let scattered = operator.coupling(value.as_slice(), columns, false)?;
        for (response, scattered) in response.iter_mut().zip(scattered) {
            *response += scattered;
        }
        let mut incident = DMatrix::zeros(operator.dimension, columns);
        for (mut output, input) in incident.column_iter_mut().zip(adjoint.value.column_iter()) {
            output.copy_from_slice(&operator.local(input.as_slice(), true));
        }
        let (positions, k0) = operator.pair_gradients(&incident, value)?;
        let mut cluster = SphereClusterGradient {
            k0,
            radii: vec![0.0; operator.positions.len()],
            epsilon: vec![Complex::default(); operator.positions.len()],
            positions,
        };
        operator.local_gradients(&adjoint.value, &response, &mut cluster)?;
        Ok(IterativeGradient {
            cluster,
            incident,
            convergence: adjoint.convergence,
        })
    }
}

impl IterativeSphereCluster {
    /// Position and wavenumber gradients through the coupling `C`, whose cotangent is
    /// `(Tᴴ Y) Xᴴ` for `local_adjoint = Tᴴ Y` and the solution `value = X`, summed one
    /// particle-pair block at a time.
    ///
    /// Chunks of consecutive particles `i`, fixed by the particle count, add their
    /// pairs `(i, j)` in order into one partial sum each, and the partial sums add in
    /// chunk order (`numerics::parallel::try_fold_ordered`), so the thread count does
    /// not change the result.
    fn pair_gradients(
        &self,
        local_adjoint: &DMatrix<Complex>,
        value: &DMatrix<Complex>,
    ) -> Result<(Vec<[f64; 3]>, f64)> {
        let modes = self.modes_per_particle();
        let particles = self.positions.len();
        try_fold_ordered(
            (0..particles).collect(),
            true,
            || (vec![[0.0; 3]; particles], 0.0),
            |(mut positions, mut k0): (Vec<[f64; 3]>, f64), _, i: usize| -> Result<_> {
                let mut cotangent = vec![Complex::default(); modes * modes];
                for j in (0..particles).filter(|&j| j != i) {
                    // The rows of particle i against the columns of particle j,
                    // sequentially: the particles already run in parallel.
                    matmul(
                        MatMut::from_column_major_slice_mut(&mut cotangent, modes, modes),
                        Accum::Replace,
                        view(local_adjoint).subrows(i * modes, modes),
                        view(value).subrows(j * modes, modes).adjoint(),
                        Complex::new(1.0, 0.0),
                        Par::Seq,
                    );
                    let displacement = std::array::from_fn(|axis| {
                        self.positions[i][axis] - self.positions[j][axis]
                    });
                    let (position, frequency) = self
                        .plan
                        .pullback(
                            Complex::new(self.k0, 0.0),
                            displacement,
                            Radial::Singular,
                            &cotangent,
                        )
                        .map(|(position, k)| (position, k.re))?;
                    k0 += frequency;
                    for (axis, derivative) in position.into_iter().enumerate() {
                        positions[i][axis] += derivative;
                        positions[j][axis] -= derivative;
                    }
                }
                Ok((positions, k0))
            },
            |(mut positions, k0), (partial, partial_k0)| {
                for (a, b) in positions.iter_mut().flatten().zip(partial.iter().flatten()) {
                    *a += b;
                }
                (positions, k0 + partial_k0)
            },
        )
    }

    /// Radius, permittivity and wavenumber gradients through the Mie coefficients,
    /// whose local cotangent is `Y (B + C X)ᴴ` restricted to each 2 x 2 block.
    ///
    /// [`SphereResidual::pullback`](crate::tmatrix::SphereResidual::pullback) does the
    /// same for the dense path of [`sphere_cluster`](super::sphere_cluster), with two
    /// differences in order. It sums the Mie gradients of all degrees and then applies
    /// the chain rule of the size parameter `k0 r` once; here each degree adds its own
    /// `k0 g` and `r g`. It also swaps each 2 x 2 block to helicity order before
    /// summing, where this sums in mode order and swaps once, which gives the same
    /// values. The first difference changes the last bits, so the two paths agree to
    /// rounding, and each keeps its order so that its results stay reproducible.
    fn local_gradients(
        &self,
        adjoint: &DMatrix<Complex>,
        response: &DMatrix<Complex>,
        cluster: &mut SphereClusterGradient,
    ) -> Result<()> {
        let modes = self.modes_per_particle();
        for (particle, degrees) in self.degrees.iter().enumerate() {
            let mut degree_cotangents = vec![Matrix2::zeros(); degrees.len()];
            for (block, &degree) in self.mode_degrees.iter().enumerate() {
                let offset = particle * modes + block * 2;
                for i in 0..2 {
                    for j in 0..2 {
                        for p in 0..adjoint.ncols() {
                            degree_cotangents[degree][(i, j)] +=
                                adjoint[(offset + i, p)] * response[(offset + j, p)].conj();
                        }
                    }
                }
            }
            for (residual, cotangent) in degrees.iter().zip(degree_cotangents) {
                let gradient = residual.clone().pullback(&to_mode_order(&cotangent))?;
                cluster.radii[particle] += self.k0 * gradient.sizes[0];
                cluster.k0 += self.radii[particle] * gradient.sizes[0];
                cluster.epsilon[particle] += gradient.epsilon[0];
            }
        }
        Ok(())
    }
}

/// Add one `modes x modes` column-major translation block, or its conjugate
/// transpose, applied to `input` into `output`.
fn apply_block(block: &[Complex], input: &[Complex], output: &mut [Complex], adjoint: bool) {
    let modes = output.len();
    if adjoint {
        for (output, column) in output.iter_mut().zip(block.chunks_exact(modes)) {
            for (entry, x) in column.iter().zip(input) {
                *output += entry.conj() * x;
            }
        }
    } else {
        for (column, x) in block.chunks_exact(modes).zip(input) {
            for (output, entry) in output.iter_mut().zip(column) {
                *output += entry * x;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{f64::consts::PI, sync::Arc};

    use nalgebra::DMatrix;
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::IterativeSphereCluster;
    use crate::{
        Complex,
        coeffs::Material,
        linalg::GmresOptions,
        sw,
        test_support::{
            EXPENSIVE_CASES, assert_same_bits_on_pools, bits, dot, patterned, prop_assert_close,
            re_dot,
        },
    };

    fn options() -> GmresOptions {
        GmresOptions {
            rtol: 2e-12,
            restart: 8,
            max_iterations: 100,
            ..GmresOptions::default()
        }
    }

    /// Sphere clusters in vacuum.
    #[derive(Clone, Debug)]
    struct Cluster {
        lmax: u32,
        radii: Vec<f64>,
        epsilon: Vec<Complex>,
        positions: Vec<[f64; 3]>,
    }

    /// One to three lossy or lossless spheres, each placed a random gap beyond touching
    /// its predecessor, along the polar axis or in a random direction.
    fn cluster() -> impl Strategy<Value = Cluster> {
        let direction = prop_oneof![
            Just([0.0, 0.0, 1.0]),
            Just([0.0, 0.0, -1.0]),
            (0.0..PI, -PI..PI).prop_map(|(theta, phi): (f64, f64)| {
                [
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]
            }),
        ];
        (1_usize..=3, 1_u32..=2)
            .prop_flat_map(move |(count, lmax)| {
                (
                    Just(lmax),
                    prop::collection::vec(0.1_f64..0.35, count),
                    prop::collection::vec(
                        (1.2_f64..4.0, prop_oneof![Just(0.0), 0.0_f64..0.05]),
                        count,
                    ),
                    prop::array::uniform3(-1.0_f64..1.0),
                    prop::collection::vec((0.05_f64..1.0, direction.clone()), count - 1),
                )
            })
            .prop_map(|(lmax, radii, epsilon, origin, steps)| {
                let mut positions = vec![origin];
                for (i, (gap, direction)) in steps.into_iter().enumerate() {
                    let distance = radii[i] + radii[i + 1] + gap;
                    let previous = positions[i];
                    positions.push(std::array::from_fn(|a| {
                        previous[a] + distance * direction[a]
                    }));
                }
                Cluster {
                    lmax,
                    radii,
                    epsilon: epsilon
                        .into_iter()
                        .map(|(re, im)| Complex::new(re, im))
                        .collect(),
                    positions,
                }
            })
    }

    impl Cluster {
        fn operator(&self, k0: f64) -> IterativeSphereCluster {
            IterativeSphereCluster::new(self.lmax, k0, &self.radii, &self.epsilon, &self.positions)
                .unwrap()
        }

        /// The signed permutation `S` mapping `(particle, l, m, pol)` to
        /// `(particle, l, -m, pol)` with sign `(-1)^m`.
        fn reciprocity(&self) -> DMatrix<Complex> {
            let modes = sw::modes(self.lmax).unwrap();
            let modes_per_particle = modes.len();
            let dimension = modes_per_particle * self.radii.len();
            let mut s = DMatrix::zeros(dimension, dimension);
            for particle in 0..self.radii.len() {
                let offset = particle * modes_per_particle;
                for (from, mode) in modes.iter().enumerate() {
                    let to = modes
                        .iter()
                        .position(|other| {
                            other.l == mode.l && other.m == -mode.m && other.pol == mode.pol
                        })
                        .unwrap();
                    let sign = if mode.m % 2 == 0 { 1.0 } else { -1.0 };
                    s[(offset + to, offset + from)] = Complex::from(sign);
                }
            }
            s
        }
    }

    proptest! {
        // Five GMRES cluster solves, an adjoint solve and a dense cluster solve per case.
        #![proptest_config(ProptestConfig::with_cases(EXPENSIVE_CASES))]

        #[test]
        fn requested_columns_match_dense_and_obey_symmetries(
            cluster in cluster(),
            seed in -2.0_f64..2.0,
            scale in 0.7_f64..1.4,
            offset in prop::array::uniform3(-1.0_f64..1.0),
        ) {
            check_requested_columns(&cluster, seed, scale, offset)?;
        }
    }

    /// Matrix-free GMRES solves equal the dense cluster T-matrix applied to the
    /// requested columns and certify their residuals; for a single sphere both equal
    /// its Mie T-matrix and its product with the columns. The dense T-matrix obeys
    /// Lorentz reciprocity `T = S Tᵀ S`, and so do the requested columns:
    /// `uᵀ S X_v = vᵀ S X_u`. Solves are invariant under translation, joint scaling
    /// and relabelling of the particles, and the pullback equals the dense one, with
    /// the incident cotangent `Tᴴ G`, the loss pairing, zero total translation
    /// gradient and the Euler identity of joint length and wavenumber scaling.
    fn check_requested_columns(
        cluster: &Cluster,
        seed: f64,
        scale: f64,
        offset: [f64; 3],
    ) -> Result<(), TestCaseError> {
        let Cluster {
            lmax,
            radii,
            epsilon,
            positions,
        } = cluster;
        for i in 0..positions.len() {
            for j in 0..i {
                let d = (0..3).fold(0.0_f64, |d, a| d.hypot(positions[i][a] - positions[j][a]));
                prop_assume!(d > radii[i] + radii[j] + 0.01);
            }
        }
        let k0 = 1.2;
        let operator = cluster.operator(k0);
        let incident = patterned(operator.dimension(), 2, seed);
        let result = operator.solve(&incident, options()).unwrap();
        let dense = crate::cluster::sphere_cluster(*lmax, k0, radii, epsilon, positions).unwrap();
        let t = dense.value().clone();
        let expected = &t * &incident;
        prop_assert_close!(&result.value, &expected, 2e-10 * expected.norm());
        for report in &result.convergence {
            prop_assert!(report.residual_norm <= options().rtol * report.rhs_norm);
        }
        if let [radius] = radii.as_slice() {
            let material = Material {
                epsilon: epsilon[0],
                ..Material::default()
            };
            let (sphere, _) =
                crate::tmatrix::sphere(*lmax, k0, &[*radius], &[material, Material::default()])
                    .unwrap();
            prop_assert_close!(&sphere, &t, 1e-13);
            prop_assert_close!(&result.value, &(&sphere * &incident), 1e-13);
        }

        let s = cluster.reciprocity();
        prop_assert_close!(&(&s * t.transpose() * &s), &t, 1e-12 * t.norm());
        let (u, v) = (incident.column(0), incident.column(1));
        let forward = (u.transpose() * &s * result.value.column(1))[0];
        let backward = (v.transpose() * &s * result.value.column(0))[0];
        prop_assert_close!(forward, backward, 1e-10 * u.norm() * v.norm() * t.norm());

        let translated = Cluster {
            positions: positions
                .iter()
                .map(|p| std::array::from_fn(|a| p[a] + offset[a]))
                .collect(),
            ..cluster.clone()
        };
        let other = translated.operator(k0).solve(&incident, options()).unwrap();
        prop_assert_close!(&other.value, &result.value, 2e-12);
        let scaled = Cluster {
            radii: radii.iter().map(|r| r * scale).collect(),
            positions: positions.iter().map(|p| p.map(|x| x * scale)).collect(),
            ..cluster.clone()
        };
        let other = scaled
            .operator(k0 / scale)
            .solve(&incident, options())
            .unwrap();
        prop_assert_close!(&other.value, &result.value, 2e-11);
        let reversed = Cluster {
            radii: radii.iter().rev().copied().collect(),
            epsilon: epsilon.iter().rev().copied().collect(),
            positions: positions.iter().rev().copied().collect(),
            ..cluster.clone()
        };
        let modes = operator.modes_per_particle();
        let count = radii.len();
        let reverse = |x: &DMatrix<Complex>| {
            DMatrix::from_fn(x.nrows(), x.ncols(), |i, j| {
                x[((count - 1 - i / modes) * modes + i % modes, j)]
            })
        };
        let other = reversed
            .operator(k0)
            .solve(&reverse(&incident), options())
            .unwrap();
        prop_assert_close!(&other.value, &reverse(&result.value), 1e-12);

        let g = patterned(operator.dimension(), 2, seed + 0.3);
        let grad = Arc::new(operator)
            .record(incident.clone(), options())
            .unwrap()
            .pullback(&g)
            .unwrap();
        prop_assert_close!(&grad.incident, &(t.adjoint() * &g), 3e-10 * g.norm());
        let dense_grad = dense.pullback(&(&g * incident.adjoint())).unwrap();
        let cluster_grad = &grad.cluster;
        prop_assert_close!(&cluster_grad.radii, &dense_grad.radii, 3e-10);
        prop_assert_close!(&cluster_grad.epsilon, &dense_grad.epsilon, 3e-10);
        prop_assert_close!(&cluster_grad.positions, &dense_grad.positions, 3e-10);
        prop_assert_close!(cluster_grad.k0, dense_grad.k0, 3e-10);
        for axis in 0..3 {
            let total: f64 = cluster_grad.positions.iter().map(|p| p[axis]).sum();
            prop_assert_close!(total, 0.0, 1e-12, "axis {}", axis);
        }
        let loss = re_dot(&g, &result.value);
        prop_assert_close!(loss, re_dot(&grad.incident, &incident), 3e-10);
        let scale_gradient = dot(radii, &cluster_grad.radii)
            + dot(
                positions.iter().flatten(),
                cluster_grad.positions.iter().flatten(),
            )
            - k0 * cluster_grad.k0;
        prop_assert_close!(scale_gradient, 0.0, 3e-10);
        Ok(())
    }

    #[test]
    fn domain_errors_are_explicit() {
        let eps = [Complex::new(2.0, 0.0)];
        assert!(IterativeSphereCluster::new(1, 0.0, &[0.2], &eps, &[[0.0; 3]]).is_err());
        assert!(IterativeSphereCluster::new(1, 1.0, &[0.2], &eps, &[]).is_err());
        assert!(
            IterativeSphereCluster::new(
                1,
                1.0,
                &[0.2, 0.3],
                &[eps[0]; 2],
                &[[0.0; 3], [0.4, 0.0, 0.0]]
            )
            .is_err()
        );
        let op = IterativeSphereCluster::new(1, 1.0, &[0.2], &eps, &[[0.0; 3]]).unwrap();
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

    /// The pair gradients add in chunks fixed by the particle count: with more
    /// particles than chunks, the solve and every gradient repeat bit for bit on every
    /// pool size.
    #[test]
    fn pullback_does_not_depend_on_the_thread_count() {
        // A 6 x 4 x 3 grid of weakly scattering spheres of slightly different radii.
        let positions: Vec<[f64; 3]> = (0..72_u32)
            .map(|i| [i % 6, i / 6 % 4, i / 24].map(|n| 1.1 * f64::from(n)))
            .collect();
        let radii: Vec<f64> = (0..72_u32)
            .map(|i| 0.2 + 0.05 * f64::from(i).sin())
            .collect();
        let epsilon = vec![Complex::new(2.25, 0.01); positions.len()];
        let cluster =
            Arc::new(IterativeSphereCluster::new(1, 1.0, &radii, &epsilon, &positions).unwrap());
        let incident = patterned(cluster.dimension(), 1, 0.4);
        let g = patterned(cluster.dimension(), 1, 1.1);
        assert_same_bits_on_pools(|| {
            let residual = cluster.record(incident.clone(), options()).unwrap();
            let solution = residual.solution().value.clone();
            let gradient = residual.pullback(&g).unwrap();
            let cluster = &gradient.cluster;
            bits(&[
                &solution,
                &cluster.k0,
                &cluster.radii,
                &cluster.epsilon,
                &cluster.positions,
                &gradient.incident,
            ])
        });
    }
}
