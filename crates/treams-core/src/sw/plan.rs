//! Couplings of complete mode blocks, computed once and shared by every pair of
//! positions, with the radial functions and harmonics evaluated once per displacement.
//!
//! Upstream: `treams.sw.translate`, which evaluates one entry at a time; a plan gives
//! the same coefficients for a whole block.
#![allow(clippy::indexing_slicing)] // Internally constructed dense indices.

use std::collections::{HashMap, HashSet};

use faer::{MatMut, MatRef};

use super::{
    CartesianTranslation, Mode,
    cartesian::{combine, harmonic},
    coupling::{Coupling, Kinds},
};
use crate::{
    Complex, Error, Result,
    basis::ModeLabel,
    numerics::{finite, parallel::try_fold_ordered},
    special::{
        Radial, RadialJet, SolidTable, Wigner3jRow, direction, spherical_hankels, spherical_radial,
        spherical_radial_sequence, tangent,
    },
};
use rayon::prelude::*;

/// The distinct degree-order pairs `(l, m)` of `modes`, in order of appearance.
fn degree_orders(modes: &[Mode]) -> Vec<(i32, i32)> {
    let mut seen = HashSet::new();
    modes
        .iter()
        .map(|mode| (mode.l, mode.m))
        .filter(|&pair| seen.insert(pair))
        .collect()
}

/// Position of degree `p` in a radial sequence.
fn degree(p: i32) -> usize {
    usize::try_from(p).unwrap_or_default()
}

/// Harmonic `(p, m)` pairs of degrees `p <= lmax` in table order; `(p, m)` sits at
/// `p * p + p + m`.
fn harmonics(lmax: i32) -> impl Iterator<Item = (i32, i32)> {
    (0..=lmax).flat_map(|p| (-p..=p).map(move |m| (p, m)))
}

/// One term of a block entry: `weight` times the harmonic table entry at `index`.
#[derive(Clone, Debug)]
struct Term {
    /// Table index `p * p + p + m` of the harmonic.
    index: usize,
    weight: Complex,
}

/// Geometry-independent spherical mode couplings, shared across position pairs.
#[derive(Clone, Debug)]
pub(crate) struct TranslationPlan {
    /// Terms of all entries, entry `e` in `terms[starts[e]..starts[e + 1]]`.
    terms: Vec<Term>,
    starts: Vec<usize>,
    /// Largest harmonic degree: the largest destination plus the largest source degree.
    lmax: i32,
    solids: SolidTable,
}

impl TranslationPlan {
    /// The couplings from the `source` modes to the `destination` modes, in helicity or
    /// parity polarization.
    pub(crate) fn between(destination: &[Mode], source: &[Mode], helicity: bool) -> Result<Self> {
        for mode in destination.iter().chain(source) {
            mode.validate()?;
        }
        let lmax = destination.iter().map(|m| m.l).max().unwrap_or(0)
            + source.iter().map(|m| m.l).max().unwrap_or(0);
        // Couplings depend on degrees and orders only, and the `(l, lambda, 0, 0)` 3j rows
        // on degrees only: evaluate each once and share them across polarizations.
        let (sources, destinations) = (degree_orders(source), degree_orders(destination));
        let index = |pairs: &[(i32, i32)]| -> HashMap<_, _> {
            pairs
                .iter()
                .enumerate()
                .map(|(i, &pair)| (pair, i))
                .collect()
        };
        let (source_index, destination_index) = (index(&sources), index(&destinations));
        let zeros: HashMap<_, _> = crate::threads::install(|| {
            sources
                .iter()
                .flat_map(|&(l, _)| destinations.iter().map(move |&(lambda, _)| (l, lambda)))
                .collect::<HashSet<_>>()
                .into_par_iter()
                .map(|(l, lambda)| ((l, lambda), Wigner3jRow::new(l, lambda, 0, 0)))
                .collect()
        });
        let couplings: Vec<Vec<_>> = crate::threads::install(|| {
            sources
                .par_iter()
                .map(|&(l, m)| {
                    destinations
                        .iter()
                        .map(|&(lambda, mu)| {
                            Coupling::new((l, m), (lambda, mu), &zeros[&(l, lambda)], Kinds::BOTH)
                        })
                        .collect()
                })
                .collect()
        });
        let mut terms = Vec::new();
        let mut starts = Vec::with_capacity(destination.len() * source.len() + 1);
        starts.push(0);
        for &from in source {
            let couplings = &couplings[source_index[&(from.l, from.m)]];
            for &to in destination {
                for (p, weight) in
                    couplings[destination_index[&(to.l, to.m)]].terms(to.pol, from.pol, helicity)
                {
                    let index = usize::try_from(p * p + p + from.m - to.m)
                        .map_err(|_| Error::InvalidInput("invalid harmonic index".into()))?;
                    terms.push(Term { index, weight });
                }
                starts.push(terms.len());
            }
        }
        Ok(Self {
            terms,
            starts,
            lmax,
            solids: SolidTable::new(lmax),
        })
    }

    /// Divide each weight by the normalization `N_pm` of its harmonic
    /// ([`crate::special::harmonic_normalization`]). The plan then sums its weights
    /// against tables of orthonormal-harmonic values `N_pm z_p P_p^m e^(i m phi)`, the
    /// normalization of the lattice sums of [`crate::lattice::sum`], instead of
    /// `z_p P_p^m e^(i m phi)`.
    pub(crate) fn weights_for_normalized_harmonics(&mut self) {
        let normalization: Vec<_> = harmonics(self.lmax)
            .map(|(l, m)| crate::special::harmonic_normalization(l, m))
            .collect();
        for term in &mut self.terms {
            term.weight /= normalization[term.index];
        }
    }

    /// The terms of each block entry, column-major (source outer, destination inner).
    fn entries(&self) -> impl Iterator<Item = &[Term]> {
        self.starts
            .windows(2)
            .map(|range| &self.terms[range[0]..range[1]])
    }

    /// The block, column-major, from harmonic values `table` at index `p * p + p + m`.
    pub(crate) fn evaluate_table(&self, table: &[Complex]) -> Vec<Complex> {
        self.entries()
            .map(|terms| {
                terms
                    .iter()
                    .map(|term| term.weight * table[term.index])
                    .sum()
            })
            .collect()
    }

    /// Add the block cotangent, weighted by the conjugate term weights, to the harmonic
    /// cotangents `table`.
    pub(crate) fn pullback_table(&self, cotangent: &[Complex], table: &mut [Complex]) {
        for (terms, &g) in self.entries().zip(cotangent) {
            for term in terms {
                table[term.index] += term.weight.conj() * g;
            }
        }
    }

    /// Radial jets `z_p(z)` of degrees `p = 0..=lmax`.
    fn radials(&self, z: Complex, radial: Radial) -> Result<Vec<RadialJet>> {
        let lmax =
            u32::try_from(self.lmax).map_err(|_| Error::InvalidInput("invalid lmax".into()))?;
        spherical_radial_sequence(lmax, z, radial)
    }

    /// Harmonic table `z_p(k r) P_p^m(cos theta) e^(i m phi)` at index `p * p + p + m`.
    /// Outgoing waves vanish at the origin, where only the regular `p = 0` term survives.
    fn values(&self, k: Complex, position: [f64; 3], radial: Radial) -> Result<Vec<Complex>> {
        let mut table = vec![Complex::default(); self.solids.len()];
        let Some((r, unit)) = direction(position) else {
            if radial == Radial::Regular {
                table[0] = spherical_radial(0, k * 0.0, radial)?.value;
            }
            return Ok(table);
        };
        if radial == Radial::Singular {
            // Values need neither the extra radial degree nor the first and
            // second derivatives retained by the pullback's table.
            let lmax =
                u32::try_from(self.lmax).map_err(|_| Error::InvalidInput("invalid lmax".into()))?;
            let radials = spherical_hankels(0, lmax, k * r)?;
            if radials.iter().any(|&value| !finite(value)) {
                return Err(Error::SpecialFunction("radial function overflow".into()));
            }
            self.solids.visit::<false>(unit, |index, p, value, _| {
                table[index] = radials[degree(p)] * value;
            });
        } else {
            let radials = self.radials(k * r, radial)?;
            self.solids.visit::<false>(unit, |index, p, value, _| {
                table[index] = radials[degree(p)].value * value;
            });
        }
        Ok(table)
    }

    /// Harmonic table with the displacement and wavenumber derivatives of each entry.
    fn table(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
    ) -> Result<Vec<CartesianTranslation>> {
        let geometry = direction(position);
        let mut table = vec![CartesianTranslation::default(); self.solids.len()];
        if geometry.is_none() && radial == Radial::Singular {
            return Ok(table);
        }
        let radials = self.radials(k * geometry.map_or(0.0, |(r, _)| r), radial)?;
        let Some((r, unit)) = geometry else {
            for (entry, (p, m)) in table.iter_mut().zip(harmonics(self.lmax)) {
                *entry = harmonic(p, m, k, None, radials[degree(p)]);
            }
            return Ok(table);
        };
        self.solids
            .visit::<true>(unit, |index, p, value, gradient| {
                let tangent = tangent(p, value, gradient, unit);
                table[index] = combine(k, (r, unit), radials[degree(p)], value, tangent);
            });
        Ok(table)
    }

    /// The block of regular or singular translation coefficients at the displacement
    /// `position`, column-major (source outer, destination inner).
    pub(crate) fn evaluate(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
    ) -> Result<Vec<Complex>> {
        Ok(self.evaluate_table(&self.values(k, position, radial)?))
    }

    /// Add the translation applied to `input` to `output`, sharing each coefficient
    /// across the input columns without storing a dense block. With `adjoint`, apply
    /// its conjugate transpose instead. The row counts follow the plan's source and
    /// destination modes, exchanged for the adjoint.
    pub(crate) fn apply(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
        input: MatRef<'_, Complex>,
        output: MatMut<'_, Complex>,
        adjoint: bool,
    ) -> Result<()> {
        let table = self.values(k, position, radial)?;
        self.apply_table(&table, input, output, adjoint);
        Ok(())
    }

    /// Apply the directional derivative of a translation directly to its incident
    /// columns, without allocating a dense particle-pair block.
    pub(crate) fn apply_pushforward(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
        position_tangent: [f64; 3],
        k_tangent: Complex,
        input: MatRef<'_, Complex>,
        output: MatMut<'_, Complex>,
    ) -> Result<()> {
        let table = self.tangent_table(k, position, radial, position_tangent, k_tangent)?;
        self.apply_table(&table, input, output, false);
        Ok(())
    }

    /// A value or tangent harmonic table, shared over all incident columns.
    fn apply_table(
        &self,
        table: &[Complex],
        input: MatRef<'_, Complex>,
        mut output: MatMut<'_, Complex>,
        adjoint: bool,
    ) {
        let (destinations, sources) = if adjoint {
            (input.nrows(), output.nrows())
        } else {
            (output.nrows(), input.nrows())
        };
        for source in 0..sources {
            let starts = &self.starts[source * destinations..][..=destinations];
            for (destination, range) in starts.windows(2).enumerate() {
                let terms = &self.terms[range[0]..range[1]];
                if terms.is_empty() {
                    continue;
                }
                let value: Complex = terms
                    .iter()
                    .map(|term| term.weight * table[term.index])
                    .sum();
                let (to, from, value) = if adjoint {
                    (source, destination, value.conj())
                } else {
                    (destination, source, value)
                };
                for column in 0..input.ncols() {
                    output[(to, column)] += value * input[(from, column)];
                }
            }
        }
    }

    /// Cache-sharing groups in harmonic order. Whole degrees retain the most reuse;
    /// when a block has more workers than degrees, consecutive orders expose enough
    /// independent work to use that budget. Grouping changes no harmonic's arithmetic.
    fn harmonic_groups(&self, workers: usize) -> Vec<(i32, std::ops::RangeInclusive<i32>)> {
        let size = if workers > degree(self.lmax + 1) {
            self.solids.len().div_ceil(workers)
        } else {
            self.solids.len()
        };
        let last = i32::try_from(size - 1).unwrap_or_default();
        (0..=self.lmax)
            .flat_map(|l| {
                (-l..=l)
                    .step_by(size)
                    .map(move |first| (l, first..=(first + last).min(l)))
            })
            .collect()
    }

    /// The block of lattice-summed couplings at the displacement `position`
    /// (destination minus source). The lattice sums take `source - destination`.
    pub(crate) fn evaluate_periodic(
        &self,
        k: Complex,
        position: [f64; 3],
        lattice: &crate::lattice::BlochLattice,
        eta: Complex,
        workers: usize,
    ) -> Result<Vec<Complex>> {
        let degrees = crate::threads::install(|| {
            self.harmonic_groups(workers)
                .into_par_iter()
                .map(|(l, orders)| {
                    crate::lattice::spherical_degree(
                        l,
                        orders,
                        k,
                        lattice,
                        position.map(|x| -x),
                        eta,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let table: Vec<_> = degrees
            .into_iter()
            .flatten()
            .zip(harmonics(self.lmax))
            .map(|(value, (l, m))| value / crate::special::harmonic_normalization(l, m))
            .collect();
        Ok(self.evaluate_table(&table))
    }

    /// Apply the angular plan to directional lattice-sum derivatives. Equal medium
    /// wavenumbers share their Ewald jets even when their input directions differ.
    pub(crate) fn pushforward_periodic(
        &self,
        ks: [Complex; 2],
        position: [f64; 3],
        lattice: &crate::lattice::BlochLattice,
        eta: Complex,
        tangents: &[crate::lattice::SumTangent; 2],
        workers: usize,
    ) -> Result<[Vec<Complex>; 2]> {
        let groups = crate::threads::install(|| {
            self.harmonic_groups(workers)
                .into_par_iter()
                .map(|(l, orders)| {
                    let orders: Vec<_> = orders.collect();
                    let mut values = [Vec::new(), Vec::new()];
                    for pol in 0..2 {
                        if pol == 1 && ks[0] == ks[1] {
                            continue;
                        }
                        let derivatives = crate::lattice::spherical_degree_derivatives(
                            l,
                            &orders,
                            ks[pol],
                            lattice,
                            position.map(|x| -x),
                            eta,
                        )?;
                        for (&m, derivative) in orders.iter().zip(derivatives) {
                            let normalization = crate::special::harmonic_normalization(l, m);
                            for target in pol..if ks[0] == ks[1] { 2 } else { pol + 1 } {
                                values[target].push(
                                    derivative.pushforward(&tangents[target]) / normalization,
                                );
                            }
                        }
                    }
                    Ok(values)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut tables = [Vec::new(), Vec::new()];
        for group in groups {
            for (table, values) in tables.iter_mut().zip(group) {
                table.extend(values);
            }
        }
        Ok(tables.map(|table| self.evaluate_table(&table)))
    }

    /// The lattice-sum gradients of the block cotangents of both polarizations.
    ///
    /// Cache-sharing groups adapt to the worker budget, but their gradients are
    /// collected and added in harmonic order, so the thread count does not change them.
    pub(crate) fn pullback_periodic(
        &self,
        ks: [Complex; 2],
        position: [f64; 3],
        lattice: &crate::lattice::BlochLattice,
        eta: Complex,
        cotangent: &[Vec<Complex>; 2],
        workers: usize,
    ) -> Result<[crate::lattice::SumGradient; 2]> {
        let [first, second] = cotangent.each_ref().map(|cotangent| {
            let mut table = vec![Complex::default(); self.solids.len()];
            self.pullback_table(cotangent, &mut table);
            table
        });
        let g: Vec<_> = first.into_iter().zip(second).map(<[_; 2]>::from).collect();
        let groups = self
            .harmonic_groups(workers)
            .into_iter()
            .map(|(l, orders)| {
                let (first, last) = (*orders.start(), *orders.end());
                let start = degree(l * l + l + first);
                (l, first, &g[start..degree(l * l + l + last + 1)])
            })
            .collect();
        let gradients = try_fold_ordered(
            groups,
            true,
            Vec::new,
            |mut gradients, _, (l, first, g): (i32, i32, &[[Complex; 2]])| {
                let mut result = vec![[crate::lattice::SumGradient::default(); 2]; g.len()];
                for pol in 0..2 {
                    if pol == 1 && ks[0] == ks[1] {
                        continue;
                    }
                    let orders: Vec<_> = (first..)
                        .zip(g)
                        .filter_map(|(m, g)| {
                            (g[pol] != Complex::default()
                                || (ks[0] == ks[1] && g[1] != Complex::default()))
                            .then_some(m)
                        })
                        .collect();
                    let derivatives = crate::lattice::spherical_degree_derivatives(
                        l,
                        &orders,
                        ks[pol],
                        lattice,
                        position.map(|r| -r),
                        eta,
                    )?;
                    for (m, d) in orders.into_iter().zip(derivatives) {
                        let index = degree(m - first);
                        let normalization = crate::special::harmonic_normalization(l, m);
                        for target in pol..if ks[0] == ks[1] { 2 } else { pol + 1 } {
                            if g[index][target] == Complex::default() {
                                continue;
                            }
                            result[index][target] = d.pullback(g[index][target] / normalization);
                            result[index][target].shift = result[index][target].shift.map(|g| -g);
                        }
                    }
                }
                gradients.extend(result);
                Ok(gradients)
            },
            |mut gradients, partial| {
                gradients.extend(partial);
                gradients
            },
        )?;
        Ok(gradients.into_iter().fold(
            [crate::lattice::SumGradient::default(); 2],
            |mut total, gradient| {
                for (total, gradient) in total.iter_mut().zip(gradient) {
                    total.add(gradient);
                }
                total
            },
        ))
    }

    /// Contract the shared harmonic derivatives before assembling the block.
    pub(crate) fn pushforward(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
        position_tangent: [f64; 3],
        k_tangent: Complex,
    ) -> Result<Vec<Complex>> {
        Ok(self.evaluate_table(&self.tangent_table(
            k,
            position,
            radial,
            position_tangent,
            k_tangent,
        )?))
    }

    /// Contract derivatives before expanding harmonic coefficients into mode pairs.
    fn tangent_table(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
        position_tangent: [f64; 3],
        k_tangent: Complex,
    ) -> Result<Vec<Complex>> {
        let table: Vec<_> = self
            .table(k, position, radial)?
            .into_iter()
            .map(|entry| {
                entry.k * k_tangent
                    + entry
                        .position
                        .into_iter()
                        .zip(position_tangent)
                        .map(|(derivative, tangent)| derivative * tangent)
                        .sum::<Complex>()
            })
            .collect();
        Ok(table)
    }

    /// Pull a block cotangent of [`evaluate`](Self::evaluate) back to the real displacement
    /// gradient and the complex wavenumber cotangent.
    pub(crate) fn pullback(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
        cotangent: &[Complex],
    ) -> Result<([f64; 3], Complex)> {
        let table = self.table(k, position, radial)?;
        let mut contracted = vec![Complex::default(); table.len()];
        self.pullback_table(cotangent, &mut contracted);
        let mut gradient = [0.0; 3];
        let mut gk = Complex::default();
        for (g, item) in contracted.iter().zip(table) {
            gk += g * item.k.conj();
            for (value, derivative) in gradient.iter_mut().zip(item.position) {
                *value += (g.conj() * derivative).re;
            }
        }
        Ok((gradient, gk))
    }
}

#[cfg(test)]
mod tests {
    //! The plan against the per-pair `sw::cartesian_translation`, its exact symmetries,
    //! its origin limit and the Lean harmonic table. Physical identities of translations
    //! are in `properties/waves.rs`.

    use std::f64::consts::PI;

    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::TranslationPlan;
    use crate::{
        Complex,
        linalg::{view, view_mut},
        special::Radial,
        sw::{self, Mode},
        test_support::{DEFAULT_CASES, bits, patterned, prop_assert_close, radial, table},
    };

    /// Displacements on both polar half-axes and in general directions.
    fn displacement() -> impl Strategy<Value = [f64; 3]> {
        prop_oneof![
            (0.3_f64..3.0).prop_map(|r| [0.0, 0.0, r]),
            (0.3_f64..3.0).prop_map(|r| [0.0, 0.0, -r]),
            (0.3_f64..3.0, 0.0..PI, -PI..PI).prop_map(|(r, theta, phi)| {
                [
                    r * theta.sin() * phi.cos(),
                    r * theta.sin() * phi.sin(),
                    r * theta.cos(),
                ]
            }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        #[test]
        fn plan_matches_direct_translation_and_its_symmetries(
            lmax in 1_u32..=3,
            helicity in any::<bool>(),
            radial in radial(),
            k in (0.5_f64..2.0, 0.0_f64..0.2).prop_map(|(re, im)| Complex::new(re, im)),
            displacement in displacement(),
            angle in -PI..PI,
            seed in 0.0_f64..10.0,
        ) {
            let plan = Case::new(lmax, helicity, radial, k);
            plan.check(displacement, angle, seed)?;
        }

        #[test]
        fn regular_plan_at_the_origin_is_the_identity(
            lmax in 1_u32..=3,
            helicity in any::<bool>(),
            k in (0.5_f64..2.0, 0.0_f64..0.2).prop_map(|(re, im)| Complex::new(re, im)),
            angle in -PI..PI,
            seed in 0.0_f64..10.0,
        ) {
            let plan = Case::new(lmax, helicity, Radial::Regular, k);
            let block = plan.evaluate([0.0; 3]);
            for (index, value) in block.iter().enumerate() {
                let identity = if index % (plan.modes.len() + 1) == 0 { 1.0 } else { 0.0 };
                prop_assert_close!(*value, Complex::new(identity, 0.0), 1e-13);
            }
            plan.check([0.0; 3], angle, seed)?;
        }
    }

    /// A plan over every mode up to `lmax` in both directions, with its evaluation
    /// parameters.
    struct Case {
        plan: TranslationPlan,
        modes: Vec<Mode>,
        helicity: bool,
        radial: Radial,
        k: Complex,
    }

    impl Case {
        fn new(lmax: u32, helicity: bool, radial: Radial, k: Complex) -> Self {
            let modes = sw::modes(lmax).unwrap();
            let plan = TranslationPlan::between(&modes, &modes, helicity).unwrap();
            Self {
                plan,
                modes,
                helicity,
                radial,
                k,
            }
        }

        fn evaluate(&self, displacement: [f64; 3]) -> Vec<Complex> {
            self.plan
                .evaluate(self.k, displacement, self.radial)
                .unwrap()
        }

        /// Block entry `(destination, source)` of the column-major block.
        fn entry(&self, block: &[Complex], to: Mode, from: Mode) -> Complex {
            let index = |mode| self.modes.iter().position(|&m| m == mode).unwrap();
            block[index(from) * self.modes.len() + index(to)]
        }

        /// Checks a block and its pullback against the per-entry `sw::cartesian_translation`,
        /// which evaluates each coefficient and its derivatives on its own, and the exact
        /// symmetries of translations:
        /// - parity: `A(-d) = (-1)^(l + lambda) P A(d) P`, where `P` swaps helicities, or
        ///   in the parity basis flips the sign of the cross-polarization couplings;
        /// - reciprocity: `A_((lambda, mu, t), (l, m, s))(d) =
        ///   (-1)^(m + mu) A_((l, -m, s), (lambda, -mu, t))(-d)`;
        /// - azimuthal covariance: `A(R_z(angle) d) = e^(i (m - mu) angle) A(d)`;
        /// - the Euler identity of the pullback, `d · g_d = Re(k conj(g_k))`.
        ///
        /// Parity flips exact signs only, so it holds bit for bit.
        fn check(&self, d: [f64; 3], angle: f64, seed: f64) -> Result<(), TestCaseError> {
            let (k, n) = (self.k, self.modes.len());
            let block = self.evaluate(d);
            // Direct actions preserve the dense block's accumulation order for
            // each output, across both polarizations and several incident columns.
            let input = patterned(n, 3, seed + 0.2);
            for adjoint in [false, true] {
                let mut actual = patterned(n, 3, seed + 0.7);
                let mut expected = actual.clone();
                for column in 0..input.ncols() {
                    for source in 0..n {
                        for destination in 0..n {
                            let coefficient = block[source * n + destination];
                            if adjoint {
                                expected[(source, column)] +=
                                    coefficient.conj() * input[(destination, column)];
                            } else {
                                expected[(destination, column)] +=
                                    coefficient * input[(source, column)];
                            }
                        }
                    }
                }
                self.plan
                    .apply(
                        k,
                        d,
                        self.radial,
                        view(&input),
                        view_mut(&mut actual),
                        adjoint,
                    )
                    .unwrap();
                prop_assert_eq!(actual, expected);
            }
            let inverted = self.evaluate(d.map(|x| -x));
            let (sin, cos) = angle.sin_cos();
            let rotated = self.evaluate([d[0] * cos - d[1] * sin, d[0] * sin + d[1] * cos, d[2]]);
            let scale = block.iter().map(|v| v.norm()).fold(1e-300, f64::max);
            let cotangent = patterned(n, n, seed);
            let (mut position, mut wavenumber) = ([0.0; 3], Complex::default());
            let mut derivative_scale = 0.0;
            for (&from, g) in self.modes.iter().zip(cotangent.column_iter()) {
                for (&to, &g) in self.modes.iter().zip(g.iter()) {
                    let value = self.entry(&block, to, from);
                    let direct =
                        sw::cartesian_translation(to, from, k, d, self.helicity, self.radial)
                            .unwrap();
                    prop_assert_close!(
                        value,
                        direct.value,
                        1e-13 * scale,
                        "{:?} <- {:?}",
                        to,
                        from
                    );
                    for (sum, derivative) in position.iter_mut().zip(direct.position) {
                        *sum += (g.conj() * derivative).re;
                    }
                    wavenumber += g * direct.k.conj();
                    derivative_scale += g.norm()
                        * (direct.position.iter().map(|v| v.norm()).sum::<f64>() + direct.k.norm());
                    let sign = if (to.l + from.l) % 2 == 0 { 1.0 } else { -1.0 };
                    let flip = |mode: Mode| Mode {
                        pol: 1 - mode.pol,
                        ..mode
                    };
                    let parity = if self.helicity {
                        sign * self.entry(&block, flip(to), flip(from))
                    } else if to.pol == from.pol {
                        sign * value
                    } else {
                        -sign * value
                    };
                    prop_assert_eq!(self.entry(&inverted, to, from), parity);
                    let reflect = |mode: Mode| Mode { m: -mode.m, ..mode };
                    let sign = if (to.m + from.m) % 2 == 0 { 1.0 } else { -1.0 };
                    let reciprocal = sign * self.entry(&inverted, reflect(from), reflect(to));
                    prop_assert_close!(value, reciprocal, 1e-12 * scale);
                    let phase = Complex::from_polar(1.0, f64::from(from.m - to.m) * angle);
                    prop_assert_close!(
                        self.entry(&rotated, to, from),
                        phase * value,
                        1e-12 * scale
                    );
                }
            }
            let (gradient, gk) = self
                .plan
                .pullback(k, d, self.radial, cotangent.as_slice())
                .unwrap();
            let tolerance = 1e-13 * derivative_scale;
            prop_assert_close!(gradient, position, tolerance);
            prop_assert_close!(gk, wavenumber, tolerance);
            let spatial: f64 = gradient.iter().zip(d).map(|(g, x)| g * x).sum();
            let euler_scale =
                derivative_scale * (1.0 + k.norm() + d.iter().map(|x| x.abs()).sum::<f64>());
            prop_assert_close!(spatial, (k * gk.conj()).re, 1e-13 * euler_scale);
            Ok(())
        }
    }

    #[test]
    fn regular_translation_is_continuous_at_the_origin() {
        // Rounding residues between coincident positions, such as `0.1 + 0.2 - 0.3`, and
        // subnormal displacements give the origin block and its derivatives: they differ
        // from them by `O(|k| r)`, far below rounding. `r^p` underflows from lmax 10 at
        // `r = 5.6e-17` and `1 / r` overflows at subnormal `r`, so the plan must not divide
        // by either (see `special::direction`).
        let modes = sw::modes(10).unwrap();
        let plan = TranslationPlan::between(&modes, &modes, true).unwrap();
        let k = Complex::new(1.3, 0.1);
        let origin = plan.evaluate(k, [0.0; 3], Radial::Regular).unwrap();
        for (index, value) in origin.iter().enumerate() {
            let identity = if index % (modes.len() + 1) == 0 {
                1.0
            } else {
                0.0
            };
            assert!((value - identity).norm() < 1e-13, "entry {index} = {value}");
        }
        let cotangent = patterned(modes.len(), modes.len(), 0.3);
        let pullback = |displacement| {
            plan.pullback(k, displacement, Radial::Regular, cotangent.as_slice())
                .unwrap()
        };
        let (position_limit, k_limit) = pullback([0.0; 3]);
        let pullback_scale = cotangent.iter().map(|g| g.norm()).sum::<f64>() * k.norm();
        let few = sw::modes(2).unwrap();
        let direct = |displacement| {
            few.iter()
                .flat_map(|&from| few.iter().map(move |&to| (to, from)))
                .map(|(to, from)| {
                    sw::cartesian_translation(to, from, k, displacement, true, Radial::Regular)
                        .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let direct_limit = direct([0.0; 3]);
        let displacements = [
            [0.1 + 0.2 - 0.3, 0.0, 0.0],
            [1e-100, -2e-100, 1e-100],
            [0.0, 0.0, 1e-300],
            [1e-310, 0.0, 0.0],
            [0.0, -5e-324, 5e-324],
        ];
        for displacement in displacements {
            let block = plan.evaluate(k, displacement, Radial::Regular).unwrap();
            for (index, (value, limit)) in block.iter().zip(&origin).enumerate() {
                assert!(
                    (value - limit).norm() < 1e-15,
                    "{displacement:?}: entry {index} = {value}"
                );
            }
            let (position, wavenumber) = pullback(displacement);
            for (derivative, limit) in position.into_iter().zip(position_limit) {
                assert!(
                    (derivative - limit).abs() < 1e-14 * pullback_scale,
                    "{displacement:?}: position {derivative} != {limit}"
                );
            }
            assert!(
                (wavenumber - k_limit).norm() < 1e-14 * pullback_scale,
                "{displacement:?}: wavenumber {wavenumber} != {k_limit}"
            );
            for (index, (term, limit)) in direct(displacement).iter().zip(&direct_limit).enumerate()
            {
                let distance = (term.value - limit.value).norm()
                    + (term.k - limit.k).norm()
                    + (0..3)
                        .map(|axis| (term.position[axis] - limit.position[axis]).norm())
                        .sum::<f64>();
                assert!(
                    distance < 1e-15,
                    "{displacement:?}: pair {index}: {term:?} != {limit:?}"
                );
            }
        }
    }

    /// Value-only translations remain finite when unused second radial derivatives
    /// overflow; compare to the independently assembled value-only polar entries.
    #[test]
    fn singular_values_do_not_evaluate_unused_radial_derivatives() {
        let case = Case::new(1, true, Radial::Singular, Complex::from(1.0));
        let r = 1e-64;
        let block = case.evaluate([0.0, 0.0, r]);
        assert!(block.iter().all(|&value| crate::numerics::finite(value)));
        let scale = block.iter().map(|value| value.norm()).fold(0.0, f64::max);
        for &from in &case.modes {
            for &to in &case.modes {
                let polar = sw::PolarTranslation::new(to, from, true, Radial::Singular).unwrap();
                let expected = polar
                    .value([Complex::from(r), Complex::default(), Complex::default()])
                    .unwrap();
                let value = case.entry(&block, to, from);
                assert!((value - expected).norm() < 1e-13 * scale);
            }
        }
        assert!(
            case.plan
                .evaluate(case.k, [0.0, 0.0, 1e-120], Radial::Singular)
                .is_err()
        );
    }

    #[test]
    fn harmonics_match_lean_model() {
        // `just formal` keeps this file equal to `Treams.Harmonics.table`.
        let cases = table::<i32, i32>(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../formal/golden/harmonics.txt"
        )));
        let bounds: Vec<_> = cases.iter().map(|(key, _)| key.clone()).collect();
        assert_eq!(bounds, (0..=8).map(|lmax| vec![lmax]).collect::<Vec<_>>());
        for (key, pairs) in cases {
            let lmax = key[0];
            assert_eq!(pairs.len() % 2, 0, "lmax {lmax}: unpaired field");
            let expected: Vec<(i32, i32)> = pairs
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&pair| pair.into())
                .collect();
            let actual: Vec<_> = super::harmonics(lmax).collect();
            assert_eq!(actual, expected, "lmax {lmax}");
            for (index, (p, m)) in actual.into_iter().enumerate() {
                assert_eq!(usize::try_from(p * p + p + m).unwrap(), index);
            }
        }
    }

    /// A singleton splits its harmonic groups on large pools; several centres expose
    /// independent blocks instead. Both forward and gradient keep identical bits.
    #[test]
    fn periodic_pullback_does_not_depend_on_the_thread_count() {
        for centres in [1, 3] {
            let basis = |offset| {
                let positions: Vec<_> = (0..centres)
                    .map(|i| [0.2 * f64::from(i) + offset, -0.1 * f64::from(i), 0.0])
                    .collect();
                sw::Basis {
                    modes: (0..positions.len())
                        .flat_map(|i| sw::modes(4).unwrap().into_iter().map(move |mode| (i, mode)))
                        .collect(),
                    positions,
                }
            };
            let (destination, source) = (basis(0.0), basis(0.13));
            let lattice =
                crate::lattice::BlochLattice::new(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[0.3, 0.1])
                    .unwrap();
            let ks = [Complex::new(1.1, 0.01), Complex::new(1.3, 0.02)];
            let g = patterned(destination.modes.len(), source.modes.len(), 0.6);
            let mut reference = None;
            for workers in [1, 2, 4, 16, 32] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                let actual = pool.install(|| {
                    let (value, residual) = sw::lattice_expansion(
                        destination.clone(),
                        source.clone(),
                        ks,
                        true,
                        lattice.clone(),
                        Complex::default(),
                    )
                    .unwrap();
                    let g = residual.pullback(&g).unwrap();
                    let expansion = &g.expansion;
                    bits(&[
                        &value,
                        &expansion.destination,
                        &expansion.source,
                        &expansion.ks,
                        &g.kpar,
                        &g.vectors,
                    ])
                });
                if let Some(reference) = &reference {
                    assert_eq!(&actual, reference, "{centres} centres, {workers} threads");
                } else {
                    reference = Some(actual);
                }
            }
        }
    }
}
