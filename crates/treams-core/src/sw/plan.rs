//! Couplings of complete mode blocks, computed once and shared by every pair of
//! positions, with the radial functions and harmonics evaluated once per displacement.
//!
//! Upstream: `treams.sw.translate`, which evaluates one entry at a time; a plan gives
//! the same coefficients for a whole block.
#![allow(clippy::indexing_slicing)] // Internally constructed dense indices.

use std::collections::{HashMap, HashSet};

use super::{
    CartesianTranslation, Mode,
    cartesian::{combine, harmonic},
    coupling::{Coupling, Kinds},
};
use crate::{
    Complex, Error, Result,
    basis::ModeLabel,
    numerics::parallel::try_fold_ordered,
    special::{
        Radial, RadialJet, SolidTable, Wigner3jRow, direction, spherical_radial,
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
        let radials = self.radials(k * r, radial)?;
        self.solids.visit::<false>(unit, |index, p, value, _| {
            table[index] = radials[degree(p)].value * value;
        });
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

    /// The block of lattice-summed couplings at the displacement `position`
    /// (destination minus source). The lattice sums take `source - destination`.
    pub(crate) fn evaluate_periodic(
        &self,
        k: Complex,
        position: [f64; 3],
        lattice: &crate::lattice::BlochLattice,
        eta: Complex,
    ) -> Result<Vec<Complex>> {
        let modes: Vec<_> = harmonics(self.lmax).collect();
        let table = crate::threads::install(|| {
            modes
                .par_iter()
                .map(|&(l, m)| {
                    Ok(crate::lattice::sum(
                        crate::lattice::Family::Spherical { l, m },
                        k,
                        lattice,
                        position.map(|x| -x),
                        eta,
                    )? / crate::special::harmonic_normalization(l, m))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(self.evaluate_table(&table))
    }

    /// The lattice-sum gradients of the block cotangents of both polarizations.
    ///
    /// The harmonics run in chunks fixed by their count, and their gradients, collected
    /// in harmonic order, add in that order, so the thread count does not change them.
    pub(crate) fn pullback_periodic(
        &self,
        ks: [Complex; 2],
        position: [f64; 3],
        lattice: &crate::lattice::BlochLattice,
        eta: Complex,
        cotangent: &[Vec<Complex>; 2],
    ) -> Result<[crate::lattice::SumGradient; 2]> {
        let modes: Vec<_> = harmonics(self.lmax).collect();
        let [first, second] = cotangent.each_ref().map(|cotangent| {
            let mut table = vec![Complex::default(); modes.len()];
            self.pullback_table(cotangent, &mut table);
            table
        });
        let g: Vec<_> = first.into_iter().zip(second).map(<[_; 2]>::from).collect();
        let gradients = try_fold_ordered(
            modes.into_iter().zip(g).collect(),
            true,
            Vec::new,
            |mut gradients, _, ((l, m), g): ((i32, i32), [Complex; 2])| {
                let mut result = [crate::lattice::SumGradient::default(); 2];
                let mut shared = None;
                for pol in 0..2 {
                    if g[pol] == Complex::default() {
                        continue;
                    }
                    let d = if let Some(d) = shared {
                        d
                    } else {
                        let d = crate::lattice::derivatives(
                            crate::lattice::Family::Spherical { l, m },
                            ks[pol],
                            lattice,
                            position.map(|r| -r),
                            eta,
                        )?;
                        if ks[0] == ks[1] {
                            shared = Some(d);
                        }
                        d
                    };
                    result[pol] = d.pullback(g[pol] / crate::special::harmonic_normalization(l, m));
                    result[pol].shift = result[pol].shift.map(|g| -g);
                }
                gradients.push(result);
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
        special::Radial,
        sw::{self, Mode},
        test_support::{
            DEFAULT_CASES, assert_same_bits_on_pools, bits, patterned, prop_assert_close, radial,
            spherical_basis, table,
        },
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
            let expected: Vec<_> = pairs
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect();
            let actual: Vec<_> = super::harmonics(lmax).collect();
            assert_eq!(actual, expected, "lmax {lmax}");
            for (index, (p, m)) in actual.into_iter().enumerate() {
                assert_eq!(usize::try_from(p * p + p + m).unwrap(), index);
            }
        }
    }

    /// The lattice-sum gradients of the harmonics add in harmonic order: with more
    /// harmonics (81 up to degree 8) than chunks, the pullback of a lattice expansion,
    /// which runs [`TranslationPlan::pullback_periodic`], repeats bit for bit on every
    /// pool size.
    #[test]
    fn periodic_pullback_does_not_depend_on_the_thread_count() {
        let destination = spherical_basis(4, [0.0; 3]);
        let source = spherical_basis(4, [0.2, -0.1, 0.3]);
        let lattice =
            crate::lattice::BlochLattice::new(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[0.3, 0.1])
                .unwrap();
        let ks = [Complex::new(1.1, 0.01); 2];
        let g = patterned(destination.modes.len(), source.modes.len(), 0.6);
        assert_same_bits_on_pools(|| {
            let (_, residual) = sw::lattice_expansion(
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
                &expansion.destination,
                &expansion.source,
                &expansion.ks,
                &g.kpar,
                &g.vectors,
            ])
        });
    }
}
