//! Cylindrical expansion matrices between bases and over periodic arrays, with analytic
//! pullbacks. Entries that share a `CouplingKey` share one coefficient, evaluated once.
//!
//! Upstream: the `treams.Expand` and `treams.ExpandLattice` operators between cylindrical
//! bases.
#![allow(clippy::indexing_slicing)] // Validated modes, basis indices and matrix shapes.

use super::{Basis, CartesianTranslation, cartesian_translation, transverse_wavenumber};
use crate::{
    Complex, Error, Result,
    basis::validate_wavenumbers,
    numerics::{broadcast, finite, parallel::Parallel},
    special::Radial,
};
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::collections::HashMap;

/// The sorted distinct axial wavenumbers of both bases: the entries of the axial
/// cotangent of `pullback_axial`.
fn axial_groups(destination: &Basis, source: &Basis) -> Vec<f64> {
    let mut groups: Vec<_> = destination
        .modes
        .iter()
        .chain(&source.modes)
        .map(|(_, m)| m.kz)
        .collect();
    groups.sort_by(f64::total_cmp);
    groups.dedup();
    groups
}

/// The key of one distinct coupling coefficient. A coefficient depends on its mode pair
/// only through these fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CouplingKey {
    /// Destination position index.
    destination: usize,
    /// Source position index.
    source: usize,
    /// Bits of the shared axial wavenumber.
    kz: u64,
    /// Index into `ks`: the source polarization, or 0 when both wavenumbers are equal.
    slot: u8,
    /// Order difference `m - mu` of the source and destination modes.
    order: i32,
}

/// The distinct coefficients of a cylindrical coupling matrix and the entries they
/// fill. A coefficient depends on the mode pair only through its key, and entries
/// of different axial wavenumbers or polarizations vanish.
#[derive(Debug)]
struct Couplings {
    /// Each key with its first entry `(row, column)`.
    requests: Vec<(CouplingKey, (usize, usize))>,
    /// The request of each coupled entry in the order of [`for_each_coupled`].
    /// Residuals keep this until their pullback, so it holds one compact index per
    /// entry instead of the entry's position.
    entries: Vec<u32>,
}

/// Call `f(row, column)` for each coupled entry of a destination and a source basis,
/// column by column: equal axial wavenumbers and polarizations.
#[allow(clippy::float_cmp)] // Axial wavenumbers are exact discrete basis labels.
fn for_each_coupled(destination: &Basis, source: &Basis, mut f: impl FnMut(usize, usize)) {
    for (j, &(_, from)) in source.modes.iter().enumerate() {
        for (i, &(_, to)) in destination.modes.iter().enumerate() {
            if to.kz == from.kz && to.pol == from.pol {
                f(i, j);
            }
        }
    }
}

impl Couplings {
    /// Equal wavenumbers share one slot.
    fn new(destination: &Basis, source: &Basis, ks: [Complex; 2]) -> Result<Self> {
        let shared = ks[0] == ks[1];
        let mut indices = HashMap::new();
        let mut requests = Vec::new();
        let mut entries = Vec::new();
        let mut overflow = false;
        for_each_coupled(destination, source, |i, j| {
            let (p, to) = destination.modes[i];
            let (q, from) = source.modes[j];
            let key = CouplingKey {
                destination: p,
                source: q,
                kz: from.kz.to_bits(),
                slot: if shared { 0 } else { from.pol },
                order: from.m - to.m,
            };
            let index = *indices.entry(key).or_insert_with(|| {
                requests.push((key, (i, j)));
                requests.len() - 1
            });
            match u32::try_from(index) {
                Ok(index) => entries.push(index),
                Err(_) => overflow = true,
            }
        });
        if overflow {
            return Err(Error::InvalidInput(
                "too many distinct cylindrical couplings".into(),
            ));
        }
        Ok(Self { requests, entries })
    }

    /// Call `f(row, column, request)` for each coupled entry.
    fn for_each(
        &self,
        destination: &Basis,
        source: &Basis,
        mut f: impl FnMut(usize, usize, usize),
    ) {
        let mut entries = self.entries.iter();
        for_each_coupled(destination, source, |i, j| {
            if let Some(&request) = entries.next() {
                f(i, j, request as usize);
            }
        });
    }

    /// The matrix whose coupled entries take their request's value.
    fn matrix(&self, destination: &Basis, source: &Basis, values: &[Complex]) -> DMatrix<Complex> {
        let mut matrix = DMatrix::zeros(destination.modes.len(), source.modes.len());
        self.for_each(destination, source, |i, j, request| {
            matrix[(i, j)] = values[request];
        });
        matrix
    }

    /// The cotangents of each request, summed over its entries by source polarization.
    fn gather(
        &self,
        destination: &Basis,
        source: &Basis,
        cotangent: &DMatrix<Complex>,
    ) -> Vec<[Complex; 2]> {
        let mut g = vec![[Complex::default(); 2]; self.requests.len()];
        self.for_each(destination, source, |i, j, request| {
            g[request][usize::from(source.modes[j].1.pol)] += cotangent[(i, j)];
        });
        g
    }
}

/// What [`expansion`] saves for its pullback: the bases, the wavenumbers, the radial
/// kind and the distinct couplings.
///
/// Equal axial wavenumbers select the coupled entries, and the pullback holds that
/// selection fixed.
///
/// The pullback evaluates the coefficients in parallel and adds their gradients in
/// request order, so its result does not depend on the thread count.
#[derive(Debug)]
pub struct ExpansionResidual {
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    radial: Radial,
    couplings: Couplings,
}

/// Expansion coefficients and their jets evaluate in parallel from this many
/// distinct couplings.
const PARALLEL: Parallel = Parallel::AtLeast(64);

/// Construct a cylindrical expansion matrix in the supplied mode order.
///
/// Upstream: the `treams.Expand` operator (`treams.operators.expand`) between cylindrical
/// bases.
pub fn expansion(
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    radial: Radial,
) -> Result<(DMatrix<Complex>, ExpansionResidual)> {
    destination.validate()?;
    source.validate()?;
    // The expansion takes no polarization convention and never couples the two
    // polarizations, so it skips the parity check.
    validate_wavenumbers(ks, true, false)?;
    // Like cartesian_translation, reject a wavenumber whose square underflows, also
    // when no entry couples.
    if ks.iter().any(|k| k.norm_sqr() == 0.0) {
        return Err(Error::InvalidInput(
            "medium wavenumber squared underflows to zero".into(),
        ));
    }
    let couplings = Couplings::new(&destination, &source, ks)?;
    let residual = ExpansionResidual {
        destination,
        source,
        ks,
        radial,
        couplings,
    };
    let requests = &residual.couplings.requests;
    let values = broadcast::map(requests.len(), PARALLEL, |r| {
        Ok(residual.translation(requests[r].1)?.value)
    })?;
    let value = residual
        .couplings
        .matrix(&residual.destination, &residual.source, &values);
    Ok((value, residual))
}

impl ExpansionResidual {
    /// Destination and source mode counts: the shape of the expansion matrix.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.destination.modes.len(), self.source.modes.len())
    }

    /// Position and complex wavenumber gradients. Axial mode labels stay fixed.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<crate::basis::ExpansionGradient> {
        self.pullback_impl::<false>(cotangent).map(|(g, _)| g)
    }

    /// Also differentiate each shared axial wavenumber. The final array follows
    /// the sorted distinct kz values from both bases; equality partitions stay fixed.
    pub fn pullback_axial(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<(crate::basis::ExpansionGradient, Vec<f64>)> {
        self.pullback_impl::<true>(cotangent)
    }

    /// The translation coefficient and derivatives of request `(i, j)`.
    fn translation(&self, (i, j): (usize, usize)) -> Result<CartesianTranslation> {
        let (p, to) = self.destination.modes[i];
        let (q, from) = self.source.modes[j];
        let position =
            std::array::from_fn(|a| self.destination.positions[p][a] - self.source.positions[q][a]);
        cartesian_translation(
            to,
            from,
            self.ks[usize::from(from.pol)],
            position,
            self.radial,
        )
    }

    fn pullback_impl<const AXIAL: bool>(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<(crate::basis::ExpansionGradient, Vec<f64>)> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "invalid cylindrical expansion cotangent".into(),
            ));
        }
        let requests = &self.couplings.requests;
        let g = self
            .couplings
            .gather(&self.destination, &self.source, cotangent);
        let jets = broadcast::map(requests.len(), PARALLEL, |r| {
            if g[r] == [Complex::default(); 2] {
                Ok(None)
            } else {
                self.translation(requests[r].1).map(Some)
            }
        })?;
        let groups = if AXIAL {
            axial_groups(&self.destination, &self.source)
        } else {
            Vec::new()
        };
        let mut axial = vec![0.0; groups.len()];
        let mut result = crate::basis::ExpansionGradient::zeros(
            self.destination.positions.len(),
            self.source.positions.len(),
        );
        for ((&(key, _), g), jet) in self.couplings.requests.iter().zip(g).zip(jets) {
            let Some(jet) = jet else { continue };
            for (out, g) in result.ks.iter_mut().zip(g) {
                *out += jet.k.conj() * g;
            }
            let total = g[0] + g[1];
            if AXIAL {
                let kz = f64::from_bits(key.kz);
                axial[groups.partition_point(|&k| k < kz)] += (total.conj() * jet.kz).re;
            }
            for axis in 0..3 {
                let derivative = (total.conj() * jet.position[axis]).re;
                result.destination[key.destination][axis] += derivative;
                result.source[key.source][axis] -= derivative;
            }
        }
        Ok((result, axial))
    }
}

/// Periodic cylindrical coupling with exact axial/polarization selection and cached orders.
///
/// Upstream: `treams.cw.translate_periodic` and the `treams.ExpandLattice` operator
/// (`treams.operators.expandlattice`) between cylindrical bases.
pub fn lattice_expansion(
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    lattice: crate::lattice::BlochLattice,
    eta: Complex,
) -> Result<(DMatrix<Complex>, LatticeExpansionResidual)> {
    destination.validate()?;
    source.validate()?;
    // The expansion takes no polarization convention and never couples the two
    // polarizations, so it skips the parity check.
    validate_wavenumbers(ks, true, false)?;
    let couplings = Couplings::new(&destination, &source, ks)?;
    let residual = LatticeExpansionResidual {
        destination,
        source,
        ks,
        lattice,
        eta,
        couplings,
    };
    let values = residual
        .couplings
        .requests
        .par_iter()
        .map(|&(key, _)| {
            let (_, krho, r, phase) = residual.geometry(key);
            Ok(crate::lattice::sum(
                crate::lattice::Family::Cylindrical { m: key.order },
                krho,
                &residual.lattice,
                [r[0], r[1], 0.0],
                residual.eta,
            )? * phase)
        })
        .collect::<Result<Vec<_>>>()?;
    let value = residual
        .couplings
        .matrix(&residual.destination, &residual.source, &values);
    Ok((value, residual))
}

/// What [`lattice_expansion`] saves for its pullback: the bases, the wavenumbers, the
/// lattice, the split and the distinct couplings.
///
/// Equal axial wavenumbers select the coupled entries, and the pullback holds that
/// selection fixed.
///
/// The pullback evaluates the lattice-sum gradients in parallel and adds them in request
/// order, so its result does not depend on the thread count.
#[derive(Debug)]
pub struct LatticeExpansionResidual {
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    lattice: crate::lattice::BlochLattice,
    eta: Complex,
    couplings: Couplings,
}

impl LatticeExpansionResidual {
    /// Destination and source mode counts: the shape of the coupling matrix.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.destination.modes.len(), self.source.modes.len())
    }

    /// The wavenumber, the transverse wavenumber, the source-to-destination
    /// displacement `r` and the axial phase `exp(-i kz r_z)` of a request.
    fn geometry(&self, key: CouplingKey) -> (Complex, Complex, [f64; 3], Complex) {
        let kz = f64::from_bits(key.kz);
        let k = self.ks[usize::from(key.slot)];
        let r: [f64; 3] = std::array::from_fn(|a| {
            self.source.positions[key.source][a] - self.destination.positions[key.destination][a]
        });
        let phase = (-Complex::i() * kz * r[2]).exp();
        (k, transverse_wavenumber(k, kz), r, phase)
    }

    /// Consume the residual and differentiate positions, medium wavenumbers and lattice geometry.
    pub fn pullback(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<crate::basis::LatticeExpansionGradient> {
        self.pullback_impl::<false>(cotangent).map(|(g, _)| g)
    }

    /// Also differentiate sorted distinct axial wavenumbers with matching groups fixed.
    pub fn pullback_axial(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<(crate::basis::LatticeExpansionGradient, Vec<f64>)> {
        self.pullback_impl::<true>(cotangent)
    }

    fn pullback_impl<const AXIAL: bool>(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<(crate::basis::LatticeExpansionGradient, Vec<f64>)> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "invalid periodic expansion cotangent".into(),
            ));
        }
        let g = self
            .couplings
            .gather(&self.destination, &self.source, cotangent);
        let gradients = self
            .couplings
            .requests
            .par_iter()
            .zip(g)
            .map(|(&(key, _), g)| {
                let kz = f64::from_bits(key.kz);
                let (k, krho, r, phase) = self.geometry(key);
                let jet = crate::lattice::derivatives(
                    crate::lattice::Family::Cylindrical { m: key.order },
                    krho,
                    &self.lattice,
                    [r[0], r[1], 0.0],
                    self.eta,
                )?;
                // Equal wavenumbers share the Ewald jet, but keep independent k cotangents.
                let total: Complex = g.iter().sum();
                let scalar_g = total * phase.conj();
                let spectral = g.map(|g| (jet.k * k / krho).conj() * g * phase.conj());
                let mut gradient = crate::lattice::SumGradient {
                    k: Complex::default(),
                    eta: Complex::default(),
                    shift: jet.shift.map(|d| (scalar_g.conj() * d).re),
                    kpar: jet.kpar.map(|d| (scalar_g.conj() * d).re),
                    vectors: jet.vectors.map(|row| row.map(|d| (scalar_g.conj() * d).re)),
                };
                gradient.shift[2] = (total.conj() * (-Complex::i() * kz) * phase * jet.value).re;
                let axial = if AXIAL {
                    (total.conj() * phase * (-jet.k * kz / krho - Complex::i() * r[2] * jet.value))
                        .re
                } else {
                    0.0
                };
                Ok((spectral, gradient, axial))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut result = crate::basis::LatticeExpansionGradient::zeros(
            self.destination.positions.len(),
            self.source.positions.len(),
            self.lattice.dimension(),
        );
        let groups = if AXIAL {
            axial_groups(&self.destination, &self.source)
        } else {
            Vec::new()
        };
        let mut axial = vec![0.0; groups.len()];
        for ((spectral, g, gkz), &(key, _)) in gradients.into_iter().zip(&self.couplings.requests) {
            if AXIAL {
                axial[groups.partition_point(|&k| k < f64::from_bits(key.kz))] += gkz;
            }
            result.add_lattice(&g);
            for (g, k) in result.expansion.ks.iter_mut().zip(spectral) {
                *g += k;
            }
            for (axis, value) in g.shift.into_iter().enumerate() {
                result.expansion.destination[key.destination][axis] -= value;
                result.expansion.source[key.source][axis] += value;
            }
        }
        Ok((result, axial))
    }
}

#[cfg(test)]
mod tests {
    //! Expansion values and pullbacks against pairwise translations. Identities of
    //! expansions are in `properties/waves.rs`.

    use super::{Basis, Complex, Radial, cartesian_translation, expansion};
    use crate::{
        cw::Mode,
        test_support::{DEFAULT_CASES, patterned, prop_assert_close, radial, re_dot},
    };
    use nalgebra::DMatrix;
    use proptest::{prelude::*, test_runner::TestCaseError};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        #[test]
        fn expansion_pullback_matches_pairwise_translations(
            mmax in 0_i32..3,
            kzs in (-0.7_f64..0.35, 0.35_f64..0.7),
            positions in prop::array::uniform3(prop::array::uniform3(-1.0_f64..1.0)),
            shared in any::<bool>(),
            radial in radial(),
            seed in -3.0_f64..3.0,
        ) {
            check_expansion_pullback(mmax, [kzs.0, kzs.1], positions, shared, radial, seed)?;
        }
    }

    /// Expansion values are the pairwise translation coefficients, and both pullbacks
    /// are the pairwise sums `Σ_ij Re(conj(g_ij) ∂T_ij)` into positions, wavenumbers and
    /// the sorted axial groups `kzs`, including zero-displacement outgoing self blocks.
    #[allow(clippy::float_cmp)] // Axial wavenumbers are exact mode labels.
    fn check_expansion_pullback(
        mmax: i32,
        kzs: [f64; 2],
        [first, second, third]: [[f64; 3]; 3],
        shared: bool,
        radial: Radial,
        seed: f64,
    ) -> Result<(), TestCaseError> {
        let modes = |count: usize| -> Vec<(usize, Mode)> {
            (0..count)
                .flat_map(|p| {
                    kzs.into_iter().flat_map(move |kz| {
                        (-mmax..=mmax).flat_map(move |m| [1, 0].map(|pol| (p, Mode { kz, m, pol })))
                    })
                })
                .collect()
        };
        // The first positions coincide: outgoing translations between them vanish.
        let destination = Basis {
            modes: modes(2),
            positions: vec![first, second],
        };
        let source = Basis {
            modes: modes(2),
            positions: vec![first, third],
        };
        let ks = if shared {
            [Complex::new(1.3, 0.05); 2]
        } else {
            [Complex::new(1.3, 0.05), Complex::new(1.5, 0.07)]
        };
        let g = patterned(destination.modes.len(), source.modes.len(), seed);
        let mut expected = DMatrix::zeros(g.nrows(), g.ncols());
        let mut gradient = (
            vec![[0.0; 3]; 2],
            vec![[0.0; 3]; 2],
            [Complex::default(); 2],
        );
        let mut axial = [0.0; 2];
        for (j, &(q, from)) in source.modes.iter().enumerate() {
            for (i, &(p, to)) in destination.modes.iter().enumerate() {
                let position =
                    std::array::from_fn(|a| destination.positions[p][a] - source.positions[q][a]);
                let pol = usize::from(from.pol);
                let jet = cartesian_translation(to, from, ks[pol], position, radial).unwrap();
                expected[(i, j)] = jet.value;
                gradient.2[pol] += jet.k.conj() * g[(i, j)];
                axial[usize::from(from.kz == kzs[1])] += re_dot([g[(i, j)]], [jet.kz]);
                for axis in 0..3 {
                    let derivative = re_dot([g[(i, j)]], [jet.position[axis]]);
                    gradient.0[p][axis] += derivative;
                    gradient.1[q][axis] -= derivative;
                }
            }
        }
        let scale = 1e-12 * (1.0 + expected.norm());
        let (value, residual) = expansion(destination.clone(), source.clone(), ks, radial).unwrap();
        prop_assert_close!(&value, &expected, 1e-14 * expected.norm());
        let actual = residual.pullback(&g).unwrap();
        let (with_axial, actual_axial) = expansion(destination, source, ks, radial)
            .unwrap()
            .1
            .pullback_axial(&g)
            .unwrap();
        for actual in [actual, with_axial] {
            prop_assert_close!(&actual.destination, &gradient.0, scale);
            prop_assert_close!(&actual.source, &gradient.1, scale);
            prop_assert_close!(actual.ks, gradient.2, scale);
        }
        prop_assert_close!(actual_axial, axial.to_vec(), scale);
        Ok(())
    }
}
