//! Regular spherical and cylindrical expansions of plane waves, with analytic pullbacks.
//!
//! Upstream: `treams.pw.to_sw`, `treams.pw.to_cw`, and the `Expand` operator from a
//! plane-wave basis to a spherical or cylindrical basis.
#![allow(clippy::indexing_slicing)] // Validated bases, label slots and matrix shapes.

use nalgebra::DMatrix;

use super::{field::phase, polarization::Direction};
use crate::{
    Complex, Error, Result,
    basis::{ModeLabel, MultipoleBasis},
    numerics::{
        self, Jet, finite,
        parallel::{try_fill_chunks, try_fold_ordered},
    },
    special::{check_pol, polarized_angular},
    sw::Mode,
};

/// Normalization and phase `i^l` of the spherical expansion of a plane wave, which
/// depend on the degree and order alone.
fn spherical_prefactor(mode: Mode) -> Complex {
    let (l, m) = (mode.l, mode.m);
    let normalization = 2.0
        * (std::f64::consts::PI * f64::from(2 * l + 1) / f64::from(l * (l + 1))).sqrt()
        * crate::special::factorial_ratio_sqrt(l, m);
    normalization * Complex::i().powi(l)
}

/// The coefficient of `mode` in the regular expansion of a unit plane wave of
/// polarization `pol`, without the position phase: the prefactor times
/// `exp(-i m phi)` times the polarized angular function of `theta`.
fn spherical_coefficient<const N: usize>(
    mode: Mode,
    prefactor: Complex,
    vector: [Complex; 3],
    direction: &Direction<N>,
    pol: u8,
    helicity: bool,
) -> Jet<N> {
    if helicity && mode.pol != pol {
        return Jet::default();
    }
    let m = mode.m;
    let axis = direction.transverse.value == Complex::default();
    let azimuth = if axis {
        Jet::constant(1.0)
    } else {
        (direction.xy[0] - Complex::i() * direction.xy[1]).powi(m)
    };
    let cosine = if axis {
        Jet::constant(if (vector[2] / direction.k.value).re >= 0.0 {
            1.0
        } else {
            -1.0
        })
    } else {
        Jet::variable(vector[2], 2) / direction.k
    };
    // Keep the same transverse branch as the Cartesian polarization. Taking a
    // second principal square root of 1-cos(theta)^2 can flip complex directions.
    let sine = direction.transverse / direction.k;
    let angular = polarized_angular(mode.l, m, [cosine, sine], [mode.pol, pol], helicity);
    prefactor * azimuth * angular
}

/// Spherical expansion coefficient for a unit-amplitude plane wave.
///
/// Upstream: `treams.pw.to_sw`.
pub fn to_sw(mode: Mode, vector: [Complex; 3], pol: u8, helicity: bool) -> Result<Complex> {
    mode.validate()?;
    check_pol(pol)?;
    let direction = Direction::<0>::new(vector)?;
    let prefactor = spherical_prefactor(mode);
    Ok(spherical_coefficient(mode, prefactor, vector, &direction, pol, helicity).value)
}

/// Regular spherical multipole amplitudes of one plane wave: the single-plane-wave
/// reference of the plane-wave property tests.
#[cfg(test)]
pub(crate) fn reference_spherical_expansion(
    basis: &crate::sw::Basis,
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> Result<Vec<Complex>> {
    basis.validate()?;
    check_pol(pol)?;
    let direction = Direction::<0>::new(vector)?;
    let labels = AngularLabels::spherical(&basis.modes);
    let angular = labels.evaluate_spherical(&basis.modes, vector, &direction, pol, helicity);
    let phases: Vec<_> = basis.positions.iter().map(|&r| phase(vector, r)).collect();
    Ok(basis
        .modes
        .iter()
        .zip(&labels.slot)
        .map(|(&(pidx, _), &slot)| phases[pidx] * angular[slot].value)
        .collect())
}

/// Whether a plane wave of axial wavenumber `kz` and polarization `pol` couples to the
/// cylindrical `mode`: equal `pol`, a real `kz`, and `kz` within 16 ulps of the mode's.
///
/// The tolerance lets a plane basis whose `kz` values come from arithmetic, such as a
/// normalization, find its cylindrical modes. It is relative, so a change of length
/// units cannot merge two labels.
pub(crate) fn cylindrical_mode_matches(mode: crate::cw::Mode, kz: Complex, pol: u8) -> bool {
    mode.pol == pol
        && kz.im == 0.0
        && (mode.kz - kz.re).abs() <= 16.0 * f64::EPSILON * mode.kz.abs().max(kz.re.abs())
}

/// The angular factor `((i kx + ky) / sqrt(kx^2 + ky^2))^m = (i exp(-i phi))^m` of the
/// cylindrical expansion of a plane wave, `i^m` on the z axis.
fn cylindrical_angular<const N: usize>(m: i32, direction: &Direction<N>) -> Jet<N> {
    if direction.transverse.value == Complex::default() {
        Jet::constant(Complex::i().powi(m))
    } else {
        (Complex::i() * direction.xy[0] + direction.xy[1]).powi(m)
    }
}

/// The coefficient of `mode` in the regular expansion of a unit plane wave of
/// polarization `pol`, without the position phase; zero unless the labels match.
fn cylindrical_coefficient<const N: usize>(
    mode: crate::cw::Mode,
    vector: [Complex; 3],
    direction: &Direction<N>,
    pol: u8,
) -> Jet<N> {
    if cylindrical_mode_matches(mode, vector[2], pol) {
        cylindrical_angular(mode.m, direction)
    } else {
        Jet::default()
    }
}

/// One plane-to-cylindrical coefficient with exact axial-label matching.
///
/// A `kz` that differs from the mode's `kz` in any bit gives zero, as in treams.
/// [`expansion`] matches the axial labels of a basis within 16 ulps instead (units in
/// the last place, a relative difference of about 3.6e-15).
///
/// Upstream: `treams.pw.to_cw`.
#[allow(clippy::float_cmp)] // A direct coefficient preserves exact discrete labels.
pub fn to_cw(mode: crate::cw::Mode, vector: [Complex; 3], pol: u8) -> Result<Complex> {
    mode.validate()?;
    if pol > 1 || vector.iter().any(|&v| !finite(v)) || vector[2].im != 0.0 {
        return Err(Error::InvalidInput(
            "finite wavevector, real axial component and polarization 0/1 required".into(),
        ));
    }
    if mode.pol != pol || mode.kz != vector[2].re {
        return Ok(Complex::default());
    }
    Ok(cylindrical_angular(mode.m, &Direction::<0>::new(vector)?).value)
}

/// Regular cylindrical multipole amplitudes of one plane wave: the single-plane-wave
/// reference of the plane-wave property tests.
#[cfg(test)]
pub(crate) fn reference_cylindrical_expansion(
    basis: &crate::cw::Basis,
    vector: [Complex; 3],
    pol: u8,
) -> Result<Vec<Complex>> {
    basis.validate()?;
    let direction = Direction::<0>::new(vector)?;
    if pol > 1 || vector[2].im != 0.0 {
        return Err(Error::InvalidInput(
            "cylindrical expansion requires real axial wavenumber and polarization 0 or 1".into(),
        ));
    }
    Ok(basis
        .modes
        .iter()
        .map(|&(pidx, mode)| {
            phase(vector, basis.positions[pidx])
                * cylindrical_coefficient(mode, vector, &direction, pol).value
        })
        .collect())
}

/// Angular plane-wave coefficients depend on a mode's label, not on its position:
/// modes that repeat at several positions share one evaluation.
struct AngularLabels {
    /// Label index of every basis mode.
    slot: Vec<usize>,
    /// The first basis mode of every label, with its spherical prefactor.
    first: Vec<(usize, Complex)>,
}

impl AngularLabels {
    fn new<K: std::hash::Hash + Eq>(
        len: usize,
        key: impl Fn(usize) -> K,
        prefactor: impl Fn(usize) -> Complex,
    ) -> Self {
        let mut index = std::collections::HashMap::new();
        let mut first = Vec::new();
        let slot = (0..len)
            .map(|i| {
                *index.entry(key(i)).or_insert_with(|| {
                    first.push((i, prefactor(i)));
                    first.len() - 1
                })
            })
            .collect();
        Self { slot, first }
    }

    fn spherical(modes: &[(usize, Mode)]) -> Self {
        Self::new(
            modes.len(),
            |i| modes[i].1,
            |i| spherical_prefactor(modes[i].1),
        )
    }

    fn cylindrical(modes: &[(usize, crate::cw::Mode)]) -> Self {
        Self::new(
            modes.len(),
            |i| (modes[i].1.kz.to_bits(), modes[i].1.m, modes[i].1.pol),
            |_| Complex::default(),
        )
    }

    fn of(basis: &MultipoleBasis) -> Self {
        match basis {
            MultipoleBasis::Spherical(b) => Self::spherical(&b.modes),
            MultipoleBasis::Cylindrical(b) => Self::cylindrical(&b.modes),
        }
    }

    /// One spherical coefficient per label.
    fn evaluate_spherical<const N: usize>(
        &self,
        modes: &[(usize, Mode)],
        vector: [Complex; 3],
        direction: &Direction<N>,
        pol: u8,
        helicity: bool,
    ) -> Vec<Jet<N>> {
        self.first
            .iter()
            .map(|&(i, prefactor)| {
                spherical_coefficient(modes[i].1, prefactor, vector, direction, pol, helicity)
            })
            .collect()
    }

    /// One angular coefficient per label of `basis`.
    fn evaluate<const N: usize>(
        &self,
        basis: &MultipoleBasis,
        vector: [Complex; 3],
        direction: &Direction<N>,
        pol: u8,
        helicity: bool,
    ) -> Vec<Jet<N>> {
        match basis {
            MultipoleBasis::Spherical(b) => {
                self.evaluate_spherical(&b.modes, vector, direction, pol, helicity)
            }
            MultipoleBasis::Cylindrical(b) => self
                .first
                .iter()
                .map(|&(i, _)| cylindrical_coefficient(b.modes[i].1, vector, direction, pol))
                .collect(),
        }
    }
}

/// What [`expansion`] saves for its pullback: its inputs. The pullback recomputes the
/// coefficients instead of keeping a Jacobian.
///
/// The pullback adds the per-mode position gradients in chunks of consecutive plane
/// modes fixed by the mode count, and the chunk sums in chunk order
/// (`numerics::parallel::try_fold_ordered`), so the thread count does not change them.
/// Each wavevector gradient belongs to one plane mode, which writes it in place.
#[derive(Debug)]
pub struct ExpansionResidual {
    pub(super) basis: MultipoleBasis,
    pub(super) vectors: Vec<[Complex; 3]>,
    pub(super) polarizations: Vec<u8>,
    pub(super) helicity: bool,
    pub(super) fixed_vectors: bool,
}
/// Plane-expansion cotangents.
#[derive(Debug)]
pub struct ExpansionGradient {
    /// Multipole position cotangents.
    pub positions: Vec<[f64; 3]>,
    /// Full complex plane-wavevector cotangents; zero when the forward holds the
    /// wavevectors fixed.
    pub vectors: Vec<[Complex; 3]>,
}
/// Expand multiple plane waves in a regular multipole basis: column `j` holds the
/// amplitudes of plane wave `j` at the basis positions.
///
/// Spherical modes use [`to_sw`]; cylindrical modes use the coefficient of [`to_cw`]
/// but match axial labels within 16 ulps (a relative difference of about 3.6e-15).
/// With `fixed_vectors`, the pullback holds the wavevectors fixed and returns zero
/// wavevector gradients.
///
/// Upstream: the `Expand` operator from a plane-wave basis.
pub fn expansion(
    basis: impl Into<MultipoleBasis>,
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    helicity: bool,
    fixed_vectors: bool,
) -> Result<(DMatrix<Complex>, ExpansionResidual)> {
    let basis = basis.into();
    basis.validate()?;
    if vectors.is_empty()
        || vectors.len() != polarizations.len()
        || polarizations.iter().any(|&p| p > 1)
        || (matches!(basis, MultipoleBasis::Cylindrical(_))
            && vectors.iter().any(|k| k[2].im != 0.0))
    {
        return Err(Error::InvalidInput(
            "require nonempty plane vectors and matching polarizations 0/1; cylindrical axial labels must be real".into(),
        ));
    }
    let labels = AngularLabels::of(&basis);
    let mut value = numerics::zeros(basis.len(), vectors.len())?;
    try_fill_chunks(
        value.as_mut_slice(),
        basis.len(),
        vectors.len() > 1,
        |j, column| -> Result<()> {
            let direction = Direction::<0>::new(vectors[j])?;
            let angular =
                labels.evaluate(&basis, vectors[j], &direction, polarizations[j], helicity);
            let phases: Vec<_> = basis
                .positions()
                .iter()
                .map(|&p| phase(vectors[j], p))
                .collect();
            for (i, (out, &slot)) in column.iter_mut().zip(&labels.slot).enumerate() {
                *out = angular[slot].value * phases[basis.position_pol(i).0];
            }
            Ok(())
        },
    )?;
    if value.iter().any(|&v| !finite(v)) {
        return Err(Error::NonFinite("plane expansion overflow".into()));
    }
    Ok((
        value,
        ExpansionResidual {
            basis,
            vectors,
            polarizations,
            helicity,
            fixed_vectors,
        },
    ))
}
impl ExpansionResidual {
    /// Multipole output and plane input mode counts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.basis.len(), self.vectors.len())
    }

    /// Validate position and wavevector tangents for the recorded operation.
    pub fn validate_tangents(
        &self,
        positions: &[[f64; 3]],
        vectors: &[[Complex; 3]],
    ) -> Result<()> {
        if positions.len() != self.basis.positions().len()
            || vectors.len() != self.vectors.len()
            || positions.iter().flatten().any(|r| !r.is_finite())
            || vectors.iter().flatten().any(|&k| !finite(k))
        {
            return Err(Error::InvalidInput(
                "invalid plane-expansion tangents".into(),
            ));
        }
        Ok(())
    }

    /// Push a direction through the position phases and angular coefficients.
    /// Cylindrical axial components remain fixed mode labels, as in the pullback.
    pub fn pushforward(
        &self,
        positions: &[[f64; 3]],
        vectors: &[[Complex; 3]],
    ) -> Result<DMatrix<Complex>> {
        self.validate_tangents(positions, vectors)?;
        let labels = AngularLabels::of(&self.basis);
        let axes = if matches!(self.basis, MultipoleBasis::Spherical(_)) {
            3
        } else {
            2
        };
        let mut tangent = numerics::zeros(self.basis.len(), self.vectors.len())?;
        try_fill_chunks(
            tangent.as_mut_slice(),
            self.basis.len(),
            self.vectors.len() > 1,
            |j, column| -> Result<()> {
                if self.fixed_vectors || vectors[j][..axes].iter().all(|&v| v == Complex::default())
                {
                    self.push_column(
                        &labels,
                        j,
                        &Direction::<0>::new(self.vectors[j])?,
                        positions,
                        vectors,
                        column,
                    );
                } else {
                    self.push_column(
                        &labels,
                        j,
                        &Direction::<3>::new(self.vectors[j])?,
                        positions,
                        vectors,
                        column,
                    );
                }
                Ok(())
            },
        )?;
        if tangent.iter().any(|&v| !finite(v)) {
            return Err(Error::NonFinite("plane expansion tangent overflow".into()));
        }
        Ok(tangent)
    }

    /// Evaluate and contract the same angular jets used by the reverse contraction.
    fn push_column<const N: usize>(
        &self,
        labels: &AngularLabels,
        j: usize,
        direction: &Direction<N>,
        positions: &[[f64; 3]],
        vectors: &[[Complex; 3]],
        column: &mut [Complex],
    ) {
        let vector = self.vectors[j];
        let angular = labels.evaluate(
            &self.basis,
            vector,
            direction,
            self.polarizations[j],
            self.helicity,
        );
        let axes = if matches!(self.basis, MultipoleBasis::Spherical(_)) {
            3
        } else {
            2
        };
        let phases: Vec<_> = self
            .basis
            .positions()
            .iter()
            .enumerate()
            .map(|(p, &position)| {
                let angle: Complex = (0..3).map(|a| vector[a] * positions[p][a]).sum::<Complex>()
                    + (0..N.min(axes))
                        .map(|a| vectors[j][a] * position[a])
                        .sum::<Complex>();
                (phase(vector, position), Complex::i() * angle)
            })
            .collect();
        for (i, (out, &slot)) in column.iter_mut().zip(&labels.slot).enumerate() {
            let angular = angular[slot];
            let (phase, angle) = phases[self.basis.position_pol(i).0];
            let derivative: Complex = (0..N.min(axes))
                .map(|a| angular.derivative[a] * vectors[j][a])
                .sum();
            *out = phase * (derivative + angular.value * angle);
        }
    }
    /// Differentiate position phases and the full direction-dependent angular coefficient.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<ExpansionGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "invalid plane-expansion cotangent".into(),
            ));
        }
        let fixed_vectors = self.fixed_vectors;
        let labels = AngularLabels::of(&self.basis);
        // Each plane mode writes its own wavevector gradient in place; the partial sums
        // carry only the position gradients.
        let mut vectors = vec![[Complex::default(); 3]; self.vectors.len()];
        let modes: Vec<_> = self.vectors.iter().zip(&mut vectors).collect();
        let positions = try_fold_ordered(
            modes,
            true,
            || vec![[0.0; 3]; self.basis.positions().len()],
            |mut positions, j, (&vector, vector_gradient)| -> Result<_> {
                // Position gradients need only values, which are identical for every
                // derivative count; fixed vectors skip the direction derivatives.
                if fixed_vectors {
                    let direction = Direction::<0>::new(vector)?;
                    self.contract(
                        &labels,
                        j,
                        &direction,
                        cotangent,
                        &mut positions,
                        vector_gradient,
                    );
                } else {
                    let direction = Direction::<3>::new(vector)?;
                    self.contract(
                        &labels,
                        j,
                        &direction,
                        cotangent,
                        &mut positions,
                        vector_gradient,
                    );
                }
                Ok(positions)
            },
            |mut total, partial| {
                for (a, b) in total.iter_mut().flatten().zip(partial.iter().flatten()) {
                    *a += b;
                }
                total
            },
        )?;
        Ok(ExpansionGradient { positions, vectors })
    }

    /// Add the gradients of column `j` to the position gradients `positions` and, with
    /// direction derivatives (`N > 0`), to its wavevector gradient `vector_gradient`.
    fn contract<const N: usize>(
        &self,
        labels: &AngularLabels,
        j: usize,
        direction: &Direction<N>,
        cotangent: &DMatrix<Complex>,
        positions: &mut [[f64; 3]],
        vector_gradient: &mut [Complex; 3],
    ) {
        let vector = self.vectors[j];
        let angular = labels.evaluate(
            &self.basis,
            vector,
            direction,
            self.polarizations[j],
            self.helicity,
        );
        let phases: Vec<_> = self
            .basis
            .positions()
            .iter()
            .map(|&p| phase(vector, p))
            .collect();
        // Cylindrical axial components are fixed labels with zero cotangents.
        let axes = if matches!(self.basis, MultipoleBasis::Spherical(_)) {
            3
        } else {
            2
        };
        for (i, &slot) in labels.slot.iter().enumerate() {
            let p = self.basis.position_pol(i).0;
            let position = self.basis.positions()[p];
            let angular = &angular[slot];
            let phase = phases[p];
            let value = phase * angular.value;
            for axis in 0..3 {
                positions[p][axis] +=
                    (cotangent[(i, j)].conj() * value * Complex::i() * vector[axis]).re;
                if N > 0 && axis < axes {
                    vector_gradient[axis] += cotangent[(i, j)]
                        * (phase
                            * (angular.derivative[axis]
                                + Complex::i() * position[axis] * angular.value))
                            .conj();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::expansion;
    use crate::{
        Complex, Error, sw,
        test_support::{assert_same_bits_on_pools, bits, patterned, spherical_basis},
    };

    /// The overflowing wave of the plane-field tests, expanded about the same points.
    #[test]
    fn overflowing_plane_expansions_report_nonfinite() {
        let k = vec![[0.6.into(), 0.0.into(), Complex::new(0.8, 2.0)]];
        let basis = |z| spherical_basis(1, [0.0, 0.0, z]);
        let at = |z| expansion(basis(z), k.clone(), vec![0], true, false);
        assert!(matches!(at(-400.0), Err(Error::NonFinite(_))));
        let (_, residual) = at(-350.0).unwrap();
        let tangent = residual.pushforward(&[[0.0, 0.0, 1e10]], &[[0.0.into(); 3]]);
        assert!(matches!(tangent, Err(Error::NonFinite(_))));
    }

    /// The position gradients add in chunks fixed by the plane-mode count: with more
    /// plane modes than chunks, every gradient repeats bit for bit on every pool size.
    #[test]
    fn pullback_does_not_depend_on_the_thread_count() {
        let modes = sw::modes(3).unwrap();
        let basis = sw::Basis {
            modes: (0..2)
                .flat_map(|p| modes.iter().map(move |&mode| (p, mode)))
                .collect(),
            positions: vec![[0.1, -0.2, 0.3], [-0.3, 0.2, -0.1]],
        };
        let count = 100_u32;
        let vectors: Vec<[Complex; 3]> = (0..count)
            .map(|j| {
                let t = f64::from(j);
                let (theta, phi) = (1.55 + 1.3 * (0.61 * t).sin(), 2.3 * t);
                [
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]
                .map(|x| Complex::new(1.2 * x, 0.0))
            })
            .collect();
        let polarizations: Vec<u8> = (0..count).map(|j| u8::from(j % 2 == 0)).collect();
        let g = patterned(basis.modes.len(), vectors.len(), 0.5);
        assert_same_bits_on_pools(|| {
            let (_, residual) = expansion(
                basis.clone(),
                vectors.clone(),
                polarizations.clone(),
                true,
                false,
            )
            .unwrap();
            let g = residual.pullback(&g).unwrap();
            bits(&[&g.positions, &g.vectors])
        });
    }
}
