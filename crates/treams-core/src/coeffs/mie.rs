//! Mie coefficients of multilayer chiral spheres and their analytic pullback.
//!
//! Upstream: `treams.coeffs.mie`. The pullback is a treams-rs extension.
#![allow(clippy::indexing_slicing)] // Fixed-size matrices of the interface equations.

use super::{Material, Matrix2, Matrix4, Matrix42, material::add_entries, validate_layers};
use crate::special::{Radial, spherical_radial};
use crate::{
    Complex, Error, MAX_DEGREE, Result,
    numerics::{finite, ratio},
};

mod saved;

// Radial entries of one interface argument x = size (n -+ kappa) for the regular and
// singular kinds: (f, f' + f/x) of the spherical function f, and their x-derivatives.
#[derive(Clone, Copy, Debug)]
struct Riccati {
    waves: [(Complex, Complex); 2],
    slopes: [(Complex, Complex); 2],
}

impl Riccati {
    fn new(l: u32, x: Complex) -> Result<Self> {
        let zero = (Complex::default(), Complex::default());
        let mut result = Self {
            waves: [zero; 2],
            slopes: [zero; 2],
        };
        for (kind, radial) in [Radial::Regular, Radial::Singular].into_iter().enumerate() {
            let f = spherical_radial(l, x, radial)?;
            result.waves[kind] = (f.value, f.first + f.value / x);
            result.slopes[kind] = (f.first, f.second + f.first / x - f.value / x.powu(2));
        }
        Ok(result)
    }
}

// Radial entries of both helicities on both sides of an interface; achiral media share
// one argument for both helicities.
fn radials(l: u32, x: [[Complex; 2]; 2]) -> Result<[[Riccati; 2]; 2]> {
    let side = |[negative, positive]: [Complex; 2]| -> Result<[Riccati; 2]> {
        let first = Riccati::new(l, negative)?;
        Ok([
            first,
            if positive == negative {
                first
            } else {
                Riccati::new(l, positive)?
            },
        ])
    };
    Ok([side(x[0])?, side(x[1])?])
}

/// One boundary between two media, as the pullback reads it.
#[derive(Clone, Debug)]
struct Interface {
    /// Arguments `size (n -+ kappa)` of the inner and outer medium, by helicity.
    x: [[Complex; 2]; 2],
    /// Impedances of the inner and outer medium.
    z: [Complex; 2],
    /// Radial functions at `x`.
    radials: [[Riccati; 2]; 2],
    /// The four-channel interface matrix `F` from inner to outer coefficients.
    matrix: Matrix4,
    /// The product of the interface matrices inside this boundary, applied to the
    /// two regular core channels.
    before: Matrix42,
}

/// What [`mie`] saves for its pullback: the interface matrices and radial functions of
/// one degree `l`, so the pullback evaluates no Bessel functions.
#[derive(Clone, Debug)]
pub struct MieResidual {
    sizes: Vec<f64>,
    materials: Vec<Material>,
    interfaces: Vec<Interface>,
    inverse: Matrix2,
    value: Matrix2,
}

/// Gradients of the inputs of [`mie`], in the order of its arguments.
#[derive(Clone, Debug)]
pub struct MieGradient {
    /// Size-parameter gradients, from inner to outer boundary.
    pub sizes: Vec<f64>,
    /// Permittivity gradients, including the embedding medium.
    pub epsilon: Vec<Complex>,
    /// Permeability gradients, including the embedding medium.
    pub mu: Vec<Complex>,
    /// Chirality gradients, including the embedding medium.
    pub kappa: Vec<Complex>,
}

impl MieGradient {
    /// Zero gradients of `boundaries` size parameters and `boundaries + 1` media.
    pub(crate) fn zeros(boundaries: usize) -> Self {
        let media = vec![Complex::default(); boundaries + 1];
        Self {
            sizes: vec![0.0; boundaries],
            epsilon: media.clone(),
            mu: media.clone(),
            kappa: media,
        }
    }

    /// Add the gradient of the same layers along another cotangent.
    pub(crate) fn accumulate(&mut self, other: &Self) {
        add_entries(&mut self.sizes, &other.sizes);
        add_entries(&mut self.epsilon, &other.epsilon);
        add_entries(&mut self.mu, &other.mu);
        add_entries(&mut self.kappa, &other.kappa);
    }
}

/// A coefficient matrix in the mode order of a T-matrix.
///
/// [`mie`] and [`mie_cyl`](fn@super::mie_cyl) index their matrices by helicity (negative,
/// positive), which is the polarization index `pol` (0, 1). A basis lists `pol = 1`
/// before `pol = 0`, so both axes swap. The swap is its own inverse, so it also takes a
/// cotangent in mode order back to helicity order.
pub(crate) fn to_mode_order(matrix: &Matrix2) -> Matrix2 {
    Matrix2::new(
        matrix[(1, 1)],
        matrix[(1, 0)],
        matrix[(0, 1)],
        matrix[(0, 0)],
    )
}

// The four-channel interface matrix F and, if TANGENT, its directional derivative
// along (dx, dz). The radial derivatives are the cached slopes times dx.
fn interface<const TANGENT: bool>(
    x: [[Complex; 2]; 2],
    z: [Complex; 2],
    radials: &[[Riccati; 2]; 2],
    dx: [[Complex; 2]; 2],
    dz: [Complex; 2],
) -> (Matrix4, Matrix4) {
    let derivative = |side: usize, pol: usize, kind: usize| {
        let (value, first) = radials[side][pol].slopes[kind];
        (value * dx[side][pol], first * dx[side][pol])
    };
    let impedance_ratio = z[1] / z[0];
    let da = (dz[1] / z[0] - impedance_ratio * dz[0] / z[0]) / (2.0 * Complex::i());
    let mut f = Matrix4::zeros();
    let mut df = Matrix4::zeros();
    for p in 0..2 {
        let square = x[1][p].powu(2);
        for q in 0..2 {
            let sign = if p == q { 1.0 } else { -1.0 };
            let a = (impedance_ratio + sign) / (2.0 * Complex::i());
            let factor = a * square;
            let dfactor = if TANGENT {
                da * square + 2.0 * a * x[1][p] * dx[1][p]
            } else {
                Complex::default()
            };
            for kind in 0..2 {
                let inner = radials[0][q].waves[kind];
                for out_kind in 0..2 {
                    let outer = radials[1][p].waves[1 - out_kind];
                    let out_sign = if out_kind == 0 { 1.0 } else { -1.0 };
                    let term = out_sign * (outer.1 * inner.0 - sign * outer.0 * inner.1);
                    let entry = (p + out_kind * 2, q + kind * 2);
                    f[entry] = term * factor;
                    if TANGENT {
                        let (dinner, douter) =
                            (derivative(0, q, kind), derivative(1, p, 1 - out_kind));
                        let dterm = out_sign
                            * (douter.1 * inner.0 + outer.1 * dinner.0
                                - sign * (douter.0 * inner.1 + outer.0 * dinner.1));
                        df[entry] = dterm * factor + term * dfactor;
                    }
                }
            }
        }
    }
    (f, df)
}

// Absorbing media can make every entry enormous while the coefficient ratio is
// well conditioned. Normalize before forming the determinant, then use the
// shared scaled complex division instead of squaring its magnitude.
fn inverse2(matrix: Matrix2) -> Result<Matrix2> {
    let scale = matrix
        .iter()
        .map(|z| z.re.abs().max(z.im.abs()))
        .fold(0.0, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return Err(Error::Singular);
    }
    let a = matrix.map(|z| z / scale);
    let determinant = a.determinant();
    if determinant == Complex::default() {
        return Err(Error::Singular);
    }
    Ok(Matrix2::new(a[(1, 1)], -a[(0, 1)], -a[(1, 0)], a[(0, 0)])
        .map(|z| ratio(z, determinant) / scale))
}

/// Mie coefficients of degree `l` of a concentric multilayer chiral sphere.
///
/// `sizes` are the size parameters `k0 r` of the boundaries, from inner to outer, and
/// `materials` has one more entry, the embedding medium last.
///
/// Upstream: `treams.coeffs.mie`.
pub fn mie(l: u32, sizes: &[f64], materials: &[Material]) -> Result<MieResidual> {
    if !(1..=MAX_DEGREE.unsigned_abs()).contains(&l) {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "require 1 <= l <= 128".into(),
        ));
    }
    validate_layers(sizes, materials)?;
    let mut q = Matrix42::zeros();
    q[(0, 0)] = Complex::new(1.0, 0.0);
    q[(1, 1)] = Complex::new(1.0, 0.0);
    let mut interfaces = Vec::with_capacity(sizes.len());
    for (i, size) in sizes.iter().enumerate() {
        let pair = [materials[i], materials[i + 1]];
        let x = pair.map(|m| [size * (m.index() - m.kappa), size * (m.index() + m.kappa)]);
        let z = pair.map(Material::impedance);
        let radials = radials(l, x)?;
        let (matrix, _) = interface::<false>(
            x,
            z,
            &radials,
            [[Complex::default(); 2]; 2],
            [Complex::default(); 2],
        );
        interfaces.push(Interface {
            x,
            z,
            radials,
            matrix,
            before: q,
        });
        q = matrix * q;
    }
    let upper = q.fixed_rows::<2>(0).into_owned();
    let inverse = inverse2(upper)?;
    let value = q.fixed_rows::<2>(2) * inverse;
    if value.iter().any(|z| !finite(*z)) {
        return Err(Error::Singular);
    }
    Ok(MieResidual {
        sizes: sizes.to_vec(),
        materials: materials.to_vec(),
        interfaces,
        inverse,
        value,
    })
}

impl MieResidual {
    /// The number of recorded layer boundaries.
    #[must_use]
    pub fn boundaries(&self) -> usize {
        self.sizes.len()
    }

    /// The Mie coefficient matrix, which the pullback reads.
    #[must_use]
    pub const fn value(&self) -> &Matrix2 {
        &self.value
    }

    /// The coefficient tangent along changes of the sizes and materials. Each
    /// `materials` entry stores changes of epsilon, mu and kappa, including the
    /// embedding medium. One combined direction travels through the cached
    /// interfaces; no Bessel functions or parameter Jacobians are evaluated.
    pub fn pushforward(&self, sizes: &[f64], materials: &[Material]) -> Result<Matrix2> {
        super::validate_layer_tangents(self.sizes.len(), sizes, materials)?;
        let optical: Vec<_> = self
            .materials
            .iter()
            .zip(materials)
            .map(|(&material, &direction)| material.pushforward(direction))
            .collect();
        let mut dq = Matrix42::zeros();
        for (i, layer) in self.interfaces.iter().enumerate() {
            let dx = std::array::from_fn(|side| {
                let material = optical[i + side];
                let dn = [
                    material.index - material.kappa,
                    material.index + material.kappa,
                ];
                std::array::from_fn(|pol| {
                    layer.x[side][pol] * (sizes[i] / self.sizes[i]) + self.sizes[i] * dn[pol]
                })
            });
            let dz = [optical[i].impedance, optical[i + 1].impedance];
            let (_, df) = interface::<true>(layer.x, layer.z, &layer.radials, dx, dz);
            dq = df * layer.before + layer.matrix * dq;
        }
        Ok((dq.fixed_rows::<2>(2) - self.value * dq.fixed_rows::<2>(0)) * self.inverse)
    }

    /// Gradients of the size parameters and the materials from `cotangent`, the
    /// gradient of a real loss with respect to the coefficient matrix.
    ///
    /// The pullback runs on the calling thread and adds the boundaries in order, from
    /// the outermost in.
    pub fn pullback(&self, cotangent: &Matrix2) -> Result<MieGradient> {
        if cotangent.iter().any(|z| !finite(*z)) {
            return Err(Error::InvalidInput("cotangents must be finite".into()));
        }
        let mut result = MieGradient::zeros(self.sizes.len());
        let lower = cotangent * self.inverse.adjoint();
        let upper = -self.value.adjoint() * lower;
        let mut gq = Matrix42::zeros();
        gq.fixed_rows_mut::<2>(0).copy_from(&upper);
        gq.fixed_rows_mut::<2>(2).copy_from(&lower);
        for (i, layer) in self.interfaces.iter().enumerate().rev() {
            let gf = gq * layer.before.adjoint();
            let contract = |dx, dz| -> Complex {
                let (_, df) = interface::<true>(layer.x, layer.z, &layer.radials, dx, dz);
                gf.iter()
                    .zip(df.iter())
                    .map(|(g, d)| g.conj() * d)
                    .sum::<Complex>()
                    .conj()
            };
            let size = self.sizes[i];
            result.sizes[i] = contract(
                layer.x.map(|row| row.map(|x| x / size)),
                [Complex::default(); 2],
            )
            .re;
            for side in 0..2 {
                let material = self.materials[i + side];
                let targets = [&mut result.epsilon, &mut result.mu, &mut result.kappa];
                for (tangent, target) in material.tangents().into_iter().zip(targets) {
                    let mut dx = [[Complex::default(); 2]; 2];
                    dx[side] = [
                        size * (tangent.index - tangent.kappa),
                        size * (tangent.index + tangent.kappa),
                    ];
                    let mut dz = [Complex::default(); 2];
                    dz[side] = tangent.impedance;
                    target[i + side] += contract(dx, dz);
                }
            }
            gq = layer.matrix.adjoint() * gq;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{Material, Matrix2, inverse2, mie};
    use crate::{
        Complex,
        test_support::{ALGEBRA_CASES, Chirality, complex, helicity_ks, prop_assert_close, re_dot},
    };

    /// A material with epsilon in [1, 6] + i [0, 0.5 loss], mu in [1, 2] + i [0, 0.2 loss]
    /// and a real chirality of at most 0.3 Re n, so both helicity indices stay positive.
    fn material(loss: f64) -> impl Strategy<Value = Material> {
        crate::test_support::material(
            1.0..6.0,
            1.0..2.0,
            [0.5 * loss, 0.2 * loss],
            Chirality::Relative(0.3),
        )
    }

    /// One to three concentric boundaries at least 0.05 apart from 0.2 up, the layer
    /// materials, and a lossless chiral embedding.
    fn sphere(loss: f64) -> impl Strategy<Value = (Vec<f64>, Vec<Material>)> {
        (1_usize..=3)
            .prop_flat_map(move |layers| {
                (
                    0.2_f64..1.0,
                    prop::collection::vec(0.05_f64..1.0, layers - 1),
                    prop::collection::vec(material(loss), layers),
                    material(0.0),
                )
            })
            .prop_map(|(first, gaps, mut materials, embedding)| {
                let sizes = std::iter::once(first)
                    .chain(gaps)
                    .scan(0.0, |size, gap| {
                        *size += gap;
                        Some(*size)
                    })
                    .collect();
                materials.push(embedding);
                (sizes, materials)
            })
    }

    /// A lossless dielectric sphere of size 0.2 to 3 and permittivity 1 to 8 in vacuum.
    fn large() -> impl Strategy<Value = (Vec<f64>, Vec<Material>)> {
        (0.2_f64..3.0, 1.0_f64..8.0).prop_map(|(size, epsilon)| {
            let sphere = Material {
                epsilon: Complex::from(epsilon),
                ..Material::default()
            };
            (vec![size], vec![sphere, Material::default()])
        })
    }

    fn cotangent() -> impl Strategy<Value = Matrix2> {
        [complex(1.0), complex(1.0), complex(1.0), complex(1.0)]
            .prop_map(|[a, b, c, d]| Matrix2::new(a, b, c, d))
    }

    proptest! {
        // A Mie evaluation of up to 3 layers costs microseconds, as an algebraic check does.
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn mie_reciprocity_mirror_and_duality((sizes, materials) in sphere(1.0), l in 1_u32..8) {
            check_mie_symmetries(l, &sizes, &materials)?;
        }

        #[test]
        fn lossless_mie_optical_theorem_and_its_derivative(
            (sizes, materials) in prop_oneof![sphere(0.0), large()], l in 1_u32..9,
        ) {
            check_lossless_mie(l, &sizes, &materials)?;
        }

        #[test]
        fn mie_pullback_scaling_identities(
            (sizes, materials) in sphere(1.0), l in 1_u32..8, g in cotangent(),
        ) {
            check_mie_scaling(l, &sizes, &materials, &g)?;
        }
    }

    fn largest(matrix: &Matrix2) -> f64 {
        matrix.iter().map(|z| z.norm()).fold(0.0, f64::max)
    }

    /// Helicity wavenumbers `n -+ kappa` of the embedding, the last material, at
    /// `k0 = 1`.
    fn embedding(materials: &[Material]) -> [Complex; 2] {
        helicity_ks(1.0, &materials.last().copied().unwrap_or_default())
    }

    /// Reciprocity `k_+^2 T_-+ = k_-^2 T_+-` in the helicity wavenumbers of the embedding;
    /// mirroring every chirality swaps the helicities; and spheres whose media all have
    /// `epsilon = mu` are dual symmetric and keep the helicity.
    fn check_mie_symmetries(
        l: u32,
        sizes: &[f64],
        materials: &[Material],
    ) -> Result<(), TestCaseError> {
        let t = mie(l, sizes, materials).unwrap().value;
        let scale = largest(&t);
        let [minus, plus] = embedding(materials).map(|k| k * k);
        prop_assert_close!(
            plus * t[(0, 1)],
            minus * t[(1, 0)],
            1e-12 * scale * plus.norm().max(minus.norm())
        );
        let mirrored: Vec<_> = materials
            .iter()
            .map(|m| Material {
                kappa: -m.kappa,
                ..*m
            })
            .collect();
        let swapped = mie(l, sizes, &mirrored).unwrap().value;
        let reversed = Matrix2::new(t[(1, 1)], t[(1, 0)], t[(0, 1)], t[(0, 0)]);
        prop_assert_close!(swapped, reversed, 1e-13 * scale);
        let dual: Vec<_> = materials
            .iter()
            .map(|m| Material {
                mu: m.epsilon,
                ..*m
            })
            .collect();
        let t = mie(l, sizes, &dual).unwrap().value;
        prop_assert_close!(
            [t[(0, 1)], t[(1, 0)]],
            [Complex::default(); 2],
            1e-14 * largest(&t)
        );
        Ok(())
    }

    /// Lossless spheres conserve energy: `T^H W + W T + 2 T^H W T = 0` with the flux
    /// weights `W = diag(k_-^-2, k_+^-2)` of the embedding, which makes `I + 2T` unitary
    /// for an achiral one. Its derivative along every lossless direction that keeps W,
    /// all but the embedding's parameters, is `Re tr(G^H dT) = 0` with `G = W (I + 2T)`:
    /// the pullback of G has vanishing size gradients and real material parts.
    ///
    /// Near a sharp resonance the halves `W` and `2 W T` of G pull back to large, nearly
    /// cancelling gradients, so each bound scales with the gradient of `<W, T>` in that
    /// parameter, the pullbacks of `W` and `i W`. A fixed bound of `1e-12 max W` fails
    /// for large spheres at resonance, which need about `2e-11 max W`.
    fn check_lossless_mie(
        l: u32,
        sizes: &[f64],
        materials: &[Material],
    ) -> Result<(), TestCaseError> {
        let t = mie(l, sizes, materials).unwrap().value;
        let weights = Matrix2::from_diagonal(&embedding(materials).map(|k| (k * k).inv()).into());
        let scale = largest(&weights);
        let two = Complex::new(2.0, 0.0);
        let residual = t.adjoint() * weights + weights * t + t.adjoint() * weights * t * two;
        prop_assert_close!(residual, Matrix2::zeros(), 1e-12 * scale);
        // The gradients of the sizes, then of the interior epsilon, mu and kappa.
        let lossless = |g: Matrix2| {
            let gradient = mie(l, sizes, materials).unwrap().pullback(&g).unwrap();
            let interior = [gradient.epsilon, gradient.mu, gradient.kappa]
                .into_iter()
                .flat_map(|values| values.into_iter().take(sizes.len()));
            let boundaries = gradient.sizes.into_iter().map(Complex::from);
            boundaries.chain(interior).collect::<Vec<_>>()
        };
        let gradient = lossless(weights * (Matrix2::identity() + t * two));
        let real = lossless(weights);
        let imaginary = lossless(weights * Complex::i());
        for (i, value) in gradient.iter().enumerate() {
            let tolerance = 1e-12 * scale.max(real[i].norm() + imaginary[i].norm());
            prop_assert_close!(value.re, 0.0, tolerance, "parameter {}", i);
        }
        Ok(())
    }

    /// T depends only on `size (n -+ kappa)` and impedance ratios, which scaling every
    /// medium by `epsilon -> s^2 epsilon` (or `mu -> s^2 mu`), `kappa -> s kappa` and every
    /// size by `1/s` leaves unchanged. At `s = 1` the pullback therefore satisfies
    /// `sum size g_size = 2 Re<g_epsilon, epsilon> + Re<g_kappa, kappa>`, and the same with mu.
    fn check_mie_scaling(
        l: u32,
        sizes: &[f64],
        materials: &[Material],
        g: &Matrix2,
    ) -> Result<(), TestCaseError> {
        let gradient = mie(l, sizes, materials).unwrap().pullback(g).unwrap();
        let size = sizes
            .iter()
            .zip(&gradient.sizes)
            .map(|(s, g)| s * g)
            .sum::<f64>();
        let epsilon = re_dot(&gradient.epsilon, materials.iter().map(|m| m.epsilon));
        let mu = re_dot(&gradient.mu, materials.iter().map(|m| m.mu));
        let kappa = re_dot(&gradient.kappa, materials.iter().map(|m| m.kappa));
        let scale = size.abs() + epsilon.abs() + mu.abs() + kappa.abs();
        prop_assert_close!(size, 2.0 * epsilon + kappa, 1e-12 * scale);
        prop_assert_close!(size, 2.0 * mu + kappa, 1e-12 * scale);
        Ok(())
    }

    proptest! {
        // One closed-form 2 x 2 inverse per case.
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn inverse_preserves_complex_scale(
            re in -1.0_f64..1.0,
            im in -1.0_f64..1.0,
            exponent in -300_i32..301,
        ) {
            let z = Complex::new(re, im);
            let matrix = Matrix2::new(3.0 + z, z, -z, 4.0 - z)
                * Complex::new(10.0_f64.powi(exponent), 0.0);
            let inverse = inverse2(matrix).unwrap();
            prop_assert!((matrix * inverse - Matrix2::identity()).norm() < 4e-15);
        }
    }

    #[test]
    fn metallic_sphere_matches_independent_mie_ratios() {
        // 120-digit direct Riccati-Bessel ratios from scripts/qualify_references.py,
        // stable at 80 digits; x = 80, epsilon = -8 + 0.4i, vacuum.
        let references = [
            (
                1,
                (-0.499_957_965_242_302_9, -0.000_033_295_596_701_331_95),
                (-0.315_374_612_638_891_9, -0.377_908_027_840_409_7),
            ),
            (
                3,
                (-0.499_774_523_335_837_05, -0.000_229_816_544_066_314_44),
                (-0.360_037_027_186_839_1, -0.335_632_030_366_205_5),
            ),
            (
                80,
                (-0.421_594_673_791_517_6, -0.391_815_903_887_81),
                (-0.280_927_898_598_179_07, -0.045_759_796_275_944_01),
            ),
            (
                99,
                (-6.117_266_839_130_22e-10, 1.109_939_769_728_892_8e-8),
                (-5.594_374_358_681_406e-10, 1.597_214_940_003_522_4e-8),
            ),
        ];
        let material = Material {
            epsilon: Complex::new(-8.0, 0.4),
            ..Material::default()
        };
        for (l, diagonal, off_diagonal) in references {
            let diagonal = Complex::new(diagonal.0, diagonal.1);
            let off_diagonal = Complex::new(off_diagonal.0, off_diagonal.1);
            let reference = Matrix2::new(diagonal, off_diagonal, off_diagonal, diagonal);
            let result = mie(l, &[80.0], &[material, Material::default()]).unwrap();
            for (actual, expected) in result.value.iter().zip(reference.iter()) {
                assert!((*actual - expected).norm() <= 2e-13 + 2e-11 * expected.norm());
            }
        }
    }
}
