//! Shared helpers of the Rust tests:
//!
//! - case budgets by cost: [`ALGEBRA_CASES`], [`DEFAULT_CASES`], [`EXPENSIVE_CASES`];
//! - cotangent pairings [`re_dot`] and [`dot`];
//! - finite differences [`central`] and [`five_point`], and [`gauss_legendre`] nodes;
//! - input strategies [`complex`], [`log_polar`], [`complex_vec`], [`complex_matrix`],
//!   [`log_uniform`], [`radial`], [`radial_kind`], [`degree_order`] and the media of
//!   [`material`], with their [`helicity_ks`];
//! - fixtures [`patterned`], [`rotate`], [`spherical_basis`] and [`cylindrical_basis`];
//! - the `key: values` text tables of [`table`];
//! - [`prop_assert_close!`], which reports both compared values and their distance;
//! - [`assert_same_bits_on_pools`], which runs a computation on pools of one to four
//!   threads and compares the [`bits`] of its results.
//!
//! `docs/development/testing.md` describes the test layout and conventions.

use std::{
    borrow::Borrow,
    fmt::Debug,
    ops::{Add, Div, Mul, Range, Sub},
    str::FromStr,
};

use nalgebra::{DMatrix, SMatrix};
use proptest::prelude::*;

use crate::{Complex, coeffs::Material, cw, numerics, special::Radial, sw};

/// Proptest cases for closed-form algebraic identities, which cost microseconds each.
pub(crate) const ALGEBRA_CASES: u32 = 256;
/// Proptest cases for single-kernel physical, scaling and adjoint identities.
pub(crate) const DEFAULT_CASES: u32 = 64;
/// Proptest cases for lattice sums, cluster solves and surface integrals, including
/// the finite differences taken through them.
pub(crate) const EXPENSIVE_CASES: u32 = 24;

/// Real Hermitian pairing `Re Σ conj(a) b`, the pairing of cotangents and changes of
/// complex values in the crate. Panics when the sequences differ in length.
pub(crate) fn re_dot<A: Borrow<Complex>, B: Borrow<Complex>>(
    a: impl IntoIterator<Item = A>,
    b: impl IntoIterator<Item = B>,
) -> f64 {
    paired(a, b, |a, b| (a.borrow().conj() * b.borrow()).re)
}

/// Euclidean pairing `Σ a b` of real values. Panics when the sequences differ in length.
pub(crate) fn dot<A: Borrow<f64>, B: Borrow<f64>>(
    a: impl IntoIterator<Item = A>,
    b: impl IntoIterator<Item = B>,
) -> f64 {
    paired(a, b, |a, b| a.borrow() * b.borrow())
}

fn paired<A, B>(
    a: impl IntoIterator<Item = A>,
    b: impl IntoIterator<Item = B>,
    term: impl Fn(A, B) -> f64,
) -> f64 {
    let (mut a, mut b) = (a.into_iter(), b.into_iter());
    let mut sum = 0.0;
    loop {
        match (a.next(), b.next()) {
            (Some(a), Some(b)) => sum += term(a, b),
            (None, None) => return sum,
            _ => panic!("paired sequences differ in length"),
        }
    }
}

/// Central difference `(f(h) - f(-h)) / 2h`, accurate to `O(h²)`.
pub(crate) fn central<T: Sub<Output = T> + Div<f64, Output = T>>(
    h: f64,
    f: impl Fn(f64) -> T,
) -> T {
    (f(h) - f(-h)) / (2.0 * h)
}

/// Fourth-order central difference `(f(-2h) - 8f(-h) + 8f(h) - f(2h)) / 12h`, for
/// derivatives of rapidly varying (outgoing) functions, normally with `h = 1e-4`.
pub(crate) fn five_point<T>(h: f64, f: impl Fn(f64) -> T) -> T
where
    T: Add<Output = T> + Sub<Output = T> + Mul<f64, Output = T> + Div<f64, Output = T>,
{
    (f(-2.0 * h) - f(-h) * 8.0 + f(h) * 8.0 - f(2.0 * h)) / (12.0 * h)
}

/// The `n` Gauss-Legendre nodes and weights on `[-1, 1]`, by Newton's method on `P_n`.
pub(crate) fn gauss_legendre(n: u32) -> Vec<(f64, f64)> {
    let order = f64::from(n);
    (1..=n)
        .map(|i| {
            let mut x = (std::f64::consts::PI * (f64::from(i) - 0.25) / (order + 0.5)).cos();
            let mut derivative = 1.0;
            for _ in 0..50 {
                let (mut lower, mut value) = (1.0, x);
                for k in 2..=n {
                    let k = f64::from(k);
                    (lower, value) = (value, ((2.0 * k - 1.0) * x * value - (k - 1.0) * lower) / k);
                }
                derivative = order * (x * value - lower) / (x * x - 1.0);
                x -= value / derivative;
            }
            (x, 2.0 / ((1.0 - x * x) * derivative * derivative))
        })
        .collect()
}

/// Complex numbers with real and imaginary parts in `[-bound, bound)`.
pub(crate) fn complex(bound: f64) -> impl Strategy<Value = Complex> {
    (-bound..bound, -bound..bound).prop_map(|(re, im)| Complex::new(re, im))
}

/// Complex numbers `10^x e^(i phase)` with `x` uniform in `decades` and the phase
/// uniform in `phases`: log-uniform moduli for arguments that span decades.
pub(crate) fn log_polar(decades: Range<f64>, phases: Range<f64>) -> impl Strategy<Value = Complex> {
    (decades, phases).prop_map(|(x, phase)| Complex::from_polar(10_f64.powf(x), phase))
}

/// `len` independent [`complex`] values.
pub(crate) fn complex_vec(len: usize, bound: f64) -> impl Strategy<Value = Vec<Complex>> {
    prop::collection::vec(complex(bound), len)
}

/// A `rows` by `cols` matrix of independent [`complex`] entries.
pub(crate) fn complex_matrix(
    rows: usize,
    cols: usize,
    bound: f64,
) -> impl Strategy<Value = DMatrix<Complex>> {
    complex_vec(rows * cols, bound).prop_map(move |values| DMatrix::from_vec(rows, cols, values))
}

/// Values `10^x` with the exponent `x` uniform in `decades`, for quantities that span
/// several orders of magnitude; shrinks toward `10^decades.start`.
pub(crate) fn log_uniform(decades: Range<f64>) -> impl Strategy<Value = f64> {
    decades.prop_map(|exponent| 10_f64.powf(exponent))
}

/// Regular or singular radial solutions; shrinks to regular.
pub(crate) fn radial() -> impl Strategy<Value = Radial> {
    prop_oneof![Just(Radial::Regular), Just(Radial::Singular)]
}

/// The radial kind of a drawn boolean, regular for `true`, for checks whose recorded
/// seeds draw `regular in any::<bool>()` rather than [`radial`].
pub(crate) const fn radial_kind(regular: bool) -> Radial {
    if regular {
        Radial::Regular
    } else {
        Radial::Singular
    }
}

/// A degree `l` from `degrees` with an order `m` drawn uniformly from `-l..=l`.
pub(crate) fn degree_order(degrees: Range<i32>) -> impl Strategy<Value = (i32, i32)> {
    degrees.prop_flat_map(|l| (Just(l), -l..=l))
}

/// Bound of the real chirality `κ` that [`material`] draws.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Chirality {
    /// `|κ| < bound`.
    Absolute(f64),
    /// `|κ| < bound Re n`, so both helicity indices `n ∓ κ` keep a positive real part
    /// for a bound below 1.
    Relative(f64),
}

/// Passive, isotropic and reciprocal media: `Re ε` uniform in `epsilon`, `Re μ` in `mu`,
/// `Im ε` and `Im μ` in `[0, loss[0]]` and `[0, loss[1]]`, and a real `κ` within
/// `chirality`.
pub(crate) fn material(
    epsilon: Range<f64>,
    mu: Range<f64>,
    loss: [f64; 2],
    chirality: Chirality,
) -> impl Strategy<Value = Material> + Clone {
    (epsilon, 0.0..=1.0, mu, 0.0..=1.0, -1.0..1.0).prop_map(
        move |(epsilon, epsilon_loss, mu, mu_loss, u): (f64, f64, f64, f64, f64)| {
            let epsilon = Complex::new(epsilon, loss[0] * epsilon_loss);
            let mu = Complex::new(mu, loss[1] * mu_loss);
            let kappa = match chirality {
                Chirality::Absolute(bound) => bound * u,
                Chirality::Relative(bound) => bound * u * numerics::complex_sqrt(epsilon * mu).re,
            };
            Material {
                epsilon,
                mu,
                kappa: Complex::new(kappa, 0.0),
            }
        },
    )
}

/// Negative and positive helicity wavenumbers `k0 (n - κ, n + κ)` of a material with
/// refractive index `n` ([`Material::index`]). When `Im n >= 0` and `κ` is real, as for
/// every medium [`material`] draws, they equal `k0` times [`Material::indices`].
pub(crate) fn helicity_ks(k0: f64, material: &Material) -> [Complex; 2] {
    let n = material.index();
    [k0 * (n - material.kappa), k0 * (n + material.kappa)]
}

/// Deterministic dense matrix with entries of modulus below `√2`; `seed` shifts the pattern.
pub(crate) fn patterned(rows: usize, cols: usize, seed: f64) -> DMatrix<Complex> {
    DMatrix::from_fn(rows, cols, |i, j| {
        let phase = f64::from(u32::try_from(i + rows * j).unwrap()) * 0.71 + seed;
        Complex::new(phase.sin(), (phase * 1.3).cos())
    })
}

/// `Rz(phi) Ry(theta) Rz(psi) position`, the rotation of the Euler angles.
pub(crate) fn rotate([phi, theta, psi]: [f64; 3], position: [f64; 3]) -> [f64; 3] {
    let z = |angle: f64| {
        let (sine, cosine) = angle.sin_cos();
        nalgebra::Matrix3::new(cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0)
    };
    let (sine, cosine) = theta.sin_cos();
    let y = nalgebra::Matrix3::new(cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine);
    (z(phi) * y * z(psi) * nalgebra::Vector3::from(position)).into()
}

/// Every spherical mode up to `lmax`, in [`sw::modes`] order, at one position.
pub(crate) fn spherical_basis(lmax: u32, position: [f64; 3]) -> sw::Basis {
    sw::Basis {
        modes: sw::modes(lmax)
            .unwrap()
            .into_iter()
            .map(|mode| (0, mode))
            .collect(),
        positions: vec![position],
    }
}

/// Cylindrical orders `-mmax..=mmax` at one axial wavenumber and position; as in
/// [`sw::modes`], polarization 1 precedes polarization 0.
pub(crate) fn cylindrical_basis(mmax: i32, kz: f64, position: [f64; 3]) -> cw::Basis {
    cw::Basis {
        modes: (-mmax..=mmax)
            .flat_map(|m| [1, 0].map(|pol| (0, cw::Mode { kz, m, pol })))
            .collect(),
        positions: vec![position],
    }
}

/// Distance between two values of a compared quantity: the absolute difference of
/// scalars and the Euclidean (Frobenius) norm of the difference of arrays, slices
/// and vectors, which is infinite when their lengths differ.
pub(crate) trait Distance: Debug {
    /// The distance between `self` and `other`.
    fn distance(&self, other: &Self) -> f64;
}

impl Distance for f64 {
    fn distance(&self, other: &Self) -> f64 {
        (self - other).abs()
    }
}

impl Distance for Complex {
    fn distance(&self, other: &Self) -> f64 {
        (self - other).norm()
    }
}

impl Distance for DMatrix<f64> {
    fn distance(&self, other: &Self) -> f64 {
        (self - other).norm()
    }
}

impl Distance for DMatrix<Complex> {
    fn distance(&self, other: &Self) -> f64 {
        (self - other).norm()
    }
}

impl<const R: usize, const C: usize> Distance for SMatrix<Complex, R, C> {
    fn distance(&self, other: &Self) -> f64 {
        (self - other).norm()
    }
}

impl<T: Distance, const N: usize> Distance for [T; N] {
    fn distance(&self, other: &Self) -> f64 {
        self.as_slice().distance(other.as_slice())
    }
}

/// Collections of different lengths are infinitely far apart, so a short or empty
/// result never passes a comparison.
impl<T: Distance> Distance for [T] {
    fn distance(&self, other: &Self) -> f64 {
        if self.len() != other.len() {
            return f64::INFINITY;
        }
        self.iter()
            .zip(other)
            .fold(0.0, |sum: f64, (a, b)| sum.hypot(a.distance(b)))
    }
}

impl<T: Distance> Distance for Vec<T> {
    fn distance(&self, other: &Self) -> f64 {
        self.as_slice().distance(other.as_slice())
    }
}

impl<T: Distance + ?Sized> Distance for &T {
    fn distance(&self, other: &Self) -> f64 {
        (**self).distance(other)
    }
}

/// Asserts `distance(actual, expected) < tolerance` inside a property, reporting both
/// expressions, their values and the distance. Optional trailing arguments format
/// additional context.
macro_rules! prop_assert_close {
    (@check $actual:expr, $expected:expr, $tolerance:expr, $context:expr) => {{
        let (actual, expected, tolerance) = ($actual, $expected, $tolerance);
        let distance = $crate::test_support::Distance::distance(&actual, &expected);
        proptest::prop_assert!(
            distance < tolerance,
            "{} = {:?}\n  is not within {:e} of {} = {:?}: distance {:e}{}",
            stringify!($actual),
            actual,
            tolerance,
            stringify!($expected),
            expected,
            distance,
            $context,
        );
    }};
    ($actual:expr, $expected:expr, $tolerance:expr $(,)?) => {
        $crate::test_support::prop_assert_close!(@check $actual, $expected, $tolerance, "")
    };
    ($actual:expr, $expected:expr, $tolerance:expr, $($context:tt)+) => {
        $crate::test_support::prop_assert_close!(
            @check $actual,
            $expected,
            $tolerance,
            format_args!(" ({})", format_args!($($context)+))
        )
    };
}
pub(crate) use prop_assert_close;

#[test]
fn distance_separates_collections_of_different_lengths() {
    let short = vec![1.0];
    assert!(short.distance(&vec![1.0, 0.0]).is_infinite());
    assert!(Vec::<f64>::new().distance(&short).is_infinite());
    assert!((vec![3.0, 0.0].distance(&vec![0.0, 4.0]) - 5.0).abs() < 1e-15);
}

/// The floats of a value as bit patterns, for results that must repeat bit for bit
/// rather than to a tolerance.
pub(crate) trait Bits {
    /// Append the bit pattern of every float in `self` to `out`, in order, with the
    /// real part of a complex value before its imaginary part.
    fn bits(&self, out: &mut Vec<u64>);
}

impl Bits for f64 {
    fn bits(&self, out: &mut Vec<u64>) {
        out.push(self.to_bits());
    }
}

impl Bits for Complex {
    fn bits(&self, out: &mut Vec<u64>) {
        out.extend([self.re.to_bits(), self.im.to_bits()]);
    }
}

impl<T: Bits, const N: usize> Bits for [T; N] {
    fn bits(&self, out: &mut Vec<u64>) {
        self.as_slice().bits(out);
    }
}

impl<T: Bits> Bits for [T] {
    fn bits(&self, out: &mut Vec<u64>) {
        for value in self {
            value.bits(out);
        }
    }
}

impl<T: Bits> Bits for Vec<T> {
    fn bits(&self, out: &mut Vec<u64>) {
        self.as_slice().bits(out);
    }
}

impl<T: Bits + nalgebra::Scalar> Bits for DMatrix<T> {
    fn bits(&self, out: &mut Vec<u64>) {
        self.as_slice().bits(out);
    }
}

/// The bit patterns of `values` in order, such as the fields of a gradient.
pub(crate) fn bits(values: &[&dyn Bits]) -> Vec<u64> {
    let mut out = Vec::new();
    for value in values {
        value.bits(&mut out);
    }
    out
}

/// Run `run` on Rayon pools of one to four threads and assert that it returns the same
/// nonempty bits on each.
///
/// The parallel regions of the crate run in place on such a pool and take its size as
/// the thread count: one thread takes the calling-thread paths, and more threads split
/// the work as a budget of that size does. `run` should do the whole computation, the
/// forward with the pullback, so that every parallel region runs on the pool.
pub(crate) fn assert_same_bits_on_pools(run: impl Fn() -> Vec<u64> + Sync) {
    let runs = [1, 2, 3, 4].map(|threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let bits = pool.install(|| {
            assert_eq!(crate::threads::current_num_threads(), threads);
            run()
        });
        (threads, bits)
    });
    let (_, reference) = &runs[0];
    assert_ne!(reference.as_slice(), []);
    for (threads, bits) in &runs {
        assert_eq!(bits.len(), reference.len(), "{threads} threads");
        let first = bits.iter().zip(reference).position(|(a, b)| a != b);
        assert_eq!(
            first, None,
            "the bits of {threads} threads and one thread differ"
        );
    }
}

/// Rows of a `key: values` text table, each split at whitespace and commas into
/// key and value fields; blank lines and `#` comment lines are skipped.
///
/// Two kinds of files use this format:
/// - Lean golden files (`formal/golden/`) hold integer rows such as
///   `2: 0,0 1,-1 1,0 1,1 2,-2 2,-1 2,0 2,1 2,2` (the `degree,order` pairs up to
///   degree 2) or `1 1 true: -1 1`.
/// - Reference tables (`references/`) hold arguments and the real and imaginary
///   parts of high-precision values below a `#` header that names the columns, such
///   as `-8 -60.0 3: 10910888627.348331 7471819238.136416` for `Γ(-8, -60 + 3i)` in
///   the `n re(z) im(z): re im` rows of `incgamma.txt`.
///
/// Panics with the offending line when a row has no `:` or a field does not parse.
pub(crate) fn table<K, V>(text: &str) -> Vec<(Vec<K>, Vec<V>)>
where
    K: FromStr<Err: Debug>,
    V: FromStr<Err: Debug>,
{
    fn fields<T: FromStr<Err: Debug>>(fields: &str, line: &str) -> Vec<T> {
        fields
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|field| !field.is_empty())
            .map(|field| {
                field
                    .parse()
                    .unwrap_or_else(|error| panic!("{line}: field {field}: {error:?}"))
            })
            .collect()
    }
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (key, values) = line
                .split_once(':')
                .unwrap_or_else(|| panic!("table row without `key:`: {line}"));
            (fields(key, line), fields(values, line))
        })
        .collect()
}
