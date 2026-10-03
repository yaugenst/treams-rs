//! Real-degree Ferrers functions of the first kind with integer order.
//!
//! Upstream: `treams.special.lpmv` at a non-integer degree and a real argument, which
//! calls scipy's `lpmv`.
//!
//! [`ferrers_real_degree`] evaluates `P_ν^m(x)` and `dP_ν^m/dx` for a non-integer degree
//! `0 < ν < 128`, an integer order `|m| <= 130` and `-1 < x <= 1`, with the conventions of
//! NIST DLMF §14.3 (Ferrers functions "on the cut", Condon–Shortley phase included for
//! `m > 0`). It implements the DLMF formulas cited below.
//!
//! # Method
//!
//! Write `μ = |m|`, `n = ⌊ν⌋`, `f = ν - n`, `y = |x|`, `w = 1 - y² = sin²θ` and
//! `t = s = (1 - y)/2`. The functions are computed for the order `-μ`; positive orders follow
//! from DLMF 14.9.3, `P_ν^μ = (-1)^μ Γ(ν+μ+1)/Γ(ν-μ+1) P_ν^{-μ}`, with the gamma ratio formed as
//! `∏_{i=1}^{μ} (ν(ν + 1) - i(i - 1))` from an exactly split `ν(ν + 1)`.
//!
//! * **About `x = 1`** (`x >= 0`). DLMF 14.3.1 with Euler's transformation (DLMF 15.8.1) gives
//!   `P_ν^{-μ}(y) = w^{μ/2}/(2^μ μ!) F(μ - ν, μ + ν + 1; μ + 1; t)`. The series is summed at a
//!   start degree `ν₀ ≡ ν (mod 1)`, `μ <= ν₀ <= ν`, as large as its alternating leading terms
//!   allow, and carried to `ν` by the degree recurrence DLMF 14.10.3 written for `P_ν` and
//!   `D_ν = P_{ν-1} - yP_ν`: `P_{ν+1} = yP_ν - c_ν D_ν`, `D_{ν+1} = wP_ν + y c_ν D_ν`,
//!   `c_ν = (ν - μ)/(ν + μ + 1)`. This form never subtracts nearly equal values near `y = 1`,
//!   and increasing the degree is stable: the regular solution dominates where
//!   `(ν + 1/2) sinθ < μ`, and both solutions oscillate elsewhere. `D_ν₀` comes from the same
//!   series; the derivative is DLMF 14.10.5, `w P' = (ν - μ) D_ν - μ y P_ν`.
//! * **About `x = 0`** (small degrees, `(ν + 1/2)|x|` small). `G = (1 - x²)^{-μ/2} P_ν^{-μ}`
//!   solves `(1 - x²) G'' - 2(μ + 1) x G' + (ν - μ)(ν + μ + 1) G = 0` (from DLMF 14.2.2), so its
//!   Taylor coefficients satisfy a two-term recurrence started from the values at zero,
//!   DLMF 14.5.1 and 14.5.2, with `1/Γ` from a Taylor polynomial and DLMF 5.5.1.
//! * **About `x = -1`** (`x < 0`). The connection formula DLMF 14.9.10,
//!   `P_ν^{-μ}(-y) = (-1)^μ [cos(νπ) P_ν^{-μ}(y) + sin(νπ) S_ν^{-μ}(y)]` with `S = -(2/π) Q`,
//!   separates the part regular at `x = -1` from the singular one. `sin(νπ)` and `cos(νπ)` come
//!   from the exactly reduced degree `ν - round(ν)`, so degrees within an ulp of an integer keep
//!   their exact offset. Both parts follow from the degenerate (logarithmic) case `c = a + b + μ`
//!   of the `1 - z` connection formulas of DLMF §15.8 applied to DLMF 14.3.1: in powers of `s`
//!   with digamma coefficients, plus a finite sum of `μ` terms for `μ >= 1`. The digamma
//!   reflection DLMF 5.5.4 turns `ψ` at non-positive arguments into `π cot(νπ)`, whose pole is
//!   cancelled analytically against the vanishing factor `n - ν` of the series coefficients. The
//!   order-zero expansion gives `Q_ν(y) = ½ Σ_k (ν+1)_k (-ν)_k/(k!)² [2ψ(k+1) - ψ(ν+k+1) -
//!   ψ(ν-k+1) - ln s] s^k`. These expansions are summed directly when `(ν + 1/2) θ` is small.
//!   Otherwise the order-zero pair is summed at a lower degree and carried to `ν` by the
//!   recurrence above; the regular part of order `-μ` comes from the series about `x = 1`, and
//!   the singular part from `S^1 = -w S^0'/sinθ` and the order recurrence DLMF 14.10.1,
//!   `S^{k+1} = -2k cotθ S^k - (ν - k + 1)(ν + k) S^{k-1}`, which is stable upwards because
//!   `Q_ν^μ(y)` dominates `P_ν^μ(y)` as `μ` grows for `y > 0`; its derivative uses
//!   `w S^μ' = (ν + μ)(ν - μ + 1) sinθ S^{μ-1} + μ y S^μ` (§14.10).
//! * The digamma function uses DLMF 5.5.2 and the asymptotic expansion DLMF 5.11.2.
//! * Quantities that can leave the double range carry a separate binary exponent, so only a final
//!   value or derivative beyond the double range is reported as an error. Sums that share the
//!   low-order bits of `f` are formed without rounding them as a whole, so their rounding errors
//!   do not accumulate systematically. At `x = 1` the exact limits are returned.
use std::f64::consts::PI;

use crate::{Error, MAX_DEGREE, Result, numerics::EULER};

// 128 is MAX_DEGREE and 130 is MAX_REAL_DEGREE_ORDER.
const INVALID: &str = "noninteger real degree in (0,128], |order|<=130 and -1<x<=1 required";
const NONFINITE: &str = "non-finite real-degree Legendre result or derivative";

/// The largest `|m|` of [`ferrers_real_degree`], two above its largest degree. Orders
/// above the degree are valid; the angular functions pass `|m| <= degree`.
const MAX_REAL_DEGREE_ORDER: i32 = MAX_DEGREE + 2;

/// Relative truncation tolerance for the series (`2^-54`).
const TOLERANCE: f64 = f64::EPSILON / 4.0;
/// Safety bound on series terms; the series converge geometrically for `t, s <= 1/2`.
const MAX_TERMS: u32 = 20_000;
/// Rescaling threshold `2^512` for recurrences that could leave the double range.
const BIG: f64 = 1.340_780_792_994_259_7e154;
const BIG_EXPONENT: i32 = 512;
/// Largest leading-ratio size `(ν₀ - μ)(ν₀ + μ + 1) t / (μ + 1)` of a regular start series.
const REGULAR_REACH: f64 = 0.5;
/// Largest `2 (ν₀ + 1/2) √s ≈ (ν₀ + 1/2) θ` at which the logarithmic series is summed.
const SINGULAR_REACH: f64 = 3.0;
const SQRT_PI: f64 = 1.772_453_850_905_516;
/// The expansion about `x = 0` is used for `(ν + 1/2)|x| <= CENTRAL_REACH` (well conditioned)
/// and `|x| <=` [`CENTRAL_LIMIT`] (or [`CENTRAL_LIMIT_POSITIVE`] for `x > 0`, where the
/// series about `x = 1` is cheaper), for degrees and orders that keep `1/Γ` cheap.
const CENTRAL_REACH: f64 = 2.5;
const CENTRAL_LIMIT: f64 = 0.5;
const CENTRAL_LIMIT_POSITIVE: f64 = 0.25;
const CENTRAL_DEGREE: f64 = 16.0;
const CENTRAL_ORDER: i32 = 20;

/// Correctly rounded `1/k!` for `k = 0..=130` ([`MAX_REAL_DEGREE_ORDER`]).
const INV_FACTORIAL: [f64; 131] = [
    1.0e0,
    1.0e0,
    5.0e-1,
    1.666_666_666_666_666_6e-1,
    4.166_666_666_666_666_4e-2,
    8.333_333_333_333_333e-3,
    1.388_888_888_888_889e-3,
    1.984_126_984_126_984e-4,
    2.480_158_730_158_73e-5,
    2.755_731_922_398_589_3e-6,
    2.755_731_922_398_589e-7,
    2.505_210_838_544_172e-8,
    2.087_675_698_786_81e-9,
    1.605_904_383_682_161_3e-10,
    1.147_074_559_772_972_5e-11,
    7.647_163_731_819_816e-13,
    4.779_477_332_387_385e-14,
    2.811_457_254_345_520_6e-15,
    1.561_920_696_858_622_5e-16,
    8.220_635_246_624_33e-18,
    4.110_317_623_312_165e-19,
    1.957_294_106_339_126_3e-20,
    8.896_791_392_450_574e-22,
    3.868_170_170_630_684e-23,
    1.611_737_571_096_118_4e-24,
    6.446_950_284_384_474e-26,
    2.479_596_263_224_797_6e-27,
    9.183_689_863_795_546e-29,
    3.279_889_237_069_838e-30,
    1.130_996_288_644_771_6e-31,
    3.769_987_628_815_905_4e-33,
    1.216_125_041_553_518e-34,
    3.800_390_754_854_743_4e-36,
    1.151_633_562_077_195e-37,
    3.387_157_535_521_162e-39,
    9.677_592_958_631_89e-41,
    2.688_220_266_286_636_3e-42,
    7.265_460_179_153_071e-44,
    1.911_963_205_040_282e-45,
    4.902_469_756_513_544e-47,
    1.225_617_439_128_385_8e-48,
    2.989_310_827_142_404_6e-50,
    7.117_406_731_291_439e-52,
    1.655_210_867_742_195_1e-53,
    3.761_842_881_232_261_6e-55,
    8.359_650_847_182_804e-57,
    1.817_315_401_561_479e-58,
    3.866_628_513_960_594e-60,
    8.055_476_070_751_236e-62,
    1.643_974_708_316_579e-63,
    3.287_949_416_633_158e-65,
    6.446_959_640_457_172e-67,
    1.239_799_930_857_148_6e-68,
    2.339_245_152_560_657_6e-70,
    4.331_935_467_704_922e-72,
    7.876_246_304_918_039e-74,
    1.406_472_554_449_649_8e-75,
    2.467_495_709_560_789_3e-77,
    4.254_302_947_518_602e-79,
    7.210_682_961_895_936_5e-81,
    1.201_780_493_649_322_6e-82,
    1.970_131_956_802_168_2e-84,
    3.177_632_188_390_594_2e-86,
    5.043_860_616_493_007e-88,
    7.881_032_213_270_323e-90,
    1.212_466_494_349_280_4e-91,
    1.837_070_445_983_758_1e-93,
    2.741_896_188_035_46e-95,
    4.032_200_276_522_735_3e-97,
    5.843_768_516_699_616e-99,
    8.348_240_738_142_31e-101,
    1.175_808_554_667_930_8e-102,
    1.633_067_437_038_793e-104,
    2.237_078_680_875_058_7e-106,
    3.023_079_298_479_809e-108,
    4.030_772_397_973_079e-110,
    5.303_647_892_069_84e-112,
    6.887_854_405_285_506e-114,
    8.830_582_570_878_855e-116,
    1.117_795_262_136_564e-117,
    1.397_244_077_670_705e-119,
    1.724_992_688_482_351_7e-121,
    2.103_649_620_100_429e-123,
    2.534_517_614_578_83e-125,
    3.017_282_874_498_607_3e-127,
    3.549_744_558_233_655_4e-129,
    4.127_609_951_434_483e-131,
    4.744_379_254_522_394_6e-133,
    5.391_340_061_957_266_6e-135,
    6.057_685_462_873_333e-137,
    6.730_761_625_414_815e-139,
    7.396_441_346_609_687e-141,
    8.039_610_159_358_355e-143,
    8.644_742_106_836_94e-145,
    9.196_534_156_209_511e-147,
    9.680_562_269_694_223e-149,
    1.008_391_903_093_148_2e-150,
    1.039_579_281_539_328e-152,
    1.060_795_185_244_212_2e-154,
    1.071_510_288_125_467e-156,
    1.071_510_288_125_467e-158,
    1.060_901_275_371_749_4e-160,
    1.040_099_289_580_146_5e-162,
    1.009_805_135_514_705_3e-164,
    9.709_664_764_564_474e-167,
    9.247_299_775_775_69e-169,
    8.723_867_712_995_935e-171,
    8.153_147_395_323_303e-173,
    7.549_210_551_225_28e-175,
    6.925_881_239_656_22e-177,
    6.296_255_672_414_745e-179,
    5.672_302_407_580_852e-181,
    5.064_555_721_054_332e-183,
    4.481_907_717_747_196_5e-185,
    3.931_497_998_023_856_7e-187,
    3.418_693_911_325_092_7e-189,
    2.947_149_923_556_114_4e-191,
    2.518_931_558_594_969_7e-193,
    2.134_687_761_521_160_7e-195,
    1.793_855_261_782_488_2e-197,
    1.494_879_384_818_74e-199,
    1.235_437_508_114_661_2e-201,
    1.012_653_695_175_951_7e-203,
    8.232_956_871_349_201e-206,
    6.639_481_347_862_259e-208,
    5.311_585_078_289_807e-210,
    4.215_543_712_928_419e-212,
    3.319_325_758_211_353e-214,
    2.593_223_248_602_619_6e-216,
    2.010_250_580_312_108_2e-218,
    1.546_346_600_240_083_3e-220,
];

/// Taylor coefficients of `1/Γ(3/2 + h)` about `h = 0` (enough for `|h| <= 1/2`).
const RGAMMA_TAYLOR: [f64; 21] = [
    std::f64::consts::FRAC_2_SQRT_PI, // 1/Γ(3/2)
    -4.117_452_644_528_31e-2,
    -5.266_544_355_255_445e-1,
    1.751_020_260_439_345_7e-1,
    5.096_686_024_770_607_4e-2,
    -4.215_516_936_853_560_4e-2,
    6.612_897_826_824_127e-3,
    2.120_731_442_572_938e-3,
    -1.110_730_254_594_890_6e-3,
    1.523_576_207_674_768_8e-4,
    2.535_520_492_381_416_5e-5,
    -1.389_680_571_791_375_6e-5,
    2.156_203_290_514_172_4e-6,
    5.794_264_054_052_672_6e-8,
    -8.913_551_118_311_116e-8,
    1.710_346_941_591_537_4e-8,
    -9.313_686_445_241_901e-10,
    -2.680_474_103_349_662_3e-10,
    7.458_932_233_316_326e-11,
    -8.012_807_061_414_718e-12,
    -8.382_343_033_451_855e-14,
];

/// `mantissa · 2^exponent` with `1/2 <= |mantissa| < 1` (or zero): a double with a wider
/// exponent range.
#[derive(Clone, Copy, Debug)]
struct Scaled {
    mantissa: f64,
    exponent: i32,
}

impl Scaled {
    fn new(value: f64, exponent: i32) -> Self {
        let (mantissa, shift) = frexp(value);
        Self {
            mantissa,
            exponent: exponent + shift,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.mantissa * other.mantissa,
            self.exponent + other.exponent,
        )
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.mantissa * factor, self.exponent)
    }

    fn add(self, other: Self) -> Self {
        if other.mantissa == 0.0 {
            return self;
        }
        if self.mantissa == 0.0 {
            return other;
        }
        let (high, low) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let aligned = scalbn(low.mantissa, low.exponent - high.exponent);
        Self::new(high.mantissa + aligned, high.exponent)
    }

    fn to_f64(self) -> f64 {
        scalbn(self.mantissa, self.exponent)
    }
}

const EXPONENT_MASK: u64 = 0x7ff << 52;

/// `value = mantissa · 2^exponent` with `1/2 <= |mantissa| < 1`; inline fast path for normal
/// numbers, `libm` otherwise.
#[inline]
fn frexp(value: f64) -> (f64, i32) {
    let bits = value.to_bits();
    let biased = (bits & EXPONENT_MASK) >> 52;
    if biased == 0 || biased == 0x7ff {
        return libm::frexp(value);
    }
    let mantissa = f64::from_bits((bits & !EXPONENT_MASK) | (0x3fe << 52));
    // biased < 2^11, so the conversion is exact.
    (
        mantissa,
        i32::from(u16::try_from(biased).unwrap_or(0)) - 1022,
    )
}

/// `value · 2^exponent`, correctly rounded; inline fast path when `2^exponent` is a normal number
/// and `value` is normalised as in [`Scaled`].
#[inline]
fn scalbn(value: f64, exponent: i32) -> f64 {
    if (-1000..=1000).contains(&exponent) {
        let power = f64::from_bits(u64::from((exponent + 1023).unsigned_abs()) << 52);
        let result = value * power;
        if result == 0.0 || result.abs() >= f64::MIN_POSITIVE {
            return result;
        }
    }
    libm::scalbn(value, exponent)
}

fn nonfinite() -> Error {
    Error::SpecialFunction(NONFINITE.into())
}

/// `value` rounded towards zero, for `|value| < 2^31` (larger magnitudes saturate). Callers use
/// it on validated degrees and small shifts; unlike `f64::floor` it needs no library call.
#[allow(clippy::cast_possible_truncation)] // Truncation is the intent; the range is bounded.
fn small_integer(value: f64) -> i32 {
    value as i32
}

/// `⌊value⌋` for `0 <= value < 2^31`.
fn floor_small(value: f64) -> f64 {
    f64::from(small_integer(value))
}

/// The integer nearest to `0 <= value < 2^31` (halves rounded up).
fn nearest_integer(value: f64) -> i32 {
    let whole = small_integer(value);
    if value - f64::from(whole) >= 0.5 {
        whole + 1
    } else {
        whole
    }
}

/// Real-degree Ferrers function `P_v^m(x)` (degree `v` not an integer) and, with
/// `DERIVATIVE`, its analytic derivative in `x`; without it the second element is `0.0`.
///
/// Upstream: `treams.special.lpmv(m, v, x)` at a non-integer degree. Differences:
/// degrees above 128, orders above 130 and `x` outside `(-1, 1]` give
/// [`Error::InvalidInput`], and results beyond the double range give
/// [`Error::SpecialFunction`], as does the infinite derivative at `x = 1` for `|m| = 1`.
pub(crate) fn ferrers_real_degree<const DERIVATIVE: bool>(
    v: f64,
    m: i32,
    x: f64,
) -> Result<(f64, f64)> {
    if !(v > 0.0 && v < f64::from(MAX_DEGREE))
        || v - floor_small(v) == 0.0
        || m.unsigned_abs() > MAX_REAL_DEGREE_ORDER.unsigned_abs()
        || !(x > -1.0 && x <= 1.0)
    {
        return Err(Error::InvalidInput(INVALID.into()));
    }
    if x >= 1.0 {
        return endpoint::<DERIVATIVE>(v, m);
    }
    let order = m.abs();
    let y = x.abs();
    let central_limit = if x > 0.0 {
        CENTRAL_LIMIT_POSITIVE
    } else {
        CENTRAL_LIMIT
    };
    let (value, slope) = if v < CENTRAL_DEGREE
        && order <= CENTRAL_ORDER
        && y <= central_limit
        && (v + 0.5) * y <= CENTRAL_REACH
    {
        central::<DERIVATIVE>(v, order, x)?
    } else if x >= 0.0 {
        regular(v, order, y)?
    } else {
        return reflected::<DERIVATIVE>(v, m, y);
    };
    if m > 0 {
        let ratio = gamma_ratio(v, order, None).scale(if order % 2 == 1 { -1.0 } else { 1.0 });
        let slope = if DERIVATIVE { slope.mul(ratio) } else { slope };
        return finish::<DERIVATIVE>(value.mul(ratio), slope, y);
    }
    finish::<DERIVATIVE>(value, slope, y)
}

/// Pairs of arguments a few ulps apart, one on each side of a switch between the
/// evaluation routes of [`ferrers_real_degree`] at degree `v` and order `m`: the edges of
/// the expansion about `x = 0`, or `x = 0` between the series about `x = 1` and the
/// connection formula, and the argument where `(ν + 1/2) θ` reaches [`SINGULAR_REACH`],
/// beyond which the logarithmic series is summed at a lower degree.
#[cfg(test)]
pub(crate) fn ferrers_route_switches(v: f64, m: i32) -> Vec<[f64; 2]> {
    let around = |x: f64| {
        [
            x * (1.0 - 4.0 * f64::EPSILON),
            x * (1.0 + 4.0 * f64::EPSILON),
        ]
    };
    let mut switches = Vec::new();
    if v < CENTRAL_DEGREE && m.abs() <= CENTRAL_ORDER {
        let reach = CENTRAL_REACH / (v + 0.5);
        switches.push(around(CENTRAL_LIMIT_POSITIVE.min(reach)));
        switches.push(around(-CENTRAL_LIMIT.min(reach)));
    } else {
        switches.push([0.0, -f64::MIN_POSITIVE]);
    }
    let s = (0.5 * SINGULAR_REACH / (v + 0.5)).powi(2);
    if s < 0.5 {
        switches.push(around(2.0f64.mul_add(s, -1.0)));
    }
    switches
}

/// `P_ν^m(-y)` for `0 < y < 1` by the connection formula DLMF 14.9.10.
fn reflected<const DERIVATIVE: bool>(v: f64, m: i32, y: f64) -> Result<(f64, f64)> {
    let order = m.abs();
    let odd = order % 2 == 1;
    let (sin, cos) = sin_cos_pi(v);
    // f π cot(πν) with f = ν - ⌊ν⌋, finite as f → 0.
    let frac = v - floor_small(v);
    let frac_cot = if frac < 1e-4 {
        pi_cot_scaled(frac)
    } else {
        frac * PI * cos / sin
    };
    if order > 0
        && f64::from(order) <= floor_small(v)
        && v <= SINGULAR_REACH / (2.0 * (0.5 * (1.0 - y)).sqrt()) - 0.5
    {
        return order_series::<DERIVATIVE>(v, m, y, sin, cos, frac_cot);
    }
    let slopes = match order {
        _ if DERIVATIVE => Slopes::Both,
        0 => Slopes::None,
        1 => Slopes::Both,
        _ => Slopes::Singular,
    };
    let zero = order_zero(v, y, frac_cot, slopes)?;
    let ((regular, regular_slope), (singular, singular_slope)) = match order {
        0 => (
            (
                Scaled::new(zero.regular, 0),
                Scaled::new(zero.regular_slope, 0),
            ),
            (
                Scaled::new(zero.singular, 0),
                Scaled::new(zero.singular_slope, 0),
            ),
        ),
        // DLMF 14.10 order steps from order zero: w P' = ν(ν+1) sinθ P^{-1} and
        // w (P^{-1})' = y P^{-1} - sinθ P.
        1 if v > 1.0 => {
            let (w, w_low) = one_minus_square(y);
            let sine = w.sqrt();
            let sine = sine + 0.5 * w_low / sine;
            let lowered = zero.regular_slope / (v * (v + 1.0) * sine);
            (
                (
                    Scaled::new(lowered, 0),
                    Scaled::new(y * lowered - sine * zero.regular, 0),
                ),
                raise_singular(v, order, y, &zero),
            )
        }
        _ => (regular(v, order, y)?, raise_singular(v, order, y, &zero)),
    };
    let (first, second) = if m > 0 {
        (
            gamma_ratio(v, order, None).scale(cos),
            Scaled::new(if odd { -sin } else { sin }, 0),
        )
    } else {
        (
            Scaled::new(if odd { -cos } else { cos }, 0),
            sine_over_gamma_ratio(v, order, sin),
        )
    };
    let value = regular.mul(first).add(singular.mul(second));
    let slope = if DERIVATIVE {
        regular_slope
            .mul(first)
            .add(singular_slope.mul(second))
            .scale(-1.0)
    } else {
        value
    };
    finish::<DERIVATIVE>(value, slope, y)
}

/// `P_ν^m(-y)` for `0 < y < 1`, `1 <= μ = |m| <= ⌊ν⌋`, from the expansion about `x = -1` in
/// `s = (1 - y)/2`: the degenerate case `c - a - b = μ` of the `1 - z` connection formulas
/// (DLMF §15.8) applied to DLMF 14.3.1, after the digamma reflection DLMF 5.5.4,
/// `P_ν^{-μ}(-y) = (-1)^μ cos(νπ) R + sin(νπ) (L + F/G)` with `G = Γ(ν+μ+1)/Γ(ν-μ+1)`,
/// `R = (w/4)^{μ/2}/μ! Σ r_k` (that is `P_ν^{-μ}(y)`),
/// `L = (-1)^μ (w/4)^{μ/2}/(π μ!) Σ r_k [ln s - ψ(k+1) - ψ(k+μ+1) + ψ(ν+μ+k+1) + ψ(ν-μ-k+1)]`,
/// `F = (-1)^{μ+1} (μ-1)!/π (w/4)^{μ/2} s^{-μ} Σ_{k<μ} (ν+1)_k (-ν)_k s^k/(k! (1-μ)_k)` and
/// `r_k = (ν+μ+1)_k (μ-ν)_k s^k/((μ+1)_k k!)`. As in [`log_series`], the pole of
/// `ψ(ν - μ - k + 1)` is cancelled against the factor `n - ν` of `r_k` (`n = ⌊ν⌋`).
fn order_series<const DERIVATIVE: bool>(
    v: f64,
    m: i32,
    y: f64,
    sin: f64,
    cos: f64,
    frac_cot: f64,
) -> Result<(f64, f64)> {
    let order = m.abs();
    let mu = f64::from(order);
    let s = 0.5 * (1.0 - y);
    let (w, w_low) = one_minus_square(y);
    // Finite sum Σ_{k<μ} f_k and Σ k f_k.
    let (mut finite, mut finite_slope, mut term) = (1.0, 0.0, 1.0);
    for k in 1..order {
        let k = f64::from(k);
        term *= (v + k) * (k - 1.0 - v) * s / (k * (k - mu));
        finite += term;
        finite_slope += k * term;
    }
    // Logarithmic part: Σ r_k, Σ k r_k, Σ r_k g_k, Σ (k r_k g_k + r_k).
    let sums = order_log_sums::<DERIVATIVE>(v, order, s, frac_cot)?;
    let (regular, regular_slope, log, log_slope) =
        (sums.regular, sums.regular_slope, sums.log, sums.log_slope);
    // Assemble with (w/4)^{μ/2}, 1/μ!, (μ-1)!/π, s^{-μ} and the gamma ratio as scaled numbers.
    let inverse_factorial = usize::try_from(order)
        .ok()
        .and_then(|index| INV_FACTORIAL.get(index).zip(INV_FACTORIAL.get(index - 1)))
        .ok_or_else(nonfinite)?;
    let base = sine_power(w, w_low, order).mul(Scaled::new(*inverse_factorial.0, -order));
    let odd = order % 2 == 1;
    let sign = if odd { -1.0 } else { 1.0 };
    let regular_part = base;
    let log_part = base.scale(sign / PI);
    let (s_mantissa, s_exponent) = frexp(s);
    // (w/4)^{μ/2} (μ - 1)! = base μ! (μ - 1)!.
    let finite_part = base
        .mul(Scaled::new(-sign / (PI * *inverse_factorial.0), 0))
        .mul(Scaled::new(1.0 / *inverse_factorial.1, 0))
        .mul(Scaled::new(s_mantissa.powi(-order), -order * s_exponent));
    let ratio = gamma_ratio(v, order, None);
    let (regular_part, log_part, finite_part) = if m > 0 {
        (
            regular_part.mul(ratio).scale(cos),
            log_part.mul(ratio).scale(sign * sin),
            finite_part.scale(sign * sin),
        )
    } else {
        let inverse = Scaled::new(1.0 / ratio.mantissa, -ratio.exponent);
        (
            regular_part.scale(sign * cos),
            log_part.scale(sin),
            finite_part.mul(inverse).scale(sin),
        )
    };
    let value = regular_part
        .scale(regular)
        .add(log_part.scale(log))
        .add(finite_part.scale(finite));
    if !DERIVATIVE {
        return finish::<false>(value, value, y);
    }
    // (1 - x²) d/dx = 2 s (1 - s) d/ds.
    let tilt = mu * (1.0 - 2.0 * s);
    let slope = regular_part
        .scale(tilt * regular + 2.0 * (1.0 - s) * regular_slope)
        .add(log_part.scale(tilt * log + 2.0 * (1.0 - s) * log_slope))
        .add(finite_part.scale(2.0 * (1.0 - s) * finite_slope - mu * finite));
    finish::<DERIVATIVE>(value, slope, y)
}

/// Sums of [`order_series`]: `Σ r_k`, `Σ k r_k`, `Σ r_k g_k` and `Σ (k r_k g_k + r_k)`.
#[derive(Clone, Copy, Debug, Default)]
struct OrderSums {
    regular: f64,
    regular_slope: f64,
    log: f64,
    log_slope: f64,
    values: Convergence,
    slopes: Convergence,
}

impl OrderSums {
    /// Adds the term `k` (`term = r_k g_k`, `plain = r_k`); returns true once converged, given
    /// that later weight ratios stay below `ratio()`. Values are frozen once converged.
    #[inline]
    fn add<const SLOPE: bool>(
        &mut self,
        k: f64,
        term: f64,
        plain: f64,
        ratio: impl Fn() -> f64,
    ) -> bool {
        if !self.values.done {
            self.regular += plain;
            self.log += term;
            self.values.update(term.abs() + plain.abs(), &ratio);
        }
        if SLOPE {
            let slope_term = k * term + plain;
            self.regular_slope += k * plain;
            self.log_slope += slope_term;
            self.slopes
                .update(slope_term.abs() + k * plain.abs(), &ratio);
            return self.values.done && self.slopes.done;
        }
        self.values.done
    }
}

/// The logarithmic sums of [`order_series`] with `n - μ = pivot`, in three phases like
/// [`log_series`]: direct terms for `k <= n - μ`, the cancelled pole at `k = n - μ + 1`, and the
/// reflected digamma beyond. One division per term serves all reciprocals.
fn order_log_sums<const SLOPE: bool>(
    v: f64,
    order: i32,
    s: f64,
    frac_cot: f64,
) -> Result<OrderSums> {
    let mu = f64::from(order);
    let whole = floor_small(v);
    let frac = v - whole;
    let pivot = whole - mu;
    let ratio_after = |k: f64| {
        let bound = (v + mu + k + 1.0) * ((mu - v + k).abs() + 1.0) / ((mu + k + 1.0) * (k + 1.0));
        s * bound.max(1.0) * (1.0 + 2.0 / (k + 1.0))
    };
    let mut sums = OrderSums::default();
    // bracket = ln s - ψ(k+1) - ψ(k+μ+1) + ψ(ν+μ+k+1) + ψ(ν-μ-k+1).
    // For low orders ψ(ν+μ+1) = ψ(ν-μ+1) + Σ_{i=1}^{2μ} 1/(ν-μ+i) and ψ(μ+1) = H_μ - γ
    // (DLMF 5.5.2) are cheaper than further digamma evaluations.
    let psi_low = digamma(1.0 + v - mu);
    let (psi_high, psi_order) = if mu <= 4.0 {
        (
            psi_low + shifted_sum(v - mu, 2 * order),
            shifted_sum(0.0, order) - EULER,
        )
    } else {
        (digamma(v + mu + 1.0), digamma(mu + 1.0))
    };
    let mut bracket = s.ln() + EULER - psi_order + psi_high + psi_low;
    let mut weight = 1.0; // r_k
    let mut k = 0.0;
    let mut count = 0;
    loop {
        if sums.add::<SLOPE>(k, weight * bracket, weight, || ratio_after(k)) {
            return Ok(sums);
        }
        if k >= pivot {
            break;
        }
        // Step to k + 1, with ψ(ν-μ-k+1) decreasing by 1/(ν-μ-k).
        let next = k + 1.0;
        let (upper, shifted, gap) = (v + mu + next, mu + next, v - mu - k);
        let (upper_next, shifted_gap) = (upper * next, shifted * gap);
        let reciprocal = 1.0 / (upper_next * shifted_gap);
        weight *= (mu - v + k) * upper * upper * gap * s * reciprocal;
        bracket +=
            (next * shifted_gap - upper * shifted_gap - upper_next * gap - upper_next * shifted)
                * reciprocal;
        k = next;
        count += 1;
        if count > MAX_TERMS {
            return Err(nonfinite());
        }
    }
    // k = n - μ + 1: r_k = -f r̂_k; f ψ(f) = f ψ(1 + f) - 1, where ψ(1 + f) is the last ψ(ν-μ-k+1).
    let next = k + 1.0;
    let (upper, shifted) = (v + mu + next, mu + next);
    weight *= upper * s / (shifted * next);
    bracket += 1.0 / upper - 1.0 / next - 1.0 / shifted;
    k = next;
    // Here the bracket holds ψ(1 + f) + the other digammas; split ψ(1 + f) off again for f ψ(f).
    if sums.add::<SLOPE>(k, weight * (1.0 - frac * bracket), -frac * weight, || {
        ratio_after(k)
    }) {
        return Ok(sums);
    }
    // k >= n - μ + 2: ψ(ν-μ-k+1) = ψ(k+μ-ν) - π cot(πν), starting from ψ(2 - f).
    let next = k + 1.0;
    let (upper, shifted) = (v + mu + next, mu + next);
    weight *= (mu - v + k) * upper * s / (shifted * next);
    bracket += 1.0 / upper - 1.0 / next - 1.0 / shifted + digamma_reflection_step(frac, frac_cot);
    k = next;
    loop {
        if sums.add::<SLOPE>(
            k,
            weight * (frac_cot - frac * bracket),
            -frac * weight,
            || ratio_after(k),
        ) {
            return Ok(sums);
        }
        let next = k + 1.0;
        let (upper, shifted, gap) = (v + mu + next, mu + next, k + mu - v);
        let (upper_next, shifted_gap) = (upper * next, shifted * gap);
        let reciprocal = 1.0 / (upper_next * shifted_gap);
        weight *= (mu - v + k) * upper * upper * gap * s * reciprocal;
        bracket += (next * shifted_gap - upper * shifted_gap - upper_next * gap
            + upper_next * shifted)
            * reciprocal;
        k = next;
        count += 1;
        if count > MAX_TERMS {
            return Err(nonfinite());
        }
    }
}

/// Converts `(P, (1 - x²) P')` to doubles, rejecting results beyond the double range.
fn finish<const DERIVATIVE: bool>(value: Scaled, slope: Scaled, y: f64) -> Result<(f64, f64)> {
    let value = value.to_f64();
    if !value.is_finite() {
        return Err(nonfinite());
    }
    if !DERIVATIVE {
        return Ok((value, 0.0));
    }
    let (w, _) = one_minus_square(y);
    let derivative = Scaled::new(slope.mantissa / w, slope.exponent).to_f64();
    if !derivative.is_finite() {
        return Err(nonfinite());
    }
    Ok((value, derivative))
}

/// Exact limits at `x = 1`: `P_ν^m(1) = δ_{m0}`; the slope is `ν(ν+1)/2` (`m = 0`), infinite
/// (`|m| = 1`), `-Γ(ν+3)/(4Γ(ν-1))` (`m = 2`), `-1/4` (`m = -2`) and zero otherwise.
fn endpoint<const DERIVATIVE: bool>(v: f64, m: i32) -> Result<(f64, f64)> {
    let value = if m == 0 { 1.0 } else { 0.0 };
    if !DERIVATIVE {
        return Ok((value, 0.0));
    }
    let derivative = match m {
        0 => 0.5 * v * (v + 1.0),
        1 | -1 => return Err(nonfinite()),
        2 => -0.25 * gamma_ratio(v, 2, None).to_f64(),
        -2 => -0.25,
        _ => 0.0,
    };
    Ok((value, derivative))
}

/// `(P_ν^{-μ}(y), (1 - y²) dP_ν^{-μ}/dy)` for `0 <= y < 1`.
fn regular(v: f64, order: i32, y: f64) -> Result<(Scaled, Scaled)> {
    let mu = f64::from(order);
    let whole = floor_small(v);
    let frac = v - whole;
    let t = 0.5 * (1.0 - y);
    let (w, w_low) = one_minus_square(y);
    let start = if mu >= whole {
        v
    } else {
        // Largest lift j with (j + frac)(2μ + 1 + j + frac) t <= REGULAR_REACH (μ + 1).
        let base = 2.0 * mu + 1.0;
        let reach = 0.5 * ((base * base + 4.0 * REGULAR_REACH * (mu + 1.0) / t).sqrt() - base);
        let lift = floor_small((reach - frac).clamp(0.0, whole - mu));
        (mu + lift) + frac
    };
    // E_k = (a + 1)_{k-1} (b)_k t^k / ((c)_k k!): P = pref (1 + a Σ E_k), D = 2(1 - t) pref Σ k E_k.
    // b + k = (b_whole + k) + frac is never rounded as a whole: all those sums share the low bits
    // of frac, so their rounding errors would not average out.
    let b_whole = mu + 1.0 + (start - frac);
    let (a, b, c) = (mu - start, b_whole + frac, mu + 1.0);
    let mut term = (b_whole * t + frac * t) / c;
    let (mut sum, mut weighted, mut size) = (term, term, term.abs());
    let mut k = 1.0;
    let mut count = 0;
    loop {
        let rising = a + k;
        term *= (rising * (b_whole + k) + rising * frac) * t / ((c + k) * (k + 1.0));
        k += 1.0;
        sum += term;
        let current = k * term;
        weighted += current;
        size += current.abs();
        if a + k > 0.0 && current.abs() <= TOLERANCE * size {
            // Every later ratio of k E_k is positive and below rho.
            let rho = t * (b + k) * (k + 1.0) / ((c + k) * k);
            if rho < 1.0 && current.abs() * rho <= TOLERANCE * size * (1.0 - rho) {
                break;
            }
        }
        count += 1;
        if count > MAX_TERMS {
            return Err(nonfinite());
        }
    }
    let mut p = 1.0 + a * sum;
    let mut d = 2.0 * (1.0 - t) * weighted;
    let mut exponent = 0;
    let mut degree = start;
    for _ in 0..small_integer(v - start) {
        let coupling = (degree - mu) / (degree + mu + 1.0);
        let next = y * p - coupling * d;
        d = w * p + (w_low * p + y * coupling * d);
        p = next;
        degree += 1.0;
        if p.abs().max(d.abs()) > BIG {
            p = scalbn(p, -BIG_EXPONENT);
            d = scalbn(d, -BIG_EXPONENT);
            exponent += BIG_EXPONENT;
        }
    }
    let slope = (v - mu) * d - mu * y * p;
    if order == 0 {
        return Ok((Scaled::new(p, exponent), Scaled::new(slope, exponent)));
    }
    let inverse_factorial = usize::try_from(order)
        .ok()
        .and_then(|index| INV_FACTORIAL.get(index))
        .ok_or_else(nonfinite)?;
    let prefactor = sine_power(w, w_low, order).mul(Scaled::new(*inverse_factorial, -order));
    Ok((
        prefactor.mul(Scaled::new(p, exponent)),
        prefactor.mul(Scaled::new(slope, exponent)),
    ))
}

/// `(P_ν^{-μ}(x), (1 - x²) dP_ν^{-μ}/dx)` from the expansion about `x = 0`.
///
/// `P_ν^{-μ}(x) = (1 - x²)^{μ/2} G(x)`, where `G` solves
/// `(1 - x²) G'' - 2(μ + 1) x G' + (ν - μ)(ν + μ + 1) G = 0`, so its Taylor coefficients obey
/// `g_{j+2} = g_j (j + μ - ν)(j + μ + ν + 1)/((j + 1)(j + 2))`; `g_0` and `g_1` are the value
/// and slope at zero (DLMF 14.5.1, 14.5.2). With `X = x²`, `G = E(X) + x O(X)`.
fn central<const SLOPE: bool>(v: f64, order: i32, x: f64) -> Result<(Scaled, Scaled)> {
    let mu = f64::from(order);
    let scale = scalbn(SQRT_PI, -order);
    let mut even = scale * rgamma(f64::midpoint(v, mu) + 1.0) * rgamma(f64::midpoint(1.0 - v, mu));
    let mut odd = -2.0 * scale * rgamma(f64::midpoint(v + mu, 1.0)) * rgamma(f64::midpoint(mu, -v));
    let square = x * x;
    let magnitude_x = x.abs();
    // Sums of a_i X^i, i a_i X^(i-1) (even) and b_i X^i, i b_i X^(i-1) (odd). The value and the
    // slope converge separately (either may be tiny next to the other); the value is frozen once
    // converged, so it does not depend on SLOPE.
    let (mut sum_even, mut sum_odd) = (even, odd);
    // The slope also needs the odd sum, to its own accuracy.
    let (mut slope_even, mut slope_odd, mut slope_sum_odd) = (0.0, 0.0, odd);
    let mut values = Convergence {
        size: even.abs() + magnitude_x * odd.abs(),
        ..Convergence::default()
    };
    let mut slopes = Convergence {
        size: odd.abs(),
        done: !SLOPE,
        ..Convergence::default()
    };
    // Running j + μ - ν, j + μ + ν + 1 and j + 1 for the current even index j = 2(i - 1).
    let (mut low, mut high, mut first) = (mu - v, mu + v + 1.0, 1.0);
    let mut index = 1.0;
    let mut count = 0;
    while !(values.done && slopes.done) {
        let reciprocal = 1.0 / (first * (first + 1.0) * (first + 2.0));
        let ratio_even = low * high * (first + 2.0) * reciprocal;
        let ratio_odd = (low + 1.0) * (high + 1.0) * first * reciprocal;
        let partial_even = even * ratio_even;
        let partial_odd = odd * ratio_odd;
        even = partial_even * square;
        odd = partial_odd * square;
        // Later ratios (times X) stay below rho.
        let rho = || square * ratio_even.abs().max(ratio_odd.abs()).max(1.0) * (1.0 + 2.0 / index);
        if !values.done {
            sum_even += even;
            sum_odd += odd;
            values.update(even.abs() + magnitude_x * odd.abs(), rho);
        }
        if !slopes.done {
            slope_even += index * partial_even;
            slope_odd += index * partial_odd;
            slope_sum_odd += odd;
            slopes.update(
                (index + 1.0) * (2.0 * magnitude_x * partial_even.abs() + partial_odd.abs()),
                rho,
            );
        }
        low += 2.0;
        high += 2.0;
        first += 2.0;
        index += 1.0;
        count += 1;
        if count > MAX_TERMS {
            return Err(nonfinite());
        }
    }
    let (w, w_low) = one_minus_square(x);
    let value = sum_even + x * sum_odd;
    let slope = if SLOPE {
        w * (2.0 * x * slope_even + slope_sum_odd + 2.0 * square * slope_odd) - mu * x * value
    } else {
        0.0
    };
    if order == 0 {
        return Ok((Scaled::new(value, 0), Scaled::new(slope, 0)));
    }
    let prefactor = sine_power(w, w_low, order);
    Ok((
        prefactor.mul(Scaled::new(value, 0)),
        prefactor.mul(Scaled::new(slope, 0)),
    ))
}

/// `1/Γ(z)` for moderate real `z`: DLMF 5.5.1 shifts into `[1, 2)`, then a Taylor polynomial.
/// The shifts multiply by `z + k` exactly where `Γ` has poles, so those zeros are exact.
fn rgamma(z: f64) -> f64 {
    let truncated = small_integer(z);
    let floor = if f64::from(truncated) > z {
        truncated - 1
    } else {
        truncated
    };
    let (mut z, mut numerator, mut denominator) = (z, 1.0, 1.0);
    for _ in floor..1 {
        numerator *= z;
        z += 1.0;
    }
    for _ in 2..=floor {
        z -= 1.0;
        denominator *= z;
    }
    numerator * rgamma_core(z - 1.5) / denominator
}

/// `1/Γ(3/2 + h)` for `|h| <= 1/2` by Estrin's scheme (short dependency chains).
fn rgamma_core(h: f64) -> f64 {
    let [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13,
        c14,
        c15,
        c16,
        c17,
        c18,
        c19,
        c20,
    ] = RGAMMA_TAYLOR;
    let h2 = h * h;
    let h4 = h2 * h2;
    let h8 = h4 * h4;
    let quad = |a: f64, b: f64, c: f64, d: f64| (a + b * h) + (c + d * h) * h2;
    let low = (quad(c0, c1, c2, c3) + quad(c4, c5, c6, c7) * h4)
        + (quad(c8, c9, c10, c11) + quad(c12, c13, c14, c15) * h4) * h8;
    let high = quad(c16, c17, c18, c19) + c20 * h4;
    low + high * (h8 * h8)
}

/// `w^{μ/2}` for `1 - y² = w + w_low` (`0 < w <= 1`).
fn sine_power(w: f64, w_low: f64, order: i32) -> Scaled {
    let half = 0.5 * f64::from(order);
    // Low orders by at most two roundings, with the first-order correction for w_low.
    let correction = 1.0 + half * (w_low / w);
    match order {
        0 => return Scaled::new(1.0, 0),
        1 => return Scaled::new(w.sqrt() * correction, 0),
        2 => return Scaled::new(w + w_low, 0),
        3 => return Scaled::new(w * w.sqrt() * correction, 0),
        4 => return Scaled::new(w * w * correction, 0),
        _ => {}
    }
    if w > 0.75 {
        // w - 1 is exact here.
        return Scaled::new((half * ((w - 1.0) + w_low).ln_1p()).exp(), 0);
    }
    let (mut mantissa, mut exponent) = frexp(w);
    if exponent % 2 != 0 {
        mantissa *= 2.0;
        exponent -= 1;
    }
    Scaled::new(mantissa.powf(half) * correction, exponent / 2 * order)
}

/// Order-zero functions at `0 < y < 1`, with `w = 1 - y²`: `P_ν(y)`, `w P_ν'(y)`, and
/// `S_ν(y) = -(2/π) Q_ν(y)` (Ferrers second kind), `w S_ν'(y)`.
#[derive(Clone, Copy, Debug)]
struct OrderZero {
    regular: f64,
    regular_slope: f64,
    singular: f64,
    singular_slope: f64,
}

/// Which order-zero slopes a caller needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slopes {
    None,
    Singular,
    Both,
}

/// [`OrderZero`] at degree `ν`: the logarithmic expansion at the largest degree `ν₀ ≡ ν`
/// (mod 1) with `(ν₀ + 1/2) θ` small, then the degree recurrence (`μ = 0`) for both solutions.
/// Only the requested slopes are formed, unless the recurrence needs both.
fn order_zero(v: f64, y: f64, frac_cot: f64, slopes: Slopes) -> Result<OrderZero> {
    let s = 0.5 * (1.0 - y);
    let frac = v - floor_small(v);
    let reach = SINGULAR_REACH / (2.0 * s.sqrt()) - 0.5;
    let start = if v <= reach {
        v
    } else {
        // Step down by ⌈ν - reach⌉ degrees, but not below 1 + f.
        let steps = floor_small(v - reach);
        let steps = if steps < v - reach {
            steps + 1.0
        } else {
            steps
        };
        (v - steps).max((1.0 + frac).min(v))
    };
    if start >= v {
        return match slopes {
            Slopes::None => log_series::<false, false>(v, y, frac_cot),
            Slopes::Singular => log_series::<true, false>(v, y, frac_cot),
            Slopes::Both => log_series::<true, true>(v, y, frac_cot),
        };
    }
    let zero = log_series::<true, true>(start, y, frac_cot)?;
    let (w, w_low) = one_minus_square(y);
    let (mut p, mut q) = (zero.regular, zero.singular);
    let (mut dp, mut dq) = (zero.regular_slope / start, zero.singular_slope / start);
    let mut degree = start;
    for _ in 0..small_integer(v - start) {
        let coupling = degree / (degree + 1.0);
        let next_p = y * p - coupling * dp;
        let next_q = y * q - coupling * dq;
        dp = w * p + (w_low * p + y * coupling * dp);
        dq = w * q + (w_low * q + y * coupling * dq);
        p = next_p;
        q = next_q;
        degree += 1.0;
    }
    Ok(OrderZero {
        regular: p,
        regular_slope: v * dp,
        singular: q,
        singular_slope: v * dq,
    })
}

/// `(S_ν^μ(y), (1 - y²) dS_ν^μ/dy)` for `μ >= 1` from order zero: `S^1 = -w S^0'/sinθ`, then the
/// order recurrence DLMF 14.10.1 upwards.
fn raise_singular(v: f64, order: i32, y: f64, zero: &OrderZero) -> (Scaled, Scaled) {
    let (w, w_low) = one_minus_square(y);
    let sine = w.sqrt();
    let sine = sine + 0.5 * w_low / sine;
    let cotangent = y / sine;
    let (mut previous, mut current) = (zero.singular, -zero.singular_slope / sine);
    let mut exponent = 0;
    for k in 1..order {
        let k = f64::from(k);
        let next = -2.0 * k * cotangent * current - (v - k + 1.0) * (v + k) * previous;
        previous = current;
        current = next;
        if current.abs() > BIG {
            previous = scalbn(previous, -BIG_EXPONENT);
            current = scalbn(current, -BIG_EXPONENT);
            exponent += BIG_EXPONENT;
        }
    }
    let mu = f64::from(order);
    let slope = (v + mu) * (v - mu + 1.0) * sine * previous + mu * y * current;
    (Scaled::new(current, exponent), Scaled::new(slope, exponent))
}

/// Convergence test of one series: the term must be negligible twice in a row and the tail,
/// bounded geometrically with ratio `rho`, as well. Once reached, it stays reached.
#[derive(Clone, Copy, Debug, Default)]
struct Convergence {
    size: f64,
    small_before: bool,
    done: bool,
}

impl Convergence {
    #[inline]
    fn update(&mut self, magnitude: f64, rho: impl Fn() -> f64) {
        if self.done {
            return;
        }
        self.size += magnitude;
        let small = magnitude <= TOLERANCE * self.size;
        if small && self.small_before {
            let rho = rho();
            self.done = rho < 1.0 && magnitude * rho <= TOLERANCE * self.size * (1.0 - rho);
        }
        self.small_before = small;
    }
}

/// Running sums of the logarithmic expansion: `Σ q_k (c_k - ln s)`, `Σ q_k`, and, as requested,
/// `Σ (k q_k (c_k - ln s) - q_k)` (`SINGULAR`) and `Σ k q_k` (`REGULAR`). The values and each
/// slope converge separately (for tiny `ν` the regular slope is tiny next to everything else);
/// the values are frozen once converged, so they do not depend on which slopes are summed.
#[derive(Clone, Copy, Debug, Default)]
struct Sums<const SINGULAR: bool, const REGULAR: bool> {
    singular: f64,
    regular: f64,
    singular_slope: f64,
    regular_slope: f64,
    values: Convergence,
    singular_slopes: Convergence,
    regular_slopes: Convergence,
}

impl<const SINGULAR: bool, const REGULAR: bool> Sums<SINGULAR, REGULAR> {
    /// Adds the term `k` (`term = q_k (c_k - ln s)`, `plain = q_k`); returns true once the
    /// remaining tails are negligible, given that later weight ratios stay below `ratio()`.
    #[inline]
    fn add(&mut self, k: f64, term: f64, plain: f64, ratio: impl Fn() -> f64) -> bool {
        // The bracket c_k - ln s grows only logarithmically; allow for it and for the k.
        let rho = || ratio() * (1.0 + 2.0 / (k + 1.0));
        if !self.values.done {
            self.singular += term;
            self.regular += plain;
            self.values.update(term.abs() + plain.abs(), rho);
        }
        if SINGULAR {
            let slope_term = k * term - plain;
            self.singular_slope += slope_term;
            self.singular_slopes.update(slope_term.abs(), rho);
        }
        if REGULAR {
            self.regular_slope += k * plain;
            self.regular_slopes.update(k * plain.abs(), rho);
        }
        self.values.done
            && (!SINGULAR || self.singular_slopes.done)
            && (!REGULAR || self.regular_slopes.done)
    }

    fn finish(&self, s: f64) -> OrderZero {
        OrderZero {
            regular: self.regular,
            regular_slope: -2.0 * (1.0 - s) * self.regular_slope,
            singular: -self.singular / PI,
            singular_slope: 2.0 / PI * (1.0 - s) * self.singular_slope,
        }
    }
}

/// [`OrderZero`] from the logarithmic expansion about `y = 1` in `s = (1 - y)/2`:
/// `P_ν = Σ q_k`, `Q_ν = ½ Σ q_k (c_k - ln s)` with `q_k = (ν+1)_k (-ν)_k s^k/(k!)²` and
/// `c_k = 2ψ(k+1) - ψ(ν+k+1) - ψ(ν+1-k)`, with the slopes selected as in [`Sums`].
///
/// From `k = n + 1` (`n = ⌊ν⌋`) on, `q_k` carries the factor `n - ν = -f`, which is split off
/// (`q_k = -f q̂_k`) and cancelled analytically against the pole of `ψ(ν + 1 - k)`:
/// `f ψ(f) = f ψ(1 + f) - 1` at `k = n + 1`, and DLMF 5.5.4, `ψ(ν+1-k) = ψ(k-ν) - π cot(πν)`,
/// beyond (`frac_cot = f π cot(πν)`). The digamma updates (DLMF 5.5.2) share one division per
/// term.
fn log_series<const SINGULAR: bool, const REGULAR: bool>(
    v: f64,
    y: f64,
    frac_cot: f64,
) -> Result<OrderZero> {
    let whole = floor_small(v);
    let frac = v - whole;
    let s = 0.5 * (1.0 - y);
    // Bound on all later weight ratios |q_{k+1}/q_k| (their limit is s).
    let ratio_after = |k: f64| {
        let bound = (v + k + 1.0) * ((v - k).abs() + 1.0) / ((k + 1.0) * (k + 1.0));
        s * bound.max(1.0)
    };
    let mut sums = Sums::<SINGULAR, REGULAR>::default();
    let psi = digamma(v + 1.0);
    let mut regular_part = -2.0 * EULER - psi - s.ln(); // 2ψ(k+1) - ψ(ν+k+1) - ln s
    let mut psi_down = psi; // ψ(ν + 1 - k)
    let mut weight = 1.0; // q_k
    let mut k = 0.0;
    let mut count = 0;
    // Terms k <= n.
    loop {
        if sums.add(k, weight * (regular_part - psi_down), weight, || {
            ratio_after(k)
        }) {
            return Ok(sums.finish(s));
        }
        if k >= whole {
            break;
        }
        let next = k + 1.0;
        let (up, gap) = (v + next, v - k);
        let reciprocal = 1.0 / (next * up * gap);
        let inv_next = up * gap * reciprocal;
        weight *= -(up * gap) * s * inv_next * inv_next;
        regular_part += (2.0 * up - next) * gap * reciprocal;
        psi_down -= next * up * reciprocal;
        k = next;
        count += 1;
        if count > MAX_TERMS {
            return Err(nonfinite());
        }
    }
    // k = n + 1: q̂_k = q_k/(-f); f ψ(f) = f ψ(1 + f) - 1 with ψ(1 + f) = psi_down.
    let next = k + 1.0;
    let up = v + next;
    let square = next * next;
    weight *= up * s / square;
    regular_part += (2.0 * up - next) / (next * up);
    k = next;
    let term = weight * (frac * (psi_down - regular_part) - 1.0);
    if sums.add(k, term, -frac * weight, || ratio_after(k)) {
        return Ok(sums.finish(s));
    }
    // k >= n + 2: ψ(ν + 1 - k) = ψ(k - ν) - π cot(πν).
    let next = k + 1.0;
    let up = v + next;
    weight *= up * (k - v) * s / (next * next);
    regular_part += (2.0 * up - next) / (next * up);
    k = next;
    // ψ(k - ν) - (2ψ(k+1) - ψ(ν+k+1) - ln s), with ψ(2 - f) = ψ(1 + f) + the reflection step.
    let mut bracket = psi_down + digamma_reflection_step(frac, frac_cot) - regular_part;
    loop {
        let term = weight * (frac * bracket - frac_cot);
        if sums.add(k, term, -frac * weight, || ratio_after(k)) {
            return Ok(sums.finish(s));
        }
        let next = k + 1.0;
        let (up, gap) = (v + next, k - v);
        let reciprocal = 1.0 / (next * up * gap);
        let inv_next = up * gap * reciprocal;
        weight *= up * gap * s * inv_next * inv_next;
        bracket += (next * up - (2.0 * up - next) * gap) * reciprocal;
        k = next;
        count += 1;
        if count > MAX_TERMS {
            return Err(nonfinite());
        }
    }
}

/// `ψ(2 - f) - ψ(1 + f)`. For `f <= 1/2` it is `π cot(πf) - 1/f + 1/(1 - f)` (DLMF 5.5.2,
/// 5.5.4) from `frac_cot = f π cot(πf)`, with an absolute error of about `ε/f` that the callers'
/// factor `f` removes; otherwise two digamma values.
fn digamma_reflection_step(frac: f64, frac_cot: f64) -> f64 {
    if frac <= 0.5 {
        (frac_cot - 1.0) / frac + 1.0 / (1.0 - frac)
    } else {
        digamma(2.0 - frac) - digamma(1.0 + frac)
    }
}

/// `Σ_{i=1}^{count} 1/(start + i)` for a small integer `count`, as one fraction.
fn shifted_sum(start: f64, count: i32) -> f64 {
    let (mut numerator, mut denominator) = (0.0, 1.0);
    for i in 1..=count {
        let factor = start + f64::from(i);
        numerator = numerator * factor + denominator;
        denominator *= factor;
    }
    numerator / denominator
}

/// `πr cot(πr)` for `|r| <= 1/2`, accurate (and finite) as `r → 0`.
fn pi_cot_scaled(r: f64) -> f64 {
    let z = PI * r;
    if z.abs() < 1e-4 {
        let z2 = z * z;
        1.0 - z2 / 3.0 - z2 * z2 / 45.0
    } else {
        z / z.tan()
    }
}

/// Digamma for `x >= 1`: DLMF 5.5.2 up to `x >= 10`, then DLMF 5.11.2.
fn digamma(mut x: f64) -> f64 {
    // Σ 1/(x + i) over the shifts, accumulated as one fraction.
    let (mut numerator, mut denominator) = (0.0, 1.0);
    let shifts = if x < 10.0 {
        small_integer(10.0 - x) + 1
    } else {
        0
    };
    for _ in 0..shifts {
        numerator = numerator * x + denominator;
        denominator *= x;
        x += 1.0;
    }
    let shift = numerator / denominator;
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let tail = inv2
        * (1.0 / 12.0
            - inv2
                * (1.0 / 120.0
                    - inv2
                        * (1.0 / 252.0
                            - inv2
                                * (1.0 / 240.0
                                    - inv2
                                        * (1.0 / 132.0
                                            - inv2 * (691.0 / 32760.0 - inv2 / 12.0))))));
    x.ln() - 0.5 * inv - tail - shift
}

/// `(sin(νπ), cos(νπ))` from the exactly reduced degree `ν - round(ν)`.
fn sin_cos_pi(v: f64) -> (f64, f64) {
    let nearest = nearest_integer(v);
    let (sin, cos) = (PI * (v - f64::from(nearest))).sin_cos();
    if nearest % 2 == 0 {
        (sin, cos)
    } else {
        (-sin, -cos)
    }
}

/// Knuth's error-free sum: `a + b = sum + error` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    (sum, (a - a_virtual) + (b - b_virtual))
}

/// Dekker's exact square: `v² = square + error`.
fn exact_square(v: f64) -> (f64, f64) {
    let split = 134_217_729.0 * v; // 2^27 + 1
    let head = split - (split - v);
    let tail = v - head;
    let square = v * v;
    (
        square,
        ((head * head - square) + 2.0 * head * tail) + tail * tail,
    )
}

/// `1 - y²` as an unevaluated sum `high + low`, accurate far beyond a double.
fn one_minus_square(y: f64) -> (f64, f64) {
    let (square, square_error) = exact_square(y);
    let (difference, error) = two_sum(1.0, -square);
    let low = error - square_error;
    let high = difference + low;
    (high, low - (high - difference))
}

/// `ν(ν + 1)` as an unevaluated sum `high + low`.
fn degree_product(v: f64) -> (f64, f64) {
    let (square, square_error) = exact_square(v);
    let (high, error) = two_sum(square, v);
    (high, error + square_error)
}

/// `Γ(ν + μ + 1)/Γ(ν - μ + 1) = ∏_{i=1}^{μ} (ν + i)(ν + 1 - i)`; with `skip = Some(i)` the
/// factor `ν + 1 - i` of that pair is left out.
fn gamma_ratio(v: f64, order: i32, skip: Option<i32>) -> Scaled {
    let (high, low) = degree_product(v);
    let mut product = 1.0;
    let mut exponent = 0;
    for i in 1..=order {
        let index = f64::from(i);
        if skip == Some(i) {
            product *= v + index;
        } else {
            // Multiply by the unrounded factor: a rounded one would carry the same error (from
            // the fractional bits of ν(ν + 1)) for every i of a binade, and these would add up.
            let (difference, error) = two_sum(high, -(index * (index - 1.0)));
            product = product * difference + product * (error + low);
        }
        if i % 32 == 0 {
            let (mantissa, shift) = frexp(product);
            product = mantissa;
            exponent += shift;
        }
    }
    Scaled::new(product, exponent)
}

/// `sin(νπ) Γ(ν - μ + 1)/Γ(ν + μ + 1)`; outside the caller domain (`μ > round(ν)`) the
/// vanishing factor `ν - round(ν)` of the gamma ratio is cancelled against `sin(νπ)`.
fn sine_over_gamma_ratio(v: f64, order: i32, sin: f64) -> Scaled {
    let nearest = nearest_integer(v);
    if nearest + 1 > order {
        let ratio = gamma_ratio(v, order, None);
        return Scaled::new(sin / ratio.mantissa, -ratio.exponent);
    }
    let ratio = gamma_ratio(v, order, Some(nearest + 1));
    let z = PI * (v - f64::from(nearest));
    let sinc = if z.abs() < 1e-4 {
        PI * (1.0 - z * z / 6.0)
    } else {
        PI * z.sin() / z
    };
    let sign = if nearest % 2 == 0 { 1.0 } else { -1.0 };
    Scaled::new(sign * sinc / ratio.mantissa, -ratio.exponent)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // Tests compare exact endpoint values.
    use super::*;

    /// `(ν, m, x, P, dP/dx)` from 30-digit references.
    const REFERENCES: [(f64, i32, f64, f64, f64); 13] = [
        (
            2.3,
            1,
            0.5,
            -1.158_877_916_558_258_6,
            -3.105_224_571_364_395,
        ),
        (
            2.3,
            -2,
            0.1,
            8.210_878_443_624_224e-2,
            5.272_898_021_258_5e-2,
        ),
        (
            0.2,
            0,
            -0.999_999,
            -1.797_687_281_513_591_6,
            1.870_981_884_756_791e5,
        ),
        (
            10.000_000_000_000_002,
            6,
            -0.9,
            7.664_247_046_048_816e4,
            1.738_989_837_866_624_4e6,
        ),
        (
            127.2,
            64,
            0.999_999,
            2.133_285_121_993_467_3e-24,
            -6.826_489_043_723_42e-17,
        ),
        (
            65.2,
            -64,
            -0.9999,
            1.336_260_716_230_590_1e6,
            -4.275_790_332_015_145e11,
        ),
        // A large order left of centre, and a degree 1e-13 above an integer next to -1.
        (
            127.9,
            121,
            -0.34,
            2.108_817_946_242_377_7e242,
            1.073_952_115_051_059e243,
        ),
        (
            127.000_000_000_000_1,
            -63,
            -0.999,
            -4.423_236_554_341_923e-87,
            1.388_205_387_672_740_3e-82,
        ),
        (
            120.6,
            0,
            1.0 - f64::EPSILON / 2.0,
            9.999_999_999_991_86e-1,
            7.332_479_999_997_015_5e3,
        ),
        (
            0.5,
            0,
            0.0,
            5.393_526_011_883_794e-1,
            5.901_702_995_080_481e-1,
        ),
        (
            127.2,
            127,
            0.0,
            -4.160_241_594_472_013e250,
            -1.949_011_429_617_312_4e251,
        ),
        (
            127.2,
            -127,
            -0.5,
            -3.491_950_613_042_738_7e-246,
            2.869_573_962_247_748_2e-244,
        ),
        (2.3, 0, 1.0, 1.0, 3.794_999_999_999_999_5),
    ];

    #[test]
    fn matches_high_precision_references() {
        for (v, m, x, value, derivative) in REFERENCES {
            let (p, q) = ferrers_real_degree::<true>(v, m, x).unwrap();
            let (p_false, zero) = ferrers_real_degree::<false>(v, m, x).unwrap();
            assert!(
                (p - value).abs() <= 1e-13 * value.abs(),
                "{v} {m} {x}: {p} vs {value}"
            );
            assert!(
                (q - derivative).abs() <= 1e-13 * derivative.abs(),
                "{v} {m} {x}: {q} vs {derivative}"
            );
            assert_eq!(p_false.to_bits(), p.to_bits());
            assert_eq!(zero.to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn endpoint_limits_are_exact() {
        assert_eq!(
            ferrers_real_degree::<true>(2.3, 0, 1.0).unwrap(),
            (1.0, 0.5 * 2.3 * 3.3)
        );
        assert_eq!(
            ferrers_real_degree::<true>(2.3, -2, 1.0).unwrap(),
            (0.0, -0.25)
        );
        assert_eq!(
            ferrers_real_degree::<true>(2.3, 7, 1.0).unwrap(),
            (0.0, 0.0)
        );
        let (value, derivative) = ferrers_real_degree::<true>(2.3, 2, 1.0).unwrap();
        assert_eq!(value, 0.0);
        assert!((derivative + 10.607_025).abs() < 1e-13 * 10.607_025);
        assert_eq!(
            ferrers_real_degree::<false>(2.3, 1, 1.0).unwrap(),
            (0.0, 0.0)
        );
        for m in [1, -1] {
            let error = ferrers_real_degree::<true>(2.3, m, 1.0).unwrap_err();
            assert!(
                matches!(&error, Error::SpecialFunction(message) if message.contains("non-finite"))
            );
        }
    }

    #[test]
    fn rejects_inputs_outside_the_domain() {
        let nan = f64::NAN;
        let cases = [
            (0.0, 0, 0.5),
            (-0.0, 0, 0.5),
            (2.0, 0, 0.5),
            (128.0, 0, 0.5),
            (128.5, 0, 0.5),
            (-0.5, 0, 0.5),
            (nan, 0, 0.5),
            (f64::INFINITY, 0, 0.5),
            (2.5, 131, 0.5),
            (2.5, i32::MIN, 0.5),
            (2.5, 0, -1.0),
            (2.5, 0, 1.000_000_1),
            (2.5, 0, nan),
        ];
        for (v, m, x) in cases {
            assert!(
                matches!(
                    ferrers_real_degree::<false>(v, m, x),
                    Err(Error::InvalidInput(_))
                ),
                "{v} {m} {x}"
            );
            assert!(
                matches!(
                    ferrers_real_degree::<true>(v, m, x),
                    Err(Error::InvalidInput(_))
                ),
                "{v} {m} {x}"
            );
        }
    }

    #[test]
    fn tiny_degrees_and_overflow() {
        let (value, derivative) = ferrers_real_degree::<true>(5e-324, 0, -0.8).unwrap();
        assert_eq!(value, 1.0);
        assert!(derivative.abs() < f64::MIN_POSITIVE);
        // P_127.2^126(-0.999999) ≈ 3.2e605 overflows; the negative order ≈ 2.29e105 does not.
        for result in [
            ferrers_real_degree::<false>(127.2, 126, -0.999_999),
            ferrers_real_degree::<true>(127.2, 126, -0.999_999),
        ] {
            assert!(
                matches!(result, Err(Error::SpecialFunction(message)) if message.contains("non-finite"))
            );
        }
        let (value, _) = ferrers_real_degree::<true>(127.2, -126, -0.999_999).unwrap();
        assert!((value / 2.286_871_2e105 - 1.0).abs() < 1e-6);
    }

    /// Degree recurrence DLMF 14.10.3 and derivative DLMF 14.10.5 across all evaluation routes,
    /// including degrees next to integers and arguments next to -1.
    #[test]
    fn recurrences_hold_across_routes() {
        let degrees: [f64; 8] = [
            1.3,
            2.000_000_001,
            2.999_999_999,
            7.5,
            20.25,
            41.000_000_000_01,
            63.7,
            125.9,
        ];
        let arguments = [
            -0.999_999_9,
            -0.99,
            -0.7,
            -0.3,
            -0.05,
            0.0,
            0.05,
            0.3,
            0.7,
            0.99,
            0.999_999_9,
        ];
        let mut checked = 0;
        for v in degrees {
            let top = small_integer(v.floor()) - 1;
            for m in [0, 1, -1, 2, -top / 2, top / 2, -top, top] {
                for x in arguments {
                    let results = [
                        ferrers_real_degree::<true>(v, m, x),
                        ferrers_real_degree::<false>(v - 1.0, m, x),
                        ferrers_real_degree::<false>(v + 1.0, m, x),
                    ];
                    // Results beyond the double range (large orders next to x = -1) are errors.
                    let [Ok((p, derivative)), Ok((lower, _)), Ok((upper, _))] = results else {
                        assert!(results.iter().all(|result| matches!(
                            result,
                            Ok(_) | Err(Error::SpecialFunction(_))
                        )));
                        continue;
                    };
                    let order = f64::from(m);
                    // Values near the underflow threshold only carry absolute accuracy.
                    let scale = (v - order + 1.0).abs() * upper.abs()
                        + (2.0 * v + 1.0) * p.abs()
                        + (v + order).abs() * lower.abs()
                        + 1e-288;
                    let recurrence =
                        (v - order + 1.0) * upper - (2.0 * v + 1.0) * x * p + (v + order) * lower;
                    assert!(
                        recurrence.abs() <= 1e-12 * scale,
                        "recurrence {v} {m} {x}: {recurrence} / {scale}"
                    );
                    checked += 1;
                    let expected = (v + order) * lower - v * x * p;
                    let deviation = (1.0 - x) * (1.0 + x) * derivative - expected;
                    let slope_scale = (v + order).abs() * lower.abs()
                        + v * p.abs()
                        + (1.0 - x) * (1.0 + x) * derivative.abs()
                        + 1e-288;
                    assert!(
                        deviation.abs() <= 1e-12 * slope_scale,
                        "derivative {v} {m} {x}: {deviation} / {slope_scale}"
                    );
                }
            }
        }
        assert!(
            checked > 600,
            "only {checked} combinations were representable"
        );
    }
}
