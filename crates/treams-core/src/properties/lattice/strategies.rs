//! Strategies that draw Ewald sums for the lattice properties.

use std::f64::consts::{PI, TAU};

use proptest::prelude::*;

use super::{
    checks::fails_with,
    ewald::{Ewald, c, cw, cylinder_point, embed, explicit_split, leading, sw, triangular},
};
use crate::{
    Complex,
    lattice::{self, SumPart, derivatives_part, resolve_split},
    test_support::{degree_order, log_uniform},
};

/// Largest `|k| rho` of the shifts `check_chain` compares with the plain Ewald sum at
/// the split `1.5 / (|k| rho)`, which keeps that split at 0.4 or more.
const ACCURATE_SPLIT_REACH: f64 = 3.75;

/// A 1D spherical sum `rho = distance * period` (at most `ACCURATE_SPLIT_REACH / |k|`)
/// off the axis at `azimuth` and `along` periods along it, at the automatic split.
pub(super) fn near_chain(
    (l, m): (i32, i32),
    k: Complex,
    period: f64,
    kpar: f64,
    (distance, azimuth, along): (f64, f64, f64),
) -> Ewald {
    let rho = (distance * period).min(ACCURATE_SPLIT_REACH / k.norm());
    let r = cylinder_point(rho, azimuth, along * period);
    Ewald::chain(sw(l, m), period, kpar, r, k, Complex::default())
}

/// Signed distances from the lattice plane or axis for [`check_tiny_normal_shift`]:
/// zero, subnormal ones, and 1e-308 to 1e-20.
///
/// [`check_tiny_normal_shift`]: super::checks::check_tiny_normal_shift
pub(super) fn tiny_distance() -> impl Strategy<Value = f64> {
    (
        prop_oneof![
            Just(0.0),
            log_uniform(-323.3..-308.0),
            log_uniform(-308.0..-20.0),
        ],
        any::<bool>(),
    )
        .prop_map(|(distance, negative)| if negative { -distance } else { distance })
}

/// 1D spherical sums 2.2 to 6 periods off the axis with the split `sqrt(2 pi) / (k L)`,
/// the automatic one for `|k| L < 8` (`w` from 5.5), where they take their spectral
/// series; `l, |m| < 13`.
pub(super) fn far_chain() -> impl Strategy<Value = Ewald> {
    (
        degree_order(0..13),
        1.0_f64..2.0,
        (0.5_f64..4.0, prop_oneof![Just(0.0), 0.05_f64..0.6]),
        -0.5_f64..0.5,
        (2.2_f64..6.0, -3.2_f64..3.2, -0.5_f64..0.5),
    )
        .prop_map(
            |((l, m), period, (kr, ki), kpar, (distance, azimuth, along))| {
                let k = c(kr, ki);
                let r = cylinder_point(distance * period, azimuth, along * period);
                Ewald::chain(
                    sw(l, m),
                    period,
                    kpar * TAU / period,
                    r,
                    k,
                    TAU.sqrt() / (k * period),
                )
            },
        )
}

/// Slopes `Im k / Re k` of the wavenumber: real, lossy or amplifying up to a quarter,
/// and nearly real of either sign down to 1e-16. Amplifying cases test rejection;
/// retaining the strategy also preserves replay of its regression seeds.
pub(super) fn slope() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        -0.25_f64..0.25,
        (log_uniform(-16.0..-3.0), any::<bool>()).prop_map(|(size, sign)| if sign {
            size
        } else {
            -size
        }),
    ]
}

/// 1D spherical sums, `l, |m| < 9`, 0.1 to 3 periods off the axis, with explicit
/// splits (`w` from 0.01 to 40); `k` with a `slope`, and Bloch vectors that include 0
/// and `+-2 pi / a`, where the order `q = 0` lies on the cut of real splits.
pub(super) fn rotated_chain() -> impl Strategy<Value = Ewald> {
    (
        degree_order(0..9),
        1.0_f64..2.0,
        (0.5_f64..4.0, slope()),
        (prop_oneof![Just(0.0), -0.5_f64..0.5], -1_i32..=1),
        (
            0.3_f64..1.5,
            prop_oneof![Just(None), (-0.3_f64..0.3).prop_map(Some)],
        ),
        (0.1_f64..3.0, -3.2_f64..3.2, -0.5_f64..0.5),
    )
        .prop_map(
            |((l, m), period, (kr, slope), (kpar, zone), (size, rotation), place)| {
                let (distance, azimuth, along) = place;
                let k = c(kr, kr * slope);
                let kpar = (kpar + f64::from(zone)) * TAU / period;
                let r = cylinder_point(distance * period, azimuth, along * period);
                Ewald::chain(
                    sw(l, m),
                    period,
                    kpar,
                    r,
                    k,
                    explicit_split(k, size, rotation),
                )
            },
        )
}

/// 2D spherical sums (`l < 4`) and 1D cylindrical sums (`|m| <= 3`) at the lattice
/// point, and at shifts mostly in the lattice plane or on the lattice axis, otherwise
/// 1e-12 to 1 off it on either side, on skewed lattices with Bloch vectors that include
/// 0 and zone edges; `k` with a non-negative `slope` or with `Im k` of 0.3 to 0.8 times
/// `Re k`, mirrored to `Re k < 0` for
/// cylindrical waves; real splits of 0.6 to 1.3 or splits of that modulus rotated off
/// `1 / k` by up to 0.3 either way, for cylindrical waves also of either sign.
pub(super) fn half_integer_sum() -> impl Strategy<Value = Ewald> {
    let family = prop_oneof![
        degree_order(0..4).prop_map(|(l, m)| (sw(l, m), 2)),
        (-3_i32..=3).prop_map(|m| (cw(m), 1)),
    ];
    (
        family,
        prop::array::uniform3(1.3_f64..1.9),
        -0.5_f64..0.5,
        (prop_oneof![Just(0.0), -0.5_f64..0.5], -1_i32..=1),
        prop_oneof![
            Just(([0.0; 2], 0.0)),
            (
                prop::array::uniform2(-0.8_f64..0.8),
                prop_oneof![
                    3 => Just(0.0),
                    1 => (log_uniform(-12.0..0.0), any::<bool>())
                        .prop_map(|(size, sign)| if sign { size } else { -size }),
                ],
            ),
        ],
        (
            0.8_f64..4.0,
            prop_oneof![slope().prop_map(f64::abs), 0.3_f64..0.8],
        ),
        (
            0.6_f64..1.3,
            prop_oneof![Just(None), (-0.3_f64..0.3).prop_map(Some)],
        ),
        prop::array::uniform2(any::<bool>()),
    )
        .prop_map(
            |(
                (wave, dim),
                pitch,
                skew,
                (kpar, zone),
                (shift, off),
                (kr, slope),
                (size, rotation),
                signs,
            )| {
                let cylindrical = matches!(wave, lattice::Family::Cylindrical { .. });
                let [mirrored, negative] = signs.map(|sign| sign && cylindrical);
                let k = c(if mirrored { -kr } else { kr }, kr * slope);
                let rows = [[pitch[0], 0.0], [skew, pitch[1]]];
                let eta = explicit_split(k, size, rotation);
                let r = if cylindrical {
                    [shift[0], off, 0.0]
                } else {
                    [shift[0], shift[1], off]
                };
                let eta = if negative { -eta } else { eta };
                let kpar = leading(dim, |j| (kpar + f64::from(zone)) * TAU / pitch[j]);
                Ewald::new(wave, dim, embed(rows, dim), kpar, r, k, eta)
            },
        )
}

/// Sums at real `splits` below every automatic one: 2D spherical, 1D spherical and 2D and
/// 1D cylindrical waves of degree and order up to 5 in skewed cells of pitch 0.8 to 1.4,
/// Bloch vectors within the first zone, shifts inside the cell (in and off the lattice
/// plane or axis) or, for a quarter of them, 1e-7 to 1e-3 from one of the nearest lattice
/// points, and `k` from 0.5 to 5 with `Im k` of 0 or 0.05 to 0.5.
pub(super) fn small_split_sum(splits: std::ops::Range<f64>) -> impl Strategy<Value = Ewald> {
    let family = prop_oneof![
        degree_order(0..6).prop_map(|(l, m)| (sw(l, m), 2)),
        degree_order(0..6).prop_map(|(l, m)| (sw(l, m), 1)),
        (-5_i32..=5).prop_map(|m| (cw(m), 2)),
        (-5_i32..=5).prop_map(|m| (cw(m), 1)),
    ];
    (
        family,
        prop::array::uniform2(0.8_f64..1.4),
        -0.3_f64..0.3,
        prop::array::uniform2(-0.5_f64..0.5),
        (
            prop::array::uniform3(-0.45_f64..0.45),
            prop_oneof![
                3 => Just(None),
                1 => (prop::array::uniform2(-1_i32..=1), log_uniform(-7.0..-3.0)).prop_map(Some),
            ],
        ),
        (0.5_f64..5.0, prop_oneof![Just(0.0), 0.05_f64..0.5]),
        splits,
    )
        .prop_map(
            |((wave, dim), pitch, skew, kpar, (shift, near), (kr, ki), eta)| {
                let rows = [[pitch[0], 0.0], [skew, pitch[1]]];
                let z = if matches!(wave, lattice::Family::Cylindrical { .. }) {
                    0.0
                } else {
                    shift[2]
                };
                let kpar = leading(dim, |j| kpar[j] * TAU / pitch[j]);
                let r = [shift[0], shift[1], z];
                let mut sum =
                    Ewald::new(wave, dim, embed(rows, dim), kpar, r, c(kr, ki), c(eta, 0.0));
                // Next to a lattice point the shift sets the direction of the offset.
                if let Some((point, distance)) = near {
                    let length = sum.r.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-3);
                    let axes = sum.axes();
                    sum.r = sum.r.map(|x| distance * x / length);
                    for i in 0..dim {
                        for j in 0..dim {
                            sum.r[axes[j]] += f64::from(point[i]) * rows[i][j];
                        }
                    }
                }
                sum
            },
        )
}

/// Sums next to a lattice point: 2D and 3D spherical waves of degree up to 6 (half of
/// them of degree 0, whose sums grow least there) in skewed cells of pitch 0.8 to 1.3,
/// mostly zero Bloch vectors, otherwise ones of 1e-9 to 1e-3 of the zone, shifts of 1e-7
/// to 1e-3 in any direction from one of the nearest lattice points, mostly real `k` from
/// 0.5 (1.2 for 3D sums, which smaller `k` make slow) to 3.5 and splits of 0.11 to 0.14,
/// real or rotated by up to 0.3 either way, with `Im k` capped as in
/// [`reflected_zero_sum`] ([`SPLIT_TURN`]): beyond 90 degrees the sums fail (see
/// [`SPLITS`]).
///
/// [`SPLITS`]: super::tables::SPLITS
pub(super) fn near_point_sum() -> impl Strategy<Value = Ewald> {
    (
        (
            prop_oneof![Just((0, 0)), degree_order(0..7)],
            2_usize..=3,
        ),
        (
            prop::array::uniform3(0.8_f64..1.3),
            prop::array::uniform3(-0.3_f64..0.3),
        ),
        prop_oneof![
            3 => Just([0.0; 3]),
            1 => prop::array::uniform3((log_uniform(-9.0..-3.0), any::<bool>()))
                .prop_map(|parts| parts.map(|(size, negative)| if negative { -size } else { size })),
        ],
        (
            prop::array::uniform3(-1_i32..=1),
            log_uniform(-7.0..-3.0),
            prop::array::uniform3(-1.0_f64..1.0),
        ),
        (0.0_f64..1.0, prop_oneof![3 => Just(0.0), 1 => 0.05_f64..0.3]),
        (0.11_f64..0.14, prop_oneof![Just(0.0), -0.3_f64..0.3]),
    )
        .prop_map(
            |(
                ((l, m), dim),
                (pitch, skew),
                kpar,
                (point, distance, direction),
                (fraction, ki),
                (size, angle),
            )| {
                let rows = triangular(pitch, skew, dim);
                let length = direction.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-3);
                let r = std::array::from_fn(|j| {
                    let lattice: f64 = (0..dim).map(|i| f64::from(point[i]) * rows[i][j]).sum();
                    lattice + distance * direction[j] / length
                });
                let lowest = if dim == 3 { 1.2 } else { 0.5 };
                let kr = fraction.mul_add(3.5 - lowest, lowest);
                let k = c(kr, ki.min(kr * (0.5 * SPLIT_TURN - angle).tan()));
                let kpar = leading(dim, |j| kpar[j] * TAU / pitch[j]);
                Ewald::new(sw(l, m), dim, rows, kpar, r, k, Complex::from_polar(size, angle))
            },
        )
}

/// The largest `|arg((k eta)^2)|` of the sums of `near_point_sum` and
/// `reflected_zero_sum`, 89.9 degrees.
const SPLIT_TURN: f64 = 89.9 * PI / 180.0;

/// Sums that vanish by a reflection: 2D spherical waves of degree 1 to 10 with `l + m`
/// odd in the plane of skewed cells of pitch 0.9 to 1.4, and 1D spherical waves of
/// degree 1 to 10 with `m != 0` (mostly +-1, whose first derivatives off the axis do not
/// vanish) on the axis of chains of pitch 0.8 to 1.5; Bloch vectors and positions
/// anywhere in the cell, `k` from 0.4 to 5, real or with `Im k` of 0.05 to 1, and splits
/// of 0.05 to 0.3, real or rotated by 0.15 either way, with `Im k` capped so that
/// `|arg((k eta)^2)| <= 89.9` degrees ([`SPLIT_TURN`]): beyond 90 degrees the Gaussians
/// of both Ewald parts grow and the jets fail (see [`ZEROS`]).
///
/// [`ZEROS`]: super::tables::ZEROS
pub(super) fn reflected_zero_sum() -> impl Strategy<Value = Ewald> {
    let family = prop_oneof![
        degree_order(1..11).prop_map(|(l, m)| {
            let m = if (l + m) % 2 == 0 {
                if m > -l { m - 1 } else { m + 1 }
            } else {
                m
            };
            (sw(l, m), 2)
        }),
        (
            1_i32..11,
            prop_oneof![3 => prop_oneof![Just(1), Just(-1)], 1 => -10_i32..=10]
        )
            .prop_map(|(l, m)| {
                let m = if m == 0 { 1 } else { m.clamp(-l, l) };
                (sw(l, m), 1)
            }),
    ];
    (
        family,
        (prop::array::uniform2(0.8_f64..1.5), -0.4_f64..0.4),
        (
            prop::array::uniform2(-0.5_f64..0.5),
            prop::array::uniform2(-0.6_f64..0.6),
        ),
        (0.4_f64..5.0, prop_oneof![Just(0.0), 0.05_f64..1.0]),
        (
            0.05_f64..0.3,
            prop_oneof![Just(0.0), Just(0.15), Just(-0.15)],
        ),
    )
        .prop_map(
            |((wave, dim), (pitch, skew), (kpar, fraction), (kr, ki), (size, angle))| {
                let ki = ki.min(kr * (0.5 * SPLIT_TURN - angle).tan());
                let rows = [[pitch[0], 0.0], [skew, pitch[1]]];
                let point: [f64; 2] =
                    std::array::from_fn(|j| (0..dim).map(|i| fraction[i] * rows[i][j]).sum());
                let r = if dim == 2 {
                    [point[0], point[1], 0.0]
                } else {
                    [0.0, 0.0, point[0]]
                };
                let kpar = leading(dim, |j| kpar[j] * TAU / pitch[j]);
                let eta = Complex::from_polar(size, angle);
                Ewald::new(wave, dim, embed(rows, dim), kpar, r, c(kr, ki), eta)
            },
        )
}

/// 1D spherical sums of degree 0 to 10 (7 to 10 in two thirds of them) and order 0 (in
/// two thirds) or any, on the axis (in two thirds) or 0.05 to 0.3 periods off it, of
/// chains of pitch 0.6 to 1.6 with Bloch vectors of zero or within the first zone, at `k`
/// of modulus 0.4 to 3 turned 20 to 45 degrees off the real axis and splits of modulus
/// 0.07 to 0.22 turned so that `(k eta)^2` turns 80 to 90 degrees (89 to 90 in two
/// thirds) off the positive real axis, all below every automatic split.
pub(super) fn turned_chain_sum() -> impl Strategy<Value = Ewald> {
    (
        prop_oneof![1 => 0_i32..11, 2 => 7_i32..11]
            .prop_flat_map(|l| (Just(l), prop_oneof![2 => Just(0), 1 => -l..=l])),
        (0.6_f64..1.6, prop_oneof![Just(0.0), -0.5_f64..0.5]),
        (
            prop_oneof![2 => Just(0.0), 1 => 0.05_f64..0.3],
            -3.2_f64..3.2,
            -0.5_f64..0.5,
        ),
        (0.4_f64..3.0, 20.0_f64..45.0),
        (
            0.07_f64..0.22,
            prop_oneof![1 => 80.0_f64..89.0, 2 => 89.0_f64..90.0],
        ),
    )
        .prop_map(
            |((l, m), (pitch, kpar), (rho, azimuth, along), (modulus, angle), (size, turn))| {
                let k = Complex::from_polar(modulus, angle.to_radians());
                let r = cylinder_point(rho * pitch, azimuth, along * pitch);
                let eta = Complex::from_polar(size, 0.5 * turn.to_radians() - k.arg());
                Ewald::chain(sw(l, m), pitch, kpar * TAU / pitch, r, k, eta)
            },
        )
}

/// Sums that vanish by symmetry: odd degrees (up to 5, every order) of spherical waves on
/// 1D, 2D and 3D lattices and odd orders of cylindrical waves on 1D and 2D ones, at a
/// zero Bloch vector, at a lattice point or half a lattice vector from one, plus a
/// lattice vector; cells with dyadic entries (pitches of 0.75 to 1.25, skews of -0.25 to
/// 0.25), so that twice the shift is exactly a lattice vector; `Re k` from 0.5 to 4 and
/// `Im k` up to 0.3 of it; the automatic split, or real and rotated splits of 0.3 to 1.2
/// times its modulus with the complete sum or either part.
pub(super) fn vanishing_sum() -> impl Strategy<Value = (Ewald, SumPart)> {
    let family = prop_oneof![
        (degree_order(0..6), 1_usize..=3).prop_map(|((l, m), dim)| {
            let l = l | 1;
            (sw(l, m.clamp(-l, l)), dim)
        }),
        ((-5_i32..=5).prop_map(|m| m | 1), 1_usize..=2).prop_map(|(m, dim)| (cw(m), dim)),
    ];
    let dyadic = |values: &'static [f64]| prop::sample::select(values);
    (
        family,
        prop::array::uniform3(dyadic(&[0.75, 1.0, 1.25])),
        prop::array::uniform3(dyadic(&[-0.25, 0.0, 0.25])),
        (
            prop::array::uniform3(-1_i32..=1),
            prop::array::uniform3(-1_i32..=1),
        ),
        (0.5_f64..4.0, 0.0_f64..0.3),
        prop_oneof![
            Just(None),
            (0.3_f64..1.2, prop_oneof![Just(0.0), -0.3_f64..0.3]).prop_map(Some)
        ],
        prop_oneof![
            Just(SumPart::Full),
            Just(SumPart::Real),
            Just(SumPart::Reciprocal)
        ],
    )
        .prop_map(
            |((wave, dim), pitch, skew, (halves, wholes), (kr, ki), split_factor, part)| {
                let rows = triangular(pitch, skew, dim);
                let k = c(kr, ki * kr);
                let mut sum =
                    Ewald::new(wave, dim, rows, [0.0; 3], [0.0; 3], k, Complex::default());
                let axes = sum.axes();
                for i in 0..dim {
                    let coefficient = 0.5 * f64::from(halves[i]) + f64::from(wholes[i]);
                    for j in 0..dim {
                        sum.r[axes[j]] += coefficient * rows[i][j];
                    }
                }
                let Some((factor, angle)) = split_factor else {
                    return (sum, SumPart::Full);
                };
                let automatic = resolve_split(k, &sum.lattice(), Complex::default());
                sum.eta = Complex::from_polar(factor * automatic.norm(), angle);
                (sum, part)
            },
        )
}

/// Sums on a lattice plane or axis, of degrees and orders up to 12, with the unit
/// `normal` along which [`check_tiny_normal_shift`] moves them off it: 2D spherical sums
/// in the plane of a skewed lattice (normal z), 1D cylindrical sums on the x axis
/// (normal y) and 1D spherical sums on the z axis (normal radial). Positions lie between
/// the lattice points, where the sums are regular; Bloch vectors include 0 and zone
/// edges; `k` has a non-negative `slope` or `Im k` of 0.3 to 0.8 times `Re k`; the split
/// is automatic, real or rotated off `1 / k`.
///
/// [`check_tiny_normal_shift`]: super::checks::check_tiny_normal_shift
pub(super) fn normal_shift() -> impl Strategy<Value = (Ewald, [f64; 3])> {
    let family = prop_oneof![
        degree_order(0..13).prop_map(|(l, m)| (sw(l, m), 2)),
        (-12_i32..=12).prop_map(|m| (cw(m), 1)),
        degree_order(0..13).prop_map(|(l, m)| (sw(l, m), 1)),
    ];
    (
        family,
        (prop::array::uniform2(1.3_f64..1.9), -0.5_f64..0.5),
        (prop_oneof![Just(0.0), -0.5_f64..0.5], -1_i32..=1),
        (prop::array::uniform2(0.1_f64..0.9), -3.2_f64..3.2),
        (
            0.8_f64..4.0,
            prop_oneof![slope().prop_map(f64::abs), 0.3_f64..0.8],
        ),
        prop_oneof![
            Just(None),
            (
                0.6_f64..1.3,
                prop_oneof![Just(None), (-0.3_f64..0.3).prop_map(Some)]
            )
                .prop_map(Some),
        ],
    )
        .prop_map(
            |(
                (wave, dim),
                (pitch, skew),
                (kpar, zone),
                (fraction, azimuth),
                (kr, slope),
                split,
            )| {
                let rows = [[pitch[0], 0.0], [skew, pitch[1]]];
                let k = c(kr, kr * slope);
                // The position in the lattice frame, at fractional coordinates.
                let point: [f64; 2] =
                    std::array::from_fn(|j| (0..dim).map(|i| fraction[i] * rows[i][j]).sum());
                let (r, normal) = match (wave, dim) {
                    (lattice::Family::Spherical { .. }, 2) => {
                        ([point[0], point[1], 0.0], [0.0, 0.0, 1.0])
                    }
                    (lattice::Family::Cylindrical { .. }, _) => {
                        ([point[0], 0.0, 0.0], [0.0, 1.0, 0.0])
                    }
                    _ => ([0.0, 0.0, point[0]], [azimuth.cos(), azimuth.sin(), 0.0]),
                };
                let kpar = leading(dim, |j| (kpar + f64::from(zone)) * TAU / pitch[j]);
                let eta = split.map_or_else(Complex::default, |(size, rotation)| {
                    explicit_split(k, size, rotation)
                });
                (
                    Ewald::new(wave, dim, embed(rows, dim), kpar, r, k, eta),
                    normal,
                )
            },
        )
}

/// 2D spherical sums of degree up to 12 and 1D cylindrical sums of order up to 12 off
/// their lattice plane or axis, at `|t| = |k s eta|^2` of 1e-2 to 6 for the distance `s`
/// and the split `eta` they take: the band in which the reciprocal parts pass from the
/// series in `t` to the Kambe chain. Skewed lattices, Bloch vectors in the first zone,
/// positions anywhere in the cell and on either side; `Re k` of 1 to 3 (of either sign
/// for cylindrical waves) and `Im k` of 0.8 to 1.2, where the image sums converge
/// absolutely (see [`check_direct_jets`]); the split automatic, or 0.8 to 1.25 times it
/// and rotated by up to 0.3 either way, where the Ewald parts cancel by little.
///
/// [`check_direct_jets`]: super::checks::check_direct_jets
pub(super) fn off_plane_sum() -> impl Strategy<Value = Ewald> {
    let family = prop_oneof![
        degree_order(0..13).prop_map(|(l, m)| (sw(l, m), 2)),
        (-12_i32..=12).prop_map(|m| (cw(m), 1)),
    ];
    (
        family,
        (prop::array::uniform2(1.3_f64..1.9), -0.5_f64..0.5),
        prop::array::uniform2(-0.5_f64..0.5),
        prop::array::uniform2(0.0_f64..1.0),
        ((1.0_f64..3.0, 0.8_f64..1.2), any::<bool>()),
        prop_oneof![Just(None), (0.8_f64..1.25, -0.3_f64..0.3).prop_map(Some)],
        (log_uniform(-2.0..0.78), any::<bool>()),
    )
        .prop_map(
            |(
                (wave, dim),
                (pitch, skew),
                kpar,
                fraction,
                ((kr, ki), mirrored),
                split_choice,
                (modulus, below),
            )| {
                let cylindrical = matches!(wave, lattice::Family::Cylindrical { .. });
                let k = c(if mirrored && cylindrical { -kr } else { kr }, ki);
                let rows = [[pitch[0], 0.0], [skew, pitch[1]]];
                let point: [f64; 2] =
                    std::array::from_fn(|j| (0..dim).map(|i| fraction[i] * rows[i][j]).sum());
                let kpar = leading(dim, |j| kpar[j] * TAU / pitch[j]);
                let r = [point[0], point[1], 0.0];
                let mut sum =
                    Ewald::new(wave, dim, embed(rows, dim), kpar, r, k, Complex::default());
                if let Some((factor, rotation)) = split_choice {
                    sum.eta = factor
                        * resolve_split(k, &sum.lattice(), sum.eta)
                        * Complex::from_polar(1.0, rotation);
                }
                let distance =
                    modulus.sqrt() / (k * resolve_split(k, &sum.lattice(), sum.eta)).norm();
                let distance = if below { -distance } else { distance };
                sum.r = if cylindrical {
                    [point[0], distance, 0.0]
                } else {
                    [point[0], point[1], distance]
                };
                sum
            },
        )
}

/// Cartesian shifts inside a cell.
pub(super) fn shift() -> impl Strategy<Value = [f64; 3]> {
    prop::array::uniform3(-0.4_f64..0.4)
}

/// Ewald sums of every family (spherical waves on 1D, 2D and 3D lattices and
/// cylindrical waves on 1D and 2D lattices) with `l, |m| < 4`, skewed lattices,
/// Bloch vectors off the axes, lossy wavenumbers, and shifts that include the origin,
/// where the self term supplies the derivatives.
pub(super) fn ewald() -> impl Strategy<Value = Ewald> {
    ewald_at(prop_oneof![Just([0.0; 3]), shift()])
}

/// [`ewald`] with the given Cartesian shifts; cylindrical shifts drop their z part. Where
/// `(k eta)^2` lies more than 45 degrees off the real axis (up to 64 degrees here), 3D
/// sums and sums above every automatic split keep the fixed shell limit and may fail
/// with "did not converge" ("Splits turned off 1/k" in
/// `docs/validation/numerical-limits.md`). Draws whose real-space jet fails so are left
/// out: 18 of 1e5 3D draws beyond 45 degrees (degrees 2 and 3 at `k` near 0.8 + 0.49i
/// and splits near 0.7), which failed the checks of six of the seven properties that
/// draw from here, and none of 1e6 1D and 2D draws.
pub(super) fn ewald_at(shift: impl Strategy<Value = [f64; 3]>) -> impl Strategy<Value = Ewald> {
    let family = prop_oneof![
        (1_usize..=3, degree_order(0..4)).prop_map(|(dim, (l, m))| (sw(l, m), dim)),
        (1_usize..=2, -3_i32..=3).prop_map(|(dim, m)| (cw(m), dim)),
    ];
    (
        family,
        prop::array::uniform3(1.3_f64..1.9),
        prop::array::uniform3(-0.5_f64..0.5),
        prop::array::uniform3(-0.6_f64..0.6),
        shift,
        (0.8_f64..2.5, 0.1_f64..0.5),
        0.7_f64..1.3,
    )
        .prop_map(|((wave, dim), pitch, skew, kpar, mut r, (kr, ki), eta)| {
            if matches!(wave, lattice::Family::Cylindrical { .. }) {
                r[2] = 0.0;
            }
            let rows = triangular(pitch, skew, dim);
            Ewald::new(wave, dim, rows, kpar, r, c(kr, ki), c(eta, 0.0))
        })
        .prop_filter(
            "a turned split whose real-space jet does not converge",
            |sum| {
                let square = (sum.k * sum.eta).powi(2);
                square.re >= square.im.abs() || {
                    let (k, lattice, real) = (sum.k, sum.lattice(), SumPart::Real);
                    let jet = derivatives_part(sum.wave, k, &lattice, sum.r, sum.eta, real);
                    !fails_with(&jet, &["did not converge"])
                }
            },
        )
}

/// The wavenumbers of [`lattice_point`].
#[derive(Clone, Copy, Debug)]
enum Wavenumber {
    /// `Re k > 0` with a `slope`.
    Lossy,
    /// `Re k < 0` with a `slope`, and either sign of a zero `Im k`.
    Mirrored,
    /// Imaginary `k` of either sign, with either sign of a zero `Re k`.
    Imaginary,
}

/// Sums at a lattice point of the families whose Ewald sums take every split: spherical
/// waves on 1D and 3D lattices and cylindrical waves on 2D lattices, with `l, |m| <= 1`,
/// where the self term reaches the values (degree 0) and derivatives (degree 1). Skewed
/// lattices, Bloch vectors that include 0, and each kind of [`Wavenumber`] with the
/// automatic split, real splits of 0.6 to 1.3 or splits of that modulus rotated off
/// `1 / k` by up to 0.3 either way. Imaginary `k` takes imaginary splits `+-i |eta|`,
/// with either sign of a zero real part, instead of real ones. Spherical Ewald sums
/// converge only for `Re(k eta) > 0` and take `Re k > 0`: for them `Re k < 0` stays
/// `Lossy` and imaginary splits take the sign with `Re(k eta) > 0`. Wavenumbers in
/// the negative imaginary half-plane test rejection.
pub(super) fn lattice_point() -> impl Strategy<Value = Ewald> {
    let family = prop_oneof![
        (prop_oneof![Just(1_usize), Just(3)], degree_order(0..2))
            .prop_map(|(dim, (l, m))| (sw(l, m), dim)),
        (-1_i32..=1).prop_map(|m| (cw(m), 2)),
    ];
    let wavenumber = prop_oneof![
        Just(Wavenumber::Lossy),
        Just(Wavenumber::Mirrored),
        Just(Wavenumber::Imaginary),
    ];
    let split = prop_oneof![
        Just(None),
        (
            0.6_f64..1.3,
            prop_oneof![Just(None), (-0.3_f64..0.3).prop_map(Some)]
        )
            .prop_map(Some),
    ];
    (
        (family, wavenumber),
        prop::array::uniform3(1.3_f64..1.9),
        prop::array::uniform3(-0.5_f64..0.5),
        prop_oneof![Just([0.0; 3]), prop::array::uniform3(-0.6_f64..0.6)],
        (0.8_f64..2.5, slope()),
        prop::array::uniform3(any::<bool>()),
        split,
    )
        .prop_map(
            |(((wave, dim), kind), pitch, skew, kpar, (size, slope), signs, split)| {
                let spherical = matches!(wave, lattice::Family::Spherical { .. });
                let zero = |negative: bool| if negative { -0.0 } else { 0.0 };
                let [negative, zero_sign, split_sign] = signs;
                let k = match kind {
                    Wavenumber::Mirrored if !spherical => c(
                        -size,
                        if slope == 0.0 {
                            zero(zero_sign)
                        } else {
                            size * slope
                        },
                    ),
                    Wavenumber::Imaginary => {
                        c(zero(zero_sign), if negative { -size } else { size })
                    }
                    _ => c(size, size * slope),
                };
                let eta = split.map_or_else(Complex::default, |(modulus, rotation)| {
                    match (kind, rotation) {
                        (Wavenumber::Imaginary, None) => {
                            // Re(k eta) = -Im k Im eta.
                            let positive = if spherical { negative } else { split_sign };
                            c(zero(split_sign), if positive { modulus } else { -modulus })
                        }
                        _ => explicit_split(k, modulus, rotation),
                    }
                });
                Ewald::new(
                    wave,
                    dim,
                    triangular(pitch, skew, dim),
                    kpar,
                    [0.0; 3],
                    k,
                    eta,
                )
            },
        )
}
