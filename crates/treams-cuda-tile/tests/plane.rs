//! Real-device comparison with the shared CPU core and linear field invariants.
#![cfg(all(feature = "cuda-tile", target_os = "linux"))]
#![allow(clippy::expect_used)]

use proptest::prelude::*;
use treams_core::{Complex, plane};
use treams_cuda_tile::PlaneWaves;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    #[ignore = "requires a CUDA 13.2+ toolkit, tileiras, and an NVIDIA GPU"]
    fn complex_plane_obeys_translation(
        kx in -2.0..2.0f64, ky in -2.0..2.0f64,
        amplitude in (-1.0..1.0f64, -1.0..1.0f64),
        point in prop::array::uniform3(-3.0..3.0f64),
        shift in prop::array::uniform3(-1.0..1.0f64),
        polarization in any::<bool>(), helicity in any::<bool>(), lossless in any::<bool>(),
    ) {
        let attenuation = if lossless { 0.0 } else { 1.0 };
        let k = [Complex::new(kx, 0.03 * attenuation), Complex::new(ky, -0.02 * attenuation), Complex::new(1.4, 0.07 * attenuation)];
        let coefficients = [Complex::new(amplitude.0, amplitude.1)];
        let gpu = PlaneWaves::new(0, &[k], &[u8::from(polarization)], &coefficients, helicity).expect("upload");
        let displaced = [point[0] + shift[0], point[1] + shift[1], point[2] + shift[2]];
        let fields = gpu.evaluate(&[point, displaced]).expect("fields");
        let phase = (Complex::i() * k.iter().zip(shift).map(|(v, x)| v * x).sum::<Complex>()).exp();
        for (left, right) in fields.first().expect("first point").iter().zip(fields.last().expect("second point")) {
            prop_assert!((*left * phase - right).norm() <= 2e-12 * (1.0 + right.norm()));
        }
    }
}

#[test]
#[ignore = "requires a CUDA 13.2+ toolkit, tileiras, and an NVIDIA GPU"]
fn complex_fields_match_core_and_preserve_superposition() {
    for helicity in [false, true] {
        for modes in [1, 3, 17, 31, 32, 33, 127, 128, 129] {
            let vectors: Vec<_> = (0..modes)
                .map(|i| {
                    let f = f64::from(i + 1);
                    [
                        Complex::new(f.sin(), 0.07),
                        Complex::new(f.cos(), -0.03),
                        Complex::new(1.2, 0.01 * f),
                    ]
                })
                .collect();
            let polarizations: Vec<_> = (0..modes).map(|i| u8::from(i % 2 == 1)).collect();
            let coefficients: Vec<_> = (0..modes)
                .map(|i| Complex::new(0.2 * f64::from(i + 1), -0.3))
                .collect();
            let gpu = PlaneWaves::new(0, &vectors, &polarizations, &coefficients, helicity)
                .expect("upload");
            for count in [1, 127, 128, 129, 257] {
                let points: Vec<_> = (0..count)
                    .map(|i| {
                        let f = f64::from(i + 1) / f64::from(count);
                        [f, -0.4 * f, 0.2 * f]
                    })
                    .collect();
                let actual = gpu.evaluate(&points).expect("GPU fields");
                let (expected, _) = plane::field(
                    vectors.clone(),
                    polarizations.clone(),
                    points.clone(),
                    Some(coefficients.clone()),
                    helicity,
                )
                .expect("CPU fields");
                for (a, e) in actual.iter().flatten().zip(expected.as_slice()) {
                    assert!((*a - e).norm() <= 2e-12 * (1.0 + e.norm()), "{a} != {e}");
                }
                let reversed: Vec<_> = points.into_iter().rev().collect();
                let reversed_field = gpu.evaluate(&reversed).expect("reordered fields");
                for (a, b) in actual
                    .iter()
                    .rev()
                    .flatten()
                    .zip(reversed_field.iter().flatten())
                {
                    assert!((*a - b).norm() <= 2e-12 * (1.0 + a.norm()));
                }
            }
            let points = [[0.1, -0.2, 0.3], [0.3, 0.2, -0.1]];
            let scale = Complex::new(-0.4, 0.7);
            let scaled: Vec<_> = coefficients.iter().map(|c| c * scale).collect();
            let scaled_gpu = PlaneWaves::new(0, &vectors, &polarizations, &scaled, helicity)
                .expect("scaled expansion");
            let actual = gpu.evaluate(&points).expect("fields");
            let scaled_actual = scaled_gpu.evaluate(&points).expect("scaled fields");
            for (a, b) in actual.iter().flatten().zip(scaled_actual.iter().flatten()) {
                assert!((*a * scale - b).norm() <= 2e-12 * (1.0 + b.norm()));
            }
            assert!(gpu.evaluate(&[]).expect("empty field").is_empty());
            assert!(gpu.evaluate(&[[f64::NAN, 0.0, 0.0]]).is_err());
        }
    }
}

#[test]
#[ignore = "requires a CUDA 13.2+ toolkit, tileiras, and an NVIDIA GPU"]
fn opposite_expansions_cancel_across_mode_reductions() {
    // Separate the positive and negative expansions so cancellation spans many
    // mode tiles instead of testing only adjacent summands.
    let half = 129;
    let vectors: Vec<_> = (0..half)
        .map(|i| {
            let angle = f64::from(i) * 0.37;
            [
                Complex::new(angle.cos(), 0.04),
                Complex::new(angle.sin(), -0.02),
                Complex::new(1.2, 0.03),
            ]
        })
        .collect();
    let vectors: Vec<_> = vectors.iter().chain(&vectors).copied().collect();
    let labels: Vec<_> = (0..half).map(|i| u8::from(i % 2 == 1)).collect();
    let labels: Vec<_> = labels.iter().chain(&labels).copied().collect();
    let amplitude = Complex::new(0.7, -0.3) / f64::from(half);
    let coefficients: Vec<_> = std::iter::repeat_n(amplitude, 129)
        .chain(std::iter::repeat_n(-amplitude, 129))
        .collect();
    let points: Vec<_> = (0..129)
        .map(|i| {
            let x = f64::from(i) * 0.09;
            [x.sin(), x.cos(), 0.3 * x]
        })
        .collect();
    for helicity in [false, true] {
        let gpu = PlaneWaves::new(0, &vectors, &labels, &coefficients, helicity)
            .expect("opposite expansions");
        for field in gpu.evaluate(&points).expect("canceling field") {
            for component in field {
                assert!(component.norm() <= 5e-13, "{component}");
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA 13.2+ toolkit, tileiras, and an NVIDIA GPU"]
fn arbitrarily_small_attenuation_is_not_treated_as_lossless() {
    // The real-k specialization must use exact zero, never a numerical cutoff.
    // A small imaginary wavenumber has an observable effect over a long path.
    for attenuation in [0.0, 1e-12] {
        let wave = [
            Complex::default(),
            Complex::default(),
            Complex::new(1.0, attenuation),
        ];
        let gpu = PlaneWaves::new(0, &[wave], &[0], &[Complex::new(1.0, 0.0)], true)
            .expect("weakly attenuating wave");
        let fields = gpu
            .evaluate(&[[0.0, 0.0, 0.0], [0.0, 0.0, 1e12]])
            .expect("long propagation");
        let intensity = |field: &[Complex; 3]| field.iter().map(Complex::norm_sqr).sum::<f64>();
        let ratio =
            intensity(fields.last().expect("end")) / intensity(fields.first().expect("start"));
        assert!((ratio - (-2e12 * attenuation).exp()).abs() <= 2e-13);
    }
}
