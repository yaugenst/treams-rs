//! Planar S-matrices: interfaces, layers, composition, illumination and observables.

#[path = "smatrix/pushforward_composition.rs"]
mod pushforward_composition;
#[path = "smatrix/pushforward_physics.rs"]
mod pushforward_physics;

use std::f64::consts::TAU;

use faer::MatRef;
use nalgebra::DMatrix;
use proptest::{prelude::*, test_runner::TestCaseError};

use crate::{
    Complex,
    coeffs::Material,
    linalg::{self, view},
    pw::wave_vector_z,
    smatrix::{self, Blocks, identity_blocks},
    test_support::{
        ALGEBRA_CASES, Chirality, DEFAULT_CASES, central, complex, complex_matrix, dot,
        helicity_ks, log_uniform, material, prop_assert_close, re_dot,
    },
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn transmittance_complete_adjoint_and_amplitude_scaling(
        scale in 0.8_f64..1.3, axis in 0_usize..3, helicity in any::<bool>(), down in any::<bool>(),
    ) {
        check_transmittance(scale, axis, helicity, usize::from(down))?;
    }

    #[test]
    fn chirality_interval_additivity_and_scale_adjoint(
        k in 0.8_f64..2.0, q in 0.1_f64..0.5, stop in 0.1_f64..1.0, axis in 0_usize..3,
    ) {
        check_chirality_intervals(k, q, stop, axis)?;
    }

    #[test]
    fn internal_illumination_matches_composition(
        (lower, upper, incoming, g) in (1_usize..=4, 1_usize..=3).prop_flat_map(|(n, p)| (
            scattering(n), scattering(n),
            prop::array::uniform2(complex_matrix(n, p, 1.0)),
            prop::array::uniform4(complex_matrix(n, p, 1.0)),
        )),
    ) {
        check_illumination_composition(&lower, &upper, &incoming, &g)?;
    }

    #[test]
    fn compact_layer_scale_adjoint(k in 1.1_f64..2.0, d in 0.1_f64..0.8, axis in 0_usize..3) {
        check_compact_layer_scale(k, d, axis)?;
    }

    #[test]
    fn compact_layer_pushforward_matches_adjoint_and_reference(
        k in 1.1_f64..2.0, d in 0.1_f64..0.8, axis in 0_usize..3, fixed_q in any::<bool>(),
    ) {
        check_compact_layer_pushforward(k, d, axis, fixed_q)?;
    }

    #[test]
    fn compact_layers_match_generic_composition(
        (media, thickness) in (1_usize..=3).prop_flat_map(|layers| (
            prop::collection::vec(lossy_medium(), layers + 2),
            prop::collection::vec(0.0_f64..1.5, layers),
        )),
        channels in prop::collection::vec(
            channel(0.5),
            1..=3,
        ),
        axis in 0_usize..3, fixed_q in any::<bool>(),
    ) {
        let (q, g): (Vec<_>, Vec<_>) = channels.into_iter().unzip();
        check_layers_match_generic_chain(&media, &thickness, &q, axis, fixed_q, g)?;
    }

    #[test]
    fn identical_media_interface_is_exact_identity(
        k in 1.1_f64..2.0, chiral in complex(0.3), z in (0.6_f64..1.4, 0.0_f64..0.05),
        angle in 0.0_f64..TAU, gap in log_uniform(-12.0..0.0), axis in 0_usize..3,
        g in prop::array::uniform4(complex_matrix(2, 2, 1.0)),
    ) {
        let other = Complex::new(1.3 * k + chiral.re, chiral.im.abs());
        let ks = [[Complex::new(k, 0.0), other]; 2];
        let q = (k * k - gap).sqrt();
        let q = [q * angle.cos(), q * angle.sin()];
        check_identical_media_interface(ks, Complex::new(z.0, z.1), q, axis, &g)?;
    }

    #[test]
    fn identical_media_remain_transparent_at_cutoff(
        k in 0.1_f64..10.0, z in 0.2_f64..3.0, component in 0_usize..2, negative in any::<bool>(),
        axis in 0_usize..3,
    ) {
        check_threshold_interface(k, z, component, negative, axis)?;
    }

    #[test]
    fn radiation_multilinearity_adjoint(
        response in complex_matrix(3, 3, 0.5),
        channels in prop::array::uniform4(complex_matrix(3, 2, 0.5)),
        g in prop::array::uniform4(complex_matrix(2, 2, 0.5)),
    ) {
        check_radiation_multilinearity(&response, &channels, &g)?;
    }

    #[test]
    fn fresnel_dimensionless_adjoint(k in 1.0_f64..3.0, q in 0.0_f64..0.6, z in 0.7_f64..1.3) {
        check_fresnel_dimensionless(k, q, z)?;
    }

    #[test]
    fn propagation_is_a_translation_group(
        (vectors, g) in (1_usize..=3).prop_flat_map(|n| (
            prop::collection::vec(
                (prop::array::uniform2(-1.0_f64..1.0), -2.0_f64..2.0, 0.0_f64..0.3),
                n,
            ),
            prop::array::uniform4(complex_matrix(n, n, 1.0)),
        )),
        first in (prop::array::uniform2(-1.0_f64..1.0), 0.0_f64..2.0),
        second in (prop::array::uniform2(-1.0_f64..1.0), 0.0_f64..2.0),
    ) {
        let vectors: Vec<_> = vectors
            .into_iter()
            .map(|([x, y], re, im)| [x.into(), y.into(), Complex::new(re, im)])
            .collect();
        let displacement = |([x, y], z): ([f64; 2], f64)| [x, y, z];
        check_propagation_group(&vectors, displacement(first), displacement(second), &g)?;
    }

    #[test]
    fn periodic_transfer_adjoint(
        (a, da, g) in (1_usize..=3).prop_flat_map(|n| (
            scattering(n),
            prop::array::uniform4(complex_matrix(n, n, 1.0)),
            complex_matrix(2 * n, 2 * n, 1.0),
        )),
    ) {
        check_periodic_adjoint(&a, &da, &g)?;
    }

    #[test]
    fn propagation_bands_recover_normal_wavenumbers(
        kz in prop::collection::vec((0.0_f64..0.5, 0.0_f64..0.3), 1..=3),
        g in prop::collection::vec(complex(1.0), 6),
        distance in 0.5_f64..1.0, period in 0.5_f64..2.0,
    ) {
        // Real parts in [0.2, 0.7), [1, 1.5) and [1.8, 2.3) keep +-kz distinct and
        // |Re kz| d below pi.
        let kz: Vec<_> = kz
            .iter()
            .zip([0.2, 1.0, 1.8])
            .map(|(&(re, im), start)| Complex::new(start + re, im))
            .collect();
        check_propagation_bands(&kz, &g[..2 * kz.len()], distance, period)?;
    }

    #[test]
    fn layer_split_preserves_value_and_gradients(
        media in prop::array::uniform3(lossy_medium()),
        thickness in 0.1_f64..1.5, fraction in 0.1_f64..0.9,
        channels in prop::collection::vec(
            channel(0.35),
            1..=2,
        ),
        axis in 0_usize..3, fixed_q in any::<bool>(),
    ) {
        let (q, g): (Vec<_>, Vec<_>) = channels.into_iter().unzip();
        check_layer_split(media, thickness, fraction, &q, g, axis, fixed_q)?;
    }

    #[test]
    fn lossless_slabs_are_unitary_and_lossy_slabs_passive(
        outer in (1.0_f64..3.0, 0.8_f64..1.5), inner in (1.2_f64..4.0, 0.8_f64..1.5, -0.3_f64..0.3),
        loss in 0.05_f64..0.5, thickness in 0.1_f64..1.5,
        q in prop::array::uniform2(-0.35_f64..0.35), axis in 0_usize..3,
    ) {
        check_slab_energy(outer, inner, loss, thickness, q, axis)?;
    }

    #[test]
    fn stacks_between_lossless_media_transmit_reciprocally(
        below in (1.0_f64..3.0, 0.8_f64..1.5), above in (1.0_f64..3.0, 0.8_f64..1.5),
        interior in prop::array::uniform2(layer_material()),
        thickness in prop::array::uniform2(0.1_f64..1.0),
        q in prop::array::uniform2(-0.35_f64..0.35), axis in 0_usize..3, chiral in any::<bool>(),
    ) {
        check_reciprocity(below, above, interior, thickness, q, axis, chiral)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

    #[test]
    fn bloch_spectra_are_cyclic_and_double(
        (a, b) in (1_usize..=3).prop_flat_map(|n| (scattering(n), scattering(n))),
    ) {
        check_bloch_spectra(&a, &b)?;
    }

    #[test]
    fn redheffer_product_is_a_monoid_with_mirror_symmetry(
        (a, b, c, g, da, db) in (1_usize..=4).prop_flat_map(|n| (
            scattering(n), scattering(n), scattering(n),
            prop::array::uniform4(complex_matrix(n, n, 1.0)),
            prop::array::uniform4(complex_matrix(n, n, 1.0)),
            prop::array::uniform4(complex_matrix(n, n, 1.0)),
        )),
    ) {
        check_redheffer_algebra(&a, &b, &c)?;
        check_redheffer_adjoint(&a, &b, &g, &da, &db)?;
    }

    #[test]
    fn redheffer_product_preserves_unitarity(
        (x, y) in (1_usize..=3).prop_flat_map(|n| {
            let bound = 0.07 / f64::from(u32::try_from(n).unwrap());
            (complex_matrix(2 * n, 2 * n, bound), complex_matrix(2 * n, 2 * n, bound))
        }),
    ) {
        check_redheffer_unitarity(&x, &y)?;
    }

    #[test]
    fn interfaces_compose_transitively_and_invert(
        media in prop::array::uniform3(lossy_medium()),
        q in prop::array::uniform2(-0.35_f64..0.35), axis in 0_usize..3,
    ) {
        check_interface_algebra(media, q, axis)?;
    }

    #[test]
    fn cartesian_interface_matches_fresnel_value_and_pullback(
        media in prop::array::uniform2(lossy_medium()),
        q in prop::array::uniform2(-0.35_f64..0.35),
        g in prop::array::uniform4(complex_matrix(2, 2, 1.0)),
    ) {
        check_interface_fresnel(media, q, &g)?;
    }
}

/// A [`material`] with `Re ε` in `[1.2, 4)`, loss `Im ε` in `[0, 0.5]`, real `μ` in
/// `[0.8, 1.5)` and `|κ| < 0.3`; at `k0 = 1` both helicities propagate for every
/// transverse wavevector of modulus below 0.68.
fn layer_material() -> impl Strategy<Value = Material> {
    material(1.2..4.0, 0.8..1.5, [0.5, 0.0], Chirality::Absolute(0.3))
}

/// A [`layer_material`] as the input of the planar kernels: its helicity wavenumbers at
/// `k0 = 1` and its impedance.
fn lossy_medium() -> impl Strategy<Value = ([Complex; 2], Complex)> {
    layer_material().prop_map(|material| planar_medium(&material))
}

/// Helicity wavenumbers at `k0 = 1`, in the order of `Material.ks`, and impedance of a
/// material.
fn planar_medium(material: &Material) -> ([Complex; 2], Complex) {
    (helicity_ks(1.0, material), material.impedance())
}

/// A lossless material with real relative permittivity, permeability and chirality.
fn lossless(epsilon: f64, mu: f64, kappa: f64) -> Material {
    Material {
        epsilon: Complex::new(epsilon, 0.0),
        mu: Complex::new(mu, 0.0),
        kappa: Complex::new(kappa, 0.0),
    }
}

/// A transverse wavevector with components below `bound` and a random cotangent of
/// its 2 by 2 polarization blocks.
fn channel(bound: f64) -> impl Strategy<Value = ([f64; 2], Blocks)> {
    (
        prop::array::uniform2(-bound..bound),
        prop::array::uniform4(complex_matrix(2, 2, 1.0)),
    )
}

/// The compact per-channel layer stack equals the explicit generic chain of
/// interfaces, propagations and Redheffer compositions: bitwise in value, and in
/// every parameter gradient, where the chain maps its normal-wavenumber gradients
/// to `(k, q)` through `∂kz/∂k = k/kz` and `∂kz/∂q = -q/kz`.
fn check_layers_match_generic_chain(
    media: &[([Complex; 2], Complex)],
    thickness: &[f64],
    q: &[[f64; 2]],
    axis: usize,
    fixed_q: bool,
    g: Vec<Blocks>,
) -> Result<(), TestCaseError> {
    let (ks, zs): (Vec<_>, Vec<_>) = media.iter().copied().unzip();
    let (values, residual) =
        smatrix::layer_stack(ks.clone(), &zs, q.to_vec(), thickness, axis, fixed_q).unwrap();
    let gradient = residual.pullback(g.clone()).unwrap();
    let mut expected = smatrix::LayerStackGradient {
        ks: vec![[Complex::default(); 2]; ks.len()],
        zs: vec![Complex::default(); ks.len()],
        q: Vec::new(),
        thickness: vec![0.0; thickness.len()],
    };
    let accumulate = |expected: &mut smatrix::LayerStackGradient,
                      medium: usize,
                      gradient: smatrix::InterfaceGradient,
                      q: &mut [f64; 2]| {
        let smatrix::InterfaceGradient {
            ks: gk,
            z: gz,
            q: gq,
        } = gradient;
        for side in 0..2 {
            for (target, g) in expected.ks[medium + side].iter_mut().zip(gk[side]) {
                *target += g;
            }
            expected.zs[medium + side] += gz[side];
            q[side] += gq[side];
        }
    };
    for ((&q, value), g) in q.iter().zip(&values).zip(g) {
        let interface = |m: usize| {
            smatrix::interface([ks[m], ks[m + 1]], [zs[m], zs[m + 1]], q, axis, fixed_q).unwrap()
        };
        let (mut chain, first) = interface(0);
        let mut steps = Vec::new();
        for (layer, &d) in thickness.iter().enumerate() {
            let normal = ks[layer + 1].map(|k| wave_vector_z(q[0].into(), q[1].into(), k));
            let vectors = normal.map(|kz| [q[0].into(), q[1].into(), kz]).to_vec();
            let (phases, propagation) = smatrix::propagation(vectors, [0.0, 0.0, d]).unwrap();
            let (spaced, spacer) = smatrix::add(chain, phases).unwrap();
            let (matching, boundary) = interface(layer + 1);
            let composed;
            (chain, composed) = smatrix::add(spaced, matching).unwrap();
            steps.push((normal, propagation, spacer, boundary, composed));
        }
        prop_assert_eq!(value, &chain);
        let mut cotangent = g;
        let mut gq = [0.0; 2];
        for (layer, (normal, propagation, spacer, boundary, composed)) in
            steps.into_iter().enumerate().rev()
        {
            let composed_g = composed.pullback(&cotangent).unwrap();
            let boundary_g = boundary.pullback(&composed_g.upper).unwrap();
            accumulate(&mut expected, layer + 1, boundary_g, &mut gq);
            let spacer_g = spacer.pullback(&composed_g.lower).unwrap();
            let (previous, propagation_g) = (spacer_g.lower, spacer_g.upper);
            let smatrix::PropagationGradient {
                vectors: gv,
                distance: gd,
            } = propagation.pullback(&propagation_g).unwrap();
            expected.thickness[layer] += gd[2];
            for (pol, gv) in gv.iter().enumerate() {
                let k = ks[layer + 1][pol];
                expected.ks[layer + 1][pol] += gv[2] * (k / normal[pol]).conj();
                if !fixed_q {
                    for (a, &q) in q.iter().enumerate() {
                        gq[a] += (gv[2] * (-q / normal[pol]).conj()).re;
                    }
                }
            }
            cotangent = previous;
        }
        let first_g = first.pullback(&cotangent).unwrap();
        accumulate(&mut expected, 0, first_g, &mut gq);
        expected.q.push(gq);
    }
    let complex_close = |actual: Vec<Complex>, expected: Vec<Complex>| {
        let scale = expected.iter().map(|z| z.norm()).sum::<f64>();
        prop_assert_close!(actual, expected, 1e-12 * (1.0 + scale));
        Ok(())
    };
    let real_close = |actual: Vec<f64>, expected: Vec<f64>| {
        let scale = expected.iter().map(|v| v.abs()).sum::<f64>();
        prop_assert_close!(actual, expected, 1e-12 * (1.0 + scale));
        Ok(())
    };
    let flat = |a: &[[Complex; 2]]| a.iter().flatten().copied().collect::<Vec<_>>();
    complex_close(flat(&gradient.ks), flat(&expected.ks))?;
    complex_close(gradient.zs, expected.zs)?;
    let flat = |a: &[[f64; 2]]| a.iter().flatten().copied().collect::<Vec<_>>();
    real_close(flat(&gradient.q), flat(&expected.q))?;
    real_close(gradient.thickness, expected.thickness)?;
    if fixed_q {
        prop_assert!(gradient.q.iter().flatten().all(|&v| v == 0.0));
    }
    Ok(())
}

/// Transmittance is invariant under incident amplitude scaling, so its incident
/// gradient is orthogonal to the incident field; every parameter pullback matches a
/// central difference.
fn check_transmittance(
    scale: f64,
    axis: usize,
    helicity: bool,
    direction: usize,
) -> Result<(), TestCaseError> {
    let modes = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
    let q = vec![[0.2, 0.13], [0.31, -0.17]];
    let block = |diagonal: f64, off: f64, im: f64| {
        DMatrix::from_fn(4, 4, |i, j| {
            Complex::new(if i == j { diagonal } else { off }, im)
        })
    };
    let matrices = [block(0.8, 0.02, 0.01), block(0.1, 0.01, -0.02)];
    let incident = DMatrix::from_fn(4, 2, |i, j| {
        let offset = 0.03 * f64::from(u32::try_from(i + j).unwrap());
        Complex::new(0.2 + offset, -0.1 * scale)
    });
    let ks = [
        [
            Complex::new(1.3, 0.02),
            Complex::new(if helicity { 1.5 } else { 1.3 }, 0.02),
        ],
        [
            Complex::new(1.2, 0.01),
            Complex::new(if helicity { 1.4 } else { 1.2 }, 0.01),
        ],
    ];
    let zs = [Complex::new(0.9, 0.01), Complex::new(1.1, -0.01)];
    let ports = |ks, zs, q: &[[f64; 2]]| smatrix::TrPorts {
        ks,
        zs,
        q: q.to_vec(),
        modes: modes.clone(),
        axis,
        helicity,
        direction,
    };
    let value = |matrices: &[DMatrix<Complex>; 2], incident: &DMatrix<Complex>, ks, zs, q: &[_]| {
        let matrices = matrices.each_ref().map(view);
        smatrix::tr_value(matrices, incident, &ports(ks, zs, q)).unwrap()
    };
    let residual = smatrix::tr(
        matrices.each_ref().map(view),
        incident.clone(),
        ports(ks, zs, &q),
        false,
    )
    .unwrap();
    let expected = residual.value().clone();
    let g = DMatrix::from_row_slice(2, 2, &[0.3, -0.2, 0.1, 0.4]);
    let gradient = residual.pullback(&g).unwrap();
    prop_assert_close!(re_dot(&gradient.incident, &incident), 0.0, 1e-12);
    let scaled = value(
        &matrices,
        &(&incident * Complex::new(scale, 0.0)),
        ks,
        zs,
        &q,
    );
    prop_assert_close!(scaled, expected, 1e-12);
    let d = Complex::new(0.07, -0.03);
    let along = |sum: Complex| (sum.conj() * d).re;
    for parameter in 0..5 {
        let fd = central(1e-5, |step| {
            let (mut matrices, mut incident, mut ks, mut zs, mut q) =
                (matrices.clone(), incident.clone(), ks, zs, q.clone());
            match parameter {
                0 => matrices.iter_mut().flatten().for_each(|v| *v += step * d),
                1 => incident.iter_mut().for_each(|v| *v += step * d),
                2 => ks.iter_mut().flatten().for_each(|v| *v += step * d),
                3 => zs.iter_mut().for_each(|v| *v += step * d),
                _ => q.iter_mut().flatten().for_each(|v| *v += step * 0.1),
            }
            dot(&value(&matrices, &incident, ks, zs, &q), &g)
        });
        let analytic = match parameter {
            0 => along(gradient.matrices.iter().flatten().sum()),
            1 => along(gradient.incident.iter().sum()),
            2 => along(gradient.ks.iter().flatten().sum()),
            3 => along(gradient.zs.iter().sum()),
            _ => gradient.q.iter().flatten().sum::<f64>() * 0.1,
        };
        let tolerance = 2e-8 * (1.0 + analytic.abs());
        prop_assert_close!(analytic, fd, tolerance, "parameter {parameter}");
    }
    Ok(())
}

/// Chirality density and oriented chirality average additively over adjacent
/// intervals. Chirality density obeys the Euler identity of joint medium and normal
/// wavenumber and interval scaling, oriented chirality that of joint transverse and
/// normal wavenumber and interval scaling.
fn check_chirality_intervals(k: f64, q: f64, stop: f64, axis: usize) -> Result<(), TestCaseError> {
    let z = [-0.2, stop];
    let g = DMatrix::from_element(3, 1, Complex::new(0.3, 0.2));
    let additive = |value: &DMatrix<Complex>, [left, right]: [DMatrix<Complex>; 2]| {
        prop_assert_close!(
            value * Complex::new(stop + 0.2, 0.0),
            left * Complex::new(0.2, 0.0) + right * Complex::new(stop, 0.0),
            1e-12
        );
        Ok(())
    };
    let halves = [[-0.2, 0.0], [0.0, stop]];
    let ks = vec![Complex::new(k, 0.1)];
    let normal = vec![Complex::new(k * 0.9, 0.12)];
    let density = |z| smatrix::chirality_density(ks.clone(), normal.clone(), z).unwrap();
    let (value, residual) = density(z);
    additive(&value, halves.map(|z| density(z).0))?;
    let gradient = residual.pullback(&g).unwrap();
    let spectral = re_dot(&gradient.ks, &ks) + re_dot(&gradient.normal, &normal);
    prop_assert_close!(spectral, dot(gradient.interval, z), 1e-12);
    let transverse = vec![[q, -0.3]];
    let normal = vec![Complex::new(1.2, 0.15)];
    let oriented = |z| {
        smatrix::oriented_chirality(transverse.clone(), normal.clone(), vec![1], axis, z).unwrap()
    };
    let (value, residual) = oriented(z);
    additive(&value, halves.map(|z| oriented(z).0))?;
    let gradient = residual.pullback(&g).unwrap();
    let scale = dot(
        gradient.transverse.iter().flatten(),
        transverse.iter().flatten(),
    ) + re_dot(&gradient.normal, &normal);
    prop_assert_close!(scale, dot(gradient.interval, z), 1e-12);
    Ok(())
}

/// Internal illumination agrees with the composed S matrix `S = L * U`: the outgoing
/// fields are `S` applied to the incident fields, the internal fields solve the
/// scattering equations of both stacks, and the block and amplitude pullbacks of
/// outgoing-field cotangents `[g0, g1, 0, 0]` equal the composition pullback of
/// `[g0 u^H, g0 d^H, g1 u^H, g1 d^H]` and `S^H g`. Row-major borrowed inputs give
/// the same fields and gradients, and every pullback pairs linearly with the
/// incident amplitudes.
fn check_illumination_composition(
    lower: &Blocks,
    upper: &Blocks,
    incoming: &[DMatrix<Complex>; 2],
    g: &[DMatrix<Complex>; 4],
) -> Result<(), TestCaseError> {
    let illuminate = |g: &[DMatrix<Complex>; 4]| {
        let (fields, residual) = smatrix::illuminate(
            lower.each_ref().map(view),
            upper.each_ref().map(view),
            incoming.clone(),
        )
        .unwrap();
        let gradient = residual.pullback(g).unwrap();
        (fields, (gradient.lower, gradient.upper, gradient.incoming))
    };
    let [inc_up, inc_down] = incoming;
    let magnitude = 1.0 + lower.iter().chain(upper).map(DMatrix::norm).sum::<f64>();
    let tolerance = 1e-12 * magnitude.powi(2) * (1.0 + inc_up.norm() + inc_down.norm());
    let (fields, (gl, gu, gi)) = illuminate(g);
    let [top, bottom, up, down] = &fields;
    let s = compose(lower, upper);
    prop_assert_close!(top.clone(), &s[0] * inc_up + &s[1] * inc_down, tolerance);
    prop_assert_close!(bottom.clone(), &s[2] * inc_up + &s[3] * inc_down, tolerance);
    prop_assert_close!(up.clone(), &lower[0] * inc_up + &lower[1] * down, tolerance);
    prop_assert_close!(
        down.clone(),
        &upper[2] * up + &upper[3] * inc_down,
        tolerance
    );
    prop_assert_close!(
        top.clone(),
        &upper[0] * up + &upper[1] * inc_down,
        tolerance
    );
    prop_assert_close!(
        bottom.clone(),
        &lower[2] * inc_up + &lower[3] * down,
        tolerance
    );
    // Every field is linear in the incident amplitudes.
    let output: f64 = fields.iter().zip(g).map(|(v, g)| re_dot(g, v)).sum();
    let input = re_dot(&gi[0], inc_up) + re_dot(&gi[1], inc_down);
    let g_scale = g.iter().map(DMatrix::norm).sum::<f64>();
    prop_assert_close!(output, input, tolerance * (1.0 + g_scale));
    // Row-major borrowed storage is the path NumPy's C order takes.
    let rows = |blocks: &Blocks| blocks.clone().map(|b| b.transpose());
    let (row_lower, row_upper) = (rows(lower), rows(upper));
    let (row_fields, row_residual) = smatrix::illuminate(
        row_lower.each_ref().map(row_view),
        row_upper.each_ref().map(row_view),
        incoming.clone(),
    )
    .unwrap();
    prop_assert_close!(row_fields, fields.clone(), tolerance);
    let smatrix::IlluminateGradient {
        lower: row_gl,
        upper: row_gu,
        incoming: row_gi,
    } = row_residual.pullback(g).unwrap();
    let gradient_tolerance = tolerance * (1.0 + g_scale);
    prop_assert_close!(row_gl, gl, gradient_tolerance);
    prop_assert_close!(row_gu, gu, gradient_tolerance);
    prop_assert_close!(row_gi, gi, gradient_tolerance);
    // Outgoing-field cotangents only: compare with the composition pullback.
    let zero = DMatrix::zeros(up.nrows(), up.ncols());
    let outgoing = [g[0].clone(), g[1].clone(), zero.clone(), zero];
    let (_, (gl, gu, gi)) = illuminate(&outgoing);
    let (_, stack) = smatrix::add(lower.clone(), upper.clone()).unwrap();
    let blocks = [
        &g[0] * inc_up.adjoint(),
        &g[0] * inc_down.adjoint(),
        &g[1] * inc_up.adjoint(),
        &g[1] * inc_down.adjoint(),
    ];
    let smatrix::AddGradient {
        lower: sl,
        upper: su,
    } = stack.pullback(&blocks).unwrap();
    prop_assert_close!(gl, sl, gradient_tolerance);
    prop_assert_close!(gu, su, gradient_tolerance);
    let expected = [
        s[0].adjoint() * &g[0] + s[2].adjoint() * &g[1],
        s[1].adjoint() * &g[0] + s[3].adjoint() * &g[1],
    ];
    prop_assert_close!(gi, expected, gradient_tolerance);
    Ok(())
}

/// A thin illumination above the Krylov threshold obeys the lossless scattering
/// power identity and differentiates the physical reflection/transmission angles.
#[test]
fn thin_internal_illumination_conserves_power_and_differentiates() -> Result<(), TestCaseError> {
    let n = 512;
    let scattering = |angle: f64| {
        let transmission = DMatrix::identity(n, n) * Complex::new(angle.cos(), 0.0);
        let reflection = DMatrix::identity(n, n) * Complex::new(0.0, angle.sin());
        [
            transmission.clone(),
            reflection.clone(),
            reflection,
            transmission,
        ]
    };
    let angles = [0.13_f64, 0.21_f64];
    let lower = scattering(angles[0]);
    let upper = scattering(angles[1]);
    let incoming = [0.2, 0.7].map(|seed| crate::test_support::patterned(n, 2, seed));
    let g = [0.1, 0.4, 0.6, 0.8].map(|seed| crate::test_support::patterned(n, 2, seed));
    let (fields, residual) = smatrix::illuminate(
        lower.each_ref().map(view),
        upper.each_ref().map(view),
        incoming.clone(),
    )
    .unwrap();
    let power_in = incoming.iter().map(DMatrix::norm_squared).sum::<f64>();
    let power_out = fields[0].norm_squared() + fields[1].norm_squared();
    prop_assert_close!(power_out, power_in, 2e-13 * power_in);
    let [tl, tu] = angles.map(|angle| Complex::new(angle.cos(), 0.0));
    let [rl, ru] = angles.map(|angle| Complex::new(0.0, angle.sin()));
    let up = (&incoming[0] * tl + &incoming[1] * (rl * tu)) / (Complex::new(1.0, 0.0) - rl * ru);
    let down = &up * ru + &incoming[1] * tu;
    prop_assert_close!(&fields[2], &up, 2e-13 * up.norm());
    prop_assert_close!(&fields[3], &down, 2e-13 * down.norm());
    let gradient = residual.pullback(&g).unwrap();
    for parameter in 0..2 {
        let blocks = if parameter == 0 {
            &gradient.lower
        } else {
            &gradient.upper
        };
        let dt = Complex::new(-angles[parameter].sin(), 0.0);
        let dr = Complex::new(0.0, angles[parameter].cos());
        let analytic: f64 = blocks
            .iter()
            .zip([dt, dr, dr, dt])
            .map(|(g, derivative)| {
                g.diagonal()
                    .iter()
                    .map(|z| (z.conj() * derivative).re)
                    .sum::<f64>()
            })
            .sum();
        let fd = central(1e-5, |h| {
            let mut changed = angles;
            changed[parameter] += h;
            let [lower, upper] = changed.map(scattering);
            let fields = smatrix::illuminate_value(
                lower.each_ref().map(view),
                upper.each_ref().map(view),
                incoming.each_ref().map(view),
            )
            .unwrap();
            fields.iter().zip(&g).map(|(v, g)| re_dot(g, v)).sum()
        });
        prop_assert_close!(analytic, fd, 2e-8 * (1.0 + analytic.abs()));
    }
    Ok(())
}

/// Layer stacks obey the Euler identity of joint thickness, wavenumber and
/// transverse scaling, and are invariant under that scaling.
fn check_compact_layer_scale(k: f64, d: f64, axis: usize) -> Result<(), TestCaseError> {
    let ks = vec![
        [Complex::new(k, 0.0); 2],
        [Complex::new(1.7 * k, 0.0), Complex::new(1.8 * k, 0.0)],
        [Complex::new(k, 0.0); 2],
    ];
    let zs = [
        Complex::new(1.0, 0.0),
        Complex::new(0.7, 0.0),
        Complex::new(1.0, 0.0),
    ];
    let q = vec![[0.1, 0.2], [0.3, 0.2]];
    let (value, residual) =
        smatrix::layer_stack(ks.clone(), &zs, q.clone(), &[d], axis, false).unwrap();
    let invariant = residual
        .pushforward(&ks, &[Complex::default(); 3], &q, &[-d])
        .unwrap();
    for block in invariant.iter().flatten() {
        prop_assert_close!(block.norm(), 0.0, 1e-12);
    }
    let g = value
        .iter()
        .map(|_| std::array::from_fn(|_| DMatrix::from_element(2, 2, Complex::new(0.2, 0.1))))
        .collect();
    let gradient = residual.pullback(g).unwrap();
    let spectral = re_dot(gradient.ks.iter().flatten(), ks.iter().flatten());
    let transverse = dot(gradient.q.iter().flatten(), q.iter().flatten());
    let spatial: f64 = gradient.thickness.iter().map(|g| g * d).sum();
    prop_assert_close!(spatial, spectral + transverse, 1e-10);
    let (scaled, _) = smatrix::layer_stack(
        ks.iter().map(|v| v.map(|k| k / 1.7)).collect(),
        &zs,
        q.iter().map(|v| v.map(|q| q / 1.7)).collect(),
        &[d * 1.7],
        axis,
        false,
    )
    .unwrap();
    prop_assert_eq!(value.len(), scaled.len());
    for (a, b) in value.iter().flatten().zip(scaled.iter().flatten()) {
        prop_assert_close!(a, b, 1e-12);
    }
    Ok(())
}

/// A simultaneous material, position and thickness direction agrees with a
/// centered reference and with the real-Hermitian adjoint pairing, including
/// fixed transverse channels.
fn check_compact_layer_pushforward(
    k: f64,
    d: f64,
    axis: usize,
    fixed_q: bool,
) -> Result<(), TestCaseError> {
    let c = Complex::new;
    let ks = vec![
        [c(k, 0.02); 2],
        [c(1.7 * k, 0.04), c(1.8 * k, 0.05)],
        [c(k, 0.01); 2],
    ];
    let zs = [c(1.0, 0.01), c(0.7, 0.02), c(1.1, 0.03)];
    let q = vec![[0.1, 0.2], [0.3, 0.2]];
    let dks = [
        [c(0.2, -0.1), c(-0.1, 0.3)],
        [c(0.3, 0.2); 2],
        [c(-0.3, 0.1); 2],
    ];
    let dzs = [c(0.1, 0.2), c(-0.2, 0.1), c(0.05, -0.2)];
    let dq = [[-0.2, 0.1], [0.1, -0.1]];
    let dd = [0.13];
    let (_, residual) =
        smatrix::layer_stack(ks.clone(), &zs, q.clone(), &[d], axis, fixed_q).unwrap();
    let tangent = residual.pushforward(&dks, &dzs, &dq, &dd).unwrap();
    let g: Vec<Blocks> = [0.2, 0.7]
        .map(|seed| {
            [0.0, 0.3, 0.6, 0.9].map(|shift| crate::test_support::patterned(2, 2, seed + shift))
        })
        .into();
    let pairing = tangent
        .iter()
        .flatten()
        .zip(g.iter().flatten())
        .map(|(d, g)| re_dot(g, d))
        .sum::<f64>();
    let gradient = residual.pullback(g).unwrap();
    let adjoint = re_dot(gradient.ks.iter().flatten(), dks.iter().flatten())
        + re_dot(&gradient.zs, dzs)
        + dot(gradient.q.iter().flatten(), dq.iter().flatten())
        + dot(&gradient.thickness, dd);
    prop_assert_close!(pairing, adjoint, 2e-12 * (1.0 + adjoint.abs()));
    let evaluate = |h: f64| {
        smatrix::layer_stack(
            ks.iter()
                .zip(dks)
                .map(|(ks, dk)| std::array::from_fn(|p| ks[p] + h * dk[p]))
                .collect(),
            &std::array::from_fn::<_, 3, _>(|i| zs[i] + h * dzs[i]),
            q.iter()
                .zip(dq)
                .map(|(q, dq)| {
                    std::array::from_fn(|p| q[p] + if fixed_q { 0.0 } else { h * dq[p] })
                })
                .collect(),
            &[d + h * dd[0]],
            axis,
            fixed_q,
        )
        .unwrap()
        .0
    };
    let h = 1e-5;
    let positive = evaluate(h);
    let negative = evaluate(-h);
    for ((actual, positive), negative) in tangent
        .iter()
        .flatten()
        .zip(positive.iter().flatten())
        .zip(negative.iter().flatten())
    {
        let reference = (positive - negative) / c(2.0 * h, 0.0);
        prop_assert_close!(actual, &reference, 1e-8 * (1.0 + reference.norm()));
    }
    Ok(())
}

/// An interface between identical media is exactly the identity for every transverse
/// wavevector, however close to a diffraction threshold, so its wavenumber and
/// impedance gradients are antisymmetric in the two sides and its transverse
/// gradient vanishes.
fn check_identical_media_interface(
    ks: [[Complex; 2]; 2],
    z: Complex,
    q: [f64; 2],
    axis: usize,
    g: &Blocks,
) -> Result<(), TestCaseError> {
    let (value, residual) = smatrix::interface(ks, [z; 2], q, axis, false).unwrap();
    prop_assert_eq!(&value, &identity_blocks(2));
    let smatrix::InterfaceGradient {
        ks: gk,
        z: gz,
        q: gq,
    } = residual.pullback(g).unwrap();
    let scale = gk
        .iter()
        .flatten()
        .chain(&gz)
        .map(|g| g.norm())
        .sum::<f64>();
    let tolerance = 1e-12 * (1.0 + scale);
    for (a, b) in gk[0].iter().zip(gk[1]) {
        prop_assert_close!(*a, -b, tolerance);
    }
    prop_assert_close!(gz[0], -gz[1], tolerance);
    prop_assert_close!(gq.to_vec(), vec![0.0; 2], tolerance);
    Ok(())
}

/// Exactly at a diffraction threshold, identical media stay transparent while the
/// derivative of the grazing channel is undefined.
fn check_threshold_interface(
    k: f64,
    z: f64,
    component: usize,
    negative: bool,
    axis: usize,
) -> Result<(), TestCaseError> {
    let ks = [[Complex::new(k, 0.0), Complex::new(1.3 * k, 0.1)]; 2];
    let mut q = [0.0; 2];
    q[component] = if negative { -k } else { k };
    let (value, residual) =
        smatrix::interface(ks, [Complex::new(z, 0.0); 2], q, axis, true).unwrap();
    prop_assert_eq!(&value, &identity_blocks(2));
    let error = residual.pullback(&value).unwrap_err().to_string();
    prop_assert!(error.contains("diffraction threshold"), "{error}");
    Ok(())
}

/// Radiation S-matrices are linear in the response and in each channel group, so
/// each pullback pairs with its input to the loss; a zero response radiates nothing.
fn check_radiation_multilinearity(
    response: &DMatrix<Complex>,
    channels: &Blocks,
    g: &Blocks,
) -> Result<(), TestCaseError> {
    let (mut scattered, forward) = smatrix::from_array(response.clone(), channels.clone()).unwrap();
    for b in [0, 3] {
        scattered[b] -= DMatrix::identity(2, 2);
    }
    let loss: f64 = g.iter().zip(&scattered).map(|(g, v)| re_dot(g, v)).sum();
    let smatrix::FromArrayGradient {
        response: gt,
        channels: gc,
    } = forward.pullback(g).unwrap();
    prop_assert_close!(re_dot(&gt, response), loss, 1e-12);
    for group in [0, 2] {
        let pairing: f64 = (group..group + 2)
            .map(|i| re_dot(&gc[i], &channels[i]))
            .sum();
        prop_assert_close!(pairing, loss, 1e-12);
    }
    let (transparent, empty) = smatrix::from_array(DMatrix::zeros(3, 3), channels.clone()).unwrap();
    prop_assert_eq!(&transparent, &identity_blocks(2));
    let gc = empty.pullback(g).unwrap().channels;
    for (gradient, channel) in gc.iter().zip(channels) {
        prop_assert_eq!(gradient, &DMatrix::zeros(channel.nrows(), channel.ncols()));
    }
    Ok(())
}

/// Fresnel coefficients are homogeneous of degree zero in the wavenumbers and in the
/// impedances.
fn check_fresnel_dimensionless(k: f64, q: f64, z: f64) -> Result<(), TestCaseError> {
    let ks = [
        [Complex::new(k, 0.1), Complex::new(k + 0.2, 0.1)],
        [Complex::new(k + 0.5, 0.2), Complex::new(k + 0.8, 0.2)],
    ];
    let kz = ks.map(|r| r.map(|k| (k * k - q * q).sqrt()));
    let impedance = [Complex::new(z, 0.03), Complex::new(z + 0.2, 0.04)];
    let (_, result) = smatrix::fresnel(ks, kz, impedance).unwrap();
    let g = std::array::from_fn(|b| {
        DMatrix::from_fn(2, 2, |i, j| {
            Complex::new(if b == i + j { 0.7 } else { 0.2 }, 0.1)
        })
    });
    let smatrix::FresnelGradient {
        ks: gk,
        kz: gkz,
        z: gz,
    } = result.pullback(&g).unwrap();
    let euler: Complex = gk
        .iter()
        .flatten()
        .zip(ks.iter().flatten())
        .chain(gkz.iter().flatten().zip(kz.iter().flatten()))
        .map(|(g, k)| g.conj() * k)
        .sum();
    prop_assert_close!(euler, Complex::default(), 1e-10);
    let impedance_euler: Complex = gz.iter().zip(impedance).map(|(g, z)| g.conj() * z).sum();
    prop_assert_close!(impedance_euler, Complex::default(), 1e-10);
    Ok(())
}

/// Propagation is a translation group, `P(r1) * P(r2) = P(r1 + r2)` for decaying
/// waves and forward displacements; its value depends on `k . r` alone, so the
/// wavevector and displacement cotangents obey the Euler identity
/// `Re Σ conj(gk) k = Σ gr r`; lossless propagation is a pure phase whose power
/// cotangent `2 P` has zero real gradients.
fn check_propagation_group(
    vectors: &[[Complex; 3]],
    first: [f64; 3],
    second: [f64; 3],
    g: &Blocks,
) -> Result<(), TestCaseError> {
    let propagate = |r: [f64; 3]| smatrix::propagation(vectors.to_vec(), r).unwrap();
    let total = std::array::from_fn(|a| first[a] + second[a]);
    let (value, residual) = propagate(total);
    let composed = compose(&propagate(first).0, &propagate(second).0);
    prop_assert_close!(composed, value.clone(), block_tolerance(&value));
    let smatrix::PropagationGradient {
        vectors: gk,
        distance: gr,
    } = residual.pullback(g).unwrap();
    let spectral = re_dot(gk.iter().flatten(), vectors.iter().flatten());
    let spatial = dot(gr, total);
    let scale = 1.0 + g.iter().map(DMatrix::norm).sum::<f64>();
    prop_assert_close!(
        spectral,
        spatial,
        1e-12 * scale * (1.0 + dot(total, total).sqrt())
    );
    let lossless: Vec<_> = vectors
        .iter()
        .map(|k| k.map(|k| Complex::new(k.re, 0.0)))
        .collect();
    let (value, residual) = smatrix::propagation(lossless, total).unwrap();
    let power = value.map(|b| b * Complex::new(2.0, 0.0));
    let smatrix::PropagationGradient {
        vectors: gk,
        distance: gr,
    } = residual.pullback(&power).unwrap();
    prop_assert_close!(gr.to_vec(), vec![0.0; 3], 1e-12);
    let real: Vec<_> = gk.iter().flatten().map(|g| g.re).collect();
    let zeros = vec![0.0; real.len()];
    prop_assert_close!(real, zeros, 1e-12);
    Ok(())
}

/// `tr(M^k)`, `k = 1..=2n`, of a `2n` by `2n` transfer matrix; by Newton's identities
/// they fix its characteristic polynomial, so equal traces mean equal spectra.
fn power_traces(m: &DMatrix<Complex>) -> Vec<Complex> {
    let mut power = m.clone();
    (0..m.nrows())
        .map(|_| {
            let trace = power.trace();
            power = &power * m;
            trace
        })
        .collect()
}

/// Bloch multipliers of a periodic cell are invariant under cyclic reordering of
/// the cell, `spec M(A * B) = spec M(B * A)`, and doubling the cell squares them,
/// `spec M(A * A) = spec M(A)^2`. The transfer matrices are not products under
/// composition; only their spectra are compared.
fn check_bloch_spectra(a: &Blocks, b: &Blocks) -> Result<(), TestCaseError> {
    let transfer = |s: Blocks| smatrix::periodic(s).unwrap().0;
    let ab = transfer(compose(a, b));
    let ba = transfer(compose(b, a));
    let tolerance = |m: &DMatrix<Complex>, power: i32| {
        1e-12 * (1.0 + f64::from(u32::try_from(m.nrows()).unwrap()) * m.norm().powi(power))
    };
    let order = i32::try_from(ab.nrows()).unwrap();
    prop_assert_close!(power_traces(&ab), power_traces(&ba), tolerance(&ab, order));
    let single = transfer(a.clone());
    let doubled = transfer(compose(a, a));
    let squared = &single * &single;
    prop_assert_close!(
        power_traces(&doubled),
        power_traces(&squared),
        tolerance(&single, 2 * order)
    );
    Ok(())
}

/// The transfer-matrix pullback pairs with an arbitrary block direction as a central
/// difference.
fn check_periodic_adjoint(
    a: &Blocks,
    da: &Blocks,
    g: &DMatrix<Complex>,
) -> Result<(), TestCaseError> {
    let (_, residual) = smatrix::periodic(a.clone()).unwrap();
    let gradient = residual.pullback(g).unwrap();
    let analytic: f64 = gradient.iter().zip(da).map(|(g, d)| re_dot(g, d)).sum();
    let numerical = central(1e-5, |step| {
        let shifted = std::array::from_fn(|b| &a[b] + &da[b] * Complex::new(step, 0.0));
        re_dot(g, &smatrix::periodic(shifted).unwrap().0)
    });
    prop_assert_close!(analytic, numerical, 1e-8 * (1.0 + analytic.abs()));
    Ok(())
}

/// A homogeneous layer of thickness `d` repeated with period `p` has the Bloch
/// wavenumbers `+-kz d / p`; chained through the propagation pullback, a wavenumber
/// cotangent `g` gives `bar kz = (g(+kz) - g(-kz)) d / p`, and the wavenumbers depend
/// on `d / p` alone, so `d bar_d + p bar_p = 0`.
fn check_propagation_bands(
    kz: &[Complex],
    g: &[Complex],
    distance: f64,
    period: f64,
) -> Result<(), TestCaseError> {
    let vectors = kz
        .iter()
        .map(|&kz| [Complex::default(), Complex::default(), kz])
        .collect();
    let (value, propagation) = smatrix::propagation(vectors, [0.0, 0.0, distance]).unwrap();
    let bands = smatrix::bands(value, period).unwrap();
    let ratio = distance / period;
    // Label every Bloch wavenumber with the mode and sign it recovers.
    let labels: Vec<(usize, f64)> = bands
        .wavenumbers()
        .iter()
        .map(|&w| {
            let candidates = kz
                .iter()
                .enumerate()
                .flat_map(|(i, &k)| [(i, 1.0, k), (i, -1.0, k)]);
            let (i, sign, k) = candidates
                .min_by(|x, y| {
                    let distance = |(_, s, k): &(usize, f64, Complex)| (w - k * s * ratio).norm();
                    distance(x).total_cmp(&distance(y))
                })
                .unwrap();
            prop_assert_close!(w, k * sign * ratio, 1e-12);
            Ok((i, sign))
        })
        .collect::<Result<_, TestCaseError>>()?;
    let mut expected = vec![Complex::default(); kz.len()];
    for (&(i, sign), &g) in labels.iter().zip(g) {
        expected[i] += g * sign * ratio;
    }
    let dimension = 2 * kz.len();
    let smatrix::BandsGradient {
        blocks,
        period: period_gradient,
    } = bands
        .pullback(g, DMatrix::zeros(dimension, dimension))
        .unwrap();
    let smatrix::PropagationGradient {
        vectors: gk,
        distance: gr,
    } = propagation.pullback(&blocks).unwrap();
    let normal: Vec<_> = gk.iter().map(|g| g[2]).collect();
    let scale = 1.0 + g.iter().map(|g| g.norm()).sum::<f64>();
    prop_assert_close!(normal, expected, 1e-12 * scale);
    prop_assert_close!(
        distance * gr[2] + period * period_gradient,
        0.0,
        1e-12 * scale
    );
    Ok(())
}

/// Scattering blocks of `n` modes near transparency: transmissions `I + X` and
/// reflections with spectral norms at most 1/2, so compositions, transfer matrices
/// and their solves are well conditioned.
fn scattering(n: usize) -> impl Strategy<Value = Blocks> {
    let scale = Complex::new(1.0 / f64::from(u32::try_from(n).unwrap()), 0.0);
    prop::array::uniform4(complex_matrix(n, n, 0.35)).prop_map(move |[t, r, s, u]| {
        let identity = DMatrix::identity(n, n);
        [
            &identity + t * scale,
            r * scale,
            s * scale,
            identity + u * scale,
        ]
    })
}

/// The `2n` by `2n` matrix `[[b0, b1], [b2, b3]]` of four blocks.
fn full(blocks: &Blocks) -> DMatrix<Complex> {
    let n = blocks[0].nrows();
    let mut matrix = DMatrix::zeros(2 * n, 2 * n);
    for (b, block) in blocks.iter().enumerate() {
        matrix
            .view_mut((n * (b / 2), n * (b % 2)), (n, n))
            .copy_from(block);
    }
    matrix
}

/// The four blocks of a `2n` by `2n` matrix.
fn split(matrix: &DMatrix<Complex>) -> Blocks {
    let n = matrix.nrows() / 2;
    std::array::from_fn(|b| {
        matrix
            .view((n * (b / 2), n * (b % 2)), (n, n))
            .clone_owned()
    })
}

/// `t^T` read from the column-major data of `t` as row-major storage, the layout
/// of a C-ordered `NumPy` block.
fn row_view(t: &DMatrix<Complex>) -> MatRef<'_, Complex> {
    MatRef::from_row_major_slice(t.as_slice(), t.ncols(), t.nrows())
}

/// Composition value, panicking on invalid input.
fn compose(lower: &Blocks, upper: &Blocks) -> Blocks {
    smatrix::add(lower.clone(), upper.clone()).unwrap().0
}

/// Distance tolerance `1e-12 (1 + |a|)` for blocks of magnitude `|a|`.
fn block_tolerance(blocks: &Blocks) -> f64 {
    1e-12 * (1.0 + blocks.iter().map(DMatrix::norm).sum::<f64>())
}

/// The Redheffer star product is a monoid with the transparent S matrix as unit,
/// commutes with the up/down mirror `[S0, S1, S2, S3] -> [S3, S2, S1, S0]` as
/// `flip(A * B) = flip(B) * flip(A)`, and preserves reciprocity
/// (`S3 = S0^T`, symmetric reflections).
fn check_redheffer_algebra(a: &Blocks, b: &Blocks, c: &Blocks) -> Result<(), TestCaseError> {
    let n = a[0].nrows();
    let tolerance = block_tolerance(a) + block_tolerance(b) + block_tolerance(c);
    let unit = identity_blocks(n);
    prop_assert_close!(compose(&unit, a), a.clone(), tolerance);
    prop_assert_close!(compose(a, &unit), a.clone(), tolerance);
    let left = compose(&compose(a, b), c);
    let right = compose(a, &compose(b, c));
    prop_assert_close!(left, right, tolerance);
    let flip = |[s0, s1, s2, s3]: Blocks| [s3, s2, s1, s0];
    let mirrored = compose(&flip(b.clone()), &flip(a.clone()));
    prop_assert_close!(flip(compose(a, b)), mirrored, tolerance);
    let reciprocal = |[s0, s1, s2, _]: &Blocks| -> Blocks {
        let half = Complex::new(0.5, 0.0);
        [
            s0.clone(),
            (s1 + s1.transpose()) * half,
            (s2 + s2.transpose()) * half,
            s0.transpose(),
        ]
    };
    let [t_up, r_up, r_down, t_down] = compose(&reciprocal(a), &reciprocal(b));
    prop_assert_close!(t_down, t_up.transpose(), tolerance);
    prop_assert_close!(r_up.transpose(), r_up, tolerance);
    prop_assert_close!(r_down.transpose(), r_down, tolerance);
    Ok(())
}

/// The composition pullback pairs with arbitrary directions of both operands as a
/// central difference of the loss `Re <g, A * B>`.
fn check_redheffer_adjoint(
    a: &Blocks,
    b: &Blocks,
    g: &Blocks,
    da: &Blocks,
    db: &Blocks,
) -> Result<(), TestCaseError> {
    let (_, residual) = smatrix::add(a.clone(), b.clone()).unwrap();
    let smatrix::AddGradient {
        lower: ga,
        upper: gb,
    } = residual.pullback(g).unwrap();
    let analytic: f64 = (0..4)
        .map(|i| re_dot(&ga[i], &da[i]) + re_dot(&gb[i], &db[i]))
        .sum();
    let numerical = central(1e-5, |step| {
        let step = Complex::new(step, 0.0);
        let a = std::array::from_fn(|i| &a[i] + &da[i] * step);
        let b = std::array::from_fn(|i| &b[i] + &db[i] * step);
        let value = compose(&a, &b);
        g.iter().zip(&value).map(|(g, v)| re_dot(g, v)).sum::<f64>()
    });
    prop_assert_close!(analytic, numerical, 1e-8 * (1.0 + analytic.abs()));
    Ok(())
}

/// Lossless S matrices are unitary, and composing two keeps the product unitary.
/// Both are Cayley transforms `(I - iH)(I + iH)^-1` of small Hermitian `H`.
fn check_redheffer_unitarity(
    x: &DMatrix<Complex>,
    y: &DMatrix<Complex>,
) -> Result<(), TestCaseError> {
    let cayley = |m: &DMatrix<Complex>| {
        let h = (m + m.adjoint()) * Complex::new(0.0, 0.5);
        let identity = DMatrix::identity(m.nrows(), m.nrows());
        split(&((&identity - &h) * (identity + &h).try_inverse().unwrap()))
    };
    let (a, b) = (cayley(x), cayley(y));
    for s in [&a, &b] {
        let s = full(s);
        prop_assert_close!(
            s.adjoint() * &s,
            DMatrix::identity(s.nrows(), s.nrows()),
            1e-13
        );
    }
    let product = full(&compose(&a, &b));
    let identity = DMatrix::identity(product.nrows(), product.nrows());
    prop_assert_close!(product.adjoint() * &product, identity, 1e-12);
    Ok(())
}

/// A zero-thickness layer has no effect: interfaces compose transitively,
/// `A|B * B|C = A|C`, and invert, `A|B * B|A = I`, on every Cartesian axis.
fn check_interface_algebra(
    media: [([Complex; 2], Complex); 3],
    q: [f64; 2],
    axis: usize,
) -> Result<(), TestCaseError> {
    let interface = |i: usize, j: usize| {
        smatrix::interface(
            [media[i].0, media[j].0],
            [media[i].1, media[j].1],
            q,
            axis,
            false,
        )
        .unwrap()
        .0
    };
    let direct = interface(0, 2);
    let tolerance = block_tolerance(&direct);
    prop_assert_close!(
        compose(&interface(0, 1), &interface(1, 2)),
        direct,
        tolerance
    );
    prop_assert_close!(
        compose(&interface(0, 1), &interface(1, 0)),
        identity_blocks(2),
        tolerance
    );
    Ok(())
}

/// Two independent derivations agree: the Cartesian tangential matching at the z
/// axis equals the explicit chiral Fresnel formula, and its pullback equals the
/// Fresnel pullback chained through the dispersion `kz = sqrt(k^2 - |q|^2)`:
/// `gk = gk_F + gkz_F conj(k / kz)` and `gq_a = Re Σ conj(gkz_F) (-q_a / kz)`.
fn check_interface_fresnel(
    media: [([Complex; 2], Complex); 2],
    q: [f64; 2],
    g: &Blocks,
) -> Result<(), TestCaseError> {
    let ks = media.map(|m| m.0);
    let zs = media.map(|m| m.1);
    let kz = ks.map(|k| k.map(|k| wave_vector_z(q[0].into(), q[1].into(), k)));
    let (value, interface) = smatrix::interface(ks, zs, q, 2, false).unwrap();
    let (fresnel_value, fresnel) = smatrix::fresnel(ks, kz, zs).unwrap();
    prop_assert_close!(
        value,
        fresnel_value.clone(),
        block_tolerance(&fresnel_value)
    );
    let smatrix::InterfaceGradient {
        ks: gk,
        z: gz,
        q: gq,
    } = interface.pullback(g).unwrap();
    let smatrix::FresnelGradient {
        ks: gk_f,
        kz: gkz_f,
        z: gz_f,
    } = fresnel.pullback(g).unwrap();
    let scale = 1.0
        + gk_f
            .iter()
            .flatten()
            .chain(gkz_f.iter().flatten())
            .chain(&gz_f)
            .map(|v| v.norm())
            .sum::<f64>();
    let mut expected_q = [0.0; 2];
    for side in 0..2 {
        for pol in 0..2 {
            let chained =
                gk_f[side][pol] + gkz_f[side][pol] * (ks[side][pol] / kz[side][pol]).conj();
            prop_assert_close!(gk[side][pol], chained, 1e-12 * scale);
            for (target, &q) in expected_q.iter_mut().zip(&q) {
                *target += (gkz_f[side][pol].conj() * (-q / kz[side][pol])).re;
            }
        }
        prop_assert_close!(gz[side], gz_f[side], 1e-12 * scale);
    }
    prop_assert_close!(gq.to_vec(), expected_q.to_vec(), 1e-12 * scale);
    Ok(())
}

/// Splitting a layer into two sublayers of the same medium changes neither the
/// stack nor its gradients: each sublayer thickness receives the whole thickness
/// gradient, the two copies of the medium share its wavenumber and impedance
/// gradients, and the transverse gradients are unchanged.
fn check_layer_split(
    media: [([Complex; 2], Complex); 3],
    thickness: f64,
    fraction: f64,
    q: &[[f64; 2]],
    g: Vec<Blocks>,
    axis: usize,
    fixed_q: bool,
) -> Result<(), TestCaseError> {
    let [a, b, c] = media;
    let (ks, zs): (Vec<_>, Vec<_>) = [a, b, c].into_iter().unzip();
    let (value, residual) =
        smatrix::layer_stack(ks, &zs, q.to_vec(), &[thickness], axis, fixed_q).unwrap();
    let (ks, zs): (Vec<_>, Vec<_>) = [a, b, b, c].into_iter().unzip();
    let sublayers = [fraction * thickness, (1.0 - fraction) * thickness];
    let (split, split_residual) =
        smatrix::layer_stack(ks, &zs, q.to_vec(), &sublayers, axis, fixed_q).unwrap();
    for (value, split) in value.iter().zip(&split) {
        prop_assert_close!(split.clone(), value.clone(), block_tolerance(value));
    }
    let whole = residual.pullback(g.clone()).unwrap();
    let parts = split_residual.pullback(g).unwrap();
    let scale = 1.0
        + whole
            .ks
            .iter()
            .flatten()
            .chain(&whole.zs)
            .map(|v| v.norm())
            .sum::<f64>()
        + whole
            .q
            .iter()
            .flatten()
            .chain(&whole.thickness)
            .map(|v| v.abs())
            .sum::<f64>();
    let tolerance = 1e-12 * scale;
    for sublayer in &parts.thickness {
        prop_assert_close!(*sublayer, whole.thickness[0], tolerance);
    }
    let merged = [0, 1, 3].map(|m| {
        if m == 1 {
            [0, 1].map(|p| parts.ks[1][p] + parts.ks[2][p])
        } else {
            parts.ks[m]
        }
    });
    for (merged, whole) in merged.iter().zip(&whole.ks) {
        prop_assert_close!(merged.to_vec(), whole.to_vec(), tolerance);
    }
    let merged = [parts.zs[0], parts.zs[1] + parts.zs[2], parts.zs[3]];
    prop_assert_close!(merged.to_vec(), whole.zs.clone(), tolerance);
    let flat = |v: &[[f64; 2]]| v.iter().flatten().copied().collect::<Vec<_>>();
    prop_assert_close!(flat(&parts.q), flat(&whole.q), tolerance);
    Ok(())
}

/// Between identical achiral lossless outer media the helicity channels carry equal
/// power per amplitude, so a slab with a lossless, possibly chiral, interior has a
/// unitary S matrix, and with loss every singular value is at most one.
fn check_slab_energy(
    outer: (f64, f64),
    inner: (f64, f64, f64),
    loss: f64,
    thickness: f64,
    q: [f64; 2],
    axis: usize,
) -> Result<(), TestCaseError> {
    let outside = planar_medium(&lossless(outer.0, outer.1, 0.0));
    let slab = |epsilon: Complex| {
        let inside = planar_medium(&Material {
            epsilon,
            mu: Complex::new(inner.1, 0.0),
            kappa: Complex::new(inner.2, 0.0),
        });
        let (ks, zs): (Vec<_>, Vec<_>) = [outside, inside, outside].into_iter().unzip();
        let (value, _) = smatrix::layer_stack(ks, &zs, vec![q], &[thickness], axis, false).unwrap();
        full(&value[0])
    };
    let lossless = slab(Complex::new(inner.0, 0.0));
    prop_assert_close!(
        lossless.adjoint() * &lossless,
        DMatrix::identity(4, 4),
        1e-12
    );
    let lossy = slab(Complex::new(inner.0, loss));
    let largest = linalg::svdvals(&lossy)
        .unwrap()
        .values()
        .iter()
        .copied()
        .fold(0.0, f64::max);
    prop_assert!(largest <= 1.0 + 1e-12, "largest singular value {largest}");
    Ok(())
}

/// Reciprocity of a lossy stack between distinct lossless outer media: the power
/// transmitted upward equals that transmitted downward for each incident helicity
/// of an achiral stack, and for TE and TM (the parity basis of an xy stack), and
/// summed over the polarizations of a chiral stack. Reflectances generally differ.
fn check_reciprocity(
    below: (f64, f64),
    above: (f64, f64),
    interior: [Material; 2],
    thickness: [f64; 2],
    q: [f64; 2],
    axis: usize,
    chiral: bool,
) -> Result<(), TestCaseError> {
    let outer = |(epsilon, mu): (f64, f64)| planar_medium(&lossless(epsilon, mu, 0.0));
    let inner = interior.map(|material| {
        planar_medium(&Material {
            kappa: if chiral {
                material.kappa
            } else {
                Complex::default()
            },
            ..material
        })
    });
    let media = [outer(below), inner[0], inner[1], outer(above)];
    let (ks, zs): (Vec<_>, Vec<_>) = media.into_iter().unzip();
    let (value, _) = smatrix::layer_stack(ks, &zs, vec![q], &thickness, axis, false).unwrap();
    // Port order is (above, below).
    let ports_k = [media[3].0, media[0].0];
    let ports_z = [media[3].1, media[0].1];
    let power = |blocks: &Blocks, helicity: bool, direction: usize| {
        let t = direction;
        let matrices = [view(&blocks[3 * t]), view(&blocks[2 - t])];
        let ports = smatrix::TrPorts {
            ks: ports_k,
            zs: ports_z,
            q: vec![q],
            modes: vec![(0, 0), (0, 1)],
            axis,
            helicity,
            direction: t,
        };
        smatrix::tr_value(matrices, &DMatrix::identity(2, 2), &ports).unwrap()
    };
    let transmitted = |power: &DMatrix<f64>| vec![power[(0, 0)], power[(0, 1)]];
    let [up, down] = [0, 1].map(|t| power(&value[0], true, t));
    if chiral {
        prop_assert_close!(up[(0, 0)] + up[(0, 1)], down[(0, 0)] + down[(0, 1)], 1e-12);
    } else {
        prop_assert_close!(transmitted(&up), transmitted(&down), 1e-12);
    }
    if !chiral && axis == 2 {
        // The parity polarizations are TE and TM only for the z normal; there the
        // parity amplitudes are a_hel = C a_par with the orthogonal symmetric C.
        let c = DMatrix::from_row_slice(
            2,
            2,
            &[-1.0, 1.0, 1.0, 1.0].map(|v| Complex::new(v * std::f64::consts::FRAC_1_SQRT_2, 0.0)),
        );
        let parity = value[0].clone().map(|b| &c * b * &c);
        let [up, down] = [0, 1].map(|t| power(&parity, false, t));
        prop_assert_close!(transmitted(&up), transmitted(&down), 1e-12);
    }
    Ok(())
}

/// Evanescent propagation against the decay direction overflows and is reported as
/// such, and a power mode listed twice is rejected instead of being counted twice.
#[test]
fn planar_kernels_reject_overflow_and_repeated_modes() {
    let evanescent = vec![[
        Complex::default(),
        Complex::default(),
        Complex::new(1.0, 1.0),
    ]];
    let error = smatrix::propagation(evanescent, [0.0, 0.0, -1e4]).unwrap_err();
    assert!(matches!(error, crate::Error::NonFinite(_)), "{error}");
    assert!(error.to_string().contains("non-finite propagation phase"));
    let identity = DMatrix::<Complex>::identity(2, 2);
    let zero = DMatrix::<Complex>::zeros(2, 2);
    let power = |modes: &[(usize, u8)]| {
        let ports = smatrix::TrPorts {
            ks: [[Complex::new(1.0, 0.0); 2]; 2],
            zs: [Complex::new(1.0, 0.0); 2],
            q: vec![[0.1, 0.2]],
            modes: modes.to_vec(),
            axis: 2,
            helicity: true,
            direction: 0,
        };
        smatrix::tr_value(
            [view(&identity), view(&zero)],
            &DMatrix::from_element(2, 1, Complex::new(1.0, 0.0)),
            &ports,
        )
    };
    let value = power(&[(0, 0), (0, 1)]).unwrap();
    assert!((value[(0, 0)] - 1.0).abs() < 1e-14 && value[(1, 0)].abs() < 1e-14);
    let error = power(&[(0, 1), (0, 1)]).unwrap_err().to_string();
    assert!(error.contains("plane modes must be distinct"), "{error}");
}
