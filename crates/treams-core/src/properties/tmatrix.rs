//! T-matrices of spheres, cylinders, EBCM particles and clusters.

#[path = "tmatrix/iterative_ebcm_forward.rs"]
mod iterative_ebcm_forward;
#[path = "tmatrix_pushforward.rs"]
mod pushforward;

use std::{collections::HashMap, ops::RangeInclusive};

use nalgebra::DMatrix;
use proptest::{prelude::*, test_runner::TestCaseError};

use crate::{
    Complex,
    cluster::{self, interaction},
    coeffs::{self, LayerGradient, Material, Matrix2, mie, to_mode_order},
    cw,
    ebcm::{Surface, qmat},
    numerics::{label_bits, parity},
    rotation,
    special::Radial,
    sw::{self, Basis, Mode},
    test_support::{
        Chirality, DEFAULT_CASES, EXPENSIVE_CASES, central, complex, complex_matrix, dot,
        gauss_legendre, helicity_ks, log_polar, material, patterned, prop_assert_close, radial,
        re_dot, rotate, spherical_basis,
    },
    tmatrix::{self, Metric, metric},
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn local_block_adjoint_matches_dense(
        (blocks, coupling, g) in prop::collection::vec(1_usize..=3, 1..=3).prop_flat_map(|sizes| {
            let n = sizes.iter().sum();
            (
                sizes.iter().map(|&m| complex_matrix(m, m, 0.3)).collect::<Vec<_>>(),
                complex_matrix(n, n, 0.1),
                complex_matrix(n, n, 1.0),
            )
        }),
    ) {
        check_local_block_adjoint(blocks, coupling, &g)?;
    }

    #[test]
    fn helicity_metrics_invariances_and_adjoint(
        (lmax, perturbation) in (1_u32..=2).prop_flat_map(|lmax| {
            let n = usize::try_from(2 * lmax * (lmax + 2)).unwrap();
            (Just(lmax), complex_matrix(n, n, 0.05))
        }),
        scale in log_polar(-1.0..1.0, -3.2..3.2),
        ks in prop::array::uniform2(0.8_f64..2.0),
        angles in prop::array::uniform3(-3.2_f64..3.2),
    ) {
        check_helicity_metrics(lmax, &perturbation, scale, ks, angles)?;
    }

    #[test]
    fn sphere_contrast_and_layer_split(
        size in 0.2_f64..3.0,
        material in chiral(1.0, 1.0),
        embedding in chiral(0.0, 1.0),
        fraction in 0.2_f64..0.8,
        l in 1_u32..9,
    ) {
        check_sphere_contrast_and_split(size, material, embedding, fraction, l)?;
    }

    #[test]
    fn sphere_matrix_energy_balance_and_rotation_invariance(
        lmax in 1_u32..=4,
        (radii, materials) in layers(0.0, 1.0),
        embedding in prop_oneof![achiral(0.0), chiral(0.0, 1.0)],
        k0 in 0.5_f64..2.0,
        seed in -2.0_f64..2.0,
        angles in prop::array::uniform3(-3.2_f64..3.2),
    ) {
        check_sphere_matrix(lmax, k0, &radii, &materials, embedding, seed, angles)?;
    }

    #[test]
    fn interaction_adjoint_identity(
        local in complex_matrix(3, 3, 0.1),
        coupling in complex_matrix(3, 3, 0.1),
        direction in complex_matrix(3, 3, 0.1),
        g in complex_matrix(3, 3, 0.1),
    ) {
        check_interaction_adjoint(&local, &coupling, &direction, &g)?;
    }

    #[test]
    fn cylinder_symmetry_contrast_and_layer_split(
        kz in prop_oneof![Just(0.0), -0.8_f64..0.8],
        order in -8_i32..=8,
        (radii, materials) in layers(1.0, 1.0),
        embedding in chiral(1.0, 1.0),
        g in [complex(1.0), complex(1.0), complex(1.0), complex(1.0)],
    ) {
        let g = Matrix2::new(g[0], g[1], g[2], g[3]);
        check_cylinder_symmetries(kz, order, &radii, &materials, embedding, &g)?;
    }

    #[test]
    fn cylinder_matrix_energy_balance(
        kzs in prop_oneof![
            prop::array::uniform2(-0.5_f64..0.5),
            (0.05_f64..0.5).prop_map(|kz| [kz, -kz]),
            (-0.5_f64..0.5).prop_map(|kz| [0.0, kz]),
        ],
        mmax in 0_u32..=4,
        (radii, materials) in layers(0.0, 0.75),
        embedding in prop_oneof![achiral(0.0), chiral(0.0, 0.75)],
        seed in -2.0_f64..2.0,
    ) {
        prop_assume!((kzs[0] - kzs[1]).abs() > 0.0);
        check_cylinder_energy_balance(kzs, mmax, &radii, &materials, embedding, seed)?;
    }

    #[test]
    fn cylinder_reciprocity(
        kz in prop_oneof![Just(0.0), 0.05_f64..0.8],
        mmax in 0_u32..=3,
        (radii, materials) in layers(1.0, 1.0),
        embedding in chiral(1.0, 1.0),
        offset in (0.0_f64..6.3, -0.5_f64..0.5),
    ) {
        check_cylinder_reciprocity(kz, mmax, &radii, &materials, embedding, offset)?;
    }

    #[test]
    fn cylinder_matrix_pullback_matches_blocks(
        kzs in prop_oneof![
            (0.05_f64..0.8).prop_map(|kz| vec![kz, -kz]),
            Just(vec![0.0]),
            (0.05_f64..0.8).prop_map(|kz| vec![-kz, 0.0, kz]),
            (0.05_f64..0.4, prop_oneof![0.45_f64..0.8, -0.8_f64..-0.45])
                .prop_map(|(kz, other)| vec![kz, -kz, other]),
        ],
        mmax in 1_u32..=3,
        (radii, materials) in layers(1.0, 1.0),
        embedding in chiral(1.0, 1.0),
        seed in -2.0_f64..2.0,
    ) {
        check_cylinder_matrix_pullback(&kzs, mmax, &radii, &materials, embedding, seed)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(EXPENSIVE_CASES))]

    #[test]
    fn ebcm_surface_scaling_adjoint(deformation in -0.2_f64..0.2) {
        check_ebcm_surface_scaling(deformation)?;
    }

    #[test]
    fn ebcm_zero_contrast_is_shape_independent(
        radius in 0.2_f64..0.5,
        deformation in -0.3_f64..0.3,
        direction in prop::array::uniform3(-1.0_f64..1.0),
        medium in chiral(1.0, 1.0),
        seed in -2.0_f64..2.0,
    ) {
        check_ebcm_zero_contrast(radius, deformation, direction, medium, seed)?;
    }

    #[test]
    fn ebcm_sphere_matches_mie(
        lmax in 1_u32..=3,
        radius in 0.15_f64..0.45,
        interior in chiral(1.0, 1.0),
        embedding in chiral(0.0, 1.0),
    ) {
        check_ebcm_sphere(lmax, radius, interior, embedding)?;
    }

    #[test]
    fn cluster_complete_directional_derivative(
        lmax in 1_u32..=2,
        radius in 0.1_f64..0.4, eps in 1.2_f64..6.0, k in 0.7_f64..1.8, dx in -0.3_f64..0.3,
    ) {
        check_cluster_directional_derivative(lmax, radius, eps, k, dx)?;
    }

    #[test]
    fn cluster_matches_particle_cluster(spheres in spheres(1..=3, 1.0, 0.0), seed in -2.0_f64..2.0) {
        check_cluster_matches_particle_cluster(&spheres, seed)?;
    }

    #[test]
    fn translation_and_cluster_reciprocity(
        spheres in spheres(1..=3, 1.0, 1.0),
        helicity in any::<bool>(),
        radial in radial(),
    ) {
        check_sphere_reciprocity(&spheres, helicity, radial)?;
    }

    #[test]
    fn lossless_cluster_energy_balance(spheres in spheres(1..=3, 0.0, 1.0), seed in -2.0_f64..2.0) {
        check_cluster_energy_balance(&spheres, seed)?;
    }

    #[test]
    fn clusters_rotate_with_their_positions(
        spheres in spheres(1..=2, 1.0, 1.0),
        angles in prop::array::uniform3(-3.2_f64..3.2),
        seed in -2.0_f64..2.0,
    ) {
        check_cluster_rotation(&spheres, angles, seed)?;
    }
}

/// A shrunk failure of `clusters_rotate_with_their_positions`, kept as an explicit
/// case so that it survives changes of the strategies: three spheres of degree 2,
/// one of them lossy and chiral, in a chiral embedding.
#[test]
fn recorded_chiral_cluster_rotates_with_its_positions() -> Result<(), TestCaseError> {
    let achiral = Material {
        epsilon: Complex::new(1.2, 0.0),
        mu: Complex::new(0.8, 0.0),
        kappa: Complex::default(),
    };
    let spheres = Spheres {
        lmax: 2,
        k0: 0.712_480_039_355_871,
        radii: vec![0.1; 3],
        materials: vec![
            achiral,
            achiral,
            Material {
                epsilon: Complex::new(1.2, 0.0),
                mu: Complex::new(1.412_081_496_870_95, 0.072_756_740_997_889_55),
                kappa: Complex::new(-0.129_884_694_596_186_82, 0.0),
            },
        ],
        embedding: Material {
            epsilon: Complex::new(1.405_504_162_742_097, 0.0),
            mu: Complex::new(0.870_953_053_901_380_9, 0.0),
            kappa: Complex::new(-0.241_465_787_793_311_15, 0.0),
        },
        positions: vec![
            [
                -0.070_543_096_257_750_99,
                0.064_496_304_069_550_2,
                -0.093_121_769_578_649_72,
            ],
            [
                0.971_630_839_701_808,
                0.285_672_528_482_815_27,
                0.277_743_797_548_510_3,
            ],
            [
                -0.335_046_714_800_322_17,
                0.722_246_550_545_535_9,
                -0.540_888_579_795_522_3,
            ],
        ],
    };
    let angles = [
        -2.697_684_266_369_771_8,
        3.022_124_935_239_106_4,
        -2.251_592_257_866_249_7,
    ];
    check_cluster_rotation(&spheres, angles, -0.742_916_986_120_987_3)
}

/// A [`material`] with epsilon in `[1.2, 5] + i [0, 0.3 loss]`, mu in `[0.8, 1.5] + i
/// [0, 0.1 loss]` and a real chirality of at most `0.4 chirality Re n`, so both helicity
/// indices keep a positive real part.
fn chiral(loss: f64, chirality: f64) -> impl Strategy<Value = Material> + Clone {
    material(
        1.2..5.0,
        0.8..1.5,
        [0.3 * loss, 0.1 * loss],
        Chirality::Relative(0.4 * chirality),
    )
}

/// An achiral [`chiral`] material.
fn achiral(loss: f64) -> impl Strategy<Value = Material> + Clone {
    chiral(loss, 0.0)
}

/// One or two concentric boundaries from radius 0.1 up, at least 0.05 apart, and
/// their interior [`chiral`] materials.
fn layers(loss: f64, chirality: f64) -> impl Strategy<Value = (Vec<f64>, Vec<Material>)> {
    (
        0.1_f64..1.0,
        prop::option::of(0.05_f64..0.5),
        chiral(loss, chirality),
        chiral(loss, chirality),
    )
        .prop_map(|(radius, gap, inner, outer)| match gap {
            None => (vec![radius], vec![inner]),
            Some(gap) => (vec![radius, radius + gap], vec![inner, outer]),
        })
}

fn dielectric(epsilon: Complex) -> Material {
    Material {
        epsilon,
        ..Material::default()
    }
}

/// Largest violation of Lorentz reciprocity `(T K)_ab = s_a s_b (T K)_{P(b) P(a)}`
/// relative to `max |T K|`, for the reciprocity partners `(P(a), s_a)` of the modes
/// and `K = diag(weights)`.
fn reciprocity_defect(t: &DMatrix<Complex>, partners: &[(usize, f64)], weights: &[Complex]) -> f64 {
    let tk = DMatrix::from_fn(t.nrows(), t.ncols(), |a, b| t[(a, b)] * weights[b]);
    let scale = tk
        .iter()
        .map(|z| z.norm())
        .fold(f64::MIN_POSITIVE, f64::max);
    let mut defect = 0.0_f64;
    for (a, &(pa, sa)) in partners.iter().enumerate() {
        for (b, &(pb, sb)) in partners.iter().enumerate() {
            defect = defect.max((tk[(a, b)] - sa * sb * tk[(pb, pa)]).norm());
        }
    }
    defect / scale
}

/// The reciprocity partner `(p, l, -m, pol)` of each spherical mode, with sign `(-1)^m`.
fn spherical_partners(basis: &Basis) -> Vec<(usize, f64)> {
    let index: HashMap<_, _> = basis
        .modes
        .iter()
        .enumerate()
        .map(|(i, &(p, mode))| ((p, mode.l, mode.m, mode.pol), i))
        .collect();
    basis
        .modes
        .iter()
        .map(|&(p, mode)| (index[&(p, mode.l, -mode.m, mode.pol)], parity(mode.m)))
        .collect()
}

/// The reciprocity partner `(p, -kz, -m, pol)` of each cylindrical mode, with sign `(-1)^m`.
fn cylindrical_partners(basis: &cw::Basis) -> Vec<(usize, f64)> {
    let key = |p: usize, kz: f64, m: i32, pol: u8| (p, label_bits(kz), m, pol);
    let index: HashMap<_, _> = basis
        .modes
        .iter()
        .enumerate()
        .map(|(i, &(p, mode))| (key(p, mode.kz, mode.m, mode.pol), i))
        .collect();
    basis
        .modes
        .iter()
        .map(|&(p, mode)| (index[&key(p, -mode.kz, -mode.m, mode.pol)], parity(mode.m)))
        .collect()
}

/// A Hermitian matrix `A + A^H` from the deterministic pattern of `seed`.
fn hermitian(n: usize, seed: f64) -> DMatrix<Complex> {
    let a = patterned(n, n, seed);
    &a + a.adjoint()
}

/// Particle centres; jittered by at most 0.1 per coordinate they stay 0.62 apart.
const CENTRES: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [0.9, 0.2, 0.3], [-0.4, 0.8, -0.5]];

/// Spheres with local multipoles up to `lmax`, in an embedding at vacuum wavenumber `k0`.
#[derive(Clone, Debug)]
struct Spheres {
    lmax: u32,
    k0: f64,
    radii: Vec<f64>,
    materials: Vec<Material>,
    embedding: Material,
    positions: Vec<[f64; 3]>,
}

/// Two or three spheres of radius 0.1 to 0.3 at the jittered [`CENTRES`], so they never
/// touch, made of [`chiral`] materials in a lossless embedding of the same chirality.
fn spheres(
    degrees: RangeInclusive<u32>,
    loss: f64,
    chirality: f64,
) -> impl Strategy<Value = Spheres> {
    (2_usize..=3, degrees, 0.7_f64..1.8)
        .prop_flat_map(move |(count, lmax, k0)| {
            (
                Just(lmax),
                Just(k0),
                prop::collection::vec(0.1_f64..0.3, count),
                prop::collection::vec(chiral(loss, chirality), count),
                chiral(0.0, chirality),
                prop::collection::vec(prop::array::uniform3(-0.1_f64..0.1), count),
            )
        })
        .prop_map(|(lmax, k0, radii, materials, embedding, jitter)| Spheres {
            lmax,
            k0,
            radii,
            materials,
            embedding,
            positions: jitter
                .iter()
                .zip(CENTRES)
                .map(|(d, c)| std::array::from_fn(|a| c[a] + d[a]))
                .collect(),
        })
}

impl Spheres {
    /// Every mode up to `lmax` at every sphere, in [`sw::modes`] order.
    fn basis(&self) -> Basis {
        let modes = sw::modes(self.lmax).unwrap();
        Basis {
            modes: (0..self.radii.len())
                .flat_map(|p| modes.iter().map(move |&mode| (p, mode)))
                .collect(),
            positions: self.positions.clone(),
        }
    }

    fn ks(&self) -> [Complex; 2] {
        helicity_ks(self.k0, &self.embedding)
    }

    /// The wavenumber of each basis mode's helicity.
    fn mode_ks(&self) -> Vec<Complex> {
        let ks = self.ks();
        self.basis()
            .modes
            .iter()
            .map(|(_, mode)| ks[usize::from(mode.pol)])
            .collect()
    }

    fn sphere(&self, i: usize) -> (DMatrix<Complex>, tmatrix::SphereResidual) {
        let materials = [self.materials[i], self.embedding];
        tmatrix::sphere(self.lmax, self.k0, &[self.radii[i]], &materials).unwrap()
    }

    /// The Mie T-matrices with their residuals, and their interaction solve.
    fn particle_cluster(
        &self,
    ) -> (
        Vec<(DMatrix<Complex>, tmatrix::SphereResidual)>,
        cluster::ParticleClusterResidual,
    ) {
        let spheres: Vec<_> = (0..self.radii.len()).map(|i| self.sphere(i)).collect();
        let local = spheres.iter().map(|(t, _)| t.clone()).collect();
        let cluster = cluster::particle_cluster(local, self.basis(), self.ks(), true).unwrap();
        (spheres, cluster)
    }

    /// The same spheres with only their permittivities, in vacuum.
    fn in_vacuum(&self) -> Self {
        Self {
            materials: self
                .materials
                .iter()
                .map(|m| dielectric(m.epsilon))
                .collect(),
            embedding: Material::default(),
            ..self.clone()
        }
    }

    /// [`cluster::sphere_cluster`] of the spheres [`Self::in_vacuum`].
    fn sphere_cluster(&self) -> cluster::SphereClusterResidual {
        let epsilon: Vec<_> = self.materials.iter().map(|m| m.epsilon).collect();
        cluster::sphere_cluster(self.lmax, self.k0, &self.radii, &epsilon, &self.positions).unwrap()
    }
}

/// The block-diagonal interaction solve and pullback equal the dense ones, with
/// one local cotangent per block.
fn check_local_block_adjoint(
    blocks: Vec<DMatrix<Complex>>,
    coupling: DMatrix<Complex>,
    g: &DMatrix<Complex>,
) -> Result<(), TestCaseError> {
    let n = coupling.nrows();
    let mut dense = DMatrix::zeros(n, n);
    let mut offset = 0;
    for block in &blocks {
        dense
            .view_mut((offset, offset), block.shape())
            .copy_from(block);
        offset += block.nrows();
    }
    let count = blocks.len();
    let sparse = cluster::interaction_blocks(blocks, coupling.clone()).unwrap();
    let dense = interaction(dense, coupling).unwrap();
    prop_assert_close!(sparse.value(), dense.value(), 1e-14);
    let cluster::InteractionGradient {
        local: blocks,
        coupling: gc,
    } = sparse.pullback_blocks(g).unwrap();
    let cluster::InteractionGradient {
        local: gd,
        coupling: expected_gc,
    } = dense.pullback(g).unwrap();
    prop_assert_close!(gc, expected_gc, 1e-14 * g.norm());
    prop_assert_eq!(blocks.len(), count);
    let mut offset = 0;
    for block in blocks {
        let expected = gd.view((offset, offset), block.shape()).into_owned();
        prop_assert_close!(&block, &expected, 1e-14 * g.norm());
        offset += block.nrows();
    }
    prop_assert_eq!(offset, gd.nrows());
    Ok(())
}

/// The helicity metrics of an absorbing `T = -0.3 I + E` in the modes up to `lmax`
/// are invariant under rotations `D T D^H`, which keep each helicity. Duality breaking
/// and chirality lie in `[0, 1]` and are invariant under complex scaling and helicity
/// relabelling, so their gradient is orthogonal to `T` and independent of the
/// wavenumbers. Circular dichroism is homogeneous of degree zero in the wavenumbers
/// and changes sign when the helicities and their wavenumbers are swapped.
fn check_helicity_metrics(
    lmax: u32,
    perturbation: &DMatrix<Complex>,
    scale: Complex,
    ks: [f64; 2],
    angles: [f64; 3],
) -> Result<(), TestCaseError> {
    let modes = sw::modes(lmax).unwrap();
    let pol: Vec<_> = modes.iter().map(|mode| mode.pol).collect();
    let swapped: Vec<_> = pol.iter().map(|p| 1 - p).collect();
    let n = pol.len();
    let t = perturbation - DMatrix::identity(n, n) * Complex::new(0.3, 0.0);
    let basis = spherical_basis(lmax, [0.0; 3]);
    let d = rotation::sw_rotation(&basis, &basis, angles)
        .unwrap()
        .value()
        .clone();
    let rotated = &d * &t * d.adjoint();
    for kind in [
        Metric::CircularDichroism,
        Metric::DualityBreaking,
        Metric::Chirality,
    ] {
        let (value, residual) = metric(&t, &pol, ks, kind).unwrap();
        let (other, _) = metric(&rotated, &pol, ks, kind).unwrap();
        prop_assert_close!(other, value, 1e-12 * (1.0 + value.abs()), "{:?}", kind);
        let tmatrix::MetricGradient {
            matrix: gradient,
            ks: gk,
        } = residual.pullback(1.0).unwrap();
        if matches!(kind, Metric::CircularDichroism) {
            prop_assert_close!(dot(gk, ks), 0.0, 1e-12 * (1.0 + gk[0].abs() + gk[1].abs()));
            let (flipped, _) = metric(&t, &swapped, [ks[1], ks[0]], kind).unwrap();
            prop_assert_close!(flipped, -value, 1e-14 * (1.0 + value.abs()));
            continue;
        }
        prop_assert!((0.0..=1.0).contains(&value), "{kind:?} = {value}");
        let tolerance = 1e-12 * gradient.norm() * t.norm();
        prop_assert_close!(gradient.dotc(&t), Complex::default(), tolerance);
        for g in gk {
            prop_assert!(g == 0.0, "{kind:?}: wavenumber gradient {g}");
        }
        let (other, _) = metric(&(&t * scale), &swapped, ks, kind).unwrap();
        prop_assert_close!(other, value, 1e-12, "{:?}", kind);
    }
    Ok(())
}

/// A sphere matched to its embedding does not scatter, and splitting a homogeneous
/// sphere into two identical shells changes nothing.
fn check_sphere_contrast_and_split(
    size: f64,
    material: Material,
    embedding: Material,
    fraction: f64,
    l: u32,
) -> Result<(), TestCaseError> {
    let matched = mie(l, &[size], &[material, material]).unwrap();
    prop_assert_close!(*matched.value(), Matrix2::zeros(), 1e-12);
    let one = mie(l, &[size], &[material, embedding]).unwrap();
    let materials = [material, material, embedding];
    let two = mie(l, &[fraction * size, size], &materials).unwrap();
    prop_assert_close!(*one.value(), *two.value(), 1e-10);
    Ok(())
}

/// The complete T-matrix of a lossless sphere conserves energy,
/// `T^H W + W T + 2 T^H W T = 0` with `W = diag(k_pol^-2)` of the embedding, and is
/// invariant under rotations `D T D^H = T`. For the cotangent `G = W (I + 2T) H` with
/// any Hermitian `H`, the pullback of every lossless direction that keeps `W` up to a
/// factor vanishes: the radii, the vacuum wavenumber and the real parts of the
/// interior materials, and those of an achiral embedding's epsilon and mu.
fn check_sphere_matrix(
    lmax: u32,
    k0: f64,
    radii: &[f64],
    interior: &[Material],
    embedding: Material,
    seed: f64,
    angles: [f64; 3],
) -> Result<(), TestCaseError> {
    let materials: Vec<_> = interior.iter().copied().chain([embedding]).collect();
    let (t, sphere) = tmatrix::sphere(lmax, k0, radii, &materials).unwrap();
    let n = t.nrows();
    let basis = spherical_basis(lmax, [0.0; 3]);
    let ks = helicity_ks(k0, &embedding);
    let w = DMatrix::from_fn(n, n, |i, j| {
        if i == j {
            ks[usize::from(basis.modes[i].1.pol)].powi(-2)
        } else {
            Complex::default()
        }
    });
    let scale = w.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let two = Complex::new(2.0, 0.0);
    let balance = t.adjoint() * &w + &w * &t + t.adjoint() * &w * &t * two;
    prop_assert_close!(&balance, &DMatrix::zeros(n, n), 1e-12 * scale);
    let d = rotation::sw_rotation(&basis, &basis, angles)
        .unwrap()
        .value()
        .clone();
    prop_assert_close!(&(&d * &t * d.adjoint()), &t, 1e-13);
    let g = &w * (DMatrix::identity(n, n) + &t * two) * hermitian(n, seed);
    let tolerance = 1e-11 * g.norm();
    let gradient = sphere.pullback(&g).unwrap();
    let layers = &gradient.layers;
    prop_assert_close!(&layers.radii, &vec![0.0; radii.len()], tolerance);
    prop_assert_close!(gradient.k0, 0.0, tolerance);
    let achiral = embedding.kappa == Complex::default();
    for (name, values) in [
        ("epsilon", &layers.epsilon),
        ("mu", &layers.mu),
        ("kappa", &layers.kappa),
    ] {
        let lossless = if achiral && name != "kappa" {
            values.len()
        } else {
            radii.len()
        };
        for (i, value) in values.iter().take(lossless).enumerate() {
            prop_assert_close!(value.re, 0.0, tolerance, "{} of medium {}", name, i);
        }
    }
    Ok(())
}

/// `X = (I - T C)⁻¹ T` solves its residual equation; its pullback equals the
/// closed forms `T̄ = Y (I + C X)ᴴ` and `C̄ = Tᴴ Y Xᴴ` with `Y = (I - T C)⁻ᴴ G` from
/// an independent nalgebra LU, and both match central differences.
fn check_interaction_adjoint(
    local: &DMatrix<Complex>,
    coupling: &DMatrix<Complex>,
    direction: &DMatrix<Complex>,
    g: &DMatrix<Complex>,
) -> Result<(), TestCaseError> {
    let forward = interaction(local.clone(), coupling.clone()).unwrap();
    let system = DMatrix::identity(3, 3) - local * coupling;
    prop_assert_close!(&system * forward.value(), local.clone(), 1e-12);
    let y = system.adjoint().lu().solve(g).unwrap();
    let response = DMatrix::identity(3, 3) + coupling * forward.value();
    let expected_gc = local.adjoint() * &y * forward.value().adjoint();
    let cluster::InteractionGradient {
        local: gt,
        coupling: gc,
    } = forward.pullback(g).unwrap();
    prop_assert_close!(&gt, &(y * response.adjoint()), 1e-14);
    prop_assert_close!(&gc, &expected_gc, 1e-14);
    let local_numeric = central(1e-5, |step| {
        let value = interaction(local + direction * Complex::from(step), coupling.clone());
        re_dot(g, value.unwrap().value())
    });
    prop_assert_close!(re_dot(&gt, direction), local_numeric, 1e-8);
    let coupling_numeric = central(1e-5, |step| {
        let value = interaction(local.clone(), coupling + direction * Complex::from(step));
        re_dot(g, value.unwrap().value())
    });
    prop_assert_close!(re_dot(&gc, direction), coupling_numeric, 1e-8);
    Ok(())
}

/// A cylinder is symmetric under the half turn about a transverse axis, which maps
/// `(kz, m)` to `(-kz, -m)`: both give the same coefficients, and the same cotangent
/// pulls back to the negated kz gradient and otherwise equal gradients. A cylinder
/// matched to its embedding does not scatter, and splitting its outer layer into two
/// identical shells changes nothing.
fn check_cylinder_symmetries(
    kz: f64,
    order: i32,
    radii: &[f64],
    interior: &[Material],
    embedding: Material,
    g: &Matrix2,
) -> Result<(), TestCaseError> {
    let k0 = 1.3;
    let materials: Vec<_> = interior.iter().copied().chain([embedding]).collect();
    let forward = coeffs::mie_cyl(kz, order, k0, radii, &materials).unwrap();
    let mirror = coeffs::mie_cyl(-kz, -order, k0, radii, &materials).unwrap();
    let value = *forward.value();
    let scale = value.norm();
    prop_assert_close!(*mirror.value(), value, 1e-15 * scale);
    let gradient = forward.pullback(g).unwrap();
    let image = mirror.pullback(g).unwrap();
    let tolerance = 1e-14 * (gradient.kz.abs() + gradient.k0.abs() + 1.0);
    prop_assert_close!(image.kz, -gradient.kz, tolerance);
    prop_assert_close!(image.k0, gradient.k0, tolerance);
    let (a, b) = (&gradient.layers, &image.layers);
    let scale = a.radii.iter().map(|x| x.abs()).sum::<f64>();
    prop_assert_close!(&b.radii, &a.radii, 1e-14 * (1.0 + scale));
    for (x, y) in [
        (&a.epsilon, &b.epsilon),
        (&a.mu, &b.mu),
        (&a.kappa, &b.kappa),
    ] {
        let scale = x.iter().map(|z| z.norm()).sum::<f64>();
        prop_assert_close!(y, x, 1e-14 * (1.0 + scale));
    }
    let matched = vec![embedding; materials.len()];
    let zero = coeffs::mie_cyl(kz, order, k0, radii, &matched).unwrap();
    prop_assert_close!(*zero.value(), Matrix2::zeros(), 1e-12);
    let outer = radii.len() - 1;
    let inner = if outer == 0 { 0.0 } else { radii[outer - 1] };
    let mut split_radii = radii.to_vec();
    split_radii.insert(outer, inner.midpoint(radii[outer]));
    let mut split = materials.clone();
    split.insert(outer, materials[outer]);
    let other = coeffs::mie_cyl(kz, order, k0, &split_radii, &split).unwrap();
    prop_assert_close!(*other.value(), value, 1e-10);
    Ok(())
}

/// The T-matrix of a lossless cylinder at two propagating axial wavenumbers conserves
/// energy, `T^H W + W T + 2 T^H W T = 0` with `W = diag(k_pol^-1)` of the embedding.
/// For the cotangent `G = W (I + 2T) H` with any Hermitian `H`, the pullback of every
/// lossless direction that keeps `W` up to a factor vanishes: the axial and vacuum
/// wavenumbers, the radii and the real parts of the interior materials, and those of
/// an achiral embedding's epsilon and mu. With `H = I` an achiral embedding's
/// chirality drops out too, as it only shifts power between the helicities.
fn check_cylinder_energy_balance(
    kzs: [f64; 2],
    mmax: u32,
    radii: &[f64],
    interior: &[Material],
    embedding: Material,
    seed: f64,
) -> Result<(), TestCaseError> {
    let k0 = 1.3;
    let materials: Vec<_> = interior.iter().copied().chain([embedding]).collect();
    let solve = || tmatrix::cylinder(&kzs, mmax, k0, radii, &materials).unwrap();
    let (t, _) = solve();
    let n = t.nrows();
    let ks = helicity_ks(k0, &embedding);
    // Polarization 1 precedes polarization 0 in every (kz, m) block.
    let w = DMatrix::from_fn(n, n, |i, j| {
        if i == j {
            ks[1 - i % 2].inv()
        } else {
            Complex::default()
        }
    });
    let two = Complex::new(2.0, 0.0);
    let balance = t.adjoint() * &w + &w * &t + t.adjoint() * &w * &t * two;
    prop_assert_close!(&balance, &DMatrix::zeros(n, n), 1e-12);
    let achiral = embedding.kappa == Complex::default();
    for h in [hermitian(n, seed), DMatrix::identity(n, n)] {
        let identity = h == DMatrix::identity(n, n);
        let g = &w * (DMatrix::identity(n, n) + &t * two) * h;
        let tolerance = 1e-10 * g.norm();
        let gradient = solve().1.pullback(&g).unwrap();
        prop_assert_close!(&gradient.kzs, &vec![0.0; 2], tolerance);
        prop_assert_close!(gradient.k0, 0.0, tolerance);
        let layers = &gradient.layers;
        prop_assert_close!(&layers.radii, &vec![0.0; radii.len()], tolerance);
        for (name, values) in [
            ("epsilon", &layers.epsilon),
            ("mu", &layers.mu),
            ("kappa", &layers.kappa),
        ] {
            let lossless = if achiral && (name != "kappa" || identity) {
                values.len()
            } else {
                radii.len()
            };
            for (i, value) in values.iter().take(lossless).enumerate() {
                prop_assert_close!(value.re, 0.0, tolerance, "{} of medium {}", name, i);
            }
        }
    }
    Ok(())
}

/// Cylinder T-matrices at the axial wavenumbers `kz` and `-kz` (or only `kz = 0`) obey
/// Lorentz reciprocity `(T K)_ab = (-1)^(m_a + m_b) (T K)_{P b, P a}` with
/// `P: (kz, m) -> (-kz, -m)` and `K = diag(k_pol)`, for lossy chiral cylinders in a
/// lossy chiral embedding, alone and coupled to a copy at a transverse offset. The
/// coupled pair is invariant under a common translation, so its position gradients
/// sum to zero.
fn check_cylinder_reciprocity(
    kz: f64,
    mmax: u32,
    radii: &[f64],
    interior: &[Material],
    embedding: Material,
    (angle, z): (f64, f64),
) -> Result<(), TestCaseError> {
    let k0 = 1.3;
    let kzs = if kz == 0.0 { vec![0.0] } else { vec![kz, -kz] };
    let materials: Vec<_> = interior.iter().copied().chain([embedding]).collect();
    let (local, _) = tmatrix::cylinder(&kzs, mmax, k0, radii, &materials).unwrap();
    let bound = i32::try_from(mmax).unwrap();
    let modes: Vec<_> = kzs
        .iter()
        .flat_map(|&kz| {
            (-bound..=bound).flat_map(move |m| [1, 0].map(|pol| cw::Mode { kz, m, pol }))
        })
        .collect();
    let ks = helicity_ks(k0, &embedding);
    let single = cw::Basis {
        modes: modes.iter().map(|&mode| (0, mode)).collect(),
        positions: vec![[0.0; 3]],
    };
    let weights: Vec<_> = modes.iter().map(|m| ks[usize::from(m.pol)]).collect();
    let defect = reciprocity_defect(&local, &cylindrical_partners(&single), &weights);
    prop_assert_close!(defect, 0.0, 1e-13, "single cylinder");
    let distance = 2.0 * radii[radii.len() - 1] + 0.3;
    let basis = cw::Basis {
        modes: (0..2)
            .flat_map(|p| modes.iter().map(move |&mode| (p, mode)))
            .collect(),
        positions: vec![
            [0.0; 3],
            [distance * angle.cos(), distance * angle.sin(), z],
        ],
    };
    let partners = cylindrical_partners(&basis);
    let solve = |positions| {
        let basis = cw::Basis {
            positions,
            ..basis.clone()
        };
        let local = vec![local.clone(), local.clone()];
        cluster::cylindrical_particle_cluster(local, basis, ks, true).unwrap()
    };
    let cluster = solve(basis.positions.clone());
    let weights: Vec<_> = weights.iter().chain(&weights).copied().collect();
    let defect = reciprocity_defect(cluster.value(), &partners, &weights);
    prop_assert_close!(defect, 0.0, 1e-12, "cylinder pair");
    let shift = |p: &[f64; 3]| [p[0] + 0.3, p[1] - 0.4, p[2] + 0.8];
    let shifted = solve(basis.positions.iter().map(shift).collect());
    let scale = cluster.value().norm();
    prop_assert_close!(shifted.value(), cluster.value(), 1e-14 * scale);
    let n = cluster.value().nrows();
    let gradient = cluster.pullback(&patterned(n, n, angle)).unwrap();
    let magnitude: f64 = gradient.positions.iter().flatten().map(|g| g.abs()).sum();
    for axis in 0..3 {
        let total: f64 = gradient.positions.iter().map(|p| p[axis]).sum();
        prop_assert_close!(total, 0.0, 1e-14 * magnitude, "axis {}", axis);
    }
    Ok(())
}

/// A cylinder T-matrix and its pullback are the block sums of the boundary solves of
/// its `(kz, m)` blocks, in kz-major, ascending-m order, each solved on its own: a
/// block that shares its mirror partner's solve still has its own coefficients and
/// pulls its own cotangent back to its own axial wavenumber, with the sign of that
/// wavenumber. The axial wavenumber gradients match central differences of
/// `Re <G, T>`, which move a mirror pair's wavenumbers apart.
fn check_cylinder_matrix_pullback(
    kzs: &[f64],
    mmax: u32,
    radii: &[f64],
    interior: &[Material],
    embedding: Material,
    seed: f64,
) -> Result<(), TestCaseError> {
    let k0 = 1.3;
    let materials: Vec<_> = interior.iter().copied().chain([embedding]).collect();
    let solve = |kzs: &[f64]| tmatrix::cylinder(kzs, mmax, k0, radii, &materials).unwrap();
    let (t, matrix) = solve(kzs);
    let n = t.nrows();
    let g = patterned(n, n, seed);
    let gradient = matrix.pullback(&g).unwrap();
    let bound = i32::try_from(mmax).unwrap();
    let mut expected = DMatrix::zeros(n, n);
    let mut kz_gradients = vec![0.0; kzs.len()];
    let mut k0_gradient = 0.0;
    let mut layers = LayerGradient::zeros(radii.len());
    // The sum of the absolute block gradients bounds the rounding of their sums.
    let mut scale = 0.0;
    let blocks = kzs
        .iter()
        .enumerate()
        .flat_map(|(i, &kz)| (-bound..=bound).map(move |m| (i, kz, m)));
    for (index, (i, kz, m)) in blocks.enumerate() {
        let block = coeffs::mie_cyl(kz, m, k0, radii, &materials).unwrap();
        let at = 2 * index;
        expected
            .fixed_view_mut::<2, 2>(at, at)
            .copy_from(&to_mode_order(block.value()));
        let cotangent = g.fixed_view::<2, 2>(at, at).into_owned();
        let part = block.pullback(&to_mode_order(&cotangent)).unwrap();
        kz_gradients[i] += part.kz;
        k0_gradient += part.k0;
        layers.accumulate(&part.layers);
        let media = [&part.layers.epsilon, &part.layers.mu, &part.layers.kappa];
        scale += part.kz.abs()
            + part.k0.abs()
            + part.layers.radii.iter().map(|x| x.abs()).sum::<f64>()
            + media
                .iter()
                .flat_map(|x| x.iter())
                .map(|z| z.norm())
                .sum::<f64>();
    }
    let size = expected.iter().map(|z| z.norm()).fold(0.0, f64::max);
    prop_assert_close!(&t, &expected, 1e-14 * size + f64::MIN_POSITIVE);
    let tolerance = 1e-13 * scale + f64::MIN_POSITIVE;
    prop_assert_close!(&gradient.kzs, &kz_gradients, tolerance);
    prop_assert_close!(gradient.k0, k0_gradient, tolerance);
    let (actual, expected) = (&gradient.layers, &layers);
    prop_assert_close!(&actual.radii, &expected.radii, tolerance);
    prop_assert_close!(&actual.epsilon, &expected.epsilon, tolerance);
    prop_assert_close!(&actual.mu, &expected.mu, tolerance);
    prop_assert_close!(&actual.kappa, &expected.kappa, tolerance);
    for (i, &analytic) in gradient.kzs.iter().enumerate() {
        let numeric = central(1e-5, |step| {
            let mut shifted = kzs.to_vec();
            shifted[i] += step;
            re_dot(&g, &solve(&shifted).0)
        });
        prop_assert_close!(analytic, numeric, 1e-9 * (1.0 + scale), "kz {}", i);
    }
    Ok(())
}

/// The EBCM surface integral is homogeneous of degree two under the joint scaling of
/// radii, slopes and inverse wavenumbers, and of degree one in the impedances.
fn check_ebcm_surface_scaling(deformation: f64) -> Result<(), TestCaseError> {
    let modes = vec![
        Mode { l: 1, m: 0, pol: 0 },
        Mode { l: 1, m: 0, pol: 1 },
        Mode { l: 2, m: 1, pol: 0 },
    ];
    let surface = deformed(deformation, 12);
    let (radii, slopes) = (surface.radii.clone(), surface.slopes.clone());
    let ks = [
        [Complex::new(2.1, 0.1), Complex::new(2.2, 0.1)],
        [Complex::new(1.3, 0.0); 2],
    ];
    let zs = [Complex::new(0.7, 0.01), Complex::new(1.0, 0.0)];
    let (value, residual) = qmat(
        modes.clone(),
        modes,
        surface,
        ks,
        zs,
        Radial::Singular,
        true,
    )
    .unwrap();
    let g = DMatrix::from_element(3, 3, Complex::new(0.3, 0.1));
    let gradient = residual.pullback(&g).unwrap();
    let geometry = dot(&gradient.radii, &radii) + dot(&gradient.slopes, &slopes)
        - re_dot(gradient.ks.iter().flatten(), ks.iter().flatten());
    prop_assert_close!(geometry, 2.0 * re_dot(&g, &value), 1e-11);
    prop_assert_close!(re_dot(gradient.zs, zs), re_dot(&g, &value), 1e-11);
    Ok(())
}

/// The surface `r = 0.3 (1 + deformation cos^2 theta)` at `nodes` midpoint nodes.
fn deformed(deformation: f64, nodes: u32) -> Surface {
    let step = std::f64::consts::PI / f64::from(nodes);
    let theta: Vec<_> = (0..nodes).map(|i| (f64::from(i) + 0.5) * step).collect();
    Surface {
        radii: theta
            .iter()
            .map(|t| 0.3 * (1.0 + deformation * t.cos().powi(2)))
            .collect(),
        slopes: theta
            .iter()
            .map(|t| -0.6 * deformation * t.cos() * t.sin())
            .collect(),
        weights: vec![step; theta.len()],
        theta,
    }
}

/// Without contrast the regular Q integral vanishes on every closed surface: the
/// reciprocity current of two regular waves of one medium is divergence free. With
/// Gauss-Legendre nodes it vanishes on a deformed surface, and so does the pullback
/// of any cotangent paired with any smooth deformation `dr = radius (a + b cos + c
/// cos^2)` of the radii and its slopes.
fn check_ebcm_zero_contrast(
    radius: f64,
    deformation: f64,
    [a, b, c]: [f64; 3],
    medium: Material,
    seed: f64,
) -> Result<(), TestCaseError> {
    let modes = sw::modes(2).unwrap();
    let ks = helicity_ks(1.3, &medium);
    let z = medium.impedance();
    let surface = gauss_legendre_surface(24, |t| {
        let (sine, cosine) = t.sin_cos();
        (
            radius * (1.0 + deformation * cosine * cosine),
            -2.0 * radius * deformation * cosine * sine,
        )
    });
    let theta = surface.theta.clone();
    let (value, residual) = qmat(
        modes.clone(),
        modes,
        surface,
        [ks; 2],
        [z; 2],
        Radial::Regular,
        true,
    )
    .unwrap();
    let n = value.nrows();
    let scale = ks[0].norm().max(ks[1].norm()).powi(2) * z.norm();
    prop_assert_close!(&value, &DMatrix::zeros(n, n), 1e-14 * scale);
    let g = patterned(n, n, seed);
    let gradient = residual.pullback(&g).unwrap();
    let radii = theta
        .iter()
        .map(|t| radius * (a + b * t.cos() + c * t.cos().powi(2)));
    let slopes = theta
        .iter()
        .map(|t| -radius * t.sin() * (b + 2.0 * c * t.cos()));
    let derivative = dot(&gradient.radii, radii) + dot(&gradient.slopes, slopes);
    prop_assert_close!(derivative, 0.0, 1e-13 * scale * g.norm());
    Ok(())
}

/// A surface `(radius, slope)(theta)` at `n` Gauss-Legendre nodes in `theta`.
fn gauss_legendre_surface(n: u32, shape: impl Fn(f64) -> (f64, f64)) -> Surface {
    let half = std::f64::consts::FRAC_PI_2;
    let (theta, weights): (Vec<_>, Vec<_>) = gauss_legendre(n)
        .into_iter()
        .map(|(x, w)| ((x + 1.0) * half, w * half))
        .unzip();
    let (radii, slopes) = theta.iter().map(|&t| shape(t)).unzip();
    Surface {
        theta,
        weights,
        radii,
        slopes,
    }
}

/// The EBCM T-matrix `-Q⁻¹ Q_reg` of a lossy chiral sphere in a chiral embedding
/// equals its Mie T-matrix: 48 Gauss-Legendre nodes integrate the smooth integrand
/// to rounding.
fn check_ebcm_sphere(
    lmax: u32,
    radius: f64,
    interior: Material,
    embedding: Material,
) -> Result<(), TestCaseError> {
    let k0 = 1.3;
    let modes = sw::modes(lmax).unwrap();
    let surface = || gauss_legendre_surface(48, |_| (radius, 0.0));
    let ks = [helicity_ks(k0, &interior), helicity_ks(k0, &embedding)];
    let zs = [interior.impedance(), embedding.impedance()];
    let q = |radial| {
        qmat(
            modes.clone(),
            modes.clone(),
            surface(),
            ks,
            zs,
            radial,
            true,
        )
        .unwrap()
        .0
    };
    let t = -q(Radial::Singular).lu().solve(&q(Radial::Regular)).unwrap();
    let (expected, _) = tmatrix::sphere(lmax, k0, &[radius], &[interior, embedding]).unwrap();
    prop_assert_close!(&t, &expected, 1e-12);
    Ok(())
}

/// The cluster pullback matches a central difference along a direction that moves
/// the wavenumber, radii, permittivities and positions at once; position gradients
/// sum to zero by translation invariance.
fn check_cluster_directional_derivative(
    lmax: u32,
    radius: f64,
    eps: f64,
    k: f64,
    dx: f64,
) -> Result<(), TestCaseError> {
    let solve = |step: f64| {
        cluster::sphere_cluster(
            lmax,
            k + step * 0.2,
            &[radius + step * 0.3, 0.25 - step * 0.1],
            &[
                Complex::new(eps + step * 0.4, 0.1 + step * 0.1),
                Complex::new(3.0 - step * 0.2, 0.2 - step * 0.3),
            ],
            &[
                [step * 0.1, 0.0, 0.0],
                [dx, 0.1 + step * 0.3, 1.5 - step * 0.2],
            ],
        )
        .unwrap()
    };
    let n = usize::try_from(4 * lmax * (lmax + 2)).unwrap();
    let g = patterned(n, n, 0.3 * f64::from(lmax));
    let gradients = solve(0.0).pullback(&g).unwrap();
    let numerical = central(1e-5, |step| re_dot(&g, solve(step).value()));
    let analytic = gradients.k0 * 0.2
        + dot(&gradients.radii, [0.3, -0.1])
        + re_dot(
            &gradients.epsilon,
            [Complex::new(0.4, 0.1), Complex::new(-0.2, -0.3)],
        )
        + dot(
            gradients.positions.iter().flatten(),
            [0.1, 0.0, 0.0, 0.0, 0.3, -0.2],
        );
    prop_assert_close!(numerical, analytic, 1e-8 * (1.0 + numerical.abs()));
    for axis in 0..3 {
        let total: f64 = gradients.positions.iter().map(|g| g[axis]).sum();
        prop_assert_close!(total, 0.0, 1e-12);
    }
    Ok(())
}

/// A vacuum sphere cluster is the particle cluster of its Mie T-matrices: equal
/// values and position gradients, and radius, permittivity and vacuum-wavenumber
/// gradients equal to the sphere pullbacks of the local gradients, the last plus both
/// embedding-wavenumber gradients.
fn check_cluster_matches_particle_cluster(
    spheres: &Spheres,
    seed: f64,
) -> Result<(), TestCaseError> {
    let vacuum = spheres.in_vacuum();
    let cluster = vacuum.sphere_cluster();
    let (local, particles) = vacuum.particle_cluster();
    let n = cluster.value().nrows();
    prop_assert_close!(
        cluster.value(),
        particles.value(),
        1e-14 * cluster.value().norm()
    );
    let g = patterned(n, n, seed);
    let gradient = cluster.pullback(&g).unwrap();
    let expected = particles.pullback(&g).unwrap();
    let tolerance = 1e-12 * g.norm();
    prop_assert_close!(&gradient.positions, &expected.positions, tolerance);
    let mut k0 = (expected.ks[0] + expected.ks[1]).re;
    for (i, ((_, sphere), g)) in local.into_iter().zip(&expected.local).enumerate() {
        let sphere = sphere.pullback(g).unwrap();
        prop_assert_close!(
            gradient.radii[i],
            sphere.layers.radii[0],
            tolerance,
            "sphere {}",
            i
        );
        let epsilon = sphere.layers.epsilon[0];
        prop_assert_close!(gradient.epsilon[i], epsilon, tolerance, "sphere {}", i);
        k0 += sphere.k0;
    }
    prop_assert_close!(gradient.k0, k0, tolerance);
    Ok(())
}

/// Lorentz reciprocity `(T K)_ab = (-1)^(m_a + m_b) (T K)_{P b, P a}` with
/// `P: m -> -m` and `K = diag(k_pol^2)` holds for regular and singular translations in
/// both bases (whose self blocks are the identity or zero), for the Mie T-matrix of a
/// lossy chiral sphere in a chiral embedding, for the interacting cluster of such
/// spheres, and for the vacuum cluster.
fn check_sphere_reciprocity(
    spheres: &Spheres,
    helicity: bool,
    radial: Radial,
) -> Result<(), TestCaseError> {
    let basis = spheres.basis();
    let partners = spherical_partners(&basis);
    let weights: Vec<_> = spheres.mode_ks().iter().map(|k| k * k).collect();
    // The parity basis needs an achiral medium.
    let (ks, translation_weights) = if helicity {
        (spheres.ks(), weights.clone())
    } else {
        (
            [spheres.ks()[0]; 2],
            vec![Complex::from(1.0); weights.len()],
        )
    };
    let (translation, _) = sw::expansion(basis.clone(), basis, ks, helicity, radial).unwrap();
    let defect = reciprocity_defect(&translation, &partners, &translation_weights);
    prop_assert_close!(defect, 0.0, 1e-14, "translation");
    let (local, cluster) = spheres.particle_cluster();
    let modes_per_particle = local[0].0.nrows();
    let single = &partners[..modes_per_particle];
    let defect = reciprocity_defect(&local[0].0, single, &weights[..modes_per_particle]);
    prop_assert_close!(defect, 0.0, 1e-13, "sphere");
    let defect = reciprocity_defect(cluster.value(), &partners, &weights);
    prop_assert_close!(defect, 0.0, 1e-13, "cluster");
    let vacuum = spheres.sphere_cluster();
    let defect = reciprocity_defect(
        vacuum.value(),
        &partners,
        &vec![Complex::from(1.0); weights.len()],
    );
    prop_assert_close!(defect, 0.0, 1e-13, "vacuum cluster");
    Ok(())
}

/// Lossless clusters conserve energy in their local bases:
/// `T^H W + W T + 2 T^H W R T = 0` with `W = diag(k_pol^-2)` and the regular
/// translations `R` between the particles, whose self blocks are the identity. Along
/// the lossless directions that keep `R` and `W`, every radius and the real part of
/// every interior material, the derivative of this identity is the pullback of
/// `G = W (I + 2 R T) H` for any Hermitian `H`, which therefore vanishes; this holds
/// for chiral spheres in a chiral embedding and for the vacuum cluster.
fn check_cluster_energy_balance(spheres: &Spheres, seed: f64) -> Result<(), TestCaseError> {
    let basis = spheres.basis();
    let n = basis.modes.len();
    let two = Complex::new(2.0, 0.0);
    let identity = DMatrix::<Complex>::identity(n, n);
    let h = hermitian(n, seed);
    let regular = |ks| {
        sw::expansion(basis.clone(), basis.clone(), ks, true, Radial::Regular)
            .unwrap()
            .0
    };
    let weights: Vec<_> = spheres.mode_ks().iter().map(|k| (k * k).inv()).collect();
    let w = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(weights));
    let scale = w.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let r = regular(spheres.ks());
    let (local, cluster) = spheres.particle_cluster();
    let t = cluster.value().clone();
    let balance = t.adjoint() * &w + &w * &t + t.adjoint() * &w * &r * &t * two;
    prop_assert_close!(&balance, &DMatrix::zeros(n, n), 1e-13 * scale);
    let g = &w * (&identity + &r * &t * two) * &h;
    let tolerance = 1e-13 * g.norm();
    let gradient = cluster.pullback(&g).unwrap();
    for (i, ((_, sphere), g)) in local.into_iter().zip(&gradient.local).enumerate() {
        let sphere = sphere.pullback(g).unwrap();
        let layers = &sphere.layers;
        prop_assert_close!(layers.radii[0], 0.0, tolerance, "sphere {}", i);
        for value in [layers.epsilon[0], layers.mu[0], layers.kappa[0]] {
            prop_assert_close!(value.re, 0.0, tolerance, "sphere {}", i);
        }
    }

    let k0 = Complex::from(spheres.k0);
    let r = regular([k0; 2]);
    let cluster = spheres.sphere_cluster();
    let t = cluster.value().clone();
    let balance = t.adjoint() + &t + t.adjoint() * &r * &t * two;
    prop_assert_close!(&balance, &DMatrix::zeros(n, n), 1e-13);
    let g = (&identity + &r * &t * two) * &h;
    let tolerance = 1e-13 * g.norm();
    let gradient = cluster.pullback(&g).unwrap();
    prop_assert_close!(&gradient.radii, &vec![0.0; spheres.radii.len()], tolerance);
    let epsilon: Vec<_> = gradient.epsilon.iter().map(|z| z.re).collect();
    prop_assert_close!(&epsilon, &vec![0.0; spheres.radii.len()], tolerance);
    Ok(())
}

/// Rotating every position by `R = Rz(phi) Ry(theta) Rz(psi)` rotates the interacting
/// T-matrix, `T(R r) = D T(r) D^H` with the block-diagonal rotation `D` of the local
/// bases: for the vacuum cluster, for chiral spheres, whose own T-matrices are
/// rotation invariant, and for arbitrary local T-matrices `L` rotated with the
/// positions, drawn with the degree decay of physical ones.
///
/// The strong dipoles of the arbitrary blocks can make `I - L C`, with the outgoing
/// translations `C` between the particles, ill conditioned (condition numbers `kappa`
/// up to 5e4 in 1000 cases), so their bound scales with `kappa`. Relative to the
/// Frobenius norm of `T`, the measured errors stay below `4e-16 kappa` for these
/// blocks and below `1.1e-15` for the physical clusters.
fn check_cluster_rotation(
    spheres: &Spheres,
    angles: [f64; 3],
    seed: f64,
) -> Result<(), TestCaseError> {
    let rotated = Spheres {
        positions: spheres
            .positions
            .iter()
            .map(|&p| rotate(angles, p))
            .collect(),
        ..spheres.clone()
    };
    let basis = spheres.basis();
    let d = rotation::sw_rotation(&basis, &basis, angles)
        .unwrap()
        .value()
        .clone();
    let covariant = |t: &DMatrix<Complex>, other: &DMatrix<Complex>, relative: f64| {
        let expected = &d * t * d.adjoint();
        prop_assert_close!(other, &expected, relative * expected.norm());
        Ok::<_, TestCaseError>(())
    };
    covariant(
        spheres.sphere_cluster().value(),
        rotated.sphere_cluster().value(),
        1e-14,
    )?;
    let (local, cluster) = spheres.particle_cluster();
    let modes_per_particle = local[0].0.nrows();
    let block = d
        .view((0, 0), (modes_per_particle, modes_per_particle))
        .into_owned();
    for (sphere, _) in &local {
        let rotated_sphere = &block * sphere * block.adjoint();
        prop_assert_close!(&rotated_sphere, sphere, 1e-14);
    }
    covariant(cluster.value(), rotated.particle_cluster().1.value(), 1e-14)?;
    let degrees: Vec<_> = basis.modes[..modes_per_particle]
        .iter()
        .map(|(_, m)| m.l)
        .collect();
    let local: Vec<_> = (0..spheres.radii.len())
        .map(|i| {
            let pattern = patterned(
                modes_per_particle,
                modes_per_particle,
                seed + f64::from(u32::try_from(i).unwrap()),
            );
            DMatrix::from_fn(modes_per_particle, modes_per_particle, |a, b| {
                pattern[(a, b)] * 0.05 * 0.3_f64.powi(degrees[a] + degrees[b] - 2)
            })
        })
        .collect();
    let turned: Vec<_> = local.iter().map(|t| &block * t * block.adjoint()).collect();
    let n = basis.modes.len();
    let outgoing = sw::expansion(
        basis.clone(),
        basis.clone(),
        spheres.ks(),
        true,
        Radial::Singular,
    )
    .unwrap()
    .0;
    let diagonal = DMatrix::from_fn(n, n, |a, b| {
        if a / modes_per_particle == b / modes_per_particle {
            local[a / modes_per_particle][(a % modes_per_particle, b % modes_per_particle)]
        } else {
            Complex::default()
        }
    });
    let singular = (DMatrix::identity(n, n) - diagonal * outgoing).singular_values();
    let condition = singular.max() / singular.min();
    let solve = |local, basis| {
        cluster::particle_cluster(local, basis, spheres.ks(), true)
            .unwrap()
            .value()
            .clone()
    };
    covariant(
        &solve(local, basis),
        &solve(turned, rotated.basis()),
        1e-14 * condition,
    )
}
