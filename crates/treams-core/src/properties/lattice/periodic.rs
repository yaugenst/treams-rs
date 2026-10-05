//! Checks of the periodic expansions built on the lattice sums: spherical and
//! cylindrical lattice expansions and the periodic spherical-to-cylindrical conversion.

use nalgebra::DMatrix;
use proptest::test_runner::TestCaseError;

use super::ewald::c;
use crate::{
    Complex, basis, cw,
    lattice::BlochLattice,
    sw::{self, Mode, lattice_expansion_from_table},
    test_support::{cylindrical_basis, dot, prop_assert_close, re_dot, spherical_basis},
};

/// Periodic expansions are linear in the lattice-sum table: the table pullback is
/// its conjugate transpose, `Re <g, A x> = Re <A^H g, x>` for random tables `x` and
/// cotangents `g`, which fails for any misplaced harmonic, channel or origin pair.
pub(super) fn check_lattice_table_adjoint(
    modes: &[Mode],
    origins: [[f64; 3]; 2],
    helicity: bool,
    channels: usize,
    table: &[Complex],
    g: &DMatrix<Complex>,
) -> Result<(), TestCaseError> {
    let destination = basis::Basis {
        modes: vec![(0, modes[0]), (1, modes[1])],
        positions: vec![[0.0; 3], origins[0]],
    };
    let source = basis::Basis {
        modes: vec![(0, modes[2]), (0, modes[3])],
        positions: vec![origins[1]],
    };
    let order = modes[0].l.max(modes[1].l) + modes[2].l.max(modes[3].l);
    let harmonics = usize::try_from((order + 1).pow(2)).unwrap();
    let input = &table[..2 * channels * harmonics];
    let (value, residual) =
        lattice_expansion_from_table(&destination, &source, helicity, channels, input).unwrap();
    let pushed = residual.pushforward(input).unwrap();
    prop_assert_close!(&pushed, &value, 1e-14 * (1.0 + value.norm()));
    let loss = re_dot(g, &value);
    let gradient = residual.pullback(g).unwrap();
    let tolerance = 1e-12 * (1.0 + g.norm() * value.norm());
    prop_assert_close!(re_dot(&gradient, input), loss, tolerance);
    Ok(())
}

/// Euler identity of joint period, wavenumber, Bloch and axial scaling of a
/// cylindrical lattice expansion.
pub(super) fn check_cylindrical_periodic_axial_scale(kz: f64, a: f64) -> Result<(), TestCaseError> {
    let basis = cw::Basis {
        modes: vec![
            (0, cw::Mode { kz, m: -1, pol: 0 }),
            (0, cw::Mode { kz, m: 1, pol: 1 }),
        ],
        positions: vec![[0.0; 3]],
    };
    let ks = [c(1.3, 0.05), c(1.5, 0.07)];
    let kpar = 0.1;
    let lattice = BlochLattice::new(&[vec![a]], &[kpar]).unwrap();
    let (_, residual) =
        cw::lattice_expansion(basis.clone(), basis, ks, lattice, c(0.9, 0.0)).unwrap();
    let scale_direction = residual
        .pushforward_axial(
            &[[0.0; 3]],
            &[[0.0; 3]],
            ks.map(|k| -k),
            &[-kpar],
            &DMatrix::from_element(1, 1, a),
            &[-kz],
        )
        .unwrap();
    prop_assert_close!(scale_direction.norm(), 0.0, 1e-11);
    let g = DMatrix::from_element(2, 2, c(0.3, 0.2));
    let (g, gkz) = residual.pullback_axial(&g).unwrap();
    let length = g.vectors.iter().sum::<f64>() * a;
    let spectral = re_dot(g.expansion.ks, ks)
        + g.kpar.iter().sum::<f64>() * kpar
        + gkz.iter().sum::<f64>() * kz;
    prop_assert_close!(length, spectral, 1e-11);
    Ok(())
}

/// Periodic spherical-to-cylindrical conversion obeys the Euler identity of joint
/// position, period, wavenumber and axial scaling, and is invariant under it.
pub(super) fn check_periodic_conversion_scale(
    k: f64,
    kz: f64,
    period: f64,
    x: f64,
    helicity: bool,
) -> Result<(), TestCaseError> {
    let source = spherical_basis(3, [x, 0.1, -0.2]);
    let destination = cylindrical_basis(3, kz, [0.2, -0.1, 0.1]);
    let ks = [c(k, 0.1); 2];
    let (value, residual) =
        sw::periodic_to_cw_matrix(destination.clone(), source.clone(), ks, period, helicity)
            .unwrap();
    let g = DMatrix::from_element(value.nrows(), value.ncols(), c(0.2, 0.1));
    let gradient = residual.pullback(&g).unwrap();
    let expansion = &gradient.expansion;
    let gradients = expansion
        .destination
        .iter()
        .chain(&expansion.source)
        .flatten();
    let positions = destination
        .positions
        .iter()
        .chain(&source.positions)
        .flatten();
    let spatial = dot(gradients, positions);
    let spectral = re_dot(expansion.ks, ks);
    let axial: f64 = gradient.kz.iter().map(|g| g * kz).sum();
    prop_assert_close!(spatial + period * gradient.period, spectral + axial, 1e-10);
    let scale = |positions: &[[f64; 3]]| positions.iter().map(|p| p.map(|x| x * 1.7)).collect();
    let scaled_source = basis::Basis {
        positions: scale(&source.positions),
        ..source
    };
    let scaled_destination = cw::Basis {
        positions: scale(&destination.positions),
        modes: destination
            .modes
            .iter()
            .map(|&(p, m)| {
                (
                    p,
                    cw::Mode {
                        kz: m.kz / 1.7,
                        ..m
                    },
                )
            })
            .collect(),
    };
    let (scaled, _) = sw::periodic_to_cw_matrix(
        scaled_destination,
        scaled_source,
        ks.map(|k| k / 1.7),
        period * 1.7,
        helicity,
    )
    .unwrap();
    prop_assert_close!(value, scaled, 1e-10);
    Ok(())
}
