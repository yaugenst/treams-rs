//! Direct sums over cube shells, without Ewald acceleration.
//!
//! Upstream: `dsumsw*` and `dsumcw*` of `lattice/_dsum.pyx`, with the terms `_fsw*` and
//! `_fcw*`.

use super::{
    cell::BlochLattice,
    geometry::visit_cube,
    inputs::Inputs,
    wave::{Family, solid_jet},
};
use crate::{
    Complex, Error, Result,
    numerics::Jet,
    special::{self, Bessel, Radial, harmonic_normalization},
};

/// One image of a direct sum: the outgoing wave `h_l^(1)(k |r|) Y_lm(r / |r|)` or
/// `H_m^(1)(k |r|) e^(i m phi)` at the displacement `r = -shift - R` (upstream `_fsw*`
/// and `_fcw*`). Values take `special::bessel`; jets take the radial jets, whose first
/// derivative the chain rule carries. The spherical harmonic is
/// `harmonic_normalization(l, m)` times the solid harmonic of the unit vector.
fn direct_term<const N: usize>(wave: Family, k: Jet<N>, r: [Jet<N>; 3]) -> Result<Jet<N>> {
    let radius = r.into_iter().map(|v| v * v).sum::<Jet<N>>().sqrt();
    let argument = k * radius;
    let (degree, spherical) = match wave {
        Family::Spherical { l, .. } => (l, true),
        Family::Cylindrical { m } => (m, false),
    };
    let radial = if N == 0 {
        Jet::constant(special::bessel(
            f64::from(degree),
            argument.value,
            Bessel::H1,
            spherical,
            0,
        )?)
    } else {
        let radial = if spherical {
            special::spherical_radial(
                u32::try_from(degree).map_err(|_| Error::InvalidInput("invalid degree".into()))?,
                argument.value,
                Radial::Singular,
            )?
        } else {
            special::cylindrical_radial(degree, argument.value, Radial::Singular)?
        };
        argument.chain(radial.value, radial.first)
    };
    let angular = match wave {
        Family::Spherical { l, m } => {
            harmonic_normalization(l, m) * solid_jet(l, m, r.map(|v| v / radius))
        }
        Family::Cylindrical { m } => ((r[0] + Complex::i() * r[1]) / radius).powi(m),
    };
    let result = radial * angular;
    if !result.finite() {
        return Err(Error::NonFinite("non-finite direct lattice term".into()));
    }
    Ok(result)
}

/// One unaccelerated cube shell of images, in the given lattice basis (upstream
/// `dsum*(..., i)` with `i = shell`).
///
/// A 1D sum whose shift lies on the axis at half a period pairs the images
/// `direction * shell` and `-direction * (shell + 1)`, which lie at equal distances, so
/// each shell adds a symmetric pair. That grouping jumps when the shift moves off the
/// half cell, so jets fail there.
pub(super) fn direct_shell<const N: usize>(
    wave: Family,
    lattice: &BlochLattice,
    inputs: &Inputs<N>,
    shell: i64,
) -> Result<Jet<N>> {
    let (dim, axis, r) = (inputs.dim, inputs.axes[0], &inputs.r);
    let half_cell = dim == 1
        && r.iter()
            .enumerate()
            .all(|(j, v)| j == axis || v.value == Complex::default())
        && (r[axis].value.re / lattice.direct[0][0]).abs() == 0.5;
    if N > 0 && half_cell {
        return Err(Error::InvalidInput("direct half-cell shell grouping changes discontinuously; use the Ewald sum for adjoints".into()));
    }
    let mut result = Jet::default();
    let mut term = |n| {
        let point = inputs.point(&inputs.direct, n);
        let shift = inputs.image(&point);
        if shift.iter().all(|v| v.value == Complex::default()) {
            return Ok(());
        }
        result += (Complex::i() * inputs.phase(&point)).exp() * direct_term(wave, inputs.k, shift)?;
        Ok(())
    };
    if half_cell {
        let direction = if r[axis].value.re / lattice.direct[0][0] > 0.0 {
            1
        } else {
            -1
        };
        term([direction * shell, 0, 0])?;
        term([-direction * (shell + 1), 0, 0])?;
    } else {
        visit_cube(dim, shell, true, term)?;
    }
    Ok(result)
}
