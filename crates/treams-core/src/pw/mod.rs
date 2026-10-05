//! Plane waves: polarization vectors, fields, translation phases, cyclic permutations of
//! the Cartesian axes, and the expansion of plane waves in regular spherical and
//! cylindrical waves.
//!
//! Upstream: `treams.pw`, `treams.misc.wave_vec_z`, and `treams.special.vpw_M`, `vpw_N`
//! and `vpw_A`.
//!
//! | File | Contents | Upstream |
//! |---|---|---|
//! | `polarization.rs` | [`polarization()`], [`field_value`] and [`wave_vector_z`]; the private `Direction` jets of a wavevector | `special.vpw_*`, `misc.wave_vec_z` |
//! | `field.rs` | [`translate`], [`phases`] and [`field()`] | `pw.translate`, `special.vpw_*` |
//! | `permute.rs` | [`permute_xyz`] and [`permutation`] | `pw.permute_xyz` |
//! | `expand.rs` | [`to_sw`], [`to_cw`] and [`expansion`] | `pw.to_sw`, `pw.to_cw`, the `Expand` operator from a plane-wave basis |
//! | `saved.rs` | Fixed numerical layouts for derivative state | treams-rs extension |
//!
//! Conventions:
//!
//! - A plane wave is `p exp(i k·r)` with a complex wavevector `k = (kx, ky, kz)`, which
//!   may be evanescent. Its wavenumber is the algebraic norm `sqrt(kx^2 + ky^2 + kz^2)`,
//!   not the Hermitian length.
//! - `pol` is the polarization index, 0 or 1: with `helicity` true, 1 is positive and 0
//!   negative helicity; with `helicity` false, 1 is the N and 0 the M polarization, as
//!   for [`crate::sw`].
//! - Gradients with respect to a complex wavevector use the pairing
//!   `dL = Re Σ conj(g)·dk`, so a real input takes the real part of its gradient.
//!
//! The physical and adjoint identities of plane waves are tested in
//! `properties/plane.rs`, which compares the expansions with the single-wave references
//! of `expand.rs`.

mod expand;
mod field;
mod permute;
mod polarization;
mod saved;

pub(crate) use expand::cylindrical_mode_matches;
pub use expand::{ExpansionGradient, ExpansionResidual, expansion, to_cw, to_sw};
#[cfg(test)]
pub(crate) use expand::{reference_cylindrical_expansion, reference_spherical_expansion};
pub use field::{
    FieldGradient, FieldResidual, PhasesGradient, PhasesResidual, field, phases, translate,
};
pub use permute::{PermutationResidual, permutation, permute_xyz};
pub use polarization::{field_value, polarization, wave_vector_z};
pub(crate) use polarization::{polarization_from_inputs, polarization_jet};
