//! Multiple scattering in clusters of particles: solves of `(I - T C) X = T B`.
//!
//! Upstream: `treams.TMatrix.cluster` and `treams.TMatrix.interaction`.
//!
//! `T` is the block-diagonal matrix of the particles' own T-matrices, `C` couples
//! every pair of particles through the singular expansion between their centres,
//! and `B` holds incident fields, one column each. The scattered fields `X` follow
//! from one of four solvers:
//!
//! | Solver | Particles | Method | Use it for |
//! |---|---|---|---|
//! | [`particle_cluster`], [`cylindrical_particle_cluster`] | any T-matrices | dense LU | the full cluster T-matrix and gradients of the particle T-matrices and positions |
//! | [`sphere_cluster`] | homogeneous nonmagnetic spheres in vacuum | dense LU | the full cluster T-matrix and gradients of radii, permittivities and positions |
//! | [`InteractionFactor`] | any T-matrices, or the spheres of [`sphere_cluster_factor`] | dense LU, kept | many incident fields with one factorization |
//! | [`IterativeSphereCluster`] | homogeneous nonmagnetic spheres in vacuum | GMRES, no dense matrix | large clusters and few incident fields |
//!
//! The dense solvers store `C` and its LU factors, `n²` entries for `n` modes in
//! total. [`IterativeSphereCluster`] stores only the geometry and the Mie
//! coefficients and applies `C` one particle pair at a time.
//!
//! | treams-rs | treams |
//! |---|---|
//! | [`particle_cluster`] | `TMatrix.cluster(tmats, positions).interaction.solve()` |
//! | [`cylindrical_particle_cluster`] | `TMatrixC.cluster(tmats, positions).interaction.solve()` |
//! | [`sphere_cluster`] | the same with `TMatrix.sphere` T-matrices |
//! | [`interaction()`] | `TMatrix.interaction.solve()` |
//! | [`InteractionFactor`], [`IterativeSphereCluster`] | none |

mod interaction;
mod iterative;
mod particles;
mod spheres;

pub(crate) use interaction::interaction_blocks;
pub use interaction::{
    IlluminateGradient, IlluminateResidual, InteractionFactor, InteractionGradient,
    InteractionResidual, interaction,
};
pub use iterative::{
    IterativeGradient, IterativeResidual, IterativeSolution, IterativeSphereCluster,
};
pub use particles::{
    ParticleClusterGradient, ParticleClusterResidual, cylindrical_particle_cluster,
    particle_cluster,
};
pub use spheres::{
    SphereClusterGradient, SphereClusterResidual, sphere_cluster, sphere_cluster_factor,
};
pub(crate) use spheres::{vacuum_sphere, validate_spheres};
