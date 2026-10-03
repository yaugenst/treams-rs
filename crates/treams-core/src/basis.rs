//! Multipole bases of modes at expansion positions, and the expansion gradients shared by
//! the spherical and cylindrical families.
//!
//! Upstream: `treams.SphericalWaveBasis` and `treams.CylindricalWaveBasis`. The gradients
//! are a treams-rs extension.
//!
//! An expansion entry depends on the displacement `destination - source` between two
//! positions. Its position cotangent therefore adds to the destination position and
//! subtracts from the source position ([`ExpansionGradient`]).
#![allow(clippy::indexing_slicing)] // Basis and matrix dimensions validated on entry.

use crate::{Complex, Error, Result, numerics::finite};
use nalgebra::DMatrix;

/// The mode label of one multipole family: [`crate::sw::Mode`] or [`crate::cw::Mode`].
pub trait ModeLabel: Copy + std::fmt::Debug {
    /// Reject labels outside the supported range.
    fn validate(self) -> Result<()>;
    /// Polarization index: 1 or 0, as in treams.
    fn pol(self) -> u8;
}

/// Modes of one multipole family at Cartesian expansion positions; [`crate::sw::Basis`]
/// and [`crate::cw::Basis`] name the two families.
///
/// Upstream: `treams.SphericalWaveBasis` and `treams.CylindricalWaveBasis`.
#[derive(Clone, Debug)]
pub struct Basis<M> {
    /// Position index (upstream `pidx`) and mode label of each matrix axis entry.
    pub modes: Vec<(usize, M)>,
    /// Expansion centres, one per position index (upstream `positions`).
    pub positions: Vec<[f64; 3]>,
}

impl<M: ModeLabel> Basis<M> {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.modes.is_empty() || self.positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(
                "basis must be nonempty with finite positions".into(),
            ));
        }
        for &(pidx, mode) in &self.modes {
            mode.validate()?;
            if pidx >= self.positions.len() {
                return Err(Error::InvalidInput(
                    "basis position index outside positions".into(),
                ));
            }
        }
        Ok(())
    }

    /// Position index and polarization index of basis entry `i`.
    pub(crate) fn position_pol(&self, i: usize) -> (usize, usize) {
        let (pidx, mode) = self.modes[i];
        (pidx, usize::from(mode.pol()))
    }
}

/// Check the medium wavenumbers `ks` (negative, positive helicity) of a basis
/// operation: both finite, nonzero unless `allow_zero`, and equal for parity
/// polarizations (`helicity` false), because parity waves exist only in an achiral
/// medium. Operations that never mix polarizations pass `helicity = true`.
pub(crate) fn validate_wavenumbers(
    ks: [Complex; 2],
    helicity: bool,
    allow_zero: bool,
) -> Result<()> {
    if ks
        .iter()
        .any(|&k| !finite(k) || (!allow_zero && k == Complex::default()))
    {
        return Err(Error::InvalidInput(
            if allow_zero {
                "medium wavenumbers must be finite"
            } else {
                "medium wavenumbers must be finite and nonzero"
            }
            .into(),
        ));
    }
    if !helicity && ks[0] != ks[1] {
        return Err(Error::InvalidInput(
            "parity polarization requires equal wavenumbers (an achiral medium)".into(),
        ));
    }
    Ok(())
}

/// A spherical or cylindrical basis, for operations that accept either family: field
/// evaluation, plane-wave expansion and the EBCM surface waves.
#[derive(Clone, Debug)]
pub enum MultipoleBasis {
    /// Spherical vector waves.
    Spherical(crate::sw::Basis),
    /// Cylindrical vector waves; axial wavenumbers are fixed mode labels.
    Cylindrical(crate::cw::Basis),
}

impl From<crate::sw::Basis> for MultipoleBasis {
    fn from(value: crate::sw::Basis) -> Self {
        Self::Spherical(value)
    }
}

impl From<crate::cw::Basis> for MultipoleBasis {
    fn from(value: crate::cw::Basis) -> Self {
        Self::Cylindrical(value)
    }
}

impl MultipoleBasis {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Spherical(b) => b.validate(),
            Self::Cylindrical(b) => b.validate(),
        }
    }

    pub(crate) fn positions(&self) -> &[[f64; 3]] {
        match self {
            Self::Spherical(b) => &b.positions,
            Self::Cylindrical(b) => &b.positions,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Spherical(b) => b.modes.len(),
            Self::Cylindrical(b) => b.modes.len(),
        }
    }

    pub(crate) fn position_pol(&self, i: usize) -> (usize, usize) {
        match self {
            Self::Spherical(b) => b.position_pol(i),
            Self::Cylindrical(b) => b.position_pol(i),
        }
    }
}

/// Periodic coupling cotangents including the lattice geometry.
#[derive(Debug)]
pub struct LatticeExpansionGradient {
    /// Position and medium-wavenumber cotangents.
    pub expansion: ExpansionGradient,
    /// Bloch wavevector cotangent in lattice coordinates.
    pub kpar: Vec<f64>,
    /// Row lattice-vector cotangent.
    pub vectors: DMatrix<f64>,
}

impl LatticeExpansionGradient {
    /// Zero cotangents of `destination` and `source` positions, both wavenumbers and a
    /// `dim`-dimensional lattice.
    pub(crate) fn zeros(destination: usize, source: usize, dim: usize) -> Self {
        Self {
            expansion: ExpansionGradient::zeros(destination, source),
            kpar: vec![0.0; dim],
            vectors: DMatrix::zeros(dim, dim),
        }
    }

    /// Add the Bloch-vector and lattice-vector cotangents of one lattice sum.
    pub(crate) fn add_lattice(&mut self, g: &crate::lattice::SumGradient) {
        for (j, value) in self.kpar.iter_mut().enumerate() {
            *value += g.kpar[j];
            for i in 0..self.vectors.nrows() {
                self.vectors[(i, j)] += g.vectors[i][j];
            }
        }
    }
}

/// Expansion cotangents, separating destination and source position dependence.
#[derive(Clone, Debug)]
pub struct ExpansionGradient {
    /// Destination Cartesian position derivatives.
    pub destination: Vec<[f64; 3]>,
    /// Source Cartesian position derivatives.
    pub source: Vec<[f64; 3]>,
    /// Complex wave number cotangents in negative, positive helicity order.
    pub ks: [Complex; 2],
}

impl ExpansionGradient {
    /// Zero cotangents of `destination` and `source` positions and both wavenumbers.
    pub(crate) fn zeros(destination: usize, source: usize) -> Self {
        Self {
            destination: vec![[0.0; 3]; destination],
            source: vec![[0.0; 3]; source],
            ks: [Complex::default(); 2],
        }
    }

    /// Add the cotangents of another pullback over the same positions.
    pub(crate) fn accumulate(&mut self, other: &Self) {
        for (a, b) in self
            .destination
            .iter_mut()
            .chain(&mut self.source)
            .flatten()
            .zip(other.destination.iter().chain(&other.source).flatten())
        {
            *a += b;
        }
        for (a, b) in self.ks.iter_mut().zip(other.ks) {
            *a += b;
        }
    }
}

/// Pull the cotangent `cot` of an entry depending on the displacement
/// `destination[p] - source[q]` (derivative `position`) and the medium wavenumber
/// of polarization `pol` (derivative `k`) back into `result`.
pub(crate) fn add_pair_gradient(
    result: &mut ExpansionGradient,
    [p, q, pol]: [usize; 3],
    cot: Complex,
    position: [Complex; 3],
    k: Complex,
) {
    result.ks[pol] += cot * k.conj();
    for (axis, derivative) in position.into_iter().enumerate() {
        let gradient = (cot.conj() * derivative).re;
        result.destination[p][axis] += gradient;
        result.source[q][axis] -= gradient;
    }
}
