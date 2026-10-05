//! Spherical expansion matrices between two bases and over a lattice, with analytic
//! pullbacks. Each pair of a destination and a source position forms one block, and
//! blocks with the same modes share one [`TranslationPlan`].
//!
//! Upstream: the `treams.Expand` and `treams.ExpandLattice` operators between spherical
//! bases.
#![allow(clippy::indexing_slicing)] // Basis and matrix dimensions validated on entry.

use super::{Basis, plan::TranslationPlan};
use crate::{
    Complex, Error, Result,
    basis::{ExpansionGradient, LatticeExpansionGradient, validate_wavenumbers},
    numerics::{self, finite},
    special::Radial,
};
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::{collections::HashMap, sync::Arc};

#[path = "expansion_lattice_state.rs"]
mod lattice_state;

/// The modes of one destination position against those of one source position.
#[derive(Clone, Debug)]
struct Block {
    destination: usize,
    source: usize,
    /// Destination minus source position.
    displacement: [f64; 3],
    rows: Vec<usize>,
    cols: Vec<usize>,
    polarizations: Vec<u8>,
    plan: Arc<TranslationPlan>,
}

impl Block {
    fn len(&self) -> usize {
        self.rows.len() * self.cols.len()
    }

    /// A shared polarization uses one buffer; distinct polarizations use two.
    fn values(
        shared: bool,
        evaluate: impl Fn(usize) -> Result<Option<Vec<Complex>>>,
    ) -> Result<[Option<Vec<Complex>>; 2]> {
        Ok([evaluate(0)?, if shared { None } else { evaluate(1)? }])
    }

    /// The block of `matrix`, column-major, with the columns of other polarizations
    /// than `pol` zeroed when `pol` is given.
    fn gather(&self, matrix: &DMatrix<Complex>, pol: Option<usize>) -> Vec<Complex> {
        let mut values = vec![Complex::default(); self.len()];
        for (column, (&col, &p)) in values
            .chunks_exact_mut(self.rows.len())
            .zip(self.cols.iter().zip(&self.polarizations))
        {
            if pol.is_none_or(|pol| usize::from(p) == pol) {
                for (value, &row) in column.iter_mut().zip(&self.rows) {
                    *value = matrix[(row, col)];
                }
            }
        }
        values
    }

    /// Write column-major block values into `matrix`, only in the columns of
    /// polarization `pol` when it is given.
    fn scatter(&self, values: &[Complex], pol: Option<usize>, matrix: &mut DMatrix<Complex>) {
        for (column, (&col, &p)) in values
            .chunks_exact(self.rows.len())
            .zip(self.cols.iter().zip(&self.polarizations))
        {
            if pol.is_none_or(|pol| usize::from(p) == pol) {
                for (&value, &row) in column.iter().zip(&self.rows) {
                    matrix[(row, col)] = value;
                }
            }
        }
    }
}

/// Validate both bases and the medium of an expansion.
fn validate_expansion(
    destination: &Basis,
    source: &Basis,
    ks: [Complex; 2],
    helicity: bool,
) -> Result<()> {
    destination.validate()?;
    source.validate()?;
    // Spherical expansions accept k = 0, where the regular expansion is the identity.
    validate_wavenumbers(ks, helicity, true)
}

/// What [`expansion`] saves for its pullback: the bases, the wavenumbers, the radial
/// kind and every block with its translation plan. The pullback recomputes the
/// derivative tables per block.
///
/// The pullback evaluates blocks in parallel and adds their gradients in block order, so
/// its result does not depend on the thread count.
#[derive(Clone, Debug)]
pub struct ExpansionResidual {
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    radial: Radial,
    helicity: bool,
    blocks: Vec<Block>,
}

/// Construct a regular or singular spherical expansion matrix, which maps source
/// coefficients to destination coefficients.
///
/// Upstream: the `treams.Expand` operator (`treams.operators.expand`) between spherical
/// bases.
pub fn expansion(
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
) -> Result<(DMatrix<Complex>, ExpansionResidual)> {
    let residual = ExpansionResidual::prepare(destination, source, ks, helicity, radial)?;
    let value = assemble(residual.shape(), &residual.blocks, BATCH, |block, _| {
        Block::values(ks[0] == ks[1], |pol| {
            expansion_block(&block.plan, ks[pol], block.displacement, radial)
        })
    })?;
    Ok((value, residual))
}

impl ExpansionResidual {
    fn prepare(
        destination: Basis,
        source: Basis,
        ks: [Complex; 2],
        helicity: bool,
        radial: Radial,
    ) -> Result<Self> {
        validate_expansion(&destination, &source, ks, helicity)?;
        let blocks = blocks(&destination, &source, helicity, false)?;
        Ok(Self {
            destination,
            source,
            ks,
            radial,
            helicity,
            blocks,
        })
    }

    /// Fixed saved-state bytes for the two basis shapes.
    pub fn state_size(
        destination_modes: usize,
        destination_positions: usize,
        source_modes: usize,
        source_positions: usize,
    ) -> Result<usize> {
        let destination = crate::saved::sw_basis_size(destination_modes, destination_positions)?;
        let source = crate::saved::sw_basis_size(source_modes, source_positions)?;
        destination
            .checked_add(source)
            .and_then(|n| n.checked_add(34))
            .ok_or_else(crate::saved::invalid)
    }

    /// Append state for a containing residual without allocating an intermediate buffer.
    pub(crate) fn write_state(&self, writer: &mut crate::saved::Writer) {
        crate::saved::write_sw_basis(writer, &self.destination);
        crate::saved::write_sw_basis(writer, &self.source);
        for k in self.ks {
            writer.complex(k);
        }
        crate::saved::write_radial(writer, self.radial);
        writer.byte(u8::from(self.helicity));
    }
    /// Destination and source mode counts: the shape of the expansion matrix.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.destination.modes.len(), self.source.modes.len())
    }

    /// Directional derivative in the positions and the two medium wavenumbers.
    pub fn pushforward(
        &self,
        destination: &[[f64; 3]],
        source: &[[f64; 3]],
        ks: [Complex; 2],
    ) -> Result<DMatrix<Complex>> {
        crate::basis::validate_expansion_tangent(
            destination,
            source,
            ks,
            (
                self.destination.positions.len(),
                self.source.positions.len(),
            ),
        )?;
        assemble(self.shape(), &self.blocks, BATCH, |block, _| {
            if self.radial == Radial::Singular && block.displacement.iter().all(|&x| x == 0.0) {
                return Ok([None, None]);
            }
            let direction = std::array::from_fn(|axis| {
                destination[block.destination][axis] - source[block.source][axis]
            });
            Block::values(self.ks[0] == self.ks[1] && ks[0] == ks[1], |pol| {
                block
                    .plan
                    .pushforward(
                        self.ks[pol],
                        block.displacement,
                        self.radial,
                        direction,
                        ks[pol],
                    )
                    .map(Some)
            })
        })
    }

    /// Position and wavenumber gradients from `cotangent`, the gradient of a real loss
    /// with respect to the matrix.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<ExpansionGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid expansion cotangent".into()));
        }
        // Blocks run in parallel; their gradients are summed in block order.
        let gradients = crate::threads::install(|| {
            self.blocks
                .par_iter()
                .map(|block| {
                    let mut gradients = [([0.0; 3], Complex::default()); 2];
                    if self.radial == Radial::Singular
                        && block.displacement.iter().all(|&x| x == 0.0)
                    {
                        return Ok(gradients);
                    }
                    for (pol, gradient) in gradients.iter_mut().enumerate() {
                        let g = block.gather(cotangent, Some(pol));
                        if g.iter().any(|&z| z != Complex::default()) {
                            *gradient = block.plan.pullback(
                                self.ks[pol],
                                block.displacement,
                                self.radial,
                                &g,
                            )?;
                        }
                    }
                    Ok(gradients)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut result = ExpansionGradient::zeros(
            self.destination.positions.len(),
            self.source.positions.len(),
        );
        for (block, gradients) in self.blocks.iter().zip(gradients) {
            for (pol, (position, k)) in gradients.into_iter().enumerate() {
                result.ks[pol] += k;
                for (axis, grad) in position.into_iter().enumerate() {
                    result.destination[block.destination][axis] += grad;
                    result.source[block.source][axis] -= grad;
                }
            }
        }
        Ok(result)
    }
}

impl crate::saved::SavedState for ExpansionResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = crate::saved::Writer::new(Self::state_size(
            self.destination.modes.len(),
            self.destination.positions.len(),
            self.source.modes.len(),
            self.source.positions.len(),
        )?);
        self.write_state(&mut writer);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut shape = crate::saved::Reader::new(bytes);
        let (dm, dp) = crate::saved::read_sw_basis_dimensions(&mut shape)?;
        let (sm, sp) = crate::saved::read_sw_basis_dimensions(&mut shape)?;
        if bytes.len() != Self::state_size(dm, dp, sm, sp)? {
            return Err(crate::saved::invalid());
        }
        let mut reader = crate::saved::Reader::new(bytes);
        let destination = crate::saved::read_sw_basis(&mut reader)?;
        let source = crate::saved::read_sw_basis(&mut reader)?;
        let ks = [reader.complex()?, reader.complex()?];
        let radial = crate::saved::read_radial(&mut reader)?;
        let helicity = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return Err(crate::saved::invalid()),
        };
        reader.finish()?;
        // Angular plans depend only on static mode labels. No translation values are reevaluated.
        Self::prepare(destination, source, ks, helicity, radial)
    }
}

fn blocks(
    destination: &Basis,
    source: &Basis,
    helicity: bool,
    normalized_harmonics: bool,
) -> Result<Vec<Block>> {
    let mut plans = HashMap::new();
    let (destination_groups, source_groups) = (destination.groups(), source.groups());
    let mut blocks = Vec::new();
    // A saturated count fails the reservation.
    numerics::reserve(
        &mut blocks,
        destination_groups.len().saturating_mul(source_groups.len()),
    )?;
    for (p, (rows, to)) in destination_groups {
        let position = destination.positions[p];
        for (&q, (cols, from)) in &source_groups {
            let key = (to.clone(), from.clone());
            let plan = if let Some(plan) = plans.get(&key) {
                Arc::clone(plan)
            } else {
                let mut plan = TranslationPlan::between(&to, from, helicity)?;
                if normalized_harmonics {
                    plan.weights_for_normalized_harmonics();
                }
                let plan = Arc::new(plan);
                plans.insert(key, Arc::clone(&plan));
                plan
            };
            blocks.push(Block {
                destination: p,
                source: q,
                displacement: std::array::from_fn(|a| position[a] - source.positions[q][a]),
                rows: rows.clone(),
                cols: cols.clone(),
                polarizations: from.iter().map(|m| m.pol).collect(),
                plan,
            });
        }
    }
    Ok(blocks)
}

/// The regular or singular translation of one block, or `None` for the singular
/// self block: outgoing waves do not couple a position to itself.
fn expansion_block(
    plan: &TranslationPlan,
    k: Complex,
    displacement: [f64; 3],
    radial: Radial,
) -> Result<Option<Vec<Complex>>> {
    if radial == Radial::Singular && displacement.iter().all(|&x| x == 0.0) {
        return Ok(None);
    }
    plan.evaluate(k, displacement, radial).map(Some)
}

/// Entries of the block values evaluated at once, bounding the transient memory
/// of the parallel assembly (16 MiB, twice that for two wavenumbers).
const BATCH: usize = 1 << 20;

/// Blocks fitting the scratch-entry budget, or one block if it alone exceeds it.
fn batch_count(blocks: &[Block], budget: usize) -> usize {
    let mut size = 0;
    blocks
        .iter()
        .take_while(|block| {
            size += block.len();
            size <= budget
        })
        .count()
        .max(1)
}

/// Evaluate the blocks in parallel, in batches of at most `batch` entries or of one
/// larger block, and scatter them in order. A missing second polarization shares
/// the first; two missing polarizations skip the block. The evaluator also receives
/// the worker budget of the current batch for nested lattice evaluations.
fn assemble(
    shape: (usize, usize),
    blocks: &[Block],
    batch: usize,
    evaluate: impl Fn(&Block, usize) -> Result<[Option<Vec<Complex>>; 2]> + Sync,
) -> Result<DMatrix<Complex>> {
    let mut value = numerics::zeros(shape.0, shape.1)?;
    let mut rest = blocks;
    while !rest.is_empty() {
        let count = batch_count(rest, batch);
        let (current, next) = rest.split_at(count);
        rest = next;
        let workers = crate::threads::current_num_threads().div_ceil(current.len());
        let values = crate::threads::install(|| {
            current
                .par_iter()
                .map(|block| evaluate(block, workers))
                .collect::<Result<Vec<_>>>()
        })?;
        for (block, [first, second]) in current.iter().zip(values) {
            if let Some(first) = first {
                block.scatter(&first, second.as_ref().map(|_| 0), &mut value);
            }
            if let Some(second) = second {
                block.scatter(&second, Some(1), &mut value);
            }
        }
    }
    Ok(value)
}

/// Periodic outgoing-to-regular spherical coupling, including nonzero lattice images of self blocks.
///
/// Upstream: `treams.sw.translate_periodic` and the `treams.ExpandLattice` operator
/// (`treams.operators.expandlattice`) between spherical bases.
pub fn lattice_expansion(
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    helicity: bool,
    lattice: crate::lattice::BlochLattice,
    eta: Complex,
) -> Result<(DMatrix<Complex>, LatticeExpansionResidual)> {
    validate_expansion(&destination, &source, ks, helicity)?;
    let blocks = blocks(&destination, &source, helicity, false)?;
    let workers = crate::threads::current_num_threads().div_ceil(blocks.len().max(1));
    let value = assemble(
        (destination.modes.len(), source.modes.len()),
        &blocks,
        BATCH,
        |block, _| {
            Block::values(ks[0] == ks[1], |pol| {
                block
                    .plan
                    .evaluate_periodic(ks[pol], block.displacement, &lattice, eta, workers)
                    .map(Some)
            })
        },
    )?;
    Ok((
        value,
        LatticeExpansionResidual {
            destination,
            source,
            ks,
            lattice,
            eta,
            helicity,
            blocks,
        },
    ))
}

/// What [`lattice_expansion`] saves for its pullback: the bases, the wavenumbers, the
/// lattice, the split and the blocks, whose angular plans every position pair shares.
///
/// The pullback evaluates bounded batches of blocks in parallel and adds their
/// gradients in block and harmonic order, so the thread count does not change it.
#[derive(Debug)]
pub struct LatticeExpansionResidual {
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    lattice: crate::lattice::BlochLattice,
    eta: Complex,
    helicity: bool,
    blocks: Vec<Block>,
}

impl LatticeExpansionResidual {
    /// Directional derivative in destination/source positions, medium wavenumbers,
    /// Bloch wavevector and lattice vectors. The Ewald split remains fixed.
    pub fn pushforward(
        &self,
        destination: &[[f64; 3]],
        source: &[[f64; 3]],
        ks: [Complex; 2],
        kpar: &[f64],
        vectors: &DMatrix<f64>,
    ) -> Result<DMatrix<Complex>> {
        crate::basis::validate_expansion_tangent(
            destination,
            source,
            ks,
            (
                self.destination.positions.len(),
                self.source.positions.len(),
            ),
        )?;
        let geometry =
            crate::lattice::SumTangent::with_lattice(kpar, vectors, self.lattice.dimension())?;
        assemble(self.shape(), &self.blocks, BATCH, |block, workers| {
            let shift = std::array::from_fn(|a| {
                source[block.source][a] - destination[block.destination][a]
            });
            let tangents = ks.map(|k| crate::lattice::SumTangent {
                k,
                shift,
                ..geometry
            });
            block
                .plan
                .pushforward_periodic(
                    self.ks,
                    block.displacement,
                    &self.lattice,
                    self.eta,
                    &tangents,
                    workers,
                )
                .map(|values| values.map(Some))
        })
    }

    /// Destination and source mode counts: the shape of the coupling matrix.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.destination.modes.len(), self.source.modes.len())
    }

    /// Position, wavenumber, Bloch-vector and lattice-vector gradients from `cotangent`;
    /// the pullback recomputes the analytic derivatives of the Ewald sums.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<LatticeExpansionGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "invalid periodic expansion cotangent".into(),
            ));
        }
        let mut result = LatticeExpansionGradient::zeros(
            self.destination.positions.len(),
            self.source.positions.len(),
            self.lattice.dimension(),
        );
        let mut rest = self.blocks.as_slice();
        while !rest.is_empty() {
            let (current, next) = rest.split_at(batch_count(rest, BATCH));
            rest = next;
            let workers = crate::threads::current_num_threads().div_ceil(current.len());
            let gradients = crate::threads::install(|| {
                current
                    .par_iter()
                    .map(|block| {
                        let g = [0, 1].map(|pol| block.gather(cotangent, Some(pol)));
                        block.plan.pullback_periodic(
                            self.ks,
                            block.displacement,
                            &self.lattice,
                            self.eta,
                            &g,
                            workers,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            for (block, gradients) in current.iter().zip(gradients) {
                for (pol, gradient) in gradients.into_iter().enumerate() {
                    result.add_lattice(&gradient);
                    result.expansion.ks[pol] += gradient.k;
                    for (axis, value) in gradient.shift.into_iter().enumerate() {
                        result.expansion.destination[block.destination][axis] += value;
                        result.expansion.source[block.source][axis] -= value;
                    }
                }
            }
        }
        Ok(result)
    }
}

/// What [`lattice_expansion_from_table`] saves for its pullback: the sparse angular
/// weights, never the table values. The pullback runs in block order and gives the same
/// result every run.
#[derive(Debug)]
pub struct LatticeExpansionFromTableResidual {
    destination: Basis,
    source: Basis,
    helicity: bool,
    blocks: Vec<Block>,
    table_shape: [usize; 4],
    shape: (usize, usize),
}

/// The lattice expansion matrix from a table of lattice sums, a linear map of the table.
///
/// The table axes are destination position, source position, wavenumber channel (one, or
/// two for distinct helicity wavenumbers) and harmonic index `p * p + p + m` for
/// `p <= lmax`, where `lmax` is the largest destination degree plus the largest source
/// degree. Each entry holds a lattice sum of orthonormal spherical harmonics, with the
/// normalization `sqrt((2p + 1) / 4 pi) sqrt((p - m)! / (p + m)!)` of
/// [`crate::lattice::sum`] for [`crate::lattice::Family::Spherical`].
///
/// Upstream: `treams.sw.translate_periodic` after its `dlms` table of
/// `treams.lattice.lsumsw` sums, which has the same axes and normalization.
pub fn lattice_expansion_from_table(
    destination: &Basis,
    source: &Basis,
    helicity: bool,
    channels: usize,
    table: &[Complex],
) -> Result<(DMatrix<Complex>, LatticeExpansionFromTableResidual)> {
    destination.validate()?;
    source.validate()?;
    if !(1..=2).contains(&channels) || (!helicity && channels != 1) {
        return Err(Error::InvalidInput(
            "lattice table needs one or two channels; parity needs one".into(),
        ));
    }
    let lmax = destination
        .modes
        .iter()
        .map(|(_, m)| m.l)
        .max()
        .unwrap_or(0)
        + source.modes.iter().map(|(_, m)| m.l).max().unwrap_or(0);
    let harmonics = usize::try_from((lmax + 1).pow(2))
        .map_err(|_| Error::InvalidInput("invalid harmonic count".into()))?;
    let table_shape = [
        destination.positions.len(),
        source.positions.len(),
        channels,
        harmonics,
    ];
    let size = table_shape
        .iter()
        .try_fold(1_usize, |size, &length| size.checked_mul(length));
    if size != Some(table.len()) || table.iter().any(|&value| !finite(value)) {
        return Err(Error::InvalidInput(
            "lattice table requires matching finite harmonic values".into(),
        ));
    }
    let blocks = blocks(destination, source, helicity, true)?;
    let shape = (destination.modes.len(), source.modes.len());
    let residual = LatticeExpansionFromTableResidual {
        destination: destination.clone(),
        source: source.clone(),
        helicity,
        blocks,
        table_shape,
        shape,
    };
    Ok((residual.apply_table(table)?, residual))
}

impl LatticeExpansionFromTableResidual {
    /// The same sparse linear angular map applied to the table direction.
    pub fn pushforward(&self, tangent: &[Complex]) -> Result<DMatrix<Complex>> {
        if tangent.len() != self.table_shape.iter().product::<usize>()
            || tangent.iter().any(|&value| !finite(value))
        {
            return Err(Error::InvalidInput(
                "lattice table tangent must be finite and match the table".into(),
            ));
        }
        self.apply_table(tangent)
    }

    /// Values and directions use one sparse linear contraction.
    fn apply_table(&self, tangent: &[Complex]) -> Result<DMatrix<Complex>> {
        let [_, sources, channels, harmonics] = self.table_shape;
        let mut result = numerics::zeros(self.shape.0, self.shape.1)?;
        for block in &self.blocks {
            let offset = (block.destination * sources + block.source) * channels * harmonics;
            for channel in 0..channels {
                let table =
                    &tangent[offset + channel * harmonics..offset + (channel + 1) * harmonics];
                let evaluated = block.plan.evaluate_table(table);
                block.scatter(&evaluated, (channels == 2).then_some(channel), &mut result);
            }
        }
        if result.iter().any(|&value| !finite(value)) {
            return Err(Error::NonFinite("lattice table sum overflows".into()));
        }
        Ok(result)
    }

    /// Destination and source mode counts: the shape of the coupling matrix.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The shape of the normalized harmonic table, which is also the shape of the
    /// table gradient.
    #[must_use]
    pub const fn table_shape(&self) -> [usize; 4] {
        self.table_shape
    }

    /// The table gradient: the conjugate transpose of the fixed angular weights applied to
    /// `cotangent`.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<Vec<Complex>> {
        if cotangent.shape() != self.shape || cotangent.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "invalid lattice table cotangent".into(),
            ));
        }
        let [_, sources, channels, harmonics] = self.table_shape;
        let mut gradient = vec![Complex::default(); self.table_shape.iter().product()];
        for block in &self.blocks {
            let offset = (block.destination * sources + block.source) * channels * harmonics;
            for channel in 0..channels {
                let local = block.gather(cotangent, (channels == 2).then_some(channel));
                block.plan.pullback_table(
                    &local,
                    &mut gradient[offset + channel * harmonics..offset + (channel + 1) * harmonics],
                );
            }
        }
        Ok(gradient)
    }
}

#[cfg(test)]
mod tests {
    //! Batched assembly of the expansion matrix. Identities of expansions are in
    //! `properties/waves.rs`.

    use super::{Basis, Block, assemble, blocks, expansion, expansion_block};
    use crate::{Complex, special::Radial, sw};

    /// Bounded batches, down to one block at a time, assemble the matrix of a single
    /// batch, for both radial kinds with equal and distinct helicity wavenumbers.
    #[test]
    fn batches_assemble_the_same_matrix() {
        // Positions with modes up to degree 1, 2 and 1 make blocks of 36, 96 and 256
        // entries. Batches of 150 entries hold two blocks or the largest one alone,
        // batches of 300 up to four blocks.
        let basis = Basis {
            modes: [1, 2, 1]
                .into_iter()
                .enumerate()
                .flat_map(|(p, lmax)| {
                    let modes = sw::modes(lmax).unwrap();
                    modes.into_iter().map(move |mode| (p, mode))
                })
                .collect(),
            positions: vec![[0.0; 3], [0.9, 0.2, 0.3], [-0.4, 0.8, -0.5]],
        };
        let blocks = blocks(&basis, &basis, true, false).unwrap();
        let k = Complex::new(1.3, 0.05);
        for ks in [[k, k], [k, k * 1.4]] {
            for radial in [Radial::Regular, Radial::Singular] {
                let (expected, _) =
                    expansion(basis.clone(), basis.clone(), ks, true, radial).unwrap();
                for batch in [1, 150, 300] {
                    let value = assemble(
                        (basis.modes.len(), basis.modes.len()),
                        &blocks,
                        batch,
                        |block, _| {
                            Block::values(ks[0] == ks[1], |pol| {
                                expansion_block(&block.plan, ks[pol], block.displacement, radial)
                            })
                        },
                    )
                    .unwrap();
                    assert_eq!(value, expected, "batch {batch}, {radial:?}");
                }
            }
        }
    }
}
