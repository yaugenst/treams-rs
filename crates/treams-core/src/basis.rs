//! Spherical basis expansion with reusable angular plans and analytic pullbacks.
#![allow(clippy::indexing_slicing)] // Basis and matrix dimensions validated on entry.

use crate::{
    Complex, Error, Result, finite, special::Radial, translation_plan::TranslationPlan, waves::Mode,
};
use nalgebra::DMatrix;
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

/// Spherical modes attached to Cartesian expansion origins.
#[derive(Clone, Debug)]
pub struct Basis {
    /// Particle index and spherical mode for each matrix axis entry.
    pub modes: Vec<(usize, Mode)>,
    /// Cartesian expansion origins.
    pub positions: Vec<[f64; 3]>,
}
impl Basis {
    fn validate(&self) -> Result<()> {
        if self.modes.is_empty() || self.positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(
                "basis must be nonempty with finite positions".into(),
            ));
        }
        for &(particle, mode) in &self.modes {
            mode.validate()?;
            if particle >= self.positions.len() {
                return Err(Error::InvalidInput(
                    "basis particle index outside positions".into(),
                ));
            }
        }
        Ok(())
    }
    fn groups(&self) -> BTreeMap<usize, (Vec<usize>, Vec<Mode>)> {
        let mut groups = BTreeMap::<usize, (Vec<usize>, Vec<Mode>)>::new();
        for (index, &(particle, mode)) in self.modes.iter().enumerate() {
            let group = groups.entry(particle).or_default();
            group.0.push(index);
            group.1.push(mode);
        }
        groups
    }
}
#[derive(Clone, Debug)]
struct Block {
    destination: usize,
    source: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    polarizations: Vec<u8>,
    plan: Arc<TranslationPlan>,
}

/// Native expansion context. Derivative tables are recomputed per pair on pullback.
#[derive(Clone, Debug)]
pub struct TranslationResidual {
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    radial: Radial,
    blocks: Vec<Block>,
    /// Expansion matrix from source coefficients to destination coefficients.
    pub value: DMatrix<Complex>,
}

/// Construct a regular or outgoing spherical expansion matrix.
pub fn translation(
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
) -> Result<TranslationResidual> {
    destination.validate()?;
    source.validate()?;
    if ks.iter().any(|&k| !finite(k)) || (!helicity && ks[0] != ks[1]) {
        return Err(Error::InvalidInput(
            "finite wave numbers required; parity requires an achiral medium".into(),
        ));
    }
    let mut plans = HashMap::new();
    let mut value = DMatrix::zeros(destination.modes.len(), source.modes.len());
    let mut blocks = Vec::new();
    let source_groups = source.groups();
    for (p, (rows, to)) in destination.groups() {
        for (&q, (cols, from)) in &source_groups {
            let key = (to.clone(), from.clone());
            let plan = if let Some(plan) = plans.get(&key) {
                Arc::clone(plan)
            } else {
                let plan = Arc::new(TranslationPlan::between(&to, from, helicity)?);
                plans.insert(key, Arc::clone(&plan));
                plan
            };
            let displacement =
                std::array::from_fn(|a| destination.positions[p][a] - source.positions[q][a]);
            let first = plan.evaluate_with(ks[0], displacement, radial)?;
            let second = if ks[0] == ks[1] {
                None
            } else {
                Some(plan.evaluate_with(ks[1], displacement, radial)?)
            };
            for (j, &col) in cols.iter().enumerate() {
                let values = if from[j].pol == 1 {
                    second.as_ref().unwrap_or(&first)
                } else {
                    &first
                };
                for (i, &row) in rows.iter().enumerate() {
                    value[(row, col)] = values[j * rows.len() + i];
                }
            }
            blocks.push(Block {
                destination: p,
                source: q,
                rows: rows.clone(),
                cols: cols.clone(),
                polarizations: from.iter().map(|m| m.pol).collect(),
                plan,
            });
        }
    }
    Ok(TranslationResidual {
        destination,
        source,
        ks,
        radial,
        blocks,
        value,
    })
}

/// Expansion cotangents, separating destination and source origin dependence.
#[derive(Clone, Debug)]
pub struct TranslationGradient {
    /// Destination Cartesian origin derivatives.
    pub destination: Vec<[f64; 3]>,
    /// Source Cartesian origin derivatives.
    pub source: Vec<[f64; 3]>,
    /// Complex wave number cotangents in negative, positive helicity order.
    pub ks: [Complex; 2],
}
impl TranslationResidual {
    /// Contract the expansion cotangent under the real Hermitian pairing.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<TranslationGradient> {
        if cotangent.shape() != self.value.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid expansion cotangent".into()));
        }
        let mut result = TranslationGradient {
            destination: vec![[0.0; 3]; self.destination.positions.len()],
            source: vec![[0.0; 3]; self.source.positions.len()],
            ks: [Complex::default(); 2],
        };
        for block in self.blocks {
            let displacement = std::array::from_fn(|a| {
                self.destination.positions[block.destination][a]
                    - self.source.positions[block.source][a]
            });
            for pol in 0..2 {
                let mut g = vec![Complex::default(); block.rows.len() * block.cols.len()];
                for (j, &col) in block.cols.iter().enumerate() {
                    if usize::from(block.polarizations[j]) != pol {
                        continue;
                    }
                    for (i, &row) in block.rows.iter().enumerate() {
                        g[j * block.rows.len() + i] = cotangent[(row, col)];
                    }
                }
                let (position, k) =
                    block
                        .plan
                        .pullback_with(self.ks[pol], displacement, self.radial, &g)?;
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
