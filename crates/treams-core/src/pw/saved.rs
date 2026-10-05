//! Fixed numerical layouts for plane-wave derivative state.
//! treams-rs extension.

use super::{ExpansionResidual, FieldResidual, PermutationResidual, PhasesResidual};
use crate::{
    Complex, Result,
    basis::MultipoleBasis,
    saved::{
        Reader, SavedState, Writer, cw_basis_size, invalid, read_basis, read_complex,
        read_f64 as read_real, sw_basis_size, write_basis,
    },
};

fn size(header: usize, parts: &[(usize, usize)]) -> Result<usize> {
    parts.iter().try_fold(header, |total, &(count, width)| {
        count
            .checked_mul(width)
            .and_then(|part| total.checked_add(part))
            .ok_or_else(invalid)
    })
}

fn read_bool(reader: &mut Reader<'_>) -> Result<bool> {
    match reader.byte()? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid()),
    }
}

fn write_points(writer: &mut Writer, points: &[[f64; 3]]) {
    for &value in points.iter().flatten() {
        writer.f64(value);
    }
}

fn read_points(reader: &mut Reader<'_>, count: usize) -> Result<Vec<[f64; 3]>> {
    (0..count)
        .map(|_| Ok([read_real(reader)?, read_real(reader)?, read_real(reader)?]))
        .collect()
}

fn write_vectors(writer: &mut Writer, vectors: &[[Complex; 3]]) {
    for &value in vectors.iter().flatten() {
        writer.complex(value);
    }
}

fn read_vectors(reader: &mut Reader<'_>, count: usize) -> Result<Vec<[Complex; 3]>> {
    (0..count)
        .map(|_| {
            Ok([
                read_complex(reader)?,
                read_complex(reader)?,
                read_complex(reader)?,
            ])
        })
        .collect()
}

fn write_polarizations(writer: &mut Writer, polarizations: &[u8]) {
    for &value in polarizations {
        writer.byte(value);
    }
}

fn read_polarizations(reader: &mut Reader<'_>, count: usize) -> Result<Vec<u8>> {
    (0..count)
        .map(|_| {
            let pol = reader.byte()?;
            if pol <= 1 { Ok(pol) } else { Err(invalid()) }
        })
        .collect()
}

impl PhasesResidual {
    /// Byte count for phases at `points` sample points and `modes` plane waves.
    pub fn state_size(points: usize, modes: usize) -> Result<usize> {
        size(16, &[(points, 24), (modes, 48)])
    }
}

impl SavedState for PhasesResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.points.len(), self.vectors.len())?);
        writer.usize(self.points.len());
        writer.usize(self.vectors.len());
        write_points(&mut writer, &self.points);
        write_vectors(&mut writer, &self.vectors);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (points, modes) = (reader.usize()?, reader.usize()?);
        if modes == 0 || bytes.len() != Self::state_size(points, modes)? {
            return Err(invalid());
        }
        let points = read_points(&mut reader, points)?;
        let vectors = read_vectors(&mut reader, modes)?;
        reader.finish()?;
        Ok(Self { points, vectors })
    }
}

impl FieldResidual {
    /// Byte count for plane fields; weighted fields retain one amplitude per mode.
    pub fn state_size(points: usize, modes: usize, weighted: bool) -> Result<usize> {
        size(
            19,
            &[(points, 24), (modes, 49 + 16 * usize::from(weighted))],
        )
    }
}

impl SavedState for FieldResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(
            self.points.len(),
            self.vectors.len(),
            self.coefficients.is_some(),
        )?);
        writer.usize(self.points.len());
        writer.usize(self.vectors.len());
        writer.byte(u8::from(self.coefficients.is_some()));
        writer.byte(u8::from(self.helicity));
        writer.byte(u8::from(self.fixed_vectors));
        write_points(&mut writer, &self.points);
        write_vectors(&mut writer, &self.vectors);
        write_polarizations(&mut writer, &self.polarizations);
        if let Some(coefficients) = &self.coefficients {
            for &value in coefficients {
                writer.complex(value);
            }
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (points, modes) = (reader.usize()?, reader.usize()?);
        let weighted = read_bool(&mut reader)?;
        let helicity = read_bool(&mut reader)?;
        let fixed_vectors = read_bool(&mut reader)?;
        if modes == 0 || bytes.len() != Self::state_size(points, modes, weighted)? {
            return Err(invalid());
        }
        let points = read_points(&mut reader, points)?;
        let vectors = read_vectors(&mut reader, modes)?;
        let polarizations = read_polarizations(&mut reader, modes)?;
        let coefficients = weighted
            .then(|| {
                (0..modes)
                    .map(|_| read_complex(&mut reader))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        reader.finish()?;
        Ok(Self {
            vectors,
            polarizations,
            points,
            coefficients,
            helicity,
            fixed_vectors,
        })
    }
}

impl PermutationResidual {
    /// Byte count for a permutation of `modes` plane waves.
    pub fn state_size(modes: usize) -> Result<usize> {
        size(17, &[(modes, 49)])
    }
}

impl SavedState for PermutationResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.vectors.len())?);
        writer.usize(self.vectors.len());
        writer.usize(self.turns);
        writer.byte(u8::from(self.helicity));
        write_vectors(&mut writer, &self.vectors);
        write_polarizations(&mut writer, &self.polarizations);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let modes = reader.usize()?;
        let turns = reader.usize()?;
        let helicity = read_bool(&mut reader)?;
        if modes == 0 || turns > 2 || bytes.len() != Self::state_size(modes)? {
            return Err(invalid());
        }
        let vectors = read_vectors(&mut reader, modes)?;
        let polarizations = read_polarizations(&mut reader, modes)?;
        reader.finish()?;
        Ok(Self {
            vectors,
            polarizations,
            turns,
            helicity,
        })
    }
}

impl ExpansionResidual {
    /// Byte count for expansion into a spherical or cylindrical multipole basis.
    pub fn state_size(
        multipoles: usize,
        positions: usize,
        modes: usize,
        cylindrical: bool,
    ) -> Result<usize> {
        let basis = if cylindrical {
            cw_basis_size(multipoles, positions)?
        } else {
            sw_basis_size(multipoles, positions)?
        };
        size(11, &[(basis, 1), (modes, 49)])
    }
}

impl SavedState for ExpansionResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(
            self.basis.len(),
            self.basis.positions().len(),
            self.vectors.len(),
            matches!(self.basis, MultipoleBasis::Cylindrical(_)),
        )?);
        writer.usize(self.vectors.len());
        writer.byte(u8::from(self.helicity));
        writer.byte(u8::from(self.fixed_vectors));
        write_basis(&mut writer, &self.basis);
        write_vectors(&mut writer, &self.vectors);
        write_polarizations(&mut writer, &self.polarizations);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let modes = reader.count(49)?;
        let helicity = read_bool(&mut reader)?;
        let fixed_vectors = read_bool(&mut reader)?;
        let basis = read_basis(&mut reader)?;
        let cylindrical = matches!(basis, MultipoleBasis::Cylindrical(_));
        if modes == 0
            || bytes.len()
                != Self::state_size(basis.len(), basis.positions().len(), modes, cylindrical)?
        {
            return Err(invalid());
        }
        let vectors = read_vectors(&mut reader, modes)?;
        if cylindrical && vectors.iter().any(|vector| vector[2].im != 0.0) {
            return Err(invalid());
        }
        let polarizations = read_polarizations(&mut reader, modes)?;
        reader.finish()?;
        Ok(Self {
            basis,
            vectors,
            polarizations,
            helicity,
            fixed_vectors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pw::{expansion, field, permutation, phases},
        test_support::{bits, patterned},
    };

    fn restore<R: SavedState>(residual: &R, size: usize) -> R {
        let bytes = residual.save_state().unwrap();
        assert_eq!(bytes.len(), size);
        let restored = R::from_state(&bytes).unwrap();
        assert_eq!(restored.save_state().unwrap(), bytes);
        assert!(R::from_state(bytes.split_last().unwrap().1).is_err());
        restored
    }

    fn vectors() -> Vec<[Complex; 3]> {
        vec![[
            Complex::new(0.4, 0.0),
            Complex::new(0.3, 0.0),
            Complex::new(0.8, 0.0),
        ]]
    }

    #[test]
    fn restored_phases_preserve_both_derivatives() {
        let (_, residual) = phases(vec![[0.2, -0.4, 0.7]], vectors()).unwrap();
        let restored = restore(&residual, PhasesResidual::state_size(1, 1).unwrap());
        let dpoints = [[0.2, 0.1, -0.3]];
        let dvectors = [[Complex::new(0.1, -0.2); 3]];
        assert_eq!(
            residual.pushforward(&dpoints, &dvectors).unwrap(),
            restored.pushforward(&dpoints, &dvectors).unwrap()
        );
        let a = residual.pullback(&patterned(1, 1, 0.8)).unwrap();
        let b = restored.pullback(&patterned(1, 1, 0.8)).unwrap();
        assert_eq!(
            bits(&[&a.points, &a.vectors]),
            bits(&[&b.points, &b.vectors])
        );
    }

    #[test]
    fn restored_fields_preserve_weighted_and_operator_derivatives() {
        for weighted in [true, false] {
            let coefficients = weighted.then(|| vec![Complex::new(0.7, -0.2)]);
            let (_, residual) = field(
                vectors(),
                vec![1],
                vec![[0.2, -0.4, 0.7]],
                coefficients,
                true,
                false,
            )
            .unwrap();
            let restored = restore(
                &residual,
                FieldResidual::state_size(1, 1, weighted).unwrap(),
            );
            let coefficients = if weighted {
                vec![Complex::new(0.2, 0.1)]
            } else {
                vec![]
            };
            let dpoints = [[0.2, 0.1, -0.3]];
            let dvectors = [[Complex::new(0.1, -0.2); 3]];
            assert_eq!(
                residual
                    .pushforward(&coefficients, &dpoints, &dvectors)
                    .unwrap(),
                restored
                    .pushforward(&coefficients, &dpoints, &dvectors)
                    .unwrap()
            );
            let a = residual.pullback(&patterned(3, 1, 0.8)).unwrap();
            let b = restored.pullback(&patterned(3, 1, 0.8)).unwrap();
            assert_eq!(
                bits(&[&a.coefficients, &a.points, &a.vectors]),
                bits(&[&b.coefficients, &b.points, &b.vectors])
            );
        }
    }

    #[test]
    fn restored_permutations_preserve_both_derivatives() {
        let (_, residual) = permutation(vectors(), vec![1], 2, true).unwrap();
        let restored = restore(&residual, PermutationResidual::state_size(1).unwrap());
        let dvectors = [[Complex::new(0.1, -0.2); 3]];
        assert_eq!(
            residual.pushforward(&dvectors).unwrap(),
            restored.pushforward(&dvectors).unwrap()
        );
        assert_eq!(
            residual.pullback(&patterned(2, 1, 0.8)).unwrap(),
            restored.pullback(&patterned(2, 1, 0.8)).unwrap()
        );
    }

    #[test]
    fn restored_expansions_preserve_both_basis_families() {
        let spherical = MultipoleBasis::Spherical(crate::sw::Basis {
            modes: vec![(0, crate::sw::Mode { l: 1, m: 0, pol: 1 })],
            positions: vec![[0.2, -0.4, 0.7]],
        });
        let cylindrical = MultipoleBasis::Cylindrical(crate::cw::Basis {
            modes: vec![(
                0,
                crate::cw::Mode {
                    kz: 0.8,
                    m: 1,
                    pol: 1,
                },
            )],
            positions: vec![[0.2, -0.4, 0.7]],
        });
        for basis in [spherical, cylindrical] {
            let cylindrical = matches!(basis, MultipoleBasis::Cylindrical(_));
            let (_, residual) = expansion(basis, vectors(), vec![1], true, false).unwrap();
            let restored = restore(
                &residual,
                ExpansionResidual::state_size(1, 1, 1, cylindrical).unwrap(),
            );
            let dpoints = [[0.2, 0.1, -0.3]];
            let dvectors = [[
                Complex::new(0.1, -0.2),
                Complex::new(0.2, 0.1),
                Complex::default(),
            ]];
            assert_eq!(
                residual.pushforward(&dpoints, &dvectors).unwrap(),
                restored.pushforward(&dpoints, &dvectors).unwrap()
            );
            let a = residual.pullback(&patterned(1, 1, 0.8)).unwrap();
            let b = restored.pullback(&patterned(1, 1, 0.8)).unwrap();
            assert_eq!(
                bits(&[&a.positions, &a.vectors]),
                bits(&[&b.positions, &b.vectors])
            );
        }
    }
}
