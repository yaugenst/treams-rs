//! Fixed-shape metric gradient state, including deferred nonsmoothness errors.
//! treams-rs extension.

use super::MetricResidual;
use crate::saved::{Reader, SavedState, Writer, invalid};
use crate::{Complex, DerivativeError, Error, Result};
use nalgebra::DMatrix;

// Status zero means a saved gradient; the remaining tags identify the only
// errors chirality and SvdvalsResidual::pullback can save in MetricResidual.
// Keep the existing tags 1..=4 and payload layout when wording changes.
const GRADIENT_ERRORS: [DerivativeError; 4] = [
    DerivativeError::ZeroChiralityContrast,
    DerivativeError::InvalidSingularValueCotangent,
    DerivativeError::UnresolvedSingularValue,
    DerivativeError::UnequalSingularValueWeights,
];

impl MetricResidual {
    /// Number of bytes in the fixed derivative state for a square metric input.
    pub fn state_size(dimension: usize) -> Result<usize> {
        if dimension == 0 {
            return Err(invalid());
        }
        dimension
            .checked_mul(dimension)
            .and_then(|count| count.checked_mul(16))
            .and_then(|bytes| bytes.checked_add(25))
            .ok_or_else(invalid)
    }
}

impl SavedState for MetricResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let status = match &self.gradient {
            Ok(_) => 0,
            Err(Error::Derivative(reason)) => GRADIENT_ERRORS
                .iter()
                .zip(1_u8..)
                .find_map(|(known, status)| (known == reason).then_some(status))
                .ok_or_else(invalid)?,
            Err(_) => return Err(invalid()),
        };
        let mut writer = Writer::new(Self::state_size(self.dimension)?);
        writer.usize(self.dimension);
        writer.byte(status);
        for gradient in self.ks_gradient {
            writer.f64(gradient);
        }
        match &self.gradient {
            Ok(gradient) => {
                for &value in gradient.iter() {
                    writer.complex(value);
                }
            }
            Err(_) => {
                // An undefined gradient has the same shape as a valid one and
                // reserves canonical zeros instead of variable-length text.
                for _ in 0..self.dimension * self.dimension {
                    writer.complex(Complex::default());
                }
            }
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let dimension = reader.usize()?;
        if bytes.len() != Self::state_size(dimension)? {
            return Err(invalid());
        }
        let status = reader.byte()?;
        let error = if status == 0 {
            None
        } else {
            Some(
                *GRADIENT_ERRORS
                    .get(usize::from(status) - 1)
                    .ok_or_else(invalid)?,
            )
        };
        let ks_gradient = [reader.f64()?, reader.f64()?];
        let gradient = if let Some(reason) = error {
            for _ in 0..dimension * dimension {
                if reader.complex()? != Complex::default() {
                    return Err(invalid());
                }
            }
            Err(Error::Derivative(reason))
        } else {
            let values = (0..dimension * dimension)
                .map(|_| reader.complex())
                .collect::<Result<Vec<_>>>()?;
            Ok(DMatrix::from_vec(dimension, dimension, values))
        };
        reader.finish()?;
        Ok(Self {
            gradient,
            ks_gradient,
            dimension,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // Serialization preserves numerical bits exactly.

    use super::*;
    use crate::tmatrix::metric::{Metric, metric};

    #[test]
    fn roundtrip_preserves_repeated_pushforward_and_pullback() {
        let matrix = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex::new(-0.3, 0.03),
                Complex::new(0.02, 0.01),
                Complex::new(0.04, -0.02),
                Complex::new(-0.2, -0.01),
            ],
        );
        let tangent = matrix.map(|z| Complex::new(z.im, z.re));
        for kind in [
            Metric::CircularDichroism,
            Metric::DualityBreaking,
            Metric::Chirality,
        ] {
            let (_, residual) = metric(&matrix, &[0, 1], [1.0, 1.3], kind).unwrap();
            let bytes = residual.save_state().unwrap();
            assert_eq!(bytes.len(), MetricResidual::state_size(2).unwrap());
            let restored = MetricResidual::from_state(&bytes).unwrap();
            assert_eq!(restored.shape(), (2, 2));
            assert_eq!(restored.save_state().unwrap(), bytes);
            for weight in [1.2, 0.0, -0.7, 1.2] {
                let expected = residual.pullback(weight).unwrap();
                let actual = restored.pullback(weight).unwrap();
                assert_eq!(actual.matrix, expected.matrix);
                assert_eq!(actual.ks, expected.ks);
                assert_eq!(
                    restored.pushforward(&tangent, [0.2, -0.1]).unwrap(),
                    residual.pushforward(&tangent, [0.2, -0.1]).unwrap()
                );
            }
        }
    }

    #[test]
    fn roundtrip_preserves_deferred_errors_and_zero_directions() {
        for reason in GRADIENT_ERRORS {
            let message = reason.to_string();
            let residual = MetricResidual {
                gradient: Err(Error::Derivative(reason)),
                ks_gradient: [0.0; 2],
                dimension: 2,
            };
            let bytes = residual.save_state().unwrap();
            assert!(bytes[25..].iter().all(|&byte| byte == 0));
            let restored = MetricResidual::from_state(&bytes).unwrap();
            assert_eq!(restored.save_state().unwrap(), bytes);
            for _ in 0..2 {
                assert_eq!(restored.pullback(1.0).unwrap_err().to_string(), message);
                assert_eq!(
                    restored
                        .pushforward(&DMatrix::identity(2, 2), [0.0; 2])
                        .unwrap_err()
                        .to_string(),
                    message
                );
                assert_eq!(restored.pullback(0.0).unwrap().matrix, DMatrix::zeros(2, 2));
                assert_eq!(
                    restored
                        .pushforward(&DMatrix::zeros(2, 2), [0.0; 2])
                        .unwrap(),
                    0.0
                );
            }
        }
    }

    #[test]
    fn metric_errors_survive_roundtrip() {
        for (entries, expected_value, message) in [
            ([0.5, 0.0, 0.0, 0.5], 0.0, "zero contrast"),
            ([0.0, 0.0, 0.0, 0.5], 1.0, "below numerical resolution"),
            (
                [0.3, 0.0, 0.1, 0.2],
                (1.0_f64 / 7.0).sqrt(),
                "below numerical resolution",
            ),
        ] {
            let matrix = DMatrix::from_row_slice(2, 2, &entries.map(|x| Complex::new(x, 0.0)));
            let (value, residual) = metric(&matrix, &[0, 1], [1.0; 2], Metric::Chirality).unwrap();
            assert!((value - expected_value).abs() < 1e-15);
            let expected = residual.pullback(1.0).unwrap_err().to_string();
            assert!(expected.contains(message), "{expected}");
            let bytes = residual.save_state().unwrap();
            let restored = MetricResidual::from_state(&bytes).unwrap();
            assert_eq!(restored.pullback(1.0).unwrap_err().to_string(), expected);
            assert_eq!(
                restored
                    .pushforward(&matrix, [0.0; 2])
                    .unwrap_err()
                    .to_string(),
                expected
            );
            assert_eq!(restored.pullback(0.0).unwrap().matrix, DMatrix::zeros(2, 2));
            assert_eq!(
                restored
                    .pushforward(&DMatrix::zeros(2, 2), [0.0; 2])
                    .unwrap(),
                0.0
            );
            assert_eq!(restored.save_state().unwrap(), bytes);
        }
    }

    #[test]
    fn rejects_invalid_shape_status_and_length_before_allocation() {
        let (_, residual) = metric(
            &DMatrix::identity(2, 2),
            &[0, 1],
            [1.0; 2],
            Metric::Chirality,
        )
        .unwrap();
        let bytes = residual.save_state().unwrap();
        for len in 0..bytes.len() {
            assert!(MetricResidual::from_state(&bytes[..len]).is_err());
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(MetricResidual::from_state(&trailing).is_err());
        for dimension in [0_u64, u64::MAX, u64::MAX / 16] {
            let mut invalid = bytes.clone();
            invalid[..8].copy_from_slice(&dimension.to_le_bytes());
            assert!(MetricResidual::from_state(&invalid).is_err());
        }
        let mut invalid = bytes.clone();
        invalid[8] = 255;
        assert!(MetricResidual::from_state(&invalid).is_err());
        let mut invalid = bytes;
        invalid[25..33].copy_from_slice(&1.0_f64.to_le_bytes());
        assert!(MetricResidual::from_state(&invalid).is_err());
        assert!(MetricResidual::state_size(usize::MAX).is_err());
    }
}
