//! Elementwise evaluation over broadcast inputs, and its pullback. treams-rs extension.
//!
//! Each input is a scalar (length one) or has the common output length, as in `NumPy`
//! broadcasting of one-dimensional arrays. The pullback turns the cotangent of each
//! output, the gradient of a real loss with respect to that output, into the
//! cotangents of the inputs. A scalar input's cotangent is the sum of its per-output
//! cotangents, added in output order on every schedule, so the thread count never
//! changes it.

use rayon::prelude::*;

use super::parallel::Parallel;
use crate::{Complex, Error, Result};

/// Common output length of scalar-or-array inputs: zero when an input is empty,
/// otherwise the longest length. Any other length is rejected with `message`.
pub(crate) fn size(lengths: &[usize], message: &str) -> Result<usize> {
    let size = if lengths.contains(&0) {
        0
    } else {
        lengths.iter().copied().max().unwrap_or_default()
    };
    if lengths.iter().any(|&n| n != 1 && n != size) {
        return Err(Error::InvalidInput(message.into()));
    }
    Ok(size)
}

/// Element `i` of a scalar-or-array input of validated length.
#[allow(clippy::indexing_slicing)] // Scalar-or-element indexing after `size` validation.
pub(crate) fn element<T: Copy>(values: &[T], i: usize) -> T {
    values[if values.len() == 1 { 0 } else { i }]
}

/// Evaluate `f` at `0..size` in order, stopping at an error.
pub(crate) fn map<T: Send>(
    size: usize,
    parallel: Parallel,
    f: impl Fn(usize) -> Result<T> + Sync + Send,
) -> Result<Vec<T>> {
    match parallel.chunk(size) {
        Some(chunk) => (0..size)
            .into_par_iter()
            .with_min_len(chunk)
            .map(f)
            .collect(),
        None => (0..size).map(f).collect(),
    }
}

/// The cotangent of one output element.
pub(crate) trait Cotangent: Sync {
    /// Whether every component is finite.
    fn finite(&self) -> bool;
}

impl Cotangent for Complex {
    fn finite(&self) -> bool {
        crate::numerics::finite(*self)
    }
}

impl<const N: usize> Cotangent for [Complex; N] {
    fn finite(&self) -> bool {
        self.iter().all(|&z| crate::numerics::finite(z))
    }
}

/// Pull back an elementwise evaluation with `A` broadcast complex arguments of the
/// given `lengths`. The cotangent needs one finite entry per output (otherwise
/// `message` is returned); `f(i, g)` gives the argument cotangents of output `i`.
pub(crate) fn pullback<G: Cotangent, const A: usize>(
    cotangent: &[G],
    size: usize,
    message: &str,
    lengths: [usize; A],
    parallel: Parallel,
    f: impl Fn(usize, &G) -> Result<[Complex; A]> + Sync + Send,
) -> Result<[Vec<Complex>; A]> {
    if cotangent.len() != size || !cotangent.iter().all(Cotangent::finite) {
        return Err(Error::InvalidInput(message.into()));
    }
    let Some(chunk) = parallel.chunk(size) else {
        // In order on this thread: gather each output's argument cotangents
        // straight into the gradients, without per-output intermediates. A
        // scalar argument's sum starts from zero and adds in output order, as
        // `Iterator::sum` does below.
        let mut sums = [Complex::default(); A];
        let mut gradients =
            lengths.map(|length| Vec::with_capacity(if length == 1 { 1 } else { size }));
        for (i, g) in cotangent.iter().enumerate() {
            let values = f(i, g)?;
            for (((gradient, sum), &length), value) in gradients
                .iter_mut()
                .zip(&mut sums)
                .zip(&lengths)
                .zip(values)
            {
                if length == 1 {
                    *sum += value;
                } else {
                    gradient.push(value);
                }
            }
        }
        for ((gradient, sum), &length) in gradients.iter_mut().zip(sums).zip(&lengths) {
            if length == 1 {
                gradient.push(sum);
            }
        }
        return Ok(gradients);
    };
    // In parallel: write each output's argument cotangents into its row in
    // place, which indexed iterators split without collecting partial vectors.
    let mut rows = vec![[Complex::default(); A]; size];
    rows.par_iter_mut()
        .zip(cotangent)
        .enumerate()
        .with_min_len(chunk)
        .try_for_each(|(i, (row, g))| {
            *row = f(i, g)?;
            Ok(())
        })?;
    if A == 1 && lengths != [1; A] {
        // The rows of a single array argument are its gradient already.
        let mut gradient = Some(rows.into_flattened());
        return Ok(std::array::from_fn(|_| gradient.take().unwrap_or_default()));
    }
    // A scalar argument's terms add in output order, as on the calling thread.
    Ok(std::array::from_fn(|a| {
        let gradient = rows
            .iter()
            .map(|row| row.get(a).copied().unwrap_or_default());
        if lengths.get(a) == Some(&1) {
            vec![gradient.sum()]
        } else {
            gradient.collect()
        }
    }))
}

/// Broadcast lengths, output order and the pullback on every schedule.
#[cfg(test)]
mod tests {
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{Parallel, element, map, pullback, size};
    use crate::{Complex, test_support::DEFAULT_CASES};

    #[test]
    fn size_accepts_scalars_and_one_common_length() {
        let message = "arrays must have equal lengths or scalar inputs";
        for (lengths, expected) in [
            (&[1, 1][..], Some(1)),
            (&[1, 4, 4], Some(4)),
            (&[4, 1, 1], Some(4)),
            (&[0, 1], Some(0)),
            (&[0], Some(0)),
            (&[], Some(0)),
            (&[0, 3], None),
            (&[2, 3], None),
            (&[2, 1, 3], None),
        ] {
            match size(lengths, message) {
                Ok(size) => assert_eq!(Some(size), expected, "{lengths:?}"),
                Err(error) => {
                    assert_eq!(expected, None, "{lengths:?}");
                    assert_eq!(error.to_string(), message);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        #[test]
        fn pullback_is_the_transposed_elementwise_product(
            size in 0_usize..40,
            scalar in any::<[bool; 2]>(),
            threshold in 0_usize..48,
            chunked in any::<bool>(),
        ) {
            check_pullback(size, scalar, threshold, chunked)?;
        }
    }

    /// For `f_i(a, b) = a_i b_i` over broadcast arguments, the evaluation keeps output
    /// order under every schedule, and the pullback pairs each cotangent with the
    /// other argument and sums it for scalar arguments, bit for bit alike in order
    /// on the calling thread and in parallel.
    fn check_pullback(
        size: usize,
        scalar: [bool; 2],
        threshold: usize,
        chunked: bool,
    ) -> Result<(), TestCaseError> {
        let parallel = if chunked {
            Parallel::Chunked(threshold)
        } else {
            Parallel::AtLeast(threshold)
        };
        let argument = |index: usize, scalar: bool| -> Vec<Complex> {
            let length = if scalar { 1 } else { size };
            (0..length)
                .map(|i| Complex::new(f64::from(u32::try_from(i + index).unwrap()), 0.5))
                .collect()
        };
        let arguments = [argument(1, scalar[0]), argument(3, scalar[1])];
        let lengths = arguments.each_ref().map(Vec::len);
        let values = map(size, parallel, |i| {
            Ok(element(&arguments[0], i) * element(&arguments[1], i))
        })?;
        prop_assert_eq!(values.len(), size);
        for (i, value) in values.iter().enumerate() {
            prop_assert_eq!(
                *value,
                element(&arguments[0], i) * element(&arguments[1], i)
            );
        }
        let cotangent: Vec<_> = (0..size)
            .map(|i| Complex::new(0.25, f64::from(u32::try_from(i).unwrap())))
            .collect();
        let gradients = pullback(&cotangent, size, "cotangent", lengths, parallel, |i, &g| {
            let [a, b] = arguments.each_ref().map(|v| element(v, i));
            Ok([g * b.conj(), g * a.conj()])
        })?;
        for (argument, gradient) in arguments.iter().zip(&gradients) {
            prop_assert_eq!(gradient.len(), argument.len());
        }
        for (index, gradient) in gradients.iter().enumerate() {
            let other = &arguments[1 - index];
            let expected: Vec<_> = (0..size)
                .map(|i| cotangent[i] * element(other, i).conj())
                .collect();
            let expected = if lengths[index] == 1 {
                vec![expected.iter().sum()]
            } else {
                expected
            };
            // Every schedule adds a scalar argument's terms in output order.
            prop_assert_eq!(gradient, &expected);
        }
        // A single argument takes its own path when it is an array.
        let [single] = pullback(
            &cotangent,
            size,
            "cotangent",
            [lengths[0]],
            parallel,
            |i, &g| Ok([g * element(&arguments[1], i).conj()]),
        )?;
        prop_assert_eq!(&single, &gradients[0]);
        Ok(())
    }

    #[test]
    fn pullback_rejects_mismatched_or_nonfinite_cotangents() {
        let zero = |_: usize, _: &[Complex; 2]| Ok([Complex::default(); 1]);
        let finite = [[Complex::new(0.5, 0.0); 2]; 2];
        let parallel = Parallel::AtLeast(1024);
        assert!(pullback(&finite, 2, "cotangent", [2], parallel, zero).is_ok());
        assert!(pullback(&finite, 3, "cotangent", [3], parallel, zero).is_err());
        assert!(pullback(&finite[..1], 2, "cotangent", [2], parallel, zero).is_err());
        let mut nonfinite = finite;
        nonfinite[1][0].im = f64::NAN;
        let error = pullback(&nonfinite, 2, "cotangent", [2], parallel, zero).unwrap_err();
        assert_eq!(error.to_string(), "cotangent");
    }
}
