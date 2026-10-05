//! Plane-port radiation of periodic multipole arrays.
//!
//! Upstream: `treams.SMatrices.from_array`.

mod saved;

use faer::MatRef;
use nalgebra::DMatrix;

use super::{Blocks, checked_dimension, dimension};
use crate::{
    Complex, Error, Result,
    linalg::{product, product_adjoint_right, product_views, view},
    numerics::finite,
};

/// The four multipole-by-plane-wave channel arrays of [`from_array`], in the order
/// incident up, incident down, emitted up, emitted down.
///
/// Each array has one row per multipole and one column per plane-wave mode. Unlike
/// [`Blocks`], the arrays need not be square.
pub type Channels = [DMatrix<Complex>; 4];

/// What [`from_array`] saves for its pullback: the response, the channel arrays and
/// the scattered fields.
#[derive(Debug)]
pub struct FromArrayResidual {
    response: DMatrix<Complex>,
    channels: Channels,
    scattered: [DMatrix<Complex>; 2],
}

/// Compose an effective response with incident/emitted, up/down channel arrays.
///
/// Returns four plane-wave scattering blocks, including direct transmission.
///
/// Upstream: `treams.SMatrices.from_array`, given the channels of
/// [`spherical_channels`](crate::channels::spherical_channels) or
/// [`cylindrical_channels`](crate::channels::cylindrical_channels).
pub fn from_array(
    response: DMatrix<Complex>,
    channels: Channels,
) -> Result<(Blocks, FromArrayResidual)> {
    let d = response.nrows();
    let c = channels[0].ncols();
    if d == 0
        || !response.is_square()
        || c == 0
        || response.iter().any(|&v| !finite(v))
        || channels
            .iter()
            .any(|a| a.shape() != (d, c) || a.iter().any(|&v| !finite(v)))
    {
        return Err(Error::InvalidInput(
            "require a finite square response and four matching multipole-by-plane channel arrays"
                .into(),
        ));
    }
    let scattered = std::array::from_fn(|direction| product(&response, &channels[direction]));
    let mut value = std::array::from_fn(|b| {
        product_views(
            view(&channels[2 + b / 2]).transpose(),
            view(&scattered[b % 2]),
        )
    });
    for block in [0, 3] {
        for i in 0..c {
            value[block][(i, i)] += 1.0;
        }
    }
    dimension(&value).map_err(|_| Error::NonFinite("non-finite array scattering matrix".into()))?;
    Ok((
        value,
        FromArrayResidual {
            response,
            channels,
            scattered,
        },
    ))
}

/// Cotangents of the inputs of [`from_array`], in the order of its arguments.
#[derive(Debug)]
pub struct FromArrayGradient {
    /// Cotangent of the effective multipole response.
    pub response: DMatrix<Complex>,
    /// Cotangents of the four channel arrays.
    pub channels: Channels,
}

impl FromArrayResidual {
    /// The shape `(n, n)` of each scattering block, with `n` plane-wave modes in each
    /// direction.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.channels[0].ncols(), self.channels[0].ncols())
    }

    /// Multipole and plane-wave mode counts of the channel inputs.
    #[must_use]
    pub fn input_shape(&self) -> (usize, usize) {
        self.channels[0].shape()
    }

    /// Apply response and channel tangents to the recorded scattered fields.
    pub fn pushforward(&self, response: &DMatrix<Complex>, channels: &Channels) -> Result<Blocks> {
        if response.shape() != self.response.shape()
            || response.iter().any(|&z| !finite(z))
            || channels
                .iter()
                .any(|a| a.shape() != self.input_shape() || a.iter().any(|&z| !finite(z)))
        {
            return Err(Error::InvalidInput(
                "invalid array scattering tangent".into(),
            ));
        }
        let scattered: [DMatrix<Complex>; 2] = std::array::from_fn(|direction| {
            product(response, &self.channels[direction])
                + product(&self.response, &channels[direction])
        });
        Ok(std::array::from_fn(|b| {
            product_views(
                view(&channels[2 + b / 2]).transpose(),
                view(&self.scattered[b % 2]),
            ) + product_views(
                view(&self.channels[2 + b / 2]).transpose(),
                view(&scattered[b % 2]),
            )
        }))
    }

    /// Effective-response and four-channel cotangents.
    pub fn pullback(&self, cotangent: &Blocks) -> Result<FromArrayGradient> {
        let c = checked_dimension(
            cotangent,
            self.channels[0].ncols(),
            "invalid array scattering cotangent shape",
        )?;
        // Blocks are C_{2+e}^T S_s with S_s = R C_s, so the emission cotangents
        // conjugate without transposing: bar S_s = conj(C_2) g_s + conj(C_3) g_{2+s}.
        let conjugate =
            |a: &DMatrix<Complex>, b: MatRef<'_, Complex>| product_views(view(a).conjugate(), b);
        let mut response = DMatrix::zeros(self.response.nrows(), self.response.ncols());
        let mut channels: Channels =
            std::array::from_fn(|_| DMatrix::zeros(self.response.nrows(), c));
        for direction in 0..2 {
            let adjoint = conjugate(&self.channels[2], view(&cotangent[direction]))
                + conjugate(&self.channels[3], view(&cotangent[2 + direction]));
            response += product_adjoint_right(&adjoint, &self.channels[direction]);
            channels[direction] = product_views(view(&self.response).adjoint(), view(&adjoint));
            channels[2 + direction] = conjugate(
                &self.scattered[0],
                view(&cotangent[2 * direction]).transpose(),
            ) + conjugate(
                &self.scattered[1],
                view(&cotangent[2 * direction + 1]).transpose(),
            );
        }
        Ok(FromArrayGradient { response, channels })
    }
}
