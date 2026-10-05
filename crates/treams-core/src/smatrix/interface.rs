//! Chiral Fresnel and Cartesian planar interfaces, and propagation of up/down
//! reference planes.
//!
//! Upstream: `treams.coeffs.fresnel`, `treams.SMatrices.interface` and
//! `treams.SMatrices.propagation`.

mod saved;

use nalgebra::DMatrix;

use super::{Blocks, checked_dimension, dimension, identity_blocks};
use crate::{
    Complex, Error, Result,
    numerics::{Jet, finite},
};

/// The chiral Fresnel coefficients `res[out][in][out pol][in pol]`, transcribed from
/// upstream `treams.coeffs.fresnel` with its variable names.
///
/// `out` and `in` index the propagation direction (0 = up, 1 = down), so `res[0][0]`
/// transmits upward from the negative medium. The inputs index the medium first,
/// negative side (below) then positive side (above), and the polarization second.
fn fresnel_values<const N: usize>(
    ks: [[Jet<N>; 2]; 2],
    kzs: [[Jet<N>; 2]; 2],
    zs: [Jet<N>; 2],
) -> [[[[Jet<N>; 2]; 2]; 2]; 2] {
    let ap = std::array::from_fn::<_, 2, _>(|i| ks[i][0] * kzs[i][1] + ks[i][1] * kzs[i][0]);
    let am = std::array::from_fn::<_, 2, _>(|i| ks[i][0] * kzs[i][1] - ks[i][1] * kzs[i][0]);
    let bp = ks[0][1] * kzs[1][1] + ks[1][1] * kzs[0][1];
    let cp = ks[0][0] * kzs[1][0] + ks[1][0] * kzs[0][0];
    let zs_diff = zs[0] - zs[1];
    let zs_prod = 4.0 * zs[0] * zs[1];
    let pref = 1.0 / (zs_diff * zs_diff * ap[0] * ap[1] + zs_prod * bp * cp);
    let mut res = [[[[Jet::default(); 2]; 2]; 2]; 2];
    for i in 0..2 {
        let j = 1 - i;
        res[i][i][0][0] = (zs[0] + zs[1]) * ks[j][0] * kzs[i][0] * bp * zs[j] * 4.0 * pref;
        res[i][i][0][1] = -(zs[i] - zs[j])
            * ks[j][0]
            * kzs[i][1]
            * (ks[j][1] * kzs[i][0] - ks[i][0] * kzs[j][1])
            * zs[j]
            * 4.0
            * pref;
        res[i][i][1][0] = -(zs[i] - zs[j])
            * ks[j][1]
            * kzs[i][0]
            * (ks[j][0] * kzs[i][1] - ks[i][1] * kzs[j][0])
            * zs[j]
            * 4.0
            * pref;
        res[i][i][1][1] = (zs[0] + zs[1]) * ks[j][1] * kzs[i][1] * cp * zs[j] * 4.0 * pref;
        res[j][i][0][0] = (-zs_diff * zs_diff * am[i] * ap[j]
            - zs_prod * bp * (ks[i][0] * kzs[j][0] - ks[j][0] * kzs[i][0]))
            * pref;
        res[j][i][0][1] =
            -2.0 * (zs[j] * zs[j] - zs[i] * zs[i]) * ks[i][0] * kzs[i][1] * ap[j] * pref;
        res[j][i][1][0] =
            -2.0 * (zs[j] * zs[j] - zs[i] * zs[i]) * ks[i][1] * kzs[i][0] * ap[j] * pref;
        res[j][i][1][1] = (zs_diff * zs_diff * am[i] * ap[j]
            - zs_prod * cp * (ks[i][1] * kzs[j][1] - ks[j][1] * kzs[i][1]))
            * pref;
    }
    res
}

/// What [`fresnel`] saves for its pullback: the wavenumbers, the normal wavenumbers and
/// the impedances. The pullback differentiates the closed forms again instead of keeping
/// a Jacobian.
#[derive(Debug)]
pub struct FresnelResidual {
    ks: [[Complex; 2]; 2],
    kzs: [[Complex; 2]; 2],
    zs: [Complex; 2],
}

/// Planar interface from full and normal wavenumbers and impedances, each indexed by
/// medium (negative side, then positive side) and then by polarization.
///
/// Returns four helicity scattering blocks with polarization order (0, 1).
///
/// Upstream: `treams.coeffs.fresnel(ks, kzs, zs)`.
pub fn fresnel(
    ks: [[Complex; 2]; 2],
    kzs: [[Complex; 2]; 2],
    zs: [Complex; 2],
) -> Result<(Blocks, FresnelResidual)> {
    if ks
        .iter()
        .flatten()
        .chain(kzs.iter().flatten())
        .chain(zs.iter())
        .any(|&v| !finite(v))
    {
        return Err(Error::InvalidInput("Fresnel inputs must be finite".into()));
    }
    if ks[0] == ks[1]
        && kzs[0] == kzs[1]
        && zs[0] == zs[1]
        && zs[0] != Complex::default()
        && ks[0].iter().all(|&k| k != Complex::default())
        && kzs[0].contains(&Complex::default())
    {
        // Identical media have no boundary. This is the exact continuation of
        // transmission through a grazing channel, whose Fresnel formula is 0/0.
        return Ok((identity_blocks(2), FresnelResidual { ks, kzs, zs }));
    }
    let values = fresnel_values::<0>(
        ks.map(|r| r.map(Jet::constant)),
        kzs.map(|r| r.map(Jet::constant)),
        zs.map(Jet::constant),
    );
    let value =
        std::array::from_fn(|b| DMatrix::from_fn(2, 2, |i, j| values[b / 2][b % 2][i][j].value));
    dimension(&value).map_err(|_| Error::Singular)?;
    Ok((value, FresnelResidual { ks, kzs, zs }))
}

/// Cotangents of the inputs of [`fresnel`], in the order of its arguments; each array
/// lists the medium on the negative side (below), then the positive side (above).
#[derive(Clone, Copy, Debug)]
pub struct FresnelGradient {
    /// Cotangents of the full wavenumbers `ks` of polarizations 0 and 1.
    pub ks: [[Complex; 2]; 2],
    /// Cotangents of the normal wavenumbers `kzs` of polarizations 0 and 1.
    pub kz: [[Complex; 2]; 2],
    /// Cotangents of the impedances `zs`.
    pub z: [Complex; 2],
}

impl FresnelResidual {
    /// Propagate one joint wavenumber, normal-wavenumber and impedance direction.
    pub fn pushforward(
        &self,
        ks: [[Complex; 2]; 2],
        kz: [[Complex; 2]; 2],
        z: [Complex; 2],
    ) -> Result<Blocks> {
        if ks
            .iter()
            .flatten()
            .chain(kz.iter().flatten())
            .chain(&z)
            .any(|&v| !finite(v))
        {
            return Err(Error::InvalidInput(
                "Fresnel tangents must be finite".into(),
            ));
        }
        let values = fresnel_values(
            std::array::from_fn(|i| {
                std::array::from_fn(|j| Jet {
                    value: self.ks[i][j],
                    derivative: [ks[i][j]],
                })
            }),
            std::array::from_fn(|i| {
                std::array::from_fn(|j| Jet {
                    value: self.kzs[i][j],
                    derivative: [kz[i][j]],
                })
            }),
            std::array::from_fn(|i| Jet {
                value: self.zs[i],
                derivative: [z[i]],
            }),
        );
        if values
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .any(|v| !v.finite())
        {
            return Err(Error::InvalidInput(
                "Fresnel derivative is undefined for this grazing-channel limit".into(),
            ));
        }
        Ok(std::array::from_fn(|b| {
            DMatrix::from_fn(2, 2, |i, j| values[b / 2][b % 2][i][j].derivative[0])
        }))
    }

    /// The gradients of all ten complex inputs under the real pairing `Re Σ conj(g)·dx`.
    pub fn pullback(&self, cotangent: &Blocks) -> Result<FresnelGradient> {
        // Jet seed offsets: `ks[i][j]` at `KS + 2 i + j`, `kzs[i][j]` at `KZS + 2 i + j`
        // and `zs[i]` at `ZS + i`.
        const KS: usize = 0;
        const KZS: usize = 4;
        const ZS: usize = 8;
        checked_dimension(cotangent, 2, "Fresnel cotangent blocks must be 2 by 2")?;
        let ks = std::array::from_fn(|i| {
            std::array::from_fn(|j| Jet::<10>::variable(self.ks[i][j], KS + 2 * i + j))
        });
        let kzs = std::array::from_fn(|i| {
            std::array::from_fn(|j| Jet::variable(self.kzs[i][j], KZS + 2 * i + j))
        });
        let zs = std::array::from_fn(|i| Jet::variable(self.zs[i], ZS + i));
        let values = fresnel_values(ks, kzs, zs);
        if values
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .any(|v| !v.finite())
        {
            return Err(Error::InvalidInput(
                "Fresnel derivative is undefined for this grazing-channel limit".into(),
            ));
        }
        let mut result = [Complex::default(); 10];
        for (b, block) in cotangent.iter().enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    for (gradient, d) in
                        result.iter_mut().zip(values[b / 2][b % 2][i][j].derivative)
                    {
                        *gradient += block[(i, j)] * d.conj();
                    }
                }
            }
        }
        Ok(FresnelGradient {
            ks: std::array::from_fn(|i| std::array::from_fn(|j| result[KS + 2 * i + j])),
            kz: std::array::from_fn(|i| std::array::from_fn(|j| result[KZS + 2 * i + j])),
            z: [result[ZS], result[ZS + 1]],
        })
    }
}

/// The four 2-by-2 scattering blocks of one interface as a 4-by-4 matrix.
type InterfaceMatrix = nalgebra::SMatrix<Complex, 4, 4>;

/// One 4-by-4 matrix (`lhs` or `rhs`) of the tangential-field matching, as jets.
type BoundaryJets<const N: usize> = [[Jet<N>; 4]; 4];

/// The normal wavenumber `sqrt(k^2 - q.q)` on the branch with a nonnegative imaginary
/// part.
fn normal_component<const N: usize>(k: Jet<N>, q: [Jet<N>; 2]) -> Result<Jet<N>> {
    let mut normal = (k * k - q[0] * q[0] - q[1] * q[1]).sqrt();
    // A grazing wave still has finite tangential fields. Only the derivative
    // of its square-root dispersion is singular, not the forward interface.
    if N > 0 && normal.value == Complex::default() {
        return Err(Error::InvalidInput(
            "interface derivative is undefined at an exact diffraction threshold".into(),
        ));
    }
    // The principal root has a nonnegative real part; take the outgoing branch.
    if normal.value.im < 0.0 {
        normal = -normal;
    }
    Ok(normal)
}

/// Tangential fields `[E_a, E_b, H_a, H_b]` of the plane waves in one medium, indexed
/// by propagation direction (0 = up) and polarization; `a` and `b` are the two axes
/// after `axis` in cyclic order.
pub(super) fn port_boundary<const N: usize>(
    ks: [Jet<N>; 2],
    z: Jet<N>,
    q: [Jet<N>; 2],
    axis: usize,
) -> Result<[[[Jet<N>; 4]; 2]; 2]> {
    let mut waves = [[[Jet::default(); 4]; 2]; 2];
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    for (pol, &wave_number) in ks.iter().enumerate() {
        let normal = normal_component(wave_number, q)?;
        for (direction, polarizations) in waves.iter_mut().enumerate() {
            let mut vector = [Jet::default(); 3];
            vector[a] = q[0];
            vector[b] = q[1];
            vector[axis] = if direction == 0 { normal } else { -normal };
            let e = crate::pw::polarization_from_inputs(vector, u8::from(pol != 0))?;
            let impedance = -Complex::i() * (if pol == 0 { -1.0 } else { 1.0 }) / z;
            polarizations[pol] = [e[a], e[b], impedance * e[a], impedance * e[b]];
        }
    }
    Ok(waves)
}

/// The tangential-field matching `lhs s = rhs` of an interface. The columns of `lhs`
/// are the outgoing waves (up in the positive medium, down in the negative one), those
/// of `rhs` the incoming waves (up in the negative medium, down in the positive one).
fn interface_boundary<const N: usize>(
    ks: [[Jet<N>; 2]; 2],
    zs: [Jet<N>; 2],
    q: [Jet<N>; 2],
    axis: usize,
) -> Result<(BoundaryJets<N>, BoundaryJets<N>)> {
    let waves = [
        port_boundary(ks[0], zs[0], q, axis)?,
        port_boundary(ks[1], zs[1], q, axis)?,
    ];
    let lhs = std::array::from_fn(|row| {
        std::array::from_fn(|col| {
            if col < 2 {
                waves[1][0][col][row]
            } else {
                -waves[0][1][col - 2][row]
            }
        })
    });
    let rhs = std::array::from_fn(|row| {
        std::array::from_fn(|col| {
            if col < 2 {
                waves[0][0][col][row]
            } else {
                -waves[1][1][col - 2][row]
            }
        })
    });
    Ok((lhs, rhs))
}

/// What [`interface()`] saves for its pullback: the media, the transverse wavevector,
/// the axis, the inverse of the tangential-field matching matrix and the blocks.
#[derive(Debug)]
pub struct InterfaceResidual {
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: [f64; 2],
    axis: usize,
    inverse: Option<InterfaceMatrix>,
    /// The scattering blocks as one matrix over (up, down) outputs and inputs.
    result: InterfaceMatrix,
    fixed_q: bool,
}

/// Blocks `[[b0, b1], [b2, b3]]` of 2-by-2 polarization blocks as one 4-by-4 matrix.
fn interface_matrix(blocks: &Blocks) -> InterfaceMatrix {
    InterfaceMatrix::from_fn(|i, j| blocks[2 * (i / 2) + j / 2][(i % 2, j % 2)])
}

/// The four 2-by-2 polarization blocks of a 4-by-4 interface matrix.
fn interface_blocks(matrix: &InterfaceMatrix) -> Blocks {
    std::array::from_fn(|block| {
        DMatrix::from_fn(2, 2, |i, j| {
            matrix[(2 * (block / 2) + i, 2 * (block % 2) + j)]
        })
    })
}

/// Interface from wavenumbers and impedances, indexed by medium (negative side, then
/// positive side), and the real transverse wavevector `q`.
///
/// `axis` is the Cartesian normal, and `q` holds the two components after it in cyclic
/// order. Returns four helicity scattering blocks with polarization order (0, 1). With
/// `fixed_q`, the pullback holds `q` fixed and returns a zero `q` gradient.
///
/// Upstream: `treams.SMatrices.interface`.
pub fn interface(
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: [f64; 2],
    axis: usize,
    fixed_q: bool,
) -> Result<(Blocks, InterfaceResidual)> {
    if axis > 2
        || q.iter().any(|x| !x.is_finite())
        || ks
            .iter()
            .flatten()
            .chain(&zs)
            .any(|&v| !finite(v) || v == Complex::default())
    {
        return Err(Error::InvalidInput("interface requires finite transverse components, nonzero finite wavenumbers/impedances, and a Cartesian normal".into()));
    }
    let (lhs, rhs) = interface_boundary::<0>(
        ks.map(|k| k.map(Jet::constant)),
        zs.map(Jet::constant),
        q.map(Jet::constant),
        axis,
    )?;
    let matrix = InterfaceMatrix::from_fn(|i, j| lhs[i][j].value);
    let identical = ks[0] == ks[1] && zs[0] == zs[1];
    let matched_grazing = identical
        && ks[0]
            .iter()
            .any(|&k| k * k - q[0] * q[0] - q[1] * q[1] == Complex::default());
    let inverse = if matched_grazing {
        None
    } else {
        Some(matrix.lu().try_inverse().ok_or(Error::Singular)?)
    };
    // The same medium on both sides is transparent for every transverse
    // wavevector. Tangential matching alone is ill-conditioned near grazing
    // incidence and cannot distinguish the ports at it, so use the exact value;
    // the saved inverse still differentiates it away from the threshold.
    let result = match inverse {
        Some(inverse) if !identical => inverse * InterfaceMatrix::from_fn(|i, j| rhs[i][j].value),
        _ => InterfaceMatrix::identity(),
    };
    let value = interface_blocks(&result);
    dimension(&value).map_err(|_| Error::Singular)?;
    Ok((
        value,
        InterfaceResidual {
            ks,
            zs,
            q,
            axis,
            inverse,
            result,
            fixed_q,
        },
    ))
}

/// Cotangents of the inputs of [`interface`], in the order of its arguments; the media
/// are listed negative side (below), then positive side (above).
#[derive(Clone, Copy, Debug)]
pub struct InterfaceGradient {
    /// Cotangents of the wavenumbers of polarizations 0 and 1 in each medium.
    pub ks: [[Complex; 2]; 2],
    /// Cotangents of the impedances `zs` of the media.
    pub z: [Complex; 2],
    /// Cotangents of the real transverse wavevector components; zero when the forward
    /// holds `q` fixed.
    pub q: [f64; 2],
}

impl InterfaceResidual {
    /// Differentiate the tangential-field solve using its saved inverse and one
    /// direction of local field derivatives.
    pub fn pushforward(
        &self,
        ks: [[Complex; 2]; 2],
        z: [Complex; 2],
        q: [f64; 2],
    ) -> Result<Blocks> {
        if ks.iter().flatten().chain(&z).any(|&v| !finite(v)) || q.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(
                "interface tangents must be finite".into(),
            ));
        }
        let inverse = self.inverse.ok_or_else(|| {
            Error::InvalidInput(
                "interface derivative is undefined at an exact diffraction threshold".into(),
            )
        })?;
        let fixed_q = self.fixed_q || (self.axis == 2 && self.q.iter().all(|&v| v == 0.0));
        let (lhs, rhs) = interface_boundary(
            std::array::from_fn(|i| {
                std::array::from_fn(|j| Jet {
                    value: self.ks[i][j],
                    derivative: [ks[i][j]],
                })
            }),
            std::array::from_fn(|i| Jet {
                value: self.zs[i],
                derivative: [z[i]],
            }),
            std::array::from_fn(|i| Jet {
                value: self.q[i].into(),
                derivative: [if fixed_q {
                    Complex::default()
                } else {
                    q[i].into()
                }],
            }),
            self.axis,
        )?;
        let lhs = InterfaceMatrix::from_fn(|i, j| lhs[i][j].derivative[0]);
        let rhs = InterfaceMatrix::from_fn(|i, j| rhs[i][j].derivative[0]);
        let tangent = interface_blocks(&(inverse * (rhs - lhs * self.result)));
        dimension(&tangent)?;
        Ok(tangent)
    }

    /// Reuse the 4-by-4 inverse for the implicit solve adjoint; recompute local field derivatives.
    pub fn pullback(&self, cotangent: &Blocks) -> Result<InterfaceGradient> {
        // Jet seed offsets: `ks[i][j]` at `KS + 2 i + j`, `zs[i]` at `ZS + i` and `q[i]`
        // at `Q + i`.
        const KS: usize = 0;
        const ZS: usize = 4;
        const Q: usize = 6;
        checked_dimension(cotangent, 2, "interface cotangent blocks must be 2 by 2")?;
        let cotangent = interface_matrix(cotangent);
        let inverse = self.inverse.ok_or_else(|| {
            Error::InvalidInput(
                "interface derivative is undefined at an exact diffraction threshold".into(),
            )
        })?;
        let adjoint = inverse.adjoint() * cotangent;
        let operator = -adjoint * self.result.adjoint();
        let ks = std::array::from_fn(|i| {
            std::array::from_fn(|j| Jet::<8>::variable(self.ks[i][j], KS + 2 * i + j))
        });
        let zs = std::array::from_fn(|i| Jet::variable(self.zs[i], ZS + i));
        // At normal incidence the xy-interface blocks depend on q only to second
        // order. Use their zero first derivative instead of an undefined
        // azimuth derivative in the intermediate polarization vectors.
        let fixed_q = self.fixed_q || (self.axis == 2 && self.q.iter().all(|&q| q == 0.0));
        let q = std::array::from_fn(|i| {
            if fixed_q {
                Jet::constant(self.q[i])
            } else {
                Jet::variable(self.q[i], Q + i)
            }
        });
        let (lhs, rhs) = interface_boundary(ks, zs, q, self.axis)?;
        let mut result = [Complex::default(); 8];
        for i in 0..4 {
            for j in 0..4 {
                for (a, g) in result.iter_mut().enumerate() {
                    *g += operator[(i, j)] * lhs[i][j].derivative[a].conj()
                        + adjoint[(i, j)] * rhs[i][j].derivative[a].conj();
                }
            }
        }
        Ok(InterfaceGradient {
            ks: std::array::from_fn(|i| std::array::from_fn(|j| result[KS + 2 * i + j])),
            z: [result[ZS], result[ZS + 1]],
            q: [result[Q].re, result[Q + 1].re],
        })
    }
}

/// What [`propagation`] saves for its pullback: the wavevectors, the displacement and
/// the diagonal phases.
#[derive(Debug)]
pub struct PropagationResidual {
    vectors: Vec<[Complex; 3]>,
    distance: [f64; 3],
    /// Upward and downward transmission phase of every mode.
    phases: Vec<[Complex; 2]>,
}

/// Propagate by a Cartesian displacement, given upgoing wavevectors for every mode.
///
/// The blocks are diagonal transmission phases and zero reflections.
///
/// Upstream: `treams.SMatrices.propagation`.
pub fn propagation(
    vectors: Vec<[Complex; 3]>,
    distance: [f64; 3],
) -> Result<(Blocks, PropagationResidual)> {
    if vectors.is_empty()
        || vectors.iter().flatten().any(|&v| !finite(v))
        || distance.iter().any(|r| !r.is_finite())
    {
        return Err(Error::InvalidInput(
            "propagation requires finite nonempty wavevectors and displacement".into(),
        ));
    }
    let phases = vectors
        .iter()
        .map(|k| {
            [1.0, -1.0].map(|sign| {
                (Complex::i()
                    * (sign * (k[0] * distance[0] + k[1] * distance[1]) + k[2] * distance[2]))
                    .exp()
            })
        })
        .collect::<Vec<_>>();
    // Evanescent waves displaced against their decay overflow.
    if phases.iter().flatten().any(|&phase| !finite(phase)) {
        return Err(Error::NonFinite("non-finite propagation phase".into()));
    }
    let n = vectors.len();
    let mut value: Blocks = std::array::from_fn(|_| DMatrix::zeros(n, n));
    for (i, phase) in phases.iter().enumerate() {
        value[0][(i, i)] = phase[0];
        value[3][(i, i)] = phase[1];
    }
    Ok((
        value,
        PropagationResidual {
            vectors,
            distance,
            phases,
        },
    ))
}

/// Cotangents of the inputs of [`propagation`], in the order of its arguments.
#[derive(Clone, Debug)]
pub struct PropagationGradient {
    /// Complex wavevector cotangents, one per mode.
    pub vectors: Vec<[Complex; 3]>,
    /// Real displacement cotangent.
    pub distance: [f64; 3],
}

impl PropagationResidual {
    /// The shape `(n, n)` of each scattering block, with `n` modes in each direction.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.vectors.len(), self.vectors.len())
    }

    /// Propagate a wavevector and displacement direction through the saved phases.
    pub fn pushforward(&self, vectors: &[[Complex; 3]], distance: [f64; 3]) -> Result<Blocks> {
        if vectors.len() != self.vectors.len()
            || vectors.iter().flatten().any(|&v| !finite(v))
            || distance.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput(
                "propagation tangents must be finite and match inputs".into(),
            ));
        }
        let n = self.vectors.len();
        let mut tangent: Blocks = std::array::from_fn(|_| DMatrix::zeros(n, n));
        for (i, ((k, dk), phase)) in self
            .vectors
            .iter()
            .zip(vectors)
            .zip(&self.phases)
            .enumerate()
        {
            let dot: [Complex; 3] =
                std::array::from_fn(|a| dk[a] * self.distance[a] + k[a] * distance[a]);
            for ((block, sign), phase) in [(0, 1.0), (3, -1.0)].into_iter().zip(phase) {
                tangent[block][(i, i)] = Complex::i() * phase * (sign * (dot[0] + dot[1]) + dot[2]);
            }
        }
        dimension(&tangent)?;
        Ok(tangent)
    }

    /// Wavevector and displacement gradients from the cotangents of the four blocks.
    pub fn pullback(&self, cotangent: &Blocks) -> Result<PropagationGradient> {
        let n = checked_dimension(
            cotangent,
            self.vectors.len(),
            "propagation cotangent shape mismatch",
        )?;
        let mut vectors = vec![[Complex::default(); 3]; n];
        let mut distance = [0.0; 3];
        for (i, (k, phase)) in self.vectors.iter().zip(&self.phases).enumerate() {
            for ((block, sign), phase) in [(0, 1.0), (3, -1.0)].into_iter().zip(phase) {
                let weight = cotangent[block][(i, i)].conj() * Complex::i() * phase;
                for (a, s) in [sign, sign, 1.0].into_iter().enumerate() {
                    vectors[i][a] += (weight * s * self.distance[a]).conj();
                    distance[a] += (weight * s * k[a]).re;
                }
            }
        }
        Ok(PropagationGradient { vectors, distance })
    }
}
