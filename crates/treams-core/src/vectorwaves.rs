//! Low-level vector waves in spherical, cylindrical and Cartesian component frames.
#![allow(clippy::indexing_slicing)] // Fixed wave argument and component arrays.

use crate::{
    Complex, Error, Result, finite,
    jet::Jet,
    special::{self, Radial},
    waves::Mode,
};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Low-level wave family. Polarization 0/1 selects M/N, or negative/positive
/// helicity when `helicity` is true. Harmonics ignore this polarization selection.
#[derive(Clone, Copy, Debug)]
pub enum Family {
    /// Tangential X harmonic in spherical components.
    HarmonicX,
    /// Tangential Y harmonic in spherical components.
    HarmonicY,
    /// Radial Z harmonic, equal to i times the scalar spherical harmonic.
    HarmonicZ,
    /// Spherical vector wave at (kr, theta, phi).
    Spherical(Radial),
    /// Cylindrical vector wave at `(kz, k_rho*rho, phi, z, k)`.
    Cylindrical(Radial),
    /// Plane vector wave at (kx, ky, kz, x, y, z).
    Plane,
}

fn select<const N: usize>(m: [Jet<N>; 3], n: [Jet<N>; 3], pol: u8, helicity: bool) -> [Jet<N>; 3] {
    if helicity {
        std::array::from_fn(|i| (n[i] + (2.0 * f64::from(pol) - 1.0) * m[i]) * FRAC_1_SQRT_2)
    } else if pol == 0 {
        m
    } else {
        n
    }
}

fn angular<const N: usize>(
    l: i32,
    m: i32,
    theta: Jet<N>,
    phi: Jet<N>,
    tangential: bool,
) -> [Jet<N>; 3] {
    let cosine = theta.map(theta.value.cos(), -theta.value.sin());
    let mut sine = theta.map(theta.value.sin(), theta.value.cos());
    // Associated Legendre functions use the principal sine factor. Computing
    // its magnitude from sin(theta) preserves small angles lost in 1-cos²(theta).
    if sine.value.re < 0.0 {
        sine = -sine;
    }
    let order = m.abs();
    // Normalize the Legendre recurrence before evaluating it. This removes
    // factorial overflow for high orders and repeated adjacent-order recurrences.
    let mut diagonal = (1.0 / (4.0 * PI)).sqrt();
    for j in 1..=order {
        diagonal *= -(f64::from(2 * j + 1) / f64::from(2 * j)).sqrt();
    }
    if m < 0 && order % 2 != 0 {
        diagonal = -diagonal;
    }
    let mut previous = Jet::default();
    let mut polynomial = Jet::constant(diagonal);
    let mut previous_derivative = Jet::default();
    let mut derivative = Jet::default();
    for j in order + 1..=l {
        let denominator = f64::from(j * j - order * order);
        let a = (f64::from(4 * j * j - 1) / denominator).sqrt();
        let b = if j == order + 1 {
            0.0
        } else {
            (f64::from((2 * j + 1) * ((j - 1) * (j - 1) - order * order))
                / (f64::from(2 * j - 3) * denominator))
                .sqrt()
        };
        if order == 0 && tangential {
            let next = a * (polynomial + cosine * derivative) - b * previous_derivative;
            previous_derivative = derivative;
            derivative = next;
        }
        let next = a * cosine * polynomial - b * previous;
        previous = polynomial;
        polynomial = next;
    }
    let phase = (Complex::i() * f64::from(m) * phi).exp();
    let power = sine.powi((order - 1).max(0));
    let harmonic = phase
        * polynomial
        * if order == 0 {
            Jet::constant(1.0)
        } else {
            power * sine
        };
    if l == 0 || !tangential {
        return [harmonic, Jet::default(), Jet::default()];
    }
    let norm = f64::from(l * (l + 1)).sqrt();
    if order == 0 {
        return [harmonic, Jet::default(), -phase * sine * derivative / norm];
    }
    let adjacent = (f64::from((l * l - order * order) * (2 * l + 1)) / f64::from(2 * l - 1)).sqrt();
    [
        harmonic,
        phase * f64::from(m) * power * polynomial / norm,
        phase * power * (f64::from(l) * cosine * polynomial - adjacent * previous) / norm,
    ]
}

#[inline]
fn evaluate<const N: usize>(
    family: Family,
    mode: Mode,
    args: [Jet<N>; 6],
    helicity: bool,
) -> Result<[Jet<N>; 3]> {
    if mode.pol > 1
        || mode.l < 0
        || mode.l > 128
        || mode.m.unsigned_abs() > 128
        || args.iter().any(|v| !finite(v.value))
    {
        return Err(Error::InvalidInput(
            "require finite wave arguments, 0 <= l <= 128, |m| <= 128 and polarization 0 or 1"
                .into(),
        ));
    }
    let result = match family {
        Family::HarmonicX | Family::HarmonicY | Family::HarmonicZ | Family::Spherical(_) => {
            if mode.m.abs() > mode.l {
                return Ok([Jet::default(); 3]);
            }
            let offset = usize::from(matches!(family, Family::Spherical(_)));
            let [y, pi, tau] = angular(
                mode.l,
                mode.m,
                args[offset],
                args[offset + 1],
                !matches!(family, Family::HarmonicZ),
            );
            let x = [Jet::default(), -pi, -Complex::i() * tau];
            let tangent = [Jet::default(), Complex::i() * tau, -pi];
            match family {
                Family::HarmonicX => x,
                Family::HarmonicY => tangent,
                Family::HarmonicZ => [Complex::i() * y, Jet::default(), Jet::default()],
                Family::Spherical(radial) => {
                    if mode.l == 0 {
                        return Ok([Jet::default(); 3]);
                    }
                    let z = args[0];
                    let l = u32::try_from(mode.l)
                        .map_err(|_| Error::InvalidInput("invalid spherical degree".into()))?;
                    if N == 0 && !helicity && mode.pol == 0 {
                        let kind = if radial == Radial::Regular {
                            special::Bessel::J
                        } else {
                            special::Bessel::H1
                        };
                        let value = special::bessel(f64::from(l), z.value, kind, true, 0)?;
                        return Ok(x.map(|v| v * value));
                    }
                    let radial_jet = special::spherical(l, z.value, radial)?;
                    let value = z.map(radial_jet.value, radial_jet.first);
                    let m = x.map(|v| v * value);
                    if !helicity && mode.pol == 0 {
                        return Ok(m);
                    }
                    let first = z.map(radial_jet.first, radial_jet.second);
                    let divided = if radial == Radial::Regular && z.value.norm() < 0.5 {
                        let lower = special::spherical(l - 1, z.value, radial)?;
                        let upper = special::spherical(l + 1, z.value, radial)?;
                        z.map(
                            (lower.value + upper.value) / f64::from(2 * l + 1),
                            (lower.first + upper.first) / f64::from(2 * l + 1),
                        )
                    } else {
                        value / z
                    };
                    let mut n = tangent.map(|v| v * (divided + first));
                    n[0] = Complex::i() * y * divided * f64::from(mode.l * (mode.l + 1)).sqrt();
                    select(m, n, mode.pol, helicity)
                }
                _ => return Err(Error::InvalidInput("invalid spherical wave family".into())),
            }
        }
        Family::Cylindrical(radial) => {
            let [kz, z, phi, axial, k, _] = args;
            let jet = special::cylindrical(mode.m, z.value, radial)?;
            let value = z.map(jet.value, jet.first);
            let first = z.map(jet.first, jet.second);
            let divided = if mode.m == 0 {
                Jet::default()
            } else if radial == Radial::Regular && z.value.norm() < 0.5 {
                let lower = special::cylindrical(mode.m - 1, z.value, radial)?;
                let upper = special::cylindrical(mode.m + 1, z.value, radial)?;
                z.map(
                    0.5 * (lower.value + upper.value),
                    0.5 * (lower.first + upper.first),
                )
            } else {
                f64::from(mode.m) * value / z
            };
            let phase = (Complex::i() * (f64::from(mode.m) * phi + kz * axial)).exp();
            let m = [
                Complex::i() * divided * phase,
                -first * phase,
                Jet::default(),
            ];
            if !helicity && mode.pol == 0 {
                return Ok(m);
            }
            if k.value == Complex::default() {
                return Err(Error::InvalidInput(
                    "cylindrical N waves require nonzero k".into(),
                ));
            }
            let transverse = (k * k - kz * kz).sqrt();
            let factor = phase / k;
            let n = [
                Complex::i() * kz * first * factor,
                -kz * divided * factor,
                transverse * value * factor,
            ];
            select(m, n, mode.pol, helicity)
        }
        Family::Plane => {
            let vector = [args[0].value, args[1].value, args[2].value];
            if N == 0 {
                return Ok(crate::plane::field_value(
                    crate::plane::polarization(vector, mode.pol, helicity)?,
                    vector,
                    [args[3].value, args[4].value, args[5].value],
                )?
                .map(Jet::constant));
            }
            let polarization = crate::plane::polarization_jet::<3>(vector, mode.pol, helicity)?
                .map(|p| Jet {
                    value: p.value,
                    derivative: std::array::from_fn(|a| {
                        (0..3)
                            .map(|b| p.derivative[b] * args[b].derivative[a])
                            .sum()
                    }),
                });
            let phase =
                (Complex::i() * (0..3).map(|a| args[a] * args[a + 3]).sum::<Jet<N>>()).exp();
            polarization.map(|p| p * phase)
        }
    };
    Ok(result)
}
fn checked<const N: usize>(result: [Jet<N>; 3]) -> Result<[Jet<N>; 3]> {
    if result.iter().any(|v| !v.finite()) {
        return Err(Error::SpecialFunction(
            "nonfinite vector wave or undefined derivative".into(),
        ));
    }
    Ok(result)
}

/// Evaluate a vector wave in its local component frame.
///
/// Harmonics use `(theta,phi)`, spherical waves `(kr,theta,phi)`, cylinders
/// `(kz,k_rho*rho,phi,z,k)`, and planes `(kx,ky,kz,x,y,z)`. Unused arguments are zero.
#[inline]
pub fn value(
    family: Family,
    mode: Mode,
    args: [Complex; 6],
    helicity: bool,
) -> Result<[Complex; 3]> {
    Ok(checked(evaluate::<0>(
        family,
        mode,
        args.map(Jet::constant),
        helicity,
    )?)?
    .map(|v| v.value))
}

/// Contract all six continuous complex arguments with a complex vector cotangent.
/// For a real physical input, use the real part of its returned cotangent.
pub fn pullback(
    family: Family,
    mode: Mode,
    args: [Complex; 6],
    helicity: bool,
    g: [Complex; 3],
) -> Result<[Complex; 6]> {
    if g.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput(
            "vector-wave cotangent must be finite".into(),
        ));
    }
    if g == [Complex::default(); 3] {
        value(family, mode, args, helicity)?;
        return Ok([Complex::default(); 6]);
    }
    match family {
        Family::HarmonicX | Family::HarmonicY | Family::HarmonicZ => {
            pullback_impl::<2>(family, mode, args, helicity, g)
        }
        Family::Spherical(_) => pullback_impl::<3>(family, mode, args, helicity, g),
        Family::Cylindrical(_) => pullback_impl::<5>(family, mode, args, helicity, g),
        Family::Plane => pullback_impl::<6>(family, mode, args, helicity, g),
    }
}
fn pullback_impl<const N: usize>(
    family: Family,
    mode: Mode,
    args: [Complex; 6],
    helicity: bool,
    g: [Complex; 3],
) -> Result<[Complex; 6]> {
    let values = checked(evaluate::<N>(
        family,
        mode,
        std::array::from_fn(|a| Jet::variable(args[a], a)),
        helicity,
    )?)?;
    Ok(std::array::from_fn(|a| {
        values
            .iter()
            .zip(g)
            .map(|(v, g)| v.derivative.get(a).copied().unwrap_or_default().conj() * g)
            .sum()
    }))
}

/// Owned wave labels and scalar-or-array arguments for a first-order pullback.
#[derive(Debug)]
pub struct Residual {
    family: Family,
    modes: Vec<Mode>,
    arguments: [Vec<Complex>; 6],
    helicity: bool,
    size: usize,
}
impl Residual {
    fn element(&self, i: usize) -> (Mode, [Complex; 6]) {
        (
            self.modes[if self.modes.len() == 1 { 0 } else { i }],
            std::array::from_fn(|a| {
                self.arguments[a][if self.arguments[a].len() == 1 { 0 } else { i }]
            }),
        )
    }
    /// Contract complex vector cotangents and reduce constant scalar arguments.
    pub fn pullback(self, cotangent: &[[Complex; 3]]) -> Result<[Vec<Complex>; 6]> {
        use rayon::prelude::*;
        if cotangent.len() != self.size || cotangent.iter().flatten().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "wave cotangent must be finite and match output".into(),
            ));
        }
        let plane = if matches!(self.family, Family::Plane)
            && self.size > 1
            && self.modes.len() == 1
            && self.arguments[..3].iter().all(|v| v.len() == 1)
            && cotangent.iter().flatten().any(|&v| v != Complex::default())
        {
            let (m, a) = self.element(0);
            Some(crate::plane::polarization_jet::<3>(
                [a[0], a[1], a[2]],
                m.pol,
                self.helicity,
            )?)
        } else {
            None
        };
        let evaluate = |(i, &g): (usize, &[Complex; 3])| {
            let (m, a) = self.element(i);
            if let Some(p) = plane {
                let phase = (Complex::i() * (0..3).map(|j| a[j] * a[j + 3]).sum::<Complex>()).exp();
                let g = g.map(|v| v * phase.conj());
                let amplitude = p
                    .into_iter()
                    .zip(g)
                    .map(|(p, g)| p.value.conj() * g)
                    .sum::<Complex>();
                let gradient = std::array::from_fn(|j| {
                    if j < 3 {
                        p.into_iter()
                            .zip(g)
                            .map(|(p, g)| p.derivative[j].conj() * g)
                            .sum::<Complex>()
                            + (Complex::i() * a[j + 3]).conj() * amplitude
                    } else {
                        (Complex::i() * a[j - 3]).conj() * amplitude
                    }
                });
                if gradient.iter().any(|&v| !finite(v)) {
                    return Err(Error::SpecialFunction(
                        "nonfinite plane-wave pullback".into(),
                    ));
                }
                Ok(gradient)
            } else {
                pullback(self.family, m, a, self.helicity, g)
            }
        };
        let gradients: Vec<[Complex; 6]> = if self.size >= 1024 {
            cotangent
                .par_iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        } else {
            cotangent
                .iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        };
        Ok(std::array::from_fn(|a| {
            if self.arguments[a].len() == 1 {
                vec![gradients.iter().map(|g| g[a]).sum()]
            } else {
                gradients.iter().map(|g| g[a]).collect()
            }
        }))
    }
}
/// Broadcast independent vector-wave evaluations while retaining only their inputs.
pub fn array(
    family: Family,
    modes: Vec<Mode>,
    arguments: [Vec<Complex>; 6],
    helicity: bool,
) -> Result<(Vec<[Complex; 3]>, Residual)> {
    use rayon::prelude::*;
    let sizes: Vec<_> = std::iter::once(modes.len())
        .chain(arguments.iter().map(Vec::len))
        .collect();
    let size = if sizes.contains(&0) {
        0
    } else {
        sizes.iter().copied().max().unwrap_or_default()
    };
    if sizes.iter().any(|&n| n != 1 && n != size) {
        return Err(Error::InvalidInput(
            "wave arrays must have equal lengths or scalar inputs".into(),
        ));
    }
    let residual = Residual {
        family,
        modes,
        arguments,
        helicity,
        size,
    };
    let plane = if matches!(family, Family::Plane)
        && size > 1
        && residual.modes.len() == 1
        && residual.arguments[..3].iter().all(|v| v.len() == 1)
    {
        let (mode, args) = residual.element(0);
        // The shared value boundary validates all fixed labels and wavevectors.
        value(family, mode, args, helicity)?;
        Some(crate::plane::polarization(
            [args[0], args[1], args[2]],
            mode.pol,
            helicity,
        )?)
    } else {
        None
    };
    let evaluate = |i| {
        let (m, a) = residual.element(i);
        if let Some(p) = plane {
            crate::plane::field_value(p, [a[0], a[1], a[2]], [a[3], a[4], a[5]])
        } else {
            value(family, m, a, helicity)
        }
    };
    let values = if size >= 1024 {
        (0..size)
            .into_par_iter()
            .map(evaluate)
            .collect::<Result<_>>()?
    } else {
        (0..size).map(evaluate).collect::<Result<_>>()?
    };
    Ok((values, residual))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Reproducible property failures.
    use super::*;
    use crate::{coordinates, cylwaves, fields};
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(40))]
        #[test]
        fn spherical_and_cylindrical_cartesian_reconstruction(l in 1_i32..5,seed in 0_i32..9,r in 0.5_f64..1.5,theta in 0.3_f64..2.8,phi in -3.0_f64..3.0,helicity in any::<bool>(),outgoing in any::<bool>()) {
            let k=Complex::new(1.3,0.05);
            let radial=if outgoing {Radial::Outgoing}else{Radial::Regular};
            let m=seed%(2*l+1)-l;
            for pol in [0,1] {
                let mode=Mode{l,m,pol};
                let position=coordinates::point([r,theta,phi],coordinates::Transform::SphToCar).unwrap();
                let local=value(Family::Spherical(radial),mode,[k*r,theta.into(),phi.into(),0.0.into(),0.0.into(),0.0.into()],helicity).unwrap();
                let actual=coordinates::vector(local,[r,theta,phi],coordinates::Transform::SphToCar).unwrap();
                let expected=fields::spherical_wave(mode,k,position,helicity,radial).unwrap().value;
                for (a,b) in actual.into_iter().zip(expected) {prop_assert!((a-b).norm()<1e-11*(1.0+b.norm()));}
                let kz=0.2;
                let krho=(k*k-kz*kz).sqrt();
                let position=coordinates::point([r,phi,0.3],coordinates::Transform::CylToCar).unwrap();
                let local=value(Family::Cylindrical(radial),mode,[kz.into(),krho*r,phi.into(),0.3.into(),k,0.0.into()],helicity).unwrap();
                let actual=coordinates::vector(local,[r,phi,0.3],coordinates::Transform::CylToCar).unwrap();
                let expected=fields::cylindrical_wave(cylwaves::Mode{kz,m,pol},k,position,helicity,radial).unwrap().value;
                for (a,b) in actual.into_iter().zip(expected) {prop_assert!((a-b).norm()<1e-11*(1.0+b.norm()));}
            }
        }
        #[test]
        fn vector_wave_all_argument_adjoints(theta in 0.3_f64..2.8,phi in -2.0_f64..2.0,helicity in any::<bool>(),pol in 0_u8..2) {
            let mode=Mode{l:3,m:-1,pol};
            let g=[Complex::new(0.3,0.2),Complex::new(-0.2,0.1),Complex::new(0.4,-0.3)];
            let direction=Complex::new(0.2,0.1);
            let h=1e-6;
            for (family,args) in [
                (Family::HarmonicX,[theta.into(),phi.into(),0.0.into(),0.0.into(),0.0.into(),0.0.into()]),
                (Family::HarmonicY,[theta.into(),phi.into(),0.0.into(),0.0.into(),0.0.into(),0.0.into()]),
                (Family::HarmonicZ,[theta.into(),phi.into(),0.0.into(),0.0.into(),0.0.into(),0.0.into()]),
                (Family::Spherical(Radial::Regular),[Complex::new(1.2,0.1),theta.into(),phi.into(),0.0.into(),0.0.into(),0.0.into()]),
                (Family::Spherical(Radial::Outgoing),[Complex::new(1.2,0.1),theta.into(),phi.into(),0.0.into(),0.0.into(),0.0.into()]),
                (Family::Cylindrical(Radial::Regular),[0.2.into(),Complex::new(1.1,0.1),phi.into(),0.3.into(),Complex::new(1.3,0.05),0.0.into()]),
                (Family::Cylindrical(Radial::Outgoing),[0.2.into(),Complex::new(1.1,0.1),phi.into(),0.3.into(),Complex::new(1.3,0.05),0.0.into()]),
                (Family::Plane,[Complex::new(0.3,0.1),Complex::new(0.4,-0.05),Complex::new(1.2,0.1),0.2.into(),0.3.into(),0.4.into()]),
            ] {
                let gradient=pullback(family,mode,args,helicity,g).unwrap();
                for a in 0..6 {
                    let mut plus=args;plus[a]+=h*direction;
                    let mut minus=args;minus[a]-=h*direction;
                    let difference=value(family,mode,plus,helicity).unwrap().into_iter().zip(value(family,mode,minus,helicity).unwrap()).zip(g).map(|((p,m),g)|(g.conj()*(p-m)).re/(2.0*h)).sum::<f64>();
                    prop_assert!((difference-(gradient[a].conj()*direction).re).abs()<2e-7*(1.0+difference.abs()));
                }
            }
        }
    }
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        #[test]
        fn normalized_high_degree_addition_theorem(l in 20_i32..129,theta in 0.0_f64..PI,phi in -3.0_f64..3.0) {
            let args=[theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default(),Complex::default()];
            for family in [Family::HarmonicX,Family::HarmonicY,Family::HarmonicZ] {
                let total=(-l..=l).map(|m| value(family,Mode{l,m,pol:0},args,false).unwrap().into_iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>();
                let expected=f64::from(2*l+1)/(4.0*PI);
                prop_assert!((total-expected).abs()<2e-11*expected);
            }
        }
    }
    #[test]
    fn regular_spherical_origin_and_harmonic_poles() {
        for l in 1..=3 {
            for m in -l..=l {
                let mode = Mode { l, m, pol: 1 };
                for theta in [0.0, 0.7, PI] {
                    let local = value(
                        Family::Spherical(Radial::Regular),
                        mode,
                        [
                            0.0.into(),
                            theta.into(),
                            0.3.into(),
                            0.0.into(),
                            0.0.into(),
                            0.0.into(),
                        ],
                        false,
                    )
                    .unwrap();
                    let actual = coordinates::vector(
                        local,
                        [0.0, theta, 0.3],
                        coordinates::Transform::SphToCar,
                    )
                    .unwrap();
                    let expected =
                        fields::spherical_wave(mode, 1.3.into(), [0.0; 3], false, Radial::Regular)
                            .unwrap()
                            .value;
                    for (a, b) in actual.into_iter().zip(expected) {
                        assert!((a - b).norm() < 1e-13);
                    }
                    pullback(
                        Family::HarmonicX,
                        mode,
                        [
                            theta.into(),
                            0.3.into(),
                            0.0.into(),
                            0.0.into(),
                            0.0.into(),
                            0.0.into(),
                        ],
                        false,
                        [Complex::new(0.3, 0.2); 3],
                    )
                    .unwrap();
                }
            }
        }
    }
}
