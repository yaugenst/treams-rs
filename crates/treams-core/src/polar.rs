//! Translation coefficients in the public spherical-coordinate convention.
#![allow(clippy::indexing_slicing)] // Fixed argument and derivative arrays.
use crate::{
    Complex, Error, Result, finite,
    jet::Jet,
    special::{self, Radial},
    waves::{self, Mode},
};

/// Fixed spherical mode-pair couplings shared across displacements and pullbacks.
#[derive(Debug)]
pub struct SphericalTranslation {
    terms: Vec<(i32, i32, Complex)>,
    radial: Radial,
}
impl SphericalTranslation {
    /// Prepare one mode pair in helicity or parity polarization.
    pub fn new(destination: Mode, source: Mode, helicity: bool, radial: Radial) -> Result<Self> {
        destination.validate()?;
        source.validate()?;
        Ok(Self {
            terms: waves::terms(destination, source, helicity),
            radial,
        })
    }
    fn evaluate<const N: usize>(&self, args: [Complex; 3]) -> Result<Jet<N>> {
        if args.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "translation arguments must be finite".into(),
            ));
        }
        let [kr, theta, phi] = std::array::from_fn(|i| Jet::<N>::variable(args[i], i));
        let [cosine, sine] = crate::vectorwaves::polar_trig(theta);
        let order = self.terms.first().map_or(0, |t| t.1);
        let phase = sine.powi(order.abs()) * (Complex::i() * f64::from(order) * phi).exp();
        let mut result = Jet::default();
        for &(degree, order, weight) in &self.terms {
            let l = u32::try_from(degree)
                .map_err(|_| Error::InvalidInput("invalid translation degree".into()))?;
            let radial = if N == 0 {
                Jet::constant(special::bessel(
                    f64::from(l),
                    kr.value,
                    if self.radial == Radial::Regular {
                        special::Bessel::J
                    } else {
                        special::Bessel::H1
                    },
                    true,
                    0,
                )?)
            } else {
                let radial = special::spherical(l, kr.value, self.radial)?;
                kr.map(radial.value, radial.first)
            };
            let angular = phase * special::legendre_factor(degree, order, cosine);
            result += weight * radial * angular;
        }
        if !result.finite() {
            return Err(Error::SpecialFunction(
                "nonfinite translation or derivative".into(),
            ));
        }
        Ok(result)
    }
    /// Evaluate at `(kr, theta, phi)`; all three arguments may be complex.
    pub fn value(&self, args: [Complex; 3]) -> Result<Complex> {
        Ok(self.evaluate::<0>(args)?.value)
    }
    /// Contract the three argument derivatives with a complex scalar cotangent.
    pub fn pullback(&self, args: [Complex; 3], g: Complex) -> Result<[Complex; 3]> {
        if !finite(g) {
            return Err(Error::InvalidInput(
                "translation cotangent must be finite".into(),
            ));
        }
        if g == Complex::default() {
            self.value(args)?;
            return Ok([Complex::default(); 3]);
        }
        Ok(self.evaluate::<3>(args)?.derivative.map(|d| d.conj() * g))
    }
}

/// Owned polar-translation inputs; constant labels retain a single coupling plan.
#[derive(Debug)]
pub struct Residual {
    modes: Vec<[Mode; 2]>,
    arguments: [Vec<Complex>; 3],
    plan: Option<SphericalTranslation>,
    helicity: bool,
    radial: Radial,
    size: usize,
}
impl Residual {
    fn args(&self, i: usize) -> [Complex; 3] {
        std::array::from_fn(|a| self.arguments[a][if self.arguments[a].len() == 1 { 0 } else { i }])
    }
    fn with_plan<T>(
        &self,
        i: usize,
        f: impl FnOnce(&SphericalTranslation) -> Result<T>,
    ) -> Result<T> {
        if let Some(plan) = &self.plan {
            f(plan)
        } else {
            let [to, from] = self.modes[i];
            f(&SphericalTranslation::new(
                to,
                from,
                self.helicity,
                self.radial,
            )?)
        }
    }
    /// Contract a scalar cotangent per translation and reduce constant arguments.
    pub fn pullback(self, g: &[Complex]) -> Result<[Vec<Complex>; 3]> {
        use rayon::prelude::*;
        if g.len() != self.size || g.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "translation cotangent must be finite and match output".into(),
            ));
        }
        let evaluate =
            |(i, &g): (usize, &Complex)| self.with_plan(i, |p| p.pullback(self.args(i), g));
        let gradients: Vec<[Complex; 3]> = if self.size >= 1024 {
            g.par_iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        } else {
            g.iter().enumerate().map(evaluate).collect::<Result<_>>()?
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
/// Broadcast polar translations, preserving scalar arguments and fixed mode pairs.
pub fn spherical(
    modes: Vec<[Mode; 2]>,
    arguments: [Vec<Complex>; 3],
    helicity: bool,
    radial: Radial,
) -> Result<(Vec<Complex>, Residual)> {
    use rayon::prelude::*;
    let lengths = [
        modes.len(),
        arguments[0].len(),
        arguments[1].len(),
        arguments[2].len(),
    ];
    let size = if lengths.contains(&0) {
        0
    } else {
        lengths.into_iter().max().unwrap_or_default()
    };
    if lengths.iter().any(|&n| n != 1 && n != size) {
        return Err(Error::InvalidInput(
            "translation arrays must have equal lengths or scalar inputs".into(),
        ));
    }
    let plan = if modes.len() == 1 {
        Some(SphericalTranslation::new(
            modes[0][0],
            modes[0][1],
            helicity,
            radial,
        )?)
    } else {
        None
    };
    let residual = Residual {
        modes,
        arguments,
        plan,
        helicity,
        radial,
        size,
    };
    let evaluate = |i| residual.with_plan(i, |p| p.value(residual.args(i)));
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

/// Cylindrical translation in `(k_rho*rho, phi, z, kz)` with a fixed order difference.
/// The two axial labels must move together; unequal labels are uncoupled.
pub fn cylindrical_value(order: i32, args: [Complex; 4], radial: Radial) -> Result<Complex> {
    Ok(cylindrical_jet::<0>(order, args, radial)?.value)
}
fn cylindrical_jet<const N: usize>(
    order: i32,
    args: [Complex; 4],
    radial: Radial,
) -> Result<Jet<N>> {
    if order.unsigned_abs() > 256 || args.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput(
            "require |order difference| <= 256 and finite arguments".into(),
        ));
    }
    let [kr, phi, z, kz] = std::array::from_fn(|i| Jet::<N>::variable(args[i], i));
    let radial = if N == 0 {
        Jet::constant(special::bessel(
            f64::from(order),
            kr.value,
            if radial == Radial::Regular {
                special::Bessel::J
            } else {
                special::Bessel::H1
            },
            false,
            0,
        )?)
    } else {
        let r = special::cylindrical(order, kr.value, radial)?;
        kr.map(r.value, r.first)
    };
    let value = radial * (Complex::i() * (f64::from(order) * phi + kz * z)).exp();
    if !value.finite() {
        return Err(Error::SpecialFunction(
            "nonfinite cylindrical translation or derivative".into(),
        ));
    }
    Ok(value)
}
/// Contract the radial argument, azimuth, axial position and common axial wavenumber.
pub fn cylindrical_pullback(
    order: i32,
    args: [Complex; 4],
    radial: Radial,
    g: Complex,
) -> Result<[Complex; 4]> {
    if !finite(g) {
        return Err(Error::InvalidInput("cotangent must be finite".into()));
    }
    if g == Complex::default() {
        cylindrical_value(order, args, radial)?;
        return Ok([Complex::default(); 4]);
    }
    Ok(cylindrical_jet::<4>(order, args, radial)?
        .derivative
        .map(|d| d.conj() * g))
}

/// Owned cylindrical polar inputs, retaining broadcast constants once.
#[derive(Debug)]
pub struct CylindricalResidual {
    orders: Vec<i32>,
    arguments: [Vec<Complex>; 4],
    radial: Radial,
    size: usize,
}
impl CylindricalResidual {
    fn element(&self, i: usize) -> (i32, [Complex; 4]) {
        (
            self.orders[if self.orders.len() == 1 { 0 } else { i }],
            std::array::from_fn(|a| {
                self.arguments[a][if self.arguments[a].len() == 1 { 0 } else { i }]
            }),
        )
    }
    /// Contract all four continuous arguments and reduce broadcast scalars.
    pub fn pullback(self, g: &[Complex]) -> Result<[Vec<Complex>; 4]> {
        use rayon::prelude::*;
        if g.len() != self.size || g.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "cotangent must be finite and match output".into(),
            ));
        }
        let evaluate = |(i, &g): (usize, &Complex)| {
            let (order, args) = self.element(i);
            cylindrical_pullback(order, args, self.radial, g)
        };
        let gradients: Vec<[Complex; 4]> = if self.size >= 64 {
            g.par_iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        } else {
            g.iter().enumerate().map(evaluate).collect::<Result<_>>()?
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
/// Evaluate a broadcast array of cylindrical polar translation coefficients.
pub fn cylindrical(
    orders: Vec<i32>,
    arguments: [Vec<Complex>; 4],
    radial: Radial,
) -> Result<(Vec<Complex>, CylindricalResidual)> {
    use rayon::prelude::*;
    let lengths = [
        orders.len(),
        arguments[0].len(),
        arguments[1].len(),
        arguments[2].len(),
        arguments[3].len(),
    ];
    let size = if lengths.contains(&0) {
        0
    } else {
        lengths.into_iter().max().unwrap_or_default()
    };
    if lengths.iter().any(|&n| n != 1 && n != size) || orders.iter().any(|m| m.unsigned_abs() > 256)
    {
        return Err(Error::InvalidInput(
            "require |order difference| <= 256 and equal lengths or scalar inputs".into(),
        ));
    }
    let residual = CylindricalResidual {
        orders,
        arguments,
        radial,
        size,
    };
    let evaluate = |i| {
        let (order, args) = residual.element(i);
        cylindrical_value(order, args, radial)
    };
    let value = if size >= 64 {
        (0..size)
            .into_par_iter()
            .map(evaluate)
            .collect::<Result<_>>()?
    } else {
        (0..size).map(evaluate).collect::<Result<_>>()?
    };
    Ok((value, residual))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(40))]
        #[test]
        fn cylindrical_cartesian_and_polar_adjoints(mu in -5_i32..6,m in -5_i32..6,phi in -3.0_f64..3.0,regular in any::<bool>()) {
            let args=[Complex::new(1.3,0.1),phi.into(),0.3.into(),0.2.into()];
            let radial=if regular {Radial::Regular}else{Radial::Outgoing};
            let k=(args[0]*args[0]+args[3]*args[3]).sqrt();
            let mode=|m| crate::cylwaves::Mode{kz:0.2,m,pol:0};
            let expected=crate::cylwaves::translate(mode(mu),mode(m),k,[phi.cos(),phi.sin(),0.3],radial).unwrap();
            let value=cylindrical_value(m-mu,args,radial).unwrap();
            prop_assert!((value-expected.value).norm()<1e-10*(1.0+value.norm()));
            let g=Complex::new(0.3,-0.2);let d=Complex::new(0.2,0.1);let h=1e-6;
            let gradient=cylindrical_pullback(m-mu,args,radial,g).unwrap();
            for i in 0..4 {
                let mut plus=args;plus[i]+=h*d;let mut minus=args;minus[i]-=h*d;
                let fd=(g.conj()*(cylindrical_value(m-mu,plus,radial).unwrap()-cylindrical_value(m-mu,minus,radial).unwrap())/(2.0*h)).re;
                prop_assert!(((gradient[i].conj()*d).re-fd).abs()<3e-7*(1.0+fd.abs()));
            }
        }

        #[test]
        fn cartesian_translation_and_all_polar_adjoints(l in 1_i32..6,lambda in 1_i32..6,seed in 0_i32..17,theta in 0.1_f64..3.0,phi in -3.0_f64..3.0,helicity in any::<bool>(),regular in any::<bool>()) {
            let to=Mode{l:lambda,m:seed%(2*lambda+1)-lambda,pol:1};
            let from=Mode{l,m:seed%(2*l+1)-l,pol:u8::from(seed%2==0)};
            let radial=if regular {Radial::Regular}else{Radial::Outgoing};
            let kr=Complex::new(1.3,0.2);
            let args=[kr,theta.into(),phi.into()];
            let plan=SphericalTranslation::new(to,from,helicity,radial).unwrap();
            let position=crate::coordinates::point([1.0,theta,phi],crate::coordinates::Transform::SphToCar).unwrap();
            let reference=waves::translate(to,from,kr,position,helicity,radial).unwrap();
            let actual=plan.value(args).unwrap();
            prop_assert!((actual-reference.value).norm()<2e-10*(1.0+actual.norm()));
            let g=Complex::new(0.3,-0.2);let direction=Complex::new(0.2,0.1);let h=1e-6;
            let gradient=plan.pullback(args,g).unwrap();
            for i in 0..3 {
                let mut plus=args;plus[i]+=h*direction;
                let mut minus=args;minus[i]-=h*direction;
                let expected=(g.conj()*(plan.value(plus).unwrap()-plan.value(minus).unwrap())/(2.0*h)).re;
                prop_assert!(((gradient[i].conj()*direction).re-expected).abs()<2e-7*(1.0+expected.abs()));
            }
        }
    }
}
