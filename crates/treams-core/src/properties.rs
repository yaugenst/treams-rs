//! Native physical and differentiation invariants, independent of Python.
#![allow(clippy::unwrap_used, clippy::expect_used)] // proptest shrinks panics to reproducible counterexamples.

use nalgebra::DMatrix;
use proptest::prelude::*;

use crate::{
    Complex,
    coeffs::{Material, mie_forward},
    interaction,
};

use crate::{
    angular::wigner3j,
    special::Radial,
    waves::{Mode, translate},
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    #[allow(clippy::indexing_slicing)] // Fixed two-medium arrays.
    fn fresnel_dimensionless_adjoint(k in 1.0_f64..3.0, q in 0.0_f64..0.6, z in 0.7_f64..1.3) {
        let ks=[[Complex::new(k,0.1),Complex::new(k+0.2,0.1)],[Complex::new(k+0.5,0.2),Complex::new(k+0.8,0.2)]];
        let kz=ks.map(|r|r.map(|k|(k*k-q*q).sqrt()));
        let impedance=[Complex::new(z,0.03),Complex::new(z+0.2,0.04)];
        let result=crate::smatrix::fresnel(ks,kz,impedance).unwrap();
        let g=std::array::from_fn(|b|DMatrix::from_fn(2,2,|i,j|Complex::new(if b==i+j {0.7}else{0.2},0.1)));
        let (gk,gkz,gz)=result.pullback(&g).unwrap();
        let euler:Complex=(0..2).flat_map(|i|(0..2).map(move|j|gk[i][j].conj()*ks[i][j]+gkz[i][j].conj()*kz[i][j])).sum();
        prop_assert!(euler.norm()<1e-10);
        prop_assert!((gz[0].conj()*impedance[0]+gz[1].conj()*impedance[1]).norm()<1e-10);
    }

    #[test]
    fn propagation_phase_conserves_power(k in 0.5_f64..3.0, distance in 0.0_f64..10.0) {
        let result=crate::smatrix::propagation(vec![[Complex::new(0.1,0.0),Complex::new(0.2,0.0),Complex::new(k,0.0)]],[0.2,-0.3,distance]).unwrap();
        let g=result.value.clone().map(|b|b*Complex::new(2.0,0.0));
        let (vectors,r)=result.pullback(&g).unwrap();
        prop_assert!(r.into_iter().all(|g|g.abs()<1e-12));
        prop_assert!(vectors.into_iter().flatten().all(|g|g.re.abs()<1e-12));
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // Fixed four-block arrays, matching native block order.
    fn smatrix_composition_and_adjoint(x in -0.3_f64..0.3, y in -0.3_f64..0.3) {
        use crate::smatrix::{Blocks, add};
        let make = |x:f64,y:f64| -> Blocks { std::array::from_fn(|b|DMatrix::from_fn(2,2,|i,j|Complex::new(x*(if i==j {1.0}else{0.2}), y*(if b==i+j {1.0}else{0.1})))) };
        let a=make(x,y);
        let b=make(y,x);
        let c=make(0.1,-0.1);
        let res=add(a.clone(),b.clone()).unwrap();
        let left=add(res.value.clone(),c.clone()).unwrap().value;
        let right=add(a.clone(),add(b.clone(),c).unwrap().value).unwrap().value;
        for (l,r) in left.iter().zip(right) {prop_assert!((l-r).norm()<1e-12);}
        let g=make(0.3,0.2);
        let da=make(0.1,-0.2);
        let db=make(-0.2,0.1);
        let (ga,gb)=res.pullback(&g).unwrap();
        let h=1e-5;
        let shifted=|sign:f64|add(std::array::from_fn(|i|&a[i]+&da[i]*Complex::new(sign*h,0.0)),std::array::from_fn(|i|&b[i]+&db[i]*Complex::new(sign*h,0.0))).unwrap().value;
        let plus=shifted(1.0);
        let minus=shifted(-1.0);
        let analytic:f64=(0..4).map(|i|ga[i].dotc(&da[i]).re+gb[i].dotc(&db[i]).re).sum();
        let numerical:f64=(0..4).map(|i|g[i].dotc(&((&plus[i]-&minus[i])*Complex::new(0.5/h,0.0))).re).sum();
        prop_assert!((analytic-numerical).abs()<1e-8);
    }



    #[test]
    fn ewald_complete_derivative(pitch in 1.3_f64..2.0, bloch in 0.05_f64..0.3, order in 0_i32..4) {
        use crate::lattice::{Lattice, Wave, sum, derivatives};
        let lattice = Lattice::new(&[vec![pitch]], &[bloch]).unwrap();
        let wave = Wave::Cylindrical { m: order };
        let k = Complex::new(2.1,0.2);
        let r = [0.19,0.13,0.0];
        let eta = Complex::new(1.2,0.0);
        let d = derivatives(wave,k,&lattice,r,eta).unwrap();
        let euler = r.iter().zip(d.position).map(|(r,d)|r*d).sum::<Complex>() + pitch*d.vectors[0][0] - k*d.k - bloch*d.bloch[0];
        prop_assert!(euler.norm() < 1e-9*(1.0+d.value.norm()));
        let h = 1e-5;
        let dk = Complex::new(0.1,0.2);
        let direction = dk*d.k + 0.1*d.position[0] - 0.07*d.position[1] + 0.2*d.vectors[0][0] + 0.13*d.bloch[0];
        let plus = Lattice::new(&[vec![pitch+0.2*h]], &[bloch+0.13*h]).unwrap();
        let minus = Lattice::new(&[vec![pitch-0.2*h]], &[bloch-0.13*h]).unwrap();
        let numerical = (sum(wave,k+h*dk,&plus,[r[0]+0.1*h,r[1]-0.07*h,0.0],eta).unwrap()
            - sum(wave,k-h*dk,&minus,[r[0]-0.1*h,r[1]+0.07*h,0.0],eta).unwrap())/(2.0*h);
        prop_assert!((numerical-direction).norm()<2e-7*(1.0+direction.norm()));
    }

    #[test]
    fn ewald_bloch_split_and_scale(pitch in 1.3_f64..2.0, bloch in 0.05_f64..0.3, order in 0_i32..4, scale in 0.8_f64..1.2) {
        use crate::lattice::{Lattice, Wave, sum};
        let lattice = Lattice::new(&[vec![pitch]], &[bloch]).unwrap();
        let wave = Wave::Cylindrical { m: order };
        let k = Complex::new(2.2, 0.3);
        let r = [0.19, 0.13, 0.0];
        let eta = Complex::new(1.2, 0.0);
        let value = sum(wave, k, &lattice, r, eta).unwrap();
        let shifted = sum(wave, k, &lattice, [r[0]+pitch, r[1], 0.0], eta).unwrap();
        prop_assert!((shifted-(-Complex::i()*bloch*pitch).exp()*value).norm() < 1e-9*(1.0+value.norm()));
        let resplit = sum(wave, k, &lattice, r, Complex::new(1.6,0.0)).unwrap();
        prop_assert!((resplit-value).norm() < 1e-8*(1.0+value.norm()));
        let scaled = Lattice::new(&[vec![pitch*scale]], &[bloch/scale]).unwrap();
        prop_assert!((sum(wave, k/scale, &scaled, r.map(|x|x*scale), eta).unwrap()-value).norm() < 1e-9*(1.0+value.norm()));
    }

    #[test]
    fn differentiated_optical_theorem(size in 0.3_f64..2.0, epsilon in 1.2_f64..6.0, l in 1_u32..6) {
        let material=Material{epsilon:Complex::new(epsilon,0.0),..Material::default()};
        let forward=mie_forward(l,&[size],&[material,Material::default()]).unwrap();
        let cotangent=crate::coeffs::Matrix2::identity()+forward.value*Complex::new(2.0,0.0);
        let gradient=forward.pullback(&cotangent).unwrap();
        prop_assert!(gradient.sizes.iter().all(|g|g.abs()<1e-10));
        prop_assert!(gradient.epsilon.iter().all(|g|g.re.abs()<1e-10));
    }

    #[test]
    fn translation_scale_derivative_identity(x in -1.0_f64..1.0, y in -1.0_f64..1.0, z in 0.5_f64..2.0, k in 0.7_f64..2.0, degree in 1_i32..5) {
        let position=[x,y,z];
        let result=translate(Mode{l:degree,m:1,pol:1},Mode{l:2,m:-1,pol:1},Complex::new(k,0.0),position,true,Radial::Outgoing).unwrap();
        let spatial:Complex=result.position.iter().zip(position).map(|(g,r)|g*r).sum();
        prop_assert!((spatial-k*result.k).norm()<1e-8*(1.0+result.value.norm()));
        let scaled=translate(Mode{l:degree,m:1,pol:1},Mode{l:2,m:-1,pol:1},Complex::new(k/1.3,0.0),position.map(|r|r*1.3),true,Radial::Outgoing).unwrap();
        prop_assert!((scaled.value-result.value).norm()<1e-10*(1.0+result.value.norm()));
    }

    #[test]
    fn wigner_orthogonality(j1 in 0_i32..9, j2 in 0_i32..9, selector in 0_i32..10) {
        let j3=(j1-j2).abs()+selector%(j1+j2-(j1-j2).abs()+1);
        let mut sum=0.0;
        for m1 in -j1..=j1 {
            let m2=-m1;
            sum+=wigner3j(j1,j2,j3,m1,m2,0).powi(2);
        }
        prop_assert!((sum*f64::from(2*j3+1)-1.0).abs()<1e-10);
    }

    #[test]
    fn lossless_optical_theorem(size in 0.2_f64..3.0, epsilon in 1.0_f64..8.0, l in 1_u32..9) {
        let material=Material{epsilon:Complex::new(epsilon,0.0),..Material::default()};
        let result=mie_forward(l,&[size],&[material,Material::default()]).unwrap();
        let extinction=-result.value.trace().re;
        let scattering=result.value.iter().map(Complex::norm_sqr).sum::<f64>();
        prop_assert!((extinction-scattering).abs()<1e-11);
    }

    #[test]
    fn homogeneous_layer_split_is_invisible(size in 0.2_f64..3.0, epsilon in 1.0_f64..5.0, fraction in 0.2_f64..0.8, l in 1_u32..7) {
        let material=Material{epsilon:Complex::new(epsilon,0.1),..Material::default()};
        let one=mie_forward(l,&[size],&[material,Material::default()]).unwrap();
        let two=mie_forward(l,&[fraction*size,size],&[material,material,Material::default()]).unwrap();
        prop_assert!((one.value-two.value).norm()<1e-10);
    }

    #[test]
    fn zero_contrast(size in 0.2_f64..3.0, epsilon in 1.0_f64..8.0, l in 1_u32..9) {
        let material=Material{epsilon:Complex::new(epsilon,0.0),..Material::default()};
        let result=mie_forward(l,&[size],&[material,material]).unwrap();
        prop_assert!(result.value.norm()<1e-12);
    }

    #[test]
    fn interaction_adjoint_identity(values in prop::collection::vec(-0.1_f64..0.1,72)) {
        let mut values=values.as_chunks::<2>().0.iter().map(|&[re, im]|Complex::new(re, im));
        let local=DMatrix::from_iterator(3,3,values.by_ref().take(9));
        let coupling=DMatrix::from_iterator(3,3,values.by_ref().take(9));
        let direction=DMatrix::from_iterator(3,3,values.by_ref().take(9));
        let g=DMatrix::from_iterator(3,3,values.by_ref().take(9));
        let forward=interaction::forward(local.clone(),coupling.clone()).unwrap();
        let residual=(DMatrix::identity(3,3)-&local*&coupling)*&forward.value-&local;
        prop_assert!(residual.norm()<1e-12);
        let (gt,gc)=forward.pullback(&g).unwrap();
        for (is_local,gradient) in [(true,gt),(false,gc)] {
            let h=Complex::new(1e-5,0.0);
            let (plus,minus)=if is_local {
                (interaction::forward(&local+&direction*h,coupling.clone()).unwrap(),interaction::forward(&local-&direction*h,coupling.clone()).unwrap())
            } else {
                (interaction::forward(local.clone(),&coupling+&direction*h).unwrap(),interaction::forward(local.clone(),&coupling-&direction*h).unwrap())
            };
            let numerical=g.dotc(&((plus.value-minus.value)/(2.0*h))).re;
            let analytic=gradient.dotc(&direction).re;
            prop_assert!((numerical-analytic).abs()<1e-8);
        }
    }
    #[test]
    fn cluster_complete_directional_derivative(radius in 0.1_f64..0.4, eps in 1.2_f64..6.0, k in 0.7_f64..1.8, dx in -0.3_f64..0.3) {
        let solve = |step: f64| crate::tmatrix::cluster(1, k + step * 0.2,
            &[radius + step * 0.3, 0.25 - step * 0.1],
            &[Complex::new(eps + step * 0.4, 0.1 + step * 0.1), Complex::new(3.0 - step * 0.2, 0.2 - step * 0.3)],
            &[[step * 0.1, 0.0, 0.0], [dx, 0.1 + step * 0.3, 1.5 - step * 0.2]]).unwrap();
        let forward = solve(0.0);
        let g = DMatrix::from_element(12, 12, Complex::new(0.7, -0.4));
        let gradients = forward.pullback(&g).unwrap();
        let h = 1e-5;
        let numerical = g.dotc(&((solve(h).value() - solve(-h).value()) / Complex::new(2.0*h,0.0))).re;
        let radius_direction = [0.3, -0.1];
        let epsilon_direction = [Complex::new(0.4,0.1),Complex::new(-0.2,-0.3)];
        let position_direction = [[0.1,0.0,0.0],[0.0,0.3,-0.2]];
        let analytic = gradients.k0 * 0.2
            + gradients.radii.iter().zip(radius_direction).map(|(g,d)|g*d).sum::<f64>()
            + gradients.epsilon.iter().zip(epsilon_direction).map(|(g,d)|(g.conj()*d).re).sum::<f64>()
            + gradients.positions.iter().flatten().zip(position_direction.iter().flatten()).map(|(g,d)|g*d).sum::<f64>();
        prop_assert!((numerical - analytic).abs() < 1e-8 * (1.0 + numerical.abs()));
        for axis in 0..3 {
            prop_assert!(gradients.positions.iter().map(|g|g.get(axis).unwrap()).sum::<f64>().abs() < 1e-12);
        }
    }

    #[test]
    fn cylinder_optical_theorem_and_its_derivative(radius in 0.1_f64..1.5, epsilon in 1.1_f64..6.0, order in -4_i32..5, kz in -0.5_f64..0.5) {
        let material=Material{epsilon:Complex::new(epsilon,0.0),kappa:Complex::new(0.02,0.0),..Material::default()};
        let forward=crate::cylinder::mie_cyl(kz,order,1.3,&[radius],&[material,Material::default()]).unwrap();
        let extinction=-forward.value.trace().re;
        let scattering=forward.value.iter().map(Complex::norm_sqr).sum::<f64>();
        prop_assert!((extinction-scattering).abs()<1e-11);
        let g=crate::coeffs::Matrix2::identity()+forward.value*Complex::new(2.0,0.0);
        let gradient=forward.pullback(&g).unwrap();
        prop_assert!(gradient.kz.abs()<1e-9 && gradient.k0.abs()<1e-9);
        prop_assert!(gradient.layers.sizes.iter().all(|g|g.abs()<1e-9));
        prop_assert!(gradient.layers.epsilon.iter().chain(&gradient.layers.mu).chain(&gradient.layers.kappa).all(|g|g.re.abs()<1e-9));
    }

    #[test]
    fn vector_wave_maxwell_and_scaling(
        l in 1_i32..7, selector in 0_i32..20, pol in 0_u8..2,
        x in -1.0_f64..1.0, y in -1.0_f64..1.0, z in 0.5_f64..2.0,
        outgoing in any::<bool>(),
    ) {
        let m=selector%(2*l+1)-l;
        let mode=Mode{l,m,pol};
        let k=Complex::new(1.2,0.1);
        let radial=if outgoing {Radial::Outgoing} else {Radial::Regular};
        for position in [[x,y,z],[0.0,0.0,z],[0.0,0.0,if outgoing {z} else {0.0}]] {
            let wave=crate::fields::spherical_wave(mode,k,position,true,radial).unwrap();
            let [jx,jy,jz]=wave.position;
            let curl=[jz[1]-jy[2],jx[2]-jz[0],jy[0]-jx[1]];
            let scale=1.0+wave.value.iter().map(Complex::norm_sqr).sum::<f64>().sqrt();
            for (actual,value) in curl.iter().zip(wave.value) {
                prop_assert!((*actual-(2.0*f64::from(pol)-1.0)*k*value).norm()<1e-10*scale);
            }
            prop_assert!((jx[0]+jy[1]+jz[2]).norm()<1e-10*scale);
            for (jacobian,dk) in wave.position.iter().zip(wave.k) {
                let spatial:Complex=jacobian.iter().zip(position).map(|(d,r)|d*r).sum();
                prop_assert!((spatial-k*dk).norm()<1e-10*scale);
            }
        }
    }

    #[test]
    fn field_amplitude_adjoint_and_coordinate_scale_identity(values in prop::collection::vec(-0.5_f64..0.5, 24), z in 0.8_f64..2.0) {
        let modes=(-1..=1).flat_map(|m|(0..2).map(move |pol|(0,Mode{l:1,m,pol}))).collect();
        let basis=crate::basis::Basis{modes,positions:vec![[0.1,0.2,0.3]]};
        let mut values=values.as_chunks::<2>().0.iter().map(|&[re,im]|Complex::new(re,im));
        let coefficients:Vec<_>=values.by_ref().take(6).collect();
        let cotangent:Vec<[Complex;3]>=(0..2).map(|_|std::array::from_fn(|_|values.next().unwrap())).collect();
        let points=vec![[0.7,0.1,z],[0.0,0.0,z]];
        let ks=[Complex::new(1.1,0.1),Complex::new(1.3,0.1)];
        let forward=crate::fields::field(basis.clone(),coefficients.clone(),points.clone(),ks,true,Radial::Outgoing).unwrap();
        let loss:f64=forward.value.iter().flatten().zip(cotangent.iter().flatten()).map(|(f,g)|(g.conj()*f).re).sum();
        let gradient=forward.pullback(&cotangent).unwrap();
        let amplitude_pairing:f64=gradient.coefficients.iter().zip(coefficients).map(|(g,c)|(g.conj()*c).re).sum();
        prop_assert!((loss-amplitude_pairing).abs()<1e-10);
        for axis in 0..3 {
            let total=gradient.points.iter().chain(&gradient.origins).map(|g|g.get(axis).unwrap()).sum::<f64>();
            prop_assert!(total.abs()<1e-10);
        }
        let spatial:f64=gradient.points.iter().flatten().chain(gradient.origins.iter().flatten()).zip(points.iter().flatten().chain(basis.positions.iter().flatten())).map(|(g,p)|g*p).sum();
        let spectral:f64=gradient.ks.iter().zip(ks).map(|(g,k)|(g.conj()*k).re).sum();
        prop_assert!((spatial-spectral).abs()<1e-10);
    }

    #[test]
    fn cylinder_matrix_differentiated_optical_theorem(radius in 0.1_f64..1.0, epsilon in 1.1_f64..5.0, mmax in 0_u32..4) {
        let material=Material{epsilon:Complex::new(epsilon,0.0),kappa:Complex::new(0.02,0.0),..Material::default()};
        let forward=crate::cylinder::cylinder(&[0.2,-0.3],mmax,1.2,&[radius],&[material,Material::default()]).unwrap();
        let dim=forward.value.nrows();
        let optical=forward.value.trace().re+forward.value.norm_squared();
        prop_assert!(optical.abs()<1e-10);
        let g=DMatrix::identity(dim,dim)+&forward.value*Complex::new(2.0,0.0);
        let gradient=forward.pullback(&g).unwrap();
        prop_assert!(gradient.kzs.iter().all(|g|g.abs()<1e-9) && gradient.k0.abs()<1e-9);
        prop_assert!(gradient.layers.sizes.iter().all(|g|g.abs()<1e-9));
        prop_assert!(gradient.layers.epsilon.iter().chain(&gradient.layers.mu).chain(&gradient.layers.kappa).all(|g|g.re.abs()<1e-9));
    }

    #[test]
    fn cylindrical_translation_scale_identity(m in -7_i32..8, mu in -7_i32..8, x in -1.0_f64..1.0, y in 0.5_f64..2.0, outgoing in any::<bool>()) {
        let k=Complex::new(1.2,0.1);
        let kz=0.3;
        let to=crate::cylwaves::Mode{kz,m:mu,pol:1};
        let source=crate::cylwaves::Mode{kz,m,pol:1};
        let position=[x,y,0.4];
        let radial=if outgoing {Radial::Outgoing}else{Radial::Regular};
        let jet=crate::cylwaves::translate(to,source,k,position,radial).unwrap();
        let spatial:Complex=jet.position.iter().zip(position).map(|(g,p)|g*p).sum();
        prop_assert!((spatial-k*jet.k-kz*jet.kz).norm()<1e-9*(1.0+jet.value.norm()));
        let scaled=crate::cylwaves::translate(crate::cylwaves::Mode{kz:kz/1.3,..to},crate::cylwaves::Mode{kz:kz/1.3,..source},k/1.3,position.map(|v|v*1.3),radial).unwrap();
        prop_assert!((scaled.value-jet.value).norm()<1e-10*(1.0+jet.value.norm()));
    }

    #[test]
    fn plane_wave_maxwell_and_origin_reconstruction(x in -1.0_f64..1.0, y in -1.0_f64..1.0, z in 0.5_f64..2.0, pol in 0_u8..2, helicity in any::<bool>()) {
        let vector=[x,y,z].map(|v|v*Complex::new(1.0,0.1));
        let k=vector.iter().map(|v|v*v).sum::<Complex>().sqrt();
        let polarization=crate::plane::polarization(vector,pol,helicity).unwrap();
        let dot:Complex=polarization.iter().zip(vector).map(|(e,k)|e*k).sum();
        prop_assert!(dot.norm()<1e-12);
        if helicity {
            let [x,y,z]=vector;
            let [ex,ey,ez]=polarization;
            let cross=[y*ez-z*ey,z*ex-x*ez,x*ey-y*ex];
            for (curl,e) in cross.iter().zip(polarization) {
                prop_assert!((Complex::i()*curl-(2.0*f64::from(pol)-1.0)*k*e).norm()<1e-12);
            }
        }
        let modes=(-1..=1).flat_map(|m|(0..2).map(move |pol|(0,Mode{l:1,m,pol}))).collect();
        let basis=crate::basis::Basis{modes,positions:vec![[0.0;3]]};
        let coefficients=crate::plane::spherical(&basis,vector,pol,helicity).unwrap();
        let field=crate::fields::field(basis,coefficients,vec![[0.0;3]],[k,k],helicity,Radial::Regular).unwrap();
        for (actual,expected) in field.value.first().unwrap().iter().zip(polarization) {
            prop_assert!((*actual-expected).norm()<1e-11);
        }
    }

}
