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
    fn periodic_conversion_scale_adjoint(k in 1.1_f64..2.0, kz in -0.7_f64..0.7, period in 1.2_f64..2.2, x in -0.3_f64..0.3, helicity in any::<bool>()) {
        let source=crate::basis::Basis{modes:(1..=3).flat_map(|l|(-l..=l).flat_map(move|m|(0..2).map(move|pol|(0,Mode{l,m,pol})))).collect(),positions:vec![[x,0.1,-0.2]]};
        let destination=crate::cylwaves::Basis{modes:(-3..=3).flat_map(|m|(0..2).map(move|pol|(0,crate::cylwaves::Mode{kz,m,pol}))).collect(),positions:vec![[0.2,-0.1,0.1]]};
        let ks=[Complex::new(k,0.1);2];
        let (value,residual)=crate::conversion::periodic_spherical_to_cylindrical(destination.clone(),source.clone(),ks,period,helicity).unwrap();
        let g=DMatrix::from_element(value.nrows(),value.ncols(),Complex::new(0.2,0.1));
        let gradient=residual.pullback(&g).unwrap();
        let spatial:f64=gradient.expansion.destination.iter().flatten().zip(destination.positions.iter().flatten()).chain(gradient.expansion.source.iter().flatten().zip(source.positions.iter().flatten())).map(|(g,x)|g*x).sum();
        let spectral:f64=gradient.expansion.ks.iter().zip(ks).map(|(g,k)|(g.conj()*k).re).sum();
        let axial:f64=gradient.kz.iter().map(|g|g*kz).sum();
        prop_assert!((spatial+period*gradient.period-spectral-axial).abs()<1e-10);
        let scaled_source=crate::basis::Basis{positions:source.positions.iter().map(|p|p.map(|x|x*1.7)).collect(),..source};
        let scaled_destination=crate::cylwaves::Basis{positions:destination.positions.iter().map(|p|p.map(|x|x*1.7)).collect(),modes:destination.modes.iter().map(|&(p,m)|(p,crate::cylwaves::Mode{kz:m.kz/1.7,..m})).collect()};
        let (scaled,_)=crate::conversion::periodic_spherical_to_cylindrical(scaled_destination,scaled_source,ks.map(|k|k/1.7),period*1.7,helicity).unwrap();
        prop_assert!((value-scaled).norm()<1e-10);
    }

    #[test]
    fn compact_layer_scale_adjoint(k in 1.1_f64..2.0, d in 0.1_f64..0.8, axis in 0_usize..3) {
        let ks=vec![[Complex::new(k,0.0);2],[Complex::new(1.7*k,0.0),Complex::new(1.8*k,0.0)],[Complex::new(k,0.0);2]];
        let zs=[Complex::new(1.0,0.0),Complex::new(0.7,0.0),Complex::new(1.0,0.0)];
        let q=vec![[0.1,0.2],[0.3,0.2]];
        let (value,residual)=crate::layers::stack(ks.clone(),&zs,q.clone(),&[d],axis).unwrap();
        let g=value.iter().map(|_|std::array::from_fn(|_|DMatrix::from_element(2,2,Complex::new(0.2,0.1)))).collect();
        let gradient=residual.pullback(g,false).unwrap();
        let spectral:f64=gradient.ks.iter().flatten().zip(ks.iter().flatten()).map(|(g,k)|(g.conj()*k).re).sum();
        let transverse:f64=gradient.q.iter().flatten().zip(q.iter().flatten()).map(|(g,q)|g*q).sum();
        let spatial:f64=gradient.thickness.iter().map(|g|g*d).sum();
        prop_assert!((spatial-spectral-transverse).abs()<1e-10);
        let (scaled,_)=crate::layers::stack(ks.iter().map(|v|v.map(|k|k/1.7)).collect(),&zs,q.iter().map(|v|v.map(|q|q/1.7)).collect(),&[d*1.7],axis).unwrap();
        for (a,b) in value.iter().flatten().zip(scaled.iter().flatten()) {prop_assert!((a-b).norm()<1e-12);}
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // Two material/helicity slots and four blocks.
    fn cartesian_interface_identity_and_adjoint(k in 1.1_f64..2.0, q in 0.1_f64..0.5, axis in 0_usize..3) {
        let ks=[[Complex::new(k,0.1),Complex::new(k+0.2,0.15)];2];
        let z=[Complex::new(0.8,0.03);2];
        let residual=crate::smatrix::interface(ks,z,[q,0.2],axis).unwrap();
        let identity=DMatrix::identity(2,2);
        for (b,value) in residual.value.iter().enumerate() {
            let expected=if b==0 || b==3 {identity.clone()}else{DMatrix::zeros(2,2)};
            prop_assert!((value-expected).norm()<1e-12);
        }
        let g=std::array::from_fn(|b|DMatrix::from_element(2,2,Complex::new(0.2*(f64::from(u32::try_from(b).unwrap())+1.0),0.1)));
        let (gk,gz,gq)=residual.pullback(&g,false).unwrap();
        for (a,b) in gk[0].iter().zip(gk[1]) {prop_assert!((a+b).norm()<1e-12);}
        prop_assert!((gz[0]+gz[1]).norm()<1e-12);
        prop_assert!(gq.iter().all(|v|v.abs()<1e-12));
    }

    #[test]
    fn cylindrical_plane_expansion_scale_adjoint(kx in 0.1_f64..0.7, x in -0.3_f64..0.3, helicity in any::<bool>()) {
        let basis=crate::cylwaves::Basis{modes:(-3..=3).flat_map(|m|(0..2).map(move|pol|(0,crate::cylwaves::Mode{kz:0.0,m,pol}))).collect(),positions:vec![[x,0.1,0.2]]};
        let vectors=vec![[Complex::new(kx,0.1),Complex::new(0.3,0.2),Complex::default()]];
        let (value,residual)=crate::plane::expansion(basis.clone(),vectors.clone(),vec![1],helicity).unwrap();
        let g=DMatrix::from_element(value.nrows(),1,Complex::new(0.2,0.1));
        let gradient=residual.pullback(&g,false).unwrap();
        let spatial:f64=gradient.origins.iter().flatten().zip(basis.positions.iter().flatten()).map(|(g,x)|g*x).sum();
        let spectral:f64=gradient.vectors.iter().flatten().zip(vectors.iter().flatten()).map(|(g,k)|(g.conj()*k).re).sum();
        prop_assert!((spatial-spectral).abs()<1e-10);
        let scaled=crate::cylwaves::Basis{positions:basis.positions.iter().map(|p|p.map(|x|x*1.7)).collect(),..basis};
        let (other,_)=crate::plane::expansion(scaled,vectors.iter().map(|v|v.map(|k|k/1.7)).collect(),vec![1],helicity).unwrap();
        prop_assert!((value-other).norm()<1e-10);
    }

    #[test]
    fn cylindrical_channels_scale_and_period_adjoint(k in 1.0_f64..2.0, period in 1.3_f64..2.2, x in -0.3_f64..0.3) {
        let basis=crate::cylwaves::Basis{modes:(-3..=3).flat_map(|m|(0..2).map(move|pol|(0,crate::cylwaves::Mode{kz:0.0,m,pol}))).collect(),positions:vec![[x,0.1,0.2]]};
        let q=vec![[0.0,0.2],[0.0,2.5]];
        let ks=[Complex::new(k,0.1);2];
        let forward=crate::channels::cylindrical(basis.clone(),ks,q.clone(),vec![0,1],period,true).unwrap();
        let g=DMatrix::from_element(forward.value.nrows(),2,Complex::new(0.2,0.1));
        let d=basis.modes.len();
        let emitted=g.rows(2*d,2*d).dotc(&forward.value.rows(2*d,2*d)).re;
        let scaled=crate::cylwaves::Basis{positions:basis.positions.iter().map(|p|p.map(|x|x*1.7)).collect(),..basis.clone()};
        let other=crate::channels::cylindrical(scaled,ks.map(|k|k/1.7),q.iter().map(|q|q.map(|v|v/1.7)).collect(),vec![0,1],period*1.7,true).unwrap();
        prop_assert!((&forward.value-other.value).norm()<1e-10);
        let gradient=forward.pullback(&g,false).unwrap();
        prop_assert!((gradient.area*period+emitted).abs()<1e-10);
        let spatial:f64=gradient.positions.iter().flatten().zip(basis.positions.iter().flatten()).map(|(g,p)|g*p).sum();
        let spectral:f64=gradient.ks.iter().zip(ks).map(|(g,k)|(g.conj()*k).re).sum();
        let transverse:f64=gradient.q.iter().flatten().zip(q.iter().flatten()).map(|(g,q)|g*q).sum();
        prop_assert!((spatial-spectral-transverse+period*gradient.area).abs()<1e-10);
    }

    #[test]
    fn plane_expansion_scale_adjoint(kx in 0.1_f64..0.7, kz in -1.0_f64..1.0, x in -0.3_f64..0.3, helicity in any::<bool>(), complex_transverse in any::<bool>()) {
        let basis=crate::basis::Basis{modes:(1..=4).flat_map(|l|(-l..=l).flat_map(move|m|(0..2).map(move|pol|(0,Mode{l,m,pol})))).collect(),positions:vec![[x,0.1,0.2]]};
        let vector=if complex_transverse {[Complex::new(kx,1.0),Complex::new(0.2,0.3),Complex::new(1.3,-0.8)]}else{[Complex::new(kx,0.0),Complex::new(0.2,0.0),Complex::new(kz,0.1)]};
        let vectors=vec![vector];
        let (value,residual)=crate::plane::expansion(basis.clone(),vectors.clone(),vec![1],helicity).unwrap();
        let k=vector.iter().map(|v|v*v).sum::<Complex>().sqrt();
        let reconstructed=crate::fields::field(basis.clone(),value.as_slice().to_vec(),basis.positions.clone(),[k,k],helicity,Radial::Regular).unwrap();
        let (expected,_)=crate::plane::field(vectors.clone(),vec![1],basis.positions.clone(),Some(vec![Complex::new(1.0,0.0)]),helicity).unwrap();
        for (a,b) in reconstructed.value.iter().flatten().zip(expected.iter()){prop_assert!((*a-b).norm()<1e-12);}
        let g=DMatrix::from_element(value.nrows(),1,Complex::new(0.2,0.1));
        let gradient=residual.pullback(&g,false).unwrap();
        let spatial:f64=gradient.origins.iter().flatten().zip(basis.positions.iter().flatten()).map(|(g,x)|g*x).sum();
        let spectral:f64=gradient.vectors.iter().flatten().zip(vectors.iter().flatten()).map(|(g,k)|(g.conj()*k).re).sum();
        prop_assert!((spatial-spectral).abs()<1e-10);
        let scaled=crate::basis::Basis{positions:basis.positions.iter().map(|p|p.map(|x|1.7*x)).collect(),..basis};
        let (other,_)=crate::plane::expansion(scaled,vectors.iter().map(|v|v.map(|k|k/1.7)).collect(),vec![1],helicity).unwrap();
        prop_assert!((value-other).norm()<1e-10);
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // Fixed two-mode matrix and Cartesian triples.
    fn plane_field_operator_and_scale_adjoint(kx in 0.1_f64..0.7, x in -0.5_f64..0.5, helicity in any::<bool>()) {
        let vectors=vec![[Complex::new(kx,0.0),Complex::new(0.2,0.0),Complex::new(1.3,0.1)],[Complex::new(1.5,0.0),Complex::new(-0.1,0.0),Complex::new(0.0,0.2)]];
        let points=vec![[x,0.2,0.1],[0.3,-0.1,0.2]];
        let coefficients=vec![Complex::new(0.7,0.1),Complex::new(-0.2,0.3)];
        let (matrix,operator)=crate::plane::field(vectors.clone(),vec![0,1],points.clone(),None,helicity).unwrap();
        let (value,weighted)=crate::plane::field(vectors.clone(),vec![0,1],points.clone(),Some(coefficients.clone()),helicity).unwrap();
        prop_assert!((&matrix*nalgebra::DVector::from_vec(coefficients.clone())-&value).norm()<1e-12);
        let g=DMatrix::from_element(6,1,Complex::new(0.2,0.1));
        let full_g=DMatrix::from_fn(6,2,|i,j|g[(i,0)]*coefficients[j].conj());
        let a=operator.pullback(&full_g,false).unwrap();
        let b=weighted.pullback(&g,false).unwrap();
        for (a,b) in a.points.iter().flatten().zip(b.points.iter().flatten()) {prop_assert!((a-b).abs()<1e-12);}
        for (a,b) in a.vectors.iter().flatten().zip(b.vectors.iter().flatten()) {prop_assert!((*a-b).norm()<1e-12);}
        let spatial:f64=b.points.iter().flatten().zip(points.iter().flatten()).map(|(g,p)|g*p).sum();
        let spectral:f64=b.vectors.iter().flatten().zip(vectors.iter().flatten()).map(|(g,k)|(g.conj()*k).re).sum();
        prop_assert!((spatial-spectral).abs()<1e-12);
        let amplitude:f64=b.coefficients.iter().zip(&coefficients).map(|(g,c)|(g.conj()*c).re).sum();
        prop_assert!((amplitude-g.dotc(&value).re).abs()<1e-12);
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // Fixed Cartesian triples and two helicities.
    fn conversion_field_and_adjoint(m in -2_i32..3, kz in -0.7_f64..0.7, x in -0.3_f64..0.3, helicity in any::<bool>()) {
        let destination=crate::basis::Basis{modes:(1..=8).flat_map(|l|(-l..=l).flat_map(move|m|(0..2).map(move|pol|(0,Mode{l,m,pol})))).collect(),positions:vec![[x,0.1,-0.2]]};
        let source=crate::cylwaves::Basis{modes:vec![(0,crate::cylwaves::Mode{kz,m,pol:1})],positions:vec![[0.2,-0.1,0.1]]};
        let ks=[Complex::new(1.3,0.1);2];
        let converted=crate::conversion::cylindrical_to_spherical(destination.clone(),source.clone(),ks,helicity).unwrap();
        let points=vec![[x+0.1,0.2,-0.1]];
        let actual=crate::fields::field(destination.clone(),converted.value.as_slice().to_vec(),points.clone(),ks,helicity,Radial::Regular).unwrap();
        let expected=crate::fields::field(source.clone(),vec![Complex::new(1.0,0.0)],points,ks,helicity,Radial::Regular).unwrap();
        for (a,b) in actual.value.iter().flatten().zip(expected.value.iter().flatten()) {prop_assert!((*a-b).norm()<1e-10);}
        let g=DMatrix::from_element(converted.value.nrows(),1,Complex::new(0.2,0.1));
        let gradient=converted.pullback(&g).unwrap();
        for a in 0..3 {prop_assert!((gradient.destination[0][a]+gradient.source[0][a]).abs()<1e-12);}
        let h=1e-5;
        let shifted=|sign:f64|{
            let mut to=destination.clone(); to.positions[0][0]+=sign*h;
            crate::conversion::cylindrical_to_spherical(to,source.clone(),ks,helicity).unwrap().value
        };
        let numeric=g.dotc(&((shifted(1.0)-shifted(-1.0))/Complex::new(2.0*h,0.0))).re;
        prop_assert!((gradient.destination[0][0]-numeric).abs()<1e-7);
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // Fixed Cartesian samples and six dipole modes.
    fn field_operator_amplitude_contraction(values in prop::collection::vec(-0.5_f64..0.5,12)) {
        let basis=crate::basis::Basis{modes:(-1..=1).flat_map(|m|(0..2).map(move|pol|(0,Mode{l:1,m,pol}))).collect(),positions:vec![[0.1,0.2,0.3]]};
        let coefficients:Vec<_>=values.as_chunks::<2>().0.iter().map(|&[re,im]|Complex::new(re,im)).collect();
        let points=vec![[0.7,0.5,0.3],[0.8,-0.3,0.2]];
        let ks=[Complex::new(1.2,0.1);2];
        let (matrix,operator)=crate::fields::operator(basis.clone(),points.clone(),ks,true,Radial::Outgoing).unwrap();
        let field=crate::fields::field(basis,coefficients.clone(),points,ks,true,Radial::Outgoing).unwrap();
        let contracted=&matrix*nalgebra::DVector::from_vec(coefficients.clone());
        for (a,b) in contracted.iter().zip(field.value.iter().flatten()) {prop_assert!((*a-b).norm()<1e-12);}
        let g=vec![[Complex::new(0.2,0.1);3];2];
        let full_g=DMatrix::from_fn(6,6,|i,j|g[i/3][i%3]*coefficients[j].conj());
        let a=operator.pullback(&full_g).unwrap();
        let b=field.pullback(&g).unwrap();
        prop_assert!(a.coefficients.is_empty());
        for (a,b) in a.points.iter().flatten().chain(a.origins.iter().flatten()).zip(b.points.iter().flatten().chain(b.origins.iter().flatten())) {prop_assert!((a-b).abs()<1e-12);}
        for (a,b) in a.ks.iter().zip(b.ks) {prop_assert!((*a-b).norm()<1e-12);}
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // Fixed triples of Euler angles.
    fn rotation_group_and_adjoint(l in 1_i32..8, theta in -6.0_f64..6.0, phi in -3.0_f64..3.0, psi in -3.0_f64..3.0) {
        let basis=crate::basis::Basis{modes:(-l..=l).map(|m|(0,Mode{l,m,pol:1})).collect(),positions:vec![[0.0;3]]};
        let angles=[phi,theta,psi];
        let rotation=crate::rotation::spherical(basis.clone(),basis.clone(),angles).unwrap();
        let inverse=crate::rotation::spherical(basis.clone(),basis.clone(),[-psi,-theta,-phi]).unwrap();
        let n=rotation.value.nrows();
        prop_assert!((&rotation.value*&inverse.value-DMatrix::identity(n,n)).norm()<1e-11);
        let g=DMatrix::from_fn(n,n,|i,j|Complex::new(if i==j {0.3} else {-0.2},0.1));
        let analytic=rotation.pullback(&g).unwrap();
        let direction=[0.2,-0.1,0.3];
        let h=1e-5;
        let shifted=|sign:f64|crate::rotation::spherical(basis.clone(),basis.clone(),std::array::from_fn(|i|angles[i]+sign*h*direction[i])).unwrap();
        let numeric=g.dotc(&((shifted(1.0).value-shifted(-1.0).value)/Complex::new(2.0*h,0.0))).re;
        prop_assert!((numeric-analytic.iter().zip(direction).map(|(g,d)|g*d).sum::<f64>()).abs()<1e-8);
        let a=crate::rotation::wigner_d(l,theta).unwrap();
        let b=crate::rotation::wigner_d(l,phi).unwrap();
        let ab=crate::rotation::wigner_d(l,theta+phi).unwrap();
        prop_assert!((a*b-ab).norm()<1e-11);
    }

    #[test]
    fn cylindrical_field_maxwell(m in -7_i32..8, pol in 0_u8..2, kz in -0.7_f64..0.7, x in -1.0_f64..1.0, y in 0.2_f64..1.2, outgoing in any::<bool>()) {
        let k=Complex::new(1.3,0.1);
        let radial=if outgoing {Radial::Outgoing} else {Radial::Regular};
        for position in [[x,y,0.3],[if outgoing {x} else {0.0},if outgoing {y} else {0.0},0.3]] {
            let mode=crate::cylwaves::Mode{kz,m,pol};
            let wave=crate::fields::cylindrical_wave(mode,k,position,true,radial).unwrap();
            let [jx,jy,jz]=wave.position;
            let curl=[jz[1]-jy[2],jx[2]-jz[0],jy[0]-jx[1]];
            let scale=1.0+wave.value.iter().map(Complex::norm_sqr).sum::<f64>().sqrt();
            for (curl,e) in curl.iter().zip(wave.value) {prop_assert!((*curl-(2.0*f64::from(pol)-1.0)*k*e).norm()<1e-10*scale);}
            prop_assert!((jx[0]+jy[1]+jz[2]).norm()<1e-10*scale);
        }
    }

    #[test]
    fn spherical_channel_scale_and_adjoint(k in 1.0_f64..2.0, qx in 0.1_f64..0.4, area in 2.0_f64..4.0, scale in 0.5_f64..2.0, helicity in any::<bool>()) {
        let modes=(1..=3).flat_map(|l|(-l..=l).flat_map(move |m|(0..2).map(move |pol|(0,Mode{l,m,pol})))).collect();
        let basis=crate::basis::Basis{modes,positions:vec![[0.1,0.2,-0.3]]};
        let ks=[Complex::new(k,0.1);2];
        let q=vec![[qx,0.2],[3.2,-0.1]];
        let forward=crate::channels::spherical(basis.clone(),ks,q.clone(),vec![0,1],area,helicity).unwrap();
        let scaled_basis=crate::basis::Basis{positions:basis.positions.iter().map(|r|r.map(|x|x*scale)).collect(),..basis.clone()};
        let scaled=crate::channels::spherical(scaled_basis,ks.map(|k|k/scale),q.iter().map(|q|q.map(|x|x/scale)).collect(),vec![0,1],area*scale*scale,helicity).unwrap();
        prop_assert!((&forward.value-scaled.value).norm()<1e-10*(1.0+forward.value.norm()));
        let g=DMatrix::from_fn(forward.value.nrows(),2,|i,j|Complex::new(if i%3==j {0.3}else{-0.2},0.1));
        let gradient=forward.pullback(&g,false).unwrap();
        let spatial:f64=gradient.positions.iter().flatten().zip(basis.positions.iter().flatten()).map(|(g,r)|g*r).sum();
        let spectral:f64=gradient.ks.iter().zip(ks).map(|(g,k)|(g.conj()*k).re).sum();
        let transverse:f64=gradient.q.iter().flatten().zip(q.iter().flatten()).map(|(g,q)|g*q).sum();
        prop_assert!((spatial-spectral-transverse+2.0*area*gradient.area).abs()<1e-8);
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // Fixed four channel blocks.
    fn radiation_multilinearity_adjoint(values in prop::collection::vec(-0.5_f64..0.5,98)) {
        let mut values=values.as_chunks::<2>().0.iter().map(|&[re,im]|Complex::new(re,im));
        let response=DMatrix::from_iterator(3,3,values.by_ref().take(9));
        let channels:crate::smatrix::Blocks=std::array::from_fn(|_|DMatrix::from_iterator(3,2,values.by_ref().take(6)));
        let g:crate::smatrix::Blocks=std::array::from_fn(|_|DMatrix::from_iterator(2,2,values.by_ref().take(4)));
        let forward=crate::smatrix::from_array(response.clone(),channels.clone()).unwrap();
        let mut scattered=forward.value.clone();
        for b in [0,3] {scattered[b]-=DMatrix::identity(2,2);}
        let loss:f64=g.iter().zip(scattered).map(|(g,v)|g.dotc(&v).re).sum();
        let (gt,gc)=forward.pullback(&g).unwrap();
        prop_assert!((gt.dotc(&response).re-loss).abs()<1e-12);
        for group in [0,2] {
            let pairing:f64=(group..group+2).map(|i|gc[i].dotc(&channels[i]).re).sum();
            prop_assert!((pairing-loss).abs()<1e-12);
        }
        let empty=crate::smatrix::from_array(DMatrix::zeros(3,3),channels).unwrap();
        prop_assert!(empty.value[1].norm()==0.0 && empty.value[2].norm()==0.0);
        prop_assert!((&empty.value[0]-DMatrix::identity(2,2)).norm()==0.0);
        let (_,gc)=empty.pullback(&g).unwrap();
        prop_assert!(gc.iter().all(|a|a.norm()==0.0));
    }

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
