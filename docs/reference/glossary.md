---
description: One name per concept, with the treams-rs Python name, the native name, the Rust name and the treams name side by side.
---

# Glossary

Each row names one concept in the Python package `treams_rs`, in its native
extension `treams_rs._native`, in the Rust crate `treams-core` and in treams.
Rust paths start at the crate root, so `sw::Mode` is `treams_core::sw::Mode`.
Python code never needs `_native`; its names appear in type annotations and
error messages.

## Records and gradients

A record is a function that returns a value and a reusable context. The context stores what is needed to compute derivatives later: `context.pullback(g)` takes the gradient `g` of a real-valued loss with respect to the value and returns the gradients with respect to the inputs, one for each differentiable input, in the order of the arguments. Gradients follow the convention dL = Re Σ conj(g)·dx.

In the Rust core, a function returns `(value, XResidual)`, and
`XResidual::pullback(&self, cotangent)` returns the input gradients as
`XGradient`.

| Term | Meaning |
| --- | --- |
| record | a function that returns a value and a context: `diff.mie(...)`, `InteractionFactor.record(...)` |
| context | the reusable object a record returns; its pullback and pushforward share saved derivative data |
| residual | the Rust form of a context: the data a forward function keeps for its gradient |
| pullback | the map from the gradient with respect to the output to the gradients with respect to the inputs |
| cotangent | the gradient of a real-valued loss with respect to one value, the `g` of a pullback |
| pairing | dL = Re Σ conj(g)·dx, which fixes the meaning of complex gradients |
| jet | a value computed together with its derivatives (`numerics::Jet` in Rust) |

## Concepts

| Concept | treams-rs Python | native (`_native`) | Rust (`treams-core`) | treams |
| --- | --- | --- | --- | --- |
| Degree and order | `l`, `m`, `lmax`, `mmax`; `degree=` and `order=` in `diff` and the framework adapters | `lmax`, `mmax`; `degree`, `order` in the angular records | fields `l`, `m`; bounds `lmax`, `mmax`; degree means `l`, order means `m` | `l`, `m`, `lmax`, `mmax` |
| Label bounds | 128 for degrees and cylindrical orders (see [conventions](../coming-from-treams/conventions.md#label-bounds)) | the same bounds | `MAX_DEGREE = 128`, `special::MAX_ORDER = 256`, `special::MAX_LABEL = 260` | none |
| Radial kind of multipole waves | `kind='regular'` or `'singular'` on physics objects (`modetype` delegates); `singular: bool` in `diff`; `modetype` in the numerical namespaces | `singular: bool` (expansions, fields, translation records, `ebcm_qmat`) | `special::Radial::{Regular, Singular}` | `modetype='regular'` or `'singular'`; `singular=True` |
| Special-function selector | `function='j'`, `'y'`, `'h1'`, `'h2'` in `diff.bessel`; `'legendre'`, `'pi'`, `'tau'` in `diff.angular` | `function: str` of `bessel_record` and `angular_record` | `special::Bessel::{J, Y, H1, H2}`, `special::Angular::{Legendre, Pi, Tau}` | `jv`, `yv`, `hankel1`, `hankel2`, `lpmv`, `pi_fun`, `tau_fun` |
| Plane-wave port direction | `kind='up'` or `'down'`; `modetype=` in `diff.smatrix_tr` | `direction: int` of `smatrix_tr` and `smatrix_tr_value` (0 up, 1 down) | `smatrix::TrPorts.direction`: 0 up, 1 down | `modetype='up'` or `'down'` |
| Planar sides | `negative_medium` (below) and `positive_medium` (above); `side='negative'` or `'positive'`; `SMatrix.material` is `(positive, negative)` | S-matrix arrays of shape `(2, 2, n, n)`, blocks `[S00, S01, S10, S11]` | `smatrix::Blocks`; `fresnel` and `interface` take `(negative, positive)` | the materials list of `SMatrices`; `illuminate(up, down)` |
| Polarization convention | `polarization='helicity'` or `'parity'` on physics objects; `poltype=` in the numerical namespaces, `operators` and `diff`; `with_polarization` (`changepoltype` delegates) | `helicity: bool` | `helicity: bool` (`true` means helicity) | `poltype`, `config.POLTYPE`, `changepoltype` |
| Polarization index and plane-wave state | `pol` (0 or 1) on bases; `plane_wave(pol=...)` also takes `'positive_helicity'`, `'negative_helicity'`, two amplitudes or three field components; `polarizations` label arrays in `diff` | `pol`; `polarizations: list[int]` | `pol: u8` (helicity 1 is positive) | `pol` |
| Mode labels | `pidx`, `l`, `m`, `pol` on `SphericalBasis`; `pidx`, `kz`, `m`, `pol` on `CylindricalBasis` | tuples `(pidx, l, m, pol)` and `(pidx, kz, m, pol)` | `sw::Mode { l, m, pol }`, `cw::Mode { kz, m, pol }`; basis entries `(position index, mode)` | `pidx`, `l`, `m`, `kz`, `pol` |
| Multipole and plane-wave bases | `SphericalBasis`, `CylindricalBasis`, `PlaneWaveBasis` (unit directions), `PlaneWavePorts` (transverse wavevector components) | mode tuples plus `positions` | `basis::Basis<M>`, `sw::Basis`, `cw::Basis`, `basis::MultipoleBasis` | `SphericalWaveBasis`, `CylindricalWaveBasis`, `PlaneWaveBasisByUnitVector`, `PlaneWaveBasisByComp` |
| Expansion centres | `positions` of bases and clusters; `destination_positions`, `source_positions` in `diff` and `advect` | `positions`, `destination_positions` and `source_positions` | `positions` (bases and gradient fields) | `positions` |
| Two bases of one matrix | `destination` and `source` in `diff` and the framework adapters; the treams names `out` and `in_` in `ebcm.qmat` and `translate_periodic` | `destination` and `source` | `destination` and `source`; displacement = destination - source | `out` and `in_`; `lambda_`, `mu`, `pol` (destination) and `l`, `m`, `qol` (source) |
| Axial wavenumbers | `kz`, one value per mode (`basis.kz`); `kzs`, the distinct values (`diff.cylinder(kzs, ...)`) | `cylinder(kzs, ...)`; `pullback_axial` returns one gradient per distinct `kz`, in increasing order | `cw::Mode.kz` | `basis.kz` (per mode), `TMatrixC.cylinder(kzs, ...)` (distinct) |
| Material and medium | `Material`; `material` (particle or layer), `medium` (surrounding space), `negative_medium`, `positive_medium`; `.material` delegates to `medium` | `epsilon`, `mu`, `kappa` arrays | `coeffs::Material { epsilon, mu, kappa }`; layers from the inside out, then the surrounding medium | `Material`; lists `[..., embedding]` |
| T-matrices | `TMatrix`, `CylindricalTMatrix`; `sphere_tmatrix`, `cylinder_tmatrix` and their multilayer forms; `diff.sphere`, `diff.cylinder` | `sphere`, `cylinder`, `mie`, `mie_cyl` | `tmatrix::sphere`, `tmatrix::cylinder`, `coeffs::mie`, `coeffs::mie_cyl` | `TMatrix`, `TMatrixC`, `TMatrix.sphere`, `TMatrixC.cylinder`, `coeffs.mie`, `coeffs.mie_cyl` |
| Waves | `Wave` (multipole coefficients), `PlaneWave`; `spherical_wave`, `cylindrical_wave`, `plane_wave` | `field`, `cylindrical_field`, `plane_field` | `fields::field`, `pw::field` | `PhysicsArray` from `spherical_wave`, `cylindrical_wave`, `plane_wave` |
| Periodic response and wave | `solve_periodic(...)` returns `PeriodicResponse`; its `scatter` returns `PeriodicWave` | `lattice_expansion`, `cylindrical_lattice_expansion` | `sw::lattice_expansion`, `cw::lattice_expansion`, `lattice::BlochLattice` | `TMatrix.latticeinteraction.solve` |
| Cluster and factor | `Cluster` (`solve`, `scatter`, `factor`); `Cluster.factor()` returns `ScatteringFactor`; `diff.particle_cluster`, `diff.sphere_cluster`, `diff.factor_interaction`, `diff.illuminate` | `particle_cluster`, `cylindrical_particle_cluster`, `sphere_cluster`, `sphere_cluster_factor`, `InteractionFactor` | `cluster::particle_cluster`, `cluster::sphere_cluster`, `cluster::sphere_cluster_factor`, `cluster::InteractionFactor` | `TMatrix.cluster` with `interaction.solve` |
| Matrix-free sphere cluster | `iterative.SphereCluster` (`solve`, `scatter`, `record`), `iterative.IterativeContext` | `IterativeSphereCluster`, `IterativeContext` | `cluster::IterativeSphereCluster`, `IterativeResidual`, `IterativeGradient`; `linalg::GmresOptions`, `Convergence` | none |
| S-matrix network and block | `SMatrix` (the network), `ScatteringBlock` (one block); `tr.SMatrices` raises an `AttributeError` naming `SMatrix` | `(2, 2, n, n)` arrays | `smatrix::Blocks`, the four blocks of one network | `SMatrices` (the network), `SMatrix` (one block) |
| S-matrix operations | `cascade`, `scatter`, `power`, `circular_dichroism`, `transfer_matrix`, `bands` (`add`, `illuminate`, `tr`, `cd`, `periodic`, `bands_kz` delegate); `diff.smatrix_*` | `smatrix_add`, `smatrix_illuminate`, `smatrix_illuminate_value`, `smatrix_periodic`, `bands`, `smatrix_from_array`, `smatrix_tr`, `smatrix_tr_value` | `smatrix::add`, `illuminate`, `illuminate_value`, `periodic`, `bands`, `from_array`, `tr`, `tr_value` | `SMatrices.add`, `illuminate`, `periodic`, `bands_kz`, `from_array`, `tr`, `cd` |
| Interface, propagation and layer stack | `interface`, `slab`, `multilayer_slab`, `propagation`, `stack`; `diff.fresnel`, `diff.interface_coefficients`, `diff.propagation_matrix`, `diff.layer_stack` | `fresnel`, `interface_coefficients`, `propagation_matrix`, `layer_stack` | `smatrix::fresnel`, `interface`, `propagation`, `layer_stack` | `coeffs.fresnel`, `SMatrices.interface`, `propagation`, `stack` |
| Expansion between two bases | `in_basis` (`expand` delegates); `operators.expand`; `diff.expansion`, `diff.plane_expansion` | `expansion`, `cylindrical_expansion`, `cw_to_sw`, `plane_expansion`, `cylindrical_plane_expansion` | `sw::expansion`, `cw::expansion`, `cw::to_sw_matrix`, `pw::expansion` | `Expand`, `expand` |
| Lattice expansion | `sw.translate_periodic`, `cw.translate_periodic`; `operators.expandlattice`; `diff.lattice_expansion`, `diff.lattice_expansion_from_table` | `lattice_expansion`, `cylindrical_lattice_expansion`, `lattice_expansion_from_table` | `sw::lattice_expansion`, `cw::lattice_expansion`, `sw::lattice_expansion_from_table` | `ExpandLattice`, `expandlattice`, `translate_periodic` |
| Translation coefficient of one mode pair | `sw.translate`, `cw.translate`, `special.tl_vsw_A`, `tl_vsw_B`, `tl_vcw`; `diff.spherical_translation`, `diff.cylindrical_translation` | ufuncs `tl_vsw_*`, `tl_vcw`; `spherical_translation_record`, `cylindrical_translation_record` | `sw::polar_translation_array`, `cw::polar_translation`, `cw::tl_vcw`, `cw::translate` | `sw.translate`, `cw.translate`, `special.tl_vsw_*`, `special.tl_vcw` |
| Change of wave family | `cw.to_sw`, `pw.to_sw`, `pw.to_cw`, `sw.periodic_to_cw`, `sw.periodic_to_pw`, `cw.periodic_to_pw`; `diff.periodic_to_cw` | ufuncs `cw_to_sw_h`, `cw_to_sw_p`, `pw_to_sw_h`, `pw_to_sw_p`, `pw_to_cw`, `sw_periodic_to_cw_h`, `sw_periodic_to_cw_p`, `sw_periodic_to_pw_h`, `sw_periodic_to_pw_p`, `cw_periodic_to_pw`; `periodic_to_cw` | `cw::to_sw`, `cw::to_sw_matrix`, `pw::to_sw`, `pw::to_cw`, `sw::periodic_to_cw`, `channels::sw_periodic_to_pw`, `channels::cw_periodic_to_pw` | the same names as treams-rs Python |
| Rotation | `sw.rotate`, `cw.rotate`, `rotate` methods, `operators.rotate`; `diff.rotation` | ufuncs `sw_rotate`, `cw_rotate`; `rotation`, `cylindrical_rotation` | `rotation::sw_rotation`, `cw_rotation`, `sw_rotate`, `cw_rotate` | `sw.rotate`, `cw.rotate`, `Rotate` |
| Wigner functions | `special.wignersmalld`, `wignerd`, `wigner3j`; `diff.wignerd` | ufuncs `wignersmalld`, `wignerd`, `wigner3j`; `wignerd_record`, `wignerd_record_scalar`, `wigner3j_scalar` | `special::wigner_small_d`, `wigner_d`, `wigner_d_array`, `wigner3j` | `special.wignersmalld`, `wignerd`, `wigner3j` |
| Lattice sums | `lattice.lsum*`, `realsum*`, `recsum*`, `dsum*` with `kpar`, `a`, `r`, `eta`; `diff.lattice_sum(..., part=)` with `'full'`, `'real'`, `'reciprocal'`, `'direct'`; `Lattice`, `WaveVector` | `lattice_sum_record`, `LatticeSumContext` | `lattice::sum`, `sum_part`, `sum_array`, `BlochLattice`, `SumPart::{Full, Real, Reciprocal, Direct}`, `Family::{Spherical, Cylindrical}` | `lsum*(l, m, k, kpar, a, r, eta)` |
| Ewald split | `eta`, which divides a lattice sum into a real-space and a reciprocal-space part; `eta=0` selects it automatically | `eta` | `eta` | `eta` |
| EBCM matrix | `ebcm.qmat(..., radial_area_factor=True)`; `diff.ebcm_qmat` | `ebcm_qmat(..., singular, radial_area_factor)` | `ebcm::qmat`, `ebcm::Surface`, `QmatResidual` | `ebcm.qmat`, which omits the radial area factor |
| Channels of periodic arrays | `diff.spherical_channels(..., area)`, `diff.cylindrical_channels(..., period)` | `spherical_channels`, `cylindrical_channels` | `channels::spherical_channels`, `cylindrical_channels`, `ChannelGradient` | `sw.periodic_to_pw`, `cw.periodic_to_pw`, `SMatrices.from_array` |
| Special-function records | `diff.bessel`, `angular`, `incgamma`, `intkambe`, `coordinates`, `vector_coordinates`, `vector_wave`, `sph_harm` | `bessel_record`, `angular_record`, `incgamma_record`, `intkambe_record`, `coordinates_record`, `vector_coordinates_record`, `vector_wave_record` | `special::bessel_array`, `angular_array`, `incgamma_array`, `intkambe_array`, `special::coordinates::{point, vector}`, `vectorwaves::vector_wave_array` | `special.jv`, `lpmv`, `incgamma`, `intkambe`, `car2sph`, `vsw_*`, `sph_harm` |
| Records | `diff.*` functions return `(value, context)`; solver objects have `record` methods | functions returning `(value, <Name>Context)` | forward functions returning `(value, XResidual)` | none |
| Errors | `ValueError` | `ValueError` | `Error::{InvalidInput, SpecialFunction, NonFinite, NotConverged, Singular}` | NaN results or NumPy warnings |
| Framework adapters | `treams_rs.advect`, `treams_rs.jax`, `treams_rs.torch`, `treams_rs.autograd` | none | none | none |
