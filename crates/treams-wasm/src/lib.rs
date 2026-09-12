//! Browser bindings for double-precision multipole scattering and electric fields.
#![allow(clippy::indexing_slicing)] // Fixed-width records and helicity labels validated at entry.

use nalgebra::{DMatrix, DVector};
use treams_core::{
    Complex, basis::Basis, coeffs::Material, fields, plane, special::Radial, tmatrix, waves,
};
use wasm_bindgen::prelude::*;

fn complex(values: &[f64]) -> Result<Vec<Complex>, JsError> {
    if !values.len().is_multiple_of(2) || values.iter().any(|x| !x.is_finite()) {
        return Err(JsError::new("expected finite interleaved complex values"));
    }
    Ok(values
        .chunks_exact(2)
        .map(|v| Complex::new(v[0], v[1]))
        .collect())
}

fn cartesian(values: &[f64]) -> Result<Vec<[f64; 3]>, JsError> {
    if !values.len().is_multiple_of(3) || values.iter().any(|x| !x.is_finite()) {
        return Err(JsError::new("expected finite Cartesian xyz triples"));
    }
    Ok(values.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect())
}

fn interleaved(values: impl IntoIterator<Item = Complex>) -> Vec<f64> {
    values.into_iter().flat_map(|z| [z.re, z.im]).collect()
}

fn basis(lmax: u32, positions: Vec<[f64; 3]>) -> treams_core::Result<Basis> {
    let modes = waves::modes(lmax)?;
    Ok(Basis {
        modes: (0..positions.len())
            .flat_map(|p| modes.iter().map(move |&m| (p, m)))
            .collect(),
        positions,
    })
}

fn plane_vector(
    ks: [Complex; 2],
    direction: &[f64],
    helicity: u8,
) -> Result<[Complex; 3], JsError> {
    if helicity > 1 {
        return Err(JsError::new("expected helicity 0 or 1"));
    }
    let direction: [f64; 3] = direction
        .try_into()
        .map_err(|_| JsError::new("expected direction xyz"))?;
    let norm = direction[0].hypot(direction[1]).hypot(direction[2]);
    if norm == 0.0 || !norm.is_finite() {
        return Err(JsError::new(
            "plane-wave direction must have a finite nonzero norm",
        ));
    }
    Ok(direction.map(|d| ks[usize::from(helicity)] * (d / norm)))
}

fn vacuum(k0: f64) -> Result<[Complex; 2], JsError> {
    if !k0.is_finite() || k0 <= 0.0 {
        return Err(JsError::new("k0 must be finite and positive"));
    }
    Ok([Complex::new(k0, 0.0); 2])
}

fn cluster_inputs(
    radii: &[f64],
    epsilon: &[f64],
    positions: &[f64],
) -> Result<(Vec<Complex>, Vec<[f64; 3]>), JsError> {
    let epsilon = complex(epsilon)?;
    let positions = cartesian(positions)?;
    if radii.is_empty()
        || radii.len() != epsilon.len()
        || radii.len() != positions.len()
        || radii.iter().any(|r| !r.is_finite() || *r <= 0.0)
    {
        return Err(JsError::new(
            "radii, permittivities and positions must describe the same nonempty cluster with positive finite radii",
        ));
    }
    for (i, position) in positions.iter().enumerate() {
        for j in 0..i {
            let [x, y, z] = std::array::from_fn(|axis| position[axis] - positions[j][axis]);
            if x.hypot(y).hypot(z) <= radii[i] + radii[j] {
                return Err(JsError::new(
                    "spherical particles must not touch or overlap",
                ));
            }
        }
    }
    Ok((epsilon, positions))
}

/// Direct unit-amplitude plane-wave electric field in vacuum, evaluated once at
/// each world-space point, independently of any multipole expansion origins.
///
/// `direction` is a nonzero real xyz vector (normalized here); `helicity` is 0 or
/// 1. `points` packs xyz triples. The returned `Float64Array` contains
/// `[Ex.re, Ex.im, Ey.re, Ey.im, Ez.re, Ez.im]` for each point.
#[wasm_bindgen]
pub fn direct_plane_field(
    k0: f64,
    direction: &[f64],
    helicity: u8,
    points: &[f64],
) -> Result<Vec<f64>, JsError> {
    let k = plane_vector(vacuum(k0)?, direction, helicity)?;
    let polarization = plane::polarization(k, helicity, true)?;
    let values = cartesian(points)?
        .into_iter()
        .map(|p| plane::field_value(polarization, k, p.map(Complex::from)))
        .collect::<treams_core::Result<Vec<_>>>()?;
    Ok(interleaved(values.into_iter().flatten()))
}

/// Total electric-field intensity and analytic shape gradient for a sphere
/// cluster in vacuum under one unit-amplitude plane wave.
///
/// Inputs follow `ScatteringSystem.cluster`: `epsilon` packs complex pairs and
/// `positions` packs xyz triples. `direction` is normalized; `helicity` is 0 or
/// 1. `target` is exactly one fixed world-space xyz point outside every sphere.
/// Frequency, incident direction, permittivities and target stay fixed in the
/// derivative. The returned `Float64Array` is
/// `[J, dr_0, ..., dr_(N-1), dx_0, dy_0, dz_0, ..., dx_(N-1), dy_(N-1), dz_(N-1)]`,
/// where `J = sum(abs(E_incident + E_scattered)^2)` (unit incident intensity).
///
/// All numerical residuals are created and consumed within this call. The
/// position derivative includes coupling, incident phase and radiation origins.
#[wasm_bindgen]
pub fn cluster_target_gradient(
    lmax: u32,
    k0: f64,
    radii: &[f64],
    epsilon: &[f64],
    positions: &[f64],
    direction: &[f64],
    helicity: u8,
    target: &[f64],
) -> Result<Vec<f64>, JsError> {
    let (epsilon, positions) = cluster_inputs(radii, epsilon, positions)?;
    let ks = vacuum(k0)?;
    let k = plane_vector(ks, direction, helicity)?;
    let target: [f64; 3] = target
        .try_into()
        .map_err(|_| JsError::new("expected one target xyz point"))?;
    if target.iter().any(|x| !x.is_finite())
        || positions.iter().zip(radii).any(|(p, r)| {
            let [x, y, z] = std::array::from_fn(|axis| target[axis] - p[axis]);
            x.hypot(y).hypot(z) <= *r
        })
    {
        return Err(JsError::new(
            "target must be finite and outside every sphere",
        ));
    }
    let solved = tmatrix::cluster(lmax, k0, radii, &epsilon, &positions)?;
    let basis = basis(lmax, positions)?;
    let (incident, incident_ctx) = plane::expansion(basis.clone(), vec![k], vec![helicity], true)?;
    let scattered = solved.value() * &incident;
    let field = fields::field(
        basis,
        scattered.as_slice().to_vec(),
        vec![target],
        ks,
        true,
        Radial::Outgoing,
    )?;
    let incident_field = plane::field_value(
        plane::polarization(k, helicity, true)?,
        k,
        target.map(Complex::from),
    )?;
    let total: [Complex; 3] =
        std::array::from_fn(|axis| incident_field[axis] + field.value[0][axis]);
    let intensity = total.iter().map(Complex::norm_sqr).sum::<f64>();
    let field_gradient = field.pullback(&[total.map(|z| 2.0 * z)])?;
    let g_b = DMatrix::from_column_slice(incident.nrows(), 1, &field_gradient.coefficients);
    let g_t = &g_b * incident.adjoint();
    let g_a = solved.value().adjoint() * g_b;
    let incident_gradient = incident_ctx.pullback(&g_a, true)?;
    let physical_gradient = solved.pullback(&g_t)?;
    let mut result = Vec::with_capacity(1 + 4 * radii.len());
    result.push(intensity);
    result.extend(physical_gradient.radii);
    for ((physical, incident), field) in physical_gradient
        .positions
        .into_iter()
        .zip(incident_gradient.origins)
        .zip(field_gradient.origins)
    {
        result.extend((0..3).map(|axis| physical[axis] + incident[axis] + field[axis]));
    }
    if result.iter().any(|x| !x.is_finite()) {
        return Err(JsError::new("nonfinite target intensity or shape gradient"));
    }
    Ok(result)
}

/// A solved T matrix with its local spherical basis and embedding medium.
///
/// Complex arrays use consecutive real/imaginary pairs. Matrices are column-major;
/// field points and field components are point-major Cartesian xyz triples.
#[wasm_bindgen]
#[derive(Debug)]
pub struct ScatteringSystem {
    matrix: DMatrix<Complex>,
    basis: Basis,
    ks: [Complex; 2],
}

#[wasm_bindgen]
impl ScatteringSystem {
    /// Solve a concentric multilayer sphere at the origin.
    ///
    /// Radii run from inner to outer. Each material is six floats:
    /// epsilon real/imaginary, mu real/imaginary, kappa real/imaginary.
    /// Include the embedding material as the final record.
    pub fn sphere(lmax: u32, k0: f64, radii: &[f64], materials: &[f64]) -> Result<Self, JsError> {
        let materials = complex(materials)?;
        if !materials.len().is_multiple_of(3) || materials.is_empty() {
            return Err(JsError::new(
                "expected epsilon, mu and kappa for every material",
            ));
        }
        let materials: Vec<_> = materials
            .chunks_exact(3)
            .map(|m| Material {
                epsilon: m[0],
                mu: m[1],
                kappa: m[2],
            })
            .collect();
        let embedding = materials
            .last()
            .ok_or_else(|| JsError::new("missing embedding material"))?;
        let ks = embedding.indices().map(|n| n * k0);
        Ok(Self {
            matrix: tmatrix::sphere(lmax, k0, radii, &materials)?.value,
            basis: basis(lmax, vec![[0.0; 3]])?,
            ks,
        })
    }

    /// Solve multiple homogeneous nonmagnetic spheres in vacuum, including all
    /// multiple scattering. Permittivities are complex pairs; positions are xyz.
    pub fn cluster(
        lmax: u32,
        k0: f64,
        radii: &[f64],
        epsilon: &[f64],
        positions: &[f64],
    ) -> Result<Self, JsError> {
        let (epsilon, positions) = cluster_inputs(radii, epsilon, positions)?;
        let solved = tmatrix::cluster(lmax, k0, radii, &epsilon, &positions)?;
        Ok(Self {
            matrix: solved.value().clone(),
            basis: basis(lmax, positions)?,
            ks: [Complex::new(k0, 0.0); 2],
        })
    }

    /// Noninteracting homogeneous nonmagnetic spheres in vacuum, with the same
    /// inputs, local basis and geometry validation as `cluster`.
    ///
    /// Only mutual rescattering is disabled: incident origin phases and coherent
    /// interference between the emitted fields are preserved. Permittivities
    /// pack real/imaginary pairs and positions pack xyz triples.
    pub fn independent_cluster(
        lmax: u32,
        k0: f64,
        radii: &[f64],
        epsilon: &[f64],
        positions: &[f64],
    ) -> Result<Self, JsError> {
        let (epsilon, positions) = cluster_inputs(radii, epsilon, positions)?;
        let ks = vacuum(k0)?;
        let basis = basis(lmax, positions)?;
        let dimension = basis.modes.len() / radii.len();
        let mut matrix = DMatrix::zeros(basis.modes.len(), basis.modes.len());
        for (particle, (&radius, &epsilon)) in radii.iter().zip(&epsilon).enumerate() {
            let local = tmatrix::sphere(
                lmax,
                k0,
                &[radius],
                &[
                    Material {
                        epsilon,
                        ..Material::default()
                    },
                    Material::default(),
                ],
            )?;
            matrix
                .view_mut(
                    (particle * dimension, particle * dimension),
                    (dimension, dimension),
                )
                .copy_from(&local.value);
        }
        Ok(Self { matrix, basis, ks })
    }

    /// Number of local spherical modes (the T matrix is modes by modes).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn modes(&self) -> usize {
        self.matrix.nrows()
    }

    /// Copy the column-major complex T matrix to a JavaScript `Float64Array`.
    #[must_use]
    pub fn tmatrix(&self) -> Vec<f64> {
        interleaved(self.matrix.iter().copied())
    }

    /// Unit-amplitude incident plane-wave coefficients for helicity 0 or 1.
    /// The nonzero real direction vector is normalized before expansion.
    pub fn plane_wave(&self, direction: &[f64], polarization: u8) -> Result<Vec<f64>, JsError> {
        let vector = plane_vector(self.ks, direction, polarization)?;
        let (value, _) =
            plane::expansion(self.basis.clone(), vec![vector], vec![polarization], true)?;
        Ok(interleaved(value.iter().copied()))
    }

    /// Apply the solved T matrix to one incident multipole vector.
    pub fn scatter(&self, incident: &[f64]) -> Result<Vec<f64>, JsError> {
        let incident = complex(incident)?;
        if incident.len() != self.modes() {
            return Err(JsError::new("expected one incident coefficient per mode"));
        }
        let scattered = &self.matrix * DVector::from_vec(incident);
        Ok(interleaved(scattered.iter().copied()))
    }

    /// Evaluate Cartesian electric fields from local multipole coefficients.
    /// Set outgoing=true for scattered fields, false for regular incident fields.
    /// Outgoing expansions describe the exterior of the particles only.
    /// Exactly zero coefficients are skipped; nonzero amplitudes are never cut off.
    pub fn electric_field(
        &self,
        coefficients: &[f64],
        points: &[f64],
        outgoing: bool,
    ) -> Result<Vec<f64>, JsError> {
        let coefficients = complex(coefficients)?;
        let points = cartesian(points)?;
        if coefficients.len() != self.modes() {
            return Err(JsError::new("expected one field coefficient per mode"));
        }
        // Preserve the outgoing expansion's singular-origin restriction even
        // when that origin has no active modes in this particular field.
        if outgoing && points.iter().any(|p| self.basis.positions.contains(p)) {
            return Err(JsError::new(
                "outgoing spherical fields are singular at expansion origins",
            ));
        }
        let (modes, coefficients): (Vec<_>, Vec<_>) = self
            .basis
            .modes
            .iter()
            .copied()
            .zip(coefficients)
            .filter(|(_, amplitude)| *amplitude != Complex::default())
            .unzip();
        if coefficients.is_empty() {
            if self
                .ks
                .iter()
                .any(|k| !k.re.is_finite() || !k.im.is_finite() || *k == Complex::default())
            {
                return Err(JsError::new("require finite nonzero field wavenumbers"));
            }
            return Ok(vec![0.0; 6 * points.len()]);
        }
        let radial = if outgoing {
            Radial::Outgoing
        } else {
            Radial::Regular
        };
        let field = fields::field(
            Basis {
                modes,
                positions: self.basis.positions.clone(),
            },
            coefficients,
            points,
            self.ks,
            true,
            radial,
        )?;
        Ok(interleaved(field.value.into_iter().flatten()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_fields_match_the_full_native_basis() -> Result<(), JsError> {
        let system = ScatteringSystem::independent_cluster(
            3,
            1.3,
            &[0.2, 0.25],
            &[2.0, 0.01, 3.0, 0.02],
            &[-0.4, 0.0, 0.0, 0.4, 0.1, 0.0],
        )?;
        let points = vec![[1.1, 0.8, 0.7], [-0.9, -0.8, 0.6], [0.1, 0.7, -1.2]];
        let flat_points: Vec<_> = points.iter().flatten().copied().collect();
        // Nonadjacent modes, both helicities and both origins; also exercise an
        // entirely inactive first origin without renumbering the remaining modes.
        for active in [[0, 15, 31, 59], [30, 35, 41, 59]] {
            let mut coefficients = vec![Complex::default(); system.modes()];
            for i in active {
                coefficients[i] = Complex::new(0.3, -0.2);
            }
            for radial in [Radial::Regular, Radial::Outgoing] {
                let full = fields::field(
                    system.basis.clone(),
                    coefficients.clone(),
                    points.clone(),
                    system.ks,
                    true,
                    radial,
                )?;
                let sparse = system.electric_field(
                    &interleaved(coefficients.iter().copied()),
                    &flat_points,
                    radial == Radial::Outgoing,
                )?;
                for (actual, expected) in sparse
                    .iter()
                    .zip(interleaved(full.value.into_iter().flatten()))
                {
                    assert!((actual - expected).abs() < 1e-13);
                }
            }
        }
        let zero = vec![0.0; 2 * system.modes()];
        for outgoing in [false, true] {
            let field = system.electric_field(&zero, &flat_points, outgoing)?;
            assert_eq!(field.len(), 6 * points.len());
            assert!(field.into_iter().all(|value| value.abs() < f64::EPSILON));
        }
        Ok(())
    }
}
