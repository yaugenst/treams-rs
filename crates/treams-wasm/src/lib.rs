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
        let positions = cartesian(positions)?;
        let epsilon = complex(epsilon)?;
        let solved = tmatrix::cluster(lmax, k0, radii, &epsilon, &positions)?;
        Ok(Self {
            matrix: solved.value().clone(),
            basis: basis(lmax, positions)?,
            ks: [Complex::new(k0, 0.0); 2],
        })
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
        if direction.len() != 3 || polarization > 1 {
            return Err(JsError::new("expected a direction xyz and helicity 0 or 1"));
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
        let vector = direction.map(|d| self.ks[usize::from(polarization)] * (d / norm));
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
    pub fn electric_field(
        &self,
        coefficients: &[f64],
        points: &[f64],
        outgoing: bool,
    ) -> Result<Vec<f64>, JsError> {
        let coefficients = complex(coefficients)?;
        let points = cartesian(points)?;
        let radial = if outgoing {
            Radial::Outgoing
        } else {
            Radial::Regular
        };
        let field = fields::field(
            self.basis.clone(),
            coefficients,
            points,
            self.ks,
            true,
            radial,
        )?;
        Ok(interleaved(field.value.into_iter().flatten()))
    }
}
