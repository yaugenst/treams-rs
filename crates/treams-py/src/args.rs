//! Python arguments as core inputs: layer materials (`coeffs::Material`), spheres,
//! spherical and cylindrical bases (`sw::Basis`, `cw::Basis`), and the names that
//! select a special function, vector wave or coordinate transform.
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    coeffs::Material,
    cw,
    special::{Angular, Bessel, coordinates::Transform},
    sw::{self, Mode},
    vectorwaves,
};

use crate::{convert::rows, coordinates::TRANSFORMS};

/// Layer materials from equally long epsilon, mu and kappa arrays.
pub(crate) fn materials(
    epsilon: &PyReadonlyArray1<'_, Complex>,
    mu: &PyReadonlyArray1<'_, Complex>,
    kappa: &PyReadonlyArray1<'_, Complex>,
) -> PyResult<Vec<Material>> {
    let (epsilon, mu, kappa) = (epsilon.as_array(), mu.as_array(), kappa.as_array());
    if epsilon.len() != mu.len() || epsilon.len() != kappa.len() {
        return Err(PyValueError::new_err(
            "epsilon, mu and kappa must have equal lengths",
        ));
    }
    Ok(epsilon
        .iter()
        .zip(&mu)
        .zip(&kappa)
        .map(|((&epsilon, &mu), &kappa)| Material { epsilon, mu, kappa })
        .collect())
}

/// Radii, permittivities and positions of homogeneous spheres in vacuum.
pub(crate) type Spheres = (Vec<f64>, Vec<Complex>, Vec<[f64; 3]>);

/// Spheres from radius and permittivity vectors and `(particles, 3)` positions.
pub(crate) fn spheres(
    radii: &PyReadonlyArray1<'_, f64>,
    epsilon: &PyReadonlyArray1<'_, Complex>,
    positions: PyReadonlyArray2<'_, f64>,
) -> PyResult<Spheres> {
    Ok((
        radii.as_array().to_vec(),
        epsilon.as_array().to_vec(),
        rows(positions.as_array(), "positions")?,
    ))
}

/// A spherical basis from `(position index, l, m, pol)` tuples and its positions.
pub(crate) fn make_basis(modes: Vec<(usize, i32, i32, u8)>, positions: Vec<[f64; 3]>) -> sw::Basis {
    sw::Basis {
        modes: modes
            .into_iter()
            .map(|(p, l, m, pol)| (p, Mode { l, m, pol }))
            .collect(),
        positions,
    }
}

/// A cylindrical basis from `(position index, kz, m, pol)` tuples and its positions.
pub(crate) fn make_cyl_basis(
    modes: Vec<(usize, f64, i32, u8)>,
    positions: Vec<[f64; 3]>,
) -> cw::Basis {
    cw::Basis {
        modes: modes
            .into_iter()
            .map(|(p, kz, m, pol)| (p, cw::Mode { kz, m, pol }))
            .collect(),
        positions,
    }
}

/// The Bessel function selected by `"j"`, `"y"`, `"h1"` or `"h2"`.
pub(crate) fn bessel_function(function: &str) -> PyResult<Bessel> {
    Ok(match function {
        "j" => Bessel::J,
        "y" => Bessel::Y,
        "h1" => Bessel::H1,
        "h2" => Bessel::H2,
        _ => return Err(PyValueError::new_err("invalid Bessel function")),
    })
}

/// The angular function selected by `"legendre"`, `"pi"` or `"tau"`.
pub(crate) fn angular_function(function: &str) -> PyResult<Angular> {
    Ok(match function {
        "legendre" => Angular::Legendre,
        "pi" => Angular::Pi,
        "tau" => Angular::Tau,
        _ => return Err(PyValueError::new_err("invalid angular function")),
    })
}

/// The vector-wave family of a `treams.special` function name such as `vsw_rA`, the
/// number of its continuous arguments, and the polarization that the name fixes:
/// 1 for `N` waves, 0 otherwise, and `None` for the helicity waves (`A`), which
/// take it from their labels.
pub(crate) fn family(function: &str) -> PyResult<(vectorwaves::Family, usize, Option<u8>)> {
    let radial = crate::context::radial(!function.contains("_r"));
    let (family, count) = match function {
        "vsh_X" => (vectorwaves::Family::HarmonicX, 2),
        "vsh_Y" => (vectorwaves::Family::HarmonicY, 2),
        "vsh_Z" | "sph_harm" => (vectorwaves::Family::HarmonicZ, 2),
        "vsw_M" | "vsw_N" | "vsw_A" | "vsw_rM" | "vsw_rN" | "vsw_rA" => {
            (vectorwaves::Family::Spherical(radial), 3)
        }
        "vcw_M" | "vcw_rM" => (vectorwaves::Family::Cylindrical(radial), 4),
        "vcw_N" | "vcw_A" | "vcw_rN" | "vcw_rA" => (vectorwaves::Family::Cylindrical(radial), 5),
        "vpw_M" | "vpw_N" | "vpw_A" => (vectorwaves::Family::Plane, 6),
        _ => return Err(PyValueError::new_err("unknown vector-wave function")),
    };
    let pol = if function.ends_with('A') {
        None
    } else {
        Some(u8::from(function.ends_with('N')))
    };
    Ok((family, count, pol))
}

/// The coordinate transform of a point function name such as `car2sph`.
pub(crate) fn transform(name: &str) -> PyResult<Transform> {
    TRANSFORMS
        .iter()
        .find(|(point, ..)| *point == name)
        .map(|&(.., transform)| transform)
        .ok_or_else(|| PyValueError::new_err("unknown coordinate transform"))
}
