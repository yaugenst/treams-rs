//! `lattice_sum_record`, the lattice expansions `lattice_expansion`,
//! `cylindrical_lattice_expansion` and `lattice_expansion_from_table` with their
//! contexts, and `lattice_cube`, `diffraction_orders` and `first_brillouin`
//! (`treams_core::lattice`, `treams_core::sw`, `treams_core::cw`).

use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4,
    ndarray::{Array2, Array4},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::borrow::Cow;
use treams_core::{Complex, fpenv::ieee, lattice};

use crate::{
    args::{make_basis, make_cyl_basis},
    broadcast::{BroadcastShapes, broadcast_context, check_broadcast, shaped},
    context::{context, detached, error},
    convert::{
        C1, C2, C4, CDyn, Cotangent, R1, R2, RDyn, layout_error, matrix, owned_matrix, rows_array,
    },
};

#[derive(Debug)]
enum Periodic {
    Spherical(treams_core::sw::LatticeExpansionResidual),
    Cylindrical(treams_core::cw::LatticeExpansionResidual),
}
impl Periodic {
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Spherical(r) => r.shape(),
            Self::Cylindrical(r) => r.shape(),
        }
    }
    fn pullback(
        self,
        g: &nalgebra::DMatrix<Complex>,
    ) -> treams_core::Result<treams_core::basis::LatticeExpansionGradient> {
        match self {
            Self::Spherical(r) => r.pullback(g),
            Self::Cylindrical(r) => r.pullback(g),
        }
    }
}
context!(LatticeExpansionContext(Periodic));
type PeriodicGradients<'py> = (R2<'py>, R2<'py>, C1<'py>, R1<'py>, R2<'py>);
fn periodic_gradient(
    py: Python<'_>,
    result: treams_core::basis::LatticeExpansionGradient,
) -> PyResult<PeriodicGradients<'_>> {
    Ok((
        rows_array(py, result.expansion.destination)?,
        rows_array(py, result.expansion.source)?,
        result.expansion.ks.to_vec().into_pyarray(py),
        result.kpar.into_pyarray(py),
        owned_matrix(py, result.vectors)?,
    ))
}
#[pymethods]
impl LatticeExpansionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<PeriodicGradients<'py>> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, Periodic::shape)?;
            periodic_gradient(py, detached(py, move || residual.pullback(&g))?)
        })
    }
    /// Also return gradients of sorted distinct shared axial wavenumbers.
    fn pullback_axial<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>, R1<'py>, R2<'py>, R1<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, Periodic::shape)?;
            let Periodic::Cylindrical(residual) = residual else {
                return Err(PyValueError::new_err(
                    "axial periodic derivatives require two cylindrical bases",
                ));
            };
            let (result, axial) = detached(py, move || residual.pullback_axial(&g))?;
            let (destination, source, ks, kpar, vectors) = periodic_gradient(py, result)?;
            Ok((
                destination,
                source,
                ks,
                kpar,
                vectors,
                axial.into_pyarray(py),
            ))
        })
    }
}

/// A periodic expansion matrix, copied into a C-ordered array, and its context.
fn finish_periodic<'py>(
    py: Python<'py>,
    value: &nalgebra::DMatrix<Complex>,
    residual: Periodic,
) -> PyResult<(C2<'py>, LatticeExpansionContext)> {
    Ok((matrix(py, value)?, LatticeExpansionContext::new(residual)))
}

/// Record the lattice expansion between two spherical bases: `sw::lattice_expansion`.
#[pyfunction]
pub(crate) fn lattice_expansion(
    py: Python<'_>,
    destination: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    kpar: Vec<f64>,
    a: Vec<Vec<f64>>,
    eta: Complex,
) -> PyResult<(C2<'_>, LatticeExpansionContext)> {
    ieee(|| {
        let destination = make_basis(destination, destination_positions);
        let source = make_basis(source, source_positions);
        let lattice = lattice::BlochLattice::new(&a, &kpar).map_err(error)?;
        let (value, residual) = detached(py, move || {
            treams_core::sw::lattice_expansion(destination, source, ks, helicity, lattice, eta)
        })?;
        finish_periodic(py, &value, Periodic::Spherical(residual))
    })
}

/// Record the lattice expansion between two cylindrical bases: `cw::lattice_expansion`.
#[pyfunction]
pub(crate) fn cylindrical_lattice_expansion(
    py: Python<'_>,
    destination: Vec<(usize, f64, i32, u8)>,
    source: Vec<(usize, f64, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    kpar: Vec<f64>,
    a: Vec<Vec<f64>>,
    eta: Complex,
) -> PyResult<(C2<'_>, LatticeExpansionContext)> {
    ieee(|| {
        let destination = make_cyl_basis(destination, destination_positions);
        let source = make_cyl_basis(source, source_positions);
        let lattice = lattice::BlochLattice::new(&a, &kpar).map_err(error)?;
        let (value, residual) = detached(py, move || {
            treams_core::cw::lattice_expansion(destination, source, ks, lattice, eta)
        })?;
        finish_periodic(py, &value, Periodic::Cylindrical(residual))
    })
}

context!(LatticeExpansionFromTableContext(
    treams_core::sw::LatticeExpansionFromTableResidual
));

#[pymethods]
impl LatticeExpansionFromTableContext {
    fn pullback<'py>(&mut self, py: Python<'py>, cotangent: Cotangent<'py>) -> PyResult<C4<'py>> {
        ieee(|| {
            let (residual, g) = self.residual.take_with_matrix(
                &cotangent,
                treams_core::sw::LatticeExpansionFromTableResidual::shape,
            )?;
            let shape = residual.table_shape();
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok(Array4::from_shape_vec(shape, gradient)
                .map_err(layout_error)?
                .into_pyarray(py))
        })
    }
}

/// Record a spherical lattice expansion from tabulated sums: `sw::lattice_expansion_from_table`.
#[pyfunction]
pub(crate) fn lattice_expansion_from_table<'py>(
    py: Python<'py>,
    destination: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    helicity: bool,
    table: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(C2<'py>, LatticeExpansionFromTableContext)> {
    ieee(|| {
        let view = table.as_array();
        let shape = view.shape();
        let channels = shape[2];
        if shape[0] != destination_positions.len() || shape[1] != source_positions.len() {
            return Err(PyValueError::new_err(
                "table position axes do not match the bases",
            ));
        }
        let values = view.as_slice().map_or_else(
            || Cow::Owned(view.iter().copied().collect::<Vec<_>>()),
            Cow::Borrowed,
        );
        let destination = make_basis(destination, destination_positions);
        let source = make_basis(source, source_positions);
        let (value, residual) = detached(py, || {
            treams_core::sw::lattice_expansion_from_table(
                &destination,
                &source,
                helicity,
                channels,
                &values,
            )
        })?;
        if residual.table_shape() != shape {
            return Err(PyValueError::new_err(
                "table axes must be destination position, source position, channel, harmonic",
            ));
        }
        // The matrix moves into an F-ordered array.
        Ok((
            owned_matrix(py, value)?,
            LatticeExpansionFromTableContext::new(residual),
        ))
    })
}

broadcast_context!(LatticeSumContext(
    lattice::SumResidual,
    [Vec<usize>; 5],
    /// The lattice dimension.
    dim: usize,
    /// The number of shift coordinates: 3 for spherical sums, 2 for cylindrical ones.
    coordinates: usize,
));
#[pymethods]
impl LatticeSumContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(CDyn<'py>, RDyn<'py>, RDyn<'py>, RDyn<'py>, CDyn<'py>)> {
        ieee(|| {
            use crate::broadcast::reduce_broadcast;
            let BroadcastShapes {
                shape,
                argument_shapes: [sk, sq, sa, sr, se],
            } = &self.shapes;
            let g = crate::broadcast::cotangent(&cotangent, shape)?;
            let residual = self.residual.take()?;
            let values = detached(py, move || residual.pullback(&g))?;
            let mut qshape = shape.clone();
            qshape.push(self.dim);
            let mut ashape = qshape.clone();
            ashape.push(self.dim);
            let mut rshape = shape.clone();
            rshape.push(self.coordinates);
            Ok((
                reduce_broadcast(values.iter().map(|g| g.k).collect(), shape, sk)?.into_pyarray(py),
                reduce_broadcast(
                    values
                        .iter()
                        .flat_map(|g| g.kpar[..self.dim].iter().copied())
                        .collect(),
                    &qshape,
                    sq,
                )?
                .into_pyarray(py),
                reduce_broadcast(
                    values
                        .iter()
                        .flat_map(|g| {
                            g.vectors[..self.dim]
                                .iter()
                                .flat_map(|r| r[..self.dim].iter().copied())
                        })
                        .collect(),
                    &ashape,
                    sa,
                )?
                .into_pyarray(py),
                reduce_broadcast(
                    values
                        .iter()
                        .flat_map(|g| g.shift[..self.coordinates].iter().copied())
                        .collect(),
                    &rshape,
                    sr,
                )?
                .into_pyarray(py),
                reduce_broadcast(values.iter().map(|g| g.eta).collect(), shape, se)?
                    .into_pyarray(py),
            ))
        })
    }
}
/// Record lattice sums, their Ewald parts or direct shells: `lattice::sum_array`.
#[pyfunction]
pub(crate) fn lattice_sum_record<'py>(
    py: Python<'py>,
    spherical: bool,
    dim: usize,
    modes: Vec<(i32, i32)>,
    k: PyReadonlyArray1<'py, Complex>,
    kpar: PyReadonlyArray2<'py, f64>,
    a: numpy::PyReadonlyArray3<'py, f64>,
    r: PyReadonlyArray2<'py, f64>,
    eta: PyReadonlyArray1<'py, Complex>,
    part: u8,
    shells: Vec<i64>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 5],
) -> PyResult<(CDyn<'py>, LatticeSumContext)> {
    ieee(|| {
        let q = kpar.as_array();
        let a = a.as_array();
        let r = r.as_array();
        let coordinates = if spherical { 3 } else { 2 };
        if !(1..=if spherical { 3 } else { 2 }).contains(&dim)
            || q.shape()[1] != dim
            || a.shape()[1..] != [dim, dim]
            || r.shape()[1] != coordinates
            || part > 3
        {
            return Err(PyValueError::new_err(
                "invalid lattice dimensions or sum component",
            ));
        }
        let cores: [&[usize]; 5] = [&[], &[dim], &[dim, dim], &[coordinates], &[]];
        for (argument, core) in argument_shapes.iter().zip(cores) {
            check_broadcast(&[shape.as_slice(), core].concat(), &[argument])?;
        }
        let geometry_count = if q.shape()[0] == 0 || a.shape()[0] == 0 {
            0
        } else {
            q.shape()[0].max(a.shape()[0])
        };
        if [q.shape()[0], a.shape()[0]]
            .iter()
            .any(|&n| n != 1 && n != geometry_count)
        {
            return Err(PyValueError::new_err("geometry batch lengths must agree"));
        }
        let lattices = (0..geometry_count)
            .map(|i| {
                let qi = if q.shape()[0] == 1 { 0 } else { i };
                let ai = if a.shape()[0] == 1 { 0 } else { i };
                lattice::BlochLattice::from_array(
                    std::array::from_fn(|j| {
                        std::array::from_fn(|h| {
                            if j < dim && h < dim {
                                a[(ai, j, h)]
                            } else {
                                0.0
                            }
                        })
                    }),
                    std::array::from_fn(|j| if j < dim { q[(qi, j)] } else { 0.0 }),
                    dim,
                )
            })
            .collect::<treams_core::Result<Vec<_>>>()
            .map_err(error)?;
        let shifts = r
            .outer_iter()
            .map(|row| std::array::from_fn(|j| if j < coordinates { row[j] } else { 0.0 }))
            .collect();
        let waves = modes
            .into_iter()
            .map(|(l, m)| {
                if spherical {
                    lattice::Family::Spherical { l, m }
                } else {
                    lattice::Family::Cylindrical { m }
                }
            })
            .collect();
        let parts = if part == 3 {
            shells.into_iter().map(lattice::SumPart::Direct).collect()
        } else {
            vec![match part {
                0 => lattice::SumPart::Full,
                1 => lattice::SumPart::Real,
                _ => lattice::SumPart::Reciprocal,
            }]
        };
        let k = k.as_array().to_vec();
        let eta = eta.as_array().to_vec();
        let (value, residual) = detached(py, move || {
            lattice::sum_array(waves, k, lattices, shifts, eta, parts)
        })?;
        Ok((
            shaped(py, value, &shape)?,
            LatticeSumContext::new(
                residual,
                BroadcastShapes {
                    shape,
                    argument_shapes,
                },
                dim,
                coordinates,
            ),
        ))
    })
}

/// All integer points of a cube, or its boundary only: `lattice::cube`.
#[pyfunction]
pub(crate) fn lattice_cube(
    py: Python<'_>,
    dim: usize,
    n: i64,
    edge: bool,
) -> PyResult<Bound<'_, PyArray2<i64>>> {
    ieee(|| {
        let values = detached(py, move || lattice::cube(dim, n, edge))?;
        Ok(Array2::from_shape_vec((values.len() / dim, dim), values)
            .map_err(layout_error)?
            .into_pyarray(py))
    })
}
/// The diffraction orders inside a circle of a 2D lattice: `lattice::diffraction_orders`.
#[pyfunction]
pub(crate) fn diffraction_orders<'py>(
    py: Python<'py>,
    b: PyReadonlyArray2<'py, f64>,
    radius: f64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    ieee(|| {
        let b = b.as_array();
        if b.shape() != [2, 2] {
            return Err(PyValueError::new_err(
                "reciprocal lattice requires shape (2, 2)",
            ));
        }
        let b = std::array::from_fn(|i| std::array::from_fn(|j| b[(i, j)]));
        let values = detached(py, move || lattice::diffraction_orders(b, radius))?;
        Ok(Array2::from_shape_vec((values.len() / 2, 2), values)
            .map_err(layout_error)?
            .into_pyarray(py))
    })
}
/// A wavevector reduced to the first Brillouin zone in 2D or 3D: `lattice::first_brillouin`.
#[pyfunction]
pub(crate) fn first_brillouin<'py>(
    py: Python<'py>,
    k: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    dim: usize,
    n: usize,
) -> PyResult<R1<'py>> {
    ieee(|| {
        let k = k.as_array();
        let b = b.as_array();
        if !(2..=3).contains(&dim) || k.len() != dim || b.shape() != [dim, dim] {
            return Err(PyValueError::new_err(
                "wavevector and lattice must match dimension",
            ));
        }
        let k = std::array::from_fn(|i| if i < dim { k[i] } else { 0.0 });
        let b = std::array::from_fn(|i| {
            std::array::from_fn(|j| if i < dim && j < dim { b[(i, j)] } else { 0.0 })
        });
        let value = detached(py, move || lattice::first_brillouin(k, b, dim, n))?;
        Ok(value[..dim].to_vec().into_pyarray(py))
    })
}
