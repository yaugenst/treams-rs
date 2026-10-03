//! The Ewald sum under test, the parser of pinned sums and small constructors.

use std::{fmt::Debug, str::FromStr};

use crate::{
    Complex,
    lattice::{
        self, BlochLattice, Derivatives, SumPart, derivatives, derivatives_part, sum, sum_part,
    },
    test_support::table,
};

/// One Ewald sum: a wave, lattice-frame rows and Bloch vector of a `dim`-dimensional
/// lattice, a Cartesian shift, the wavenumber and the split.
#[derive(Clone, Debug)]
pub(super) struct Ewald {
    pub(super) wave: lattice::Family,
    pub(super) dim: usize,
    pub(super) rows: [[f64; 3]; 3],
    pub(super) kpar: [f64; 3],
    pub(super) r: [f64; 3],
    pub(super) k: Complex,
    pub(super) eta: Complex,
}

impl Ewald {
    /// The sum on the leading `dim` block of `rows` and `kpar`.
    pub(super) fn new(
        wave: lattice::Family,
        dim: usize,
        rows: [[f64; 3]; 3],
        kpar: [f64; 3],
        r: [f64; 3],
        k: Complex,
        eta: Complex,
    ) -> Self {
        let (rows, kpar) = (embed(rows, dim), leading(dim, |j| kpar[j]));
        Self {
            wave,
            dim,
            rows,
            kpar,
            r,
            k,
            eta,
        }
    }

    /// A sum on a chain of the given period and Bloch vector.
    pub(super) fn chain(
        wave: lattice::Family,
        period: f64,
        kpar: f64,
        r: [f64; 3],
        k: Complex,
        eta: Complex,
    ) -> Self {
        let rows = [[period, 0.0, 0.0], [0.0; 3], [0.0; 3]];
        Self::new(wave, 1, rows, [kpar, 0.0, 0.0], r, k, eta)
    }

    /// The same sum at the split `eta`.
    pub(super) fn at(&self, eta: Complex) -> Self {
        Self {
            eta,
            ..self.clone()
        }
    }

    pub(super) fn lattice(&self) -> BlochLattice {
        BlochLattice::from_array(self.rows, self.kpar, self.dim).unwrap()
    }

    pub(super) fn part(&self, part: SumPart) -> crate::Result<Complex> {
        sum_part(self.wave, self.k, &self.lattice(), self.r, self.eta, part)
    }

    pub(super) fn try_sum(&self) -> crate::Result<Complex> {
        sum(self.wave, self.k, &self.lattice(), self.r, self.eta)
    }

    pub(super) fn sum(&self) -> Complex {
        self.try_sum().unwrap()
    }

    pub(super) fn try_derivatives(&self) -> crate::Result<Derivatives> {
        derivatives(self.wave, self.k, &self.lattice(), self.r, self.eta)
    }

    pub(super) fn derivatives(&self) -> Derivatives {
        self.try_derivatives().unwrap()
    }

    pub(super) fn part_derivatives(&self, part: SumPart) -> Derivatives {
        derivatives_part(self.wave, self.k, &self.lattice(), self.r, self.eta, part).unwrap()
    }

    /// The same sum of another wave, or zero for an invalid label (`|m| > l`).
    pub(super) fn of(&self, wave: lattice::Family) -> Complex {
        match wave {
            lattice::Family::Spherical { l, m } if m.abs() > l => Complex::default(),
            _ => Self {
                wave,
                ..self.clone()
            }
            .sum(),
        }
    }

    /// Cartesian axis of each lattice coordinate: z for 1D spherical lattices,
    /// otherwise x, y and z in order.
    pub(super) fn axes(&self) -> [usize; 3] {
        if matches!(self.wave, lattice::Family::Spherical { .. }) && self.dim == 1 {
            [2, 0, 1]
        } else {
            [0, 1, 2]
        }
    }

    /// The same sum with the shift moved onto the lattice's axis or plane, where
    /// the forward reciprocal sums take their special paths.
    pub(super) fn in_frame(&self) -> Self {
        let axes = self.axes();
        let mut r = [0.0; 3];
        for &axis in &axes[..self.dim] {
            r[axis] = self.r[axis];
        }
        Self { r, ..self.clone() }
    }

    /// The sum with the orthogonal map `o` applied to the shift, the lattice rows and
    /// the Bloch vector, or `None` when `o` moves the lattice out of its frame.
    pub(super) fn transformed(&self, o: [[f64; 3]; 3]) -> Option<Self> {
        let axes = self.axes();
        let apply = |v: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|i| (0..3).map(|j| o[i][j] * v[j]).sum())
        };
        let frame = |v: [f64; 3]| -> Option<[f64; 3]> {
            let mut cartesian = [0.0; 3];
            for (j, &axis) in axes.iter().enumerate().take(self.dim) {
                cartesian[axis] = v[j];
            }
            let image = apply(cartesian);
            let outside = axes[self.dim..].iter().any(|&axis| image[axis] != 0.0);
            (!outside).then(|| leading(self.dim, |j| image[axes[j]]))
        };
        let mut rows = [[0.0; 3]; 3];
        for (row, given) in rows.iter_mut().zip(self.rows).take(self.dim) {
            *row = frame(given)?;
        }
        Some(Self {
            rows,
            kpar: frame(self.kpar)?,
            r: apply(self.r),
            ..self.clone()
        })
    }

    /// Reciprocal rows `b` with `a_i . b_j = 2 pi delta_ij`.
    pub(super) fn reciprocal(&self) -> [[f64; 3]; 3] {
        lattice::reciprocal(self.rows, self.dim).unwrap()
    }
}

/// Parses a table of pinned sums with [`table`], one sum per line, `#` starting a
/// comment: `wave dim re(k) im(k) x y z re(eta) im(eta) rows kpar fields: values` with the
/// wave `s l m` (spherical) or `c m` (cylindrical) and the `dim x dim` lattice rows and
/// the Bloch vector in the lattice frame; returns each sum with its further fields and
/// the values after the colon.
pub(super) fn pinned<V: FromStr<Err: Debug>>(text: &str) -> Vec<(Ewald, Vec<String>, Vec<V>)> {
    let rows = text
        .lines()
        .map(|line| line.split_once('#').map_or(line, |(row, _)| row))
        .collect::<Vec<_>>()
        .join("\n");
    table::<String, V>(&rows)
        .into_iter()
        .map(|(key, values)| {
            let (wave, fields) = match key[0].as_str() {
                "s" => (
                    sw(key[1].parse().unwrap(), key[2].parse().unwrap()),
                    &key[3..],
                ),
                _ => (cw(key[1].parse().unwrap()), &key[2..]),
            };
            let dim: usize = fields[0].parse().unwrap();
            let x = |i: usize| -> f64 { fields[1 + i].parse().unwrap() };
            let rows = std::array::from_fn(|i| {
                leading(dim, |j| if i < dim { x(7 + dim * i + j) } else { 0.0 })
            });
            let kpar = leading(dim, |j| x(7 + dim * dim + j));
            let sum = Ewald::new(
                wave,
                dim,
                rows,
                kpar,
                [x(2), x(3), x(4)],
                c(x(0), x(1)),
                c(x(5), x(6)),
            );
            (sum, fields[8 + dim * (dim + 1)..].to_vec(), values)
        })
        .collect()
}

pub(super) fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

pub(super) const fn sw(l: i32, m: i32) -> lattice::Family {
    lattice::Family::Spherical { l, m }
}

pub(super) const fn cw(m: i32) -> lattice::Family {
    lattice::Family::Cylindrical { m }
}

/// The leading `dim x dim` block of `rows`, zero elsewhere.
pub(super) fn embed<const R: usize>(rows: [[f64; R]; R], dim: usize) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| leading(dim, |j| if i < dim { rows[i][j] } else { 0.0 }))
}

/// A lattice-frame vector with the leading `dim` components `f(j)`, zero elsewhere.
pub(super) fn leading(dim: usize, f: impl Fn(usize) -> f64) -> [f64; 3] {
    std::array::from_fn(|j| if j < dim { f(j) } else { 0.0 })
}

/// Lower triangular rows with the diagonal `pitch` and the entries `skew` below it, in the
/// leading `dim x dim` block.
pub(super) fn triangular(pitch: [f64; 3], skew: [f64; 3], dim: usize) -> [[f64; 3]; 3] {
    let rows = [
        [pitch[0], 0.0, 0.0],
        [skew[0], pitch[1], 0.0],
        [skew[1], skew[2], pitch[2]],
    ];
    embed(rows, dim)
}

/// The point at distance `rho` from the z axis at `azimuth`, and `z` along it.
pub(super) fn cylinder_point(rho: f64, azimuth: f64, z: f64) -> [f64; 3] {
    [rho * azimuth.cos(), rho * azimuth.sin(), z]
}

pub(super) fn diagonal(d: [f64; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { d[i] } else { 0.0 }))
}

/// A split of modulus 0.3 to 1.5, real or rotated off `1 / k` by up to 0.3 either way.
pub(super) fn explicit_split(k: Complex, size: f64, rotation: Option<f64>) -> Complex {
    rotation.map_or_else(
        || c(size, 0.0),
        |angle| size * k.norm() / k * Complex::from_polar(1.0, angle),
    )
}
