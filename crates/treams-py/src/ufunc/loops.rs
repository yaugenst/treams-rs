//! The `NumPy` inner loops: adapters from operand dtypes to the core kernels.
//! Loop parameter types spell out the operand dtypes of each loop.

// NumPy calls the loops as raw C callbacks, which read operands through `ffi`.
#![allow(unsafe_code)]

use std::{
    ffi::{c_char, c_long, c_void},
    ops::{Add, Mul, Sub},
};

use numpy::npyffi::npy_intp;

use super::ffi::{Args, Call, Dtype, Operand, drive, guard, offset};
use super::kinds::{angular, bessel, ewald, parallel_from, pol, poltype, vsh};
use crate::context::radial;
use treams_core::{
    Complex, Error, Result,
    basis::ModeLabel,
    cw,
    lattice::SumPart,
    rotation,
    special::{self, Angular, Bessel, MAX_LABEL},
    sw::{Mode, PolarTranslation},
    vectorwaves::{self, WaveLabel},
};

/// Real or complex operand of loops registered for both dtypes.
pub(super) trait Argument: Dtype + Into<Complex> {
    /// The value of this dtype that a loop writes for `value`.
    fn from_complex(value: Complex) -> Self;
}
impl Argument for f64 {
    fn from_complex(value: Complex) -> Self {
        value.re
    }
}
impl Argument for Complex {
    fn from_complex(value: Complex) -> Self {
        value
    }
}

/// Define `NumPy` loops for elementwise kernels over scalar operands.
///
/// `name[generics] @ threshold, (argument: type, ...) -> output { body }`, one
/// after another. The body is safe code returning `Result<output>`; the
/// registered dtype row derives from the argument and output types.
macro_rules! scalar_loops {
    ($($name:ident $([$($generics:tt)*])? @ $parallel:expr,
       ($($argument:ident: $ty:ty),+ $(,)?) -> $output:ty $body:block)+) => {$(
        pub(super) unsafe extern "C" fn $name $(<$($generics)*>)? (
            args: Args<($($ty,)+), $output>,
            dimensions: *mut npy_intp,
            steps: *mut npy_intp,
            _data: *mut c_void,
        ) {
            let kernel = |$($argument: $ty),+| -> Result<$output> { $body };
            guard(true, || {
                // SAFETY: NumPy supplies operands of the dtype row derived from
                // `Args`, which the kernel reads; vector outputs have the only
                // core dimension of their derived signature.
                unsafe {
                    let call = args.call(dimensions, steps);
                    let ($($argument,)+) = call.inputs;
                    call.store($parallel, |i| kernel($($argument.read(i)),+))
                }
            });
        }
    )+};
}

#[allow(clippy::cast_possible_truncation)] // Checked integral labels within the supported domain.
fn label(value: f64) -> Result<i32> {
    let bound = f64::from(MAX_LABEL);
    if value.fract() != 0.0 || !(-bound..=bound).contains(&value) {
        return Err(Error::InvalidInput(
            // 260 is MAX_LABEL.
            "special-function labels must be integers in [-260, 260]".into(),
        ));
    }
    Ok(value as i32)
}

fn integer_wave_mode(l: c_long, m: c_long, pol: c_long) -> Result<Mode> {
    let invalid = || Error::InvalidInput("invalid integer wave labels".into());
    Ok(Mode {
        l: i32::try_from(l).map_err(|_| invalid())?,
        m: i32::try_from(m).map_err(|_| invalid())?,
        pol: u8::try_from(pol).map_err(|_| invalid())?,
    })
}

/// [`integer_wave_mode`] as the label of a vector wave.
fn integer_wave_label(l: c_long, m: c_long, pol: c_long) -> Result<WaveLabel> {
    let Mode { l, m, pol } = integer_wave_mode(l, m, pol)?;
    Ok(WaveLabel { l, m, pol })
}

fn cylindrical_mode(kz: f64, m: c_long, pol: c_long) -> Result<cw::Mode> {
    Ok(cw::Mode {
        kz,
        m: i32::try_from(m).map_err(|_| Error::InvalidInput("invalid order".into()))?,
        pol: special::pol_index(pol)?,
    })
}

scalar_loops! {
    bessel_loop[const KIND: u8, const SPHERICAL: bool, const DERIVATIVE: u8]
        @ parallel_from::BESSEL,
    (order: f64, z: Complex) -> Complex {
        let kind = match KIND {
            bessel::J => Bessel::J,
            bessel::Y => Bessel::Y,
            bessel::H1 => Bessel::H1,
            _ => Bessel::H2,
        };
        special::bessel(order, z, kind, SPHERICAL, DERIVATIVE)
    }
    // lpmv takes (order, degree, z); pi_fun and tau_fun take (degree, order, z).
    angular_loop[const KIND: u8] @ parallel_from::DEFAULT,
    (first: f64, second: f64, z: Complex) -> Complex {
        match KIND {
            angular::LEGENDRE => special::angular_value(second, first, z, Angular::Legendre),
            angular::PI_FUN => special::angular_value(first, second, z, Angular::Pi),
            _ => special::angular_value(first, second, z, Angular::Tau),
        }
    }
    legendre_real_loop @ parallel_from::DEFAULT, (order: f64, degree: f64, x: f64) -> f64 {
        special::lpmv_real(degree, order, x)
    }
    wigner_small_d_loop @ parallel_from::DEFAULT,
    (l: f64, m: f64, k: f64, theta: Complex) -> Complex {
        special::wigner_small_d(label(l)?, label(m)?, label(k)?, theta)
    }
    wigner_d_loop @ parallel_from::DEFAULT,
    (l: f64, m: f64, k: f64, phi: Complex, theta: Complex, psi: Complex) -> Complex {
        special::wigner_d(label(l)?, label(m)?, label(k)?, [phi, theta, psi])
    }
    wigner3j_loop @ parallel_from::DEFAULT,
    (l1: f64, l2: f64, l3: f64, m1: f64, m2: f64, m3: f64) -> f64 {
        Ok(special::wigner3j(
            label(l1)?, label(l2)?, label(l3)?, label(m1)?, label(m2)?, label(m3)?,
        ))
    }
    gamma_loop @ parallel_from::DEFAULT, (n: f64, z: Complex) -> Complex {
        special::incgamma(n, z)
    }
    refractive_indices_loop @ parallel_from::SERIAL,
    (epsilon: Complex, mu: Complex, kappa: Complex) -> [Complex; 2] {
        Ok(treams_core::coeffs::Material { epsilon, mu, kappa }.indices())
    }
    refractive_indices_real_loop @ parallel_from::SERIAL,
    (epsilon: f64, mu: f64, kappa: f64) -> [f64; 2] {
        Ok(treams_core::coeffs::real_refractive_indices(epsilon, mu, kappa))
    }
    wave_vector_z_loop[T: Argument] @ parallel_from::SERIAL, (kx: T, ky: T, k: T) -> Complex {
        Ok(treams_core::pw::wave_vector_z(kx.into(), ky.into(), k.into()))
    }
    first_brillouin_1d_loop @ parallel_from::DEFAULT, (k: f64, b: f64) -> f64 {
        treams_core::lattice::first_brillouin_1d(k, b)
    }
}

// The Kambe cost depends on the order's parity: odd orders sum incomplete-gamma
// series (about 30 us per element), even orders take an erfc closed form (under
// a microsecond). Only a broadcast odd order is known to be slow throughout.
pub(super) unsafe extern "C" fn kambe_loop(
    args: Args<(f64, Complex, Complex), Complex>,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    guard(true, || {
        // SAFETY: NumPy supplies operands of the dtype row derived from `Args`.
        unsafe {
            let call = args.call(dimensions, steps);
            let (n, z, eta) = call.inputs;
            let odd = call.len > 0 && n.broadcast() && n.read(0) % 2.0 != 0.0;
            let parallel = if odd {
                parallel_from::SLOW
            } else {
                parallel_from::DEFAULT
            };
            call.store(parallel, |i| {
                special::intkambe(label(n.read(i))?, z.read(i), eta.read(i))
            })
        }
    });
}

/// Byte offsets of the entries of a `dim x dim` core block with two core strides,
/// computed once per loop call (zero outside the block).
fn cell_offsets(strides: [npy_intp; 2], dim: usize) -> [[isize; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i < dim && j < dim {
                offset(strides[0], i).wrapping_add(offset(strides[1], j))
            } else {
                0
            }
        })
    })
}

/// Copy the `DIM x DIM` cell (zero padded to 3 x 3) of element `index`.
///
/// # Safety
/// `cell` is the operand of a registered `(i,i)` core of this call whose
/// entries lie at `offsets` (from [`cell_offsets`]), `DIM` <= 3 is its core
/// size and `index` is below the loop count.
unsafe fn cell_input<T: Dtype + Default, const DIM: usize>(
    cell: Operand<T>,
    index: usize,
    offsets: &[[isize; 3]; 3],
) -> [[T; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i < DIM && j < DIM {
                // SAFETY: Guaranteed by the caller; transposed, reversed and
                // unaligned cells are read through their core strides.
                unsafe { cell.read_at(index, offsets[i][j]) }
            } else {
                T::default()
            }
        })
    })
}

fn cell_dimension(dim: usize) -> Result<usize> {
    let invalid = || Error::InvalidInput("cell dimension must be 1, 2 or 3".into());
    (1..=3).contains(&dim).then_some(dim).ok_or_else(invalid)
}

/// Cell entries whose determinant a volume loop forms: `f64` or `Wrapping<c_long>`.
pub(super) trait CellEntry:
    Dtype + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
}
impl<T: Dtype + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> CellEntry for T {}

/// Cell volumes (areas, lengths) of one loop call.
///
/// # Safety
/// `call` is a `(i,i)->()` loop call with core size `DIM` and entry `offsets`
/// from [`cell_offsets`].
unsafe fn volume_cells<T: CellEntry, const DIM: usize>(
    call: &Call<(T,), T>,
    offsets: &[[isize; 3]; 3],
) -> Result<()> {
    for index in 0..call.len {
        // SAFETY: Guaranteed by the caller for every index below the loop count.
        let cell = unsafe { cell_input::<T, DIM>(call.inputs.0, index, offsets) };
        let value = treams_core::lattice::volume(cell, DIM)?;
        // SAFETY: As above; the output holds one T per index.
        unsafe { call.output.write_at(index, 0, value) };
    }
    Ok(())
}

pub(super) unsafe extern "C" fn volume_loop<T: CellEntry>(
    args: Args<(T,), T>,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    guard(false, || {
        // SAFETY: NumPy supplies operands of the dtype row derived from `Args`
        // with the (i,i)->() core; each matrix is copied before its output is
        // written.
        unsafe {
            let call = args.call(dimensions, steps);
            let dim = cell_dimension(call.core_len(0))?;
            let offsets = cell_offsets([call.core_stride(0), call.core_stride(1)], dim);
            // A constant dimension unrolls the per-cell copies and drops their
            // bounds checks; at run time they cost a large part of a cell.
            match dim {
                1 => volume_cells::<T, 1>(&call, &offsets),
                2 => volume_cells::<T, 2>(&call, &offsets),
                _ => volume_cells::<T, 3>(&call, &offsets),
            }
        }
    });
}

/// Reciprocal cells of one loop call.
///
/// # Safety
/// `call` is a `(i,i)->(i,i)` loop call with core size `DIM` and input and
/// output entry offsets from [`cell_offsets`].
unsafe fn reciprocal_cells<const DIM: usize>(
    call: &Call<(f64,), f64>,
    input: &[[isize; 3]; 3],
    output: &[[isize; 3]; 3],
) -> Result<()> {
    for index in 0..call.len {
        // SAFETY: Guaranteed by the caller for every index below the loop count.
        let cell = unsafe { cell_input::<f64, DIM>(call.inputs.0, index, input) };
        let value = treams_core::lattice::reciprocal(cell, DIM)?;
        for (row, offsets) in value.iter().zip(output).take(DIM) {
            for (&v, &shift) in row.iter().zip(offsets).take(DIM) {
                // SAFETY: As above, for the output cell of this index.
                unsafe { call.output.write_at(index, shift, v) };
            }
        }
    }
    Ok(())
}

pub(super) unsafe extern "C" fn reciprocal_loop(
    args: Args<(f64,), f64>,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    guard(false, || {
        // SAFETY: NumPy supplies operands of the dtype row derived from `Args`
        // with the (i,i)->(i,i) core, two core strides per operand. Each input
        // cell is copied before its output is written.
        unsafe {
            let call = args.call(dimensions, steps);
            let dim = cell_dimension(call.core_len(0))?;
            let input = cell_offsets([call.core_stride(0), call.core_stride(1)], dim);
            let output = cell_offsets([call.core_stride(2), call.core_stride(3)], dim);
            // A constant dimension unrolls the per-cell copies (see `volume_loop`).
            match dim {
                1 => reciprocal_cells::<1>(&call, &input, &output),
                2 => reciprocal_cells::<2>(&call, &input, &output),
                _ => reciprocal_cells::<3>(&call, &input, &output),
            }
        }
    });
}

/// A lattice-sum mode label: `c_long`, or an integral `f64`.
pub(super) trait Label: Dtype {
    fn mode(self) -> Result<i32>;
}
impl Label for c_long {
    fn mode(self) -> Result<i32> {
        i32::try_from(self)
            .map_err(|_| Error::InvalidInput("lattice mode exceeds i32 range".into()))
    }
}
impl Label for f64 {
    fn mode(self) -> Result<i32> {
        label(self)
    }
}

/// The Ewald split of lattice-sum part `PART`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(super) struct Eta<const PART: u8>(Complex);
impl<const PART: u8> Dtype for Eta<PART> {
    const TYPE: c_char = Complex::TYPE;
}

/// The last lattice-sum operand: an Ewald split, or a direct shell index.
pub(super) trait Tail: Dtype {
    fn part(self) -> (Complex, SumPart);
}
impl<const PART: u8> Tail for Eta<PART> {
    fn part(self) -> (Complex, SumPart) {
        let part = match PART {
            ewald::FULL => SumPart::Full,
            ewald::REAL => SumPart::Real,
            _ => SumPart::Reciprocal,
        };
        (self.0, part)
    }
}
impl Tail for c_long {
    fn part(self) -> (Complex, SumPart) {
        (Complex::default(), SumPart::Direct(self))
    }
}

/// Components of the displacement `r` of a lattice sum: z too for spherical
/// three-dimensional or shifted sums.
const fn lattice_width(spherical: bool, dim: usize, shifted: bool) -> usize {
    2 + (spherical && (dim == 3 || shifted)) as usize
}

/// Core signature of a lattice sum: labels, k, q (dim), a (dim, dim), r and the
/// tail; the one-dimensional sum about a lattice point has scalar operands only.
pub(super) fn lattice_core(
    labels: usize,
    spherical: bool,
    dim: usize,
    shifted: bool,
) -> Option<String> {
    let r = lattice_width(spherical, dim, shifted);
    let (q, a) = match (dim, shifted) {
        (1, false) => return None,
        (1, true) => ("()".to_owned(), "()".to_owned()),
        _ => (format!("({dim})"), format!("({dim},{dim})")),
    };
    Some(format!("{}(),{q},{a},({r}),()->()", "(),".repeat(labels)))
}

#[allow(clippy::type_complexity)] // The operand types are the dtype row.
pub(super) unsafe extern "C" fn lattice_loop<
    L: Label,
    E: Tail,
    const SPHERICAL: bool,
    const DIM: usize,
    const SHIFTED: bool,
    const LABELS: usize,
>(
    args: Args<([L; LABELS], Complex, f64, f64, f64, E), Complex>,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    use treams_core::lattice::{self, BlochLattice, sum_part};
    // Spherical sums other than the unshifted 1d sum take a second mode label.
    const { assert!(LABELS == 1 + (SPHERICAL && (DIM != 1 || SHIFTED)) as usize) };
    guard(true, || {
        // SAFETY: NumPy supplies operands of the dtype row derived from `Args`
        // with the core dimensions of `lattice_core`. Each evaluation copies
        // its inputs.
        unsafe {
            let call = args.call(dimensions, steps);
            let (labels, k, q, a, r, tail) = call.inputs;
            // Core strides: the shifted 1d displacement, or q, a (two) and r.
            let cores = if DIM == 1 { usize::from(SHIFTED) } else { 4 };
            let strides: [npy_intp; 4] =
                std::array::from_fn(|j| if j < cores { call.core_stride(j) } else { 0 });
            let geometry = |i| {
                let q = std::array::from_fn(|j| {
                    if j < DIM {
                        q.read_at(i, offset(strides[0], j))
                    } else {
                        0.0
                    }
                });
                let a = std::array::from_fn(|j| {
                    std::array::from_fn(|h| {
                        if j < DIM && h < DIM {
                            a.read_at(i, offset(strides[1], j) + offset(strides[2], h))
                        } else {
                            0.0
                        }
                    })
                });
                BlochLattice::from_array(a, q, DIM)
            };
            let fixed = (call.len > 0 && q.broadcast() && a.broadcast())
                .then(|| geometry(0))
                .transpose()?;
            call.store(parallel_from::SLOW, |i| {
                let mode = |j: usize| labels.get(j).map_or(Ok(0), |label| label.read(i).mode());
                let l = mode(0)?;
                let wave = if SPHERICAL {
                    lattice::Family::Spherical { l, m: mode(1)? }
                } else {
                    lattice::Family::Cylindrical { m: l }
                };
                let varying;
                let lattice = if let Some(ref lattice) = fixed {
                    lattice
                } else {
                    varying = geometry(i)?;
                    &varying
                };
                let mut position = [0.0; 3];
                if DIM == 1 && !SHIFTED {
                    position[if SPHERICAL { 2 } else { 0 }] = r.read(i);
                } else {
                    let stride = strides[if DIM == 1 { 0 } else { 3 }];
                    let width = lattice_width(SPHERICAL, DIM, SHIFTED);
                    for (j, x) in position.iter_mut().enumerate().take(width) {
                        *x = r.read_at(i, offset(stride, j));
                    }
                }
                let (eta, part) = tail.read(i).part();
                sum_part(wave, k.read(i), lattice, position, eta, part)
            })
        }
    });
}

/// Evaluate a vector wave, padding its arguments to the core's six slots.
fn wave<const K: usize>(
    family: vectorwaves::Family,
    label: WaveLabel,
    arguments: [Complex; K],
    helicity: bool,
) -> Result<[Complex; 3]> {
    let arguments = std::array::from_fn(|i| arguments.get(i).copied().unwrap_or_default());
    vectorwaves::vector_wave(family, label, arguments, helicity)
}

scalar_loops! {
    sph_harm_loop[T: Argument] @ parallel_from::DEFAULT,
    (m: f64, l: f64, phi: f64, theta: T) -> Complex {
        vectorwaves::sph_harm(label(l)?, label(m)?, theta.into(), phi.into())
    }
    vsh_loop[T: Argument, const FAMILY: u8] @ parallel_from::DEFAULT,
    (l: c_long, m: c_long, theta: T, phi: f64) -> [Complex; 3] {
        let family = match FAMILY {
            vsh::HARMONIC_X => vectorwaves::Family::HarmonicX,
            vsh::HARMONIC_Y => vectorwaves::Family::HarmonicY,
            _ => vectorwaves::Family::HarmonicZ,
        };
        wave(family, integer_wave_label(l, m, 0)?, [theta.into(), phi.into()], false)
    }
    // Parity M and N spherical waves.
    vsw_loop[T: Argument, const REGULAR: bool, const POL: u8] @ parallel_from::DEFAULT,
    (l: c_long, m: c_long, kr: Complex, theta: T, phi: f64) -> [Complex; 3] {
        let wave_label = integer_wave_label(l, m, c_long::from(POL))?;
        wave(
            vectorwaves::Family::Spherical(radial(!REGULAR)),
            wave_label,
            [kr, theta.into(), phi.into()],
            false,
        )
    }
    vsw_a_loop[T: Argument, const REGULAR: bool] @ parallel_from::DEFAULT,
    (l: c_long, m: c_long, kr: Complex, theta: T, phi: f64, pol: c_long) -> [Complex; 3] {
        let wave_label = integer_wave_label(l, m, pol)?;
        wave(
            vectorwaves::Family::Spherical(radial(!REGULAR)),
            wave_label,
            [kr, theta.into(), phi.into()],
            true,
        )
    }
    vcw_m_loop[const REGULAR: bool] @ parallel_from::DEFAULT,
    (kz: f64, m: c_long, kr: Complex, phi: f64, z: f64) -> [Complex; 3] {
        let arguments = [kz.into(), kr, phi.into(), z.into()];
        wave(
            vectorwaves::Family::Cylindrical(radial(!REGULAR)),
            integer_wave_label(0, m, 0)?,
            arguments,
            false,
        )
    }
    vcw_n_loop[const REGULAR: bool] @ parallel_from::DEFAULT,
    (kz: f64, m: c_long, kr: Complex, phi: f64, z: f64, k: Complex) -> [Complex; 3] {
        let arguments = [kz.into(), kr, phi.into(), z.into(), k];
        wave(
            vectorwaves::Family::Cylindrical(radial(!REGULAR)),
            integer_wave_label(0, m, 1)?,
            arguments,
            false,
        )
    }
    vcw_a_loop[const REGULAR: bool] @ parallel_from::DEFAULT,
    (kz: f64, m: c_long, kr: Complex, phi: f64, z: f64, k: Complex, pol: c_long) -> [Complex; 3] {
        let arguments = [kz.into(), kr, phi.into(), z.into(), k];
        wave(
            vectorwaves::Family::Cylindrical(radial(!REGULAR)),
            integer_wave_label(0, m, pol)?,
            arguments,
            true,
        )
    }
}

// Plane waves share one polarization across stride-0 wave vectors; POL `A`
// reads the helicity from a label operand.
#[allow(clippy::type_complexity)] // The operand types are the dtype row.
pub(super) unsafe extern "C" fn plane_wave_loop<T: Argument, const POL: u8, const LABELS: usize>(
    args: Args<([T; 3], [f64; 3], [c_long; LABELS]), [Complex; 3]>,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    const { assert!(LABELS == (POL == pol::A) as usize) };
    guard(false, || {
        // SAFETY: NumPy supplies operands of the dtype row derived from `Args`;
        // the vector output has the only core dimension.
        unsafe {
            let call = args.call(dimensions, steps);
            let (k, x, labels) = call.inputs;
            let pol = |i| {
                labels
                    .first()
                    .map_or(Ok(POL), |p| special::pol_index(p.read(i)))
            };
            let wave_vector = |i| k.map(|k| k.read(i).into());
            let direction = k.iter().all(|k| k.broadcast()) && labels.iter().all(|p| p.broadcast());
            let fixed = (call.len > 1 && direction)
                .then(|| treams_core::pw::polarization(wave_vector(0), pol(0)?, POL == pol::A))
                .transpose()?;
            call.store(parallel_from::DEFAULT, |i| {
                let k = wave_vector(i);
                let polarization = match fixed {
                    Some(p) => p,
                    None => treams_core::pw::polarization(k, pol(i)?, POL == pol::A)?,
                };
                let position = x.map(|x| Complex::from(x.read(i)));
                treams_core::pw::field_value(polarization, k, position)
            })
        }
    });
}

scalar_loops! {
    // Regular translations are dominated by the cheaper Bessel series.
    tl_vcw_loop[T: Argument, const REGULAR: bool]
        @ if REGULAR { parallel_from::PHASE } else { parallel_from::BESSEL },
    (kz: f64, mu: c_long, qz: f64, m: c_long, kr: T, phi: f64, z: f64) -> Complex {
        cw::tl_vcw(kz, mu, qz, m, [kr.into(), phi.into(), z.into()], radial(!REGULAR))
    }
    sw_rotate_loop @ parallel_from::DEFAULT,
    (lambda: c_long, mu: c_long, p: c_long, l: c_long, m: c_long, q: c_long,
     phi: f64, theta: f64, psi: f64) -> Complex {
        let destination = integer_wave_mode(lambda, mu, p)?;
        let source = integer_wave_mode(l, m, q)?;
        rotation::sw_rotate(destination, source, [phi, theta, psi])
    }
    cw_rotate_loop @ parallel_from::DEFAULT,
    (kz: f64, mu: c_long, p: c_long, qz: f64, m: c_long, q: c_long, phi: f64) -> Complex {
        rotation::cw_rotate(kz, mu, p, qz, m, q, phi)
    }
    cw_translate_loop[T: Argument, const REGULAR: bool] @ parallel_from::BESSEL,
    (kz: f64, mu: c_long, p: c_long, qz: f64, m: c_long, q: c_long, kr: T, phi: f64, z: f64)
        -> Complex {
        let arguments = [kr.into(), phi.into(), z.into()];
        cw::translate(kz, mu, p, qz, m, q, arguments, radial(!REGULAR))
    }
    pw_translate_loop[T: Argument] @ parallel_from::PHASE,
    (kx: T, ky: T, kz: T, x: f64, y: f64, z: f64) -> Complex {
        treams_core::pw::translate([kx.into(), ky.into(), kz.into()], [x, y, z])
    }
    pw_to_sw_loop[T: Argument, const HELICITY: bool] @ parallel_from::DEFAULT,
    (l: c_long, m: c_long, p: c_long, kx: T, ky: T, kz: T, q: c_long) -> Complex {
        let k = [kx.into(), ky.into(), kz.into()];
        treams_core::pw::to_sw(integer_wave_mode(l, m, p)?, k, special::pol_index(q)?, HELICITY)
    }
    pw_to_cw_loop[T: Argument] @ parallel_from::DEFAULT,
    (kz: f64, m: c_long, p: c_long, kx: f64, ky: T, qz: f64, q: c_long) -> Complex {
        let k = [kx.into(), ky.into(), qz.into()];
        treams_core::pw::to_cw(cylindrical_mode(kz, m, p)?, k, special::pol_index(q)?)
    }
    cw_to_sw_loop[const HELICITY: bool] @ parallel_from::DEFAULT,
    (l: c_long, m: c_long, p: c_long, kz: f64, mu: c_long, q: c_long, k: Complex) -> Complex {
        let destination = integer_wave_mode(l, m, p)?;
        cw::to_sw(destination, cylindrical_mode(kz, mu, q)?, k, HELICITY)
    }
    // One cyclic xyz permutation (TURNS 1) or its inverse (TURNS 2).
    pw_permute_loop[T: Argument, const HELICITY: bool, const TURNS: usize]
        @ parallel_from::DEFAULT,
    (kx: T, ky: T, kz: T, p: c_long, q: c_long) -> Complex {
        let k = [kx.into(), ky.into(), kz.into()];
        let (p, q) = (special::pol_index(p)?, special::pol_index(q)?);
        treams_core::pw::permute_xyz(k, p, q, TURNS, HELICITY)
    }
    sw_to_pw_loop[T: Argument, const HELICITY: bool] @ parallel_from::DEFAULT,
    (kx: f64, ky: f64, kz: T, p: c_long, l: c_long, m: c_long, q: c_long, area: f64)
        -> Complex {
        let k = [kx.into(), ky.into(), kz.into()];
        let mode = integer_wave_mode(l, m, q)?;
        treams_core::channels::sw_periodic_to_pw(mode, k, special::pol_index(p)?, area, HELICITY)
    }
    cw_to_pw_loop[T: Argument] @ parallel_from::DEFAULT,
    (kx: f64, ky: T, kz: f64, p: c_long, qz: f64, m: c_long, q: c_long, period: f64)
        -> Complex {
        let k = [kx.into(), ky.into(), kz.into()];
        let mode = cylindrical_mode(qz, m, q)?;
        treams_core::channels::cw_periodic_to_pw(mode, k, special::pol_index(p)?, period)
    }
    sw_to_cw_loop[const HELICITY: bool] @ parallel_from::DEFAULT,
    (kz: f64, mu: c_long, p: c_long, l: c_long, m: c_long, q: c_long, k: Complex, period: f64)
        -> Complex {
        let (destination, source) = (cylindrical_mode(kz, mu, p)?, integer_wave_mode(l, m, q)?);
        treams_core::sw::periodic_to_cw(destination, source, k, period, HELICITY)
    }
}

/// Spherical translation coefficients of one loop call, sharing one plan per
/// mode pair across broadcast labels.
///
/// # Safety
/// `call` holds the operands of a registered translation loop.
unsafe fn translations<
    X: Argument,
    Y: Argument,
    const LABELS: usize,
    const HELICITY: bool,
    const REGULAR: bool,
>(
    call: &Call<([c_long; LABELS], X, Y, f64), Complex>,
    modes: impl Fn([c_long; LABELS]) -> Result<(Mode, Mode)> + Sync,
) -> Result<()> {
    let (labels, kr, theta, phi) = call.inputs;
    // SAFETY: Guaranteed by the caller; elements are read below the loop count.
    unsafe {
        let modes = |i| modes(labels.map(|label| label.read(i)));
        let prepare = |i| {
            let (destination, source) = modes(i)?;
            PolarTranslation::new(destination, source, HELICITY, radial(!REGULAR))
        };
        let broadcast = call.len > 1 && labels.iter().all(|x| x.broadcast());
        let plan = broadcast.then(|| prepare(0)).transpose()?;
        call.store(parallel_from::BESSEL, |i| {
            let arguments = [kr.read(i).into(), theta.read(i).into(), phi.read(i).into()];
            if LABELS == 6 && PolarTranslation::is_self_term(radial(!REGULAR), arguments[0]) {
                // The polarized self coefficient is zero, but labels must be
                // valid for every broadcast layout, not only with a shared plan.
                if plan.is_none() {
                    let (destination, source) = modes(i)?;
                    destination.validate()?;
                    source.validate()?;
                }
                return Ok(Complex::default());
            }
            match &plan {
                Some(plan) => plan.value(arguments),
                None => prepare(i)?.value(arguments),
            }
        })
    }
}

// Special A and B coefficients from the parity wave (l, m, POL) to (lambda, mu,
// M): POL M gives A, N gives B.
pub(super) unsafe extern "C" fn tl_vsw_loop<T: Argument, const POL: u8, const REGULAR: bool>(
    args: Args<([c_long; 4], Complex, T, f64), Complex>,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    guard(true, || {
        let modes = |[lambda, mu, l, m]: [c_long; 4]| {
            let destination = integer_wave_mode(lambda, mu, c_long::from(pol::M))?;
            Ok((destination, integer_wave_mode(l, m, c_long::from(POL))?))
        };
        // SAFETY: NumPy supplies operands of the dtype row derived from `Args`.
        unsafe {
            translations::<_, _, 4, { poltype::PARITY }, REGULAR>(
                &args.call(dimensions, steps),
                modes,
            )
        }
    });
}

// Polarized coefficients from (l, m, q) to (lambda, mu, p).
pub(super) unsafe extern "C" fn sw_translate_loop<
    T: Argument,
    const HELICITY: bool,
    const REGULAR: bool,
>(
    args: Args<([c_long; 6], T, f64, f64), Complex>,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    guard(true, || {
        let modes = |[lambda, mu, p, l, m, q]: [c_long; 6]| {
            Ok((
                integer_wave_mode(lambda, mu, p)?,
                integer_wave_mode(l, m, q)?,
            ))
        };
        // SAFETY: NumPy supplies operands of the dtype row derived from `Args`.
        unsafe { translations::<_, _, 6, HELICITY, REGULAR>(&args.call(dimensions, steps), modes) }
    });
}

// Points use (d)->(d); vectors (d),(d)->(d) with positions in the input system.
pub(super) unsafe extern "C" fn coordinate_loop<
    V: Argument,
    const TRANSFORM: u8,
    const VECTORS: usize,
>(
    args: Args<([V; VECTORS], f64), V>,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    guard(false, || {
        // SAFETY: NumPy supplies operands of the dtype row derived from `Args`
        // with the core dimension (2 or 3) of the transform, whose component
        // strides follow the outer strides.
        unsafe {
            let call = args.call(dimensions, steps);
            let transform = crate::coordinates::TRANSFORMS[usize::from(TRANSFORM)].2;
            let dim = transform.dimension();
            let (vector, position) = call.inputs;
            // One component stride per operand: the vector, the position, the output.
            let vector_step = if VECTORS == 1 { call.core_stride(0) } else { 0 };
            let (position_step, output_step) =
                (call.core_stride(VECTORS), call.core_stride(VECTORS + 1));
            let output = call.output;
            drive(
                call.len,
                parallel_from::SERIAL,
                |i| {
                    let p = std::array::from_fn(|axis| {
                        if axis < dim {
                            position.read_at(i, offset(position_step, axis))
                        } else {
                            0.0
                        }
                    });
                    if let Some(vector) = vector.first() {
                        let v = std::array::from_fn(|axis| {
                            if axis < dim {
                                vector.read_at(i, offset(vector_step, axis)).into()
                            } else {
                                Complex::default()
                            }
                        });
                        special::coordinates::vector(v, p, transform)
                    } else {
                        Ok(special::coordinates::point(p, transform)?.map(Complex::from))
                    }
                },
                |i, value| {
                    for (axis, value) in value.into_iter().take(dim).enumerate() {
                        output.write_at(i, offset(output_step, axis), V::from_complex(value));
                    }
                },
            )
        }
    });
}
