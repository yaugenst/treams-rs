//! The Ewald sum: real-space part, reciprocal-space part and self term.
//!
//! Upstream: `lsum*`, `realsum*` and `recsum*` of `lattice/_esum.pyx`. The reduction
//! into a cell, the exact zeros by symmetry, the checks at small splits and the
//! spectral series of 1D spherical sums are treams-rs extensions.

use std::{
    cell::{Cell, OnceCell},
    f64::consts::PI,
};

use super::{
    SumPart,
    accuracy::{
        Checked, EARLY_FAILURE_LOSS, Rounding, TERM_ULPS, below_automatic, cancellation,
        derivative_scale, too_small,
    },
    cell::{BlochLattice, Reduction},
    evaluate,
    inputs::{Evaluation, Inputs},
    real::{RealKambe, real_term},
    reciprocal::{Diffraction, reciprocal_term, self_term},
    sheets::Split,
    shells::{
        NOT_CONVERGED, SHELL_TOLERANCE, Shells, is_shell_limit, norm, not_converged, peak_shells,
        real_shells, reciprocal_shells,
    },
    spectral::{
        REAL_ULPS, SPECTRAL_FIRST_W, SPECTRAL_SWITCH_LOSS, prefer_spectral_sw1d, spectral_sw1d,
    },
    wave::Family,
};
use crate::{Complex, Error, Result, numerics::Jet, special::SmallSplitKambe};

/// The Ewald parts selected by `evaluation` with split `eta`. Both sums run over shells of
/// reduced bases, and the reciprocal sum over the Bloch vector reduced into a cell
/// of the reduced reciprocal basis, so their results do not depend on the given basis
/// or on the Bloch vector's cell.
pub(super) fn ewald<const N: usize>(
    wave: Family,
    lattice: &BlochLattice,
    mut inputs: Inputs<N>,
    eta: Complex,
    evaluation: Evaluation,
) -> Result<Jet<N>> {
    let CellReduction {
        phase,
        direct_rows: direct,
        reciprocal_rows: rows,
        kpar,
        kpar_reduced: reduced,
        given_shift,
    } = reduce_to_cell(lattice, &mut inputs);
    let inputs = &inputs;
    let dim = inputs.dim;
    let symmetry = Symmetry::of(wave, inputs, &direct, &kpar);
    if symmetry.vanishes && (N == 0 || matches!(evaluation, Evaluation::EtaDerivative)) {
        return Ok(Jet::default());
    }
    // Only splits below every automatic one check the cancellation of their parts.
    let small = below_automatic(eta);
    let checked = symmetry.checked(inputs, evaluation);
    let real_limit = real_shells(lattice, inputs.k.value, eta);
    // The shells before which neither part may stop (see `peak_shells`): the real-space
    // terms fall like `exp(-Re((k eta)^2) |R|^2 / 2)`, the reciprocal ones like
    // `exp(-Re(1 / (k eta)^2) |q + G|^2 / 2)`.
    let square = (inputs.k.value * eta).powi(2);
    let real_floor = peak_shells(
        wave,
        dim,
        N > 0,
        square.re,
        norm(inputs.r.map(|r| r.value.re)),
        (&lattice.reciprocal, lattice.direct_reduction.as_ref()),
    );
    let reciprocal_floor = peak_shells(
        wave,
        dim,
        N > 0,
        square.inv().re,
        norm(kpar.map(|q| q.value.re)),
        (&lattice.direct, lattice.reciprocal_reduction.as_ref()),
    );
    // Complete sums may settle their real-space shells (see `Shells::sum`) and fail
    // early (see `EARLY_FAILURE_LOSS`).
    let settle = small && matches!(evaluation, Evaluation::Part(SumPart::Full));
    let early = settle && probes::early_failure_enabled();
    let automatic_split = AutomaticSplit {
        wave,
        k: inputs.k.value,
        lattice,
        shift: given_shift,
        evaluation,
    };
    let monitor = SmallSplitMonitor::new(
        (small && !matches!(evaluation, Evaluation::EtaDerivative))
            .then(|| SmallSplitKambe::new(eta)),
        early,
        automatic_split,
        &checked,
    );
    let real_summands = RealSummands {
        wave,
        inputs,
        direct: &direct,
        eta,
        evaluation,
        monitor: &monitor,
    };
    let real_summand = |n, bounds: &mut Rounding<N>| real_summands.at(n, bounds);
    let real_part = || {
        Shells::sum(
            dim,
            [real_limit, real_floor],
            &mut { real_summand },
            settle,
            small,
        )
    };
    if !evaluation.reciprocal() {
        let real = real_part()?;
        let sum = phase * real.sum;
        if small {
            cancellation(&[&real], &Jet::default(), 0.0, &phase, &sum, &checked)?;
        }
        return Ok(symmetry.zeroed(sum));
    }
    // 1D spherical sums may take the spectral series of the complete sum (less the
    // real part for the reciprocal part alone), evaluated once.
    let sw1d = match wave {
        Family::Spherical { l, m } if dim == 1 && probes::spectral_series_enabled() => Some((l, m)),
        _ => None,
    };
    let series_cell = OnceCell::new();
    let series = || {
        *series_cell.get_or_init(|| {
            let (mut series, mut bound) = spectral_sw1d(
                sw1d?,
                inputs.k,
                inputs.r,
                kpar[0],
                rows[0][0],
                inputs.measure,
            )?;
            if !evaluation.real() {
                let real = real_part().ok()?.sum;
                series -= real;
                bound += Rounding::relative(&real, REAL_ULPS);
            }
            Some((phase * series, bound.times(&phase)))
        })
    };
    // Far off the axis the Ewald sum cancels (see `sum`), so a series that keeps its
    // accuracy is taken without it.
    let rho = (inputs.r[0].value.re).hypot(inputs.r[1].value.re);
    if sw1d.is_some()
        && inputs.k.value.norm() * rho * eta.norm() >= SPECTRAL_FIRST_W
        && let Some((series, bound)) = series()
        && f64::EPSILON * bound.value <= SPECTRAL_SWITCH_LOSS * series.value.norm().max(1.0)
    {
        return Ok(series);
    }
    let real = if evaluation.real() {
        real_part().map(Some)
    } else {
        Ok(None)
    };
    let [maximum, extension] =
        reciprocal_shells(lattice, inputs.k.value, eta, kpar.map(|q| q.value.re));
    // A 1D spherical sum whose reciprocal part passes `extension` shells takes its
    // spectral series where that is accurate. Calibrated: for a degree-3 chain the series
    // lies 2e-15 from the sum at the automatic split, and the Ewald sum continued to the
    // larger radius 3e-13.
    let maximum = if sw1d.is_some() { extension } else { maximum };
    // Absolute rounding error bounds of the reciprocal part in units of epsilon.
    let mut rounding = Rounding::default();
    let split = Split::new(inputs.k.value, eta);
    let reciprocal_summands = ReciprocalSummands {
        wave,
        inputs,
        rows: &rows,
        kpar: &kpar,
        kpar_reduced: reduced,
        eta,
        split: &split,
    };
    let mut reciprocal_summand =
        |n: [i64; 3], _: &mut Rounding<N>| reciprocal_summands.at(n, &mut rounding);
    let reciprocal = Shells::sum(
        dim,
        [maximum, reciprocal_floor],
        &mut reciprocal_summand,
        false,
        small,
    );
    let self_term = if inputs.r.iter().all(|r| r.value == Complex::default()) {
        self_term(wave, inputs.k, inputs.r, eta)
    } else {
        Jet::default()
    };
    let total = |real: &Option<Shells<N>>, reciprocal: &Shells<N>| {
        phase * (real.map_or_else(Jet::default, |real| real.sum) + reciprocal.sum + self_term)
    };
    // Below every automatic split both parts continue to `SHELL_TOLERANCE` of the
    // complete sum, which fails where they cancel beyond use (and goes to
    // `prefer_spectral_sw1d` then).
    let mut complete = |mut real: Option<Shells<N>>, mut reciprocal: Shells<N>| {
        let scale = Some(total(&real, &reciprocal).norm().max(1.0));
        let mut tail = 0.0;
        if !reciprocal.extend(dim, extension, &mut reciprocal_summand, scale)? {
            tail += reciprocal.tail();
        }
        if let Some(real) = &mut real
            && !real.extend(dim, real_limit, &mut { real_summand }, scale)?
        {
            tail += real.tail();
        }
        let sum = total(&real, &reciprocal);
        let parts: Vec<&Shells<N>> = real.iter().chain([&reciprocal]).collect();
        let rounding = cancellation(&parts, &self_term, tail, &phase, &sum, &checked)?;
        Ok((sum, rounding, tail))
    };
    // The rounding the cancelling parts predict and what their last shells may leave
    // out, for `prefer_spectral_sw1d`.
    let (mut predicted, mut truncation) = (Rounding::default(), 0.0);
    let settled = matches!(&real, Ok(Some(real)) if real.settled);
    let result = match (real, reciprocal) {
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        (Ok(real), Ok(reciprocal)) if small => {
            complete(real, reciprocal).map(|(sum, rounding, tail)| {
                (predicted, truncation) = (rounding, tail);
                sum
            })
        }
        (Ok(real), Ok(reciprocal)) => Ok(total(&real, &reciprocal)),
    };
    let result = if sw1d.is_none() {
        result.map(|sum| symmetry.zeroed(sum))
    } else {
        let mut rounding = rounding.times(&phase);
        rounding += predicted;
        rounding.value += truncation / f64::EPSILON;
        prefer_spectral_sw1d(result, rounding, series, inputs.k.value)
            .map(|sum| symmetry.zeroed(sum))
    };
    if !settled {
        return result;
    }
    verify_settled(result?, predicted.value, automatic_split)
}

/// What the real-space terms of one sum read.
struct RealSummands<'a, const N: usize> {
    wave: Family,
    inputs: &'a Inputs<N>,
    /// The reduced direct rows.
    direct: &'a [[Jet<N>; 3]; 3],
    eta: Complex,
    evaluation: Evaluation,
    monitor: &'a SmallSplitMonitor<'a, N>,
}

impl<const N: usize> RealSummands<'_, N> {
    /// The real-space term at the lattice point with the coordinates `n`, adding its
    /// rounding bounds below every automatic split to `bounds`. Always inlined: it is
    /// the body of the term loop of [`Shells::sum`].
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn at(&self, n: [i64; 3], bounds: &mut Rounding<N>) -> Result<Jet<N>> {
        let Self {
            wave,
            inputs,
            direct,
            eta,
            evaluation,
            monitor,
        } = *self;
        let point = inputs.point(direct, n);
        let shift = inputs.image(&point);
        if shift.iter().all(|r| r.value == Complex::default()) {
            return Ok(Jet::default());
        }
        probes::count_real_term();
        // Below every automatic split only: the rounding of the largest term so far and
        // the bound of this one.
        let mut small = monitor
            .kambe
            .as_ref()
            .map(|kambe| (kambe, monitor.largest.get(), Rounding::default()));
        let kambe = match (&mut small, evaluation) {
            (Some((kambe, current, bound)), _) => RealKambe::SmallSplit {
                kambe,
                largest: current,
                bound,
            },
            (None, Evaluation::EtaDerivative) => RealKambe::EtaDerivative,
            (None, _) => RealKambe::Plain,
        };
        let term = real_term(wave, inputs.k, shift, eta, kambe);
        let phase = (Complex::i() * inputs.phase(&point)).exp();
        let Some((_, current, bound)) = small else {
            return Ok(term * phase);
        };
        monitor
            .largest
            .set(current.max(Rounding::relative(&term, 1.0)));
        let (term, bound) = (term * phase, bound.times(&phase));
        *bounds += bound;
        monitor.add_to_running(&term, bound)?;
        Ok(term)
    }
}

/// What the reciprocal-space terms of one sum read.
struct ReciprocalSummands<'a, const N: usize> {
    wave: Family,
    inputs: &'a Inputs<N>,
    /// The reduced reciprocal rows.
    rows: &'a [[Jet<N>; 3]; 3],
    /// The Bloch vector reduced into a cell of `rows`.
    kpar: &'a [Jet<N>; 3],
    /// Whether the reduction moved the Bloch vector.
    kpar_reduced: bool,
    eta: Complex,
    split: &'a Split,
}

impl<const N: usize> ReciprocalSummands<'_, N> {
    /// The reciprocal-space term at the reciprocal lattice point with the coordinates
    /// `n`, adding its absolute rounding bounds in units of epsilon to `rounding`. Always
    /// inlined: it is the body of the term loop of [`Shells::sum`].
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn at(&self, n: [i64; 3], rounding: &mut Rounding<N>) -> Result<Jet<N>> {
        let Self {
            wave,
            inputs,
            rows,
            kpar,
            kpar_reduced,
            eta,
            split,
        } = *self;
        let dim = inputs.dim;
        let vector = inputs.point(rows, n);
        let mut q = [Jet::default(); 3];
        for j in 0..dim {
            q[inputs.axes[j]] = vector[j] + kpar[j];
        }
        let phase = (-Complex::i()
            * q.into_iter()
                .zip(inputs.r)
                .map(|(q, r)| q * r)
                .sum::<Jet<N>>())
        .exp();
        let mut bound = Rounding::default();
        let term = reciprocal_term(
            wave,
            dim,
            inputs.k,
            Diffraction {
                q,
                exact: !kpar_reduced && n == [0; 3],
            },
            inputs.r,
            eta,
            split,
            inputs.measure,
            &mut bound,
        )?;
        *rounding += bound.times(&phase);
        Ok(term * phase)
    }
}

/// The shift and the Bloch vector of one sum, each reduced into a cell of its reduced
/// basis.
struct CellReduction<const N: usize> {
    /// The Bloch phase `exp(-i q · R)` that compensates subtracting the lattice point `R`
    /// from the shift.
    phase: Jet<N>,
    /// The reduced direct rows.
    direct_rows: [[Jet<N>; 3]; 3],
    /// The reduced reciprocal rows.
    reciprocal_rows: [[Jet<N>; 3]; 3],
    /// The Bloch vector in lattice coordinates, reduced into a cell of `reciprocal_rows`.
    kpar: [Jet<N>; 3],
    /// Whether the reduction moved the Bloch vector.
    kpar_reduced: bool,
    /// The shift as given, for the sum at the automatic split ([`AutomaticSplit`]).
    given_shift: [f64; 3],
}

/// Reduces the shift of `inputs` into a cell of the reduced direct basis, in place, and
/// the Bloch vector into a cell of the reduced reciprocal basis.
fn reduce_to_cell<const N: usize>(
    lattice: &BlochLattice,
    inputs: &mut Inputs<N>,
) -> CellReduction<N> {
    let dim = inputs.dim;
    // Coordinates of a lattice-frame vector in the basis `rows`, whose dual is `dual`.
    let coordinates = |dual: &[[f64; 3]; 3], vector: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|i| {
            if i < dim {
                (0..dim).map(|j| dual[i][j] * vector[j]).sum::<f64>() / (2.0 * PI)
            } else {
                0.0
            }
        })
    };
    // A Bloch phase compensates the shift's reduction. The cell point is combined from
    // the given rows, so a shift equal to one of them reduces to exactly zero.
    let direct_reduction = lattice.direct_reduction.as_ref();
    let shift = std::array::from_fn(|j| inputs.r[inputs.axes[j]].value.re);
    let cells = Reduction::given(
        direct_reduction,
        Reduction::round(
            direct_reduction,
            coordinates(&lattice.reciprocal, shift),
            dim,
        ),
        dim,
    );
    let given_shift = inputs.r.map(|r| r.value.re);
    let cell = inputs.combination(&inputs.direct, cells);
    for (&axis, &cell) in inputs.axes.iter().zip(&cell).take(dim) {
        inputs.r[axis] -= cell;
    }
    let phase = (-Complex::i() * inputs.phase(&cell)).exp();
    let direct_rows = inputs.reduce(&inputs.direct, direct_reduction);
    let reciprocal_reduction = lattice.reciprocal_reduction.as_ref();
    let reciprocal_rows = inputs.reduce(&inputs.reciprocal, reciprocal_reduction);
    let cells = Reduction::round(
        reciprocal_reduction,
        coordinates(&lattice.direct, lattice.kpar),
        dim,
    );
    let mut kpar = inputs.kpar;
    let kpar_reduced = cells.iter().any(|&c| c != 0.0);
    if kpar_reduced {
        let point = inputs.combination(&reciprocal_rows, cells);
        for (kpar, point) in kpar.iter_mut().zip(point).take(dim) {
            *kpar -= point;
        }
    }
    CellReduction {
        phase,
        direct_rows,
        reciprocal_rows,
        kpar,
        kpar_reduced,
        given_shift,
    }
}

/// A sum that vanishes by symmetry ([`vanishes`]) is exactly zero, as are its derivatives
/// that move `k` alone; jets keep the other derivatives and their check of the
/// cancellation.
struct Symmetry<const N: usize> {
    /// Whether the sum vanishes by symmetry.
    vanishes: bool,
    /// Whether each jet slot moves `k` alone, with the shift, the Bloch vector and the
    /// lattice vectors fixed.
    k_alone: [bool; N],
}

impl<const N: usize> Symmetry<N> {
    /// The symmetry of the sum with the reduced direct rows `direct` and the reduced
    /// Bloch vector `kpar`.
    fn of(wave: Family, inputs: &Inputs<N>, direct: &[[Jet<N>; 3]; 3], kpar: &[Jet<N>; 3]) -> Self {
        Self {
            vanishes: vanishes(wave, inputs, direct, kpar),
            k_alone: std::array::from_fn(|slot| {
                inputs
                    .r
                    .iter()
                    .chain(&inputs.kpar)
                    .chain(inputs.direct.iter().flatten())
                    .all(|input| input.derivative[slot] == Complex::default())
            }),
        }
    }

    /// `sum` with the components that vanish by symmetry set to exactly zero.
    fn zeroed(&self, mut sum: Jet<N>) -> Jet<N> {
        if self.vanishes {
            sum.value = Complex::default();
            for (derivative, &k_alone) in sum.derivative.iter_mut().zip(&self.k_alone) {
                if k_alone {
                    *derivative = Complex::default();
                }
            }
        }
        sum
    }

    /// The components whose cancellation a sum below every automatic split checks: those
    /// that [`Self::zeroed`] keeps, without the split derivative of the parts.
    fn checked(&self, inputs: &Inputs<N>, evaluation: Evaluation) -> Checked<N> {
        Checked {
            value: !self.vanishes,
            units: std::array::from_fn(|slot| {
                if matches!(evaluation, Evaluation::EtaDerivative)
                    || (self.vanishes && self.k_alone[slot])
                {
                    None
                } else {
                    Some(derivative_scale(slot, inputs.dim, inputs.k.value))
                }
            }),
        }
    }
}

/// The same sum at the automatic split, against which sums below every automatic split
/// fail early ([`SmallSplitMonitor::hopeless`]) or verify their settled shells
/// ([`verify_settled`]).
#[derive(Clone, Copy)]
struct AutomaticSplit<'a> {
    wave: Family,
    k: Complex,
    lattice: &'a BlochLattice,
    /// The shift as given, before its reduction into a cell.
    shift: [f64; 3],
    evaluation: Evaluation,
}

impl AutomaticSplit<'_> {
    fn sum<const N: usize>(&self) -> Result<Jet<N>> {
        evaluate::<N>(
            self.wave,
            self.k,
            self.lattice,
            self.shift,
            Complex::default(),
            self.evaluation,
        )
    }
}

/// The real-space terms below every automatic split: the Kambe integrals that keep their
/// rounding below that of the largest term so far, and the early failure of complete
/// sums ([`EARLY_FAILURE_LOSS`]).
struct SmallSplitMonitor<'a, const N: usize> {
    /// The Kambe integrals of the real-space terms below every automatic split, except
    /// for the split derivative.
    kambe: Option<SmallSplitKambe>,
    /// The largest moduli of the value and derivatives of a real-space term so far,
    /// whose rounding the closed forms of the later terms may reach.
    largest: Cell<Rounding<N>>,
    /// Whether the sum fails early once the rounding its real-space terms predict is
    /// [`hopeless`](Self::hopeless).
    early: bool,
    /// The rounding the real-space terms predict so far.
    running: Cell<Rounding<N>>,
    /// The sum at the automatic split, on whose scales the sum fails early, evaluated
    /// once; `None` where it fails.
    automatic_sum: OnceCell<Option<Jet<N>>>,
    automatic_split: AutomaticSplit<'a>,
    checked: &'a Checked<N>,
}

impl<'a, const N: usize> SmallSplitMonitor<'a, N> {
    fn new(
        kambe: Option<SmallSplitKambe>,
        early: bool,
        automatic_split: AutomaticSplit<'a>,
        checked: &'a Checked<N>,
    ) -> Self {
        Self {
            kambe,
            largest: Cell::new(Rounding::default()),
            early,
            running: Cell::new(Rounding::default()),
            automatic_sum: OnceCell::new(),
            automatic_split,
            checked,
        }
    }

    /// With [`Self::early`], adds the rounding of the real-space term `term` and its bound
    /// `bound` to the running rounding and checks whether it is hopeless.
    fn add_to_running(&self, term: &Jet<N>, bound: Rounding<N>) -> Result<()> {
        if self.early {
            let mut total = self.running.get();
            total += Rounding::relative(term, TERM_ULPS);
            total += bound;
            self.running.set(total);
            self.hopeless(&total)?;
        }
        Ok(())
    }

    /// Fails where the rounding `bound` exceeds [`EARLY_FAILURE_LOSS`] of the scale of a
    /// checked component of the sum at the automatic split.
    fn hopeless(&self, bound: &Rounding<N>) -> Result<()> {
        let checked = self.checked;
        let beyond = |bound: f64, scale: f64| f64::EPSILON * bound > EARLY_FAILURE_LOSS * scale;
        // The scales are at least 1 for the value and `s` for a derivative, so smaller
        // losses need no automatic sum.
        let slots = || bound.derivative.iter().zip(checked.units);
        let candidate = (checked.value && beyond(bound.value, 1.0))
            || slots().any(|(&bound, unit)| unit.is_some_and(|unit| beyond(bound, unit)));
        if !candidate {
            return Ok(());
        }
        let Some(automatic) = self
            .automatic_sum
            .get_or_init(|| self.automatic_split.sum::<N>().ok())
        else {
            return Ok(());
        };
        let scale = automatic.value.norm().max(1.0);
        if checked.value && beyond(bound.value, scale) {
            return Err(too_small(f64::EPSILON * bound.value / scale, false));
        }
        for ((&bound, unit), derivative) in slots().zip(&automatic.derivative) {
            if let Some(unit) = unit {
                let scale = scale.mul_add(unit, derivative.norm());
                if beyond(bound, scale) {
                    return Err(too_small(f64::EPSILON * bound / scale, true));
                }
            }
        }
        Ok(())
    }
}

/// `sum`, whose real-space shells settled at their limit ([`Shells::settled`]), if it
/// agrees with the sum at the automatic split within the rounding `predicted` that its
/// parts predict for the value (without the truncation that the comparison verifies)
/// plus twice the convergence tolerance of each shell sum.
fn verify_settled<const N: usize>(
    sum: Jet<N>,
    predicted: f64,
    automatic_split: AutomaticSplit<'_>,
) -> Result<Jet<N>> {
    // A larger split cannot help where the automatic one does not converge either.
    let automatic = automatic_split.sum::<N>().map_err(|error| {
        if is_shell_limit(&error) {
            Error::NotConverged(format!(
                "{NOT_CONVERGED}, neither at this split nor at the automatic one"
            ))
        } else {
            error
        }
    })?;
    let scale = automatic.value.norm().max(1.0);
    let relative = f64::EPSILON.mul_add(predicted / scale, 4.0 * SHELL_TOLERANCE);
    let agrees = |a: Complex, b: Complex| (a - b).norm() <= relative * (b.norm() + scale);
    if agrees(sum.value, automatic.value)
        && sum
            .derivative
            .iter()
            .zip(&automatic.derivative)
            .all(|(&a, &b)| agrees(a, b))
    {
        Ok(sum)
    } else {
        Err(not_converged())
    }
}

/// Whether the sum vanishes by symmetry.
///
/// That holds for odd degrees of spherical and odd orders of cylindrical waves where the
/// reduced Bloch vector `kpar` is exactly zero and twice the reduced shift is exactly one
/// of the combinations of the reduced rows `direct` with coefficients in `{-1, 0, 1}`
/// (`2 r` lies in their cell).
///
/// # Why
///
/// Inversion maps the images `-r - R` onto themselves and changes the sign of the waves,
/// so every Ewald part and the self term vanish at any split and `k`, while the computed
/// parts would leave rounding of the order of the terms. The derivatives that move `k`
/// alone vanish too; the others keep the check of the predicted loss ([`cancellation`]).
pub(super) fn vanishes<const N: usize>(
    wave: Family,
    inputs: &Inputs<N>,
    direct: &[[Jet<N>; 3]; 3],
    kpar: &[Jet<N>; 3],
) -> bool {
    let odd = match wave {
        Family::Spherical { l, .. } => l % 2 != 0,
        Family::Cylindrical { m } => m % 2 != 0,
    };
    let dim = inputs.dim;
    let lattice_axis = |axis: &usize| inputs.axes[..dim].contains(axis);
    if !odd
        || kpar[..dim].iter().any(|q| q.value != Complex::default())
        || (0..3).any(|axis| !lattice_axis(&axis) && inputs.r[axis].value != Complex::default())
    {
        return false;
    }
    let twice: [f64; 3] = std::array::from_fn(|j| {
        if j < dim {
            2.0 * inputs.r[inputs.axes[j]].value.re
        } else {
            0.0
        }
    });
    // Integer coefficients in {-1, 0, 1} for every reduced row.
    let count = 3_usize.pow(u32::try_from(dim).unwrap_or(3));
    (0..count).any(|index| {
        let mut rest = index;
        let coefficients: [f64; 3] = std::array::from_fn(|i| {
            if i >= dim {
                return 0.0;
            }
            let digit = rest % 3;
            rest /= 3;
            match digit {
                0 => 0.0,
                1 => 1.0,
                _ => -1.0,
            }
        });
        let point = inputs.combination(direct, coefficients);
        (0..dim).all(|j| point[j].value.re == twice[j])
    })
}

#[cfg(test)]
pub(crate) mod probes {
    //! Test probes of the Ewald sum: switches that turn off one decision of [`ewald`], a
    //! count of its real-space terms, and the reduced integrals and the spectral series
    //! on their own. Builds without tests take the no-op twins of the reads.
    //!
    //! [`ewald`]: super::ewald

    use std::{cell::Cell, thread::LocalKey};

    use super::super::{
        Derivatives,
        cell::BlochLattice,
        inputs::{Inputs, unpack},
        reduced::{Reduced, reduced_kambe},
        sheets::{Split, lower_side, split_sheet},
        spectral::spectral_sw1d,
        wave::Family,
    };
    use crate::{Complex, numerics::Jet};

    thread_local! {
        static EWALD_ONLY: Cell<bool> = const { Cell::new(false) };
        static LATE_FAILURE: Cell<bool> = const { Cell::new(false) };
        static REAL_TERMS: Cell<u64> = const { Cell::new(0) };
    }

    /// Restores a switch when dropped, so a panicking test leaves it as it was.
    struct Restore {
        switch: &'static LocalKey<Cell<bool>>,
        previous: bool,
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            self.switch.set(self.previous);
        }
    }

    /// Runs `run` with `switch` on in this thread.
    fn with_switch<T>(switch: &'static LocalKey<Cell<bool>>, run: impl FnOnce() -> T) -> T {
        let _restore = Restore {
            switch,
            previous: switch.replace(true),
        };
        run()
    }

    /// Runs `run` with this thread's 1D spherical sums kept on the Ewald sum: they skip
    /// the spectral series, so tests can compare the two.
    pub(crate) fn ewald_only<T>(run: impl FnOnce() -> T) -> T {
        with_switch(&EWALD_ONLY, run)
    }

    /// Runs `run` with this thread's complete sums below every automatic split skipping
    /// the early failure on [`EARLY_FAILURE_LOSS`], so tests can compare them with the
    /// check of the complete sum alone.
    ///
    /// [`EARLY_FAILURE_LOSS`]: super::super::accuracy::EARLY_FAILURE_LOSS
    pub(crate) fn late_failure<T>(run: impl FnOnce() -> T) -> T {
        with_switch(&LATE_FAILURE, run)
    }

    /// A count of the real-space terms this thread's Ewald sums evaluate, which tests read
    /// to see how far a sum went before it stopped.
    pub(crate) struct RealTerms {
        start: u64,
    }

    impl RealTerms {
        /// The real-space terms evaluated since [`real_terms`] created this count.
        pub(crate) fn count(&self) -> u64 {
            REAL_TERMS.get() - self.start
        }
    }

    /// Starts counting the real-space terms of this thread's Ewald sums.
    pub(crate) fn real_terms() -> RealTerms {
        RealTerms {
            start: REAL_TERMS.get(),
        }
    }

    /// Whether 1D spherical sums may take the spectral series (see [`ewald_only`]).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(super) fn spectral_series_enabled() -> bool {
        !EWALD_ONLY.get()
    }

    /// Whether complete sums may fail early (see [`late_failure`]).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(super) fn early_failure_enabled() -> bool {
        !LATE_FAILURE.get()
    }

    /// Counts one real-space term (see [`real_terms`]).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(super) fn count_real_term() {
        REAL_TERMS.set(REAL_TERMS.get() + 1);
    }

    /// `F_n(v, t)` of one reciprocal point, of squared transverse modulus `q2` at the
    /// distance `s` from the lattice plane or axis (`t = (k s eta)^2`), from its series in
    /// `t` and from Kambe integrals, each with its rounding bound, on the sheet of the
    /// automatic split.
    pub(crate) fn reduced_paths(
        twice_n: i32,
        k: Complex,
        q2: f64,
        eta: Complex,
        distance: f64,
    ) -> [(Complex, f64); 2] {
        let v = lower_side((q2 / (k * k) - 1.0) / (2.0 * eta * eta));
        let t = (k * distance * eta).powi(2);
        let sheet = Some(split_sheet(&Split::new(k, eta), q2.into()));
        let mut reduced = Reduced::<0>::new(Jet::constant(v), Jet::constant(t), sheet);
        let kambe = if twice_n % 2 == 0 {
            reduced_kambe(twice_n, v, t.sqrt(), sheet)
        } else {
            (reduced.by_even_recurrence(twice_n), 0.0)
        };
        [reduced.series(twice_n), kambe]
    }

    /// The spectral series of a 1D spherical sum with its derivatives in every input, as
    /// [`derivatives`] returns them, and the rounding bounds of each component in units
    /// of epsilon (as real parts, in the same layout), or `None` where [`spectral_sw1d`]
    /// is not available.
    ///
    /// [`derivatives`]: super::super::derivatives
    pub(crate) fn spectral_sw1d_derivatives(
        (l, m): (i32, i32),
        k: Complex,
        lattice: &BlochLattice,
        r: [f64; 3],
    ) -> Option<(Derivatives, Derivatives)> {
        let inputs = Inputs::<6>::new(Family::Spherical { l, m }, k, lattice, r, true, true);
        let (series, bound) = spectral_sw1d(
            (l, m),
            inputs.k,
            inputs.r,
            inputs.kpar[0],
            inputs.reciprocal[0][0],
            inputs.measure,
        )?;
        let bound = Jet {
            value: bound.value.into(),
            derivative: bound.derivative.map(Complex::from),
        };
        Some((unpack(series, 1), unpack(bound, 1)))
    }
}

/// The reads of the test probes in builds without tests: constants that fold away.
#[cfg(not(test))]
mod probes {
    /// Always true: 1D spherical sums may take the spectral series.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(super) const fn spectral_series_enabled() -> bool {
        true
    }

    /// Always true: complete sums may fail early.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(super) const fn early_failure_enabled() -> bool {
        true
    }

    /// Counts nothing.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(super) const fn count_real_term() {}
}
