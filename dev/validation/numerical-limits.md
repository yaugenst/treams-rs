# Numerical and platform limits

## How to read these limits

Accepted mode labels do not guarantee accuracy over the whole range. Extreme
orders and arguments, resonant conditioning and many layers need their own
convergence checks. Passing tests cover only the parameter ranges tested.

Errors are relative to max(|S|, 1) for a sum or value S. For a derivative dS they
are relative to |dS| + max(|S|, 1) s, with s = |k| for derivatives with respect
to a length and 1 / |k| for derivatives with respect to an inverse length.

## Special functions and waves

- **Fractional Legendre** functions support 0 < degree <= 128 and
  |order| <= degree. Their implementation uses scaled DLMF series and
  recurrences, retaining the fractional offset of degrees close to an integer.
  Values or derivatives outside the float64 range raise an error.
    - `lpmv` keeps the zero extension of treams at |order| > degree instead of
      the general Ferrers function.
    - Its real argument returns float64 for Python scalars and arrays alike.
    - Outside [-1, 1] it returns the real analytic value where one exists: even
      orders, including 0, and the zero extension. Odd orders with
      |order| <= degree have an imaginary continuation and raise `ValueError`;
      pass a complex argument.
    - Fractional complex arguments and fractional pi/tau are not implemented.
- **Legendre next to the poles**: the associated Legendre functions and pi form
  sin(theta) as `sqrt(1 - z * z)`, which cancels. The relative error grows roughly
  like |m| eps / (1 - |z|). For degree and |order| 12 it is 6.6e-11 at
  z = 1 - 1e-6 and 3.3e-9 at 1 - 1e-8, as in the `lpmv` of SciPy and treams.
  `sqrt((1 - z) (1 + z))` would avoid it.
- **Wigner 3j** symbols differ from treams on purpose where its recurrence is
  unstable: treams errs by up to 2e-3 below degree 90 and by a factor of 1800 at
  degree 260 (see [differences from treams](../coming-from-treams/differences.md)).
  The native recurrence continues in each direction while the symbols grow.
  Sampled extreme-order symbols up to degree 260 stay within 5e-13 of exact Racah
  values after scaling by sqrt(2 j3 + 1).
- **Spherical Bessel functions on the negative real axis** take the limit from
  above the axis, as the AMOS functions do, for either sign of zero in the
  imaginary part (a float argument is `+0j`). The parity identities therefore hold
  for the Bessel and Hankel functions, their derivatives and singular polar
  translations.
- **Spherical translations** evaluate their angular factor on the unit sphere and
  do not depend on the unit of length. Rust properties rescale displacements and
  wavenumbers by 1e-12 to 1e12 up to degree 80; the Python tests use 1e-150 to
  1e150.
    - Displacements whose components are all subnormal count as coincident
      positions: regular translations take their exact limits at zero
      displacement and singular ones are zero.
    - Singular translations over other displacements below about 1e-154 raise the
      Hankel-function overflow.
- **Spherical fields** evaluate their solid harmonics in bounded internal length
  units, then return their analytic derivatives in the caller's units. Reference,
  Maxwell and gradient tests cover degrees up to 40 with length units from 1e-150
  to 1e150. Fields whose dimensionless Bessel values exceed float64 still fail;
  a change of units cannot make those values representable.
- **Incomplete gamma functions** Gamma(n, z) are checked against these
  references:
    - 1494 40-digit mpmath values at 1e-13 relative: half-integer degrees in
      [-8, 8], Re z in [-60, 60], |Im z| <= 20, both sides of the branch cut;
    - every degree |n| <= 128 next to the negative real axis
      (|z| + Re z <= 1, |z| <= 600) below 1e-13;
    - two 60-digit sweeps within 1.5e-13 on the scale
      |Gamma(n, z)| + |z^(n-1) e^-z| (relative for n < 1): 38859 points over
      |n| <= 128 with |z| log-uniform in [1e-3, 600], and 39395 positive-degree
      points with |z| / n in (0, 1.5].

    For positive degrees with |z| < n, a Kummer sum of the lower function replaces
    the cancelling recurrence and power series. The largest errors, 1.2e-13 to
    1.4e-13, come from the continued fraction at degrees below -88 with |z| near
    |n|, and at |z| near 1.14 n next to the positive real axis. These checks do not
    cover arguments outside these ranges.

- **Kambe integrals** are checked against 181 quadrature values of orders -14 to 7
  with real eta at 1e-13 relative. They are also checked against 1249 70-digit
  values at the arguments of the lattice sums, x = sqrt(-2 v w^2) and eta = -i/w
  with |w| in [0.5, 6] ([references](https://github.com/yaugenst/treams-rs/tree/79a92e31fc29542ef0de77664ae887ff4eff14e4/crates/treams-core/references)).
    - Even orders stay within 1e-11 there.
    - Odd orders keep what the gamma series keeps of their base pair, which
      cancels by up to e^(|w|^2). Over a 20808-value grid, at |w| near 3, 4, 5
      and 6 the relative error reached 3e-11, 8e-9, 2e-5 and 0.14 (medians 6e-15,
      1e-13, 1e-11 and 2e-9), the largest for values below 1e-10.
    - Where |1/(2 eta^2)| exceeds 715 the series terms overflow and the integrals
      are non-finite.
    - At far evanescent reciprocal orders (imaginary x with |x| / |w| from 13,
      |w| >= 4.96) the integrals are finite and within their reported bound. The
      bound grows like e^(w^2): 1e-4 at w = 5, no digits at w = 8 and 12, as the
      `special.intkambe` docstring states. The lattice sums reach these arguments
      only from w = 4.96, where they take the spectral series.
    - The public `intkambe` of order -2 at eta = -i / w keeps its closed form,
      which cancels by 1 / |w| as in treams (1e-7 at |w| = 1e-8). The lattice
      sums do not reach it there.

    These checks do not cover arguments outside these ranges.

## Lattice sums

An Ewald sum splits a slowly converging lattice sum into a real-space part and a
reciprocal-space part, which both converge fast. The Ewald split parameter `eta`
sets how the work divides between the two parts; the exact sum does not depend on
it (see the [glossary](../reference/glossary.md)). The Rust function
[`lattice::sum`](../rust/treams_core/lattice/fn.sum.html)
states the same rules for the Rust code.

### Convergence and failures

- `eta = 0` selects the automatic split
  `eta = sqrt(2 pi) / (k L) max(|k L| / 8, 1)`, with the cell length
  `L = measure^(1/dim)`.
- Both Ewald parts add shells until two consecutive shells add less than 2e-13 of
  the part, never before the moduli of their far terms peak.
- The parts fail after 200, 32 or 16 shells in one, two or three dimensions. A
  large |k| L or a small |k eta| d ([below](#small-k-eta-d)) raises this limit up
  to four times.
- Exact diffraction thresholds are singular.

A sum that would be inaccurate fails instead. In Python each failure raises
`ValueError`.

| Message | Cause | Remedy |
| --- | --- | --- |
| "Ewald sums require Im(k) >= 0; gain media are unsupported" | A wavenumber with Im k < 0. | Use a passive wavenumber; finite direct shells remain available for complex k. |
| "spherical Ewald sums require Re(k) >= 0" | A spherical wavenumber on the unsupported Kambe branch. | Use a wavenumber in the supported domain; finite direct shells remain available for complex k. |
| "Ewald sum did not converge within the shell limit" | The parts need more shells than the limit: at small \|k eta\| d, at splits whose (k eta)^2 lies more than 45 degrees off the real axis, or where far terms peak beyond the limit. | Use the automatic split (`eta = 0`) where the limit comes from an explicit split. Below every automatic split the message adds "use a larger split (eta = 0 selects one)". |
| "Ewald split too small: ...; use a larger split (eta = 0 selects one)" | The parts of an explicit split below every automatic one cancel ([below](#explicit-splits-below-the-automatic-one)). | Use a larger split or `eta = 0`. |
| "Ewald sum lost its accuracy to cancelling Kambe integrals; reduce the split parameter" | A 1D spherical sum off the axis cancels where its spectral series is not available ([below](#1d-spherical-sums-off-the-axis)). | Reduce the split. |
| "non-finite Ewald summand" | (k eta)^2 turns 90 degrees or more off the real axis, so the Gaussians of both parts grow. | Change the split or reduce the order. |

Use the automatic split to avoid the explicit-split limits described below. It
does not remove the other limits.

### Branches and sheets

- **Supported Ewald wavenumbers** have Im k >= 0, with Re k >= 0 additionally
  required for spherical waves. Pure positive-imaginary k is supported, as is
  Re k < 0 for cylindrical waves. The full sum, both Ewald parts and their
  recorded derivatives reject inputs outside this domain before summation.
- **Re k < 0 for spherical waves** selects the wrong root of k^2 in the even
  Kambe integrals. Earlier implementations could return another function or
  fail to converge; the current implementation raises `ValueError`.
- **Im k < 0 (gain)** is unsupported for every Ewald family. The reciprocal
  roots continue from the evanescent orders rather than the outgoing gain
  branch; some families also depend on the split. These inputs raise
  `ValueError`. Within the supported domain every family gives the sum of the
  automatic split at every split where its Ewald parts converge.
- **Finite direct shells** accept every finite nonzero complex k. This does not
  assert convergence of the infinite direct sum for gain media.
- **At a lattice point** the sums exclude the image there. They take their self
  term on the sheet arg v = 2 arg k - pi - arg((k eta)^2) (DLMF 8.2.10), so they
  are the limit of the shifted sums at every split where their Ewald parts
  converge within the supported wavenumber domain.

### Near thresholds, planes and axes

- **Next to a diffraction threshold** the rounded q . q / k^2 - 1 loses about
  eps k^2 / |k_q^2|.
    - The zeroth order of an unreduced Bloch vector within 1/16 of its threshold
      takes compensated products: 2D sums stay within 2e-14 down to k_q = 0.002.
    - Other orders carry the rounding of their reciprocal vector or Bloch
      reduction and keep the rounded form of treams. 2D spherical sums of degree 0
      in the unit square at k_q of the order (0, 2 pi) of 4.5e-5 k are 2.1e-8 off
      (treams 6.4e-9).
- **Near the lattice plane or axis** the reciprocal orders of 2D spherical, 1D
  cylindrical and 1D spherical sums take the series of their reduced integrals in
  t = (k s eta)^2. The series applies for |t| <= 1; 1D spherical sums use it for
  |t| <= 1e-2, and for their orders below the Kambe base pair up to 1.
    - Values and derivatives are continuous down to the plane or axis, subnormal
      distances and signed zeros included.
    - Against mpmath the values are within 5e-14, 2e-14 and 2e-13, and the
      derivatives within 4e-13 (3e-12 for 1D spherical sums). The 2e-13 is at
      degree 7, 0.1 off the axis, with the split 0.35 (treams 3e-15).
    - Beyond |t| = 1, 2D spherical and 1D cylindrical sums run one Kambe
      recurrence chain per reciprocal point, whose correlated errors the sums
      cancel. They lose up to 2e-11 at |t| of 9 to 35 with the split 0.9 at
      k = 1.2 + 0.7i.

### 1D spherical sums off the axis

These sums need odd Kambe orders at w = k rho eta, for the distance rho from the
axis (w = 2.5 rho / period at the automatic split for k period < 8). There the
Ewald reciprocal part cancels by up to e^(w^2).

- Each component whose rounding bound exceeds 1e-12 comes from the spectral series
  (outgoing cylindrical waves over the diffraction orders) where that series
  bounds it more tightly. The series is tried first from w = 2.5.
- Complete sums are within 1.4e-13 (derivatives 5.7e-13) from w = 2 to 4, and
  within 1.3e-14 (7.4e-15) beyond.
- Next to the axis the Ewald sums remain. Explicit splits of 0.16 and 3.4 lose
  3e-9 and 4e-11 there.
- The sums fail with "lost its accuracy" beyond 1e-3 where the series is not
  available: for Re k <= 0, within about 0.015 periods of the axis (more than 512
  diffraction orders on each side), next to a diffraction threshold, or where a
  term overflows (degrees near 128 at small k rho).

### Sums that vanish by symmetry

Odd degrees or orders vanish at a Bloch vector that is zero modulo the reciprocal
lattice where twice the shift is a lattice vector. These sums are exactly zero at
every split, and so is their k derivative. The other derivatives keep the loss
check below. Twice the shift is compared exactly, so a point off by one unit in
the last place (ulp) is summed, and the sums tend to zero next to such points.

### Explicit splits below the automatic one

These are splits with Re(1 / (2 eta^2)) > 16 / pi, which for real splits means
below 0.31. The Ewald parts grow like e^Re(1 / (2 eta^2)) against the sum and
cancel.

- Both parts continue to 2e-13 of the complete sum with compensated additions.
- The real-space terms take the series of their Kambe integrals where the closed
  forms would lose more.
- Each sum predicts the rounding of the cancellation from its terms. It fails with
  "Ewald split too small" where that exceeds 1e-3 of the value or of a derivative
  of a jet (a value computed together with its derivatives). It fails early,
  before its far shells, once the real-space loss exceeds 2e-3 of the scale of the
  automatic split.
- The prediction exceeded the actual error of every probe sum more than 1e-11 off
  by a median factor of 45 at real and 87 at rotated splits (at least 4).
- No kept probe sum was more than 1.2e-4 off, or 7.7e-4 next to lattice points
  and diffraction thresholds (a 3D sum at k = 5.3 and eta = 0.133 that treams has
  1.0 off).
- A value-only call may return a sum whose jet fails.

The checks run only below every automatic split. Timings at these splits are in
the [performance evidence](../performance/evidence.md#lattice-sums-at-explicit-small-splits-pre-release-timings).

### Small |k eta| d

Here d is the smallest height of the reduced unit cell, 2 pi over its longest
reciprocal basis vector; d = L in square and cubic cells. Real-space terms fall
like exp(-(k eta R)^2 / 2), so the real-space limit grows until they fall below
exp(-50) of the nearest terms, up to four times its base. n shells reach the
radius n d.

- At small |k| L the Bloch phases cancel the terms within each shell. Below every
  automatic split (Re(1 / (2 eta^2)) > 16 / pi), a complete sum is returned where
  it matches the automatic split, if its real-space shells reach the limit but its
  last two shells add less than 2e-13 in all.
- Otherwise complete sums at small |k eta| d fail at explicit splits, below every
  automatic split or not, with "did not converge" (or "Ewald split too small" where
  their parts cancel, see above). So do their real-space parts alone (`realsumsw`,
  `realsumcw`).
- Apart from such settled sums, probe sums of degree up to 8 at |k| L of 0.05 to
  0.4 failed below a |k eta| d that depends on the sum: 0.04 to 0.07 in 2D and
  0.11 to 0.125 in 3D at degrees 0 and 1. Higher degrees and orders failed up to
  the same splits or converged at smaller ones, some at every probed split (down
  to 0.015 and 0.05).
- In the cube, sums that vanish by its point symmetry, which the rule for odd sums
  above does not cover, failed at larger splits. At a zero Bloch vector and shift,
  degrees 2 and 4 failed up to 0.15, and at |k| L = 0.05 degrees 6 and 8 failed at
  every probed split up to 0.4.
- Square, hexagonal, sheared and elongated cells gave the same limits in
  |k eta| d, so in elongated cells the limits in |k eta| L grow like L / d. A
  0.5 x 2 rectangle fails up to |k eta| L = 0.12, a 0.5 x 1 x 2 box up to 0.24.
- Above every automatic split, the sums that converge next to these limits and do
  not vanish by symmetry agree with the automatic split to about 1e-11 of
  max(|S|, 1) in 2D and 2e-10 in 3D. Below it they keep the loss check above.
- Sums and Ewald parts that fail to converge first sum up to the grown shell
  limit.

treams returns such sums after its fixed shell limit. Where the native sums fail
in the unit square and cube above every automatic split, treams is 5e-4 to 15 of
max(|S|, 1) off at degrees 0 and 1 and 3e-15 to 0.4 at higher degrees, within
1e-6 for most probe sums of degree 4 and higher (61 of 99). Below every automatic
split its parts cancel as well (up to 6e21 off). The automatic split is the
workaround.

### Splits turned off 1/k

These are splits where (k eta)^2 lies off the real axis.

- Below every automatic split, the real-space limit of 1D and 2D sums also grows
  at splits turned more than 45 degrees. 3D sums, and sums above every automatic
  split, keep the fixed limit there.
- Sums fail with "did not converge" where they need more shells than their limit:
    - 2D spherical sums of lossy k at real splits of 0.35 to 0.7 (treams within
      1.2e-10 and 6.8e-8);
    - a 1D cylindrical sum of order 11 at Re k < 0 whose far terms peak 100
      periods out;
    - a 1D spherical sum on the axis of degree 10 at k = 0.604 + 0.257i turned
      89.89 degrees, which needs more than its grown 800 shells (treams 5.7e-7
      off).
- Where (k eta)^2 turns 90 degrees or more, the Gaussians of both parts grow. Sums
  fail with "non-finite Ewald summand" or "did not converge" unless they vanish by
  symmetry.
- 2D jets that converge near 90 degrees take up to 1.5 s in a debug build.

### Kambe chains

At splits well below the automatic one, the downward Kambe recurrence carries the
rounding of its closed-form pair into lower orders.

- Beyond |t| = 1, each diffraction order of a 2D spherical or 1D cylindrical sum
  below every automatic split bounds its contribution by the chain and by single
  integrals, and takes the tighter bound. Other splits keep the chain.
- 1D cylindrical sums of order 8 at k = 11.3 with period 0.9, 0.99 off the axis,
  lose 3.2e-11 at 0.6 times the automatic split and 2.9e-9 at 0.5 times. Single
  integrals alone give 5.7e-13 at 0.6 times the automatic split but lose the
  cancellation of the chain elsewhere.

### Cases less accurate than treams

Errors are against extended-precision spectral, image or Lerch-transcendent sums,
relative to max(|S|, 1). For 2D sums, "off" is the distance from the lattice plane.

| Configuration | treams-rs error | treams error |
| --- | --- | --- |
| 2D spherical, hexagonal cell of side 1, k = 4.1 + 1i, 0.45 off | 2.4e-13 | 4e-15 |
| 2D spherical, split 0.9, k = 1.2 + 0.7i, 2.5 off, degree and order 12 | 2.5e-12 | 7.2e-13 |
| 2D spherical, rectangular cell, k = 9.7, eta = 0.11, 6.7 off, degree 12 order -5 | 1.5e-13; 6.2e-11 in other modes | 1.7e-14; up to 0.16 in other modes |
| 2D, degree 0, k = 8.34 + 1.66i, eta = 0.1354 + 0.0205i, 0.9 off | 2.2e-6 | 6.0e-7 |
| 2D sums that vanish by the half turn about the normal (odd orders, an in-plane shift at or half a lattice vector from a lattice point, a zero Bloch vector, off the plane), k = 2.15, eta = 0.1403 | 1.9e-12 to 5.7e-12 | 3e-15 to 1.2e-14 |
| 1D spherical, degree 0 on the axis, shift 0.5 along the axis, k = 0.34, eta = 0.137 | 8.3e-6 | 1.5e-7 |
| 1D spherical, degree 0 on the axis, eta = 0.2 | 7.7e-12 | 2.1e-12 |

1D spherical sums of degree 0 on the axis at small splits keep 7 to 15 ulps of
real-space parts up to 1e10. 32 of 808 probe sums are more than 3 times behind
treams, up to 54 times. Degrees 1 to 8 stay within 1.3e-12 where they trail.

## Floating-point environment

Rust results do not depend on the caller's flush-to-zero mode; see
[floating-point environment](../design/floating-point.md). This protection starts
at the native call. Argument conversion and the NumPy arithmetic of the Python
layer come first, so float32 subnormals, and float64 values that Python computes
from subnormal inputs, can still become zero when this mode is enabled.

## Scale and platform

- Dense outputs and LU storage grow quadratically with the channel dimension, and
  factorization work grows cubically.
- Solving for the requested illuminations avoids the full interacting T-matrix.
  The matrix-free sphere method avoids global quadratic storage in both the forward
  solve and physical-parameter adjoint. It recomputes pair translations in
  each GMRES iteration and checks the actual residual, so it can be slower than a
  reused dense factor. See [large problems](../performance/large-problems.md).
- Optimized homogeneous sphere clusters need non-overlapping nonmagnetic spheres
  in vacuum. The general local-T-matrix path supports other materials and cutoffs.
  The caller must ensure that enclosing particle surfaces do not overlap.
- The EBCM degree-6 benchmark contains analytically zero entries with severe
  cancellation. Both implementations reach a double-precision roundoff floor. The
  strict comparison check stays, and no degree-6 speed claim is made (see the
  [differences from treams](../coming-from-treams/differences.md) and its
  high-precision reproducer).
- HDF5 layout compatibility is not certification against every external T-matrix
  database.
- The release targets CPython 3.12–3.15 on the platforms listed under
  [Install](../getting-started/install.md). Historical numerical measurements
  cover the hosts and versions recorded with them; a wheel build does not
  extend those measurements to other platforms.
