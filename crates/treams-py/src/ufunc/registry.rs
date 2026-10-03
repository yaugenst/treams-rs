//! The ufunc tables: the names, loops, core signature and docstring of every ufunc,
//! and the constructors of the lattice-sum and coordinate-transform families.
//!
//! A ufunc has two names. The attribute of `_native` is unique; the ufuncs of
//! `sw`, `cw` and `pw` start with their namespace (`sw_translate_sh` serves
//! `sw.translate`). The ufunc name, `__name__`, appears in `NumPy` error
//! messages. For a ufunc that a namespace exposes as itself, it is the public
//! name (`periodic_to_pw` for `cw.periodic_to_pw`), and a row gives it where it
//! differs from the attribute. Ufuncs behind a Python wrapper keep the
//! attribute as their name.
//!
//! A docstring opens with the call under the ufunc name and the treams argument
//! names, such as `jv(v, z, /, out=None, *, where=True)`: `NumPy` shows its own
//! call with `x1, x2` first, and the API reference reads this line instead. The
//! formula, the treams function it mirrors and the differences from it follow.

use std::{
    ffi::{CStr, CString, c_char, c_long},
    num::Wrapping,
};

use numpy::npyffi::NPY_TYPES;
use pyo3::prelude::*;
use treams_core::Complex;

use super::ffi::{Dtype, Loop, create};
// Const generic arguments must be single identifiers, so the tables import the kinds.
use super::kinds::{
    angular::{LEGENDRE, PI_FUN, TAU_FUN},
    bessel::{H1, H2, J, Y},
    derivative::{DERIVATIVE, VALUE},
    family::{CYLINDRICAL, SPHERICAL},
    origin::{SHIFTED, UNSHIFTED},
    pol::{A, M, N},
    poltype::{HELICITY, PARITY},
    radial::{REGULAR, SINGULAR},
    turns::{CYCLIC, INVERSE},
    vsh::{HARMONIC_X, HARMONIC_Y, HARMONIC_Z},
};
use super::loops::{
    Tail, angular_loop, bessel_loop, coordinate_loop, cw_rotate_loop, cw_to_pw_loop, cw_to_sw_loop,
    cw_translate_loop, first_brillouin_1d_loop, gamma_loop, kambe_loop, lattice_core, lattice_loop,
    legendre_real_loop, plane_wave_loop, pw_permute_loop, pw_to_cw_loop, pw_to_sw_loop,
    pw_translate_loop, reciprocal_loop, refractive_indices_loop, refractive_indices_real_loop,
    sph_harm_loop, sw_rotate_loop, sw_to_cw_loop, sw_to_pw_loop, sw_translate_loop, tl_vcw_loop,
    tl_vsw_loop, vcw_a_loop, vcw_m_loop, vcw_n_loop, volume_loop, vsh_loop, vsw_a_loop, vsw_loop,
    wave_vector_z_loop, wigner_d_loop, wigner_small_d_loop, wigner3j_loop,
};

/// One row of a ufunc table.
pub(super) struct Ufunc {
    /// The attribute of `_native`.
    pub(super) attribute: &'static str,
    /// The ufunc name, `__name__`.
    pub(super) name: &'static str,
    pub(super) loops: &'static [Loop],
    /// The core signature of loops with core dimensions other than a vector
    /// output's.
    pub(super) core: Option<&'static str>,
    /// The docstring, which opens with the call under `name`.
    pub(super) doc: &'static CStr,
}

/// Declare ufuncs as `c"docstring" attribute: loop, ...;`, with an optional ufunc
/// name (`attribute as "name"`) and core signature (`attribute "(i,i)->()"`).
///
/// `NumPy` takes the first loop whose dtype row matches, so real rows precede
/// complex ones.
macro_rules! ufuncs {
    (@core) => { None };
    (@core $core:literal) => { Some($core) };
    (@name $attribute:ident) => { stringify!($attribute) };
    (@name $attribute:ident $name:literal) => { $name };
    ($($doc:literal $attribute:ident $(as $name:literal)? $($core:literal)?: $($function:expr),+;)+) => {
        [$(Ufunc {
            attribute: stringify!($attribute),
            name: ufuncs!(@name $attribute $($name)?),
            loops: &[$(Loop::new($function)),+],
            core: ufuncs!(@core $($core)?),
            doc: $doc,
        }),+]
    };
}

pub(super) const UFUNCS: &[Ufunc] = &ufuncs! {
    c"jv(v, z, /, out=None, *, where=True)\n\n\
      ``J_v(z)``, the Bessel function of the first kind of real order ``v``.\n\n\
      Mirrors ``treams.special.jv``.\n\n\
      Differences from treams: complex128 results also for real ``z``, with the principal-branch \
      value (``arg z = pi``) at ``z < 0``, where treams returns NaN for non-integer ``v``; \
      ``ValueError`` for non-finite input, at a pole and on overflow."
    jv: bessel_loop::<J, CYLINDRICAL, VALUE>;

    c"yv(v, z, /, out=None, *, where=True)\n\n\
      ``Y_v(z)``, the Bessel function of the second kind of real order ``v``.\n\n\
      Mirrors ``treams.special.yv``.\n\n\
      Differences from treams: complex128 results also for real ``z``, with the principal-branch \
      value (``arg z = pi``) at ``z < 0``, where treams returns NaN; ``ValueError`` for \
      non-finite input, at ``z = 0`` and on overflow."
    yv: bessel_loop::<Y, CYLINDRICAL, VALUE>;

    c"hankel1(v, z, /, out=None, *, where=True)\n\n\
      ``H1_v(z) = J_v(z) + i Y_v(z)``, the Hankel function of the first kind.\n\n\
      Mirrors ``treams.special.hankel1``.\n\n\
      Differences from treams: ``ValueError`` for non-finite input, at ``z = 0`` and on \
      overflow, where treams returns NaN or infinity."
    hankel1: bessel_loop::<H1, CYLINDRICAL, VALUE>;

    c"hankel2(v, z, /, out=None, *, where=True)\n\n\
      ``H2_v(z) = J_v(z) - i Y_v(z)``, the Hankel function of the second kind.\n\n\
      Mirrors ``treams.special.hankel2``.\n\n\
      Differences from treams: ``ValueError`` for non-finite input, at ``z = 0`` and on \
      overflow, where treams returns NaN or infinity."
    hankel2: bessel_loop::<H2, CYLINDRICAL, VALUE>;

    c"jv_d(v, z, /, out=None, *, where=True)\n\n\
      ``J_v'(z) = (J_(v-1)(z) - J_(v+1)(z)) / 2``, the derivative of ``J_v``.\n\n\
      Mirrors ``treams.special.jv_d``.\n\n\
      Differences from treams: complex128 results also for real ``z``, with the principal-branch \
      value (``arg z = pi``) at ``z < 0``, where treams returns NaN for non-integer ``v``; \
      ``ValueError`` for non-finite input, at a pole and on overflow."
    jv_d: bessel_loop::<J, CYLINDRICAL, DERIVATIVE>;

    c"yv_d(v, z, /, out=None, *, where=True)\n\n\
      ``Y_v'(z) = (Y_(v-1)(z) - Y_(v+1)(z)) / 2``, the derivative of ``Y_v``.\n\n\
      Mirrors ``treams.special.yv_d``.\n\n\
      Differences from treams: complex128 results also for real ``z``, with the principal-branch \
      value (``arg z = pi``) at ``z < 0``, where treams returns NaN; ``ValueError`` for \
      non-finite input, at ``z = 0`` and on overflow."
    yv_d: bessel_loop::<Y, CYLINDRICAL, DERIVATIVE>;

    c"hankel1_d(v, z, /, out=None, *, where=True)\n\n\
      ``H1_v'(z) = (H1_(v-1)(z) - H1_(v+1)(z)) / 2``, the derivative of ``H1_v``.\n\n\
      Mirrors ``treams.special.hankel1_d``.\n\n\
      Differences from treams: ``ValueError`` for non-finite input, at ``z = 0`` and on \
      overflow, where treams returns NaN or infinity."
    hankel1_d: bessel_loop::<H1, CYLINDRICAL, DERIVATIVE>;

    c"hankel2_d(v, z, /, out=None, *, where=True)\n\n\
      ``H2_v'(z) = (H2_(v-1)(z) - H2_(v+1)(z)) / 2``, the derivative of ``H2_v``.\n\n\
      Mirrors ``treams.special.hankel2_d``.\n\n\
      Differences from treams: ``ValueError`` for non-finite input, at ``z = 0`` and on \
      overflow, where treams returns NaN or infinity."
    hankel2_d: bessel_loop::<H2, CYLINDRICAL, DERIVATIVE>;

    c"spherical_jn(n, z, /, out=None, *, where=True)\n\n\
      ``j_n(z) = sqrt(pi / (2 z)) J_(n+1/2)(z)``, the spherical Bessel function of the first \
      kind.\n\n\
      Mirrors ``treams.special.spherical_jn`` with ``derivative=False``.\n\n\
      Differences from treams: complex128 results also for real ``z``; ``ValueError`` for \
      non-finite input and on overflow."
    spherical_jn: bessel_loop::<J, SPHERICAL, VALUE>;

    c"spherical_yn(n, z, /, out=None, *, where=True)\n\n\
      ``y_n(z) = sqrt(pi / (2 z)) Y_(n+1/2)(z)``, the spherical Bessel function of the second \
      kind.\n\n\
      Mirrors ``treams.special.spherical_yn`` with ``derivative=False``.\n\n\
      Differences from treams: complex128 results also for real ``z``; ``ValueError`` for \
      non-finite input, at ``z = 0``, on overflow and for ``n = 0`` at \
      ``|z|`` below about ``1e-162``."
    spherical_yn: bessel_loop::<Y, SPHERICAL, VALUE>;

    c"spherical_hankel1(n, z, /, out=None, *, where=True)\n\n\
      ``h1_n(z) = j_n(z) + i y_n(z)``, the spherical Hankel function of the first kind.\n\n\
      Mirrors ``treams.special.spherical_hankel1``.\n\n\
      Differences from treams: ``ValueError`` at ``z = 0``, where treams raises \
      ``ZeroDivisionError``; also for non-finite input, on overflow and for ``n = 0`` at \
      ``|z|`` below about ``1e-162``."
    spherical_hankel1: bessel_loop::<H1, SPHERICAL, VALUE>;

    c"spherical_hankel2(n, z, /, out=None, *, where=True)\n\n\
      ``h2_n(z) = j_n(z) - i y_n(z)``, the spherical Hankel function of the second kind.\n\n\
      Mirrors ``treams.special.spherical_hankel2``.\n\n\
      Differences from treams: ``ValueError`` at ``z = 0``, where treams raises \
      ``ZeroDivisionError``; also for non-finite input, on overflow and for ``n = 0`` at \
      ``|z|`` below about ``1e-162``."
    spherical_hankel2: bessel_loop::<H2, SPHERICAL, VALUE>;

    c"spherical_jn_d(n, z, /, out=None, *, where=True)\n\n\
      ``j_n'(z) = (n j_(n-1)(z) - (n + 1) j_(n+1)(z)) / (2 n + 1)``, the derivative of \
      ``j_n``.\n\n\
      Mirrors ``treams.special.spherical_jn_d``.\n\n\
      Differences from treams: complex128 results also for real ``z``; ``ValueError`` for \
      non-finite input and on overflow."
    spherical_jn_d: bessel_loop::<J, SPHERICAL, DERIVATIVE>;

    c"spherical_yn_d(n, z, /, out=None, *, where=True)\n\n\
      ``y_n'(z) = (n y_(n-1)(z) - (n + 1) y_(n+1)(z)) / (2 n + 1)``, the derivative of \
      ``y_n``.\n\n\
      Mirrors ``treams.special.spherical_yn_d``.\n\n\
      Differences from treams: complex128 results also for real ``z``; ``ValueError`` for \
      non-finite input, at ``z = 0`` and on overflow."
    spherical_yn_d: bessel_loop::<Y, SPHERICAL, DERIVATIVE>;

    c"spherical_hankel1_d(n, z, /, out=None, *, where=True)\n\n\
      ``h1_n'(z) = (n h1_(n-1)(z) - (n + 1) h1_(n+1)(z)) / (2 n + 1)``, the derivative of \
      ``h1_n``.\n\n\
      Mirrors ``treams.special.spherical_hankel1_d``.\n\n\
      Differences from treams: ``ValueError`` at ``z = 0``, where treams raises \
      ``ZeroDivisionError``; also for non-finite input and on overflow."
    spherical_hankel1_d: bessel_loop::<H1, SPHERICAL, DERIVATIVE>;

    c"spherical_hankel2_d(n, z, /, out=None, *, where=True)\n\n\
      ``h2_n'(z) = (n h2_(n-1)(z) - (n + 1) h2_(n+1)(z)) / (2 n + 1)``, the derivative of \
      ``h2_n``.\n\n\
      Mirrors ``treams.special.spherical_hankel2_d``.\n\n\
      Differences from treams: ``ValueError`` at ``z = 0``, where treams raises \
      ``ZeroDivisionError``; also for non-finite input and on overflow."
    spherical_hankel2_d: bessel_loop::<H2, SPHERICAL, DERIVATIVE>;

    c"lpmv(m, v, z, /, out=None, *, where=True)\n\n\
      ``P_v^m(z)``, the associated Legendre function of order ``m`` and degree ``v``, with \
      the branch cut at real ``|z| > 1``.\n\n\
      Mirrors ``treams.special.lpmv``.\n\n\
      Differences from treams: ``ValueError`` for degrees outside ``[0, 128]``, non-integer \
      orders and non-finite ``z``. At real ``|z| > 1`` and an integer degree ``v >= |m| > 0``, \
      where treams returns NaN, this function returns the value for even ``m`` and raises \
      ``ValueError`` for odd ``m``. A non-integer degree ``v > |m|`` needs ``z`` in \
      ``(-1, 1]`` with zero imaginary part and raises ``ValueError`` elsewhere. With such a \
      degree, treams returns NaN or infinity at complex ``z`` and outside ``(-1, 1]``, except \
      a value at real ``z > 1`` for ``m = 0``."
    lpmv: legendre_real_loop, angular_loop::<LEGENDRE>;

    c"pi_fun(l, m, x, /, out=None, *, where=True)\n\n\
      ``pi_lm(x) = m P_l^m(x) / sqrt(1 - x**2)``.\n\n\
      Mirrors ``treams.special.pi_fun``.\n\n\
      Differences from treams: complex128 results also for real ``x``, with values at real \
      ``|x| > 1``, where treams returns NaN; ``ValueError`` for degrees outside ``[0, 128]`` \
      and non-finite ``x``."
    pi_fun: angular_loop::<PI_FUN>;

    c"tau_fun(l, m, x, /, out=None, *, where=True)\n\n\
      ``tau_lm(x) = d P_l^m(cos(theta)) / d theta`` at ``x = cos(theta)``.\n\n\
      Mirrors ``treams.special.tau_fun``.\n\n\
      Differences from treams: complex128 results also for real ``x``, with values at real \
      ``|x| > 1``, where treams returns NaN unless ``l = |m| <= 1``; 0 for ``|m| > l``, where \
      treams returns, for example, ``tau_fun(0, -1, x) = 0.5``; ``ValueError`` for degrees \
      outside ``[0, 128]`` and non-finite ``x``."
    tau_fun: angular_loop::<TAU_FUN>;

    c"wignersmalld(l, m, k, theta, /, out=None, *, where=True)\n\n\
      ``d^l_mk(theta)``, the Wigner small-d function.\n\n\
      Mirrors ``treams.special.wignersmalld``.\n\n\
      Differences from treams: complex128 results also for real ``theta``; degrees up to 128; \
      ``ValueError`` for non-finite ``theta``."
    wignersmalld: wigner_small_d_loop;

    c"wignerd(l, m, k, phi, theta, psi, /, out=None, *, where=True)\n\n\
      ``D^l_mk(phi, theta, psi) = exp(-i m phi) d^l_mk(theta) exp(-i k psi)``, the Wigner D \
      function.\n\n\
      Mirrors ``treams.special.wignerd``.\n\n\
      Differences from treams: complex ``phi`` and ``psi`` keep their imaginary parts, which \
      treams drops; degrees up to 128; ``ValueError`` for non-finite angles."
    wignerd: wigner_d_loop;

    c"wigner3j(j1, j2, j3, m1, m2, m3, /, out=None, *, where=True)\n\n\
      The Wigner 3j symbol ``(j1 j2 j3; m1 m2 m3)``, and 0 where the selection rules forbid \
      it.\n\n\
      Mirrors ``treams.special.wigner3j``.\n\n\
      Differences from treams: labels within ``[-260, 260]``; each recurrence continues while \
      the symbols grow, which keeps extreme orders accurate where treams loses digits."
    wigner3j: wigner3j_loop;

    c"incgamma(l, z, /, out=None, *, where=True)\n\n\
      ``Gamma(l, z)``, the upper incomplete gamma function of integer or half-integer degree \
      ``l``, with the branch cut on the negative real axis.\n\n\
      Mirrors ``treams.special.incgamma``.\n\n\
      Differences from treams: complex128 results also for real ``z``, with the principal-branch \
      value (``arg z = pi``) at ``z < 0``, where treams returns NaN for half-integer ``l`` and \
      for ``l <= 0``; ``|l|`` up to 128; ``ValueError`` for non-finite ``z``."
    incgamma: gamma_loop;

    c"intkambe(n, z, eta, /, out=None, *, where=True)\n\n\
      ``I_n(z, eta)``, the integral of ``t**n exp(-z**2 t**2 / 2 + 1 / (2 t**2))`` over ``t`` \
      from ``eta`` to infinity: the Kambe integral of the Ewald lattice sums.\n\n\
      Mirrors ``treams.special.intkambe``.\n\n\
      Differences from treams: complex128 results also for real ``z`` and ``eta``; ``|n|`` up \
      to 260; ``ValueError`` for non-finite input."
    intkambe: kambe_loop;

    c"refractive_indices(epsilon, mu, kappa, /, out=None)\n\n\
      ``(sqrt(epsilon mu) - kappa, sqrt(epsilon mu) + kappa)``, the refractive indices of \
      negative and positive helicity, with nonnegative imaginary parts.\n\n\
      Mirrors ``treams.misc.refractive_index``."
    refractive_indices: refractive_indices_real_loop, refractive_indices_loop;

    c"wave_vector_z(kx, ky, k, /, out=None, *, where=True)\n\n\
      ``kz = sqrt(k**2 - kx**2 - ky**2)`` with ``Im(kz) >= 0``, and ``Re(kz) >= 0`` where \
      ``kz`` is real.\n\n\
      Mirrors ``treams.misc.wave_vec_z``."
    wave_vector_z: wave_vector_z_loop::<f64>, wave_vector_z_loop::<Complex>;

    c"firstbrillouin1d(kpar, b, /, out=None, *, where=True)\n\n\
      ``kpar`` reduced to the first Brillouin zone ``(-b/2, b/2]`` of a lattice with \
      reciprocal pitch ``b``.\n\n\
      Mirrors ``treams.misc.firstbrillouin1d``.\n\n\
      Differences from treams: broadcasts over arrays, where treams takes scalars; \
      ``ValueError`` unless ``b > 0`` and both arguments are finite, where treams raises \
      ``ZeroDivisionError`` for ``b = 0``, returns NaN for non-finite input and an unreduced \
      value for ``b < 0``."
    first_brillouin_1d as "firstbrillouin1d": first_brillouin_1d_loop;

    c"sph_harm(m, l, phi, theta, /, out=None, *, where=True)\n\n\
      ``Y_lm(theta, phi) = N_lm P_l^m(cos(theta)) exp(i m phi)`` with \
      ``N_lm = sqrt((2 l + 1) (l - m)! / (4 pi (l + m)!))``; the order of the arguments \
      follows SciPy.\n\n\
      Mirrors ``treams.special.sph_harm``.\n\n\
      Differences from treams: 0 for ``|m| > l``, where treams returns NaN at real ``theta``; \
      degrees up to 128; ``ValueError`` for non-finite angles."
    sph_harm: sph_harm_loop::<f64>, sph_harm_loop::<Complex>;

    c"vsh_X(l, m, theta, phi, /, out=None)\n\n\
      ``X_lm(theta, phi)``, the vector spherical harmonic tangent to the sphere of the ``M`` \
      waves, in spherical components ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsh_X``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    vsh_X: vsh_loop::<f64, HARMONIC_X>, vsh_loop::<Complex, HARMONIC_X>;

    c"vsh_Y(l, m, theta, phi, /, out=None)\n\n\
      ``Y_lm(theta, phi) = r_hat x X_lm(theta, phi)``, in spherical components \
      ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsh_Y``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    vsh_Y: vsh_loop::<f64, HARMONIC_Y>, vsh_loop::<Complex, HARMONIC_Y>;

    c"vsh_Z(l, m, theta, phi, /, out=None)\n\n\
      ``Z_lm(theta, phi) = i Y_lm(theta, phi) r_hat``, in spherical components \
      ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsh_Z``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    vsh_Z: vsh_loop::<f64, HARMONIC_Z>, vsh_loop::<Complex, HARMONIC_Z>;

    c"vsw_M(l, m, x, theta, phi, /, out=None)\n\n\
      ``M_lm(x, theta, phi) = h_l(x) X_lm(theta, phi)``, the singular vector spherical wave \
      ``M`` at ``x = k r``, in spherical components ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsw_M``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` at ``x = 0``, where treams \
      raises ``ZeroDivisionError``, and for non-finite input."
    vsw_M: vsw_loop::<f64, SINGULAR, M>, vsw_loop::<Complex, SINGULAR, M>;

    c"vsw_N(l, m, x, theta, phi, /, out=None)\n\n\
      ``N_lm(x, theta, phi)``, the curl of ``M_lm`` with respect to ``x = k r``: the singular \
      vector spherical wave ``N``, in spherical components ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsw_N``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` at ``x = 0``, where treams \
      raises ``ZeroDivisionError``, and for non-finite input."
    vsw_N: vsw_loop::<f64, SINGULAR, N>, vsw_loop::<Complex, SINGULAR, N>;

    c"vsw_A(l, m, x, theta, phi, p, /, out=None)\n\n\
      ``A_lm(x, theta, phi) = (N_lm +- M_lm) / sqrt(2)``, the singular vector spherical wave \
      of helicity ``p`` (1 positive, 0 negative), in spherical components \
      ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsw_A``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` at ``x = 0``, where treams \
      raises ``ZeroDivisionError``, and for non-finite input."
    vsw_A: vsw_a_loop::<f64, SINGULAR>, vsw_a_loop::<Complex, SINGULAR>;

    c"vsw_rM(l, m, x, theta, phi, /, out=None)\n\n\
      ``M_lm(x, theta, phi) = j_l(x) X_lm(theta, phi)``, the regular vector spherical wave \
      ``M`` at ``x = k r``, in spherical components ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsw_rM``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    vsw_rM: vsw_loop::<f64, REGULAR, M>, vsw_loop::<Complex, REGULAR, M>;

    c"vsw_rN(l, m, x, theta, phi, /, out=None)\n\n\
      ``N_lm(x, theta, phi)``, the curl of ``M_lm`` with respect to ``x = k r``: the regular \
      vector spherical wave ``N``, in spherical components ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsw_rN``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    vsw_rN: vsw_loop::<f64, REGULAR, N>, vsw_loop::<Complex, REGULAR, N>;

    c"vsw_rA(l, m, x, theta, phi, p, /, out=None)\n\n\
      ``A_lm(x, theta, phi) = (N_lm +- M_lm) / sqrt(2)``, the regular vector spherical wave \
      of helicity ``p`` (1 positive, 0 negative), in spherical components \
      ``(r, theta, phi)``.\n\n\
      Mirrors ``treams.special.vsw_rA``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    vsw_rA: vsw_a_loop::<f64, REGULAR>, vsw_a_loop::<Complex, REGULAR>;

    c"vcw_M(kz, m, xrho, phi, z, /, out=None)\n\n\
      ``M_kzm(xrho, phi, z)``, the singular vector cylindrical wave ``M`` built on \
      ``H_m(xrho) exp(i (kz z + m phi))`` with ``xrho = k_rho rho``, in cylindrical \
      components ``(rho, phi, z)``.\n\n\
      Mirrors ``treams.special.vcw_M``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` at ``xrho = 0``, \
      where treams raises ``ZeroDivisionError``, and for non-finite input."
    vcw_M: vcw_m_loop::<SINGULAR>;

    c"vcw_N(kz, m, xrho, phi, z, k, /, out=None)\n\n\
      ``N_kzm(xrho, phi, z)``, the curl of ``M_kzm`` divided by ``k``: the singular vector \
      cylindrical wave ``N``, in cylindrical components ``(rho, phi, z)``.\n\n\
      Mirrors ``treams.special.vcw_N``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` at ``xrho = 0`` and \
      ``k = 0``, where treams raises ``ZeroDivisionError``, and for non-finite input."
    vcw_N: vcw_n_loop::<SINGULAR>;

    c"vcw_A(kz, m, xrho, phi, z, k, pol, /, out=None)\n\n\
      ``A_kzm = (N_kzm +- M_kzm) / sqrt(2)``, the singular vector cylindrical wave of \
      helicity ``pol`` (1 positive, 0 negative), in cylindrical components ``(rho, phi, z)``.\n\n\
      Mirrors ``treams.special.vcw_A``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` at ``xrho = 0`` and \
      ``k = 0``, where treams raises ``ZeroDivisionError``, and for non-finite input."
    vcw_A: vcw_a_loop::<SINGULAR>;

    c"vcw_rM(kz, m, xrho, phi, z, /, out=None)\n\n\
      ``M_kzm(xrho, phi, z)``, the regular vector cylindrical wave ``M`` built on \
      ``J_m(xrho) exp(i (kz z + m phi))`` with ``xrho = k_rho rho``, in cylindrical \
      components ``(rho, phi, z)``.\n\n\
      Mirrors ``treams.special.vcw_rM``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` for non-finite input."
    vcw_rM: vcw_m_loop::<REGULAR>;

    c"vcw_rN(kz, m, xrho, phi, z, k, /, out=None)\n\n\
      ``N_kzm(xrho, phi, z)``, the curl of ``M_kzm`` divided by ``k``: the regular vector \
      cylindrical wave ``N``, in cylindrical components ``(rho, phi, z)``.\n\n\
      Mirrors ``treams.special.vcw_rN``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` at ``k = 0``, where \
      treams raises ``ZeroDivisionError``, and for non-finite input."
    vcw_rN: vcw_n_loop::<REGULAR>;

    c"vcw_rA(kz, m, xrho, phi, z, k, pol, /, out=None)\n\n\
      ``A_kzm = (N_kzm +- M_kzm) / sqrt(2)``, the regular vector cylindrical wave of helicity \
      ``pol`` (1 positive, 0 negative), in cylindrical components ``(rho, phi, z)``.\n\n\
      Mirrors ``treams.special.vcw_rA``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` at ``k = 0``, where \
      treams raises ``ZeroDivisionError``, and for non-finite input."
    vcw_rA: vcw_a_loop::<REGULAR>;

    c"tl_vcw(kz1, mu, kz2, m, xrho, phi, z, /, out=None, *, where=True)\n\n\
      ``H_(m-mu)(xrho) exp(i ((m - mu) phi + kz1 z))`` for ``kz1 == kz2``, else 0: the \
      coefficient from singular to regular vector cylindrical waves.\n\n\
      Mirrors ``treams.special.tl_vcw``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` at ``xrho = 0``, where \
      treams returns NaN, and for non-finite input."
    tl_vcw: tl_vcw_loop::<Complex, SINGULAR>;

    c"tl_vcw_r(kz1, mu, kz2, m, xrho, phi, z, /, out=None, *, where=True)\n\n\
      ``J_(m-mu)(xrho) exp(i ((m - mu) phi + kz1 z))`` for ``kz1 == kz2``, else 0: the \
      coefficient between vector cylindrical waves of one kind.\n\n\
      Mirrors ``treams.special.tl_vcw_r``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` for non-finite input."
    tl_vcw_r: tl_vcw_loop::<f64, REGULAR>, tl_vcw_loop::<Complex, REGULAR>;

    c"sw_rotate(lambda_, mu, pol, l, m, qol, phi, theta, psi, /, out=None, *, where=True)\n\n\
      ``D^l_(mu m)(phi, theta, psi)`` for ``lambda_ == l`` and ``pol == qol``, else 0: the \
      coefficient that rotates the spherical mode ``(l, m, qol)`` into ``(lambda_, mu, pol)`` \
      by the z-y-z Euler angles.\n\n\
      Mirrors ``treams.sw.rotate``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    sw_rotate: sw_rotate_loop;

    c"cw_rotate(kz, mu, pol, qz, m, qol, phi, /, out=None, *, where=True)\n\n\
      ``exp(-i m phi)`` for equal labels ``(kz, mu, pol) == (qz, m, qol)``, else 0: the \
      coefficient that rotates a cylindrical mode about the z axis.\n\n\
      Mirrors ``treams.cw.rotate``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` for non-finite input."
    cw_rotate: cw_rotate_loop;

    c"sw_periodic_to_pw_h(kx, ky, kz, pol, l, m, qol, area, /, out=None, *, where=True)\n\n\
      The amplitude of the plane wave ``(kx, ky, kz, pol)`` in the field of a lattice in the \
      xy plane of singular spherical waves ``(l, m, qol)``, with unit cell area ``area``, in \
      the helicity basis.\n\n\
      Mirrors ``treams.sw.periodic_to_pw`` with ``poltype='helicity'``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for ``area = 0`` and a zero \
      wave vector, where treams raises ``ZeroDivisionError``, and for non-finite input."
    sw_periodic_to_pw_h: sw_to_pw_loop::<f64, HELICITY>, sw_to_pw_loop::<Complex, HELICITY>;

    c"sw_periodic_to_pw_p(kx, ky, kz, pol, l, m, qol, area, /, out=None, *, where=True)\n\n\
      The amplitude of the plane wave ``(kx, ky, kz, pol)`` in the field of a lattice in the \
      xy plane of singular spherical waves ``(l, m, qol)``, with unit cell area ``area``, in \
      the parity basis.\n\n\
      Mirrors ``treams.sw.periodic_to_pw`` with ``poltype='parity'``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for ``area = 0`` and a zero \
      wave vector, where treams raises ``ZeroDivisionError``, and for non-finite input."
    sw_periodic_to_pw_p: sw_to_pw_loop::<f64, PARITY>, sw_to_pw_loop::<Complex, PARITY>;

    c"periodic_to_pw(kx, ky, kz, pol, qz, m, qol, area, /, out=None, *, where=True)\n\n\
      The amplitude of the plane wave ``(kx, ky, kz, pol)`` in the field of a lattice on the \
      x axis of singular cylindrical waves ``(qz, m, qol)`` with period ``area``.\n\n\
      Mirrors ``treams.cw.periodic_to_pw``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` at a diffraction \
      threshold (``ky = 0``), where treams returns a large finite value; ``ValueError`` for \
      ``area = 0``, where treams raises ``ZeroDivisionError``, and for non-finite input."
    cw_periodic_to_pw as "periodic_to_pw": cw_to_pw_loop::<f64>, cw_to_pw_loop::<Complex>;

    c"sw_periodic_to_cw_h(kz, m, pol, l, mu, qol, k, area, /, out=None, *, where=True)\n\n\
      The amplitude of the singular cylindrical wave ``(kz, m, pol)`` in the field of a \
      lattice on the z axis of singular spherical waves ``(l, mu, qol)`` with wavenumber \
      ``k`` and period ``area``, in the helicity basis.\n\n\
      Mirrors ``treams.sw.periodic_to_cw`` with ``poltype='helicity'``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for ``k = 0`` and \
      ``area = 0``, where treams raises ``ZeroDivisionError``, and for non-finite input."
    sw_periodic_to_cw_h: sw_to_cw_loop::<HELICITY>;

    c"sw_periodic_to_cw_p(kz, m, pol, l, mu, qol, k, area, /, out=None, *, where=True)\n\n\
      The amplitude of the singular cylindrical wave ``(kz, m, pol)`` in the field of a \
      lattice on the z axis of singular spherical waves ``(l, mu, qol)`` with wavenumber \
      ``k`` and period ``area``, in the parity basis.\n\n\
      Mirrors ``treams.sw.periodic_to_cw`` with ``poltype='parity'``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for ``k = 0`` and \
      ``area = 0``, where treams raises ``ZeroDivisionError``, and for non-finite input."
    sw_periodic_to_cw_p: sw_to_cw_loop::<PARITY>;

    c"sw_translate_sh(lambda_, mu, pol, l, m, qol, kr, theta, phi, /, out=None, *, \
      where=True)\n\n\
      The coefficient that translates the singular spherical wave ``(l, m, qol)`` into the \
      regular wave ``(lambda_, mu, pol)`` across the displacement ``(kr, theta, phi)``, in \
      the helicity basis.\n\n\
      Mirrors ``treams.sw.translate`` with ``poltype='helicity'`` and ``singular=True``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    sw_translate_sh: sw_translate_loop::<f64, HELICITY, SINGULAR>, sw_translate_loop::<Complex, HELICITY, SINGULAR>;

    c"sw_translate_rh(lambda_, mu, pol, l, m, qol, kr, theta, phi, /, out=None, *, \
      where=True)\n\n\
      The coefficient that translates the spherical wave ``(l, m, qol)`` into the wave \
      ``(lambda_, mu, pol)`` of the same kind across the displacement ``(kr, theta, phi)``, \
      in the helicity basis.\n\n\
      Mirrors ``treams.sw.translate`` with ``poltype='helicity'`` and ``singular=False``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    sw_translate_rh: sw_translate_loop::<f64, HELICITY, REGULAR>, sw_translate_loop::<Complex, HELICITY, REGULAR>;

    c"sw_translate_sp(lambda_, mu, pol, l, m, qol, kr, theta, phi, /, out=None, *, \
      where=True)\n\n\
      The coefficient that translates the singular spherical wave ``(l, m, qol)`` into the \
      regular wave ``(lambda_, mu, pol)`` across the displacement ``(kr, theta, phi)``, in \
      the parity basis.\n\n\
      Mirrors ``treams.sw.translate`` with ``poltype='parity'`` and ``singular=True``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    sw_translate_sp: sw_translate_loop::<f64, PARITY, SINGULAR>, sw_translate_loop::<Complex, PARITY, SINGULAR>;

    c"sw_translate_rp(lambda_, mu, pol, l, m, qol, kr, theta, phi, /, out=None, *, \
      where=True)\n\n\
      The coefficient that translates the spherical wave ``(l, m, qol)`` into the wave \
      ``(lambda_, mu, pol)`` of the same kind across the displacement ``(kr, theta, phi)``, \
      in the parity basis.\n\n\
      Mirrors ``treams.sw.translate`` with ``poltype='parity'`` and ``singular=False``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    sw_translate_rp: sw_translate_loop::<f64, PARITY, REGULAR>, sw_translate_loop::<Complex, PARITY, REGULAR>;

    c"cw_translate_s(kz, mu, pol, qz, m, qol, krr, phi, z, /, out=None, *, where=True)\n\n\
      The coefficient that translates the singular cylindrical wave ``(qz, m, qol)`` into the \
      regular wave ``(kz, mu, pol)`` across the displacement ``(krr, phi, z)``: ``tl_vcw`` \
      for ``pol == qol``, else 0.\n\n\
      Mirrors ``treams.cw.translate`` with ``singular=True``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` at ``krr = 0``, where \
      treams returns NaN, and for non-finite input."
    cw_translate_s: cw_translate_loop::<f64, SINGULAR>, cw_translate_loop::<Complex, SINGULAR>;

    c"cw_translate_r(kz, mu, pol, qz, m, qol, krr, phi, z, /, out=None, *, where=True)\n\n\
      The coefficient that translates the cylindrical wave ``(qz, m, qol)`` into the wave \
      ``(kz, mu, pol)`` of the same kind across the displacement ``(krr, phi, z)``: \
      ``tl_vcw_r`` for ``pol == qol``, else 0.\n\n\
      Mirrors ``treams.cw.translate`` with ``singular=False``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` for non-finite input."
    cw_translate_r: cw_translate_loop::<f64, REGULAR>, cw_translate_loop::<Complex, REGULAR>;

    c"pw_to_sw_h(l, m, polsw, kx, ky, kz, polpw, /, out=None, *, where=True)\n\n\
      The amplitude of the regular spherical wave ``(l, m, polsw)`` in the plane wave \
      ``(kx, ky, kz, polpw)``, in the helicity basis.\n\n\
      Mirrors ``treams.pw.to_sw`` with ``poltype='helicity'``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for a wave vector with \
      ``kx**2 + ky**2 + kz**2 = 0``, where treams raises ``ZeroDivisionError``, and for \
      non-finite input."
    pw_to_sw_h: pw_to_sw_loop::<f64, HELICITY>, pw_to_sw_loop::<Complex, HELICITY>;

    c"pw_to_sw_p(l, m, polsw, kx, ky, kz, polpw, /, out=None, *, where=True)\n\n\
      The amplitude of the regular spherical wave ``(l, m, polsw)`` in the plane wave \
      ``(kx, ky, kz, polpw)``, in the parity basis.\n\n\
      Mirrors ``treams.pw.to_sw`` with ``poltype='parity'``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for a wave vector with \
      ``kx**2 + ky**2 + kz**2 = 0``, where treams raises ``ZeroDivisionError``, and for \
      non-finite input."
    pw_to_sw_p: pw_to_sw_loop::<f64, PARITY>, pw_to_sw_loop::<Complex, PARITY>;

    c"to_cw(kzcw, m, polcw, kx, ky, kzpw, polpw, /, out=None, *, where=True)\n\n\
      The amplitude of the regular cylindrical wave ``(kzcw, m, polcw)`` in the plane wave \
      ``(kx, ky, kzpw, polpw)``, and 0 unless ``kzcw == kzpw``.\n\n\
      Mirrors ``treams.pw.to_cw``.\n\n\
      Differences from treams: orders ``|m|`` up to 128; ``ValueError`` for non-finite input."
    pw_to_cw as "to_cw": pw_to_cw_loop::<f64>, pw_to_cw_loop::<Complex>;

    c"cw_to_sw_h(l, m, polsw, kz, mu, polcw, k, /, out=None, *, where=True)\n\n\
      The amplitude of the regular spherical wave ``(l, m, polsw)`` in the regular \
      cylindrical wave ``(kz, mu, polcw)`` of wavenumber ``k``, in the helicity basis.\n\n\
      Mirrors ``treams.cw.to_sw`` with ``poltype='helicity'``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for ``k = 0``, where treams \
      raises ``ZeroDivisionError``, and for non-finite input."
    cw_to_sw_h: cw_to_sw_loop::<HELICITY>;

    c"cw_to_sw_p(l, m, polsw, kz, mu, polcw, k, /, out=None, *, where=True)\n\n\
      The amplitude of the regular spherical wave ``(l, m, polsw)`` in the regular \
      cylindrical wave ``(kz, mu, polcw)`` of wavenumber ``k``, in the parity basis.\n\n\
      Mirrors ``treams.cw.to_sw`` with ``poltype='parity'``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for ``k = 0``, where treams \
      raises ``ZeroDivisionError``, and for non-finite input."
    cw_to_sw_p: cw_to_sw_loop::<PARITY>;

    c"pw_permute_xyz_h(kx, ky, kz, p, q, /, out=None, *, where=True)\n\n\
      The amplitude of polarization ``p`` after one cyclic permutation of the Cartesian axes, \
      for the plane wave ``(kx, ky, kz)`` of polarization ``q``, in the helicity basis.\n\n\
      Mirrors ``treams.pw.permute_xyz`` with ``poltype='helicity'`` and ``inverse=False``.\n\n\
      Differences from treams: ``ValueError`` for a wave vector with \
      ``kx**2 + ky**2 + kz**2 = 0`` and for non-finite input."
    pw_permute_xyz_h: pw_permute_loop::<f64, HELICITY, CYCLIC>, pw_permute_loop::<Complex, HELICITY, CYCLIC>;

    c"pw_permute_xyz_p(kx, ky, kz, p, q, /, out=None, *, where=True)\n\n\
      The amplitude of polarization ``p`` after one cyclic permutation of the Cartesian axes, \
      for the plane wave ``(kx, ky, kz)`` of polarization ``q``, in the parity basis.\n\n\
      Mirrors ``treams.pw.permute_xyz`` with ``poltype='parity'`` and ``inverse=False``.\n\n\
      Differences from treams: ``ValueError`` for a wave vector with \
      ``kx**2 + ky**2 + kz**2 = 0`` and for non-finite input."
    pw_permute_xyz_p: pw_permute_loop::<f64, PARITY, CYCLIC>, pw_permute_loop::<Complex, PARITY, CYCLIC>;

    c"pw_permute_xyz_inverse_h(kx, ky, kz, p, q, /, out=None, *, where=True)\n\n\
      The amplitude of polarization ``p`` after the inverse cyclic permutation of the \
      Cartesian axes, for the plane wave ``(kx, ky, kz)`` of polarization ``q``, in the \
      helicity basis.\n\n\
      Mirrors ``treams.pw.permute_xyz`` with ``poltype='helicity'`` and ``inverse=True``.\n\n\
      Differences from treams: ``ValueError`` for a wave vector with \
      ``kx**2 + ky**2 + kz**2 = 0`` and for non-finite input."
    pw_permute_xyz_inverse_h: pw_permute_loop::<f64, HELICITY, INVERSE>, pw_permute_loop::<Complex, HELICITY, INVERSE>;

    c"pw_permute_xyz_inverse_p(kx, ky, kz, p, q, /, out=None, *, where=True)\n\n\
      The amplitude of polarization ``p`` after the inverse cyclic permutation of the \
      Cartesian axes, for the plane wave ``(kx, ky, kz)`` of polarization ``q``, in the \
      parity basis.\n\n\
      Mirrors ``treams.pw.permute_xyz`` with ``poltype='parity'`` and ``inverse=True``.\n\n\
      Differences from treams: ``ValueError`` for a wave vector with \
      ``kx**2 + ky**2 + kz**2 = 0`` and for non-finite input."
    pw_permute_xyz_inverse_p: pw_permute_loop::<f64, PARITY, INVERSE>, pw_permute_loop::<Complex, PARITY, INVERSE>;

    c"tl_vsw_A(lambda_, mu, l, m, x, theta, phi, /, out=None, *, where=True)\n\n\
      The translation coefficient ``A`` from the parity wave ``(l, m)`` to ``(lambda_, mu)`` \
      of equal parity, from singular to regular vector spherical waves, across the \
      displacement ``(x, theta, phi)`` with ``x = k r``.\n\n\
      Mirrors ``treams.special.tl_vsw_A``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` at ``x = 0``, where treams \
      raises ``ZeroDivisionError``, and for non-finite input."
    tl_vsw_A: tl_vsw_loop::<f64, M, SINGULAR>, tl_vsw_loop::<Complex, M, SINGULAR>;

    c"tl_vsw_B(lambda_, mu, l, m, x, theta, phi, /, out=None, *, where=True)\n\n\
      The translation coefficient ``B`` from the parity wave ``(l, m)`` to ``(lambda_, mu)`` \
      of opposite parity, from singular to regular vector spherical waves, across the \
      displacement ``(x, theta, phi)`` with ``x = k r``.\n\n\
      Mirrors ``treams.special.tl_vsw_B``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` at ``x = 0``, where treams \
      raises ``ZeroDivisionError``, and for non-finite input."
    tl_vsw_B: tl_vsw_loop::<f64, N, SINGULAR>, tl_vsw_loop::<Complex, N, SINGULAR>;

    c"tl_vsw_rA(lambda_, mu, l, m, x, theta, phi, /, out=None, *, where=True)\n\n\
      The translation coefficient ``A`` from the parity wave ``(l, m)`` to ``(lambda_, mu)`` \
      of equal parity, between vector spherical waves of one kind, across the displacement \
      ``(x, theta, phi)`` with ``x = k r``.\n\n\
      Mirrors ``treams.special.tl_vsw_rA``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    tl_vsw_rA: tl_vsw_loop::<f64, M, REGULAR>, tl_vsw_loop::<Complex, M, REGULAR>;

    c"tl_vsw_rB(lambda_, mu, l, m, x, theta, phi, /, out=None, *, where=True)\n\n\
      The translation coefficient ``B`` from the parity wave ``(l, m)`` to ``(lambda_, mu)`` \
      of opposite parity, between vector spherical waves of one kind, across the displacement \
      ``(x, theta, phi)`` with ``x = k r``.\n\n\
      Mirrors ``treams.special.tl_vsw_rB``.\n\n\
      Differences from treams: degrees up to 128; ``ValueError`` for non-finite input."
    tl_vsw_rB: tl_vsw_loop::<f64, N, REGULAR>, tl_vsw_loop::<Complex, N, REGULAR>;
};

/// Ufuncs reached only through the Rust scalar wrappers. A row's attribute and
/// name are those of its wrapper, so error messages show the name the caller
/// used (`translate` for `pw.translate`).
pub(super) const HIDDEN: [Ufunc; 6] = ufuncs! {
    c"translate(kx, ky, kz, x, y, z, /, out=None, *, where=True)\n\n\
      ``exp(i (kx x + ky y + kz z))``, the phase that translates a plane wave by \
      ``(x, y, z)``.\n\n\
      Mirrors ``treams.pw.translate``.\n\n\
      Differences from treams: real positions only; ``ValueError`` for non-finite input."
    pw_translate as "translate": pw_translate_loop::<f64>, pw_translate_loop::<Complex>;

    c"vpw_M(kx, ky, kz, x, y, z, /, out=None)\n\n\
      ``M_k(r) = -i phi_hat(k) exp(i k.r)``, the vector plane wave ``M``, in Cartesian \
      components.\n\n\
      Mirrors ``treams.special.vpw_M``.\n\n\
      Differences from treams: ``ValueError`` for ``kx**2 + ky**2 + kz**2 = 0`` and for \
      non-finite input."
    vpw_M: plane_wave_loop::<f64, M, 0>, plane_wave_loop::<Complex, M, 0>;

    c"vpw_N(kx, ky, kz, x, y, z, /, out=None)\n\n\
      ``N_k(r) = -theta_hat(k) exp(i k.r)``, the vector plane wave ``N``, in Cartesian \
      components.\n\n\
      Mirrors ``treams.special.vpw_N``.\n\n\
      Differences from treams: ``ValueError`` for ``kx**2 + ky**2 + kz**2 = 0`` and for \
      non-finite input."
    vpw_N: plane_wave_loop::<f64, N, 0>, plane_wave_loop::<Complex, N, 0>;

    c"vpw_A(kx, ky, kz, x, y, z, p, /, out=None)\n\n\
      ``A_k(r) = (N_k(r) +- M_k(r)) / sqrt(2)``, the vector plane wave of helicity ``p`` (1 \
      positive, 0 negative), in Cartesian components.\n\n\
      Mirrors ``treams.special.vpw_A``.\n\n\
      Differences from treams: ``ValueError`` for ``kx**2 + ky**2 + kz**2 = 0`` and for \
      non-finite input."
    vpw_A: plane_wave_loop::<f64, A, 1>, plane_wave_loop::<Complex, A, 1>;

    c"volume(a, /, out=None)\n\n\
      The signed volume (area, length) of the cell whose rows ``a`` are the lattice vectors; \
      integer cells give integers.\n\n\
      Mirrors ``treams.lattice.volume``.\n\n\
      Differences from treams: also one-dimensional cells."
    cell_volume as "volume" "(i,i)->()": volume_loop::<Wrapping<c_long>>, volume_loop::<f64>;

    c"reciprocal(a, /, out=None)\n\n\
      The reciprocal lattice vectors ``b`` as rows, with ``a_i . b_j = 2 pi delta_ij`` for \
      the lattice vectors ``a_i``, the rows of ``a``.\n\n\
      Mirrors ``treams.lattice.reciprocal``.\n\n\
      Differences from treams: also one-dimensional cells."
    cell_reciprocal as "reciprocal" "(i,i)->(i,i)": reciprocal_loop;
};

// The dtype rows derived for a few loops, checked when the crate compiles.
const _: () = {
    const D: c_char = NPY_TYPES::NPY_DOUBLE as c_char;
    const Z: c_char = NPY_TYPES::NPY_CDOUBLE as c_char;
    const L: c_char = NPY_TYPES::NPY_LONG as c_char;
    let jv = Loop::new(bessel_loop::<J, CYLINDRICAL, VALUE>);
    assert!(matches!(jv.types(), [D, Z, Z])); // dD->D
    let wigner3j = Loop::new(wigner3j_loop);
    assert!(matches!(wigner3j.types(), [D, D, D, D, D, D, D])); // dddddd->d
    let vsw = Loop::new(vsw_a_loop::<Complex, REGULAR>);
    assert!(matches!(vsw.types(), [L, L, Z, Z, D, L, Z]) && matches!(vsw.components, Some(3)));
    let sum = Loop::new(lattice_loop::<c_long, c_long, SPHERICAL, 2, SHIFTED, 2>);
    assert!(matches!(sum.types(), [L, L, Z, D, D, D, L, Z]) && sum.components.is_none());
};

/// Register the eight lattice sums `prefix` + family whose last operand is `E`;
/// `part` says what they return, as in "The Ewald sum".
pub(super) fn lattice_family<E: Tail>(
    module: &Bound<'_, PyModule>,
    prefix: &str,
    part: &str,
) -> PyResult<()> {
    fn sums<E: Tail, const S: bool, const DIM: usize, const SHIFT: bool, const LABELS: usize>(
        module: &Bound<'_, PyModule>,
        name: String,
        [part, failures]: [&str; 2],
        geometry: &str,
    ) -> PyResult<()> {
        let loops = [
            Loop::new(lattice_loop::<c_long, E, S, DIM, SHIFT, LABELS>),
            Loop::new(lattice_loop::<f64, E, S, DIM, SHIFT, LABELS>),
        ];
        // Only direct shells (integer tails) take integer mode labels too.
        let loops = if E::TYPE == c_long::TYPE {
            &loops[..]
        } else {
            &loops[1..]
        };
        let core = lattice_core(LABELS, S, DIM, SHIFT);
        let doc = lattice_doc::<E>(&name, [part, failures], geometry, S, LABELS, core.is_none())?;
        let function = create(module.py(), &name, loops, core.as_deref(), doc)?;
        module.add(name, function)
    }
    // The reciprocal-space terms diverge at a diffraction threshold, and the Ewald
    // parts stop where their series do not converge.
    let failures = match prefix {
        "lsum" => {
            " It also raises ``ValueError`` at a diffraction threshold ``k = |kpar + G|``, with \
             ``G`` a reciprocal lattice vector, where a part does not converge and where the \
             split ``eta`` is too small for full accuracy."
        }
        "recsum" => {
            " It also raises ``ValueError`` at a diffraction threshold ``k = |kpar + G|``, with \
             ``G`` a reciprocal lattice vector, where the sum does not converge and where the \
             split ``eta`` is too small for full accuracy."
        }
        "realsum" => {
            " It also raises ``ValueError`` where the sum does not converge and where the split \
             ``eta`` is too small for full accuracy."
        }
        _ => "",
    };
    let text = [part, failures];
    let name = |family: &str| format!("{prefix}{family}");
    sums::<E, SPHERICAL, 1, UNSHIFTED, 1>(
        module,
        name("sw1d"),
        text,
        "The lattice lies on the z axis: ``a`` is the pitch, ``kpar`` the Bloch wavenumber \
         and ``r`` the shift along z; ``m = 0``.",
    )?;
    sums::<E, SPHERICAL, 1, SHIFTED, 2>(
        module,
        name("sw1d_shift"),
        text,
        "The lattice lies on the z axis: ``a`` is the pitch, ``kpar`` the Bloch wavenumber \
         and ``r`` a shift ``(x, y, z)``.",
    )?;
    sums::<E, SPHERICAL, 2, UNSHIFTED, 2>(
        module,
        name("sw2d"),
        text,
        "The lattice lies in the xy plane: the rows of ``a``, shape ``(2, 2)``, are the \
         lattice vectors, ``kpar`` has shape ``(2,)`` and ``r`` is a shift ``(x, y)``.",
    )?;
    sums::<E, SPHERICAL, 2, SHIFTED, 2>(
        module,
        name("sw2d_shift"),
        text,
        "The lattice lies in the xy plane: the rows of ``a``, shape ``(2, 2)``, are the \
         lattice vectors, ``kpar`` has shape ``(2,)`` and ``r`` is a shift ``(x, y, z)``.",
    )?;
    sums::<E, SPHERICAL, 3, UNSHIFTED, 2>(
        module,
        name("sw3d"),
        text,
        "The rows of ``a``, shape ``(3, 3)``, are the lattice vectors; ``kpar`` and ``r`` \
         have shape ``(3,)``.",
    )?;
    sums::<E, CYLINDRICAL, 1, UNSHIFTED, 1>(
        module,
        name("cw1d"),
        text,
        "The lattice lies on the x axis: ``a`` is the pitch, ``kpar`` the Bloch wavenumber \
         and ``r`` the shift along x.",
    )?;
    sums::<E, CYLINDRICAL, 1, SHIFTED, 1>(
        module,
        name("cw1d_shift"),
        text,
        "The lattice lies on the x axis: ``a`` is the pitch, ``kpar`` the Bloch wavenumber \
         and ``r`` a shift ``(x, y)``.",
    )?;
    sums::<E, CYLINDRICAL, 2, UNSHIFTED, 1>(
        module,
        name("cw2d"),
        text,
        "The lattice lies in the xy plane: the rows of ``a``, shape ``(2, 2)``, are the \
         lattice vectors; ``kpar`` and ``r`` have shape ``(2,)``.",
    )
}

/// The docstring of the lattice sum `name`, leaked like its name: the call, `part`
/// of the sum and its summand, the last operand, the lattice `geometry`, the treams
/// function and the differences from it, which end with the part's `failures`.
fn lattice_doc<E: Tail>(
    name: &str,
    [part, failures]: [&str; 2],
    geometry: &str,
    spherical: bool,
    labels: usize,
    elementwise: bool,
) -> PyResult<&'static CStr> {
    let order_check = if spherical && labels == 2 {
        "``|m| > l``, "
    } else {
        ""
    };
    let (labels, summand, bound) = if spherical {
        (
            if labels == 2 { "l, m" } else { "l" },
            "``h_l(k |r + R|) Y_lm(-r - R) exp(i kpar.R)``,",
            "degrees up to 128",
        )
    } else {
        (
            "m",
            "``H_m(k |r + R|) exp(i m phi) exp(i kpar.R)``, where ``phi`` is the azimuth of \
             ``-r - R``,",
            "orders ``|m|`` up to 128",
        )
    };
    let (tail, about_tail) = if E::TYPE == c_long::TYPE {
        (
            "i",
            "Shell ``i`` holds the lattice points whose integer coordinates in the basis ``a`` \
             have the largest magnitude ``i``; the shells ``0, 1, 2, ...`` add up to the sum.",
        )
    } else {
        (
            "eta",
            "``eta`` sets how the sum divides into a real-space and a reciprocal-space part \
             (the Ewald split); ``eta = 0`` picks it automatically.",
        )
    };
    let options = if elementwise {
        "out=None, *, where=True"
    } else {
        "out=None"
    };
    let doc = format!(
        "{name}({labels}, k, kpar, a, r, {tail}, /, {options})\n\n\
         {part} over the lattice points ``R`` of {summand} without the term at ``r + R = 0``. \
         {about_tail} {geometry}\n\n\
         Mirrors ``treams.lattice.{name}``.\n\n\
         Differences from treams: {bound}; ``ValueError`` for {order_check}non-finite input, \
         ``k = 0`` and a cell of zero volume.{failures}"
    );
    Ok(Box::leak(CString::new(doc)?.into_boxed_c_str()))
}

/// Docstrings of the point and vector ufunc of each transform, in `TRANSFORMS` order.
const COORDINATE_DOCS: [[&CStr; 2]; 8] = [
    [
        c"car2cyl(r, /, out=None)\n\n\
          Cartesian points ``r`` in cylindrical coordinates; the last axis holds the \
          components.\n\n\
          Mirrors ``treams.special.car2cyl``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
        c"vcar2cyl(v, r, /, out=None)\n\n\
          Cartesian vector components ``v`` at the Cartesian points ``r`` in the cylindrical \
          basis; the last axis holds the components.\n\n\
          Mirrors ``treams.special.vcar2cyl``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
    ],
    [
        c"car2sph(r, /, out=None)\n\n\
          Cartesian points ``r`` in spherical coordinates; the last axis holds the \
          components.\n\n\
          Mirrors ``treams.special.car2sph``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
        c"vcar2sph(v, r, /, out=None)\n\n\
          Cartesian vector components ``v`` at the Cartesian points ``r`` in the spherical \
          basis; the last axis holds the components.\n\n\
          Mirrors ``treams.special.vcar2sph``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
    ],
    [
        c"cyl2car(r, /, out=None)\n\n\
          Cylindrical points ``r`` in Cartesian coordinates; the last axis holds the \
          components.\n\n\
          Mirrors ``treams.special.cyl2car``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
        c"vcyl2car(v, r, /, out=None)\n\n\
          Cylindrical vector components ``v`` at the cylindrical points ``r`` in the \
          Cartesian basis; the last axis holds the components.\n\n\
          Mirrors ``treams.special.vcyl2car``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
    ],
    [
        c"cyl2sph(r, /, out=None)\n\n\
          Cylindrical points ``r`` in spherical coordinates; the last axis holds the \
          components.\n\n\
          Mirrors ``treams.special.cyl2sph``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
        c"vcyl2sph(v, r, /, out=None)\n\n\
          Cylindrical vector components ``v`` at the cylindrical points ``r`` in the \
          spherical basis; the last axis holds the components.\n\n\
          Mirrors ``treams.special.vcyl2sph``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
    ],
    [
        c"sph2car(r, /, out=None)\n\n\
          Spherical points ``r`` in Cartesian coordinates; the last axis holds the \
          components.\n\n\
          Mirrors ``treams.special.sph2car``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
        c"vsph2car(v, r, /, out=None)\n\n\
          Spherical vector components ``v`` at the spherical points ``r`` in the Cartesian \
          basis; the last axis holds the components.\n\n\
          Mirrors ``treams.special.vsph2car``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
    ],
    [
        c"sph2cyl(r, /, out=None)\n\n\
          Spherical points ``r`` in cylindrical coordinates; the last axis holds the \
          components.\n\n\
          Mirrors ``treams.special.sph2cyl``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
        c"vsph2cyl(v, r, /, out=None)\n\n\
          Spherical vector components ``v`` at the spherical points ``r`` in the cylindrical \
          basis; the last axis holds the components.\n\n\
          Mirrors ``treams.special.vsph2cyl``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
    ],
    [
        c"car2pol(r, /, out=None)\n\n\
          Cartesian points ``r`` in polar coordinates; the last axis holds the components.\n\n\
          Mirrors ``treams.special.car2pol``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
        c"vcar2pol(v, r, /, out=None)\n\n\
          Cartesian vector components ``v`` at the Cartesian points ``r`` in the polar basis; \
          the last axis holds the components.\n\n\
          Mirrors ``treams.special.vcar2pol``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
    ],
    [
        c"pol2car(r, /, out=None)\n\n\
          Polar points ``r`` in Cartesian coordinates; the last axis holds the components.\n\n\
          Mirrors ``treams.special.pol2car``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
        c"vpol2car(v, r, /, out=None)\n\n\
          Polar vector components ``v`` at the polar points ``r`` in the Cartesian basis; the \
          last axis holds the components.\n\n\
          Mirrors ``treams.special.vpol2car``.\n\n\
          Differences from treams: ``ValueError`` for non-finite input.",
    ],
];

/// Point and vector ufuncs of one coordinate transform.
pub(super) fn coordinate_ufuncs<const KIND: u8>(py: Python<'_>) -> PyResult<[Py<PyAny>; 2]> {
    let (point, vector, transform) = crate::coordinates::TRANSFORMS[usize::from(KIND)];
    let [point_doc, vector_doc] = COORDINATE_DOCS[usize::from(KIND)];
    let dim = transform.dimension();
    let ufunc = |name, loops: &[Loop], core: String, doc| {
        Ok::<_, PyErr>(create(py, name, loops, Some(&core), doc)?.unbind())
    };
    let points = [Loop::new(coordinate_loop::<f64, KIND, 0>)];
    let vectors = [
        Loop::new(coordinate_loop::<f64, KIND, 1>),
        Loop::new(coordinate_loop::<Complex, KIND, 1>),
    ];
    Ok([
        ufunc(point, &points, format!("({dim})->({dim})"), point_doc)?,
        ufunc(
            vector,
            &vectors,
            format!("({dim}),({dim})->({dim})"),
            vector_doc,
        )?,
    ])
}

/// Whether `doc` opens with `name(`, the call that the API reference shows.
const fn opens_with_call(doc: &CStr, name: &str) -> bool {
    let (doc, name) = (doc.to_bytes(), name.as_bytes());
    if doc.len() <= name.len() || doc[name.len()] != b'(' {
        return false;
    }
    let mut i = 0;
    while i < name.len() {
        if doc[i] != name[i] {
            return false;
        }
        i += 1;
    }
    true
}

// Every static docstring opens with the call under its ufunc name, checked when the
// crate compiles; the lattice docstrings start with the name by construction.
const _: () = {
    let mut i = 0;
    while i < UFUNCS.len() {
        assert!(opens_with_call(UFUNCS[i].doc, UFUNCS[i].name));
        i += 1;
    }
    let mut i = 0;
    while i < HIDDEN.len() {
        assert!(opens_with_call(HIDDEN[i].doc, HIDDEN[i].name));
        i += 1;
    }
    let mut i = 0;
    while i < COORDINATE_DOCS.len() {
        let (point, vector, _) = crate::coordinates::TRANSFORMS[i];
        assert!(opens_with_call(COORDINATE_DOCS[i][0], point));
        assert!(opens_with_call(COORDINATE_DOCS[i][1], vector));
        i += 1;
    }
};
