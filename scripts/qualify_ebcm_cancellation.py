# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy", "mpmath", "treams==0.4.5"]
# ///
"""Diagnose the degree-6 m=0 EBCM cancellation floor without relaxing benchmarks.

Run in the development environment:
uv run --no-sync --with mpmath python scripts/qualify_ebcm_cancellation.py
"""

import json
import warnings

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from treams import ebcm as upstream

from treams_rs import _native, ebcm


def wave(degree, argument, theta, pol, outgoing):
    """Independent high-precision m=0 spherical helicity wave in (r, theta, phi)."""
    cosine = mp.cos(theta)
    previous, polynomial = mp.mpf(1), cosine
    dprevious, derivative = mp.mpf(0), mp.mpf(1)
    for n in range(2, degree + 1):
        previous, polynomial = (
            polynomial,
            ((2 * n - 1) * cosine * polynomial - (n - 1) * previous) / n,
        )
        dprevious, derivative = (
            derivative,
            ((2 * n - 1) * (previous + cosine * derivative) - (n - 1) * dprevious) / n,
        )
    tau = -mp.sin(theta) * derivative
    radial = mp.hankel1 if outgoing else mp.besselj
    value = mp.sqrt(mp.pi / (2 * argument)) * radial(degree + mp.mpf("0.5"), argument)
    first = degree * value / argument - mp.sqrt(mp.pi / (2 * argument)) * radial(
        degree + mp.mpf("1.5"), argument
    )
    normalization = 1j * mp.sqrt((2 * degree + 1) / (8 * mp.pi * degree * (degree + 1)))
    return (
        normalization * degree * (degree + 1) * polynomial * value / argument,
        normalization * tau * (value / argument + first),
        -(2 * pol - 1) * normalization * tau * value,
    )


def main():
    mp.mp.dps = 80
    index = np.sqrt((3.1 + 0.2j) * (1.2 + 0.1j))
    ks = 1.3 * np.array([[index - 0.07, index + 0.07], [1, 1]])
    zs = [np.sqrt((1.2 + 0.1j) / (3.1 + 0.2j)), 1.0]
    out = ([1, 1, 3, 3], [0] * 4, [1, 0, 1, 0])
    incoming = ([6, 6], [0, 0], [1, 0])

    def radius(theta):
        return 0.3 * (1 + 0.23 * np.cos(theta) ** 2)

    def slope(theta):
        return -0.138 * np.cos(theta) * np.sin(theta)

    native = {
        order: ebcm.qmat(radius, slope, ks, zs, out, incoming, order=order, legacy=True)
        for order in (48, 96, 192)
    }
    results = []
    for i, (degree, pol) in enumerate(zip(out[0], out[2], strict=True)):
        for j, source_pol in enumerate(incoming[2]):
            fr = upstream._j_real(
                degree, 6, 0, radius, slope, pol, source_pol, *ks, *zs
            )
            fi = upstream._j_imag(
                degree, 6, 0, radius, slope, pol, source_pol, *ks, *zs
            )

            def precise(theta, degree=degree, pol=pol, source_pol=source_pol):
                r = mp.mpf("0.3") * (1 + mp.mpf("0.23") * mp.cos(theta) ** 2)
                dr = -mp.mpf("0.138") * mp.cos(theta) * mp.sin(theta)
                a = wave(degree, mp.mpc(ks[0, pol]) * r, theta, pol, False)
                b = wave(6, mp.mpc(ks[1, source_pol]) * r, theta, source_pol, True)
                return (
                    mp.sin(theta)
                    * (
                        (2 * pol - 1) * mp.mpc(zs[1])
                        + (2 * source_pol - 1) * mp.mpc(zs[0])
                    )
                    * (
                        r * (a[1] * b[2] - a[2] * b[1])
                        - dr * (a[2] * b[0] - a[0] * b[2])
                    )
                )

            sample = 0.413
            np.testing.assert_allclose(
                complex(precise(mp.mpf(sample))),
                fr(sample) + 1j * fi(sample),
                rtol=2e-13,
            )
            reflected_error = max(
                abs(precise(t) + precise(mp.pi - t))
                for t in [mp.mpf("0.2"), mp.mpf("0.7"), mp.mpf("1.1")]
            )
            exact = mp.quad(precise, [0, mp.pi], method="gauss-legendre", maxdegree=4)
            with warnings.catch_warnings(record=True) as captured:
                reference = quad(fr, 0, np.pi)[0] + 1j * quad(fi, 0, np.pi)[0]
                tight = (
                    quad(fr, 0, np.pi, epsabs=1e-13, epsrel=1e-13)[0]
                    + 1j * quad(fi, 0, np.pi, epsabs=1e-13, epsrel=1e-13)[0]
                )
                mass = quad(lambda t, fr=fr, fi=fi: abs(fr(t) + 1j * fi(t)), 0, np.pi)[
                    0
                ]
            assert abs(exact) < mp.mpf("1e-65")
            assert reflected_error < mp.mpf("1e-65")
            results.append(
                {
                    "out": [degree, 0, pol],
                    "in": [6, 0, source_pol],
                    "analytic_value": 0,
                    "mpmath_absolute_value": str(abs(exact)),
                    "mpmath_reflection_error": str(reflected_error),
                    "integral_of_absolute_integrand": mass,
                    "epsilon_times_absolute_integral": np.finfo(float).eps * mass,
                    "upstream_default": [reference.real, reference.imag],
                    "upstream_tight": [tight.real, tight.imag],
                    "quadrature_warnings": sorted({str(w.message) for w in captured}),
                    "native_by_quadrature_order": {
                        order: [a[i, j].real, a[i, j].imag]
                        for order, a in native.items()
                    },
                }
            )
    print(
        json.dumps(
            {
                "native_profile": _native.build_profile(),
                "mpmath_digits": mp.mp.dps,
                "symmetry": "For m=0 and odd l_out+l_in, this equatorially symmetric surface has an odd integrand under theta -> pi-theta.",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
