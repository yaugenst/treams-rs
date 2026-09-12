# Qualification against published applications

These checks exercise published scientific workflows, including original authors'
numerical data. They complement kernel tests; agreement with another implementation
alone is not an independent proof of accuracy. Every comparison names its source,
sampling grid, truncation order, and tolerance. Runtime timings in these results
are execution diagnostics, not controlled performance benchmarks.

![Electron-beam spectra and periodic-array transmission, with author data and upstream comparisons](../benchmarks/papers/qualification.png)

Lines use the full original notebook grids; circles show the separately sampled
author reference tables. Narrow cylinder resonances explain the visual difference
between line interpolation and the sparse points. Both grids are evaluated and
checked separately. The cylinder notebook and regression table also use the
slightly different dispersion constants recorded below.

## Electron-beam spectroscopy

Stamatopoulou and Rockstuhl,
[A T-matrix scattering formalism for electron-beam spectroscopy](https://arxiv.org/abs/2602.12743v1),
arXiv:2602.12743v1 (2026), Figure 2(a,b).

The public [author repository](https://github.com/tfp-photonics/treams_ebeam/tree/9aa9974d0e56c60873dc880f11557e622d233ea9)
provides sphere/cylinder notebooks and stored regression spectra. We evaluate the
electron source, cylindrical translation, cylindrical-to-spherical conversion,
Mie response, and cathodoluminescence (CL) / electron energy loss (EELS) pairings
using `treams-rs`. The observable formulas follow paper equations 6, 21, 25, and 26.
The same explicit calculations run against upstream `treams==0.4.5`.

Both particles have radius 50 nm, electron impact parameter 60 nm, and reduced
speed 0.7. The sphere has permittivity `16+0.5j` and order 4; the infinite cylinder
uses the specified silver-like Drude model and order 12. We reproduce both full
200-point notebook grids. Separately, 50 points per object are checked against
the authors' stored test values (200 CL/EELS values). These test grids differ from
the notebooks. The cylinder notebook's rounded `hbar=6.582e-16 eV s` is preserved
in its dispersion model; the stored-test calculation uses SciPy constants.

Executing the authors' actual numerical notebook cells exposed a cylinder-source
error: the notebook omits the conversion from the helicity T-matrix to the parity
electron-source basis. Its own regression test includes that conversion. We apply
that one-line repair explicitly; the unmodified notebook differs by as much as
50% in EELS. The unmodified sphere notebook and repaired cylinder notebook match
our results within `8e-17` absolute. Both original and repaired execution values
are preserved in [ebeam-notebook-execution.json](../benchmarks/papers/ebeam-notebook-execution.json).
The plotted cylinder curve is therefore a **corrected source reproduction**, not
a claim to reproduce the unmodified notebook's inconsistent polarization pairing.

The independent stored-data tolerance is `atol=1e-8, rtol=0`, matching the precision
of the rounded source tables. Maximum error is `4.84e-9` for the sphere and
`2.66e-12` for the cylinder. Fresh upstream calculations agree much more closely.
CL and EELS are nonnegative, CL does not exceed EELS, and rotational invariance of
the sphere is checked with Hypothesis. Cylinder CL vanishes because every axial
channel is evanescent. Increasing the multipole cutoff by two at five frequencies
changes sphere EELS by up to 0.24% and cylinder EELS by up to 0.62%; reproducing the
author's truncation is not a claim of arbitrary-precision convergence.

Sources: [sphere notebook](https://github.com/tfp-photonics/treams_ebeam/blob/9aa9974d0e56c60873dc880f11557e622d233ea9/docs/sphere.ipynb),
[cylinder notebook](https://github.com/tfp-photonics/treams_ebeam/blob/9aa9974d0e56c60873dc880f11557e622d233ea9/docs/cylinder.ipynb),
[sphere data](https://github.com/tfp-photonics/treams_ebeam/blob/9aa9974d0e56c60873dc880f11557e622d233ea9/tests/test_cl_eels_sphere.py),
[cylinder data](https://github.com/tfp-photonics/treams_ebeam/blob/9aa9974d0e56c60873dc880f11557e622d233ea9/tests/test_cl_eels_cylinder.py).
Only numerical reference values are stored locally, with source hashes and attribution;
the addon source implementation is not vendored.

## CPC treams paper companion spectra

Beutel, Fernandez-Corbaton, and Rockstuhl,
[treams – a T-matrix-based scattering code for nanophotonics](https://doi.org/10.1016/j.cpc.2023.109076),
Computer Physics Communications 297, 109076 (2024).

Table 3 refers readers to accompanying executable examples. We reproduce the
200-point dielectric sphere efficiency spectrum, 50-point absorbing chiral slab
transmission/reflection spectra, and periodic chiral spheres above a dielectric
slab. Parameters and grids come from immutable upstream commit
`1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39`:
[sphere](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/sphere.py),
[slab](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/slab.py),
[periodic array](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/array_spheres.py).
These are companion-code reproductions, not digitized plots from the article.

All qualified samples agree with fresh upstream values within
`rtol=2e-9, atol=2e-11`. Passive sphere/slab inequalities hold; the lossless periodic
stack conserves `T+R=1` within `1e-10`. The original array order 3 differs from order
5 by up to `1.1e-4` in transmittance at the five convergence samples.

The array's final original sample, wavelength 350 nm, is an exact diffraction
threshold and is explicitly recorded as excluded; 99 interior samples qualify.
Upstream assigns finite surrogates to both the
[radiation pole](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/src/treams/sw.pyx#L660-L678)
and the
[lattice pole](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/src/treams/lattice/_esum.pyx#L84-L95).
That finite output is not an independent exact-threshold reference. Four extra
evaluations at `350*(1±1e-4)` and `350*(1±1e-6)` nm verify agreement away from the
pole, power conservation, and convergence of the two sides. This does not silently
replace the author's endpoint or claim an exact endpoint/derivative formulation.

The article's quasi-BIC Figure 5 is **not reproduced** here. Appendix F supplies
the algorithm, but `ellipsoid.h5` is absent from the retrieved
[arXiv ancillary archive](https://arxiv.org/src/2309.03182v1/anc).
The publisher supplement returned HTTP 403 during this investigation. Replacing
the unavailable FEM matrix by another ellipsoid model would be a different
qualification, so no Figure 5 claim is made.

## Thermal radiation

Mazo-Vásquez et al.,
[Studying thermal radiation with T matrices](https://doi.org/10.1103/41m5-9ztm),
Physical Review B 112, 054307 (2025), Figure 2(a,c).
The [original notebook and data](https://github.com/jdmazo-vasquez/TMatricesThermalRadiation/tree/ec710fc1f15f379d126659043ef1c2ce4f0eafae)
describe four 250 nm SiC spheres separated by 520 nm, with a temperature of 500 K.
Their material data and all 300 original frequencies are retained. At multipole
order 10, the local interaction system has 960 channels and is expanded into a
240-channel global basis. We reproduce the absorption spectrum and three complete
200-angle thermal-emission cuts against the authors' original numeric tables.

All 300 absorption values agree within **0.577%**; all 600 angular values agree
within **2.18e-7 relative**. The tolerance for the absorption reproduction is 1%,
consistent with the paper's stated 99% multipole-convergence criterion. Increasing
the cutoff through order 12, and through order 16 at the lowest frequency, changes
order-10 absorption by at most **0.057%** over 22 checks. Positive absorption and
azimuth/helicity symmetry are verified; the maximum symmetry discrepancy is
`1.56e-14`. A 52-point upstream comparison is retained separately.

This application exposed severe small-particle, high-order scaling in the dense
interaction solve. Native equilibration removes the observed sensitivity: at the
lowest frequency the stable order-10 value is `1.7544335861e-4 µm²`, while the
author table gives `1.7464535052e-4 µm²`. The remaining disagreement with those
historical values is therefore documented, not hidden behind an upstream-equality
claim. Original angular filenames use an earlier figure numbering; the source
fixture records their mapping, units, and the stale notebook save-cell convention.

Results: [thermal-result.json](../benchmarks/papers/thermal-result.json),
[source data and hashes](../benchmarks/papers/thermal-source.json),
[upstream comparison](../benchmarks/papers/thermal-upstream-comparison.json).

![Thermal absorption and directional emission compared with the author's numeric figure data](../benchmarks/papers/thermal-qualification.png)

```sh
uv run --no-sync python scripts/papers_thermal.py --samples 300 --convergence \
  --upstream-reference benchmarks/papers/thermal-upstream-comparison.json
```

This command recomputes every native sample and reuses the identical-parameter
upstream subset. Omit `--upstream-reference` to recompute upstream for all 300
frequencies too. Source material is fixed to the cited commit; default operation
uses the local numerical fixture. This is scientific reproduction of selected
figures, not replication of every example in the publication.

## Reproduce and inspect

From a built checkout:

```sh
uv run --no-sync python scripts/qualify_papers.py
uv run --no-sync pytest -q tests/test_paper_qualification.py
uv run --no-sync --with matplotlib python scripts/qualify_papers.py --plot
```

The default script takes seconds. It checks full spectra, physical inequalities,
and source-data tolerances, and writes
[qualification.json](../benchmarks/papers/qualification.json).
The JSON includes all grids and values, package versions, and the installed native
binary and script SHA-256 hashes. Source URLs and hashes are in
[source-provenance.json](../benchmarks/papers/source-provenance.json).
The compact [author reference tables](../benchmarks/papers/ebeam-author-reference.json)
are exercised by ordinary tests, so the reference checks do not require network
access. The 20 inexpensive tests include resonance samples, two original thermal
absorption samples, and a Hypothesis rotation/passivity invariant.
