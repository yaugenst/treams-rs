# GPU opportunities and limits

The RTX 4080 SUPER result for weighted plane fields measures one implementation
and workload. It is not a GPU speedup ceiling. Conversely, the dense complex128
qualification does not support expecting a large acceleration from moving any
large matrix to this GPU: the tested persistent triangular solves were slower
than the optimized CPU solves, and large dense factorizations were near parity.
See [the recorded dense qualification](../benchmarks/gpu-qualification.json).

The follow-up improved the existing field kernel by 12–22% in warm host-to-host
time and qualified [repeated sampling](gpu-sampling.md) at 7.12–7.74× versus the
strongest prepared CPU alternative tested. Setup and operator memory limit the
latter to workloads with enough reuse. That establishes a concrete larger win;
it does not establish a similar gain for dense factorization or all scattering
workloads. The default numerical contract remains complex128/f64.

The following opportunities are ranked by proximity to existing code and
reusable work. Beyond the measured plane-field and sampling paths, these are
hypotheses to qualify, not measured speedup claims.

| Opportunity | Concrete workload | What would change | Main limitation |
| --- | --- | --- | --- |
| Plane-field throughput and repeated sampling | Many plane modes evaluated at a large set of points, or changing illumination coefficients at fixed points | The loop and resident sampling/pullback are now qualified; batching many coefficients and reducing scalar objectives on-device remain opportunities | Expensive double-precision exponentials/trigonometry in the fused path; cached operator memory, setup, and transfer costs in the sampling path |
| Spherical/cylindrical field maps and their pullbacks | Large near-field maps or objectives sampled around many particles | Fuse field evaluation and native cotangent contractions over samples; reuse radial/angular values shared by modes | Complex special functions, axis/origin cases, recurrence stability and gradient reductions need numerical qualification |
| Matrix-free particle interaction and adjoint | Many low-order particles with a few requested illuminations | Generate and contract translation blocks on the GPU, retaining Krylov vectors and reductions there | Pair work is still quadratic; repeated radial evaluation and poor convergence can dominate; a GPU does not fix the preconditioner |
| Many independent spectra or geometries | Large batches of small or medium scattering problems, preferably sharing mode order and topology | Batch operator construction, solves and reduced observables; transfer inputs and final observables | Each frequency/geometry generally changes the factorization; uploading CPU-built matrices alone leaves much of the work on the host |
| Ewald/Bloch parameter sweeps | Many periodic lattice sums across frequencies or Bloch vectors | Batch shell terms and share recurrence tables across angular orders | Adaptive convergence, cancellation and complex special functions make isolated small sums poor GPU workloads |

The CPU already avoids a full point-by-mode matrix for weighted fields and a
global dense coupling matrix for matrix-free sphere solves. Those memory savings
must not be credited to GPU hardware. Likewise, improving radial-table reuse or
avoiding unnecessary illuminations can benefit both processors.

At fixed geometry, wavenumber and sample points, several illuminations can reuse
the same sampling operator. Applying that operator and its Hermitian transpose
with dense algebra is useful when construction is expensive and it fits in
memory. Both CPU and GPU baselines must get this reuse. A three-component
complex128 operator with 262,144 points and 1,024 modes already takes 12 GiB,
before coefficients, outputs and library workspace. Tiling or recomputation is
necessary for larger cases; explicitly caching the entire operator is not a
general replacement for the fused field path.

FFT or NUFFT methods could reduce the amount of work for structured sampling or
translation-invariant arrangements. This is an algorithmic specialization,
separate from GPU acceleration. Arbitrary spherical multipoles, irregular
geometries and complex evanescent wavevectors do not become a standard Fourier
transform merely because their field contains a phase factor. Any approximation
would need an explicit tolerance and comparison with the direct field operator
and its adjoint.

FP32 or mixed-precision solvers are a separate possible backend, not an implicit
optimization of the complex128 path. Iterative refinement can only be relied on
when high-precision residuals and convergence checks establish the requested
accuracy. Resonances and ill-conditioned high-order multipoles are particularly
important counterexamples to a blanket lower-precision policy. No such precision
change is part of this investigation.
