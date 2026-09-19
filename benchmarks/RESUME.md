# Linux core qualification

The Linux correctness run is complete for commit `2843a70` on this machine
(Ryzen 9 9950X, Python 3.13.1). The [qualification record](linux-core-qualification.json)
links the accepted 615 cases, full thermal reproduction, native build hash,
raw evidence and passing audits. No numerical implementation change was needed.
macOS is outside the current task. Browser and GPU work remain deferred on their
separate experimental branches.

## Accepted evidence

- `results/linux-core-correctness-frozen-20260919/`: independent references,
  physical invariants, published workflows, high-order cluster references,
  527 upstream comparisons and the full thermal reproduction.
- `results/linux-core-correctness-20260919/`: 51 gradient and 33 boundary cases.
  Its first three accuracy groups are retained but superseded: the installed
  extension changed during the initial upstream group. The rerun used a copied
  package matching the binary used by the gradient and boundary groups.
- Both accepted audits have zero integrity errors and zero unresolved native
  numerical gates. The 18 overflowing reference inputs and five upstream cluster
  disagreements remain explicit; independent cluster-reference checks pass.

The raw evidence, including the superseded run, is preserved in
[`linux-core-qualification-20260919.tar.gz`](results/linux-core-qualification-20260919.tar.gz).
Its checksum is in the qualification record. Extract from the repository root:

```sh
tar -xzf benchmarks/results/linux-core-qualification-20260919.tar.gz
```

To repeat the integrity audits while the qualified package and source are installed:

```sh
uv run --script scripts/audit_qualification.py "$PWD" linux \
  accuracy-references accuracy-physics accuracy-upstream --cohort corrected \
  --results-root benchmarks/results/linux-core-correctness-frozen-20260919
uv run --script scripts/audit_qualification.py "$PWD" linux \
  gradient correction-boundary --cohort corrected \
  --results-root benchmarks/results/linux-core-correctness-20260919
```

The auditor checks source/build identities, case coverage, original tolerances,
numerical gate fields, timing consistency and archived array hashes/ZIP integrity.
It does not rerun the solver or recompute the numerical arrays.

## Remaining work

The other 685 cases in `correction-plan.json` cover the broad performance grid,
size/thread scaling and requested illuminations. They have not been rerun, so the
proposed dense-memory improvement and fresh overall speed/memory claims remain
unqualified. Keep this separate from the completed correctness acceptance.

If resuming those measurements, use one frozen release package on an otherwise
idle Linux host, preserve the declared repeats/tolerances and 16 GiB process-group
guard, record new fingerprints, and retain every outcome in fresh directories.
Audit and visually inspect the final performance report before updating claims.

Historical Mac/Linux results and September 12 pause records remain unchanged in
`results/comparison/`, `results/corrected/`, `results/corrections/` and
`results/fixes/`. Their old queue/renderer assumed a combined CPU/GPU campaign;
do not run them unchanged on core-only `main`.
