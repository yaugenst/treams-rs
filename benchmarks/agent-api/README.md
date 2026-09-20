# Wheel-only agent evaluation

`protocol.json` freezes the incremental comparison before scored attempts.
Its baseline (`85986f1`) already contains the initial physical API redesign.
`original-api-comparison.json` separately freezes the comparison against the
branch's starting API (`2843a70`), whose numerical/Python source matches qualified
main (`0137eca`). Both comparisons retain the same prompts and references; gains
against the starting API must not be attributed solely to later help fixes. `tasks.json`
contains ten development tasks and three transfer tasks; only the corresponding
`*-prompts.json` is sent to each fresh agent. `reference.json` and the grader
remain outside the tool sandbox. Grader amendments preserve previous source and
explain corrections applied identically to every cohort.

The tasks adapt the physical workflows in Beutel, Fernandez-Corbaton and
Rockstuhl, [treams](https://doi.org/10.1016/j.cpc.2023.109076), CPC 297 (2024),
and its pinned [sphere](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/sphere.py),
[slab](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/slab.py)
and [periodic-array](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/array_spheres.py)
examples. Parameters are smaller evaluation inputs, not a reproduction of a
published figure. The sensitivity and inverse-design objectives are evaluation
extensions of those physical problems.

References use upstream treams 0.4.5 and independent Fresnel/Airy controls.
Sensitivity references use fourth-order finite differences only in the hidden
grader; solutions must execute native analytic pullbacks. Both example inputs
and changed numeric inputs are replayed in a separate wheel-only process before
the trusted grader sees the output. Optimization checks recompute the returned
design independently, including its final derivative and loss threshold.

The design follows the prior Photonoodle campaign: identical prompts and medium
effort across Luna, Sol, Sonnet and Opus, a 15-minute limit, fresh workspaces,
independent outcome grading, then transfer tasks after freezing the candidate.
That earlier eight-attempt result was 1 to 7 full passes and 82.7 to 32.5 aggregate
agent minutes; it combined implementation and discovery fixes and was not a
model ranking or a controlled ablation. This campaign uses repeated attempts.

The reusable launchers are adapted from the local Photonoodle campaign scripts.
They mask global instructions, skills, plugins and history. Tool processes can
read only the installed wheel and declared dependencies, system runtimes and
that attempt's workspace; package directories are read-only and tool network
access is blocked. Model transport/authentication runs outside that sandbox.
No evaluation agent decides grading, acceptance or API changes.

Run from the redesign worktree, using the development environment only for the
trusted controller and grader:

```sh
just build-wheel
uv run --script scripts/agent_eval/install.py --wheel /path/to/wheel.whl \
  --env /tmp/treams-eval-wheel --output /path/to/new-installation-record
export TREAMS_EVAL_VENV=/tmp/treams-eval-wheel
uv run --script scripts/agent_eval/preflight.py
uv run --script scripts/agent_eval/run.py \
  --suite benchmarks/agent-api/main-prompts.json --output /path/to/new-cohort
uv run --no-sync python scripts/agent_eval/grade.py \
  --cohort /path/to/new-cohort --reference benchmarks/agent-api/reference.json
```

Use unique output directories; no attempt is overwritten or selectively retried.
`watch.py` scores completed attempts while the controller runs. `summarize.py`
exports submitted code, reports, replay results, tool actions and token usage;
private model reasoning and injected system prompts remain excluded. Provider
capacity refusals are retained and reported separately from API failures.
Token totals come from completed provider usage records; missing totals remain
missing rather than being reported as zero. Effective tokens mean uncached input plus output, with Claude cache creation counted as
uncached input. Agent wall time includes discovery, coding and checking; it is
not a solver benchmark. Exact model IDs, wheel hashes and dependency versions
are retained with each cohort.

The source audit found that real-chirality and lossless power references cannot
establish helicity labeling or reverse-incidence setup by numerical equality
alone. `source-audit.json` records parent reviews bound to exact submitted source
hashes; pass-to-partial overrides are retained alongside automated outcomes by
`summarize.py --source-audit`. The first help revisions incorrectly described
port ordering; those affected attempts remain in the results.

`freeze_polarization.py` freezes two supplementary, nondegenerate cases before
new model calls. Complex chirality distinguishes helicities; asymmetric lossy
layers distinguish incident-side reflection. Existing prompts and numerical
references are preserved. After correcting the guide, the original transfer
cases are confirmation tasks, not unseen tasks again. See
[`the campaign report`](../results/agent-api-20260920/REPORT.md) for final results
and explicit acceptance-gate decisions.
