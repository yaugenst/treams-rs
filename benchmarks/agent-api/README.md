# Wheel-only agent evaluation

`protocol.json` freezes, before any scored attempt, the incremental comparison
against baseline `85986f1`, which already contains the initial physical API
redesign. `original-api-comparison.json` freezes the comparison against the
original API (`2843a70`), whose numerical and Python source matches qualified
main (`0137eca`); its gains measure the whole redesign, not only the help and
namespace changes after `85986f1`. Both comparisons use the same prompts and
references. `tasks.json` contains ten development tasks and three transfer
tasks; only the corresponding `*-prompts.json` is sent to each fresh agent.
`reference.json` and the grader remain outside the tool sandbox. Grader
amendments preserve previous source and explain corrections applied identically
to every cohort.

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

Four agent configurations receive identical prompts at medium reasoning effort;
the evidence directories label them luna, sol, sonnet and opus, and
`protocol.json` maps the labels to model IDs. Every attempt has a 15-minute limit
and a fresh workspace, and repeated attempts of each task are retained. Outcomes
are graded independently of the agents. Transfer tasks run only after the
candidate API is frozen.

The launchers mask global instructions, skills, plugins and history. Tool
processes can read only the installed wheel and declared dependencies, system
runtimes, that attempt's workspace and, under Codex, the Codex CLI installation
and any paths granted with `TREAMS_EVAL_READ_PATHS` (see
[Running an evaluation](#running-an-evaluation)); package directories are
read-only and tool network access is blocked. Model transport and
authentication run outside that sandbox. No evaluation agent decides grading,
acceptance or API changes.

## Running an evaluation

Requirements:

- Linux with bubblewrap at `/usr/bin/bwrap`; the Claude launcher and the tool
  sandbox that the graders and `preflight.py` share call it by that path.
- For Claude models, Python 3.11 or newer at `/usr/bin/python3`, which runs the
  Claude launcher and its tool prefix.
- uv, and the Codex and Claude Code CLIs signed in for the models under test.
  `run.py` and `preflight.py` need the Codex CLI even when only Claude models
  are evaluated.
- Network access for model transport only.

The scripts read these environment variables:

| Variable | Read by | Meaning |
| --- | --- | --- |
| `TREAMS_EVAL_VENV` | every launcher and grader | Evaluation environment created by `install.py` (required). |
| `TREAMS_EVAL_CODEX` | `runner.py` | Codex executable; default: `codex` on `PATH`. The directory two levels above the resolved executable (its installation) is readable inside the Codex sandbox, so it must not contain `CODEX_HOME`, `~/.claude` or other private data; `preflight.py` fails when it exposes a forbidden path. |
| `TREAMS_EVAL_CLAUDE` | `claude_runner.py` | Claude Code executable; default: `claude` on `PATH`. |
| `CODEX_HOME` | `runner.py`, `preflight.py` | Codex state directory; default: `~/.codex`. |
| `TREAMS_EVAL_READ_PATHS` | `runner.py` | Further read-only paths for the Codex sandbox, separated by `:`. |
| `TREAMS_EVAL_FORBIDDEN` | `preflight.py` | Further paths that sandboxed tools must not be able to read, separated by `:`. |
| `TREAMS_EVAL_CONTEXT_MARKERS` | `preflight.py` | Further strings that must not appear in the Codex model context, such as a phrase unique to your global instructions, separated by `:` (so a marker cannot contain `:`; any distinctive substring works). |

Both sandboxes can read the evaluation environment and its base Python
installation; the Codex sandbox can also read the installation of the Codex CLI
and `TREAMS_EVAL_READ_PATHS`. `preflight.py` checks the sandboxes without model
calls: tools must not be able to read the Codex credentials, instructions and
memories in `CODEX_HOME`, the Claude Code credentials in `~/.claude` or the
further forbidden paths. It writes its report to the untracked
`benchmarks/results/local/agent-api/isolation/`. `run.py` creates each attempt
workspace under `/tmp/treams-api-agent-trials/<cohort>/`, where `<cohort>` is
the name of its output directory, and never reuses one, so every cohort on a
host needs a new name.

Run from the repository root, using the development environment only for the
trusted controller and grader. `EVAL_DIR` is a new directory outside the
repository:

```sh
just build-wheel
uv run --script scripts/agent_eval/install.py --wheel dist/<wheel>.whl \
  --env "$EVAL_DIR/venv" --output "$EVAL_DIR/installation"
export TREAMS_EVAL_VENV="$EVAL_DIR/venv"
uv run --script scripts/agent_eval/preflight.py
COHORT="$EVAL_DIR/main-$(date -u +%Y%m%dT%H%M%SZ)"
uv run --script scripts/agent_eval/run.py \
  --suite benchmarks/agent-api/main-prompts.json --output "$COHORT"
uv run --no-sync python scripts/agent_eval/grade.py \
  --cohort "$COHORT" --reference benchmarks/agent-api/reference.json
```

Use unique output directories; no attempt is overwritten or selectively retried.
`watch.py` scores completed attempts while the controller runs. `summarize.py`
exports submitted code, reports, replay results, tool actions and token usage;
private model reasoning and injected system prompts remain excluded. Provider
capacity refusals are retained and reported separately from API failures.
Token totals come from completed provider usage records; missing totals remain
missing rather than being reported as zero. Effective tokens mean uncached input
plus output, with Claude cache creation counted as uncached input. Agent wall
time includes discovery, coding and checking; it is not a solver benchmark.
Exact model IDs, wheel hashes and dependency versions are retained with each
cohort.

Real-chirality and lossless power references cannot establish helicity labeling
or reverse-incidence setup by numerical equality alone, so `source-audit.json`
records source reviews bound to exact submitted source hashes;
`summarize.py --source-audit` reports their pass-to-partial overrides alongside
the automated outcomes. Attempts made while the help text described port
ordering incorrectly remain in the results.

`freeze_polarization.py` freezes two supplementary, nondegenerate cases
(`polarization-protocol.json`) before new model calls: complex chirality
distinguishes helicities, and asymmetric lossy layers distinguish incident-side
reflection. Existing prompts and numerical references are unchanged. Runs of the
original transfer cases with the corrected guide are confirmation tasks, not
unseen tasks. [The campaign report](../results/agent-api-20260920/REPORT.md) has
the final results and the acceptance-gate decisions.

[`review-verification.json`](../results/agent-api-20260920/review-verification.json),
summarized in the report, records a compatibility replay of the saved final
solutions on the reviewed wheel, without fresh model calls or agent timing; the
campaign scores, source audits and wheel hashes stay as recorded.
`review-evidence.tar.gz` holds the replay command and check logs, and
`evidence.tar.gz` the original submitted solutions.
