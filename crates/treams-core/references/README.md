# Reference tables

High-precision values that the Rust tests compare treams-rs against. mpmath
computes them at 30 to 100 digits, far beyond double precision, so a test
failure points at treams-rs and not at its reference. The tables are committed
data: CI reads them and never regenerates them.

## Tables

Two terms appear below:

- The Kambe integral is `I_n(z, eta) = ∫_eta^∞ t^n exp(-z² t²/2 + 1/(2t²)) dt`.
  The Ewald lattice sums are built from it.
- The Ewald split `eta` is the parameter that divides a lattice sum into a
  real-space part (a sum over the lattice points near the evaluation point) and
  a reciprocal part (a sum over diffraction orders). Both parts converge fast,
  and their total does not depend on `eta`.

| File | Quantity | Precision | Generator | Read by |
| --- | --- | --- | --- | --- |
| [`incgamma.txt`](incgamma.txt) | Upper incomplete gamma function `Gamma(n, z)` on a grid of degrees and arguments, on both sides of the cut | 40 digits | `incgamma` | `incgamma_matches_reference_table` |
| [`kambe.txt`](kambe.txt) | Kambe integral `I_n(z, eta)` at a real split `eta` | 40 digits; Gauss-Legendre agrees with tanh-sinh to 1e-15 | `kambe` | `intkambe_matches_quadrature` |
| [`kambe_lattice.txt`](kambe_lattice.txt) | `I_n(x, eta)` at the complex arguments the lattice sums pass to it | 70 digits; 100 digits agree to 1e-48 | `kambe-lattice` | `intkambe_at_lattice_arguments` |
| [`lattice_sums.txt`](lattice_sums.txt) | Ewald lattice sums, their real-space and reciprocal parts, and their `k` and Bloch-vector derivatives | 30 to 60 digits, by section | `lattice-sums` | `sums_match_high_precision_references` |
| [`lattice_chain.txt`](lattice_chain.txt) | 1D spherical lattice sums 2 to 5 periods off the axis and their first derivatives | 30 digits; derivatives by central differences at 55 digits | `lattice-chain` | `far_off_axis_chains_match_high_precision_references` |

The special-function unit tests in `special::integrals` read the first three
tables through their `reference()` helper. The lattice property tests read the
last two. Each file's header comments give the exact grid, recipe and row
format.

## Row formats

`#` starts a comment. Every other line has the form `key: values`:

- In `incgamma.txt`, `kambe.txt`, `kambe_lattice.txt` and `lattice_chain.txt`
  the key is a list of numbers, such as `n re(z) im(z)`, and the values are
  pairs `re im` of complex numbers.
- `lattice_sums.txt` rows have the form `key part jet tolerance: values`. The
  key names the sum: `s l m` or `c m`, the dimension, `k`, the shift, the split
  `eta`, the lattice rows and the Bloch vector. Then come the fields `part`
  (full, real or reciprocal), `jet` (0 or 1) and `tolerance`. The values are
  the reference sum and, where given, its `k` and Bloch-vector derivatives, as
  `re im` pairs.

## Generators

`scripts/generate_references.py` has one subcommand per table. With `--check`
it reads a table (default: the committed file; `--output` names another),
recomputes every row from its key and prints the worst relative error; it
writes nothing. Without `--check` it writes the table to `--output` (default:
the committed file). The cases of `lattice_sums.txt` are chosen by hand, so
`lattice-sums` takes its cases from the table, computes each by the recipe of
its section, and keeps the comments in place.

```sh
uv run python scripts/generate_references.py kambe --check
uv run python scripts/generate_references.py lattice-chain --workers 4 --output /tmp/chain.txt
```

Every generator reproduces the committed values to within one unit in the last
place of each part, the tolerance of `--check`. All rows of `incgamma.txt`,
`kambe.txt` and `lattice_chain.txt` match bit for bit. In `kambe_lattice.txt`,
4 of 1249 rows differ in parts below 1e-60 of the value. In `lattice_sums.txt`,
9 of 65 rows differ by one unit in the last place.

| Subcommand | Run time on one core |
| --- | --- |
| `incgamma` | 15 s |
| `kambe` | 30 s |
| `kambe-lattice` | 1 min |
| `lattice-sums` | 15 min |
| `lattice-chain` | 50 min |

## Which data lives here

- Values that mpmath computes live here, next to their generator in
  `scripts/generate_references.py`. The cases of `lattice_sums.txt` are chosen
  by hand, but their values are computed.
- Expected values written by hand, and failures that proptest shrank, stay
  inline next to the test that reads them, under a comment that says where they
  come from.
- Values that Lean computes from the formal models live in
  [`formal/golden/`](../../../formal/golden), with the models in
  [`formal/`](../../../formal/README.md).
