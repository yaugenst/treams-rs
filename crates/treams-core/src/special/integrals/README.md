# Gamma and Kambe integrals

These functions provide the incomplete gamma and Kambe integrals used by
[Ewald lattice sums](../../lattice/README.md).

[gamma.rs](gamma.rs) evaluates the upper incomplete gamma function and scaled
sequences of its orders. [kambe.rs](kambe.rs) combines recurrences and series
to evaluate Kambe integrals, including rounding bounds for difficult inputs.
[double.rs](double.rs) represents numbers as pairs of doubles to retain
precision in small-split series.

[mod.rs](mod.rs) checks public inputs and exposes scalar and array
calculations. Array results keep their arguments for analytic gradients.
Internal lattice calculations can request sequences of related orders and
error bounds, avoiding independent evaluations of each integral. The
[reference tables](../../../references/README.md) provide high-precision checks.
