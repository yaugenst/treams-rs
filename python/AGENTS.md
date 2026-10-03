# Python and adapters

Read [source ownership](../docs/development/architecture.md) and [testing](../docs/development/testing.md#python).

- Values and derivatives come from `_native`; Python adds physical meaning, argument checks and treams-style messages.
- Advect, JAX, PyTorch and h5py stay optional imports; the adapters share `_framework*.py`.
- treams names live in `_upstream.py`; treams-style members are one-line delegations at the end of each class.
- Tests go into `tests/<domain>/` with a marker; derivative tests use `treams_rs.testing.check_pullback`.
