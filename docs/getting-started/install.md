---
description: Install treams-rs from GitHub with the Rust toolchain, choose extras for gradients and HDF5, and set the thread count.
---

# Install

## Requirements

- Python 3.12 or 3.13.
- The Rust toolchain through [rustup](https://rustup.rs). The repository's
  `rust-toolchain.toml` selects Rust 1.94, and rustup downloads it on the first
  build.

treams-rs is tested on Linux x86-64 and macOS arm64. Other platforms are
untested.

## Install from GitHub

treams-rs is not on PyPI yet. pip builds it from the repository and compiles
the Rust core in release mode, which takes a few minutes:

```sh
pip install "treams-rs @ git+https://github.com/yaugenst/treams-rs"
```

The package needs only NumPy at run time. Extras add optional packages:

| Extra | Adds | For |
| --- | --- | --- |
| `advect` | Advect | `treams_rs.advect` |
| `jax` | JAX | `treams_rs.jax` |
| `torch` | PyTorch | `treams_rs.torch` |
| `io` | h5py | HDF5 files in `treams_rs.io` |

Name the extras in brackets:

```sh
pip install "treams-rs[jax,io] @ git+https://github.com/yaugenst/treams-rs"
```

Check the installation with the offline help, which prints a short example
and the main entry points:

```sh
python -m treams_rs
```

## Threads

The Rust core runs in parallel on its own pool of threads and uses every CPU the
process may use by default. `tr.set_num_threads(1)` or `with tr.threads(1):`
fixes the count at any time, for example for timings on one core; before
Python imports `treams_rs`, `TREAMS_RS_NUM_THREADS` sets the default. Results
do not depend on the count. [Threads and process pools](../guide/threads.md)
covers the other variables, `multiprocessing` and threadpoolctl.

## From a checkout

To change treams-rs or run its tests, set up a development checkout as
described in [Development](../development/index.md). That setup also installs
treams, SciPy and the test tools, which the package itself does not need.
