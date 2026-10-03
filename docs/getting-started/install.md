---
description: Install treams-rs from PyPI, choose extras for gradients and HDF5, build from source, and set the thread count.
---

# Install

## Install from PyPI

Use CPython 3.12–3.15:

```sh
pip install treams-rs
```

The 0.1.0 core wheel targets are:

| Operating system | Architectures |
| --- | --- |
| Linux, glibc 2.17 or newer | x86-64, arm64 (`aarch64`) |
| macOS | Intel (`x86_64`), Apple silicon (`arm64`) |
| Windows | x86-64 |

Installing a wheel needs no Rust toolchain. NumPy is the only required
runtime dependency.

## Optional packages

Extras add framework adapters and HDF5 interchange:

| Extra | Adds | For |
| --- | --- | --- |
| `advect` | Advect | `treams_rs.advect` |
| `jax` | JAX | `treams_rs.jax` |
| `torch` | PyTorch | `treams_rs.torch` |
| `io` | h5py | HDF5 files in `treams_rs.io` |

Name the extras in brackets:

```sh
pip install "treams-rs[jax,io]"
```

For 0.1.0, use CPython 3.12–3.14 with the `torch` or `io` extras. The
CPython 3.15 release matrix excludes their standard PyPI installs because
the release dependency set has no matching PyPI PyTorch or h5py wheels.
The Linux development matrix uses PyTorch's separate CPU wheel index on
3.15; see [development setup](../development/index.md#setup).

Check the installation with the offline help, which prints a short example
and the main entry points:

```sh
python -m treams_rs
```

## Threads

The Rust core owns its worker pool and defaults to the available CPU budget.
Set `TREAMS_RS_NUM_THREADS` before importing the package, or call
`treams_rs.set_num_threads(n)` to change the budget in the running process.
`with treams_rs.threads(n):` restores the previous budget on exit. See
[Threads and process pools](../guide/threads.md) for multiprocessing,
environment-variable precedence and interaction with other libraries.

## From a checkout

To change treams-rs or run its tests, set up a development checkout as
described in [Development](../development/index.md). That setup also installs
treams, SciPy and the test tools, which the package itself does not need.

## Build from source

A source build needs Rust 1.94.0 or newer and a C linker. Install Rust through
[rustup](https://rustup.rs), then build the published source distribution:

```sh
pip install --no-binary treams-rs treams-rs
```

To install the current repository revision instead:

```sh
pip install "treams-rs @ git+https://github.com/yaugenst/treams-rs"
```

The checkout's `rust-toolchain.toml` selects Rust 1.94.0. Source builds
compile the Rust core in release mode and take longer than wheel installs.
