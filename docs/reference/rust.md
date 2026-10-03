---
description: The rustdoc of treams-core, the internal Rust crate that holds the numerical code.
---

# Rust crate

The rustdoc of `treams-core` is at
<https://yaugenst.github.io/treams-rs/rust/treams_core/>. `treams-core` is an
internal crate: only the Python bindings in `treams-py` use it, and it is not
published to crates.io. Its documentation includes private items, so it also
serves as the reference for the numerical code. Build it locally with
`just docs-rust`, which writes it to `site/rust/`.
