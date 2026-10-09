# Reference

- [Python API](python/index.md): public functions, classes and native return types.
- [Rust API](rust.md): the numerical core, including private implementation details.
- [Glossary](glossary.md): shared names across Python, Rust and treams.

## How the reference is built

`just docs` writes the [Python API reference](python/index.md) from
`treams_rs.support_catalog()`, which reads the signatures and docstrings of the
installed package. Each public module gets one page: signatures appear as Python
code, and docstring sections such as `Args:` and `Returns:` become labelled lists.
`just docs` also fills the generated tables of other pages, such as the
[treams name map](../coming-from-treams/names.md), and writes `llms.txt` from
the site navigation in `mkdocs.yml` and the `description` line at the top of
each page. `just docs-check` fails when a generated file differs from its
source, so the generated files always match the code. The same reference
prints offline with `python -m treams_rs --format markdown`.
