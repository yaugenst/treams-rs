# Documentation

The site is built with [MkDocs](https://www.mkdocs.org/) and the Material
theme from the Markdown files under `docs/`. The `nav` of
[`mkdocs.yml`](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/mkdocs.yml) is the one list of pages: it sets their order
on the site and in `llms.txt`.

## Build and preview

```sh
just docs-serve   # preview with live reload at http://127.0.0.1:8000
just docs-build   # strict build into site/
just docs-rust    # add the treams-core rustdoc under site/rust/
```

`just docs-build` and `just docs-serve` use the `docs` dependency group in an
isolated environment, so they leave `.venv` untouched. The strict build fails
on a warning, such as a broken link or a page missing from the nav. They need
no extension build: the site holds only committed Markdown.

The site uses the Material theme with its own colors, fonts and logo:
`theme` in `mkdocs.yml`, `docs/stylesheets/extra.css` and `docs/assets/`.
Sections are collapsed until opened; the active section stays expanded.
Python and Rust API references have top-level entries. Keep `navigation.sections`
disabled so the full page inventory does not fill the sidebar. The small
`docs/javascripts/navigation.js` gives the theme's disclosure controls accessible
names and state, and adds Space-key activation alongside Enter.
The small `docs/javascripts/history-loader.js` loads the history stylesheet
and interactive script only when their content is present, including after
instant navigation. The navigation and asset lifecycle checks run with
`node --test tests/browser/*.test.mjs`.

`just docs-rust` documents `treams-core` with its private items, which makes
the rustdoc the reference for the numerical code. The site serves it at
`rust/treams_core/` within each documentation version.

## Adding a page

1. Write `docs/<section>/<name>.md`. It starts with front matter that holds a
   `description` of at most 140 characters:

    ```markdown
    ---
    description: Solve a finite cluster once and scatter many illuminations.
    ---
    ```

2. Add the page to the `nav` of `mkdocs.yml`.
3. Run `just docs` to add it to `llms.txt`.

`just docs-check` fails when a page is missing from the nav, a nav entry has no
file, or a page has no description. `tests/api/test_docs.py` fails when a
description is longer than 140 characters, or when a comment or docstring cites
a `docs/` path or a site page that does not exist.

## Links to the repository

The site holds only the `docs/` tree. The hook `docs/_hooks/site.py`
turns a relative link that leaves `docs/`, such as
`[conftest](../../tests/conftest.py)`, into a GitHub link at the commit that
the site was built from (`TREAMS_RS_DOCS_SOURCE_REF`, or `main` locally). A link to a missing file
stops the build with the page and the target named. Write repository links as
relative paths, so they also work in the GitHub file view.

The published site keeps links to its own pages in the displayed version,
including rustdoc. Every version also publishes `llms.txt` and the Markdown
sources beside the HTML; links in that index name the same version's sources.

## Code examples

Every ```` ```python ```` fence in `README.md` and `docs/` names a mode after
the language, and `tests/api/test_docs.py` enforces it:

| Fence | Meaning |
|---|---|
| ```` ```python exec ```` | runs as a test |
| ```` ```python exec jax ```` | runs as a test, skipped without JAX |
| ```` ```python exec torch ```` | runs as a test, skipped without PyTorch |
| ```` ```python exec autograd ```` | runs as a test, skipped without HIPS Autograd |
| ```` ```python no-exec ```` | does not run: fragments and treams code |

`FENCE_MODES` in the hook is the one list of modes; the hook removes the mode
words before MkDocs renders the page. The tests also run docstring examples
after `::` and `>>>` doctests of the API catalog. Keep each example
self-contained, with valid physical inputs and an `assert` that checks the
claimed result, and run `uv run --no-sync pytest tests/api/test_docs.py` after
changing one.

## Generated content

`just docs` writes three kinds of files from
[`scripts/generate_docs.py`](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/scripts/generate_docs.py):

- **The Python API reference** under `docs/reference/python/`: one page per
  public module and a page of returned native types, rendered from
  `treams_rs.support_catalog()`.
- **Generated regions** of hand-written pages, between
  `<!-- generated: NAME -->` and `<!-- end generated -->`: `upstream-names`
  (the treams name map, from `_upstream.py`) and `rust-crosswalk` (the module
  table of the `treams-core` crate docs).
- **`llms.txt`**: every page of the nav with its description, for tools that
  read the documentation as text.

`just docs-check` fails when one of these files differs from its source, and
the Python tests run the same check. Never edit a generated file; change the
docstring, the stub or the source table and run `just docs`.

The reference comes from the API catalog rather than from a MkDocs plugin such
as mkdocstrings. One source then serves the site, `llms.txt` and
`python -m treams_rs --format markdown`. The catalog reads the installed
package, so it includes native ufuncs, whose signatures exist only at run time,
and the adapter modules without their optional frameworks installed.

`docs/history/` is written by a private history build from the archived
conversation records. Do not edit it by hand; it has a typos exclusion for its
data and keeps every file under the 1,000 KB limit.

## Examples gallery

The gallery in `docs/examples/` repeats the examples of the treams
documentation and adds two gradient examples. An example `<name>` has these
files:

| File | Content |
| --- | --- |
| `docs/examples/<name>.py` | The treams-rs script. It prints a table and draws no figures. |
| `docs/examples/output/<name>.txt` | What the script prints. |
| `docs/examples/upstream/<treams name>.py` | For a ported example only: the treams example with the same sizes, its source and its license. |
| `docs/examples/<name>.md` | The page: the scripts in tabs, the output and the differences from treams. |

[`tests/api/test_examples.py`](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/tests/api/test_examples.py) runs every
script and compares the numbers it prints with `output/<name>.txt` to a
relative tolerance of 1e-6. With treams installed, it also runs each treams
script and compares the variables that its `ORACLE` entry names, to a relative
tolerance of 1e-10 unless the entry gives a looser one with a reason. A script
whose first line reads `# Requires: jax` skips without JAX.

To add an example:

1. Write the treams-rs script, with sizes small enough to run in under a
   second.
2. For a port of a treams example, write the treams script and add an `ORACLE`
   entry, keyed by the treams name, to `tests/api/test_examples.py`.
3. Run `just docs-examples` to write `output/<name>.txt`.
4. Write the page from an existing one: include the scripts and the output
   with `--8<--` lines and list the differences from treams.
5. Add the page to the table in `docs/examples/index.md` and to the Examples
   section of `mkdocs.yml`.

`test_gallery_is_complete` fails when a piece is missing, and when a page shows
a treams tab without a treams script or the other way round. Run
`just docs-examples` again whenever a script changes what it prints, and
review the change in `output/`.

## Publishing

The reusable [Docs workflow](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/.github/workflows/docs.yml) builds the site
and rustdoc for pull requests. Main publishes `dev`; a release publishes its
version and updates `latest`. Mike keeps every version in `gh-pages`, and
the workflow deploys the complete tree through GitHub Pages artifacts.
[Releasing](releasing.md#documentation-site) describes setup and corrections
to released documentation.
