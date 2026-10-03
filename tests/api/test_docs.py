"""Execute the examples of the documentation pages and of the docstrings."""

import doctest
import os
import re
import runpy
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

import treams_rs as tr

from _support import ROOT, catalog, jax_x64

# Generated reference pages hold signature fences, not examples; their
# docstring examples run through DOCSTRINGS below.
GENERATED = ("docs/reference/",)
DOCUMENTS = [
    path
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    if not path.relative_to(ROOT).as_posix().startswith(GENERATED)
]
# Fences inside tabs and admonitions are indented; the closing fence matches.
FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```python(?P<mode>[^\n]*)\n(?P<code>.*?)^(?P=indent)```",
    re.M | re.S,
)
# The site hook strips exactly these modes, so both read one list.
FENCE_MODES = runpy.run_path(str(ROOT / "docs/_hooks/site.py"))["FENCE_MODES"]


def _examples():
    """Runnable fences and the ``(page, index)`` of fences without a valid mode.

    Every ```python fence must say ``exec`` (optionally naming a framework to
    import first) or ``no-exec`` (fragments and upstream reproducers). Pages are
    named by their repository path, so equal file names in different folders
    give distinct ids.
    """
    examples, unmarked = [], []
    for path in DOCUMENTS:
        page = path.relative_to(ROOT).as_posix()
        for index, match in enumerate(FENCE.finditer(path.read_text())):
            words = tuple(match["mode"].split())
            if words not in FENCE_MODES:
                unmarked.append((page, index))
            elif words[0] == "exec":
                code = textwrap.dedent(match["code"])
                examples.append(
                    pytest.param(path, code, words[1:], id=f"{page}:{index}")
                )
    return examples, unmarked


EXAMPLES, UNMARKED = _examples()


@pytest.fixture
def restore_jax_precision():
    """Documentation may enable JAX double precision globally; undo it."""
    with jax_x64(enabled=None):
        yield


@pytest.mark.workflows
@pytest.mark.parametrize("path,code,requires", EXAMPLES)
def test_executable_documentation(request, path, code, requires):
    for module in requires:
        pytest.importorskip(module)
        if module == "jax":
            request.getfixturevalue("restore_jax_precision")
    exec(compile(code, str(path), "exec"), {"__name__": "__example__"})


def _docstring_blocks(doc):
    return [
        textwrap.dedent(block)
        for block in re.findall(r"::\n\n((?:    [^\n]*\n|\n)+)", doc + "\n")
    ]


DOCSTRINGS = {
    "treams_rs": tr.__doc__,
    **{path: doc for path, doc in catalog()["modules"].items() if "::\n\n" in doc},
    **{row["path"]: row["doc"] for row in catalog()["api"] if "::\n\n" in row["doc"]},
}


@pytest.mark.workflows
@pytest.mark.parametrize("path", DOCSTRINGS)
def test_docstring_examples_run(request, path):
    # The adapter quickstarts need their optional framework.
    framework = {"treams_rs.jax": "jax", "treams_rs.torch": "torch"}.get(path)
    if framework is not None:
        pytest.importorskip(framework)
        if framework == "jax":
            request.getfixturevalue("restore_jax_precision")
    blocks = _docstring_blocks(DOCSTRINGS[path])
    assert blocks, f"{path}: a '::' example block must follow a blank line"
    namespace = {}
    for block in blocks:
        exec(compile(block, path, "exec"), namespace)
    # The module quickstarts end with a derivative of their objective.
    result = {
        "treams_rs": "derivative",
        "treams_rs.advect": "gradient",
        "treams_rs.jax": "gradient",
        "treams_rs.torch": "gradient",
    }.get(path)
    if result is not None:
        value = np.asarray(namespace[result])
        assert np.all(np.isfinite(value))
        assert np.any(value != 0)


DOCTESTS = {row["doc"]: row["path"] for row in catalog()["api"] if ">>>" in row["doc"]}


@pytest.mark.workflows
@pytest.mark.parametrize("doc", DOCTESTS, ids=DOCTESTS.values())
def test_docstring_doctests_pass(doc):
    test = doctest.DocTestParser().get_doctest(doc, {}, DOCTESTS[doc], None, 0)
    runner = doctest.DocTestRunner()
    runner.run(test)
    assert runner.summarize(verbose=False).failed == 0


@pytest.mark.interface
def test_documentation_has_examples_and_fences_are_marked():
    assert not UNMARKED, (
        f"fences {UNMARKED} must be `python exec [jax|torch]` or `python no-exec`"
    )
    # Snippet includes belong in no-exec fences; the included scripts run in
    # their own tests.
    included = [param.id for param in EXAMPLES if "--8<--" in param.values[1]]
    assert not included, f"exec fences {included} include snippets"
    assert EXAMPLES
    assert DOCSTRINGS
    assert DOCTESTS


SITE = "https://yaugenst.github.io/treams-rs/"
# Site pages that repository files link to before the page exists. The test
# fails once such a page exists, so each entry leaves with its page.
UNBUILT_PAGES = frozenset[str]()
# Tool, build and dependency trees, immutable evidence and generated pages.
UNLINKED = {
    ".venv",
    "target",
    "dist",
    ".git",
    "formal/.lake",
    "benchmarks/results",
    "docs/reference",
}
FENCED_CODE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$", re.M | re.S)
LINK = re.compile(
    r"\]\(<?(?P<inline>[^)\s>]+)>?(?:\s+\"[^\"]*\")?\)"
    r"|^[ \t]*\[[^\]]+\]:[ \t]+<?(?P<reference>[^\s>]+)",
    re.M,
)


def _markdown_files():
    for directory, subdirectories, names in os.walk(ROOT):
        relative = Path(directory).relative_to(ROOT)
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if (relative / name).as_posix() not in UNLINKED
        )
        yield from (
            Path(directory) / name for name in sorted(names) if name.endswith(".md")
        )


def _site_page_exists(page):
    docs = ROOT / "docs"
    return (docs / f"{page}.md").is_file() or (docs / page / "index.md").is_file()


def _link_resolves(path, target):
    if target.startswith(SITE):
        page = target.removeprefix(SITE).partition("#")[0].strip("/")
        page = re.sub(r"^(?:latest|dev|\d+\.\d+\.\d+)(?:/|$)", "", page)
        # The rustdoc pages are built by cargo, not from docs/.
        if page == "rust" or page.startswith("rust/") or page in UNBUILT_PAGES:
            return True
        return _site_page_exists(page or "index")
    if re.match(r"[a-z][a-z0-9+.-]*:|#", target):
        return True
    return (path.parent / target.partition("#")[0]).exists()


@pytest.mark.interface
def test_repository_links_resolve():
    """Relative links name existing files and site links name existing pages."""
    broken = [
        (path.relative_to(ROOT).as_posix(), target)
        for path in _markdown_files()
        for match in LINK.finditer(FENCED_CODE.sub("", path.read_text()))
        if not _link_resolves(path, target := match["inline"] or match["reference"])
    ]
    assert not broken, f"links to missing files or pages: {broken}"
    built = sorted(page for page in UNBUILT_PAGES if _site_page_exists(page))
    assert not built, f"remove {built} from UNBUILT_PAGES"


# Directories whose comments and docstrings cite documentation pages.
SOURCE_DIRECTORIES = ("python", "tests", "scripts", "crates", "formal")
SOURCE_SUFFIXES = (".py", ".pyi", ".rs", ".lean", ".md", ".toml")
# Build output, caches and fixtures that quote recorded evidence verbatim.
SOURCE_EXCLUDED = {"target", "__pycache__", ".lake", "fixtures"}
DOC_PATH = re.compile(r"(?<![\w/.-])docs/[A-Za-z0-9_./-]+\.md\b")
SITE_URL = re.compile(re.escape(SITE) + r"[A-Za-z0-9_./#-]*")


def _source_files():
    for top in SOURCE_DIRECTORIES:
        for directory, subdirectories, names in os.walk(ROOT / top):
            subdirectories[:] = sorted(
                name for name in subdirectories if name not in SOURCE_EXCLUDED
            )
            yield from (
                Path(directory) / name
                for name in sorted(names)
                if name.endswith(SOURCE_SUFFIXES)
            )


@pytest.mark.interface
def test_documentation_paths_in_sources_exist():
    """Comments and docstrings cite repository pages and site pages that exist."""
    this = Path(__file__).resolve()
    missing = []
    for path in _source_files():
        if path.resolve() == this:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        page = path.relative_to(ROOT).as_posix()
        missing += [
            (page, match[0])
            for match in DOC_PATH.finditer(text)
            if not (ROOT / match[0]).is_file()
        ]
        missing += [
            (page, match[0])
            for match in SITE_URL.finditer(text)
            if not _link_resolves(path, match[0].rstrip("."))
        ]
    assert not missing, f"documentation paths that do not exist: {missing}"


def _front_matter(path):
    text = path.read_text()
    if not text.startswith("---\n"):
        return {}
    return yaml.safe_load(text.split("---\n", 2)[1]) or {}


@pytest.mark.interface
def test_page_descriptions_are_short():
    """llms.txt and search results show each page description on one line."""
    long = {
        path.relative_to(ROOT).as_posix(): len(description)
        for path in sorted((ROOT / "docs").rglob("*.md"))
        if len(description := _front_matter(path).get("description", "")) > 140
    }
    assert not long, f"descriptions longer than 140 characters: {long}"


# The definition of records and gradients, quoted verbatim wherever the
# vocabulary is defined, so that no restatement drifts from it.
RECORD_DEFINITION = (
    "A record is a function that returns a value and a context. The context"
    " stores what is needed to compute gradients later and can be used once:"
    " `context.pullback(g)` takes the gradient `g` of a real-valued loss with"
    " respect to the value and returns the gradients with respect to the"
    " inputs, one for each differentiable input, in the order of the"
    " arguments. Gradients follow the convention dL = Re Σ conj(g)·dx."
)
RUST_DEFINITION = (
    "In the Rust core, a function returns `(value, XResidual)`, and"
    " `XResidual::pullback(self, cotangent)` returns the input gradients as"
    " `XGradient`."
)


def _normalized(text):
    """``text`` with the ``//!`` prefixes removed and whitespace collapsed."""
    return " ".join(re.sub(r"^[ \t]*//[!/]", "", text, flags=re.M).split())


@pytest.mark.interface
@pytest.mark.parametrize(
    "source,rust",
    [
        pytest.param("docs/reference/glossary.md", True, id="glossary"),
        pytest.param("docs/differentiation/index.md", False, id="differentiation"),
        pytest.param("python/treams_rs/diff.py", False, id="diff.py"),
        pytest.param("crates/treams-core/src/lib.rs", True, id="treams-core"),
        pytest.param("crates/treams-py/src/lib.rs", True, id="treams-py"),
    ],
)
def test_record_definition_is_quoted_verbatim(source, rust):
    text = _normalized((ROOT / source).read_text())
    assert RECORD_DEFINITION in text
    assert not rust or RUST_DEFINITION in text
    if source.endswith("diff.py"):
        assert RECORD_DEFINITION in _normalized(tr.diff.__doc__)


@pytest.mark.interface
def test_rust_module_docs_name_their_treams_counterpart():
    """Each treams-core module doc opens with a summary line and says which treams
    namespace it mirrors ('Upstream:') or that treams has none ('treams-rs
    extension'); the site publishes these docs as the Rust reference."""
    source = ROOT / "crates/treams-core/src"
    missing = []
    for path in sorted(source.rglob("*.rs")):
        module = path.relative_to(source)
        if module.parts[0] == "properties" or module.name == "test_support.rs":
            continue
        lines = path.read_text().splitlines()
        doc = []
        for line in lines:
            if line.startswith("//!"):
                doc.append(line[3:].strip())
            elif doc:
                break
        text = " ".join(doc)
        if (
            not doc
            or not doc[0]
            or not re.search(r"Upstream:|treams-rs extension", text)
        ):
            missing.append(module.as_posix())
    assert not missing, f"module docs without a summary and counterpart: {missing}"
