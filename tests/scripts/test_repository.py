"""Repository configuration: YAML files parse, and the Formal workflow watches every
Rust file that a Lean model describes."""

import re

import pytest
import yaml

from _support import ROOT

pytestmark = pytest.mark.interface

FORMAL_WORKFLOW = ROOT / ".github" / "workflows" / "formal.yml"
FORMAL_PAGE = ROOT / "docs" / "design" / "formal-proofs.md"
MODEL_TABLE = "## Rust items and their models"
RUST_SOURCE = re.compile(r"crates/treams-core/src/[\w/]+\.rs")


def _formal_paths(trigger: str) -> list[str]:
    workflow = yaml.safe_load(FORMAL_WORKFLOW.read_text(encoding="utf-8"))
    # YAML 1.1 reads the bare key `on` as the boolean true.
    return workflow[True][trigger]["paths"]


def _watched_rust_files() -> set[str]:
    return {
        path for path in _formal_paths("pull_request") if path.startswith("crates/")
    }


def _modelled_rust_files() -> set[str]:
    """The Rust files linked from the Rust-file column of the model table."""
    text = FORMAL_PAGE.read_text(encoding="utf-8")
    section = text.split(MODEL_TABLE, 1)[1].split("\n## ", 1)[0]
    files = set()
    for row in section.splitlines():
        cells = row.split(" | ")
        if row.startswith("| `") and len(cells) == 5:
            files.update(RUST_SOURCE.findall(cells[1]))
    return files


def test_formal_workflow_watches_the_same_paths_on_pull_request_and_push():
    paths = _formal_paths("pull_request")
    assert paths == _formal_paths("push")
    assert {"formal/**", "justfile", ".github/workflows/formal.yml"} <= set(paths)
    assert len(paths) == len(set(paths))


def test_formal_workflow_paths_name_existing_rust_files():
    for path in _watched_rust_files():
        assert (ROOT / path).is_file(), path


def test_formal_workflow_watches_exactly_the_modelled_rust_files():
    modelled = _modelled_rust_files()
    assert len(modelled) >= 6, "the model table in formal-proofs.md was not found"
    assert modelled == _watched_rust_files()


def test_lean_comments_cite_watched_rust_files():
    watched = _watched_rust_files()
    for lean in sorted((ROOT / "formal").rglob("*.lean")):
        if ".lake" in lean.parts:
            continue
        for path in RUST_SOURCE.findall(lean.read_text(encoding="utf-8")):
            assert path in watched, f"{lean.relative_to(ROOT)} cites {path}"


YAML_FILES = sorted(
    [
        ROOT / "mkdocs.yml",
        ROOT / "CITATION.cff",
        *(ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"),
        *(ROOT / ".github" / "workflows").glob("*.yml"),
    ]
)


@pytest.mark.parametrize(
    "path", YAML_FILES, ids=[str(p.relative_to(ROOT)) for p in YAML_FILES]
)
def test_yaml_file_parses_to_a_mapping(path):
    assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
