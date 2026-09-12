"""Execute marked public documentation examples and check the agent index links."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    (path, index, code)
    for path in sorted((ROOT / "docs").glob("*.md"))
    for index, code in enumerate(
        re.findall(
            r"^```python exec\n(.*?)^```", path.read_text(), re.MULTILINE | re.DOTALL
        )
    )
]


@pytest.mark.parametrize(
    "path,index,code", EXAMPLES, ids=[f"{p.name}:{i}" for p, i, _ in EXAMPLES]
)
def test_executable_documentation(path, index, code):
    exec(compile(code, f"{path}:{index}", "exec"), {"__name__": "__example__"})


def test_agent_index_is_local_and_complete():
    assert EXAMPLES
    text = (ROOT / "llms.txt").read_text()
    links = re.findall(r"\]\(([^)]+)\)", text)
    assert "docs/api.md" in links
    assert "docs/agents.md" in links
    assert "docs/testing.md" in links
    for link in links:
        assert (ROOT / link).is_file(), link
