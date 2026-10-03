# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print exact pins at the declared floors of the treams-rs runtime requirements.

Reads ``pyproject.toml`` and writes one ``name==floor`` requirement per line for
the base dependencies and the requested extras, keeping environment markers.
Installing the output together checks that the declared lower bounds are
simultaneously installable and sufficient. A requirement without a ``>=`` floor
is rejected so every supported minimum stays explicit.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from typing import cast

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_REQUIREMENT = re.compile(
    r"\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?"
    r"\s*(?P<specifiers>[^;]*?)\s*(?:;\s*(?P<marker>.+?))?\s*"
)


def minimum_pins(extras: list[str], pyproject: Path = _PYPROJECT) -> list[str]:
    """Return exact floor pins for the base requirements and the given extras."""
    project = cast(
        "dict[str, object]",
        tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"],
    )
    requirements = list(cast("list[str]", project["dependencies"]))
    optional = cast("dict[str, list[str]]", project.get("optional-dependencies", {}))
    for extra in extras:
        if extra not in optional:
            message = (
                f"unknown extra {extra!r}; declared: {', '.join(sorted(optional))}"
            )
            raise ValueError(message)
        requirements.extend(optional[extra])

    pins: list[str] = []
    for requirement in requirements:
        match = _REQUIREMENT.fullmatch(requirement)
        if match is None:
            message = f"cannot parse requirement {requirement!r}"
            raise ValueError(message)
        floors = [
            specifier.strip().removeprefix(">=").strip()
            for specifier in match["specifiers"].split(",")
            if specifier.strip().startswith(">=")
        ]
        if len(floors) != 1:
            message = f"requirement {requirement!r} must declare exactly one >= floor"
            raise ValueError(message)
        marker = f"; {match['marker']}" if match["marker"] else ""
        pins.append(f"{match['name']}=={floors[0]}{marker}")
    return pins


def main(argv: list[str] | None = None) -> int:
    """Print the floor pins for the requested extras, one per line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extras", nargs="*", help="Optional dependency extras.")
    extras = cast("list[str]", parser.parse_args(argv).extras)
    print("\n".join(minimum_pins(extras)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
