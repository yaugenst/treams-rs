"""Offline installed help: quickstart, an API path, --search text, or --format json."""

import argparse
import json
from typing import Any

from . import __doc__ as quickstart
from .support import _markdown, support_catalog


def _entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for row in rows for entry in (row, *_entries(row.get("members", [])))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "topic", nargs="?", help="API path, e.g. sphere_tmatrix or advect.Cluster"
    )
    parser.add_argument(
        "--search", metavar="TEXT", help="Find public names and one-line descriptions"
    )
    parser.add_argument(
        "--format", choices=("json", "markdown"), help="Export the complete catalog"
    )
    args = parser.parse_args()
    if not (args.topic or args.search or args.format):
        print(quickstart)
        return
    if sum(bool(value) for value in (args.topic, args.search, args.format)) != 1:
        parser.error("choose one topic, --search, or --format")
    catalog = support_catalog()
    if args.format:
        print(
            _markdown(catalog)
            if args.format == "markdown"
            else json.dumps(catalog, indent=2)
        )
        return
    entries = _entries(catalog["api"])
    if args.search:
        terms = [
            term.strip().casefold()
            for term in args.search.replace(r"\|", "|").split("|")
            if term.strip()
        ]
        matches = [
            row
            for row in entries
            if any(term in row["path"].casefold() for term in terms)
        ]
        for row in matches:
            print(row["path"] + " — " + (row["doc"].splitlines() or [""])[0])
        if not matches:
            parser.error(f"no names match {args.search!r}")
        return
    path = (
        args.topic if args.topic.startswith("treams_rs.") else "treams_rs." + args.topic
    )
    if path in catalog["modules"]:
        print(catalog["modules"][path])
        print("\nPublic API (request a name for its full contract):")
        for row in catalog["api"]:
            if row["path"].rsplit(".", 1)[0] == path:
                print(row["path"] + row.get("signature", ""))
        return
    entry = next((row for row in entries if row["path"] == path), None)
    if entry is None:
        parser.error(f"unknown API path {args.topic!r}; use --search to find a name")
    print(entry["path"] + entry.get("signature", "") + "\n\n" + entry["doc"])
    for member in entry.get("members", []):
        print("\n" + member["path"] + member.get("signature", ""))
        print(member["doc"])
    for pullback in entry.get("pullbacks", []):
        print(
            "\n"
            + pullback["context"]
            + "."
            + pullback["method"]
            + pullback["signature"]
        )
        print(pullback["doc"])


if __name__ == "__main__":
    main()
