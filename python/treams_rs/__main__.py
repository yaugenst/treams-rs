"""Print the installed API catalog offline: python -m treams_rs [--format markdown]."""

import argparse
import json

from .support import _markdown, support_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    catalog = support_catalog()
    print(
        _markdown(catalog)
        if args.format == "markdown"
        else json.dumps(catalog, indent=2)
    )


if __name__ == "__main__":
    main()
