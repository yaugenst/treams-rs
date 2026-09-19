# /// script
# requires-python = ">=3.12,<3.14"
# ///
"""Generate or verify the offline agent index and source-derived Python reference."""

import argparse
from pathlib import Path

from treams_rs.support import _markdown, support_catalog

ROOT = Path(__file__).resolve().parents[1]


def generated_files() -> dict[Path, str]:
    catalog = support_catalog()
    pages = {
        "docs/agents.md": "Agent quickstart and choosing an execution path",
        "docs/api.md": "Complete source-derived Python signatures, docstrings and pullback contracts",
        "docs/testing.md": "Runnable gradient and native pullback checks",
        "docs/architecture.md": "Rust/Python boundary and complex adjoint convention",
        "docs/adapters.md": "Advect, JAX and PyTorch execution contracts",
        "docs/large-problems.md": "Requested illuminations and measured memory/performance tradeoffs",
        "docs/iterative.md": "Matrix-free sphere solver and convergence",
        "docs/wasm.md": "Direct JavaScript/TypeScript WASM build and qualification",
        "docs/status.md": "Verified coverage and remaining limits",
        "docs/paper-qualification.md": "Published-problem qualification and discrepancies",
        "docs/upstream-findings.md": "Documented reference defects and numerical conventions",
        "CONTRIBUTING.md": "Contributor doorway",
        "docs/development.md": "Code ownership, local checks and documentation generation",
        "AGENTS.md": "Repository-wide instructions for automated contributors",
    }
    index = [
        "# treams-rs",
        "",
        f"> Private T-matrix scattering package, version {catalog['version']}. Rust numerics and analytic first-order pullbacks; typed Python and direct WASM interfaces.",
        "",
        "Read these Markdown files at the same Git revision. No public documentation service is required.",
        "With only an installed wheel, run `python -m treams_rs --format markdown` or",
        "`python -m treams_rs` (JSON) for signatures, docstrings and backend contracts from that installation.",
        "`treams_rs.support_catalog()` never imports optional frameworks.",
        "",
        "## Documentation",
        "",
        *(f"- [{title}]({path})" for path, title in pages.items()),
        "",
    ]
    return {
        ROOT / "docs/api.md": _markdown(catalog),
        ROOT / "llms.txt": "\n".join(index),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in generated_files().items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(f"{path.relative_to(ROOT)} is stale; run just docs")
        else:
            path.write_text(content)
    print(
        "Agent documentation is current"
        if args.check
        else "Agent documentation generated"
    )


if __name__ == "__main__":
    main()
