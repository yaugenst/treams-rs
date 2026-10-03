# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Validate the treams-rs sdist and a wheel rebuilt from it."""

from __future__ import annotations

from _support.release_artifacts import source_build_main

if __name__ == "__main__":
    raise SystemExit(source_build_main())
