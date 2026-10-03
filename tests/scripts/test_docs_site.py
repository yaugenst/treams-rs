# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check versioned links and the published Markdown without a native build."""

import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HOOK = runpy.run_path(str(Path(__file__).resolve().parents[2] / "docs/_hooks/site.py"))


class DocumentationSiteTest(unittest.TestCase):
    def test_versioned_sources_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs/guide").mkdir(parents=True)
            (root / "site").mkdir()
            (root / "Cargo.toml").write_text("# source file\n")
            (root / "llms.txt").write_text(
                "The Markdown sources linked below are in this repository, at the same Git revision as this file.\n"
                "- [Guide](docs/guide/example.md)\n"
            )
            config = {
                "docs_dir": root / "docs",
                "site_dir": root / "site",
                "config_file_path": root / "mkdocs.yml",
            }
            page = SimpleNamespace(
                file=SimpleNamespace(src_uri="guide/example.md", url="guide/example/")
            )
            with patch.dict(
                "os.environ",
                {"TREAMS_RS_DOCS_SOURCE_REF": "a" * 40, "GITHUB_SHA": "b" * 40},
            ):
                HOOK["on_pre_build"](config)
                markdown = HOOK["on_page_markdown"](
                    "[Source](../../Cargo.toml)\n"
                    "[Rust](https://yaugenst.github.io/treams-rs/latest/rust/treams_core/index.html#modules)\n"
                    "`[Literal](https://yaugenst.github.io/treams-rs/latest/)`\n"
                    "```python exec\nassert True\n```\n",
                    page,
                    config,
                    None,
                )
                self.assertIn(f"/blob/{'a' * 40}/Cargo.toml", markdown)
                self.assertNotIn("python exec", markdown)
                html = HOOK["on_post_page"](
                    '<a href="https://yaugenst.github.io/treams-rs/latest/reference/glossary/#record">Glossary</a>'
                    '<a href="https://yaugenst.github.io/treams-rs/latest/rust/treams_core/">Rust</a>'
                    f'<a href="https://github.com/yaugenst/treams-rs/blob/{"a" * 40}/llms.txt">Index</a>',
                    page,
                    config,
                )
                self.assertIn('href="../../reference/glossary/#record"', html)
                self.assertIn('href="../../rust/treams_core/"', html)
                self.assertIn('href="../../llms.txt"', html)
                HOOK["on_post_build"](config)
                published = (root / "site/guide/example.md").read_text()
                self.assertEqual(
                    published,
                    markdown.replace(
                        "](https://yaugenst.github.io/treams-rs/latest/rust/",
                        "](../rust/",
                    ),
                )
                index = (root / "site/llms.txt").read_text()
                self.assertIn("this documentation version", index)
                self.assertIn("](guide/example.md)", index)


if __name__ == "__main__":
    unittest.main()
