"""MkDocs hook for the treams-rs site.

The hook rewrites every page before MkDocs renders it:

1. It removes the mode words from python fence headers. The documentation
   tests run ```` ```python exec ```` fences and skip ```` ```python no-exec ````
   fences; the site shows both as highlighted Python.
2. It turns links that leave ``docs/`` (for example
   ``../tests/api/test_docs.py``) into GitHub links, because the built site
   holds only the ``docs/`` tree. Ordinary links point at the GitHub page of
   the file or folder; image links point at the raw file, so the image shows.
   A link to a missing file stops the build with the page and the target
   named. Fenced code and inline code keep their text unchanged.

``FENCE_MODES`` is the one list of fence modes; ``tests/api/test_docs.py``
loads it from this file. The module imports nothing from MkDocs at import
time, so the tests can load it without MkDocs installed.
"""

import os
import posixpath
import re
from pathlib import Path

FENCE_MODES = (
    ("exec",),
    ("exec", "jax"),
    ("exec", "torch"),
    ("exec", "autograd"),
    ("no-exec",),
)
REPOSITORY = "https://github.com/yaugenst/treams-rs"
SITE = "https://yaugenst.github.io/treams-rs/"
_page_sources = {}

# Longest modes first, so that "exec jax" is not read as "exec".
_MODES = "|".join(
    r"[ \t]+".join(map(re.escape, words))
    for words in sorted(FENCE_MODES, key=len, reverse=True)
)
# Fences inside tabs and admonitions are indented.
_FENCE_HEADER = re.compile(
    rf"^(?P<head>[ \t]*```python)[ \t]+(?:{_MODES})[ \t]*$", re.M
)
_FENCE_LINE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})")
_PARENT_LINK = re.compile(
    r"(?P<image>!?)(?P<text>\[[^\]]*\])\((?P<target>\.\./[^)\s]*)"
)
# An inline code span opens and closes with backtick runs of equal length.
_CODE_SPAN = re.compile(r"(`+).+?(?<!`)\1(?!`)")
_SITE_LINK = re.compile(
    r'(?P<open>href=")' + re.escape(SITE) + r'latest/(?P<path>[^"]*)"'
)


_MARKDOWN_SITE_LINK = re.compile(
    r"(?P<open>\]\()" + re.escape(SITE) + r"latest/(?P<path>[^)\s]*)"
)
_MARKDOWN_SITE_AUTOLINK = re.compile(
    r"<" + re.escape(SITE) + r"latest/(?P<path>[^>\s]*)>"
)


def _relative_site_link(path, source):
    path, marker, anchor = path.partition("#")
    relative = posixpath.relpath(path or ".", posixpath.dirname(source))
    slash = "/" if not path or path.endswith("/") else ""
    return f"{relative}{slash}{marker}{anchor}"


def _local_html_links(output, source):
    return _SITE_LINK.sub(
        lambda match: f'{match["open"]}{_relative_site_link(match["path"], source)}"',
        output,
    )


def on_pre_build(config):
    """Clear page sources when a preview rebuilds the site."""
    _page_sources.clear()


def on_page_markdown(markdown, page, config, files):
    """Remove fence mode words and point repository links at GitHub."""
    docs = Path(config["docs_dir"]).resolve()
    root = Path(config["config_file_path"]).resolve().parent
    source = page.file.src_uri
    revision = os.environ.get("TREAMS_RS_DOCS_SOURCE_REF") or "main"

    def repository_link(match):
        target = match["target"]
        path, hash_, anchor = target.partition("#")
        resolved = Path(os.path.normpath((docs / source).parent / path))
        if resolved.is_relative_to(docs):
            return match[0]
        if not resolved.is_relative_to(root) or not resolved.exists():
            from mkdocs.exceptions import PluginError

            raise PluginError(f"{source}: link target {target} does not exist")
        # Images need the raw file; other links open the GitHub page.
        kind = "raw" if match["image"] else "tree" if resolved.is_dir() else "blob"
        relative = resolved.relative_to(root).as_posix()
        url = f"{REPOSITORY}/{kind}/{revision}/{relative}{hash_}{anchor}"
        return f"{match['image']}{match['text']}({url}"

    def rewrite_links(line, pattern, replacement):
        # Leave links that start inside an inline code span alone.
        spans = [span.span() for span in _CODE_SPAN.finditer(line)]

        def rewrite(match):
            if any(start <= match.start() < end for start, end in spans):
                return match[0]
            return replacement(match)

        return pattern.sub(rewrite, line)

    def published_link(match):
        return match["open"] + _relative_site_link(match["path"], source)

    def published_autolink(match):
        return f"[{match[0][1:-1]}]({_relative_site_link(match['path'], source)})"

    lines, published, fence = [], [], None
    for line in markdown.splitlines(keepends=True):
        published_line = None
        marker = _FENCE_LINE.match(line)
        if fence is None and marker:
            fence = marker["marker"]
            line = _FENCE_HEADER.sub(r"\g<head>", line)
        elif fence is not None:
            # A closing fence repeats the opening character at least as often.
            closing = marker and line.strip() == marker["marker"]
            if closing and marker["marker"].startswith(fence):
                fence = None
        else:
            line = rewrite_links(line, _PARENT_LINK, repository_link)
            published_line = rewrite_links(line, _MARKDOWN_SITE_LINK, published_link)
            published_line = rewrite_links(
                published_line, _MARKDOWN_SITE_AUTOLINK, published_autolink
            )
        lines.append(line)
        published.append(line if published_line is None else published_line)
    result = "".join(lines)
    _page_sources[source] = "".join(published)
    return result


def on_page_content(html, page, config, files):
    """Make the generated models table's scroll region keyboard-accessible."""
    if page.file.src_uri == "history/index.md":
        html = html.replace(
            '<div class="history-models">',
            '<div class="history-models" tabindex="0" role="region" '
            'aria-labelledby="models">',
        )
    return html


def on_post_page(output, page, config):
    """Keep links to this project's latest docs inside the displayed version."""

    revision = os.environ.get("TREAMS_RS_DOCS_SOURCE_REF") or "main"
    index = posixpath.relpath("llms.txt", posixpath.dirname(page.file.url))
    output = output.replace(
        f'href="{REPOSITORY}/blob/{revision}/llms.txt"', f'href="{index}"'
    )
    return _local_html_links(output, page.file.url)


def on_post_build(config):
    """Publish the Markdown and agent index beside this version's HTML."""
    site = Path(config["site_dir"])
    root = Path(config["config_file_path"]).resolve().parent
    for path in (site / "rust").rglob("*.html"):
        original = path.read_text(encoding="utf-8")
        rewritten = _local_html_links(original, path.relative_to(site).as_posix())
        if rewritten != original:
            path.write_text(rewritten, encoding="utf-8")
    for source, markdown in _page_sources.items():
        target = site / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    index = (root / "llms.txt").read_text(encoding="utf-8")
    index = index.replace(
        "The Markdown sources linked below are in this repository, at the same Git revision as this file.",
        "The Markdown sources linked below belong to this documentation version.",
    ).replace(
        f"The documentation site is {SITE}.",
        "The documentation site for these sources is [here](./).",
    )
    (site / "llms.txt").write_text(index.replace("](docs/", "]("), encoding="utf-8")
