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
import re
from pathlib import Path

FENCE_MODES = (("exec",), ("exec", "jax"), ("exec", "torch"), ("no-exec",))
REPOSITORY = "https://github.com/yaugenst/treams-rs"

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


def on_page_markdown(markdown, page, config, files):
    """Remove fence mode words and point repository links at GitHub."""
    from mkdocs.exceptions import PluginError

    docs = Path(config["docs_dir"]).resolve()
    root = Path(config["config_file_path"]).resolve().parent
    source = page.file.src_uri
    revision = os.environ.get("GITHUB_SHA", "main")

    def repository_link(match):
        target = match["target"]
        path, hash_, anchor = target.partition("#")
        resolved = Path(os.path.normpath((docs / source).parent / path))
        if resolved.is_relative_to(docs):
            return match[0]
        if not resolved.is_relative_to(root) or not resolved.exists():
            raise PluginError(f"{source}: link target {target} does not exist")
        # Images need the raw file; other links open the GitHub page.
        kind = "raw" if match["image"] else "tree" if resolved.is_dir() else "blob"
        relative = resolved.relative_to(root).as_posix()
        url = f"{REPOSITORY}/{kind}/{revision}/{relative}{hash_}{anchor}"
        return f"{match['image']}{match['text']}({url}"

    def rewrite_links(line):
        # Leave links that start inside an inline code span alone.
        spans = [span.span() for span in _CODE_SPAN.finditer(line)]

        def rewrite(match):
            if any(start <= match.start() < end for start, end in spans):
                return match[0]
            return repository_link(match)

        return _PARENT_LINK.sub(rewrite, line)

    lines, fence = [], None
    for line in markdown.splitlines(keepends=True):
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
            line = rewrite_links(line)
        lines.append(line)
    return "".join(lines)
