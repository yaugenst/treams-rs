import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { commandText } from "../../docs/javascripts/history-text.mjs";

test("command excerpts cut before an incomplete redaction marker", () => {
  for (const marker of ["[a private folder]", "[computer name]", "[e-mail address]"]) {
    for (let cut = 2; cut < marker.length; cut++) {
      assert.equal(commandText(`pwd && ls ${marker.slice(0, cut)}…`), "pwd && ls …");
    }
    assert.equal(commandText(`ls ${marker}…`), `ls ${marker}…`);
    assert.equal(commandText(`ls ${marker}/src`), `ls ${marker}/src`);
  }
});

test("ordinary shell, Python and regex excerpts retain their contents", () => {
  for (const text of ["ls src…", "echo [computer name]", "test [ -f file…",
    "print([columns…", "rg '[a-z…", "python -c 'a = […", "echo [computer".slice(0, -2), "echo hello"]) {
    assert.equal(commandText(text), text);
  }
  assert.equal(commandText(`${"[computer name]".slice(0, -2)}…`), "…");
});

test("the published ci-merge reproducer retains its complete command prefix", () => {
  const path = new URL("../../docs/history/data/c/ci-merge.w1.json", import.meta.url);
  const { lines } = JSON.parse(readFileSync(path, "utf8"));
  assert.equal(commandText(lines[2][2]),
    "pwd && rg --files -g AGENTS.md -g '*treams*' -g '!**/node_modules/**' -g '!**/.git/**' …");
});

test("all exported command markers have readable display text", () => {
  const root = new URL("../../docs/history/data/c/", import.meta.url);
  const fragments = ["[a private folder]", "[computer name]", "[e-mail address]"]
    .flatMap((marker) => Array.from({ length: marker.length - 2 }, (_, i) => `${marker.slice(0, i + 2)}…`));
  // Include helper-agent records in subdirectories, not only top-level work.
  for (const file of readdirSync(root, { recursive: true }).filter((name) => name.endsWith(".json"))) {
    const { lines = [] } = JSON.parse(readFileSync(new URL(file, root), "utf8"));
    for (const [, verb, text] of lines) {
      if (verb === "ran") {
        assert.ok(!fragments.some((fragment) => commandText(text).endsWith(fragment)), `${file}: ${text}`);
      }
    }
  }
});
