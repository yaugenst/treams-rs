import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const root = new URL("../../", import.meta.url);
const source = readFileSync(new URL("docs/javascripts/history-loader.js", root), "utf8");
const tick = () => new Promise((resolve) => setImmediate(resolve));

function browser(page = "ordinary") {
  const assets = [], errors = [];
  let current = page, navigate;
  runInNewContext(source, {
    URL,
    console: { error: (error) => errors.push(error) },
    document: {
      currentScript: { src: "https://example.test/project/0.2.0/javascripts/history-loader.js" },
      querySelector: (selector) => current === "interactive"
        || (current === "about" && selector.includes(".history-r")),
      createElement: (tag) => ({ tag, remove() { this.removed = true; } }),
      body: { append: (element) => assets.push(element) },
    },
    document$: { subscribe: (callback) => { navigate = callback; callback(); } },
  });
  return {
    assets, errors,
    visit(page) { current = page; navigate(); },
  };
}

test("ordinary pages do not request history assets", () => {
  const config = readFileSync(new URL("mkdocs.yml", root), "utf8");
  assert.doesNotMatch(config, /^\s*- history\/history\.(css|js)\s*$/m);
  assert.match(config, /^\s*- javascripts\/history-loader\.js\s*$/m);
  const b = browser();
  b.visit("ordinary");
  assert.equal(b.assets.length, 0);
});

test("direct and instant visits load each asset once, inside the site version", async () => {
  for (const initial of ["ordinary", "interactive"]) {
    const b = browser(initial);
    b.visit("interactive");
    b.visit("interactive");
    assert.equal(b.assets.length, 1);
    assert.equal(b.assets[0].href, "https://example.test/project/0.2.0/history/history.css");
    b.assets[0].onload();
    await tick();
    assert.equal(b.assets.length, 2);
    assert.equal(b.assets[1].src, "https://example.test/project/0.2.0/history/history.js");
    b.assets[1].onload();
    b.visit("ordinary");
    b.visit("interactive");
    await tick();
    assert.equal(b.assets.length, 2);
    assert.deepEqual(b.errors, []);
  }
});

test("About uses only styles; leaving before styles arrive does not load the script", async () => {
  const b = browser("about");
  b.assets[0].onload();
  await tick();
  assert.equal(b.assets.length, 1);
  b.visit("interactive");
  b.visit("ordinary");
  await tick();
  assert.equal(b.assets.length, 1);
  b.visit("interactive");
  await tick();
  assert.equal(b.assets.length, 2);
});

test("a failed stylesheet or script can be retried on the next visit", async () => {
  const b = browser("interactive");
  b.assets[0].onerror();
  await tick();
  assert.equal(b.assets[0].removed, true);
  b.visit("interactive");
  b.assets[1].onload();
  await tick();
  b.assets[2].onerror();
  await tick();
  assert.equal(b.assets[2].removed, true);
  b.visit("interactive");
  await tick();
  b.assets[3].onload();
  assert.equal(b.assets.length, 4);
  assert.equal(b.errors.length, 2);
});
