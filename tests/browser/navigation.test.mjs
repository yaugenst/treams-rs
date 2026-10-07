import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const source = readFileSync(new URL("../../docs/javascripts/navigation.js", import.meta.url), "utf8");

function browser(initiallyExpanded = false) {
  let mount;
  const attributes = new Map();
  const label = Object.assign(new EventTarget(), {
    id: "section_label", htmlFor: "section",
    setAttribute: (key, value) => attributes.set(key, value),
  });
  const toggle = Object.assign(new EventTarget(), {
    checked: initiallyExpanded,
    click: () => change(!toggle.checked),
  });
  const nav = {
    querySelector: () => ({ textContent: "  Python API  " }),
    setAttribute() {},
  };
  function change(value) {
    toggle.checked = value;
    toggle.dispatchEvent(new Event("change"));
  }
  runInNewContext(source, {
    AbortController,
    document: {
      querySelectorAll: () => [label],
      querySelector: () => nav,
      getElementById: () => toggle,
    },
    document$: { subscribe(callback) { mount = callback; callback(); } },
  });
  return { label, nav, attributes, change, mount };
}

test("disclosures have names, controlled regions and live expanded state", () => {
  for (const initial of [false, true]) {
    const b = browser(initial);
    assert.equal(b.attributes.get("role"), "button");
    assert.equal(b.attributes.get("aria-label"), "Python API");
    assert.equal(b.attributes.get("aria-controls"), b.nav.id);
    assert.equal(b.attributes.get("aria-expanded"), String(initial));
    b.change(!initial);
    assert.equal(b.attributes.get("aria-expanded"), String(!initial));
  }
});

test("Space and Enter activate once after instant navigation, without scrolling", () => {
  const b = browser();
  b.mount();
  b.mount();
  const space = Object.assign(new Event("keydown", { cancelable: true }), { key: " " });
  b.label.dispatchEvent(space);
  assert.equal(space.defaultPrevented, true);
  assert.equal(b.attributes.get("aria-expanded"), "true");
  const enter = Object.assign(new Event("keydown", { cancelable: true }), { key: "Enter" });
  b.label.dispatchEvent(enter);
  assert.equal(enter.defaultPrevented, true);
  assert.equal(b.attributes.get("aria-expanded"), "false");
});
