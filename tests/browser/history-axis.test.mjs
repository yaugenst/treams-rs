import assert from "node:assert/strict";
import test from "node:test";
import { layoutAxisLabels } from "../../docs/javascripts/history-axis.mjs";

const label = (left, width, required = false, priority = 1) => ({ left, width, required, priority });

test("the first day survives a nearby break label on a phone", () => {
  // #21: Day 1 at 128–161 px, with Day 2 at 138–171 px.
  assert.deepEqual(layoutAxisLabels([label(0, 33, true), label(10, 33, false, 0)], 220),
    [{ left: 0, row: 0 }, null]);
});

test("five-pixel gaps no longer join the day and break labels", () => {
  // #21: Day 21, 3 h later and Day 22 had only 5 px between their text.
  assert.deepEqual(layoutAxisLabels([
    label(0, 40, true), label(45, 46, false, 0), label(96, 43),
  ], 160), [{ left: 0, row: 0 }, null, { left: 96, row: 0 }]);
});

test("every phase start survives, even when two require separate rows", () => {
  // #21: Day 3 at 817–852 px and Day 10 at 850–891 px.
  assert.deepEqual(layoutAxisLabels([label(0, 35, true), label(33, 41, true)], 90),
    [{ left: 0, row: 0 }, { left: 33, row: 1 }]);
});

test("end labels fit inside the track and outrank intermediate labels", () => {
  assert.deepEqual(layoutAxisLabels([
    label(0, 35, true), label(55, 35), label(95, 40, false, 2),
  ], 100), [{ left: 0, row: 0 }, null, { left: 60, row: 0 }]);
});

test("resizing recalculates rows and hidden labels without retaining old placement", () => {
  const labels = [label(0, 35, true), label(40, 35, true), label(120, 35)];
  assert.deepEqual(layoutAxisLabels(labels, 60),
    [{ left: 0, row: 0 }, { left: 25, row: 1 }, null]);
  assert.deepEqual(layoutAxisLabels([labels[0], label(60, 35, true), labels[2]], 200),
    [{ left: 0, row: 0 }, { left: 60, row: 0 }, { left: 120, row: 0 }]);
  assert.deepEqual(layoutAxisLabels(labels, 0), [null, null, null]);
});
