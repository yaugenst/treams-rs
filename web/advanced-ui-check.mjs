import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdir, readFile } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { chromium } from "playwright";

const root = resolve("dist");
const server = createServer(async (req, res) => {
  try {
    const path = resolve(
      root,
      `.${decodeURIComponent(new URL(req.url, "http://localhost").pathname)}`,
    );
    if (!path.startsWith(root + sep)) throw new Error("outside root");
    const data = await readFile(path);
    res.setHeader(
      "Content-Type",
      {
        ".html": "text/html",
        ".js": "text/javascript",
        ".css": "text/css",
        ".wasm": "application/wasm",
      }[extname(path)] ?? "application/octet-stream",
    );
    res.end(data);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${server.address().port}/advanced.html`;
const browser = await chromium.launch({
  headless: true,
  ...(process.platform === "darwin"
    ? {
        executablePath:
          process.env.CHROME ??
          "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      }
    : {}),
});
await mkdir("output", { recursive: true });
const errors = [];
try {
  for (const viewport of [
    { width: 390, height: 844 },
    { width: 320, height: 568 },
    { width: 1440, height: 1000 },
  ]) {
    const mobile = viewport.width < 760;
    const context = await browser.newContext({
      viewport,
      isMobile: mobile,
      hasTouch: mobile,
      deviceScaleFactor: 1,
      reducedMotion: "reduce",
    });
    const page = await context.newPage();
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error")
        errors.push(`${message.text()} ${message.location().url}`);
    });
    await page.addInitScript(() => {
      window.advancedResults = 0;
      document.addEventListener(
        "advanced-result",
        () => window.advancedResults++,
      );
    });
    const settled = async (after) => {
      await page.waitForFunction(
        (after) => window.advancedResults > after,
        after,
        { timeout: 60000 },
      );
      const values = await page
        .locator("#first-value, #second-value")
        .allTextContents();
      assert(values.every((value) => /^\d+(\.\d+)?%$/.test(value)));
      assert(
        Math.abs(
          values.reduce((sum, value) => sum + parseFloat(value), 0) - 100,
        ) <= 0.11,
      );
      assert.equal(
        await page.locator('#spectrum path[stroke-width="2.2"]').count(),
        2,
        "both calculated spectrum curves are rendered",
      );
    };
    await page.goto(url);
    await settled(0);
    assert.equal(
      await page.locator("#pause").getAttribute("aria-label"),
      "Play animation",
    );
    const image = () =>
      page.locator("#scene").evaluate((canvas) => canvas.toDataURL());
    const pausedImage = await image();
    await page.waitForTimeout(80);
    assert.equal(
      await image(),
      pausedImage,
      "reduced motion starts with a still scene",
    );

    for (const kind of ["array", "crystal"]) {
      if (kind === "crystal") {
        const before = await page.evaluate(() => window.advancedResults);
        await page.locator("#crystal-tab").click();
        await settled(before);
      }
      const layout = await page.evaluate(
        (kind) => ({
          width: document.documentElement.scrollWidth,
          height: document.documentElement.scrollHeight,
          scrollY,
          legendBottom: document
            .querySelector(".plot-key")
            .getBoundingClientRect().bottom,
          controlsTop: document
            .querySelector(".advanced-controls")
            .getBoundingClientRect().top,
          readouts: [
            ...document.querySelectorAll(".advanced-readout strong"),
          ].map((el) => {
            const range = document.createRange();
            range.selectNodeContents(el);
            return range.getBoundingClientRect().toJSON();
          }),
          controls: [
            ...document.querySelectorAll(
              `#${kind}-controls input, #scene, #spectrum, .plot-key, .showcase-tabs button`,
            ),
          ].map((el) => ({
            id: el.id,
            ...el.getBoundingClientRect().toJSON(),
          })),
        }),
        kind,
      );
      await page.screenshot({
        path: `output/advanced-${kind}-${viewport.width}.png`,
      });
      assert(
        layout.width <= viewport.width,
        `${kind} ${viewport.width}: no horizontal overflow (${layout.width}px)`,
      );
      if (mobile)
        assert(
          layout.height <= viewport.height,
          `${kind} ${viewport.width}: page fits viewport`,
        );
      assert.equal(
        layout.scrollY,
        0,
        "switching experiments does not scroll the page",
      );
      assert(
        layout.readouts[0].right <= layout.readouts[1].left,
        "reflection and transmission readouts do not overlap",
      );
      if (mobile)
        assert(
          layout.legendBottom <= layout.controlsTop,
          `${kind} ${viewport.width}: spectrum legend is above controls`,
        );
      for (const bounds of layout.controls)
        assert(
          bounds.top >= 0 &&
            bounds.bottom <= viewport.height &&
            bounds.left >= 0 &&
            bounds.right <= viewport.width,
          `${kind} ${viewport.width}: ${bounds.id} stays visible with the scene`,
        );
      if (mobile) {
        const scene = await page.locator("#scene").boundingBox();
        await page.locator(".advanced-explanation summary").click();
        assert.deepEqual(
          await page.locator("#scene").boundingBox(),
          scene,
          "opening the explanation keeps the scene visible",
        );
        const details = await page.locator("#explanation").boundingBox();
        assert(
          details.y >= 0 && details.y + details.height <= viewport.height,
          "explanation fits the screen",
        );
        await page.screenshot({
          path: `output/advanced-${kind}-details-${viewport.width}.png`,
        });
        await page.locator(".advanced-explanation summary").click();
      }
      if (kind === "crystal") {
        const fieldSpan = await page.locator("#scene").evaluate((canvas) => {
          // The bright cyan field trace must have room to show its oscillation.
          const pixels = canvas
            .getContext("2d")
            .getImageData(0, 0, canvas.width, canvas.height).data;
          const rows = [];
          for (let y = 0; y < canvas.height; y++)
            for (let x = 0; x < canvas.width; x++) {
              const at = 4 * (y * canvas.width + x);
              if (
                pixels[at] > 100 &&
                pixels[at] < 160 &&
                pixels[at + 1] > 195 &&
                pixels[at + 2] > 205 &&
                pixels[at + 3] > 220
              ) {
                rows.push(y);
                break;
              }
            }
          return rows.length ? (rows.at(-1) - rows[0]) / canvas.height : 0;
        });
        assert(
          fieldSpan > 0.15,
          "the calculated crystal wave remains visible on a short screen",
        );
        const power = await page.locator("#spectrum").innerHTML();
        await page.locator("#bloch-plot").click();
        assert.equal(
          await page.locator("#plot-label").textContent(),
          "BLOCH DISPERSION",
        );
        assert.equal(
          await page.locator("#bloch-plot").getAttribute("aria-pressed"),
          "true",
        );
        assert.notEqual(await page.locator("#spectrum").innerHTML(), power);
        await page.screenshot({
          path: `output/advanced-bloch-${viewport.width}.png`,
        });
        await page.locator("#power-plot").click();
      }
      if (viewport.width !== 390) continue;

      const immediate = await page.evaluate(async (kind) => {
        const canvas = document.getElementById("scene"),
          ctx = canvas.getContext("2d");
        const scale = Math.min(
          canvas.clientWidth / 4.8,
          canvas.clientHeight / 2.4,
        );
        const sample = () =>
          kind === "array"
            ? [
                ...ctx.getImageData(
                  Math.round(canvas.width * 0.5 + scale * 0.2),
                  Math.round(canvas.height * 0.54),
                  1,
                  1,
                ).data,
              ]
            : canvas.toDataURL();
        const before = sample();
        const id = kind === "array" ? "radius" : "fill";
        const input = document.getElementById(id);
        input.value = kind === "array" ? ".17" : ".4";
        input.dispatchEvent(new Event("input", { bubbles: true }));
        await new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        );
        return {
          before,
          after: sample(),
          label: document.getElementById(`${id}-value`).textContent,
        };
      }, kind);
      assert.notDeepEqual(
        immediate.after,
        immediate.before,
        `${kind}: geometry redraws within two frames`,
      );
      assert.equal(immediate.label, kind === "array" ? "0.170 μm" : "40%");
      // Start a real calculation, then change it repeatedly while its spectrum is pending.
      await page.waitForTimeout(45);
      const count = await page.evaluate((kind) => {
        const input = document.getElementById(
          kind === "array" ? "wavelength" : "frequency",
        );
        for (const value of kind === "array"
          ? [1.1, 1.3, 1.575]
          : [0.13, 0.4, 0.3]) {
          input.value = String(value);
          input.dispatchEvent(new Event("input", { bubbles: true }));
        }
        return window.advancedResults;
      }, kind);
      await settled(count);
      const expected = await page.evaluate(async (kind) => {
        const { init, point } = await import("./advanced-physics.js");
        await init();
        const value = (id) => Number(document.getElementById(id).value);
        const state =
          kind === "array"
            ? {
                kind,
                wavelength: value("wavelength"),
                period: value("period"),
                radius: value("radius"),
                angle: value("angle"),
              }
            : {
                kind,
                frequency: value("frequency"),
                index: value("index"),
                fill: value("fill"),
                periods: value("periods"),
              };
        const p = point(state).values;
        return (kind === "array" ? [p[0], p[1]] : [p[1], p[0]]).map(
          (v) => 100 * v,
        );
      }, kind);
      const actual = await page
        .locator("#first-value, #second-value")
        .allTextContents();
      actual.forEach((value, i) =>
        assert(
          Math.abs(parseFloat(value) - expected[i]) < 0.051,
          `${kind}: latest input wins over in-flight work`,
        ),
      );

      const spectrum = page.locator("#spectrum"),
        bounds = await spectrum.boundingBox();
      const cdp = await context.newCDPSession(page);
      const dragCount = await page.evaluate(() => window.advancedResults);
      const x = bounds.x + bounds.width * 0.62,
        y = bounds.y + bounds.height * 0.5;
      await cdp.send("Input.dispatchTouchEvent", {
        type: "touchStart",
        touchPoints: [{ x: bounds.x + bounds.width * 0.3, y }],
      });
      await cdp.send("Input.dispatchTouchEvent", {
        type: "touchMove",
        touchPoints: [{ x, y }],
      });
      await cdp.send("Input.dispatchTouchEvent", {
        type: "touchEnd",
        touchPoints: [],
      });
      const parameter = kind === "array" ? "wavelength" : "frequency";
      const dragged = Number(await page.locator(`#${parameter}`).inputValue());
      assert.equal(
        Number(await spectrum.getAttribute("aria-valuenow")),
        dragged,
      );
      assert(
        Math.abs(dragged - (kind === "array" ? 1.26 : 0.427)) < 0.006,
        "touching the curve tunes its spectral coordinate",
      );
      await settled(dragCount);
      const keyCount = await page.evaluate(() => window.advancedResults);
      await spectrum.focus();
      await page.keyboard.press("ArrowRight");
      assert(
        Math.abs(
          Number(await page.locator(`#${parameter}`).inputValue()) -
            dragged -
            (kind === "array" ? 0.005 : 0.001),
        ) < 1e-9,
      );
      await settled(keyCount);
      await cdp.detach();
    }
    await context.close();
  }
  assert.deepEqual(errors, [], "no browser or worker errors");
  console.log(
    "Advanced showcases: real WASM powers/spectra, immediate geometry, latest-state updates, touch/keyboard plots, Bloch toggle and mobile/desktop layout passed.",
  );
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
