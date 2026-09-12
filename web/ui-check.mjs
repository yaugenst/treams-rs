import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { resolve, extname, sep } from "node:path";
import { chromium } from "playwright";
const root = resolve("dist");
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(
      new URL(req.url, "http://localhost").pathname,
    );
    const path = resolve(
      root,
      `.${pathname === "/" ? "/index.html" : pathname}`,
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
const localUrl = `http://127.0.0.1:${server.address().port}/`;
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
try {
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
    deviceScaleFactor: 1,
    reducedMotion: "reduce",
    permissions: ["clipboard-read", "clipboard-write"],
  });
  const page = await context.newPage(),
    errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(localUrl);
  const settled = () =>
    page.waitForFunction(
      () =>
        document.getElementById("status").textContent === "" &&
        document.getElementById("metric").textContent !== "—",
      null,
      { timeout: 30000 },
    );
  await settled();
  const cdp = await context.newCDPSession(page);
  const touch = (type, x, y) =>
    cdp.send("Input.dispatchTouchEvent", {
      type,
      touchPoints: type === "touchEnd" ? [] : [{ x, y }],
    });
  const field = page.locator("#field"),
    box = await field.boundingBox();
  const start = {
    x: box.x + box.width * (0.5 - 0.43 / 3.2),
    y: box.y + box.height / 2,
  };
  const destination = { x: start.x - 30, y: start.y - 50 };
  await touch("touchStart", start.x, start.y);
  await touch("touchMove", destination.x, destination.y);
  const immediate = await page.evaluate(async ({ x, y }) => {
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );
    const canvas = document.getElementById("field"),
      rect = canvas.getBoundingClientRect();
    const pixel = [
      ...canvas
        .getContext("2d")
        .getImageData(
          Math.round(((x - rect.left + 10) * canvas.width) / rect.width),
          Math.round(((y - rect.top) * canvas.height) / rect.height),
          1,
          1,
        ).data,
    ];
    return { status: document.getElementById("status").textContent, pixel };
  }, destination);
  assert(
    immediate.status,
    "field should still be catching up during immediate drag feedback",
  );
  assert.deepEqual(
    immediate.pixel,
    [255, 253, 247, 255],
    "particle must already be drawn under the finger before field completion",
  );
  await page.screenshot({ path: "output/mobile-drag.png" });
  await touch("touchEnd");
  await settled();
  console.log(
    "Touch drag: particle moves in two animation frames while field is pending.",
  );
  const names = [
    "01 Shape the light",
    "02 Find a resonance",
    "03 Mix the waves",
    "04 Add a shell",
    "05 Flip the light",
    "06 Let it improve",
  ];
  for (const name of names) {
    await page.getByRole("button", { name, exact: true }).click();
    await settled();
    const layout = await page.evaluate(() => ({
      overflow:
        document.documentElement.scrollHeight > innerHeight ||
        document.documentElement.scrollWidth > innerWidth,
      field: document.getElementById("field").getBoundingClientRect().toJSON(),
      controls: [...document.querySelectorAll(".primary-control input")].map(
        (e) => e.getBoundingClientRect().toJSON(),
      ),
    }));
    assert(
      !layout.overflow,
      `${name}: page must not scroll away from the field`,
    );
    for (const rect of [layout.field, ...layout.controls])
      assert(
        rect.top >= 0 &&
          rect.bottom <= 844 &&
          rect.left >= 0 &&
          rect.right <= 390,
        `${name}: field and primary controls fit together`,
      );
    if (name.startsWith("02")) {
      const b = await field.boundingBox();
      await touch("touchStart", b.x + b.width * 0.3, b.y + b.height * 0.5);
      await touch("touchMove", b.x + b.width * 0.65, b.y + b.height * 0.5);
      assert.equal(await page.locator("#wavelength").inputValue(), "2.01");
      await touch("touchEnd");
      await settled();
      await page.screenshot({ path: "output/mobile-resonance.png" });
      // Reproduce the reported faint quadrupole at its actual physical strength.
      await page.locator("#wavelength").fill("1.45");
      await page.locator("#wavelength").dispatchEvent("change");
      await settled();
      await page
        .getByRole("button", { name: "Quadrupole", exact: true })
        .click();
      await settled();
      assert.equal(await page.locator("#metric").textContent(), "0.00714");
      assert.match(await page.locator("#metric-unit").textContent(), /0.0901%/);
      assert.equal(
        await page.locator(".plot-heading span").first().textContent(),
        "QUADRUPOLE SPECTRUM",
      );
      await page.waitForTimeout(250); // finish the documented field crossfade
      const boosted = await field.evaluate((c) => c.toDataURL());
      await page.screenshot({ path: "output/mobile-quadrupole.png" });
      const boostButton = page.locator("#field-contrast");
      assert.match(await boostButton.textContent(), /Field ×[\d.]+ · boosted/);
      await boostButton.click();
      await page.evaluate(
        () =>
          new Promise((r) =>
            requestAnimationFrame(() => requestAnimationFrame(r)),
          ),
      );
      assert.notEqual(await field.evaluate((c) => c.toDataURL()), boosted);
      assert.equal(
        await page.locator("#metric").textContent(),
        "0.00714",
        "contrast never changes the physical score",
      );
    }
    if (name.startsWith("03")) {
      const before = await field.evaluate((c) => c.toDataURL());
      const b = await field.boundingBox();
      await touch("touchStart", b.x + b.width * 0.5, b.y + b.height * 0.5);
      await touch("touchMove", b.x + b.width * 0.75, b.y + b.height * 0.3);
      await touch("touchEnd");
      await settled();
      assert.equal(
        await page.locator("#control-relative-phase").inputValue(),
        "90",
      );
      assert.equal(
        await page.locator("#control-quadrupole-share").inputValue(),
        "70",
      );
      assert.notEqual(await field.evaluate((c) => c.toDataURL()), before);
    }
    if (name.startsWith("04")) {
      assert.equal(await page.locator("#metric").textContent(), "18%");
      const b = await field.boundingBox();
      await touch(
        "touchStart",
        b.x + b.width * (0.5 + 0.29 / 3.2),
        b.y + b.height * 0.5,
      );
      await touch(
        "touchMove",
        b.x + b.width * (0.5 + 0.4 / 3.2),
        b.y + b.height * 0.5,
      );
      await touch("touchEnd");
      await settled();
      assert(
        Math.abs(
          Number(await page.locator("#control-shell-thickness").inputValue()) -
            0.17,
        ) < 0.006,
      );
    }
    if (name.startsWith("05")) {
      const before = await page.locator("#metric").textContent();
      await page
        .getByRole("button", { name: "Helicity +", exact: true })
        .click();
      await settled();
      assert.notEqual(await page.locator("#metric").textContent(), before);
    }
    if (name.startsWith("06")) {
      const before = parseFloat(await page.locator("#metric").textContent());
      await page
        .getByRole("button", { name: "Step once", exact: true })
        .click();
      await settled();
      assert(parseFloat(await page.locator("#metric").textContent()) > before);
      await page.screenshot({ path: "output/mobile-gradient.png" });
      for (const id of ["run-optimization", "improve", "improvement"]) {
        const bounds = await page.locator(`#${id}`).boundingBox();
        assert(
          bounds && bounds.y >= 0 && bounds.y + bounds.height <= 844,
          `${id} is visible beside the field on mobile`,
        );
      }
      await page.getByRole("button", { name: "Reset", exact: true }).click();
      await settled();
      const positions = () =>
        page
          .locator(".position-row input")
          .evaluateAll((inputs) => inputs.map((input) => input.value));
      const initialPositions = await positions();
      await page
        .getByRole("button", { name: "Run optimization", exact: true })
        .click();
      await page.getByRole("button", { name: "Pause", exact: true }).click();
      await settled();
      assert.deepEqual(
        await positions(),
        initialPositions,
        "Pause discards the in-flight proposed move",
      );
      await page
        .getByRole("button", { name: "Run optimization", exact: true })
        .click();
      await page.waitForFunction(() =>
        /Step [2-9]/.test(document.getElementById("improvement").textContent),
      );
      assert.notDeepEqual(
        await positions(),
        initialPositions,
        "Run moves the particles automatically",
      );
      await page.screenshot({ path: "output/mobile-optimization-running.png" });
      await page.waitForFunction(
        () =>
          document
            .getElementById("improvement")
            .textContent.includes("No improving step fits"),
        null,
        { timeout: 45000 },
      );
      await settled();
      assert(
        parseFloat(await page.locator("#metric").textContent()) > 2 * before,
      );
      assert.equal(
        await page.locator("#run-optimization").getAttribute("aria-pressed"),
        "false",
      );
      await page.waitForTimeout(1200);
      await settled();
      assert.equal(
        await page.locator("#run-optimization").getAttribute("aria-pressed"),
        "false",
        "stopped run does not restart after final refinement",
      );
      await page.screenshot({ path: "output/mobile-optimization-stopped.png" });
      await page.getByRole("button", { name: "Reset", exact: true }).click();
      await settled();
      await page
        .getByRole("button", { name: "Run optimization", exact: true })
        .click();
      await page.locator("#wavelength").fill("1.8");
      await page.locator("#wavelength").dispatchEvent("change");
      await settled();
      assert.equal(
        await page.locator("#run-optimization").getAttribute("aria-pressed"),
        "false",
        "manual controls interrupt the run",
      );
      assert.deepEqual(
        await positions(),
        initialPositions,
        "manual edit discards stale optimization geometry",
      );
      await page.getByRole("button", { name: "Reset", exact: true }).click();
      await settled();
      await page.locator("#run-optimization").focus();
      await page.keyboard.press("Enter");
      await page.waitForFunction(() =>
        document.getElementById("improvement").textContent.startsWith("Step 1"),
      );
      assert.equal(
        await page.evaluate(() => document.activeElement.id),
        "run-optimization",
        "keyboard focus survives accepted steps",
      );
      await page.keyboard.press("Enter");
      await settled();
      assert.equal(
        await page.locator("#run-optimization").getAttribute("aria-pressed"),
        "false",
        "keyboard can pause after a step",
      );
    }
    console.log(`${name}: visible controls and live interaction passed.`);
  }
  await page
    .getByRole("button", { name: "Copy a link to this experiment" })
    .click();
  assert(new URL(page.url()).hash.length > 100);
  await page.reload();
  await settled();
  assert.equal(
    await page.locator("#title").textContent(),
    "Give light a destination",
  );
  await page.getByRole("button", { name: "More", exact: true }).click();
  await page
    .getByRole("slider", { name: "Refractive index", exact: true })
    .press("ArrowRight");
  await settled();
  const sticky = await field.boundingBox();
  assert(
    sticky.y >= 0 && sticky.y + sticky.height < 844,
    "advanced controls retain visible field",
  );
  for (const size of [
    { width: 375, height: 667 },
    { width: 320, height: 568 },
  ]) {
    await page.setViewportSize(size);
    await page.getByRole("button", { name: "01 Shape the light" }).click();
    await settled();
    assert(
      await page.evaluate(
        () =>
          document.documentElement.scrollHeight <= innerHeight &&
          document.documentElement.scrollWidth <= innerWidth,
      ),
      "small mobile page stays in viewport",
    );
  }
  await context.close();
  const desktop = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: "reduce",
  });
  await desktop.goto(localUrl);
  await desktop.waitForFunction(
    () => document.getElementById("status").textContent === "",
  );
  await desktop.locator("#control-particle-radius").fill("0.35");
  await desktop.locator("#control-particle-radius").dispatchEvent("change");
  await desktop.waitForFunction(
    () => document.getElementById("status").textContent === "",
  );
  await desktop.waitForTimeout(250);
  await desktop
    .locator("#field")
    .screenshot({ path: "output/sphere-edges.png" });
  await desktop.getByRole("button", { name: "03 Mix the waves" }).click();
  await desktop.waitForFunction(
    () => document.getElementById("status").textContent === "",
  );
  await desktop.screenshot({ path: "output/desktop-mixer.png" });
  await desktop.emulateMedia({ reducedMotion: "no-preference" });
  await desktop.getByRole("button", { name: "06 Let it improve" }).click();
  await desktop.waitForFunction(
    () => document.getElementById("status").textContent === "",
  );
  await desktop.evaluate(() => {
    window.motionCheck = new Promise((resolve) =>
      document.addEventListener(
        "lab-result",
        async () => {
          const canvas = document.getElementById("field");
          const values = [
            ...document.querySelectorAll(".position-row input"),
          ].map((input) => Number(input.value));
          const x = values[0],
            y = values[1],
            dx = x + 0.43,
            dy = y,
            length = Math.hypot(dx, dy);
          const radius = 0.23 - (6 * 3.2) / canvas.width;
          const px = Math.round(
            (0.5 + (x + (dx / length) * radius) / 3.2) * canvas.width,
          );
          const py = Math.round(
            (0.5 - (y + (dy / length) * radius) / 3.2) * canvas.height,
          );
          const sample = () => [
            ...canvas.getContext("2d").getImageData(px, py, 1, 1).data,
          ];
          const first = sample();
          await new Promise((r) => setTimeout(r, 260));
          resolve({ first, last: sample() });
        },
        { once: true },
      ),
    );
  });
  await desktop.getByRole("button", { name: "Step once", exact: true }).click();
  const motion = await desktop.evaluate(() => window.motionCheck);
  assert.notDeepEqual(
    motion.first,
    [255, 253, 247, 255],
    "particle has not jumped to the accepted endpoint in the first frame",
  );
  assert.deepEqual(
    motion.last,
    [255, 253, 247, 255],
    "particle animates into the accepted endpoint",
  );
  await desktop.screenshot({ path: "output/desktop-optimization.png" });
  assert.deepEqual(errors, [], "no page exceptions");
  console.log(
    "Share/reload, advanced controls, narrow phones and desktop render passed.",
  );
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
