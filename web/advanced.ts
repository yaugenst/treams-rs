import {
  initialArray,
  initialCrystal,
  limits,
  selected,
  type AdvancedState,
  type Point,
  type Curve,
} from "./advanced-physics.js";
const el = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;
const canvas = el<HTMLCanvasElement>("scene"),
  ctx = canvas.getContext("2d")!;
const plot = document.getElementById("spectrum") as unknown as SVGSVGElement;
const worker = new Worker(new URL("./advanced-worker.js", import.meta.url), {
  type: "module",
});
let arrayState = initialArray(),
  crystalState = initialCrystal();
let state: AdvancedState =
  location.hash === "#crystal" ? crystalState : arrayState;
let revision = 0,
  timer: ReturnType<typeof setTimeout> | undefined,
  data: Point | undefined,
  curve: Curve | undefined;
let pending = true,
  bloch = false,
  phase = 0,
  lastTime = 0;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
let paused = reducedMotion.matches;
const percentage = (n: number) =>
  `${(100 * n).toFixed(n > 0 && n < 0.001 ? 3 : 1)}%`;
const status = (message: string) => {
  el("status").textContent = message;
};
function sync() {
  const array = state.kind === "array";
  el("array-controls").hidden = !array;
  el("crystal-controls").hidden = array;
  el("array-tab").setAttribute("aria-pressed", String(array));
  el("crystal-tab").setAttribute("aria-pressed", String(!array));
  el("plot-switch").hidden = array;
  el("gap-key").hidden = array;
  el("title").textContent = array
    ? "A small cell. An infinite surface."
    : "Make a forbidden color.";
  el("subtitle").textContent = array
    ? "Every particle hears every other particle. Change their spacing."
    : "Repeat two materials. Watch a photonic bandgap open.";
  el("scene-label").textContent = array
    ? "SQUARE ARRAY · ℓ ≤ 4 · SCHEMATIC"
    : "1D PHOTONIC CRYSTAL · Re(Ex)";
  el("scene-note").textContent = array
    ? "Beam width shows computed power. Geometry repeats infinitely."
    : "Computed field through a finite stack. Unit incident amplitude.";
  canvas.setAttribute(
    "aria-label",
    array
      ? "Infinite square array with calculated diffraction beam powers and directions"
      : "Calculated electric field inside an alternating dielectric stack",
  );
  el("explanation").innerHTML = array
    ? `<p>This is an <b>infinite square array</b> of lossless dielectric spheres (index 3.5), illuminated from air. The solver includes the electromagnetic coupling to every repeated cell using Ewald lattice sums.</p><p>Reflection and transmission add the power in every propagating diffraction order. Shorter wavelengths or an angled beam can open extra orders. Beam directions and widths use those calculated channels; the sphere drawing is a schematic, not a sampled near field.</p><p>Spherical multipoles through order 4 give an exploratory model. Sharp resonances need denser sampling and higher-order convergence checks. Exact grazing thresholds leave gaps in the spectrum.</p>`
    : `<p>This is a <b>one-dimensional photonic crystal</b>: alternating layers with indices 1.45 and the selected second index. Lengths are measured in periods <i>a</i>; frequency is <i>a/λ</i>. Light enters from air at normal incidence.</p><p>The field and power curves are calculated for the finite stack with a scattering-matrix solver. Shading identifies the forbidden bands of the <em>infinite</em> repeating crystal, from its Bloch eigenvalues.</p><p>Switch to Bloch to see phase advance Re(K)a/π and amplitude decay Im(K)a per period. More periods suppress transmission inside the gap; equal indices close it. Materials are lossless and dispersion is held constant.</p>`;
  const values: Record<string, [number, string]> = {
    wavelength: [
      arrayState.wavelength,
      `${arrayState.wavelength.toFixed(3)} μm`,
    ],
    period: [arrayState.period, `${arrayState.period.toFixed(3)} μm`],
    radius: [arrayState.radius, `${arrayState.radius.toFixed(3)} μm`],
    angle: [arrayState.angle, `${arrayState.angle}°`],
    frequency: [crystalState.frequency, crystalState.frequency.toFixed(3)],
    index: [crystalState.index, `${crystalState.index.toFixed(2)} / 1.45`],
    fill: [crystalState.fill, `${Math.round(crystalState.fill * 100)}%`],
    periods: [crystalState.periods, String(crystalState.periods)],
  };
  for (const [id, [value, text]] of Object.entries(values)) {
    el<HTMLInputElement>(id).value = String(value);
    el(`${id}-value`).textContent = text;
  }
  el("pause").textContent = paused ? "▷" : "Ⅱ";
  el("pause").setAttribute(
    "aria-label",
    paused ? "Play animation" : "Pause animation",
  );
  el("pause").setAttribute("aria-pressed", String(paused));
  const [lo, hi] = limits(state);
  plot.setAttribute(
    "aria-label",
    array ? "Wavelength in micrometers" : "Frequency a over wavelength",
  );
  plot.setAttribute("aria-valuemin", String(lo));
  plot.setAttribute("aria-valuemax", String(hi));
  plot.setAttribute("aria-valuenow", String(selected(state)));
  draw();
  drawPlot();
}
function request() {
  revision++;
  pending = true;
  document.querySelector<HTMLElement>(".advanced-stage")!.dataset.pending =
    "true";
  status("Calculating…");
  sync();
  if (timer === undefined)
    timer = setTimeout(() => {
      timer = undefined;
      worker.postMessage({ id: revision, state });
    }, 35);
}
worker.onmessage = (
  event: MessageEvent<{
    id: number;
    type: string;
    result: Point;
    curve: Curve;
    elapsed: number;
    message: string;
  }>,
) => {
  const message = event.data;
  if (message.id !== revision) return;
  if (message.type === "point") {
    data = message.result;
    pending = false;
    document.querySelector<HTMLElement>(".advanced-stage")!.dataset.pending =
      "false";
    const p = data.values;
    el("first-value").textContent = percentage(
      state.kind === "array" ? p[0]! : p[1]!,
    );
    el("second-value").textContent = percentage(
      state.kind === "array" ? p[1]! : p[0]!,
    );
    el("response-note").textContent =
      state.kind === "array"
        ? `${p[3]} open diffraction ${p[3] === 1 ? "order" : "orders"}`
        : p[3]! > 1e-7
          ? `In the gap · decay ${p[3]!.toFixed(2)} / period`
          : "Pass band · propagating Bloch wave";
    el("timing").textContent =
      `${Math.round(message.elapsed)} ms · selected response`;
    status("Scanning spectrum…");
    draw();
    document.dispatchEvent(new CustomEvent("advanced-point"));
  } else if (message.type === "curve") {
    curve = message.curve;
    status(
      curve.gaps
        ? `${curve.gaps} grazing ${curve.gaps === 1 ? "gap" : "gaps"}`
        : "",
    );
    drawPlot();
    document.dispatchEvent(new CustomEvent("advanced-result"));
  } else if (message.type === "error") {
    status(
      /graz/i.test(message.message)
        ? "Grazing threshold · tune slightly away"
        : "Calculation unavailable · try another setting",
    );
    el("response-note").textContent = message.message.replace(/^Error: /, "");
    el("first-value").textContent = "—";
    el("second-value").textContent = "—";
    data = undefined;
    draw();
  }
};
worker.onerror = () => status("Solver could not start. Reload to retry.");
for (const id of [
  "wavelength",
  "period",
  "radius",
  "angle",
  "frequency",
  "index",
  "fill",
  "periods",
]) {
  el<HTMLInputElement>(id).addEventListener("input", (event) => {
    const value = Number((event.target as HTMLInputElement).value);
    if (id === "wavelength") arrayState.wavelength = value;
    else if (id === "period") arrayState.period = value;
    else if (id === "radius") arrayState.radius = value;
    else if (id === "angle") arrayState.angle = value;
    else if (id === "frequency") crystalState.frequency = value;
    else if (id === "index") crystalState.index = value;
    else if (id === "fill") crystalState.fill = value;
    else crystalState.periods = value;
    request();
  });
}
function choose(kind: "array" | "crystal") {
  state = kind === "array" ? arrayState : crystalState;
  data = undefined;
  curve = undefined;
  el("first-value").textContent = "—";
  el("second-value").textContent = "—";
  history.replaceState(null, "", `#${kind}`);
  request();
}
el("array-tab").onclick = () => choose("array");
el("crystal-tab").onclick = () => choose("crystal");
el("reset").onclick = () => {
  arrayState = initialArray();
  crystalState = initialCrystal();
  choose(state.kind);
};
el("pause").onclick = () => {
  paused = !paused;
  sync();
};
reducedMotion.onchange = () => {
  paused = reducedMotion.matches;
  sync();
};
el("power-plot").onclick = () => {
  bloch = false;
  drawPlot();
};
el("bloch-plot").onclick = () => {
  bloch = true;
  drawPlot();
};
function tune(value: number) {
  const [lo, hi] = limits(state),
    bounded = Math.max(lo, Math.min(hi, value));
  if (state.kind === "array")
    state.wavelength = Math.round(bounded * 200) / 200;
  else state.frequency = Math.round(bounded * 1000) / 1000;
  request();
}
const tunePointer = (event: PointerEvent) => {
  const b = plot.getBoundingClientRect(),
    [lo, hi] = limits(state);
  tune(
    lo +
      Math.max(
        0,
        Math.min(1, (((event.clientX - b.left) / b.width) * 600 - 30) / 562),
      ) *
        (hi - lo),
  );
};
plot.onpointerdown = (event) => {
  plot.setPointerCapture(event.pointerId);
  tunePointer(event);
};
plot.onpointermove = (event) => {
  if (plot.hasPointerCapture(event.pointerId)) tunePointer(event);
};
plot.onpointerup = (event) => plot.releasePointerCapture(event.pointerId);
plot.onkeydown = (event) => {
  if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
    event.preventDefault();
    const [lo, hi] = limits(state);
    tune(
      event.key === "Home"
        ? lo
        : event.key === "End"
          ? hi
          : selected(state) +
            (event.key === "ArrowLeft" ? -1 : 1) *
              (state.kind === "array" ? 0.005 : 0.001),
    );
  }
};
function drawPlot() {
  const bands = bloch && state.kind === "crystal";
  el("power-plot").setAttribute("aria-pressed", String(!bands));
  el("bloch-plot").setAttribute("aria-pressed", String(bands));
  el("plot-label").textContent = bands ? "BLOCH DISPERSION" : "POWER SPECTRUM";
  document.querySelector(".reflection-key")!.textContent = bands
    ? "Phase · Re(K)a/π"
    : "Reflection";
  document.querySelector(".transmission-key")!.textContent = bands
    ? "Decay · Im(K)a"
    : "Transmission";
  const [lo, hi] = limits(state),
    x = (v: number) => 30 + ((v - lo) / (hi - lo)) * 562;
  let max = 1;
  if (bands && curve)
    for (let i = 0; i < curve.axis.length; i++)
      max = Math.max(max, curve.values[i * curve.stride + 3]!);
  const y = (v: number) => 109 - (v / max) * 100;
  let markup = "";
  if (state.kind === "crystal" && curve)
    for (let i = 0; i < curve.axis.length - 1; i++)
      if (curve.values[i * 5 + 3]! > 1e-7)
        markup += `<rect x="${x(curve.axis[i]!)}" y="8" width="${x(curve.axis[i + 1]!) - x(curve.axis[i]!) + 0.2}" height="101" fill="#eabc7919"/>`;
  for (const v of [0, max / 2, max])
    markup += `<path d="M30 ${y(v)}H592" stroke="#263449"/><text x="23" y="${y(v) + 3}" text-anchor="end" fill="#8498af" font-size="9">${v.toFixed(v === 0 ? 0 : 1)}</text>`;
  if (curve) {
    for (let line = 0; line < 2; line++) {
      let path = "",
        move = true;
      const column = bands
        ? 2 + line
        : state.kind === "array"
          ? line
          : 1 - line;
      for (let i = 0; i < curve.axis.length; i++) {
        const value = curve.values[i * curve.stride + column]!;
        if (!Number.isFinite(value)) {
          move = true;
          continue;
        }
        path += `${move ? "M" : "L"}${x(curve.axis[i]!).toFixed(2)} ${y(value).toFixed(2)}`;
        move = false;
      }
      markup += `<path d="${path}" fill="none" stroke="${line ? "#7ed5e2" : "#eabc79"}" stroke-width="2.2" stroke-linejoin="round" opacity="${pending ? 0.35 : 1}"/>`;
    }
  }
  const marker = x(selected(state));
  markup += `<path d="M${marker} 6V111" stroke="#f0f4f7" stroke-width="1.5"/><circle cx="${marker}" cy="6" r="3" fill="#f0f4f7"/>`;
  markup += `<text x="30" y="127" fill="#a4b4c8" font-size="10">${lo}</text><text x="312" y="127" text-anchor="middle" fill="#a4b4c8" font-size="10">${state.kind === "array" ? "Wavelength · μm" : "Frequency · a / λ"}</text><text x="592" y="127" text-anchor="end" fill="#a4b4c8" font-size="10">${hi}</text>`;
  plot.innerHTML = markup;
}
function line(points: [number, number][], color: string, width = 1) {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}
function arrow(
  start: [number, number],
  end: [number, number],
  power: number,
  color: string,
) {
  if (power < 0.00005) return;
  const width = 1 + 8 * Math.sqrt(Math.max(0, power));
  ctx.globalAlpha =
    Math.max(0.25, Math.sqrt(Math.min(1, power))) * (pending ? 0.35 : 1);
  line([start, end], color, width);
  const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(...end);
  ctx.lineTo(
    end[0] - 16 * Math.cos(angle - 0.6),
    end[1] - 16 * Math.sin(angle - 0.6),
  );
  ctx.lineTo(
    end[0] - 16 * Math.cos(angle + 0.6),
    end[1] - 16 * Math.sin(angle + 0.6),
  );
  ctx.fill();
  const t = (phase / (2 * Math.PI)) % 1;
  ctx.fillStyle = "#f0f4f7";
  ctx.beginPath();
  ctx.arc(
    start[0] + (end[0] - start[0]) * t,
    start[1] + (end[1] - start[1]) * t,
    Math.max(2, width / 3),
    0,
    2 * Math.PI,
  );
  ctx.fill();
  ctx.globalAlpha = 1;
}
function draw() {
  const w = canvas.clientWidth,
    h = canvas.clientHeight;
  if (!w || !h) return;
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui";
  if (state.kind === "array") {
    const scale = Math.min(w / 4.8, h / 2.4),
      center: [number, number] = [w * 0.5, h * 0.54];
    const project = (x: number, y: number, z = 0): [number, number] => [
      center[0] + scale * (0.8 * x + 0.52 * y),
      center[1] + scale * (-0.2 * x + 0.28 * y + 0.86 * z),
    ];
    const p = state.period,
      radius = state.radius * scale;
    for (let j = -4; j <= 4; j++)
      for (let i = 4; i >= -4; i--) {
        const [x, y] = project(i * p, j * p);
        if (x < -radius || x > w + radius || y < 25 || y > h - 20) continue;
        const glow = ctx.createRadialGradient(
          x - radius * 0.25,
          y - radius * 0.4,
          0,
          x,
          y,
          radius,
        );
        glow.addColorStop(0, "#5894a9");
        glow.addColorStop(0.4, "#285368");
        glow.addColorStop(1, "#11273c");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = i === 0 && j === 0 ? "#afdeea" : "#5a879974";
        ctx.lineWidth = i === 0 && j === 0 ? 1.5 : 0.7;
        ctx.stroke();
      }
    ctx.setLineDash([4, 4]);
    line(
      [
        project(-p / 2, -p / 2),
        project(p / 2, -p / 2),
        project(p / 2, p / 2),
        project(-p / 2, p / 2),
        project(-p / 2, -p / 2),
      ],
      "#eabc79a8",
    );
    ctx.setLineDash([]);
    const theta = (state.angle * Math.PI) / 180,
      d = Math.min(1.5, (h / scale) * 0.35);
    arrow(
      project(-0.35 - Math.sin(theta) * d, 0, -Math.cos(theta) * d),
      project(-0.35, 0),
      1,
      "#c8d7e5",
    );
    if (data) {
      const a = data.values;
      for (let i = 0; i < a[3]!; i++) {
        const at = 4 + i * 7,
          kx = a[at + 2]!,
          ky = a[at + 3]!,
          kz = a[at + 4]!;
        arrow(center, project(kx * d, ky * d, -kz * d), a[at + 5]!, "#eabc79");
        arrow(center, project(kx * d, ky * d, kz * d), a[at + 6]!, "#7ed5e2");
      }
    }
    ctx.fillStyle = "#cfdeea";
    ctx.font = "10px system-ui";
    ctx.fillText(`a = ${p.toFixed(3)} μm`, 13, h - 30);
  } else {
    const n = state.periods,
      left = w * 0.075,
      right = w * 0.925,
      x = (z: number) => left + ((z + 1) / (n + 2)) * (right - left),
      top = h < 140 ? 22 : 33,
      bottom = h - 32,
      mid = (top + bottom) / 2;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = "#314b6860";
      ctx.fillRect(x(i), top, x(i + state.fill) - x(i), bottom - top);
      ctx.fillStyle = `rgba(85,182,196,${0.13 + ((state.index - 1.45) / 2.05) * 0.3})`;
      ctx.fillRect(
        x(i + state.fill),
        top,
        x(i + 1) - x(i + state.fill),
        bottom - top,
      );
      line(
        [
          [x(i), top],
          [x(i), bottom],
        ],
        "#6dacc438",
      );
    }
    let amplitude = 1;
    if (data?.field)
      for (let i = 0; i < data.field.length; i += 2)
        amplitude = Math.max(
          amplitude,
          Math.hypot(data.field[i]!, data.field[i + 1]!),
        );
    const yScale = (bottom - top - 8) / (2 * amplitude);
    ctx.setLineDash([3, 4]);
    for (const v of [-1, 0, 1]) {
      line(
        [
          [left, mid - v * yScale],
          [right, mid - v * yScale],
        ],
        "#69869b40",
      );
      ctx.fillStyle = "#8ba4b8";
      ctx.font = "9px system-ui";
      if (v !== 0 || h >= 140) ctx.fillText(String(v), 6, mid - v * yScale + 3);
    }
    ctx.setLineDash([]);
    if (data?.field && data.positions) {
      const points: [number, number][] = [];
      for (let i = 0; i < data.positions.length; i++)
        points.push([
          x(data.positions[i]!),
          mid -
            (data.field[2 * i]! * Math.cos(phase) +
              data.field[2 * i + 1]! * Math.sin(phase)) *
              yScale,
        ]);
      ctx.globalAlpha = pending ? 0.3 : 1;
      ctx.shadowColor = "#7ed5e2";
      ctx.shadowBlur = 8;
      line(points, "#8be0e8", 2);
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = "#bdd2e0";
    ctx.font = "10px system-ui";
    ctx.fillText("air", x(-0.9), h - 28);
    ctx.fillText(`${n} periods`, w * 0.44, h - 28);
    ctx.fillText("air", x(n + 0.3), h - 28);
  }
}
new ResizeObserver(() => {
  const dpr = Math.min(devicePixelRatio, 2);
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}).observe(canvas);
function animate(time: number) {
  if (!paused && !document.hidden) {
    phase += (Math.min(50, time - lastTime) / 1000) * 2.7;
    draw();
  }
  lastTime = time;
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
request();
