import {
  initial,
  presets,
  geometryValid,
  fromHash,
  type State,
  type Result,
  type Experiment,
  type Particle,
} from "./model.js";
import { padField, patternGain } from "./field-view.js";
import { navigation } from "./navigation.js";
function element<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`Missing ${id}`);
  return e as T;
}
let toastTimer: ReturnType<typeof setTimeout>;
let refineTimer: ReturnType<typeof setTimeout>;
let gradients = true;
let fieldFade = 1;
let transitionStarted = 0;
let displayField: Float64Array = new Float64Array();
let contrast = 1,
  revealPattern = true;
let optimizing = false,
  optimizationSteps = 0,
  optimizationStart = 0;
let optimizationTimer: ReturnType<typeof setTimeout>;
let optimizationMessage = "Run to follow the gradient. Pause at any time.";
let particleMotion: { from: Particle[]; started: number } | undefined;
let state = initial();
try {
  state = fromHash(location.hash);
} catch {
  toast("That experiment link is invalid. Starting a fresh experiment.");
}
let result: Result | undefined,
  busy = false,
  pending = false,
  revision = 0,
  requestedImprove = false,
  grid = 80;
let display = "phase",
  playing = !matchMedia("(prefers-reduced-motion: reduce)").matches,
  dirty = true;
const canvas = element<HTMLCanvasElement>("field"),
  context = canvas.getContext("2d")!;
const raster = document.createElement("canvas"),
  rasterContext = raster.getContext("2d")!;
const transition = document.createElement("canvas"),
  transitionContext = transition.getContext("2d")!;
const worker = new Worker("./worker.js", { type: "module" }),
  status = element("status"),
  nav = element("experiments");
const syncNavigation = navigation(
  nav,
  Object.fromEntries(
    Object.keys(presets).map((key) => [key, () => select(key as Experiment)]),
  ),
);
function select(experiment: Experiment) {
  stopOptimization();
  optimizationSteps = 0;
  optimizationMessage = "Run to follow the gradient. Pause at any time.";
  state = initial(experiment);
  result = undefined;
  updateControls();
  schedule(true);
}
function updateControls() {
  const focusedId = document.activeElement?.id;
  const preset = presets[state.experiment],
    cluster = state.experiment === "particles" || state.experiment === "design";
  canvas.style.touchAction = "none";
  document.body.dataset.experiment = state.experiment;
  element("title").textContent = preset.title;
  element("description").textContent = preset.description;
  element("explanation").textContent = preset.explanation;
  element("eyebrow").textContent =
    `EXPERIMENT 0${Object.keys(presets).indexOf(state.experiment) + 1} / 08`;
  syncNavigation(state.experiment);
  element<HTMLInputElement>("wavelength").value = String(state.wavelength);
  element("wavelength-value").textContent = `${state.wavelength.toFixed(2)} μm`;
  element<HTMLInputElement>("angle").value = String(state.angle);
  element("angle-value").textContent = `${state.angle}°`;
  element("angle-group").hidden = state.experiment === "mixer";
  element("drag-hint").textContent = cluster
    ? state.experiment === "design"
      ? "Drag a particle or target"
      : "Drag a particle to move it"
    : {
        resonance: "Drag the field or spectrum to tune",
        mixer: "Drag to mix amplitude and phase",
        shell: "Drag the coating edge",
        chirality: "Drag to change material chirality",
      }[state.experiment as "resonance" | "mixer" | "shell" | "chirality"];
  const parameters = element("parameters");
  parameters.replaceChildren();
  parameters.append(
    slider(
      state.experiment === "shell" ? "Core radius" : "Particle radius",
      0.12,
      0.35,
      0.005,
      state.radius,
      "μm",
      (value) => {
        const next = structuredClone(state);
        next.radius = value;
        next.particles.forEach((p) => (p.radius = value));
        if (!geometryValid(next)) {
          toast("Leave a little space between the particles and target.");
          return false;
        }
        state = next;
        return true;
      },
    ),
  );
  parameters.append(
    slider(
      state.experiment === "shell" ? "Core index" : "Refractive index",
      1.4,
      4,
      0.05,
      Math.sqrt(state.epsilon),
      "",
      (value) => {
        state.epsilon = value * value;
        return true;
      },
    ),
  );
  if (state.experiment === "shell")
    parameters.append(
      slider(
        "Shell thickness",
        0.01,
        0.25,
        0.005,
        state.shell,
        "μm",
        (value) => {
          state.shell = value;
          return true;
        },
      ),
      slider("Shell index", 1, 2.8, 0.05, state.shellIndex, "", (value) => {
        state.shellIndex = value;
        return true;
      }),
    );
  if (state.experiment === "chirality")
    parameters.append(
      slider(
        "Material chirality",
        -0.5,
        0.5,
        0.01,
        state.chirality,
        "κ",
        (value) => {
          state.chirality = value;
          return true;
        },
      ),
    );
  if (state.experiment === "mixer")
    parameters.append(
      slider(
        "Quadrupole share",
        0,
        100,
        1,
        state.mixture * 100,
        "%",
        (value) => {
          state.mixture = value / 100;
          return true;
        },
      ),
      slider("Relative phase", -180, 180, 1, state.phase, "°", (value) => {
        state.phase = value;
        return true;
      }),
    );
  const actions = element("actions");
  actions.replaceChildren();
  if (state.experiment === "particles")
    actions.append(
      action(
        state.compare ? "Mutual scattering: off" : "Mutual scattering: on",
        () => {
          state.compare = !state.compare;
          updateControls();
          schedule();
        },
        !state.compare,
      ),
    );
  if (state.experiment === "resonance") {
    const group = document.createElement("div");
    group.className = "mode-choice";
    group.setAttribute("aria-label", "Scattered multipole shown");
    for (const [order, label] of [
      "All waves",
      "Dipole",
      "Quadrupole",
    ].entries()) {
      const b = action(
        label,
        () => {
          state.order = order;
          updateControls();
          schedule();
        },
        state.order === order,
      );
      b.classList.toggle("active", state.order === order);
      group.append(b);
    }
    actions.append(group);
  }
  if (state.experiment === "chirality") {
    const group = document.createElement("div");
    group.className = "mode-choice";
    for (const h of [0, 1])
      group.append(
        action(
          `Helicity ${h === 0 ? "−" : "+"}`,
          () => {
            state.helicity = h;
            updateControls();
            schedule();
          },
          state.helicity === h,
        ),
      );
    actions.append(group);
  }
  if (state.experiment === "mixer") {
    const equation = document.createElement("p");
    equation.className = "equation";
    equation.textContent = "b = T a";
    actions.append(equation);
  }
  if (state.experiment === "design") {
    const buttons = document.createElement("div");
    buttons.className = "optimization-actions";
    const run = action(
      "Run optimization",
      () => {
        if (optimizing) {
          pauseOptimization();
          return;
        }
        optimizing = true;
        optimizationSteps = 0;
        optimizationStart = result!.score;
        optimizationMessage = "Finding the next improving step…";
        requestStep();
      },
      false,
    );
    run.id = "run-optimization";
    run.className = "primary-button";
    const b = action("Step once", singleStep, false);
    b.id = "improve";
    buttons.append(run, b);
    actions.append(buttons);
    const note = document.createElement("p");
    note.id = "improvement";
    note.className = "action-note";
    note.setAttribute("role", "status");
    note.textContent = optimizationMessage;
    actions.append(note);
    const overlay = action(
      gradients ? "Gradient arrows: on" : "Gradient arrows: off",
      () => {
        gradients = !gradients;
        overlay.textContent = gradients
          ? "Gradient arrows: on"
          : "Gradient arrows: off";
        overlay.setAttribute("aria-pressed", String(gradients));
        dirty = true;
      },
      gradients,
    );
    actions.append(overlay);
    const caption = document.createElement("p");
    caption.className = "action-note";
    caption.textContent =
      "Arrows point uphill in target intensity. Lengths show relative sensitivity, not force.";
    actions.append(caption);
  }
  if (cluster) actions.append(positionControls());
  const primary = {
    particles: ["control-particle-radius", "wavelength"],
    resonance: ["wavelength", "control-particle-radius"],
    mixer: ["control-quadrupole-share", "control-relative-phase"],
    shell: ["control-shell-thickness", "control-shell-index"],
    chirality: ["control-material-chirality", "wavelength"],
    design: ["wavelength", "control-particle-radius"],
  }[state.experiment];
  document.querySelectorAll<HTMLElement>(".control-group").forEach((group) => {
    group.classList.toggle(
      "primary-control",
      primary.includes(group.querySelector("input")!.id),
    );
  });
  element("spectrum").hidden = state.experiment !== "resonance";
  element("field-contrast").hidden = !(
    state.experiment === "resonance" && state.order
  );
  element("metric").textContent = "—";
  element("metric-unit").textContent = "";
  element("metric-detail").textContent = "";
  updateOptimizationControls();
  dirty = true;
  updatePlay();
  if (focusedId)
    document.getElementById(focusedId)?.focus({ preventScroll: true });
}
function action(label: string, click: () => void, pressed: boolean) {
  const b = document.createElement("button");
  b.className = "secondary-button";
  b.textContent = label;
  b.setAttribute("aria-pressed", String(pressed));
  b.onclick = click;
  return b;
}
function slider(
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  unit: string,
  change: (value: number) => boolean,
) {
  const group = document.createElement("div");
  group.className = "control-group";
  const htmlLabel = document.createElement("label"),
    input = document.createElement("input"),
    output = document.createElement("output");
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  input.id = `control-${label.toLowerCase().replaceAll(" ", "-")}`;
  htmlLabel.htmlFor = input.id;
  htmlLabel.append(label, output);
  const format = (v: number) => `${v.toFixed(step >= 1 ? 0 : 2)} ${unit}`;
  output.textContent = format(value);
  input.oninput = () => {
    const v = Number(input.value);
    if (!change(v)) {
      input.value = String(value);
      return;
    }
    value = v;
    output.textContent = format(v);
    schedule(true);
  };
  input.onchange = () => schedule();
  group.append(htmlLabel, input);
  return group;
}
function positionControls() {
  const details = document.createElement("details");
  details.className = "positions";
  const summary = document.createElement("summary");
  summary.textContent = "Precise positions";
  details.append(summary);
  const rows = state.particles.map((p, i) => ({
    label: `Particle ${i + 1}`,
    values: [p.x, p.y],
    set: (next: State, axis: number, v: number) => {
      next.particles[i]![axis === 0 ? "x" : "y"] = v;
    },
  }));
  if (state.experiment === "design")
    rows.push({
      label: "Target",
      values: state.target,
      set: (next, axis, v) => {
        next.target[axis] = v;
      },
    });
  for (const row of rows) {
    const group = document.createElement("div");
    group.className = "position-row";
    const label = document.createElement("span");
    label.textContent = row.label;
    group.append(label);
    for (const axis of [0, 1]) {
      const input = document.createElement("input");
      input.type = "number";
      input.min = "-1.1";
      input.max = "1.1";
      input.step = ".02";
      input.value = row.values[axis]!.toFixed(2);
      input.setAttribute(
        "aria-label",
        `${row.label} ${axis === 0 ? "x" : "y"} in micrometres`,
      );
      input.onchange = () => {
        const next = structuredClone(state),
          v = input.valueAsNumber;
        row.set(next, axis, v);
        if (Number.isFinite(v) && Math.abs(v) <= 1.1 && geometryValid(next)) {
          state = next;
          schedule();
        } else {
          input.value = row.values[axis]!.toFixed(2);
          toast("Keep particles apart and the target outside them.");
        }
      };
      group.append(input);
    }
    details.append(group);
  }
  return details;
}
function updateOptimizationControls() {
  const run = document.getElementById(
    "run-optimization",
  ) as HTMLButtonElement | null;
  const step = document.getElementById("improve") as HTMLButtonElement | null;
  if (run) {
    run.textContent = optimizing ? "Pause" : "Run optimization";
    run.setAttribute("aria-pressed", String(optimizing));
    run.disabled = !optimizing && (busy || pending || !result);
  }
  if (step) step.disabled = optimizing || busy || pending || !result;
  const note = document.getElementById("improvement");
  if (note) note.textContent = optimizationMessage;
}
function stopOptimization(message?: string) {
  optimizing = false;
  clearTimeout(optimizationTimer);
  if (message) optimizationMessage = message;
  updateOptimizationControls();
}
function pauseOptimization() {
  stopOptimization(`Paused · ${optimizationSteps} accepted steps.`);
  // Invalidate an in-flight proposed move and refine the last accepted geometry.
  schedule();
}
function requestStep() {
  requestedImprove = true;
  schedule();
}
function singleStep() {
  stopOptimization();
  optimizationSteps = 0;
  optimizationStart = result!.score;
  requestStep();
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden && optimizing) pauseOptimization();
});
function schedule(coarse = false) {
  if (!requestedImprove) {
    if (optimizing)
      stopOptimization(`Paused · ${optimizationSteps} accepted steps.`);
    particleMotion = undefined;
  }
  clearTimeout(refineTimer);
  pending = true;
  revision++;
  grid =
    requestedImprove && optimizing
      ? 32
      : coarse
        ? 24
        : innerWidth < 620
          ? 56
          : 72;
  status.textContent = result ? "Field catching up…" : "Finding the field…";
  element("live-parameter").textContent =
    state.experiment === "mixer"
      ? `φ ${state.phase.toFixed(0)}° · mix ${(state.mixture * 100).toFixed(0)}%`
      : state.experiment === "shell"
        ? `Coating ${state.shell.toFixed(2)} μm`
        : state.experiment === "chirality"
          ? `κ ${state.chirality.toFixed(2)}`
          : `λ ${state.wavelength.toFixed(2)} μm`;
  dirty = true;
  dispatch();
  if (coarse) refineTimer = setTimeout(() => schedule(), 180);
}
function dispatch() {
  if (busy || !pending) return;
  busy = true;
  pending = false;
  const improve = requestedImprove;
  requestedImprove = false;
  updateOptimizationControls();
  worker.postMessage({ id: revision, state, n: grid, improve });
}
worker.onmessage = (
  event: MessageEvent<Result | { id: number; error: string }>,
) => {
  busy = false;
  const data = event.data;
  if (
    data.id === revision ||
    (!("error" in data) && JSON.stringify(data.state) === JSON.stringify(state))
  ) {
    if ("error" in data) {
      stopOptimization("Stopped: the solver could not complete this setting.");
      status.textContent = "Could not solve this setting";
      toast(data.error);
    } else {
      const sameExperiment = result?.state.experiment === data.state.experiment;
      result = data;
      displayField = padField(data.field, data.n);
      contrast = patternGain(data.field);
      if (sameExperiment) {
        transition.width = raster.width;
        transition.height = raster.height;
        transitionContext.drawImage(raster, 0, 0);
      }
      transitionStarted = sameExperiment ? performance.now() : 0;
      if (data.improvement) {
        const wasOptimizing = optimizing;
        if (data.improvement.accepted) {
          optimizationSteps++;
          if (!matchMedia("(prefers-reduced-motion: reduce)").matches)
            particleMotion = {
              from: state.particles,
              started: performance.now(),
            };
          optimizationMessage = `Step ${optimizationSteps} · ${optimizationStart.toFixed(2)}× → ${data.score.toFixed(2)}×`;
        } else {
          stopOptimization(
            `${optimizationSteps} steps · ${data.improvement.message}`,
          );
        }
        state = data.state;
        updateControls();
        if (optimizing) {
          // Let the accepted geometry and its field transition finish before the next solve.
          optimizationTimer = setTimeout(requestStep, 240);
        } else if (wasOptimizing) {
          refineTimer = setTimeout(() => schedule(), 240);
        }
      }
      status.textContent = pending ? "Refining…" : "";
      updateReadout();
      dirty = true;
    }
    if (data.id === revision)
      requestAnimationFrame(() =>
        document.dispatchEvent(
          new CustomEvent("lab-result", {
            detail: "error" in data ? data.error : null,
          }),
        ),
      );
  }
  updateOptimizationControls();
  dispatch();
};
worker.onerror = () => {
  busy = false;
  pending = false;
  stopOptimization("Stopped: the solver could not start.");
  status.textContent = "The solver could not start. Reload to retry.";
  document.dispatchEvent(
    new CustomEvent("lab-result", { detail: status.textContent }),
  );
};
function updateReadout() {
  if (!result) return;
  const r = result,
    s = r.state;
  let label = "",
    metric = "",
    unit = "",
    detail = "";
  switch (s.experiment) {
    case "particles":
      label = "PEAK SAMPLED INTENSITY";
      metric = `${r.peak.toFixed(2)}×`;
      unit = "incident intensity";
      detail = s.compare
        ? "Independent particles · coherent interference remains"
        : "Full interaction · repeated scattering included";
      break;
    case "resonance":
      label = s.order
        ? `${s.order === 1 ? "DIPOLE" : "QUADRUPOLE"} SCATTERING`
        : "SCATTERING EFFICIENCY";
      metric = s.order
        ? (r.orders[s.order - 1] ?? 0).toPrecision(3)
        : r.score.toFixed(2);
      unit = s.order
        ? `Qsca · ${((r.orders[s.order - 1]! / r.score) * 100).toPrecision(3)}% of total`
        : "Qsca · cross section / geometric area";
      detail = `Total Qsca ${r.score.toPrecision(3)} · dipole ${(((r.orders[0] ?? 0) / r.score) * 100).toPrecision(3)}% · quadrupole ${(((r.orders[1] ?? 0) / r.score) * 100).toPrecision(3)}%`;
      break;
    case "mixer":
      label = "SCATTERED MODE POWER";
      metric = r.score.toFixed(3);
      unit = "Σ|b|² / Σ|a|²";
      detail = `Electric dipole ${((1 - s.mixture) * 100).toFixed(0)}% + quadrupole ${(s.mixture * 100).toFixed(0)}% · fixed incident coefficient norm`;
      break;
    case "shell":
      label = "SCATTERING WITH THE COATING";
      metric = `${((r.score / r.reference) * 100).toFixed(0)}%`;
      unit = "of the same bare core";
      detail = `Coated ${r.score.toFixed(3)} μm² · bare ${r.reference.toFixed(3)} μm²`;
      break;
    case "chirality":
      label = `SCATTERING · HELICITY ${s.helicity === 0 ? "−" : "+"}`;
      metric = r.score.toFixed(3);
      unit = "μm² cross section";
      detail = `Opposite ${r.reference.toFixed(3)} μm² · selected-vs-opposite contrast ${(((r.score - r.reference) / (r.score + r.reference)) * 100).toFixed(1)}%`;
      break;
    case "design":
      label = "INTENSITY AT YOUR TARGET";
      metric = `${r.score.toFixed(2)}×`;
      unit = "incident intensity";
      detail = "Fixed point objective · total electric field";
      break;
  }
  element("metric-label").textContent = label;
  element("metric").textContent = metric;
  element("metric-unit").textContent = unit;
  element("metric-detail").textContent = detail;
  element("field-label").textContent =
    (s.experiment === "resonance" && s.order
      ? `SCATTERED ${s.order === 1 ? "DIPOLE" : "QUADRUPOLE"}`
      : "TOTAL FIELD") + (display === "phase" ? " · Re(Ez)" : " · |E|²");
  element("timing").textContent =
    `${Math.round(r.milliseconds)} ms solve + field`;
  const button = element("field-contrast");
  button.hidden = !(s.experiment === "resonance" && s.order);
  button.textContent = revealPattern
    ? `Field ×${contrast.toFixed(1)} · boosted`
    : "True strength · boost?";
  button.setAttribute("aria-pressed", String(revealPattern));
  element("legend-end").textContent =
    display === "intensity"
      ? `≥${(7 / displayGain() ** 2).toPrecision(2)}`
      : "+";
  drawSpectrum();
}
function displayGain() {
  return result?.state.experiment === "resonance" &&
    result.state.order &&
    revealPattern
    ? contrast
    : 1;
}
element("field-contrast").onclick = () => {
  revealPattern = !revealPattern;
  transitionStarted = 0;
  updateReadout();
  dirty = true;
};
function changeWavelength(value: number, coarse = false) {
  state.wavelength =
    Math.round(Math.max(1.1, Math.min(2.5, value)) * 100) / 100;
  element<HTMLInputElement>("wavelength").value = String(state.wavelength);
  element("wavelength-value").textContent = `${state.wavelength.toFixed(2)} μm`;
  drawSpectrum();
  schedule(coarse);
}
element<HTMLInputElement>("wavelength").oninput = (event) =>
  changeWavelength(Number((event.target as HTMLInputElement).value), true);
element("wavelength").onchange = () => schedule();
element<HTMLInputElement>("angle").oninput = (event) => {
  state.angle = Number((event.target as HTMLInputElement).value);
  element("angle-value").textContent = `${state.angle}°`;
  schedule(true);
};
element("angle").onchange = () => schedule();
element("reset").onclick = () => select(state.experiment);
element("more-controls").onclick = () => {
  const controls = element("controls"),
    button = element("more-controls");
  const expanded = controls.classList.toggle("expanded");
  button.textContent = expanded ? "Less" : "More";
  button.setAttribute("aria-expanded", String(expanded));
};
function updatePlay() {
  element("play").textContent = playing ? "Ⅱ" : "▷";
  element("play").setAttribute(
    "aria-label",
    playing ? "Pause field animation" : "Play field animation",
  );
}
element("play").onclick = () => {
  playing = !playing;
  updatePlay();
  dirty = true;
};
document.querySelectorAll<HTMLButtonElement>("[data-display]").forEach(
  (button) =>
    (button.onclick = () => {
      display = button.dataset.display!;
      document
        .querySelectorAll<HTMLButtonElement>("[data-display]")
        .forEach((b) => {
          b.classList.toggle("selected", b === button);
          b.setAttribute("aria-pressed", String(b === button));
        });
      element("legend-ramp").classList.toggle(
        "intensity",
        display === "intensity",
      );
      element("legend-start").textContent = display === "intensity" ? "0" : "−";
      element("legend-end").textContent = display === "intensity" ? "≥7" : "+";
      updateReadout();
      dirty = true;
    }),
);
function toast(message: string) {
  const box = element("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (box.hidden = true), 3800);
}
element("share").onclick = async () => {
  const url = new URL(location.href);
  url.hash = encodeURIComponent(JSON.stringify(state));
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url.href);
    toast("Experiment link copied");
  } catch {
    toast(
      "Your experiment is saved in the address bar. Copy its link to share.",
    );
  }
};
window.onhashchange = () => {
  try {
    state = fromHash(location.hash);
    result = undefined;
    updateControls();
    schedule();
  } catch {
    toast("That experiment link is invalid.");
  }
};
let phase = 0,
  previousTime = 0;
function render(time: number) {
  if (particleMotion && time - particleMotion.started >= 220) {
    particleMotion = undefined;
    dirty = true;
  }
  if (playing && !document.hidden)
    phase += Math.min(40, time - previousTime) * 0.002;
  previousTime = time;
  const fadeTarget = busy || pending ? 0.48 : 1;
  fieldFade += (fadeTarget - fieldFade) * 0.18;
  element("stage").classList.toggle("calculating", busy || pending);
  if (
    !document.hidden &&
    (dirty ||
      busy ||
      pending ||
      particleMotion ||
      Math.abs(fieldFade - fadeTarget) > 0.005 ||
      time - transitionStarted < 220 ||
      (playing && display === "phase"))
  ) {
    draw();
    dirty = false;
  }
  requestAnimationFrame(render);
}
function draw() {
  const width = canvas.clientWidth,
    height = canvas.clientHeight,
    dpr = Math.min(devicePixelRatio, 2);
  if (
    canvas.width !== Math.round(width * dpr) ||
    canvas.height !== Math.round(height * dpr)
  ) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.fillStyle = "#fffdf7";
  context.fillRect(0, 0, width, height);
  const span = 3.2;
  const s = state;
  if (result) {
    const { n } = result,
      field = displayField,
      gain = displayGain();
    if (raster.width !== n) {
      raster.width = n;
      raster.height = n;
    }
    const pixels = rasterContext.createImageData(n, n),
      c = Math.cos(phase),
      sin = Math.sin(phase);
    for (let i = 0; i < n * n; i++) {
      const offset = i * 6;
      let r = 255,
        g = 253,
        b = 247;
      if (Number.isFinite(field[offset])) {
        if (display === "phase") {
          const value =
              (field[offset + 4]! * c + field[offset + 5]! * sin) * gain,
            t = Math.abs(Math.tanh(value * 1.5));
          const color = value > 0 ? [215, 75, 63] : [36, 99, 197];
          r += t * (color[0]! - r);
          g += t * (color[1]! - g);
          b += t * (color[2]! - b);
        } else {
          let intensity = 0;
          for (let j = 0; j < 6; j++) intensity += field[offset + j]! ** 2;
          const t = Math.min(
              1,
              Math.log1p(intensity * gain ** 2) / Math.log(8),
            ),
            a = t < 0.55 ? t / 0.55 : (t - 0.55) / 0.45;
          const start = t < 0.55 ? [255, 253, 247] : [36, 99, 197],
            end = t < 0.55 ? [36, 99, 197] : [33, 58, 99];
          r = start[0]! + a * (end[0]! - start[0]!);
          g = start[1]! + a * (end[1]! - start[1]!);
          b = start[2]! + a * (end[2]! - start[2]!);
        }
      }
      pixels.data.set([r, g, b, 255], i * 4);
    }
    rasterContext.putImageData(pixels, 0, 0);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    const blend = Math.min(1, (performance.now() - transitionStarted) / 220);
    context.globalAlpha = fieldFade;
    if (blend < 1 && transition.width)
      context.drawImage(transition, 0, 0, width, height);
    context.globalAlpha = fieldFade * blend;
    context.drawImage(raster, 0, 0, width, height);
    context.globalAlpha = 1;
  }
  let particles =
    s.experiment === "particles" || s.experiment === "design"
      ? s.particles
      : [{ x: 0, y: 0, radius: s.radius }];
  if (particleMotion) {
    const t = Math.min(1, (performance.now() - particleMotion.started) / 220);
    const blend = t * t * (3 - 2 * t);
    particles = particles.map((p, i) => ({
      ...p,
      x: particleMotion!.from[i]!.x * (1 - blend) + p.x * blend,
      y: particleMotion!.from[i]!.y * (1 - blend) + p.y * blend,
    }));
  }
  const circle = (x: number, y: number, r: number) => {
    context.beginPath();
    context.arc(
      (x / span + 0.5) * width,
      (0.5 - y / span) * height,
      (r / span) * width,
      0,
      2 * Math.PI,
    );
  };
  particles.forEach((p, i) => {
    if (s.experiment === "shell") {
      circle(p.x, p.y, p.radius + s.shell);
      context.fillStyle = "#e2eafb";
      context.fill();
      context.strokeStyle = "#2463c5";
      context.lineWidth = 1;
      context.stroke();
    }
    circle(p.x, p.y, p.radius);
    context.fillStyle = "#fffdf7";
    context.fill();
    context.strokeStyle = "#213a63";
    context.lineWidth = 1.3;
    context.stroke();
    const cx = (p.x / span + 0.5) * width,
      cy = (0.5 - p.y / span) * height,
      radius = (p.radius / span) * width;
    context.save();
    context.clip();
    context.strokeStyle = "#213a6338";
    context.lineWidth = 0.7;
    context.beginPath();
    for (let line = -2; line <= 4; line++) {
      context.moveTo(cx - radius, cy + line * radius * 0.4);
      context.lineTo(cx - radius * 0.45, cy + (line - 0.7) * radius * 0.4);
    }
    context.stroke();
    context.restore();
    context.beginPath();
    context.ellipse(
      cx + 0.8,
      cy - 0.5,
      radius + 1,
      radius + 0.3,
      -0.2,
      0.2,
      5.7,
    );
    context.strokeStyle = "#213a6355";
    context.lineWidth = 0.7;
    context.stroke();
    context.font = "16px Kalam, cursive";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = "#213a63";
    context.fillText(
      String(i + 1),
      (p.x / span + 0.5) * width,
      (0.5 - p.y / span) * height,
    );
  });
  if (s.experiment === "design") {
    const x = (s.target[0] / span + 0.5) * width,
      y = (0.5 - s.target[1] / span) * height;
    context.strokeStyle = "#c13b32";
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(x, y, 9, 0, 2 * Math.PI);
    context.moveTo(x - 15, y);
    context.lineTo(x - 5, y);
    context.moveTo(x + 5, y);
    context.lineTo(x + 15, y);
    context.moveTo(x, y - 15);
    context.lineTo(x, y - 5);
    context.moveTo(x, y + 5);
    context.lineTo(x, y + 15);
    context.stroke();
    if (gradients && result && !busy && !pending && !particleMotion) {
      const offset = 1 + s.particles.length,
        g = result.gradient;
      const largest = Math.max(
        ...s.particles.map((_, i) =>
          Math.hypot(g[offset + i * 3]!, g[offset + i * 3 + 1]!),
        ),
      );
      if (largest > 1e-12)
        s.particles.forEach((p, i) => {
          const dx = g[offset + i * 3]!,
            dy = -g[offset + i * 3 + 1]!,
            length = Math.hypot(dx, dy),
            angle = Math.atan2(dy, dx);
          if (length < largest * 1e-4) return;
          const r = (p.radius / span) * width,
            arrow = (48 * length) / largest;
          context.save();
          context.translate(
            (p.x / span + 0.5) * width,
            (0.5 - p.y / span) * height,
          );
          context.rotate(angle);
          context.strokeStyle = "#fffdf7";
          context.lineWidth = 5;
          context.beginPath();
          context.moveTo(r + 4, 0);
          context.lineTo(r + 4 + arrow, 0);
          context.stroke();
          context.strokeStyle = "#213a63";
          context.lineWidth = 2.5;
          context.beginPath();
          context.moveTo(r + 4, 0);
          context.lineTo(r + 4 + arrow, 0);
          context.moveTo(r - 2 + arrow, -5);
          context.lineTo(r + 4 + arrow, 0);
          context.lineTo(r - 2 + arrow, 5);
          context.stroke();
          context.restore();
        });
    }
  }
  if (s.experiment !== "mixer") {
    const a = (s.angle * Math.PI) / 180;
    context.save();
    context.translate(width * 0.13, height * 0.8);
    context.rotate(-a);
    context.strokeStyle = "#213a63";
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(-15, 0);
    context.lineTo(16, 0);
    context.moveTo(10, -5);
    context.lineTo(16, 0);
    context.lineTo(10, 5);
    context.stroke();
    context.restore();
  }
}
function drawSpectrum() {
  const panel = element("spectrum");
  if (!result || result.state.experiment !== "resonance") return;
  if (!panel.children.length) {
    panel.innerHTML =
      '<div class="plot-heading"><span>SCATTERING SPECTRUM</span><span>Touch to tune wavelength</span></div><canvas id="spectrum-plot" role="img" aria-label="Scattering efficiency versus wavelength, from 1.1 to 2.5 micrometres"></canvas><div class="range-ends"><span>1.1 μm</span><span>2.5 μm</span></div>';
    const plot = element<HTMLCanvasElement>("spectrum-plot");
    let moving = false;
    const tune = (e: PointerEvent) => {
      const box = plot.getBoundingClientRect();
      changeWavelength(1.1 + ((e.clientX - box.left) / box.width) * 1.4, true);
    };
    plot.onpointerdown = (e) => {
      moving = true;
      plot.setPointerCapture(e.pointerId);
      tune(e);
    };
    plot.onpointermove = (e) => {
      if (moving) tune(e);
    };
    plot.onpointerup = plot.onpointercancel = () => {
      moving = false;
      schedule();
    };
  }
  const plot = element<HTMLCanvasElement>("spectrum-plot"),
    ctx = plot.getContext("2d")!,
    w = plot.clientWidth,
    h = plot.clientHeight,
    dpr = Math.min(devicePixelRatio, 2);
  plot.width = w * dpr;
  plot.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const values = result.spectrum;
  panel.querySelector(".plot-heading span")!.textContent = result.state.order
    ? `${result.state.order === 1 ? "DIPOLE" : "QUADRUPOLE"} SPECTRUM`
    : "SCATTERING SPECTRUM";
  let max = 0;
  for (let i = 1; i < values.length; i += 2) max = Math.max(max, values[i]!);
  max *= 1.15;
  ctx.strokeStyle = "#213a6324";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath();
    ctx.moveTo(0, (h * i) / 4);
    ctx.lineTo(w, (h * i) / 4);
    ctx.stroke();
  }
  ctx.beginPath();
  for (let i = 0; i < values.length; i += 2) {
    const x = ((values[i]! - 1.1) / 1.4) * w,
      y = h - 8 - (values[i + 1]! / max) * (h - 16);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#c13b32";
  ctx.lineWidth = 2;
  ctx.stroke();
  const x = ((state.wavelength - 1.1) / 1.4) * w;
  ctx.strokeStyle = "#2463c5";
  ctx.beginPath();
  ctx.moveTo(x, 4);
  ctx.lineTo(x, h - 4);
  ctx.stroke();
  ctx.fillStyle = "#637080";
  ctx.font = "11px system-ui";
  ctx.fillText(`Qsca · max ${max.toFixed(1)}`, 7, 13);
}
let drag = -1;
let fieldGesture = false;
function tuneField(event: PointerEvent) {
  const p = pointerPosition(event);
  const set = (id: string, value: number) => {
    const input = element<HTMLInputElement>(id);
    const step = Number(input.step);
    input.value = String(
      Math.round(
        Math.max(Number(input.min), Math.min(Number(input.max), value)) / step,
      ) * step,
    );
    input.dispatchEvent(new Event("input", { bubbles: true }));
  };
  if (state.experiment === "resonance")
    changeWavelength(1.1 + (p.x / 3.2 + 0.5) * 1.4, true);
  if (state.experiment === "mixer") {
    set("control-relative-phase", (p.x / 1.6) * 180);
    set("control-quadrupole-share", (p.y / 3.2 + 0.5) * 100);
  }
  if (state.experiment === "shell")
    set("control-shell-thickness", Math.hypot(p.x, p.y) - state.radius);
  if (state.experiment === "chirality")
    set("control-material-chirality", p.x / 3.2);
}
function pointerPosition(event: PointerEvent) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width - 0.5) * 3.2,
    y: (0.5 - (event.clientY - rect.top) / rect.height) * 3.2,
  };
}
canvas.onpointerdown = (event) => {
  if (optimizing) pauseOptimization();
  if (state.experiment !== "particles" && state.experiment !== "design") {
    if (
      state.experiment === "shell" &&
      Math.abs(
        Math.hypot(...Object.values(pointerPosition(event))) -
          state.radius -
          state.shell,
      ) > 0.2
    )
      return;
    fieldGesture = true;
    canvas.setPointerCapture(event.pointerId);
    tuneField(event);
    return;
  }
  const p = pointerPosition(event);
  drag =
    state.experiment === "design" &&
    Math.hypot(p.x - state.target[0], p.y - state.target[1]) < 0.2
      ? 2
      : state.particles.findIndex(
          (s) => Math.hypot(s.x - p.x, s.y - p.y) < s.radius + 0.13,
        );
  if (drag >= 0) canvas.setPointerCapture(event.pointerId);
};
canvas.onpointermove = (event) => {
  if (fieldGesture) {
    tuneField(event);
    return;
  }
  if (drag < 0) return;
  const p = pointerPosition(event),
    next = structuredClone(state);
  p.x = Math.max(-1.1, Math.min(1.1, p.x));
  p.y = Math.max(-1.1, Math.min(1.1, p.y));
  if (drag === 2) next.target = [p.x, p.y];
  else Object.assign(next.particles[drag]!, p);
  if (!geometryValid(next)) return;
  state = next;
  schedule(true);
};
canvas.onpointerup = canvas.onpointercancel = () => {
  if (fieldGesture) {
    fieldGesture = false;
    schedule();
    return;
  }
  if (drag < 0) return;
  drag = -1;
  const precise = document.querySelector(".positions");
  if (precise) {
    const open = (precise as HTMLDetailsElement).open;
    const replacement = positionControls();
    replacement.open = open;
    precise.replaceWith(replacement);
  }
  schedule();
};
new ResizeObserver(() => {
  dirty = true;
  drawSpectrum();
}).observe(canvas);
updateControls();
schedule(true);
requestAnimationFrame(render);

// Optional browser-native agent tools share the visible controls' state and solver.
interface BrowserTool {
  name: string;
  description: string;
  inputSchema: object;
  annotations: { readOnlyHint: boolean };
  execute: (input: unknown) => unknown;
}
const modelContext = (
  document as Document & {
    modelContext?: {
      registerTool: (
        tool: BrowserTool,
        options: { signal: AbortSignal },
      ) => unknown;
    };
  }
).modelContext;
if (modelContext?.registerTool) {
  const lifecycle = new AbortController();
  const read = () => ({
    state,
    solving: busy || pending,
    score: result?.score,
    label: element("metric-label").textContent,
    detail: element("metric-detail").textContent,
  });
  const completed = (action: () => void) =>
    new Promise((resolve, reject) => {
      document.addEventListener(
        "lab-result",
        (event) => {
          const error = (event as CustomEvent).detail;
          if (error) reject(new Error(error));
          else resolve(read());
        },
        { once: true },
      );
      action();
    });
  const tools: BrowserTool[] = [
    {
      name: "read_experiment",
      description:
        "Read current experiment parameters and the latest computed score.",
      inputSchema: {
        type: "object",
        properties: {},
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true },
      execute: read,
    },
    {
      name: "configure_experiment",
      description:
        "Set a complete experiment state obtained from read_experiment. Validate geometry, update the visible controls, and compute its field.",
      inputSchema: {
        type: "object",
        properties: { state: { type: "object" } },
        required: ["state"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false },
      execute: (input) => {
        const next = fromHash(
          encodeURIComponent(JSON.stringify((input as { state: State }).state)),
        );
        return completed(() => {
          state = next;
          result = undefined;
          updateControls();
          schedule();
        });
      },
    },
    {
      name: "improve_focus",
      description:
        "In the target experiment, take one accepted analytic adjoint step and return the new intensity.",
      inputSchema: {
        type: "object",
        properties: {},
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false },
      execute: () => {
        if (state.experiment !== "design" || busy || pending)
          throw new Error(
            "Open the target experiment and wait for its solve first",
          );
        return completed(singleStep);
      },
    },
  ];
  for (const tool of tools) {
    try {
      void Promise.resolve(
        modelContext.registerTool(tool, { signal: lifecycle.signal }),
      ).catch((error) => console.warn("Agent tool registration:", error));
    } catch (error) {
      console.warn("Agent tool registration:", error);
    }
  }
  window.addEventListener("pagehide", () => lifecycle.abort(), { once: true });
}
