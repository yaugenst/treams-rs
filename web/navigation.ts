import { initial, presets, type Experiment } from "./model.js";

type ExperimentId = Experiment | "array" | "crystal";
const experiments: [ExperimentId, string][] = [
  ...Object.entries(presets).map(
    ([key, preset]) =>
      [key as Experiment, preset.short] as [Experiment, string],
  ),
  ["array", "Infinite metasurface"],
  ["crystal", "Open a bandgap"],
];

export function navigation(
  nav: HTMLElement,
  local: Partial<Record<ExperimentId, () => void>>,
) {
  for (const [i, [key, label]] of experiments.entries()) {
    const action = local[key];
    const item = action
      ? document.createElement("button")
      : document.createElement("a");
    if (item instanceof HTMLAnchorElement)
      item.href =
        key === "array" || key === "crystal"
          ? `./advanced.html#${key}`
          : `./index.html#${encodeURIComponent(JSON.stringify(initial(key)))}`;
    else item.onclick = action!;
    item.id = `${key}-tab`;
    item.className = "nav-item";
    item.dataset.experiment = key;
    item.innerHTML = `<span class="nav-number">0${i + 1}</span>${label}`;
    nav.append(item);
  }
  return (selected: ExperimentId) => {
    for (const item of nav.querySelectorAll<HTMLElement>(".nav-item")) {
      const active = item.dataset.experiment === selected;
      if (active && !item.classList.contains("active")) {
        // Scroll only the experiment strip, keeping the scene in place.
        const bounds = item.getBoundingClientRect(),
          strip = nav.getBoundingClientRect();
        if (bounds.left < strip.left || bounds.right > strip.right)
          nav.scrollLeft += bounds.left - strip.left;
      }
      item.classList.toggle("active", active);
      item.setAttribute("aria-current", active ? "page" : "false");
      if (item instanceof HTMLButtonElement)
        item.setAttribute("aria-pressed", String(active));
    }
  };
}
