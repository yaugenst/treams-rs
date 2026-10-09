// Keep Material's checkbox navigation, with named disclosure buttons and Space.
(() => {
  let cleanup = () => {};
  const mount = () => {
    cleanup();
    const events = new AbortController();
    for (const label of document.querySelectorAll(".md-nav--primary label.md-nav__link[for][id]")) {
      const nav = document.querySelector(`nav[aria-labelledby="${label.id}"]`);
      const toggle = document.getElementById(label.htmlFor);
      if (!nav || !toggle) continue;
      const name = nav.querySelector(".md-nav__title").textContent.trim();
      nav.id = `${label.htmlFor}_children`;
      label.setAttribute("role", "button");
      label.setAttribute("aria-label", name);
      label.setAttribute("aria-controls", nav.id);
      const sync = () => {
        label.setAttribute("aria-expanded", String(toggle.checked));
        nav.setAttribute("aria-expanded", String(toggle.checked));
      };
      sync();
      toggle.addEventListener("change", sync, { signal: events.signal });
      label.addEventListener("keydown", (event) => {
        if (event.key === " " || event.key === "Enter") {
          event.preventDefault();
          // Handle both keys at every viewport, without Material toggling again.
          event.stopImmediatePropagation();
          toggle.click();
        }
      }, { capture: true, signal: events.signal });
    }
    cleanup = () => events.abort();
  };
  if (typeof document$ !== "undefined") document$.subscribe(mount);
  else mount();
})();
