// Resolve from this script so instant navigation keeps the selected site version.
(() => {
  const base = new URL("../history/", document.currentScript.src);
  let styles, script;

  const load = (tag, file) => new Promise((resolve, reject) => {
    const element = document.createElement(tag);
    const url = new URL(file, base).href;
    if (tag === "link") {
      element.rel = "stylesheet";
      element.href = url;
    } else {
      element.src = url;
    }
    element.onload = resolve;
    element.onerror = () => {
      element.remove();
      reject(new Error(`Could not load ${url}`));
    };
    // Material reconciles head links on each navigation. Keep these outside
    // both the head and the page container so a loaded stylesheet stays active.
    document.body.append(element);
  });

  document$.subscribe(() => {
    // The records' About page uses redaction styles without interactive content.
    if (!document.querySelector("[data-history], .history-r")) return;
    styles ??= load("link", "history.css").catch((error) => {
      styles = null;
      throw error;
    });
    styles.then(() => {
      // Navigation may have left history while the stylesheet was in flight.
      if (!document.querySelector("[data-history]")) return;
      // history.js subscribes to Material's replayed document$ itself, including
      // the current page, and cleans up its mounts on subsequent navigation.
      script ??= load("script", "history.js").catch((error) => {
        script = null;
        throw error;
      });
      return script;
    }).catch(console.error);
  });
})();
