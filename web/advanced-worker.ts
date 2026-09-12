import {
  init,
  point,
  spectrum,
  type AdvancedState,
} from "./advanced-physics.js";
const ready = init();
let pending: { id: number; state: AdvancedState } | undefined,
  active = false,
  newest = 0;
self.onmessage = (
  event: MessageEvent<{ id: number; state: AdvancedState }>,
) => {
  pending = event.data;
  newest = event.data.id;
  if (!active) void run();
};
async function run() {
  active = true;
  try {
    await ready;
    while (pending) {
      const request = pending;
      pending = undefined;
      const start = performance.now();
      try {
        const result = point(request.state);
        self.postMessage({
          id: request.id,
          type: "point",
          result,
          elapsed: performance.now() - start,
        });
        const curve = await spectrum(
          request.state,
          () => newest !== request.id,
        );
        if (curve && newest === request.id)
          self.postMessage({
            id: request.id,
            type: "curve",
            curve,
            elapsed: performance.now() - start,
          });
      } catch (error) {
        self.postMessage({
          id: request.id,
          type: "error",
          message: String(error),
        });
      }
    }
  } catch (error) {
    self.postMessage({ id: newest, type: "error", message: String(error) });
  } finally {
    active = false;
  }
}
