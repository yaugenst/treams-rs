import { init, simulate, improve } from "./physics.js";
import type { State } from "./model.js";
const scope = self as unknown as DedicatedWorkerGlobalScope;
const ready = init();
scope.onmessage = async (
  event: MessageEvent<{
    id: number;
    state: State;
    n: number;
    improve: boolean;
  }>,
) => {
  const { id, n } = event.data;
  try {
    await ready;
    const improved = event.data.improve ? improve(event.data.state) : undefined;
    const result = simulate(id, improved?.state ?? event.data.state, n);
    if (improved)
      result.improvement = {
        message: improved.message,
        accepted: improved.accepted,
      };
    scope.postMessage(result, [result.field.buffer]);
  } catch (error) {
    scope.postMessage({
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
