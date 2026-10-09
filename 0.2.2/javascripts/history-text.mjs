// Plain-text markers emitted by the private history exporter (see history/about).
const REDACTIONS = ["[a private folder]", "[computer name]", "[e-mail address]"];

// Some archived command excerpts were cut inside a marker. Keep only the text
// before that marker: the excerpt contains no recoverable remainder to expand.
export function commandText(text) {
  if (!text.endsWith("…")) return text;
  const start = text.lastIndexOf("[");
  if (start < 0) return text;
  const tail = text.slice(start, -1);
  // A bare bracket is ambiguous (shell tests, Python lists, regular expressions).
  if (tail.length > 1 && REDACTIONS.some((marker) => marker !== tail && marker.startsWith(tail))) {
    return `${text.slice(0, start)}…`;
  }
  return text;
}
