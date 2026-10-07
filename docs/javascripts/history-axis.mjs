// Place required labels first, using extra rows only when they cannot share one.
export function layoutAxisLabels(labels, width, gap = 12) {
  const rows = [], placements = labels.map(() => null);
  const ordered = labels.map((label, index) => ({ ...label, index })).sort((a, b) =>
    Number(b.required) - Number(a.required) || b.priority - a.priority);
  for (const label of ordered) {
    if (width <= 0 || label.width > width) continue;
    const left = Math.max(0, Math.min(label.left, width - label.width));
    const right = left + label.width;
    let row = rows.findIndex((kept) => kept.every((k) => right + gap <= k.left || left >= k.right + gap));
    if (row < 0) {
      if (!label.required && rows.length) continue;
      row = rows.length;
      rows.push([]);
    }
    rows[row].push({ left, right });
    placements[label.index] = { left, row };
  }
  return placements;
}
