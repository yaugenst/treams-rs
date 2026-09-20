# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10"]
# ///
"""Plot retained per-attempt outcomes and agent wall time (not solver timing)."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--before", required=True)
parser.add_argument("--after", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = json.loads(args.summary.read_text())
groups = [
    [r for r in rows if r["cohort"] == name] for name in (args.before, args.after)
]
assert groups[0] and len(groups[0]) == len(groups[1]), "Need complete matched cohorts"
assert Counter((r["model"], r["case"]) for r in groups[0]) == Counter(
    (r["model"], r["case"]) for r in groups[1]
)
colors = {
    "pass": "#087F8C",
    "partial": "#E3B552",
    "fail": "#C45B55",
    "incomplete": "#B7BEC8",
}
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"}
)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(12.8, 6.2), gridspec_kw={"width_ratios": [1, 1.4]}
)
fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.22, wspace=0.5)
fig.suptitle(
    "More completed scattering tasks, less agent time",
    x=0.09,
    y=0.95,
    ha="left",
    fontsize=21,
    fontweight="bold",
)
fig.text(
    0.09,
    0.875,
    "Same 10 prompts · 4 models · 3 fresh attempts per cell · medium effort · 15-minute limit",
    color="#58636F",
)
for y, group in enumerate(groups):
    counts = Counter(r["outcome"] for r in group)
    offset = 0
    for outcome, color in colors.items():
        width = counts[outcome]
        left.barh(y, width, left=offset, height=0.5, color=color, edgecolor="white")
        if width >= 3:
            left.text(
                offset + width / 2,
                y,
                str(width),
                va="center",
                ha="center",
                fontweight="bold",
                color="white" if outcome in ("pass", "fail") else "#243443",
                fontsize=11,
            )
        offset += width
    details = " · ".join(
        f"{counts[name]} {name}" for name in colors if name != "pass" and counts[name]
    )
    left.text(
        0,
        y + 0.39,
        details or "All requested outcomes passed",
        fontsize=9,
        color="#58636F",
    )
left.set(
    yticks=[0, 1],
    yticklabels=["Original API", "Redesign"],
    xlim=(0, len(groups[0])),
    ylim=(1.7, -0.6),
    xlabel="Attempts",
)
left.set_title("Requested task outcomes", loc="left", pad=18, fontweight="bold")
models = [
    ("gpt-5.6-luna", "Luna"),
    ("gpt-5.6-sol", "Sol"),
    ("claude-sonnet-5", "Sonnet"),
    ("claude-opus-5", "Opus"),
]
for y, (model, _) in enumerate(models):
    values = [
        sum(r["elapsed_seconds"] for r in g if r["model"] == model) / 60 for g in groups
    ]
    right.plot(values, [y, y], color="#CBD1D8", lw=2)
    for value, color, offset in zip(
        values, ("#929CAA", colors["pass"]), (0.12, -0.12), strict=True
    ):
        right.scatter(value, y, s=55, color=color, zorder=3)
        right.annotate(
            f"{value:.1f}",
            (value, y),
            xytext=(0, 12 if offset < 0 else -20),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=10,
        )
right.set(
    yticks=range(4),
    yticklabels=[label for _, label in models],
    ylim=(3.7, -0.7),
    xlabel="Aggregate agent wall time (minutes)",
)
right.set_xlim(left=0)
right.set_title(
    "Time across 30 attempts per model", loc="left", pad=18, fontweight="bold"
)
for axis in (left, right):
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#EEF0F3")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
minutes = [sum(r["elapsed_seconds"] for r in g) / 60 for g in groups]
fig.text(
    0.09,
    0.12,
    f"{minutes[0]:.1f} → {minutes[1]:.1f} aggregate minutes ({100 * (1 - minutes[1] / minutes[0]):.0f}% less)",
    fontweight="bold",
    fontsize=13,
)
fig.legend(
    handles=[Patch(color=c, label=k.capitalize()) for k, c in colors.items()],
    loc="lower right",
    bbox_to_anchor=(0.97, 0.09),
    ncol=4,
    frameon=False,
    fontsize=9,
)
fig.text(
    0.09,
    0.045,
    "Wheel-only tools; all attempts retained, including provider errors. Gray time markers: original; teal: redesign. Workflow timings, not solver benchmarks.",
    fontsize=8.5,
    color="#58636F",
)
fig.text(
    0.09,
    0.02,
    "The predeclared 40% time-reduction target was not met.",
    fontsize=8.5,
    color="#58636F",
)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output)
fig.savefig(args.output.with_suffix(".png"), dpi=160)
print(args.output)
