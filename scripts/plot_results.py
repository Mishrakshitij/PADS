#!/usr/bin/env python3
"""Regenerate README figures from measured data and transcribed manuscript tables."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#667085", "#E6A23C", "#26828E", "#4B57A8"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#D0D5DD")
    axis.tick_params(length=0, pad=8)
    axis.set_axisbelow(True)


def save(figure, target):
    figure.savefig(target, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(target.relative_to(ROOT) if target.is_relative_to(ROOT) else target)


def plot_dataset(target):
    rows = read_csv(ROOT / "datasets" / "summary.csv")
    domains = [row["domain"].capitalize() for row in rows]
    four_class = "politeness_label_0_rows" in rows[0]
    count_column = "source_rows" if four_class else "rows"
    counts = np.array([int(row[count_column]) for row in rows])
    denominators = np.array([int(row["annotated_rows"]) for row in rows]) if four_class else counts
    if np.any(denominators <= 0):
        raise ValueError("Each domain must have labeled records to plot.")
    labels = ["0 · Impolite", "1 · Somewhat impolite", "2 · Somewhat polite", "3 · Polite"] if four_class else ["Prediction 0", "Prediction 1"]
    columns = [f"politeness_label_{index}_rows" for index in range(4)] if four_class else ["prediction_0_rows", "prediction_1_rows"]
    shares = [np.array([int(row[column]) for row in rows]) / denominators for column in columns]
    if not np.allclose(sum(shares), 1):
        raise ValueError("Label counts do not sum to the number of annotated records.")
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    for axis in axes:
        style(axis)
        axis.set_yticks(positions, domains)
        axis.invert_yaxis()
        axis.grid(axis="x", color="#EAECF0")
    bars = axes[0].barh(positions, counts, color=COLORS[2], height=0.58)
    axes[0].bar_label(bars, labels=[f"{n:,}" for n in counts], padding=7, fontsize=10)
    axes[0].set_xlim(0, max(counts) * 1.28)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    axes[0].set_xlabel("CSV records")
    axes[0].set_title("Released records by domain", loc="left", fontweight="bold", pad=16)
    left = np.zeros(len(rows))
    palette = COLORS if four_class else [COLORS[0], COLORS[2]]
    for share, label, color in zip(shares, labels, palette):
        axes[1].barh(positions, share, left=left, color=color, label=label, height=0.58)
        left += share
    axes[1].set_xlim(0, 1)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel("Share of annotated records" if four_class else "Share of records")
    axes[1].set_title("Four politeness categories" if four_class else "Stored binary predictions", loc="left", fontweight="bold", pad=16)
    axes[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    figure.suptitle(f"PADS data release · {sum(counts):,} records across six domains", x=0.08, ha="left", fontsize=16, fontweight="bold", y=1.02)
    if four_class:
        omitted = sum(int(row["unscorable_rows"]) for row in rows)
        figure.text(0.08, -0.05, f"Class proportions exclude {omitted:,} blank source records retained without a label.", fontsize=9, color="#475467")
    figure.tight_layout(w_pad=3)
    save(figure, target / "dataset-overview.png")


def plot_paper(target):
    rows = read_csv(ROOT / "docs" / "paper-results.csv")
    if any(row["status"] != "manuscript-reported" for row in rows):
        raise ValueError("Expected manuscript-reported result provenance.")
    figure, axis = plt.subplots(figsize=(11, 5.5))
    style(axis)
    positions = np.arange(len(rows))
    width = 0.18
    for index, (method, color) in enumerate(zip(["Baseline", "PRIS", "PRSIS", "PRRIS"], COLORS)):
        values = np.array([float(row[method]) for row in rows])
        axis.bar(positions + (index - 1.5) * width, values, width=width,
                 color=color, label=method)
    axis.set_xticks(positions, [row["domain"].capitalize() for row in rows])
    axis.set_ylim(0, 0.9)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylabel("Task success rate · higher is better")
    axis.grid(axis="y", color="#EAECF0")
    axis.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.08))
    figure.suptitle("PADS · manuscript-reported task success", x=0.08, ha="left", fontsize=16, fontweight="bold", y=1.03)
    figure.text(0.08, -0.015, "Source: author manuscript, Table 5 · Template+Retrieval simulator · 15,000 dialogues in table caption\nRecorded values and evaluation notes: docs/paper-results.md", fontsize=9, color="#475467")
    figure.tight_layout()
    save(figure, target / "paper-success-rates.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    plot_dataset(args.output_dir)
    plot_paper(args.output_dir)


if __name__ == "__main__":
    main()
