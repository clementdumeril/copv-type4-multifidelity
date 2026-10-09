#!/usr/bin/env python3
"""Build the README figures from the data committed in this repository.

    python code/plotting/make_readme_figures.py

Outputs (reports/figures/):
  readme_hero.png         CalculiX fiber-index field of case 11l_0122 next to the
                          critical-zone counts of the fast model and CalculiX (384 cases)
  readme_ga_check.png     fiber index of the 17 GA-optimised designs: fast model,
                          corrected fast model, CalculiX p95 and CalculiX maximum
The field render itself comes from render_fiber_field.py (needs the raw .frd files).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
FIG = REPO / "reports" / "figures"

ZONES = ["cylinder", "junction", "left_dome", "right_dome", "boss"]
ZONE_LABELS = ["cylinder", "dome-cylinder\njunction", "dome\n(fixed boss)", "dome\n(sliding boss)", "boss\nedge"]


def hero() -> None:
    df = pd.read_csv(DATA / "fiber_proxy_dataset.csv")
    fast = df["python_raw_critical_zone"].value_counts().reindex(ZONES, fill_value=0)
    fe = df["calculix_fiber_proxy_zone"].value_counts().reindex(ZONES, fill_value=0)

    img = mpimg.imread(FIG / "fiber_index_field_11l_0122.png")
    h, w = img.shape[:2]
    img = img[int(0.14 * h): int(0.82 * h), int(0.2 * w): int(0.85 * w)]
    case = df.set_index("case_id").loc["11l_0122"]
    vmax = round(float(case["calculix_fiber_stress_ratio_max"]), 2)

    fig = plt.figure(figsize=(13, 4.6), dpi=150)
    ax0 = fig.add_axes([0.0, 0.02, 0.5, 0.9])
    ax0.imshow(img)
    ax0.axis("off")
    ax0.set_title(f"CalculiX, case 11l_0122 at 70 MPa: peak fiber index {vmax:.2f} at the dome polar opening\n"
                  f"(fast model: {float(case['python_fiber_stress_ratio_max']):.2f}, uniform along the cylinder)",
                  fontsize=10, loc="left")
    cax = fig.add_axes([0.51, 0.18, 0.012, 0.6])
    bar = fig.colorbar(cm.ScalarMappable(colors.Normalize(0.0, vmax), "inferno"), cax=cax)
    bar.set_label(r"$|\sigma_{11}|/X$", fontsize=10)
    bar.ax.tick_params(labelsize=8)

    ax1 = fig.add_axes([0.62, 0.2, 0.36, 0.66])
    x = np.arange(len(ZONES))
    ax1.bar(x - 0.2, fast.values, 0.4, label="fast analytical model", color="#9aa5b1")
    ax1.bar(x + 0.2, fe.values, 0.4, label="CalculiX shell model", color="#c2410c")
    for xi, v in zip(x + 0.2, fe.values):
        ax1.text(xi, v + 6, str(v), ha="center", fontsize=8)
    ax1.text(x[0] - 0.2, fast.values[0] + 6, str(fast.values[0]), ha="center", fontsize=8)
    ax1.set_xticks(x, ZONE_LABELS, fontsize=8)
    ax1.set_ylabel("cases where the zone is critical")
    ax1.set_title(f"Where the fiber index peaks, {len(df)} DOE cases", fontsize=10)
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "readme_hero.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ga_check() -> None:
    df = pd.read_csv(DATA / "ga_statistical_validation.csv")
    df = df.sort_values("calculix_fibre_fi_max").reset_index(drop=True)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    ax.scatter(x, df["python_raw_fibre_fi"], marker="v", color="#9aa5b1", label="fast model, raw", zorder=3)
    ax.scatter(x, df["python_corrected_fibre_fi_p95"], marker="o", color="#2563eb", label="fast model + MLP correction (p95 margin)", zorder=3)
    ax.scatter(x, df["calculix_fibre_fi_p95"], marker="s", facecolor="none", edgecolor="#c2410c", label="CalculiX, p95 over plies and elements", zorder=3)
    ax.scatter(x, df["calculix_fibre_fi_max"], marker="x", color="#7f1d1d", label="CalculiX, pointwise maximum", zorder=3)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x, [f"s{s}-{r}" for s, r in zip(df["seed"], df["rank"])], rotation=60, fontsize=7)
    ax.set_xlabel("GA design (seed-rank), sorted by CalculiX maximum")
    ax.set_ylabel(r"fiber index $|\sigma_{11}|/X$ at 87 MPa")
    n = len(df)
    ax.set_title(f"{n} GA designs re-run in CalculiX: corrected ≥ p95 in {int(df['is_conservative_vs_p95'].sum())}/{n}, "
                 f"≥ maximum in {int(df['is_conservative_vs_max'].sum())}/{n}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "readme_ga_check.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    ga_check()
    print("wrote", FIG / "readme_hero.png", FIG / "readme_ga_check.png")
