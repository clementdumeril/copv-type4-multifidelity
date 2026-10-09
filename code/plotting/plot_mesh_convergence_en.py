"""Mesh study on the Jin et al. (2022) EX-A / EX-B vessels: fibre-proxy burst pressure vs mesh density."""
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
csv_path = REPO / "data" / "jin2022_mesh_convergence.csv"
output_png = REPO / "reports" / "figures" / "paik2023_mesh_convergence_en.png"

with open(csv_path, "r", encoding="utf-8") as f:
    rows = [{k: (v if k in ("published_case", "axial_mode") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
labels = sorted({row["published_case"] for row in rows})
modes = list(dict.fromkeys(row["axial_mode"] for row in rows))

fig, axes = plt.subplots(len(labels), 1, figsize=(8.0, 3.3 * len(labels)), sharex=True)
if len(labels) == 1:
    axes = [axes]

colors = {"single_boss_reference": "#245b9b", "both_boss_rings": "#c44d2d"}

# Read baseline rows from baseline csv if exists, else default to 55.06 for EX-A
baseline_p = {"EX-A": 55.06}

for ax, label in zip(axes, labels):
    exp = float(next(row for row in rows if row["published_case"] == label)["experimental_burst_pressure_mpa"])
    base = baseline_p.get(label, math.nan)
    for mode in modes:
        sub = sorted(
            [row for row in rows if row["published_case"] == label and row["axial_mode"] == mode],
            key=lambda row: int(row["mesh"]),
        )
        mode_display = "Sliding boss" if mode == "single_boss_reference" else "Double boss"
        ax.plot(
            [int(row["mesh"]) for row in sub],
            [float(row["fibre_proxy_pressure_mpa"]) for row in sub],
            marker="o",
            label=f"{mode_display} (Max)",
            color=colors.get(mode, None),
        )
        if any(math.isfinite(float(row.get("fibre_proxy_pressure_p99_mpa", math.nan))) for row in sub):
            ax.plot(
                [int(row["mesh"]) for row in sub],
                [float(row["fibre_proxy_pressure_p99_mpa"]) for row in sub],
                marker="s",
                linestyle="--",
                label=f"{mode_display} (p99)",
                color=colors.get(mode, None),
                alpha=0.72,
            )
    ax.axhline(exp, color="#1f8a4c", linestyle="--", label="Experimental burst")
    if math.isfinite(base):
        ax.axhline(base, color="0.45", linestyle=":", label="24x24 reference")
    ax.set_title(f"{label} : Mesh Convergence (fibre proxy)")
    ax.set_ylabel("Pressure [MPa]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

axes[-1].set_xlabel("Mesh Density (n_meridional = n_theta)")
fig.tight_layout()
fig.savefig(output_png, dpi=220)
plt.close(fig)
print("Mesh convergence English plot generated successfully at", output_png)
