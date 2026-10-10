import json
import math
from pathlib import Path
import matplotlib.pyplot as plt

json_path = Path("type4_calibration/outputs/validation/paik2023_progressive_probe/paik2023_progressive_probe_results.json")
output_png = Path("livrables/figures/paik2023_progressive_damage_probe_en.png")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = data["rows"]
labels = sorted({row["published_case"] for row in rows})
factors = sorted({float(row["degradation_factor"]) for row in rows})

fig, ax = plt.subplots(figsize=(8.4, 4.8))
series_names = ["Experiment", "Published FE", "Local Reference"] + [f"Local Softening {factor:.2g}" for factor in factors]
width = 0.80 / len(series_names)
x = list(range(len(labels)))

# Baseline reference value for local reference (CalculiX 24x24 EX-A = 48.3)
baseline_val = {"EX-A": 48.28}

def values_for_series(name: str) -> list[float]:
    vals = []
    for label in labels:
        if name == "Experiment":
            vals.append(float(next(row for row in rows if row["published_case"] == label)["experimental_burst_pressure_mpa"]))
        elif name == "Published FE":
            vals.append(float(next(row for row in rows if row["published_case"] == label)["published_fe_pressure_mpa"]))
        elif name == "Local Reference":
            vals.append(baseline_val.get(label, 48.28))
        else:
            factor = float(name.rsplit(" ", 1)[-1])
            match = next(row for row in rows if row["published_case"] == label and abs(float(row["degradation_factor"]) - factor) < 1e-12)
            vals.append(float(match["probe_fibre_proxy_pressure_mpa"]))
    return vals

colors = ["#1f8a4c", "#245b9b", "#c44d2d", "#7b4ab8", "#d79b23", "#4a8f8f"]
for idx, name in enumerate(series_names):
    vals = values_for_series(name)
    offsets = [pos - 0.4 + width / 2.0 + idx * width for pos in x]
    ax.bar(offsets, vals, width=width, label=name, color=colors[idx % len(colors)])
    for xpos, value in zip(offsets, vals):
        if math.isfinite(value):
            ax.text(xpos, value + 1.0, f"{value:.1f}", ha="center", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Pressure [MPa]")
ax.set_title("Paik 2023: Exploratory local matrix stiffness softening study")
ymax = max(float(row["experimental_burst_pressure_mpa"]) for row in rows) * 1.22
ax.set_ylim(0.0, ymax)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(output_png, dpi=220)
plt.close(fig)
print("Progressive damage English plot generated successfully at", output_png)
