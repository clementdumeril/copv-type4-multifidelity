import json
import matplotlib.pyplot as plt

manifest_path = "type4_calibration/outputs/comparison_tables/material_sensitivity_sweep_manifest.json"
output_png = "livrables/figures/material_sensitivity_sweep_en.png"

with open(manifest_path, "r", encoding="utf-8") as f:
    data = json.load(f)

py_corrected_p95_fi = data["python_corrected_p95_fi"]
base_calc_fibre_max = data["baseline_calculix"]["max"]
runs = data["runs"]

fi_max_vals = [r["calculix_fibre_fi_max"] for r in runs]
fi_p95_vals = [r["calculix_fibre_fi_p95"] for r in runs]

fig, ax = plt.subplots(figsize=(7.2, 4.8))

ax.hist(fi_max_vals, bins=8, alpha=0.6, label="CalculiX Fibre Failure Index (Max)", color="#c44d2d", edgecolor="black")
ax.hist(fi_p95_vals, bins=8, alpha=0.6, label="CalculiX Fibre Failure Index (p95)", color="#245b9b", edgecolor="black")

ax.axvline(base_calc_fibre_max, color="#1f8a4c", linestyle="--", linewidth=1.8, label=f"Reference Max ({base_calc_fibre_max:.3f})")
ax.axvline(py_corrected_p95_fi, color="#e67e22", linestyle="-", linewidth=2.0, label=f"Python Corrected p95 ({py_corrected_p95_fi:.3f})")

ax.set_xlabel("Fibre failure index proxy")
ax.set_ylabel("Frequency / Number of runs")
ax.set_title("Sensitivity to composite properties: CalculiX FI under ±10% variation")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(output_png, dpi=220)
plt.close(fig)
print("Sensitivity English plot generated successfully at", output_png)
