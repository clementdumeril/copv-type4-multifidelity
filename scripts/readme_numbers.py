"""Recompute every number quoted in README.md from the files in data/.

Usage: python scripts/readme_numbers.py   (needs numpy and pandas; runs in a few seconds)
"""
import json
import math
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

met = json.loads((DATA / "extended_statistical_metrics.json").read_text(encoding="utf-8"))
meta, cv, ev = met["dataset_metadata"], met["cross_validation_metrics"], met["evaluation_metrics"]
print(f"designs {meta['total_rows']}: train {meta['train_count']}, held-out {meta['eval_count']} "
      f"(holdout {meta['holdout_count']}, boundary {meta['boundary_count']}, pressure {meta['pressure_scale_count']}), "
      f"duplicates {meta['duplicates']}, train/eval overlap {meta['data_leakage_train_eval_intersection']}")
for k in ("OLS", "Ridge", "MLP"):
    print(f"{k:5s} R2 CV {cv[k]['r2_mean']:.3f} +/- {cv[k]['r2_std']:.3f} | held-out {ev[k]['eval_r2']:.3f} | "
          f"boundary {ev[k]['boundary_r2']:.3f} | pressure {ev[k]['pressure_r2']:.3f}")

C = pd.read_csv(DATA / "fiber_proxy_dataset.csv").fiber_correction_ratio_calculix_over_python
print(f"C = FI_FE / FI_fast: min {C.min():.2f}, median {C.median():.2f}, max {C.max():.2f}")

ga = pd.read_csv(DATA / "ga_statistical_validation.csv")
print(f"GA check: {len(ga)} designs from {ga.seed.nunique()} seeds; corrected >= CalculiX p95: "
      f"{int(ga.is_conservative_vs_p95.sum())}, >= CalculiX max: {int(ga.is_conservative_vs_max.sum())}, "
      f"raw fast model >= p95: {int((ga.python_raw_fibre_fi >= ga.calculix_fibre_fi_p95).sum())}, "
      f"CalculiX max > 1: {int((ga.calculix_fibre_fi_max > 1).sum())}")

loc = pd.read_csv(DATA / "baseline_fiber_peak_location.csv")
pole = loc.peak_radius_mm < 1.25 * loc.boss_radius_mm
print(f"peak within 1.25 boss radii of the pole: {int(pole.sum())} (left boss {int((pole & (loc.peak_z_mm < 0)).sum())})")

cases = json.loads((DATA / "doe_11l_single_boss_cases.json").read_text(encoding="utf-8"))["cases"]
n_bad = sum(c["geometry"]["inner_radius_mm"] * math.sin(math.radians(min(c["winding"]["helical_angles_deg"])))
            > c["geometry"]["boss_radius_mm"] for c in cases)
print(f"designs where no helical ply reaches the polar opening: {n_bad} of {len(cases)}")

m = pd.read_csv(DATA / "paper_mesh_study.csv")
b = m[m.variant == "baseline"]
peak = b.pivot_table(index="case_id", columns="mesh", values="max_fiber_stress_ratio_abs")
p95 = b.pivot_table(index="case_id", columns="mesh", values="fiber_stress_ratio_p95_abs")
print(f"design 11l_0122 max index: {peak.loc['11l_0122', 16]:.2f} (16x16) -> {peak.loc['11l_0122', 64]:.2f} (64x64)")
print(f"p95 change 48 -> 64, four designs: max {((p95[64] - p95[48]).abs() / p95[64] * 100).max():.2f} %")

e = pd.read_csv(DATA / "paper_endcap.csv").join(
    b.set_index(["case_id", "mesh"])[["max_fiber_stress_ratio_abs", "fiber_stress_ratio_p95_abs"]],
    on=["case_id", "mesh"], rsuffix="_base")
e64 = e[e.mesh == 64]
dmax = (e64.max_fiber_stress_ratio_abs / e64.max_fiber_stress_ratio_abs_base - 1) * 100
dp95 = (e64.fiber_stress_ratio_p95_abs / e64.fiber_stress_ratio_p95_abs_base - 1) * 100
print(f"end-cap force at 64x64: max index {dmax.min():+.0f} to {dmax.max():+.0f} %, p95 within {dp95.abs().max():.1f} %")
