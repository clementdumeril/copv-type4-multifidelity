#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_utils import (  # noqa: E402
    COMPARISON,
    FIGURES,
    PYTHON_RESULTS,
    SIM_RESULTS,
    as_float,
    ensure_dirs,
    load_materials,
    read_csv,
    write_csv,
    write_json,
)


PYTHON_SUMMARY = PYTHON_RESULTS / "python_case_results.csv"
PYTHON_POINTS = PYTHON_RESULTS / "python_radial_points.csv"
SIM_CALCULIX = SIM_RESULTS / "calculix"
DATASET = COMPARISON / "fiber_proxy_dataset.csv"


def material_strengths(name: str, materials: dict[str, Any]) -> tuple[float, float]:
    mat = materials["composites"][name]
    return float(mat["Xt_mpa"]), float(mat["Xc_mpa"])


def update_best(best: dict[str, Any], row: dict[str, str], value: float) -> None:
    if value > float(best.get("fiber_stress_ratio_max", -math.inf)):
        best.clear()
        best.update(
            {
                "fiber_stress_ratio_max": value,
                "fiber_proxy_layer": row.get("layer", ""),
                "fiber_proxy_ply_name": row.get("name", ""),
                "fiber_proxy_angle_deg": row.get("angle_deg", ""),
                "fiber_proxy_radius_mm": row.get("radius_mm", ""),
                "fiber_proxy_sigma_11_mpa": row.get("sigma1_mpa", ""),
                "fiber_proxy_tau_12_mpa": row.get("tau12_mpa", ""),
                "fiber_proxy_combined": row.get("combined", ""),
            }
        )


def python_fiber_proxy_by_case(summary_rows: dict[str, dict[str, str]], materials: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with PYTHON_POINTS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = row["case_id"]
            summary = summary_rows.get(cid)
            if not summary:
                continue
            xt, xc = material_strengths(summary.get("composite_material", "carbon_epoxy_t700_like"), materials)
            s1 = as_float(row, "sigma1_mpa")
            ratio = abs(s1) / (xt if s1 >= 0.0 else xc)
            update_best(out.setdefault(cid, {}), row, ratio)
    for cid, values in out.items():
        pressure = as_float(summary_rows[cid], "pressure_mpa")
        ratio = float(values.get("fiber_stress_ratio_max", 0.0) or 0.0)
        values["fiber_proxy_pressure_mpa"] = pressure / ratio if ratio > 1e-12 else ""
    return out


def load_calculix_summary(case_id: str) -> dict[str, Any]:
    path = SIM_CALCULIX / case_id / "calculix_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def counts(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda item: item[0]))


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def stats(values: list[float]) -> dict[str, float]:
    vals = finite(values)
    if not vals:
        return {}
    arr = np.array(vals, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def plot_dataset(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    calc_ratio = np.array([float(row["calculix_fiber_stress_ratio_max"]) for row in rows], dtype=float)
    py_ratio = np.array([float(row["python_fiber_stress_ratio_max"]) for row in rows], dtype=float)
    correction = np.array([float(row["fiber_correction_ratio_calculix_over_python"]) for row in rows], dtype=float)
    calc_pressure = np.array([float(row["calculix_fiber_proxy_pressure_mpa"]) for row in rows], dtype=float)
    py_pressure = np.array([float(row["python_fiber_proxy_pressure_mpa"]) for row in rows], dtype=float)
    labels = [str(row["case_id"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(calc_ratio, py_ratio, s=24, color="#245b9b", edgecolor="white", linewidth=0.35)
    limit = max(float(np.max(calc_ratio)), float(np.max(py_ratio)), 1.0) * 1.05
    ax.plot([0.0, limit], [0.0, limit], "k--", linewidth=1.0, label="Python = CalculiX")
    ax.set_xlabel("CalculiX fibre stress ratio max")
    ax.set_ylabel("Python fibre stress ratio max")
    ax.set_title("Fibre-proxy criterion: Python vs CalculiX")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fiber_proxy_python_vs_calculix.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    x = np.arange(len(rows))
    ax.plot(x, correction, marker="o", markersize=3, linewidth=1.0, color="#c44d2d")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x[:: max(1, len(x) // 24)])
    ax.set_xticklabels([labels[i] for i in x[:: max(1, len(x) // 24)]], rotation=35, ha="right")
    ax.set_ylabel("CalculiX/Python fibre ratio")
    ax.set_title("Fibre-proxy correction factor by DOE case")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fiber_proxy_correction_factor_by_case.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(calc_pressure, py_pressure, s=24, color="#1f8a4c", edgecolor="white", linewidth=0.35)
    limit = max(float(np.max(calc_pressure)), float(np.max(py_pressure)), 1.0) * 1.05
    ax.plot([0.0, limit], [0.0, limit], "k--", linewidth=1.0, label="Python = CalculiX")
    ax.set_xlabel("CalculiX proxy pressure at fibre ratio = 1 [MPa]")
    ax.set_ylabel("Python proxy pressure at fibre ratio = 1 [MPa]")
    ax.set_title("Predicted fibre-limit pressure")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fiber_proxy_pressure_python_vs_calculix.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    materials = load_materials()
    python_rows = {row["case_id"]: row for row in read_csv(PYTHON_SUMMARY)}
    python_proxy = python_fiber_proxy_by_case(python_rows, materials)
    rows: list[dict[str, Any]] = []
    for cid, py in sorted(python_rows.items()):
        summary = load_calculix_summary(cid)
        if summary.get("solver_status") != "ccx_completed":
            continue
        calc_ratio = float(summary.get("max_fiber_stress_ratio_abs", 0.0) or 0.0)
        py_ratio = float(python_proxy.get(cid, {}).get("fiber_stress_ratio_max", 0.0) or 0.0)
        if calc_ratio <= 0.0 or py_ratio <= 0.0:
            continue
        pressure = as_float(py, "pressure_mpa")
        row = {
            "case_id": cid,
            "pressure_mpa": pressure,
            "inner_radius_mm": py.get("inner_radius_mm", ""),
            "cylindrical_length_mm": py.get("cylindrical_length_mm", ""),
            "boss_radius_mm": py.get("boss_radius_mm", ""),
            "dome_shape": py.get("dome_shape", ""),
            "dome_radius_factor": py.get("dome_radius_factor", ""),
            "helical_angle_min_deg": py.get("helical_angle_min_deg", ""),
            "helical_angle_max_deg": py.get("helical_angle_max_deg", ""),
            "transition_angle_max_deg": py.get("transition_angle_max_deg", ""),
            "hoop_angle_deg": py.get("hoop_angle_deg", ""),
            "helical_pairs": py.get("helical_pairs", ""),
            "transition_pairs": py.get("transition_pairs", ""),
            "hoop_pairs": py.get("hoop_pairs", ""),
            "total_thickness_mm": py.get("total_thickness_mm", ""),
            "layer_count": py.get("layer_count", ""),
            "python_fiber_stress_ratio_max": py_ratio,
            "calculix_fiber_stress_ratio_max": calc_ratio,
            "fiber_correction_ratio_calculix_over_python": calc_ratio / py_ratio,
            "python_fiber_proxy_pressure_mpa": python_proxy[cid]["fiber_proxy_pressure_mpa"],
            "calculix_fiber_proxy_pressure_mpa": summary.get("fiber_proxy_pressure_mpa", ""),
            "python_raw_max_combined": py.get("max_combined", ""),
            "calculix_raw_max_combined": summary.get("max_combined", ""),
            "python_raw_critical_zone": py.get("critical_zone_python", ""),
            "calculix_raw_critical_zone": summary.get("critical_zone", ""),
            "calculix_fiber_proxy_zone": summary.get("fiber_proxy_zone", ""),
            "calculix_fiber_proxy_ply": summary.get("fiber_proxy_ply", ""),
            "calculix_fiber_proxy_ply_name": summary.get("fiber_proxy_ply_name", ""),
            "calculix_fiber_proxy_angle_deg": summary.get("fiber_proxy_local_angle_deg", ""),
            "calculix_fiber_proxy_sigma_11_mpa": summary.get("fiber_proxy_sigma_11_mpa", ""),
            "calculix_fiber_proxy_tau_12_mpa": summary.get("fiber_proxy_tau_12_mpa", ""),
            "calculix_max_matrix_tension_hashin": summary.get("max_matrix_tension_hashin", ""),
            "calculix_max_matrix_compression_hashin": summary.get("max_matrix_compression_hashin", ""),
            "calculix_max_matrix_stress_ratio_abs": summary.get("max_matrix_stress_ratio_abs", ""),
            "calculix_max_shear_ratio_abs": summary.get("max_shear_ratio_abs", ""),
            "criterion_interpretation": "mode-separated fibre stress proxy for burst-scale comparison; raw matrix/shear indices kept as early-damage diagnostics",
        }
        rows.append(row)
    write_csv(DATASET, rows)
    if rows:
        plot_dataset(rows)
    correction = [float(row["fiber_correction_ratio_calculix_over_python"]) for row in rows]
    calc_pressure = [as_float(row, "calculix_fiber_proxy_pressure_mpa") for row in rows]
    py_pressure = [as_float(row, "python_fiber_proxy_pressure_mpa") for row in rows]
    write_json(
        COMPARISON / "fiber_proxy_dataset_manifest.json",
        {
            "case_count": len(rows),
            "source_python": str(PYTHON_POINTS),
            "source_calculix": str(SIM_CALCULIX),
            "target_observable": "max_fiber_stress_ratio_abs = max(|sigma_11|/Xt_or_Xc)",
            "pressure_proxy": "P_fiber_proxy = applied_pressure / max_fiber_stress_ratio_abs",
            "correction_ratio": "calculix_fiber_stress_ratio_max / python_fiber_stress_ratio_max",
            "correction_ratio_stats": stats(correction),
            "python_fiber_proxy_pressure_stats_mpa": stats(py_pressure),
            "calculix_fiber_proxy_pressure_stats_mpa": stats(calc_pressure),
            "calculix_fiber_proxy_zone_counts": counts([str(row["calculix_fiber_proxy_zone"]) for row in rows]),
            "warning": "This replaces raw max_combined as the burst-scale comparison target. Matrix/shear modes remain diagnostics, not final burst evidence.",
        },
    )
    print(DATASET)


if __name__ == "__main__":
    main()
