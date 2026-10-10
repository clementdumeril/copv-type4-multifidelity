#!/usr/bin/env python3
"""Assemble a dataset for a CalculiX variant campaign, in the format of data/fiber_proxy_dataset.csv.

The fast model only sees the cylinder, so its columns (python_*) and the design inputs are
copied from the reference dataset. The CalculiX columns are replaced by the variant's results,
and C is recomputed. Cases without a completed CalculiX run are dropped and listed.

    python code/paper/build_variant_dataset.py --runs <campaign folder> --out data/fiber_proxy_dataset_strict.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, help="campaign folder (TYPE4_OUTPUTS_DIR of the variant)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = pd.read_csv(DATA / "fiber_proxy_dataset.csv")
    calc = Path(args.runs) / "simulation_results" / "calculix"
    rows, missing = [], []
    for _, r in ref.iterrows():
        f = calc / r["case_id"] / "calculix_summary.json"
        s = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if s.get("solver_status") != "ccx_completed":
            missing.append(r["case_id"])
            continue
        row = r.to_dict()
        row["calculix_fiber_stress_ratio_max"] = s["max_fiber_stress_ratio_abs"]
        row["calculix_fiber_stress_ratio_p95"] = s.get("fiber_stress_ratio_p95_abs")
        row["calculix_fiber_stress_ratio_p99"] = s.get("fiber_stress_ratio_p99_abs")
        row["calculix_fiber_proxy_zone"] = s.get("fiber_proxy_zone", "")
        row["calculix_fiber_proxy_ply_name"] = s.get("fiber_proxy_ply_name", "")
        row["calculix_fiber_proxy_angle_deg"] = s.get("fiber_proxy_local_angle_deg", "")
        row["calculix_raw_max_combined"] = s.get("max_combined", "")
        row["calculix_raw_critical_zone"] = s.get("critical_zone", "")
        row["fiber_correction_ratio_calculix_over_python"] = s["max_fiber_stress_ratio_abs"] / r["python_fiber_stress_ratio_max"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} cases written to {args.out}; {len(missing)} without a completed run")


if __name__ == "__main__":
    main()
