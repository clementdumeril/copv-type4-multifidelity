#!/usr/bin/env python3
"""Write data/doe_11l_strict_cases.json: the geometrically consistent DOE cases, strict turnaround.

A geodesic helical ply reaches down to r_t = R sin(alpha). If r_t is larger than the boss
radius, no ply of the layup can reach the polar opening; the DOE model hides this with a
residual coverage (coverage_epsilon = 0.20-0.35) of every ply below its turnaround radius.

This script keeps the cases where the smallest helical turnaround radius is at most the boss
radius, and sets `winding.strict_turnaround = true` so that each ply stops at its own
turnaround radius (weights below 5 % are dropped). It also lists the inconsistent cases.
"""
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
src = json.loads((DATA / "doe_11l_single_boss_cases.json").read_text(encoding="utf-8"))
keep, dropped = [], []
for case in src["cases"]:
    g, w = case["geometry"], case["winding"]
    r_t = g["inner_radius_mm"] * math.sin(math.radians(min(w["helical_angles_deg"])))
    if r_t <= g["boss_radius_mm"]:
        case["winding"]["strict_turnaround"] = True
        keep.append(case)
    else:
        dropped.append(case["case_id"])
out = dict(src, cases=keep)
out.setdefault("metadata", {})["variant"] = (
    "strict_turnaround: plies end at their geodesic turnaround radius; only cases whose helical "
    "turnaround radius is <= the boss radius")
out["metadata"]["excluded_inconsistent_cases"] = dropped
(DATA / "doe_11l_strict_cases.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"kept {len(keep)} consistent cases, excluded {len(dropped)}")
