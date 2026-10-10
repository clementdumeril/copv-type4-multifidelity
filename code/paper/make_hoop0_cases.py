#!/usr/bin/env python3
"""Write data/doe_11l_hoop0_cases.json: the 384 DOE cases with no hoop coverage in the dome.

Identical to data/doe_11l_single_boss_cases.json except `winding.hoop_coverage_epsilon = 0`,
so hoop plies stop at their turnaround radius (0.985 x inner radius) instead of keeping
coverage_epsilon (0.20-0.35) of their thickness down to the polar opening.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
src = json.loads((DATA / "doe_11l_single_boss_cases.json").read_text(encoding="utf-8"))
for case in src["cases"]:
    case["winding"]["hoop_coverage_epsilon"] = 0.0
src.setdefault("metadata", {})["variant"] = "hoop_coverage_epsilon = 0 (no hoop plies below the hoop turnaround radius)"
(DATA / "doe_11l_hoop0_cases.json").write_text(json.dumps(src, indent=2), encoding="utf-8")
print("wrote", DATA / "doe_11l_hoop0_cases.json", len(src["cases"]), "cases")
