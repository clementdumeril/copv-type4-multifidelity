#!/usr/bin/env python3
"""Write data/doe_11l_minimal_pins_cases.json: the DOE with no boss ring clamped axially.

In the DOE the left boss blocks U3 and the right boss slides, and the dome peak lands on the
fixed-boss side in every dome-critical design. `minimal_pins` (already supported by the case
generator) removes rigid-body modes with three points only, so neither boss is clamped.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
src = json.loads((DATA / "doe_11l_single_boss_cases.json").read_text(encoding="utf-8"))
for case in src["cases"]:
    case["boundary_conditions"]["axial_constraint"] = "minimal_pins"
src.setdefault("metadata", {})["variant"] = "axial_constraint = minimal_pins (no boss ring clamped)"
(DATA / "doe_11l_minimal_pins_cases.json").write_text(json.dumps(src, indent=2), encoding="utf-8")
print("wrote", DATA / "doe_11l_minimal_pins_cases.json")
