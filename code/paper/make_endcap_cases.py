#!/usr/bin/env python3
"""Write data/doe_11l_endcap_cases.json: the DOE with the boss end-cap force applied.

In the DOE `loading.include_boss_endcap_force` is False, so the polar edges carry no axial
force p * pi * r_boss^2. This variant switches it on (the generator applies it with *CLOAD
on both boss rings); everything else is unchanged.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
src = json.loads((DATA / "doe_11l_single_boss_cases.json").read_text(encoding="utf-8"))
for case in src["cases"]:
    case.setdefault("loading", {})["include_boss_endcap_force"] = True
src.setdefault("metadata", {})["variant"] = "include_boss_endcap_force = True"
(DATA / "doe_11l_endcap_cases.json").write_text(json.dumps(src, indent=2), encoding="utf-8")
print("wrote", DATA / "doe_11l_endcap_cases.json")
