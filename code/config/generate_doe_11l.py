#!/usr/bin/env python3
"""Generate a fixed-volume 11 L DOE for Type IV vessel correction.

This campaign is intentionally different from the historical broad DOE:

* every case has the same internal target volume, 11 L;
* the cylindrical length is derived from the selected radius and dome geometry;
* initial/boundary-condition metadata are written in every case;
* the main axial support is a single boss reference, with the opposite
  boss left axially free to avoid a non-physical two-end axial clamp;
* smooth dome coverage parameters are sampled explicitly;
* case roles distinguish training, holdout, boundary and pressure-scale checks.

The default case count, 384, is chosen as a practical minimum for the current
small MLP. A PINN would require field-level samples from the CalculiX solution,
not only scalar case summaries.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "doe_11l_single_boss_cases.json"

DEFAULT_SEED = 20260527
DEFAULT_CASES = 384
TARGET_VOLUME_L = 11.0
TARGET_VOLUME_MM3 = TARGET_VOLUME_L * 1_000_000.0

RANGES = {
    "inner_radius_mm": (76.0, 96.0),
    "boss_radius_ratio": (0.28, 0.36),
    "dome_radius_factor": (1.00, 1.16),
    "helical_angle_deg": (12.0, 26.0),
    "transition_angle_deg": (30.0, 56.0),
    "hoop_angle_deg": (86.5, 89.5),
    "overlap_factor": (0.94, 1.06),
    "helical_ply_mm": (0.55, 1.20),
    "transition_ply_mm": (0.45, 1.05),
    "hoop_ply_mm": (0.90, 2.20),
    "coverage_epsilon": (0.20, 0.35),
    "coverage_transition_width_mm": (6.0, 18.0),
}

DOME_SHAPES = ["hemispherical", "isotensoid_like", "variable_contour"]
TOW_WIDTHS = [6.0, 6.35]
PRESSURE_SCHEDULE = {
    "train": [70.0],
    "holdout": [70.0],
    "boundary": [70.0],
    "pressure_scale": [52.5, 87.5],
}


def latin_hypercube(n: int, d: int, rng: random.Random) -> list[list[float]]:
    samples = [[0.0 for _ in range(d)] for _ in range(n)]
    for j in range(d):
        values = []
        for i in range(n):
            lo = i / n
            hi = (i + 1) / n
            values.append(lo + rng.random() * (hi - lo))
        rng.shuffle(values)
        for i, value in enumerate(values):
            samples[i][j] = value
    return samples


def spherical_cap_volume_mm3(r_inner: float, boss_r: float, dome_factor: float) -> float:
    dome_r = max(r_inner * dome_factor, r_inner * 1.0001)
    boss_r = min(boss_r, 0.92 * r_inner)
    phi_boss = math.acos(max(-0.999999, min(0.999999, boss_r / dome_r)))
    phi_join = math.acos(max(-0.999999, min(0.999999, r_inner / dome_r)))

    def primitive(phi: float) -> float:
        s = math.sin(phi)
        return s - (s**3) / 3.0

    return math.pi * dome_r**3 * (primitive(phi_boss) - primitive(phi_join))


def cylindrical_length_for_volume(r_inner: float, boss_r: float, dome_factor: float, target_volume_mm3: float) -> float:
    two_caps = 2.0 * spherical_cap_volume_mm3(r_inner, boss_r, dome_factor)
    cylinder_volume = target_volume_mm3 - two_caps
    if cylinder_volume <= 0.0:
        raise ValueError("Target volume is smaller than the two dome caps.")
    return cylinder_volume / (math.pi * r_inner**2)


def internal_volume_l(r_inner: float, length: float, boss_r: float, dome_factor: float) -> float:
    cylinder = math.pi * r_inner**2 * length
    caps = 2.0 * spherical_cap_volume_mm3(r_inner, boss_r, dome_factor)
    return (cylinder + caps) / 1_000_000.0


def map_row(row: list[float], keys: list[str]) -> dict[str, float]:
    return {
        key: RANGES[key][0] + row[i] * (RANGES[key][1] - RANGES[key][0])
        for i, key in enumerate(keys)
    }


def role_for_index(i: int, total: int) -> str:
    if i >= int(0.9375 * total):
        return "pressure_scale"
    if i >= int(0.8750 * total):
        return "boundary"
    if i >= int(0.7500 * total):
        return "holdout"
    return "train"


def apply_boundary_focus(values: dict[str, float], idx: int) -> None:
    knobs = [
        ("boss_radius_ratio", idx & 1),
        ("dome_radius_factor", (idx >> 1) & 1),
        ("helical_angle_deg", (idx >> 2) & 1),
        ("coverage_transition_width_mm", (idx >> 3) & 1),
    ]
    for key, high in knobs:
        lo, hi = RANGES[key]
        values[key] = hi if high else lo


def make_case(case_number: int, row: list[float], role: str, pressure_index: int) -> dict:
    keys = list(RANGES.keys())
    values = map_row(row, keys)
    if role == "boundary":
        apply_boundary_focus(values, case_number)

    r_inner = round(values["inner_radius_mm"], 2)
    boss_r = round(values["boss_radius_ratio"] * r_inner, 2)
    dome_factor = round(values["dome_radius_factor"], 3)
    length = cylindrical_length_for_volume(r_inner, boss_r, dome_factor, TARGET_VOLUME_MM3)
    length = round(length, 2)
    volume_l = internal_volume_l(r_inner, length, boss_r, dome_factor)

    if not 2.5 <= length / r_inner <= 7.6:
        raise ValueError(f"Unphysical length/radius ratio for case {case_number}: {length / r_inner:.3f}")
    if abs(volume_l - TARGET_VOLUME_L) > 0.003:
        raise ValueError(f"Volume drift for case {case_number}: {volume_l:.5f} L")

    role_pressures = PRESSURE_SCHEDULE[role]
    pressure = role_pressures[pressure_index % len(role_pressures)]
    helical_angle = round(values["helical_angle_deg"], 2)
    transition_angle = round(values["transition_angle_deg"], 2)
    hoop_angle = round(values["hoop_angle_deg"], 2)
    tow_width = TOW_WIDTHS[case_number % len(TOW_WIDTHS)]
    helical_pairs = 4 + (case_number % 3 == 0)
    transition_pairs = 1 + (case_number % 4 != 0)
    hoop_pairs = 2 + ((case_number // 3) % 2)

    return {
        "case_id": f"11l_{case_number:04d}",
        "campaign": {
            "name": "DOE-B_11L_fixed_volume",
            "role": role,
            "target_volume_l": TARGET_VOLUME_L,
            "computed_internal_volume_l": round(volume_l, 6),
            "volume_method": "cylinder plus two spherical-cap dome volumes truncated at boss radius",
            "ml_readiness": "MLP scalar case; PINN requires field-level exports from CalculiX",
        },
        "loading": {"pressure_mpa": pressure},
        "geometry": {
            "inner_radius_mm": r_inner,
            "cylindrical_length_mm": length,
            "boss_radius_mm": boss_r,
            "dome_shape": DOME_SHAPES[case_number % len(DOME_SHAPES)],
            "dome_radius_factor": dome_factor,
        },
        "winding": {
            "helical_angles_deg": [helical_angle],
            "transition_angles_deg": [transition_angle],
            "hoop_angle_deg": hoop_angle,
            "helical_pairs": int(helical_pairs),
            "transition_pairs": int(transition_pairs),
            "hoop_pairs": int(hoop_pairs),
            "tow_width_mm": tow_width,
            "tow_thickness_mm": 0.125,
            "overlap_factor": round(values["overlap_factor"], 3),
            "coverage_model": "smooth_logistic_turnaround",
            "coverage_epsilon": round(values["coverage_epsilon"], 4),
            "coverage_transition_width_mm": round(values["coverage_transition_width_mm"], 2),
        },
        "thickness": {
            "helical_ply_mm": round(values["helical_ply_mm"], 3),
            "transition_ply_mm": round(values["transition_ply_mm"], 3),
            "hoop_ply_mm": round(values["hoop_ply_mm"], 3),
        },
        "materials": {
            "composite": "carbon_epoxy_t700_like",
            "liner": "pa6_liner",
            "boss": "aluminum_6061_t6",
        },
        "initial_conditions": {
            "stress": "zero",
            "displacement": "zero",
            "thermal_field": "none",
            "winding_residual_stress": "not_included",
            "damage_state": "undamaged_linear_elastic",
        },
        "boundary_conditions": {
            "axial_constraint": "single_boss_reference",
            "left_boss": "U3=0",
            "right_boss": "axially free; no U3 constraint on RIGHT_BOSS",
            "pin_a": "U1=U2=0",
            "pin_b": "U2=0",
            "contact": "not_active_in_main_doe",
            "rationale": (
                "bibliography-aligned reference/sliding support: one boss fixes the axial datum, "
                "the opposite boss is not axially clamped, and pins remove residual rigid-body modes"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed-volume 11 L DOE-B campaign.")
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    if args.case_count < 256:
        raise SystemExit("For this MLP/PINN-oriented campaign, keep at least 256 cases.")

    rng = random.Random(args.seed)
    keys = list(RANGES.keys())
    lhs = latin_hypercube(args.case_count, len(keys), rng)
    cases = []
    pressure_counter = 0
    for i, row in enumerate(lhs, start=1):
        role = role_for_index(i - 1, args.case_count)
        if role == "pressure_scale":
            pressure_counter += 1
        cases.append(make_case(i, row, role, pressure_counter))

    role_counts: dict[str, int] = {}
    volumes = []
    length_ratios = []
    for case in cases:
        role = case["campaign"]["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
        volumes.append(case["campaign"]["computed_internal_volume_l"])
        length_ratios.append(case["geometry"]["cylindrical_length_mm"] / case["geometry"]["inner_radius_mm"])

    out = {
        "schema_version": 2,
        "description": (
            "DOE-B fixed-volume 11 L campaign generated after numerical-review feedback. "
            "All cases are 11 L internal-volume containers; cylindrical length is derived, not sampled freely."
        ),
        "seed": args.seed,
        "case_count": len(cases),
        "target_volume_l": TARGET_VOLUME_L,
        "role_counts": role_counts,
        "volume_l_min": min(volumes),
        "volume_l_max": max(volumes),
        "length_radius_ratio_min": min(length_ratios),
        "length_radius_ratio_max": max(length_ratios),
        "professor_feedback_response": [
            "Initial and boundary conditions are explicit in each case.",
            "The main axial boundary condition is single_boss_reference, not a two-boss axial clamp.",
            "The container volume is fixed to 11 L; geometry variations preserve volume.",
            "The DOE is separated from the historical broad DOE to avoid mixing calibration domains.",
            "Smooth dome coverage parameters are sampled with a nonzero numerical floor before retraining the MLP.",
            "The campaign is appropriate for an MLP correction; a PINN requires additional field-level exports.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps({k: out[k] for k in ["case_count", "role_counts", "volume_l_min", "volume_l_max"]}, indent=2))


if __name__ == "__main__":
    main()
