from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
GENETIC_ROOT = ROOT.parent
CONFIG = ROOT / "config"
CASES_FILE = Path(os.environ.get("TYPE4_CASES_FILE", CONFIG / "doe_cases.json"))
OUTPUTS = Path(os.environ.get("TYPE4_OUTPUTS_DIR", ROOT / "outputs"))
PYTHON_RESULTS = OUTPUTS / "python_results"
SIM_RESULTS = OUTPUTS / "simulation_results"
COMPARISON = OUTPUTS / "comparison_tables"
MODELS = OUTPUTS / "correction_models"
FIGURES = OUTPUTS / "figures"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    for path in [PYTHON_RESULTS, SIM_RESULTS, COMPARISON, MODELS, FIGURES, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def add_genetic_to_path() -> None:
    path = str(GENETIC_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    geom = case["geometry"]
    load = case["loading"]
    wind = case["winding"]
    thick = case.get("thickness", {})
    helical_angles = [float(a) for a in wind.get("helical_angles_deg", [])]
    transition_angles = [float(a) for a in wind.get("transition_angles_deg", [])]
    explicit_plies = wind.get("explicit_plies", [])
    layup = make_layup(case) if explicit_plies else []
    helical_like = [
        abs(float(ply["angle_deg"]))
        for ply in layup
        if abs(float(ply["angle_deg"])) < 80.0
    ]
    hoop_like = [
        abs(float(ply["angle_deg"]))
        for ply in layup
        if abs(float(ply["angle_deg"])) >= 80.0
    ]
    return {
        "case_id": case["case_id"],
        "pressure_mpa": load["pressure_mpa"],
        "inner_radius_mm": geom["inner_radius_mm"],
        "cylindrical_length_mm": geom["cylindrical_length_mm"],
        "boss_radius_mm": geom["boss_radius_mm"],
        "dome_shape": geom["dome_shape"],
        "dome_radius_factor": geom.get("dome_radius_factor", 1.0),
        "helical_angle_min_deg": min(abs(a) for a in helical_angles) if helical_angles else (min(helical_like) if helical_like else 0.0),
        "helical_angle_max_deg": max(abs(a) for a in helical_angles) if helical_angles else (max(helical_like) if helical_like else 0.0),
        "transition_angle_max_deg": max(abs(a) for a in transition_angles) if transition_angles else 0.0,
        "hoop_angle_deg": wind.get("hoop_angle_deg", max(hoop_like) if hoop_like else 90.0),
        "helical_pairs": wind.get("helical_pairs", 0),
        "transition_pairs": wind.get("transition_pairs", 0),
        "hoop_pairs": wind.get("hoop_pairs", 0),
        "tow_width_mm": wind.get("tow_width_mm", 0.0),
        "tow_thickness_mm": wind.get("tow_thickness_mm", 0.0),
        "overlap_factor": wind.get("overlap_factor", 1.0),
        "helical_ply_mm": thick.get("helical_ply_mm", 0.0),
        "transition_ply_mm": thick.get("transition_ply_mm", 0.0),
        "hoop_ply_mm": thick.get("hoop_ply_mm", 0.0),
        "composite_material": case["materials"]["composite"],
        "liner_material": case["materials"]["liner"],
        "boss_material": case["materials"]["boss"],
    }


def make_layup(case: dict[str, Any]) -> list[dict[str, Any]]:
    wind = case["winding"]
    if wind.get("explicit_plies"):
        rows: list[dict[str, Any]] = []
        ply = 1
        for block_idx, block in enumerate(wind["explicit_plies"], start=1):
            count = int(block["count"])
            if count <= 0:
                continue
            thickness = float(block["thickness_mm"])
            base_name = str(block.get("name", f"block_{block_idx}"))
            angle = float(block["angle_deg"])
            if bool(block.get("signed_pair", False)):
                # Agne et al. report stacks such as +/-42_28.  Here the suffix
                # is treated as total signed plies, not as 28 plus/minus pairs.
                plus_count = count // 2 + count % 2
                minus_count = count // 2
                signed_blocks = [(angle, "plus", plus_count), (-angle, "minus", minus_count)]
                for signed_angle, label, signed_count in signed_blocks:
                    for rep in range(signed_count):
                        row = {
                            "ply": ply,
                            "name": f"{base_name}_{rep + 1}_{label}",
                            "angle_deg": signed_angle,
                            "thickness_mm": thickness,
                        }
                        for meta_key in (
                            "family_id",
                            "material",
                            "zone_materials",
                            "turnaround_radius_mm",
                            "coverage_zones",
                            "dome_angle_policy",
                            "dropoff_at_turnaround",
                            "coverage_model",
                            "coverage_epsilon",
                            "coverage_transition_width_mm",
                            "coverage_outside_zones_weight",
                        ):
                            if meta_key in block:
                                row[meta_key] = block[meta_key]
                        rows.append(row)
                        ply += 1
            else:
                for rep in range(count):
                    row = {
                        "ply": ply,
                        "name": f"{base_name}_{rep + 1}",
                        "angle_deg": angle,
                        "thickness_mm": thickness,
                    }
                    for meta_key in (
                        "family_id",
                        "material",
                        "zone_materials",
                        "turnaround_radius_mm",
                        "coverage_zones",
                        "dome_angle_policy",
                        "dropoff_at_turnaround",
                        "coverage_model",
                        "coverage_epsilon",
                        "coverage_transition_width_mm",
                        "coverage_outside_zones_weight",
                    ):
                        if meta_key in block:
                            row[meta_key] = block[meta_key]
                    rows.append(row)
                    ply += 1
        return rows

    thick = case["thickness"]
    rows: list[dict[str, Any]] = []
    ply = 1

    def add_pair(base_name: str, angle: float, thickness: float, pair_idx: int) -> None:
        nonlocal ply
        for sign, label in [(1.0, "plus"), (-1.0, "minus")]:
            rows.append(
                {
                    "ply": ply,
                    "name": f"{base_name}_{pair_idx}_{label}",
                    "angle_deg": sign * float(angle),
                    "thickness_mm": float(thickness),
                }
            )
            ply += 1

    for rep in range(int(wind["helical_pairs"])):
        for angle in wind["helical_angles_deg"]:
            add_pair("helical", float(angle), thick["helical_ply_mm"], rep + 1)
    for rep in range(int(wind["transition_pairs"])):
        for angle in wind["transition_angles_deg"]:
            add_pair("transition", float(angle), thick["transition_ply_mm"], rep + 1)
    for rep in range(int(wind["hoop_pairs"])):
        add_pair("hoop", float(wind["hoop_angle_deg"]), thick["hoop_ply_mm"], rep + 1)
    return rows


def ply_material_name(case: dict[str, Any], ply: dict[str, Any]) -> str:
    return str(ply.get("material") or case["materials"]["composite"])


def ply_material_name_for_zone(case: dict[str, Any], ply: dict[str, Any], zone: str) -> str:
    """Return a ply material, with optional zone-specific material states."""
    zone_materials = ply.get("zone_materials")
    if isinstance(zone_materials, dict):
        if zone in zone_materials:
            return str(zone_materials[zone])
        if zone in {"left_dome", "right_dome", "boss"} and "dome" in zone_materials:
            return str(zone_materials["dome"])
        if zone in {"left_dome", "right_dome", "junction", "boss"} and "dome_or_junction" in zone_materials:
            return str(zone_materials["dome_or_junction"])
    return ply_material_name(case, ply)


def case_materials(base_materials: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Merge database materials with optional case-local material overrides."""
    overrides = case.get("material_overrides", {})
    out = dict(base_materials)
    composites = dict(base_materials.get("composites", {}))
    composites.update(overrides.get("composites", {}))
    out["composites"] = composites
    if "liners" in overrides:
        liners = dict(base_materials.get("liners", {}))
        liners.update(overrides.get("liners", {}))
        out["liners"] = liners
    if "bosses" in overrides:
        bosses = dict(base_materials.get("bosses", {}))
        bosses.update(overrides.get("bosses", {}))
        out["bosses"] = bosses
    return out


def total_thickness(case: dict[str, Any]) -> float:
    return sum(float(layer["thickness_mm"]) for layer in make_layup(case))


def smooth_coverage_enabled(case: dict[str, Any]) -> bool:
    wind = case.get("winding", {})
    model = str(wind.get("coverage_model", wind.get("dome_coverage_model", ""))).lower()
    return "smooth" in model or "taper" in model


def coverage_epsilon(case: dict[str, Any]) -> float:
    wind = case.get("winding", {})
    value = float(wind.get("coverage_epsilon", wind.get("coverage_min_weight", 0.06)))
    return max(1e-4, min(0.40, value))


def coverage_transition_width_mm(case: dict[str, Any], ply: dict[str, Any]) -> float:
    wind = case.get("winding", {})
    if "coverage_transition_width_mm" in ply:
        return max(0.50, float(ply["coverage_transition_width_mm"]))
    if "coverage_transition_width_mm" in wind:
        return max(0.50, float(wind["coverage_transition_width_mm"]))
    tow_width = float(wind.get("tow_width_mm", 6.0) or 6.0)
    return max(4.0, 1.5 * tow_width)


def stable_logistic(x: float) -> float:
    if x >= 50.0:
        return 1.0
    if x <= -50.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def ply_turnaround_radius_mm(case: dict[str, Any] | None, ply: dict[str, Any]) -> float | None:
    if ply.get("turnaround_radius_mm") not in (None, ""):
        return float(ply["turnaround_radius_mm"])
    if case is None:
        return None
    geom = case.get("geometry", {})
    wind = case.get("winding", {})
    r_inner = float(geom.get("inner_radius_mm", 0.0) or 0.0)
    angle = abs(float(ply.get("angle_deg", 0.0)))
    if angle < 80.0:
        return r_inner * math.sin(math.radians(angle))
    if smooth_coverage_enabled(case):
        if wind.get("hoop_turnaround_radius_mm") not in (None, ""):
            return float(wind["hoop_turnaround_radius_mm"])
        return 0.985 * r_inner
    return None


def local_ply_angle_deg(
    ply: dict[str, Any],
    zone: str,
    radius_mm: float,
    default_geodesic_angle_deg: float,
    case: dict[str, Any] | None = None,
) -> float:
    nominal = float(ply["angle_deg"])
    if zone == "cylinder" or abs(nominal) >= 80.0:
        return nominal
    turnaround = ply_turnaround_radius_mm(case, ply)
    if turnaround is None:
        return math.copysign(float(default_geodesic_angle_deg), nominal)
    ratio = max(-0.999, min(0.999, float(turnaround) / max(radius_mm, float(turnaround) + 1e-6)))
    alpha_geo = math.degrees(math.asin(ratio))
    if zone == "junction":
        alpha_geo = 0.50 * abs(nominal) + 0.50 * alpha_geo
    return math.copysign(alpha_geo, nominal)


def ply_coverage_weight(
    ply: dict[str, Any],
    zone: str,
    radius_mm: float,
    case: dict[str, Any] | None = None,
) -> float:
    coverage = ply.get("coverage_zones")
    smooth = case is not None and smooth_coverage_enabled(case)
    eps = coverage_epsilon(case) if case is not None else 0.0
    if coverage and zone not in set(str(item) for item in coverage):
        if "coverage_outside_zones_weight" in ply:
            return max(0.0, float(ply["coverage_outside_zones_weight"]))
        return eps if smooth else 0.0
    if zone == "cylinder":
        return 1.0
    turnaround = ply_turnaround_radius_mm(case, ply)
    if not smooth:
        if bool(ply.get("dropoff_at_turnaround", False)) and turnaround is not None:
            return 0.0 if radius_mm + 1e-9 < float(turnaround) else 1.0
        return 1.0
    if turnaround is None:
        return 1.0
    width = coverage_transition_width_mm(case, ply)
    raw = stable_logistic((radius_mm - float(turnaround)) / width)
    return max(eps, min(1.0, eps + (1.0 - eps) * raw))


def ply_is_active(ply: dict[str, Any], zone: str, radius_mm: float, case: dict[str, Any] | None = None) -> bool:
    return ply_coverage_weight(ply, zone, radius_mm, case) > 1e-12


def active_layup(
    case: dict[str, Any],
    zone: str,
    radius_mm: float,
    layup: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    base = layup if layup is not None else make_layup(case)
    smooth = smooth_coverage_enabled(case)
    rows: list[dict[str, Any]] = []
    for ply in base:
        weight = ply_coverage_weight(ply, zone, radius_mm, case)
        if weight <= 1e-12:
            continue
        if smooth:
            local = dict(ply)
            local["nominal_thickness_mm"] = float(ply["thickness_mm"])
            local["coverage_weight"] = weight
            local["thickness_mm"] = float(ply["thickness_mm"]) * weight
            rows.append(local)
        else:
            rows.append(ply)
    return rows or base


def safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return math.nan
    return num / den


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    cases_path = Path(path) if path is not None else CASES_FILE
    data = read_json(cases_path)
    return data["cases"]


def load_materials() -> dict[str, Any]:
    return read_json(CONFIG / "material_database.json")

