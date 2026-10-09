#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_utils import (  # noqa: E402
    SIM_RESULTS,
    active_layup,
    case_materials,
    ensure_dirs,
    load_cases,
    load_materials,
    local_ply_angle_deg,
    make_layup,
    ply_material_name_for_zone,
    write_csv,
    write_json,
)

FLOAT_RE = re.compile(r"[+-]?\d\.\d{5}E[+-]\d{3}")


def frd_blocks(path: Path, block_name: str) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(" -4") and block_name in line:
            start = i + 1
            break
    if start is None:
        return []
    out: list[str] = []
    for line in lines[start:]:
        if line.startswith(" -4") or line.startswith(" -3"):
            if out:
                break
            continue
        if line.startswith(" -1"):
            out.append(line)
    return out


def parse_result_id(line: str) -> int | None:
    try:
        return int(line[3:13])
    except Exception:
        return None


def parse_floats(line: str) -> list[float]:
    # FRD result rows pack the node/value fields without a separator, e.g.
    # " -1    1199291.00872E-001...".  The first 13 chars contain the -1 flag
    # and node id; parsing the full line would merge the node id into value 1.
    payload = line[13:] if line.startswith(" -1") else line
    return [float(value) for value in FLOAT_RE.findall(payload)]


def parse_frd_nodes(path: Path) -> dict[int, tuple[float, float, float]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    if not path.exists():
        return nodes
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if line.startswith(" -3"):
            break
        if not line.startswith(" -1"):
            continue
        nid = parse_result_id(line)
        vals = parse_floats(line)
        if nid is not None and len(vals) == 3:
            nodes[nid] = (vals[0], vals[1], vals[2])
    return nodes


def parse_displacements(frd: Path) -> dict[str, float]:
    max_mag = 0.0
    count = 0
    for line in frd_blocks(frd, "DISP"):
        vals = parse_floats(line)
        if len(vals) < 3:
            continue
        ux, uy, uz = vals[:3]
        max_mag = max(max_mag, math.sqrt(ux * ux + uy * uy + uz * uz))
        count += 1
    return {"frd_displacement_node_count": count, "max_displacement_mm": max_mag}


def parse_stresses(frd: Path) -> dict[str, float]:
    max_vm = 0.0
    count = 0
    for line in frd_blocks(frd, "STRESS"):
        vals = parse_floats(line)
        if len(vals) < 6:
            continue
        sxx, syy, szz, sxy, syz, sxz = vals[:6]
        vm = math.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy * sxy + syz * syz + sxz * sxz)
        )
        max_vm = max(max_vm, vm)
        count += 1
    return {"frd_stress_value_count": count, "max_von_mises_mpa": max_vm}


def parse_stress_rows(frd: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for line in frd_blocks(frd, "STRESS"):
        nid = parse_result_id(line)
        vals = parse_floats(line)
        if nid is None or len(vals) < 6:
            continue
        rows.append(
            {
                "node_id": nid,
                "sxx": vals[0],
                "syy": vals[1],
                "szz": vals[2],
                "sxy": vals[3],
                "syz": vals[4],
                "sxz": vals[5],
            }
        )
    return rows


def count_dat_integration_point_stresses(dat: Path) -> dict[str, float | int | str]:
    if not dat.exists():
        return {"dat_integration_point_output": "missing", "dat_stress_line_count": 0}
    text = dat.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_stress = False
    count = 0
    max_vm = 0.0
    for line in text:
        low = line.lower()
        if "stresses" in low and ("integration point" in low or "integ" in low):
            in_stress = True
            continue
        if in_stress and line.strip() == "":
            continue
        vals = [float(value) for value in FLOAT_RE.findall(line)]
        if in_stress and len(vals) >= 6:
            sxx, syy, szz, sxy, syz, sxz = vals[:6]
            vm = math.sqrt(
                0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                + 3.0 * (sxy * sxy + syz * syz + sxz * sxz)
            )
            max_vm = max(max_vm, vm)
            count += 1
    return {
        "dat_integration_point_output": "present" if count else "not_detected",
        "dat_stress_line_count": count,
        "dat_max_von_mises_mpa": max_vm,
    }


def orientation_stats(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    rows = []
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    angles = [float(row["geodesic_angle_deg"]) for row in rows]
    thicknesses = [float(row["thickness_mm"]) for row in rows]
    zones = sorted({row["zone"] for row in rows})
    return {
        "element_count": len(rows),
        "min_local_angle_deg": min(angles),
        "max_local_angle_deg": max(angles),
        "min_thickness_mm": min(thicknesses),
        "max_thickness_mm": max(thicknesses),
        "zones": ",".join(zones),
    }


def load_orientation_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        x = float(row["x_mm"])
        y = float(row["y_mm"])
        z = float(row["z_mm"])
        theta = math.atan2(y, x)
        out.append(
            {
                "element_id": int(row["element_id"]),
                "x": x,
                "y": y,
                "z": z,
                "r": float(row["radius_mm"]),
                "theta": theta,
                "zone": row["zone"],
                "geodesic_angle_deg": float(row["geodesic_angle_deg"]),
                "thickness_factor": float(row["thickness_factor"]),
                "thickness_mm": float(row["thickness_mm"]),
                "tow_coverage": float(row["tow_coverage"]),
            }
        )
    add_local_bases(out)
    return out


def add_local_bases(rows: list[dict]) -> None:
    by_theta: dict[int, list[dict]] = {}
    for row in rows:
        key = round((math.degrees(row["theta"]) % 360.0) * 1000)
        by_theta.setdefault(key, []).append(row)
        row["circumferential"] = unit((-math.sin(row["theta"]), math.cos(row["theta"]), 0.0))
    for group in by_theta.values():
        group.sort(key=lambda item: item["z"])
        for idx, row in enumerate(group):
            if len(group) == 1:
                tangent = (0.0, 0.0, 1.0)
            elif idx == 0:
                tangent = sub(center(group[idx + 1]), center(row))
            elif idx == len(group) - 1:
                tangent = sub(center(row), center(group[idx - 1]))
            else:
                tangent = sub(center(group[idx + 1]), center(group[idx - 1]))
            row["meridional"] = unit(tangent)


def center(row: dict) -> tuple[float, float, float]:
    return (row["x"], row["y"], row["z"])


def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def tensor_vec(stress: dict[str, float | int], v: tuple[float, float, float]) -> tuple[float, float, float]:
    sxx = float(stress["sxx"])
    syy = float(stress["syy"])
    szz = float(stress["szz"])
    sxy = float(stress["sxy"])
    syz = float(stress["syz"])
    sxz = float(stress["sxz"])
    return (
        sxx * v[0] + sxy * v[1] + sxz * v[2],
        sxy * v[0] + syy * v[1] + syz * v[2],
        sxz * v[0] + syz * v[1] + szz * v[2],
    )


def project_stress(
    stress: dict[str, float | int],
    meridional: tuple[float, float, float],
    circumferential: tuple[float, float, float],
    angle_deg: float,
) -> tuple[float, float, float]:
    alpha = math.radians(angle_deg)
    fiber = unit(add(scale(meridional, math.cos(alpha)), scale(circumferential, math.sin(alpha))))
    transverse = unit(add(scale(meridional, -math.sin(alpha)), scale(circumferential, math.cos(alpha))))
    sigma_11 = dot(fiber, tensor_vec(stress, fiber))
    sigma_22 = dot(transverse, tensor_vec(stress, transverse))
    tau_12 = dot(fiber, tensor_vec(stress, transverse))
    return sigma_11, sigma_22, tau_12


def nearest_element_index(
    point: tuple[float, float, float],
    orientation_rows: list[dict],
    last_index: int | None = None,
) -> int:
    x, y, z = point
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    if last_index is not None:
        candidates = range(max(0, last_index - 48), min(len(orientation_rows), last_index + 49))
    else:
        candidates = range(len(orientation_rows))
    best_idx = 0
    best_d2 = float("inf")
    for idx in candidates:
        row = orientation_rows[idx]
        dtheta = abs((theta - row["theta"] + math.pi) % (2.0 * math.pi) - math.pi)
        d2 = (z - row["z"]) ** 2 + (r - row["r"]) ** 2 + (row["r"] * dtheta) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = idx
    if best_d2 > 400.0 and last_index is not None:
        return nearest_element_index(point, orientation_rows, None)
    return best_idx


def tsai_wu(s1: float, s2: float, t12: float, mat: dict) -> float:
    xt = float(mat["Xt_mpa"])
    xc = float(mat["Xc_mpa"])
    yt = float(mat["Yt_mpa"])
    yc = float(mat["Yc_mpa"])
    s = float(mat["X12_mpa"])
    f1 = 1.0 / xt - 1.0 / xc
    f2 = 1.0 / yt - 1.0 / yc
    f11 = 1.0 / (xt * xc)
    f22 = 1.0 / (yt * yc)
    f66 = 1.0 / (s * s)
    f12 = -0.5 * math.sqrt(f11 * f22)
    return f1 * s1 + f2 * s2 + f11 * s1 * s1 + f22 * s2 * s2 + f66 * t12 * t12 + 2.0 * f12 * s1 * s2


def hashin(s1: float, s2: float, t12: float, mat: dict) -> float:
    xt = float(mat["Xt_mpa"])
    xc = float(mat["Xc_mpa"])
    yt = float(mat["Yt_mpa"])
    yc = float(mat["Yc_mpa"])
    s = float(mat["X12_mpa"])
    if s1 >= 0.0:
        fiber = (s1 / xt) ** 2 + (t12 / s) ** 2
    else:
        fiber = (s1 / xc) ** 2
    if s2 >= 0.0:
        matrix = (s2 / yt) ** 2 + (t12 / s) ** 2
    else:
        matrix = (s2 / (2.0 * s)) ** 2 + ((yc / (2.0 * s)) ** 2 - 1.0) * (s2 / yc) + (t12 / s) ** 2
    return max(fiber, matrix)


def hashin_mode_parts(s1: float, s2: float, t12: float, mat: dict) -> dict[str, float]:
    xt = float(mat["Xt_mpa"])
    xc = float(mat["Xc_mpa"])
    yt = float(mat["Yt_mpa"])
    yc = float(mat["Yc_mpa"])
    s = float(mat["X12_mpa"])
    fiber_tension = (s1 / xt) ** 2 + (t12 / s) ** 2 if s1 >= 0.0 else 0.0
    fiber_compression = (s1 / xc) ** 2 if s1 < 0.0 else 0.0
    matrix_tension = (s2 / yt) ** 2 + (t12 / s) ** 2 if s2 >= 0.0 else 0.0
    matrix_compression = (
        (s2 / (2.0 * s)) ** 2
        + ((yc / (2.0 * s)) ** 2 - 1.0) * (s2 / yc)
        + (t12 / s) ** 2
        if s2 < 0.0
        else 0.0
    )
    return {
        "fiber_stress_ratio_abs": abs(s1) / (xt if s1 >= 0.0 else xc),
        "fiber_tension_hashin": fiber_tension,
        "fiber_compression_hashin": fiber_compression,
        "matrix_tension_hashin": matrix_tension,
        "matrix_compression_hashin": matrix_compression,
        "matrix_stress_ratio_abs": abs(s2) / (yt if s2 >= 0.0 else yc),
        "shear_ratio_abs": abs(t12) / s,
    }


def puck_simplified(s1: float, s2: float, t12: float, mat: dict) -> float:
    xt = float(mat["Xt_mpa"])
    xc = float(mat["Xc_mpa"])
    yt = float(mat["Yt_mpa"])
    yc = float(mat["Yc_mpa"])
    s = float(mat["X12_mpa"])
    fiber = s1 / xt if s1 >= 0.0 else -s1 / xc
    if s2 >= 0.0:
        matrix = math.sqrt((s2 / yt) ** 2 + (t12 / s) ** 2)
    else:
        matrix = 0.65 * (-s2 / yc) + math.sqrt(0.35 * (s2 / yc) ** 2 + (t12 / s) ** 2)
    return max(fiber, matrix)


def percentile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = max(0.0, min(1.0, q)) * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def boundary_epsilon_mm(element: dict, layup: list[dict], cap_mm: float = 0.18) -> float:
    factor = float(element.get("thickness_factor", 1.0))
    if not layup:
        return cap_mm
    min_ply = min(float(ply["thickness_mm"]) * factor for ply in layup)
    return max(1e-4, min(cap_mm, 0.20 * min_ply))


def compute_ply_boundary_radii(element: dict, layup: list[dict]) -> list[float]:
    """Return r_3d positions of ply interfaces AND shell surfaces in the 3D-expanded shell.

    Includes r_inner_3d and r_outer_3d so that section-point nodes near the free
    surfaces (which carry bending-enhanced stresses absent from the Lekhnitskii
    reference model) are filtered by the same boundary-epsilon check used for
    interior ply interfaces.
    """
    t = element["thickness_mm"]
    r_inner_3d = element["r"] - t / 2.0     # OFFSET=0 → mid-surface at mesh nodes
    r_outer_3d = element["r"] + t / 2.0
    boundaries: list[float] = [r_inner_3d, r_outer_3d]
    cumulative = 0.0
    for ply in reversed(layup):
        cumulative += float(ply["thickness_mm"]) * float(element.get("thickness_factor", 1.0))
        if cumulative < t - 1e-9:
            boundaries.append(r_inner_3d + cumulative)
    return boundaries


def assign_ply_from_thickness_position(
    r_3d: float,
    element: dict,
    layup: list[dict],
) -> dict:
    """Map a 3D-expanded FRD node to its ply by through-thickness position.

    With *EL FILE,OUTPUT=3D the shell is expanded symmetrically around the
    reference surface (mesh node radius), from r-t/2 to r+t/2.  CalculiX
    stacks plies in REVERSED order in this expansion: the last ply in the
    layup definition is innermost (closest to r-t/2), the first ply is
    outermost (closest to r+t/2).
    """
    t = element["thickness_mm"]
    r_ref = element["r"]
    r_inner_3d = r_ref - t / 2.0       # OFFSET=0 → expansion symmetric about mesh nodes
    t_from_inner = r_3d - r_inner_3d
    t_frac = max(0.0, min(1.0, t_from_inner / t)) if t > 0 else 0.0
    cumulative = 0.0
    for ply in reversed(layup):
        cumulative += float(ply["thickness_mm"]) * float(element.get("thickness_factor", 1.0))
        if cumulative / t >= t_frac:
            return ply
    return layup[0]


def local_ply_failure(
    case: dict,
    frd: Path,
    orientation_csv: Path,
    material: dict | None = None,
    materials: dict | None = None,
    max_samples: int = 200000,
) -> tuple[dict, list[dict], list[dict]]:
    nodes = parse_frd_nodes(frd)
    stress_rows = parse_stress_rows(frd)
    orientation_rows = load_orientation_rows(orientation_csv)
    layup = make_layup(case)
    material_db = case_materials(materials or {}, case)
    default_material = material or material_db.get("composites", {}).get(case["materials"]["composite"], {})
    if not nodes or not stress_rows or not orientation_rows:
        return {}, [], []
    step = max(1, len(stress_rows) // max_samples)
    sampled = stress_rows[::step]
    local_rows: list[dict] = []
    mode_rows: list[dict] = []
    last_idx: int | None = None

    # ------------------------------------------------------------------
    # Per-ply averaging (Option α) to align with the Python Lekhnitskii
    # reference, which evaluates one ply-mean stress per ply.  CalculiX
    # gives multiple section points through each ply; taking their max is
    # too conservative versus Python (which uses the radial mean over each
    # ply).  Here we group section-point stresses by (element_id, ply),
    # average the stress tensor components, then project and compute FI.
    # ------------------------------------------------------------------
    _BOUNDARY_EPSILON = 0.18  # mm — exclude duplicates at ply interfaces
    bucket_sum: dict[tuple, dict] = {}
    bucket_meta: dict[tuple, dict] = {}
    for stress in sampled:
        nid = int(stress["node_id"])
        point = nodes.get(nid)
        if point is None:
            continue
        idx = nearest_element_index(point, orientation_rows, last_idx)
        last_idx = idx
        element = orientation_rows[idx]
        r_3d = math.sqrt(point[0] * point[0] + point[1] * point[1])
        element_layup = active_layup(case, element["zone"], float(element["r"]), layup)
        eps = boundary_epsilon_mm(element, element_layup)
        if any(abs(r_3d - b) < eps for b in compute_ply_boundary_radii(element, element_layup)):
            continue
        ply = assign_ply_from_thickness_position(r_3d, element, element_layup)
        key = (element["element_id"], int(ply["ply"]))
        if key not in bucket_sum:
            bucket_sum[key] = {
                "sxx": 0.0, "syy": 0.0, "szz": 0.0,
                "sxy": 0.0, "syz": 0.0, "sxz": 0.0,
                "count": 0,
            }
            local_angle = local_ply_angle_deg(
                ply,
                element["zone"],
                float(element["r"]),
                float(element["geodesic_angle_deg"]),
                case,
            )
            bucket_meta[key] = {
                "element": element,
                "ply": ply,
                "local_angle": local_angle,
                "sample_node": (nid, point, r_3d),
            }
        b = bucket_sum[key]
        b["sxx"] += float(stress["sxx"]); b["syy"] += float(stress["syy"]); b["szz"] += float(stress["szz"])
        b["sxy"] += float(stress["sxy"]); b["syz"] += float(stress["syz"]); b["sxz"] += float(stress["sxz"])
        b["count"] += 1

    max_record: dict | None = None
    max_by_zone: dict[str, dict] = {}
    max_by_mode: dict[str, dict] = {}
    values_by_mode: dict[str, list[float]] = {}
    combined_values: list[float] = []
    for key, b in bucket_sum.items():
        n = b["count"]
        if n == 0:
            continue
        mean_stress = {
            "sxx": b["sxx"]/n, "syy": b["syy"]/n, "szz": b["szz"]/n,
            "sxy": b["sxy"]/n, "syz": b["syz"]/n, "sxz": b["sxz"]/n,
        }
        meta = bucket_meta[key]
        element = meta["element"]
        ply = meta["ply"]
        local_angle = meta["local_angle"]
        nid, point, r_3d = meta["sample_node"]
        s1, s2, t12 = project_stress(mean_stress, element["meridional"], element["circumferential"], local_angle)
        material_name = ply_material_name_for_zone(case, ply, element["zone"])
        ply_material = material_db.get("composites", {}).get(material_name, default_material)
        if not ply_material:
            raise KeyError(f"Unknown ply material: {material_name}")
        tw = tsai_wu(s1, s2, t12, ply_material)
        hs = hashin(s1, s2, t12, ply_material)
        pk = puck_simplified(s1, s2, t12, ply_material)
        parts = hashin_mode_parts(s1, s2, t12, ply_material)
        combined = max(tw, hs, pk)
        combined_values.append(combined)
        record = {
            "case_id": case["case_id"],
            "node_id": nid,
            "element_id": element["element_id"],
            "ply": int(ply["ply"]),
            "ply_name": ply["name"],
            "material": material_name,
            "zone": element["zone"],
            "x_mm": point[0],
            "y_mm": point[1],
            "z_mm": point[2],
            "radius_mm": r_3d,
            "local_angle_deg": local_angle,
            "sigma_11_mpa": s1,
            "sigma_22_mpa": s2,
            "tau_12_mpa": t12,
            "tsai_wu": tw,
            "hashin": hs,
            "puck": pk,
            "combined": combined,
            **parts,
            "section_point_count": n,
        }
        if max_record is None or combined > max_record["combined"]:
            max_record = record
        zone = element["zone"]
        if zone not in max_by_zone or combined > max_by_zone[zone]["combined"]:
            max_by_zone[zone] = record
        for mode, value in parts.items():
            values_by_mode.setdefault(mode, []).append(float(value))
            if mode not in max_by_mode or value > max_by_mode[mode][mode]:
                max_by_mode[mode] = {**record, "record_type": "mode_maximum", "metric": mode}
    if max_record:
        pressure = float(case.get("loading", {}).get("pressure_mpa", 0.0) or 0.0)
        local_rows.append(max_record)
        # Also emit per-zone critical records so reviewers can diagnose each zone
        # independently — the global max alone can hide a benign cylinder behind a
        # localized junction concentration.
        for zone_name in ("cylinder", "left_dome", "right_dome", "junction", "boss"):
            rec = max_by_zone.get(zone_name)
            if rec is not None and rec is not max_record:
                local_rows.append(rec)
        for mode in (
            "fiber_stress_ratio_abs",
            "fiber_tension_hashin",
            "fiber_compression_hashin",
            "matrix_tension_hashin",
            "matrix_compression_hashin",
            "matrix_stress_ratio_abs",
            "shear_ratio_abs",
        ):
            rec = max_by_mode.get(mode)
            if rec is not None:
                mode_rows.append(rec)
        fiber_rec = max_by_mode.get("fiber_stress_ratio_abs", {})
        fiber_ratio = float(fiber_rec.get("fiber_stress_ratio_abs", 0.0) or 0.0)
        fiber_hashin = max(
            float(max_by_mode.get("fiber_tension_hashin", {}).get("fiber_tension_hashin", 0.0) or 0.0),
            float(max_by_mode.get("fiber_compression_hashin", {}).get("fiber_compression_hashin", 0.0) or 0.0),
        )
        fiber_p95 = percentile(values_by_mode.get("fiber_stress_ratio_abs", []), 0.95)
        fiber_p99 = percentile(values_by_mode.get("fiber_stress_ratio_abs", []), 0.99)
        fiber_p995 = percentile(values_by_mode.get("fiber_stress_ratio_abs", []), 0.995)
        summary = {
            "local_failure_projection": "per_ply_section_mean_global_projection",
            "failure_sample_stride": step,
            "failure_stress_samples": len(sampled),
            "max_tsai_wu": max_record["tsai_wu"],
            "max_hashin": max_record["hashin"],
            "max_puck": max_record["puck"],
            "max_combined": max_record["combined"],
            "combined_p95": percentile(combined_values, 0.95),
            "combined_p99": percentile(combined_values, 0.99),
            "combined_p995": percentile(combined_values, 0.995),
            "max_combined_cylinder": max_by_zone.get("cylinder", {}).get("combined", 0.0),
            "max_combined_dome": max(
                max_by_zone.get("left_dome", {}).get("combined", 0.0),
                max_by_zone.get("right_dome", {}).get("combined", 0.0),
            ),
            "max_combined_junction": max_by_zone.get("junction", {}).get("combined", 0.0),
            "max_combined_boss": max_by_zone.get("boss", {}).get("combined", 0.0),
            "max_fiber_stress_ratio_abs": fiber_ratio,
            "fiber_proxy_pressure_mpa": pressure / fiber_ratio if fiber_ratio > 1e-12 else 0.0,
            "fiber_stress_ratio_p95_abs": fiber_p95,
            "fiber_stress_ratio_p99_abs": fiber_p99,
            "fiber_stress_ratio_p995_abs": fiber_p995,
            "fiber_proxy_pressure_p95_mpa": pressure / fiber_p95 if fiber_p95 > 1e-12 else 0.0,
            "fiber_proxy_pressure_p99_mpa": pressure / fiber_p99 if fiber_p99 > 1e-12 else 0.0,
            "fiber_proxy_pressure_p995_mpa": pressure / fiber_p995 if fiber_p995 > 1e-12 else 0.0,
            "max_fiber_hashin_mode": fiber_hashin,
            "fiber_proxy_zone": fiber_rec.get("zone", ""),
            "fiber_proxy_ply": fiber_rec.get("ply", ""),
            "fiber_proxy_ply_name": fiber_rec.get("ply_name", ""),
            "fiber_proxy_material": fiber_rec.get("material", ""),
            "fiber_proxy_local_angle_deg": fiber_rec.get("local_angle_deg", 0.0),
            "fiber_proxy_sigma_11_mpa": fiber_rec.get("sigma_11_mpa", 0.0),
            "fiber_proxy_tau_12_mpa": fiber_rec.get("tau_12_mpa", 0.0),
            "max_matrix_tension_hashin": max_by_mode.get("matrix_tension_hashin", {}).get("matrix_tension_hashin", 0.0),
            "max_matrix_compression_hashin": max_by_mode.get("matrix_compression_hashin", {}).get("matrix_compression_hashin", 0.0),
            "max_matrix_stress_ratio_abs": max_by_mode.get("matrix_stress_ratio_abs", {}).get("matrix_stress_ratio_abs", 0.0),
            "max_shear_ratio_abs": max_by_mode.get("shear_ratio_abs", {}).get("shear_ratio_abs", 0.0),
            "critical_zone": max_record["zone"],
            "critical_angle_deg": max_record["local_angle_deg"],
            "critical_ply": max_record["ply"],
            "critical_ply_name": max_record["ply_name"],
            "critical_material": max_record.get("material", ""),
            "critical_element_id": max_record["element_id"],
            "critical_node_id": max_record["node_id"],
            "sigma_11_mpa_at_critical": max_record["sigma_11_mpa"],
            "sigma_22_mpa_at_critical": max_record["sigma_22_mpa"],
            "tau_12_mpa_at_critical": max_record["tau_12_mpa"],
        }
        return summary, local_rows, mode_rows
    return {}, [], []


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process the CalculiX FRD and orientation fields.")
    parser.add_argument("--case-id", default="cal_001")
    args = parser.parse_args()

    ensure_dirs()
    case_dir = SIM_RESULTS / "calculix" / args.case_id
    manifest_path = case_dir / "calculix_case_manifest.json"
    run_status_path = case_dir / "calculix_run_status.json"
    render_status_path = case_dir / "paraview_render_status.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_status = json.loads(run_status_path.read_text(encoding="utf-8")) if run_status_path.exists() else {}
    render_status = json.loads(render_status_path.read_text(encoding="utf-8")) if render_status_path.exists() else {}
    inp = Path(manifest["inp"])
    frd = inp.with_suffix(".frd")
    stats = {
        "case_id": args.case_id,
        "base_case_id": manifest.get("base_case_id", manifest.get("case_id", args.case_id)),
        "output_id": manifest.get("output_id", args.case_id),
        "solver_status": run_status.get("solver_status", "not_run"),
        "render_status": render_status.get("render_status", "not_run"),
        "inp": str(inp),
        "frd": str(frd),
        "preview_vtk": manifest.get("preview_vtk", ""),
        "tow_paths_vtk": manifest.get("tow_paths_vtk", ""),
        "paraview_screenshot": render_status.get("screenshot", ""),
        "model_scope": manifest.get("model_scope", ""),
        "n_meridional": manifest.get("n_meridional", ""),
        "n_theta": manifest.get("n_theta", ""),
        "el_print_requested": manifest.get("el_print_requested", False),
        "certification_warning": "CalculiX demonstrator for research/pre-design only; not a certified hydrogen vessel simulation.",
    }
    stats.update(orientation_stats(Path(manifest["orientation_csv"])))
    if frd.exists():
        stats.update(parse_displacements(frd))
        stats.update(parse_stresses(frd))
        stats.update(count_dat_integration_point_stresses(inp.with_suffix(".dat")))
        cases = {case["case_id"]: case for case in load_cases()}
        base_case_id = manifest.get("base_case_id", manifest.get("case_id", args.case_id))
        case = cases.get(base_case_id)
        if case:
            materials = load_materials()
            failure_summary, critical_rows, mode_rows = local_ply_failure(case, frd, Path(manifest["orientation_csv"]), materials=materials)
            stats.update(failure_summary)
            if critical_rows:
                write_csv(case_dir / "calculix_critical_ply_projection.csv", critical_rows)
            if mode_rows:
                write_csv(case_dir / "calculix_mode_failure_projection.csv", mode_rows)
    write_json(case_dir / "calculix_summary.json", stats)
    write_csv(SIM_RESULTS / "calculix" / "calculix_summary.csv", [stats])
    print(case_dir / "calculix_summary.json")


if __name__ == "__main__":
    main()
