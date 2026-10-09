#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_utils import (  # noqa: E402
    SIM_RESULTS,
    active_layup,
    case_materials,
    ensure_dirs,
    flatten_case,
    load_cases,
    load_materials,
    local_ply_angle_deg,
    make_layup,
    ply_material_name_for_zone,
    write_json,
)


@dataclass(frozen=True)
class SurfacePoint:
    z: float
    r: float
    zone: str
    meridional_tangent: tuple[float, float]


@dataclass(frozen=True)
class ElementInfo:
    eid: int
    node_ids: tuple[int, int, int, int, int, int, int, int]
    corner_node_ids: tuple[int, int, int, int]
    center: tuple[float, float, float]
    theta: float
    r: float
    z: float
    zone: str
    angle_deg: float
    thickness_factor: float
    thickness_mm: float
    coverage: float
    meridional: tuple[float, float, float]
    circumferential: tuple[float, float, float]


def unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(x / n for x in v)  # type: ignore[return-value]


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def local_basis(theta: float, dzds: float, drds: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    e_theta = (-math.sin(theta), math.cos(theta), 0.0)
    e_meridional = (drds * math.cos(theta), drds * math.sin(theta), dzds)
    return unit(e_meridional), unit(e_theta)


def make_profile(case: dict, n_dome: int, n_cyl: int) -> list[SurfacePoint]:
    geom = case["geometry"]
    r_inner = float(geom["inner_radius_mm"])
    length = float(geom["cylindrical_length_mm"])
    boss_r = float(geom["boss_radius_mm"])
    dome_factor = float(geom.get("dome_radius_factor", 1.0))
    dome_R = max(r_inner * dome_factor, r_inner * 1.0001)
    boss_r = min(boss_r, 0.92 * r_inner)
    # Continuous spherical-cap profile: the previous implementation used
    # phi_join=0 for every dome factor, so cases with dome_R > r_inner jumped
    # from r=dome_R at the junction to r=r_inner in the cylinder.  The cap now
    # joins the cylinder exactly at r_inner while still allowing flatter domes.
    phi_boss = math.acos(max(-0.999999, min(0.999999, boss_r / dome_R)))
    phi_join = math.acos(max(-0.999999, min(0.999999, r_inner / dome_R)))
    left_join = -0.5 * length
    right_join = 0.5 * length

    pts: list[SurfacePoint] = []

    def append_unique(point: SurfacePoint) -> None:
        if pts and abs(pts[-1].z - point.z) < 1e-9 and abs(pts[-1].r - point.r) < 1e-9:
            return
        pts.append(point)

    for i in range(n_dome + 1):
        u = i / n_dome
        phi = phi_boss + (phi_join - phi_boss) * u
        r = dome_R * math.cos(phi)
        z = left_join - dome_R * (math.sin(phi) - math.sin(phi_join))
        dzds = math.cos(phi)
        drds = math.sin(phi)
        append_unique(SurfacePoint(z=z, r=r, zone="left_dome" if i < n_dome else "junction", meridional_tangent=(dzds, drds)))

    for i in range(1, n_cyl):
        u = i / n_cyl
        z = left_join + u * length
        append_unique(SurfacePoint(z=z, r=r_inner, zone="cylinder", meridional_tangent=(1.0, 0.0)))

    append_unique(SurfacePoint(z=right_join, r=r_inner, zone="junction", meridional_tangent=(1.0, 0.0)))

    for i in range(1, n_dome + 1):
        u = i / n_dome
        phi = phi_join + (phi_boss - phi_join) * u
        r = dome_R * math.cos(phi)
        z = right_join + dome_R * (math.sin(phi) - math.sin(phi_join))
        dzds = math.cos(phi)
        drds = -math.sin(phi)
        append_unique(SurfacePoint(z=z, r=r, zone="right_dome" if i < n_dome else "boss", meridional_tangent=(dzds, drds)))
    return pts


def interpolate_profile(profile: list[SurfacePoint], t: float) -> SurfacePoint:
    pos = t * (len(profile) - 1)
    i = min(max(int(math.floor(pos)), 0), len(profile) - 2)
    f = pos - i
    a = profile[i]
    b = profile[i + 1]
    zone = a.zone if f < 0.5 else b.zone
    return SurfacePoint(
        z=a.z * (1.0 - f) + b.z * f,
        r=a.r * (1.0 - f) + b.r * f,
        zone=zone,
        meridional_tangent=(
            a.meridional_tangent[0] * (1.0 - f) + b.meridional_tangent[0] * f,
            a.meridional_tangent[1] * (1.0 - f) + b.meridional_tangent[1] * f,
        ),
    )


def geodesic_angle_deg(case: dict, point: SurfacePoint, base_angle: float) -> float:
    boss_r = float(case["geometry"]["boss_radius_mm"])
    if point.zone == "cylinder":
        return abs(base_angle)
    ratio = max(-0.999, min(0.999, boss_r / max(point.r, boss_r + 1e-6)))
    alpha_geo = math.degrees(math.asin(ratio))
    if point.zone == "junction":
        return 0.50 * abs(base_angle) + 0.50 * alpha_geo
    return alpha_geo


def dome_thickness_factor(case: dict, point: SurfacePoint) -> float:
    r_inner = float(case["geometry"]["inner_radius_mm"])
    boss_r = float(case["geometry"]["boss_radius_mm"])
    wind = case["winding"]
    overlap = float(wind["overlap_factor"])
    if point.zone == "cylinder":
        return 1.0
    model = str(wind.get("dome_thickness_model", "")).lower()
    if "paik" in model or "clairaut" in model:
        return paik_clairaut_thickness_factor(case, point, overlap)
    concentration = (r_inner / max(point.r, boss_r + 1e-6)) ** 0.65
    if point.zone == "junction":
        concentration = 0.5 + 0.5 * concentration
    return max(0.75, min(2.65, overlap * concentration))


def paik_clairaut_thickness_factor(case: dict, point: SurfacePoint, overlap: float) -> float:
    """Dome thickness amplification from the Paik/Jin/Bai Clairaut law.

    The source formula is singular at the polar opening because the winding
    angle tends to 90 deg. A finite cap is therefore kept as a numerical
    reconstruction parameter for the shell model.
    """
    geom = case["geometry"]
    wind = case["winding"]
    r_cyl = float(geom["inner_radius_mm"])
    r0 = float(wind.get("clairaut_polar_radius_mm", geom.get("boss_radius_mm", 0.0)))
    band_width = float(wind.get("helical_band_width_mm", wind.get("tow_width_mm", 1.0)))
    max_factor = float(wind.get("dome_thickness_factor_cap", 4.0))
    min_factor = float(wind.get("dome_thickness_factor_floor", 0.75))
    r0 = max(1e-6, min(r0, 0.999 * r_cyl))
    r = max(point.r, r0 + 1e-6)
    ratio0 = max(-0.999999, min(0.999999, r0 / r_cyl))
    ratio = max(-0.999999, min(0.999999, r0 / r))
    cos_alpha0 = math.sqrt(max(1e-12, 1.0 - ratio0 * ratio0))
    cos_alpha = math.sqrt(max(1e-12, 1.0 - ratio * ratio))
    radial_fraction = max(0.0, min(1.0, (r_cyl - r) / max(r_cyl - r0, 1e-6)))
    denominator = r + 2.0 * band_width * radial_fraction**4
    factor = (cos_alpha0 / cos_alpha) * (r_cyl / max(denominator, 1e-6))
    if point.zone == "junction":
        factor = 0.5 + 0.5 * factor
    return max(min_factor, min(max_factor, overlap * factor))


def build_mesh(case: dict, n_meridional: int, n_theta: int) -> tuple[dict[int, tuple[float, float, float]], list[ElementInfo]]:
    profile_base = make_profile(case, n_dome=max(6, n_meridional // 4), n_cyl=max(8, n_meridional // 2))
    n_i = 2 * (len(profile_base) - 1)
    n_j = 2 * n_theta
    profile = [interpolate_profile(profile_base, i / n_i) for i in range(n_i + 1)]

    nodes: dict[int, tuple[float, float, float]] = {}

    def node_id(i: int, j: int) -> int:
        return i * n_j + (j % n_j) + 1

    for i, p in enumerate(profile):
        for j in range(n_j):
            theta = 2.0 * math.pi * j / n_j
            nodes[node_id(i, j)] = (p.r * math.cos(theta), p.r * math.sin(theta), p.z)

    layup = make_layup(case)
    base_total = sum(float(ply["thickness_mm"]) for ply in layup)
    elements: list[ElementInfo] = []
    eid = 1
    for i in range(0, n_i, 2):
        for j in range(0, n_j, 2):
            p_mid = interpolate_profile(profile_base, (i + 1) / n_i)
            theta_mid = 2.0 * math.pi * (j + 1) / n_j
            mer, circ = local_basis(theta_mid, p_mid.meridional_tangent[0], p_mid.meridional_tangent[1])
            base_angle = min(abs(float(ply["angle_deg"])) for ply in layup)
            angle = geodesic_angle_deg(case, p_mid, base_angle)
            factor = dome_thickness_factor(case, p_mid)
            active = active_layup(case, p_mid.zone, p_mid.r, layup)
            active_total = sum(float(ply["thickness_mm"]) for ply in active)
            coverage = min(1.0, max(0.0, active_total / max(base_total, 1e-12)))
            node_ids = (
                node_id(i, j),
                node_id(i + 2, j),
                node_id(i + 2, j + 2),
                node_id(i, j + 2),
                node_id(i + 1, j),
                node_id(i + 2, j + 1),
                node_id(i + 1, j + 2),
                node_id(i, j + 1),
            )
            corner_node_ids = (node_ids[0], node_ids[1], node_ids[2], node_ids[3])
            center = (
                p_mid.r * math.cos(theta_mid),
                p_mid.r * math.sin(theta_mid),
                p_mid.z,
            )
            elements.append(
                ElementInfo(
                    eid=eid,
                    node_ids=node_ids,
                    corner_node_ids=corner_node_ids,
                    center=center,
                    theta=theta_mid,
                    r=p_mid.r,
                    z=p_mid.z,
                    zone=p_mid.zone,
                    angle_deg=angle,
                    thickness_factor=factor,
                    thickness_mm=active_total * factor,
                    coverage=coverage,
                    meridional=mer,
                    circumferential=circ,
                )
            )
            eid += 1
    return nodes, elements


def calculix_material_name(material_name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in material_name.upper()).strip("_")
    return f"MAT_{cleaned[:64]}" if cleaned else "MAT_COMPOSITE"


def material_block(materials: dict, material_name: str, ccx_name: str | None = None) -> str:
    mat = materials["composites"][material_name]
    e1, e2, e3 = mat["young_modulus_mpa"]
    g12, g13, g23 = mat["shear_modulus_mpa"]
    nu12, nu13, nu23 = mat["poisson_ratios"]
    density = float(mat["density_kg_m3"]) * 1e-12
    material_label = ccx_name or calculix_material_name(material_name)
    return "\n".join(
        [
            f"*MATERIAL,NAME={material_label}",
            "*ELASTIC,TYPE=ENGINEERING CONSTANTS",
            f"{e1:.6g},{e2:.6g},{e3:.6g},{nu12:.6g},{nu13:.6g},{nu23:.6g},{g12:.6g},{g13:.6g},",
            f"{g23:.6g}",
            "*DENSITY",
            f"{density:.8e}",
        ]
    )

def orientation_vector(element: ElementInfo, angle_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    alpha = math.radians(angle_deg)
    fiber = unit(add(scale(element.meridional, math.cos(alpha)), scale(element.circumferential, math.sin(alpha))))
    in_plane_b = unit(add(scale(element.meridional, -math.sin(alpha)), scale(element.circumferential, math.cos(alpha))))
    return fiber, in_plane_b


def write_inp(
    case: dict,
    case_dir: Path,
    nodes: dict[int, tuple[float, float, float]],
    elements: list[ElementInfo],
    output_id: str | None = None,
    el_print: bool = False,
) -> Path:
    materials = case_materials(load_materials(), case)
    layup = make_layup(case)
    flat = flatten_case(case)
    job_id = output_id or case["case_id"]
    inp = case_dir / f"{job_id}_calculix_composite.inp"
    lines: list[str] = [
        "** Type IV COPV CalculiX composite-shell demonstrator",
        f"** Case: {case['case_id']}",
        "** Units: mm, N, MPa. Research/pre-design model, not certification.",
        "*NODE",
    ]
    for nid, xyz in sorted(nodes.items()):
        lines.append(f"{nid},{xyz[0]:.8f},{xyz[1]:.8f},{xyz[2]:.8f}")
    lines.append("*ELEMENT,TYPE=S8R,ELSET=EALL")
    for element in elements:
        lines.append(f"{element.eid}," + ",".join(str(n) for n in element.node_ids))

    left_boss_nodes = [nid for nid, xyz in nodes.items() if xyz[2] < min(p[2] for p in nodes.values()) + 1e-6]
    right_boss_nodes = [nid for nid, xyz in nodes.items() if xyz[2] > max(p[2] for p in nodes.values()) - 1e-6]
    pin_a = min(left_boss_nodes)
    pin_b = max(left_boss_nodes, key=lambda nid: nodes[nid][1])
    pin_c = min(right_boss_nodes)
    lines.append("*NSET,NSET=LEFT_BOSS")
    lines.extend(chunk_numbers(left_boss_nodes))
    lines.append("*NSET,NSET=RIGHT_BOSS")
    lines.extend(chunk_numbers(right_boss_nodes))
    lines.append("*NSET,NSET=PIN_A")
    lines.append(str(pin_a))
    lines.append("*NSET,NSET=PIN_B")
    lines.append(str(pin_b))
    lines.append("*NSET,NSET=PIN_C")
    lines.append(str(pin_c))
    used_materials = sorted(
        {
            ply_material_name_for_zone(case, ply, element.zone)
            for element in elements
            for ply in active_layup(case, element.zone, element.r, layup)
        }
    )
    for material_name in used_materials:
        lines.append(material_block(materials, material_name, calculix_material_name(material_name)))

    for element in elements:
        elset = f"E{element.eid:05d}"
        lines.append(f"*ELSET,ELSET={elset}")
        lines.append(str(element.eid))
        section_lines = [f"*SHELL SECTION,COMPOSITE,ELSET={elset},OFFSET=0."]
        section_layup = active_layup(case, element.zone, element.r, layup)
        for ply in section_layup:
            local_angle = local_ply_angle_deg(ply, element.zone, element.r, element.angle_deg, case)
            orientation_name = f"O{element.eid:05d}_{int(ply['ply']):03d}"
            fiber, plane = orientation_vector(element, local_angle)
            lines.append(f"*ORIENTATION,NAME={orientation_name},SYSTEM=RECTANGULAR")
            lines.append(
                f"{fiber[0]:.8f},{fiber[1]:.8f},{fiber[2]:.8f},"
                f"{plane[0]:.8f},{plane[1]:.8f},{plane[2]:.8f}"
            )
            material_name = calculix_material_name(ply_material_name_for_zone(case, ply, element.zone))
            section_lines.append(f"{float(ply['thickness_mm']) * element.thickness_factor:.8f},,{material_name},{orientation_name}")
        lines.extend(section_lines)

    pressure = float(case["loading"]["pressure_mpa"])
    bc = case.get("boundary_conditions", {})
    load_cfg = case.get("loading", {})
    axial_constraint = bc.get("axial_constraint", "single_boss_reference")
    include_boss_endcap_force = bool(load_cfg.get("include_boss_endcap_force", False))
    boss_r = float(case["geometry"]["boss_radius_mm"])
    boss_endcap_force = pressure * math.pi * boss_r * boss_r
    condition_comments = [
        "** Initial and boundary conditions",
        "** Initial state: stress-free, zero displacement and no thermal/prestress field.",
        "** Winding residual stresses, autofrettage and progressive damage are not included.",
        f"** Load: uniform internal pressure {pressure:.8f} MPa applied on EALL with *DLOAD, P.",
        f"** Boss end-cap force included: {include_boss_endcap_force}.",
        f"** Axial constraint mode: {axial_constraint}.",
        "** both_boss_rings: LEFT_BOSS and RIGHT_BOSS block U3.",
        "** single_boss_reference: LEFT_BOSS blocks U3 and RIGHT_BOSS is axially free.",
        "** minimal_pins: no full boss ring is clamped; three points remove rigid-body modes only.",
    ]
    boundary_lines = condition_comments + ["*BOUNDARY"]
    if axial_constraint == "both_boss_rings":
        boundary_lines.extend(["LEFT_BOSS,3,3,0.", "RIGHT_BOSS,3,3,0.", "PIN_A,1,2,0.", "PIN_B,2,2,0."])
    elif axial_constraint == "single_boss_reference":
        boundary_lines.extend(["LEFT_BOSS,3,3,0.", "PIN_A,1,2,0.", "PIN_B,2,2,0."])
    elif axial_constraint == "minimal_pins":
        boundary_lines.extend(["PIN_A,1,3,0.", "PIN_B,2,3,0.", "PIN_C,3,3,0."])
    else:
        raise ValueError(f"Unknown axial_constraint: {axial_constraint}")
    lines.extend(
        boundary_lines
        + [
            "*STEP",
            "*STATIC",
            "*DLOAD",
            f"EALL,P,{-pressure:.8f}",
        ]
    )
    if include_boss_endcap_force:
        left_force_per_node = -boss_endcap_force / max(len(left_boss_nodes), 1)
        right_force_per_node = boss_endcap_force / max(len(right_boss_nodes), 1)
        lines.extend(
            [
                "** Equivalent axial pressure force on polar boss openings.",
                "** This approximates pressure acting on missing boss/end-cap surfaces.",
                "*CLOAD",
                f"LEFT_BOSS,3,{left_force_per_node:.8f}",
                f"RIGHT_BOSS,3,{right_force_per_node:.8f}",
            ]
        )
    lines.extend(
        [
            "*NODE PRINT,NSET=LEFT_BOSS",
            "U",
            "*NODE FILE",
            "U",
        ]
    )
    if el_print:
        lines.extend(
            [
                "*EL PRINT,ELSET=EALL",
                "S",
            ]
        )
    lines.extend(
        [
            "*EL FILE, OUTPUT=3D",
            "S",
            "*END STEP",
        ]
    )
    inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inp


def chunk_numbers(values: Iterable[int], width: int = 16) -> list[str]:
    vals = list(values)
    return [",".join(str(v) for v in vals[i : i + width]) for i in range(0, len(vals), width)]


def write_orientation_csv(case_dir: Path, elements: list[ElementInfo]) -> Path:
    path = case_dir / "element_orientation_thickness.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "element_id",
                "x_mm",
                "y_mm",
                "z_mm",
                "radius_mm",
                "theta_deg",
                "zone",
                "geodesic_angle_deg",
                "thickness_factor",
                "thickness_mm",
                "tow_coverage",
            ],
        )
        writer.writeheader()
        for element in elements:
            writer.writerow(
                {
                    "element_id": element.eid,
                    "x_mm": element.center[0],
                    "y_mm": element.center[1],
                    "z_mm": element.center[2],
                    "radius_mm": element.r,
                    "theta_deg": math.degrees(element.theta) % 360.0,
                    "zone": element.zone,
                    "geodesic_angle_deg": element.angle_deg,
                    "thickness_factor": element.thickness_factor,
                    "thickness_mm": element.thickness_mm,
                    "tow_coverage": element.coverage,
                }
            )
    return path


def write_preview_vtk(case_dir: Path, nodes: dict[int, tuple[float, float, float]], elements: list[ElementInfo]) -> Path:
    used_nodes: list[int] = sorted({nid for e in elements for nid in e.corner_node_ids})
    map_id = {nid: i for i, nid in enumerate(used_nodes)}
    path = case_dir / "calculix_composite_shell_preview.vtk"
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("CalculiX Type IV composite shell preview\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(used_nodes)} float\n")
        for nid in used_nodes:
            x, y, z = nodes[nid]
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        f.write(f"POLYGONS {len(elements)} {len(elements) * 5}\n")
        for element in elements:
            ids = [map_id[nid] for nid in element.corner_node_ids]
            f.write("4 " + " ".join(str(i) for i in ids) + "\n")
        f.write(f"CELL_DATA {len(elements)}\n")
        write_cell_scalar(f, "fiber_angle_deg", [e.angle_deg for e in elements])
        write_cell_scalar(f, "thickness_mm", [e.thickness_mm for e in elements])
        write_cell_scalar(f, "thickness_factor", [e.thickness_factor for e in elements])
        write_cell_scalar(f, "tow_coverage", [e.coverage for e in elements])
        zone_code = {"left_dome": 1, "junction": 2, "cylinder": 3, "right_dome": 4, "boss": 5}
        write_cell_scalar(f, "zone_id", [zone_code.get(e.zone, 0) for e in elements])
        f.write("VECTORS fiber_direction float\n")
        for element in elements:
            fiber, _ = orientation_vector(element, element.angle_deg)
            f.write(f"{fiber[0]:.8f} {fiber[1]:.8f} {fiber[2]:.8f}\n")
    return path


def write_cell_scalar(handle, name: str, values: list[float]) -> None:
    handle.write(f"SCALARS {name} float 1\n")
    handle.write("LOOKUP_TABLE default\n")
    for value in values:
        handle.write(f"{float(value):.8f}\n")


def write_tow_paths(case: dict, case_dir: Path, elements: list[ElementInfo]) -> Path:
    by_z = sorted({round(e.z, 8): e for e in elements}.values(), key=lambda e: e.z)
    n_paths = 32
    points: list[tuple[float, float, float]] = []
    lines: list[list[int]] = []
    angles: list[float] = []
    for path_idx in range(n_paths):
        theta = 2.0 * math.pi * path_idx / n_paths
        line: list[int] = []
        handedness = 1.0 if path_idx % 2 == 0 else -1.0
        previous_z = by_z[0].z
        for element in by_z:
            dz = element.z - previous_z
            alpha = max(math.radians(element.angle_deg), math.radians(2.0))
            theta += handedness * dz * math.tan(alpha) / max(element.r, 1e-6)
            previous_z = element.z
            x = element.r * math.cos(theta)
            y = element.r * math.sin(theta)
            points.append((x, y, element.z))
            line.append(len(points) - 1)
        lines.append(line)
        angles.append(1.0 if handedness > 0 else -1.0)

    path = case_dir / "calculix_tow_paths.vtk"
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Finite-width tow path centerlines for ParaView rendering\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        total_size = sum(len(line) + 1 for line in lines)
        f.write(f"LINES {len(lines)} {total_size}\n")
        for line in lines:
            f.write(str(len(line)) + " " + " ".join(str(i) for i in line) + "\n")
        f.write(f"CELL_DATA {len(lines)}\n")
        write_cell_scalar(f, "tow_family", angles)
        write_cell_scalar(f, "tow_width_mm", [float(case["winding"]["tow_width_mm"]) for _ in lines])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a CalculiX composite-shell Type IV COPV demonstrator.")
    parser.add_argument("--case-id", default="cal_001")
    parser.add_argument("--output-id", default="")
    parser.add_argument("--n-meridional", type=int, default=24)
    parser.add_argument("--n-theta", type=int, default=24)
    parser.add_argument("--el-print", action="store_true", help="Also request integration-point stresses in the .dat file.")
    args = parser.parse_args()

    ensure_dirs()
    cases = {case["case_id"]: case for case in load_cases()}
    if args.case_id not in cases:
        raise SystemExit(f"Unknown case_id {args.case_id}")
    case = cases[args.case_id]
    output_id = args.output_id or args.case_id
    case_dir = SIM_RESULTS / "calculix" / output_id
    case_dir.mkdir(parents=True, exist_ok=True)

    nodes, elements = build_mesh(case, args.n_meridional, args.n_theta)
    inp = write_inp(case, case_dir, nodes, elements, output_id=output_id, el_print=args.el_print)
    orientation_csv = write_orientation_csv(case_dir, elements)
    preview_vtk = write_preview_vtk(case_dir, nodes, elements)
    tow_vtk = write_tow_paths(case, case_dir, elements)
    write_json(
        case_dir / "calculix_case_manifest.json",
        {
            "case_id": args.case_id,
            "base_case_id": args.case_id,
            "output_id": output_id,
            "inp": str(inp),
            "orientation_csv": str(orientation_csv),
            "preview_vtk": str(preview_vtk),
            "tow_paths_vtk": str(tow_vtk),
            "element_count": len(elements),
            "node_count": len(nodes),
            "n_meridional": args.n_meridional,
            "n_theta": args.n_theta,
            "el_print_requested": args.el_print,
            "model_status": "calculix_input_generated",
            "model_scope": "composite S8R shell demonstrator with local ply orientations; research/pre-design only",
            "units": {"length": "mm", "force": "N", "stress": "MPa"},
            "initial_conditions": {
                "stress": "zero initial stress",
                "displacement": "zero reference displacement",
                "thermal_field": "not applied",
                "winding_residual_stress": "not included",
                "damage_state": "undamaged linear-elastic material",
            },
            "load_definition": {
                "type": "uniform internal pressure",
                "calculix_keyword": "*DLOAD, EALL, P",
                "pressure_mpa": float(case["loading"]["pressure_mpa"]),
                "sign_convention": "negative pressure value in the generated .inp follows CalculiX shell normal convention",
                "include_boss_endcap_force": bool(case.get("loading", {}).get("include_boss_endcap_force", False)),
                "boss_endcap_force_N_per_side": (
                    float(case["loading"]["pressure_mpa"]) * math.pi * float(case["geometry"]["boss_radius_mm"]) ** 2
                    if bool(case.get("loading", {}).get("include_boss_endcap_force", False))
                    else 0.0
                ),
            },
            "boundary_conditions": {
                "axial_constraint": case.get("boundary_conditions", {}).get("axial_constraint", "single_boss_reference"),
                "left_boss": "U3=0 on LEFT_BOSS nodes",
                "right_boss": "U3=0 on RIGHT_BOSS nodes for both_boss_rings mode",
                "rigid_body_pinning": "PIN_A/PIN_B/PIN_C are available for minimal rigid-body suppression",
                "contact_liner_boss_composite": "not included in the main DOE shell model; studied separately in contact benchmarks",
            },
            "known_limitations": [
                "linear static shell model",
                "no progressive damage",
                "no delamination",
                "no matrix crack evolution",
                "no winding residual stress",
                "no complete liner/boss/composite contact in the main DOE",
                "failure indices are initiation proxies, not certified burst pressure",
            ],
            "coverage_model": case.get("winding", {}).get("coverage_model", "legacy_all_plies_active"),
            "ply_materials": sorted(
                {
                    ply_material_name_for_zone(case, ply, element.zone)
                    for element in elements
                    for ply in active_layup(case, element.zone, element.r, make_layup(case))
                }
            ),
        },
    )
    print(inp)


if __name__ == "__main__":
    main()
