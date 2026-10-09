#!/usr/bin/env python3
"""
Generate a full Type IV tank visualization for ParaView:
- cylindrical section
- upper and lower domes
- finite-width filament winding bands over cylinder and domes

This is a visualization mesh derived from the common input. It is not a
certification-grade FEA mesh.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT.parent / "hydrogen_type4_copv"
OUT = ROOT / "type4_copv_comparison_output"
COMMON_INPUT = SIM_DIR / "comparison_input.json"


def load_common_input() -> dict:
    with COMMON_INPUT.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_polydata(path: Path, points: list[tuple[float, float, float]], polys: list[tuple[int, ...]], cell_data: dict[str, list[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{path.stem}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        f.write(f"POLYGONS {len(polys)} {sum(len(poly) + 1 for poly in polys)}\n")
        for poly in polys:
            f.write(str(len(poly)) + " " + " ".join(str(i) for i in poly) + "\n")
        f.write(f"CELL_DATA {len(polys)}\n")
        for name, values in cell_data.items():
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for value in values:
                f.write(f"{float(value):.8f}\n")


def local_geodesic(radius: float, polar_opening: float, rho: float) -> tuple[float, float]:
    alpha = math.asin(min(0.9999, polar_opening / max(rho, polar_opening * 1.001)))
    cyl_alpha = math.radians(15.0)
    factor = radius * math.cos(cyl_alpha) / max(rho * math.cos(alpha), 1e-6)
    return math.degrees(alpha), min(factor, 8.0)


def build_tank_body(data: dict) -> Path:
    inner_r = data["geometry"]["internal_radius_mm"]
    total_t = sum(layer["thickness_mm"] for layer in data["layers"])
    radius = inner_r + total_t
    polar_opening = 25.0
    cyl_len = 420.0
    n_theta = 160
    n_z = 72
    n_phi = 48
    phi_p = math.asin(polar_opening / radius)

    points: list[tuple[float, float, float]] = []
    polys: list[tuple[int, ...]] = []
    cell_data = {"region": [], "local_angle_deg": [], "local_thickness_factor": []}

    # Cylinder
    cyl_start = len(points)
    for iz in range(n_z + 1):
        z = -cyl_len / 2.0 + cyl_len * iz / n_z
        for it in range(n_theta):
            th = 2.0 * math.pi * it / n_theta
            points.append((radius * math.cos(th), radius * math.sin(th), z))
    for iz in range(n_z):
        for it in range(n_theta):
            p0 = cyl_start + iz * n_theta + it
            p1 = cyl_start + iz * n_theta + ((it + 1) % n_theta)
            p2 = cyl_start + (iz + 1) * n_theta + ((it + 1) % n_theta)
            p3 = cyl_start + (iz + 1) * n_theta + it
            polys.append((p0, p1, p2, p3))
            cell_data["region"].append(0.0)
            cell_data["local_angle_deg"].append(15.0)
            cell_data["local_thickness_factor"].append(1.0)

    # Domes
    for dome_sign, region in [(1.0, 1.0), (-1.0, -1.0)]:
        start = len(points)
        for ip in range(n_phi + 1):
            u = ip / n_phi
            phi = math.pi / 2.0 - u * (math.pi / 2.0 - phi_p)
            rho = radius * math.sin(phi)
            z = dome_sign * (cyl_len / 2.0 + radius * math.cos(phi))
            for it in range(n_theta):
                th = 2.0 * math.pi * it / n_theta
                points.append((rho * math.cos(th), rho * math.sin(th), z))
        for ip in range(n_phi):
            phi_mid = math.pi / 2.0 - (ip + 0.5) / n_phi * (math.pi / 2.0 - phi_p)
            rho_mid = radius * math.sin(phi_mid)
            angle, factor = local_geodesic(radius, polar_opening, rho_mid)
            for it in range(n_theta):
                p0 = start + ip * n_theta + it
                p1 = start + ip * n_theta + ((it + 1) % n_theta)
                p2 = start + (ip + 1) * n_theta + ((it + 1) % n_theta)
                p3 = start + (ip + 1) * n_theta + it
                polys.append((p0, p1, p2, p3))
                cell_data["region"].append(region)
                cell_data["local_angle_deg"].append(angle)
                cell_data["local_thickness_factor"].append(factor)

    vtk = OUT / "type4_full_tank_body.vtk"
    write_polydata(vtk, points, polys, cell_data)
    return vtk


def dome_path(radius: float, polar_opening: float, cyl_len: float, top: bool, theta0: float, sign: float, n: int) -> list[tuple[float, float, float, float, float]]:
    phi_eq = math.pi / 2.0
    phi_p = math.asin(polar_opening / radius)
    theta = theta0
    last_phi = phi_eq
    path = []
    dome_sign = 1.0 if top else -1.0
    for i in range(n + 1):
        u = i / n
        phi = phi_eq - u * (phi_eq - phi_p)
        rho = radius * math.sin(phi)
        if i > 0:
            mid_phi = 0.5 * (last_phi + phi)
            mid_rho = max(radius * math.sin(mid_phi), polar_opening * 1.001)
            alpha = math.asin(min(0.9999, polar_opening / mid_rho))
            dphi = abs(phi - last_phi)
            theta += sign * math.tan(alpha) * radius / mid_rho * dphi
        z = dome_sign * (cyl_len / 2.0 + radius * math.cos(phi))
        angle, factor = local_geodesic(radius, polar_opening, rho)
        path.append((rho, theta, z, angle, factor))
        last_phi = phi
    return path


def cylinder_path(radius: float, cyl_len: float, theta0: float, sign: float, n: int) -> list[tuple[float, float, float, float, float]]:
    alpha = math.radians(15.0)
    path = []
    for i in range(n + 1):
        u = i / n
        z = -cyl_len / 2.0 + cyl_len * u
        theta = theta0 + sign * (z + cyl_len / 2.0) * math.tan(alpha) / radius
        path.append((radius, theta, z, 15.0, 1.0))
    return path


def ribbon_from_path(points: list[tuple[float, float, float, float, float]], tow_width: float, direction: int, out_points: list, out_polys: list, cell_data: dict) -> None:
    strip = []
    for rho, theta, z, angle, factor in points:
        dtheta = tow_width / max(2.0 * rho, 1e-6)
        left = (rho * math.cos(theta - dtheta), rho * math.sin(theta - dtheta), z)
        right = (rho * math.cos(theta + dtheta), rho * math.sin(theta + dtheta), z)
        strip.append((len(out_points), len(out_points) + 1, angle, factor))
        out_points.extend([left, right])
    for i in range(len(strip) - 1):
        a0, a1, angle0, factor0 = strip[i]
        b0, b1, angle1, factor1 = strip[i + 1]
        out_polys.append((a0, a1, b1, b0))
        cell_data["tow_direction"].append(direction)
        cell_data["local_angle_deg"].append(0.5 * (angle0 + angle1))
        cell_data["local_thickness_factor"].append(0.5 * (factor0 + factor1))
        cell_data["tow_width_mm"].append(tow_width)


def build_tows(data: dict) -> Path:
    inner_r = data["geometry"]["internal_radius_mm"]
    total_t = sum(layer["thickness_mm"] for layer in data["layers"])
    radius = inner_r + total_t + 3.0
    cyl_len = 420.0
    polar_opening = 25.0
    tow_width = data["filament"].get("tow_width_mm_reference", 6.0)
    overlap_fraction = 0.18
    pitch = tow_width * (1.0 - overlap_fraction)
    n_tows = max(24, math.ceil(2.0 * math.pi * radius / pitch))
    points: list[tuple[float, float, float]] = []
    polys: list[tuple[int, ...]] = []
    cell_data = {"tow_direction": [], "local_angle_deg": [], "local_thickness_factor": [], "tow_width_mm": []}

    for direction, sign in [(1, 1.0), (-1, -1.0)]:
        for k in range(n_tows):
            theta0 = 2.0 * math.pi * k / n_tows + (math.pi / n_tows if direction < 0 else 0.0)
            top = dome_path(radius, polar_opening, cyl_len, True, theta0 + sign * cyl_len * math.tan(math.radians(15.0)) / radius, sign, 72)
            cyl = cylinder_path(radius, cyl_len, theta0, sign, 96)
            bottom = dome_path(radius, polar_opening, cyl_len, False, theta0, -sign, 72)
            ribbon_from_path(bottom, tow_width, direction, points, polys, cell_data)
            ribbon_from_path(cyl, tow_width, direction, points, polys, cell_data)
            ribbon_from_path(top, tow_width, direction, points, polys, cell_data)

    coverage = {
        "tow_width_mm": tow_width,
        "outer_visual_radius_mm": radius,
        "n_tows_per_direction": n_tows,
        "equator_pitch_mm": 2.0 * math.pi * radius / n_tows,
        "target_overlap_fraction": overlap_fraction,
        "equator_width_to_pitch_ratio": tow_width / (2.0 * math.pi * radius / n_tows),
        "note": "Dense visual coverage estimate for the ParaView rendering; not a certified winding program.",
    }
    (OUT / "type4_full_tank_tow_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    vtk = OUT / "type4_full_tank_finite_width_tows.vtk"
    write_polydata(vtk, points, polys, cell_data)
    return vtk


def write_pv_script(body: Path, tows: Path) -> Path:
    script = OUT / "open_type4_full_tank_paraview.py"
    state = OUT / "type4_full_tank_state.pvsm"
    screenshot = OUT / "paraview_full_tank_with_tows.png"
    script.write_text(
        f'''from paraview.simple import *

body_file = r"{body}"
tows_file = r"{tows}"
state_file = r"{state}"
screenshot_file = r"{screenshot}"

_DisableFirstRenderCameraReset()
body = LegacyVTKReader(registrationName="full_tank_body", FileNames=[body_file])
tows = LegacyVTKReader(registrationName="full_tank_finite_width_tows", FileNames=[tows_file])
body.UpdatePipeline()
tows.UpdatePipeline()

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1500, 950]
view.Background = [0.08, 0.08, 0.09]

body_display = Show(body, view, "GeometryRepresentation")
body_display.Representation = "Surface With Edges"
body_display.Opacity = 0.50
body_display.EdgeColor = [0.02, 0.02, 0.02]
ColorBy(body_display, ("CELLS", "local_thickness_factor"))
body_display.SetScalarBarVisibility(view, True)
body_lut = GetColorTransferFunction("local_thickness_factor")
body_lut.ApplyPreset("Turbo", True)
body_lut.RescaleTransferFunction(1.0, 8.0)

tow_display = Show(tows, view, "GeometryRepresentation")
tow_display.Representation = "Surface"
tow_display.Opacity = 0.92
ColorBy(tow_display, ("CELLS", "tow_direction"))
tow_display.SetScalarBarVisibility(view, False)
tow_lut = GetColorTransferFunction("tow_direction")
tow_lut.RescaleTransferFunction(-1.0, 1.0)

view.CameraPosition = [620.0, -900.0, 240.0]
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelScale = 380.0
view.CameraParallelProjection = 1

Render(view)
SaveState(state_file)
SaveScreenshot(screenshot_file, view)
print("Saved full tank state:", state_file)
print("Saved full tank screenshot:", screenshot_file)
''',
        encoding="utf-8",
    )
    (OUT / "open_full_tank_in_paraview.cmd").write_text(
        f'''@echo off
set "PV=C:\\Program Files\\ParaView 6.1.0\\bin\\paraview.exe"
start "" "%PV%" "{body}" "{tows}"
''',
        encoding="utf-8",
    )
    return script


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_common_input()
    body = build_tank_body(data)
    tows = build_tows(data)
    script = write_pv_script(body, tows)
    print(json.dumps({"body": str(body), "tows": str(tows), "script": str(script)}, indent=2))


if __name__ == "__main__":
    main()
