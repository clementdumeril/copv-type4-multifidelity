#!/usr/bin/env python3
"""
Generate dome-focused ParaView files for a Type IV hydrogen COPV.

The goal is visual clarity:
- the dome shell is colored by local geodesic winding angle/thickness factor
- helical tows are finite-width ribbon bands, not centerline "strings"

This is a visualization/model-preparation artifact, not a certified FEA mesh.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


GENETIC_DIR = Path(__file__).resolve().parent
SIM_DIR = GENETIC_DIR.parent / "hydrogen_type4_copv"
COMMON_INPUT = SIM_DIR / "comparison_input.json"
OUT_DIR = GENETIC_DIR / "type4_copv_comparison_output"


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
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        size = sum(len(poly) + 1 for poly in polys)
        f.write(f"POLYGONS {len(polys)} {size}\n")
        for poly in polys:
            f.write(str(len(poly)) + " " + " ".join(str(i) for i in poly) + "\n")
        f.write(f"CELL_DATA {len(polys)}\n")
        for name, values in cell_data.items():
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in values:
                f.write(f"{float(value):.8f}\n")


def build_dome_shell(data: dict) -> Path:
    radius = data["geometry"]["internal_radius_mm"] + 0.5 * sum(layer["thickness_mm"] for layer in data["layers"])
    polar_opening = 25.0
    phi_polar = math.asin(polar_opening / radius)
    n_phi = 56
    n_theta = 144
    points: list[tuple[float, float, float]] = []
    polys: list[tuple[int, ...]] = []
    cell_data = {
        "local_geodesic_angle_deg": [],
        "local_thickness_factor": [],
        "radius_ratio": [],
    }

    for ip in range(n_phi + 1):
        u = ip / n_phi
        phi = math.pi / 2.0 - u * (math.pi / 2.0 - phi_polar)
        rho = radius * math.sin(phi)
        z = radius * math.cos(phi)
        for it in range(n_theta):
            theta = 2.0 * math.pi * it / n_theta
            points.append((rho * math.cos(theta), rho * math.sin(theta), z))

    cylinder_reference_angle = math.radians(15.0)
    cyl_rho = radius
    for ip in range(n_phi):
        phi_mid = math.pi / 2.0 - (ip + 0.5) / n_phi * (math.pi / 2.0 - phi_polar)
        rho_mid = radius * math.sin(phi_mid)
        local_angle = math.degrees(math.asin(min(0.9999, polar_opening / rho_mid)))
        thickness_factor = (cyl_rho * math.cos(cylinder_reference_angle)) / max(rho_mid * math.cos(math.radians(local_angle)), 1e-6)
        thickness_factor = min(thickness_factor, 8.0)
        for it in range(n_theta):
            p0 = ip * n_theta + it
            p1 = ip * n_theta + ((it + 1) % n_theta)
            p2 = (ip + 1) * n_theta + ((it + 1) % n_theta)
            p3 = (ip + 1) * n_theta + it
            polys.append((p0, p1, p2, p3))
            cell_data["local_geodesic_angle_deg"].append(local_angle)
            cell_data["local_thickness_factor"].append(thickness_factor)
            cell_data["radius_ratio"].append(rho_mid / radius)

    vtk_path = OUT_DIR / "type4_dome_geodesic_shell.vtk"
    write_polydata(vtk_path, points, polys, cell_data)
    return vtk_path


def geodesic_centerline(radius: float, polar_opening: float, theta0: float, sign: float, n_steps: int) -> list[tuple[float, float, float, float, float]]:
    """Return rho, theta, z, local_angle_deg, thickness_factor along one dome path."""
    phi_equator = math.pi / 2.0
    phi_polar = math.asin(polar_opening / radius)
    theta = theta0
    last_phi = phi_equator
    points = []
    cyl_alpha = math.radians(15.0)
    for i in range(n_steps + 1):
        u = i / n_steps
        phi = phi_equator - u * (phi_equator - phi_polar)
        rho = radius * math.sin(phi)
        z = radius * math.cos(phi)
        if i > 0:
            mid_phi = 0.5 * (last_phi + phi)
            mid_rho = max(radius * math.sin(mid_phi), polar_opening * 1.001)
            alpha = math.asin(min(0.9999, polar_opening / mid_rho))
            dphi = abs(phi - last_phi)
            theta += sign * math.tan(alpha) * radius / mid_rho * dphi
        alpha_here = math.degrees(math.asin(min(0.9999, polar_opening / max(rho, polar_opening * 1.001))))
        thickness_factor = (radius * math.cos(cyl_alpha)) / max(rho * math.cos(math.radians(alpha_here)), 1e-6)
        thickness_factor = min(thickness_factor, 8.0)
        points.append((rho, theta, z, alpha_here, thickness_factor))
        last_phi = phi
    return points


def build_tow_ribbons(data: dict) -> Path:
    radius = data["geometry"]["internal_radius_mm"] + sum(layer["thickness_mm"] for layer in data["layers"]) + 2.0
    polar_opening = 25.0
    tow_width = data["filament"].get("tow_width_mm_reference", 6.0)
    n_steps = 90
    n_tows_per_direction = 12
    points: list[tuple[float, float, float]] = []
    polys: list[tuple[int, ...]] = []
    cell_data = {
        "tow_direction": [],
        "tow_width_mm": [],
        "local_angle_deg": [],
        "thickness_factor": [],
    }

    for direction, sign in [(1, 1.0), (-1, -1.0)]:
        for k in range(n_tows_per_direction):
            theta0 = 2.0 * math.pi * k / n_tows_per_direction + (0.17 if direction < 0 else 0.0)
            center = geodesic_centerline(radius, polar_opening, theta0, sign, n_steps)
            strip_indices = []
            for rho, theta, z, local_angle, thickness_factor in center:
                dtheta = tow_width / max(2.0 * rho, 1e-6)
                p_left = (rho * math.cos(theta - dtheta), rho * math.sin(theta - dtheta), z)
                p_right = (rho * math.cos(theta + dtheta), rho * math.sin(theta + dtheta), z)
                strip_indices.append((len(points), len(points) + 1, local_angle, thickness_factor))
                points.extend([p_left, p_right])
            for i in range(n_steps):
                a0, a1, angle0, thick0 = strip_indices[i]
                b0, b1, angle1, thick1 = strip_indices[i + 1]
                polys.append((a0, a1, b1, b0))
                cell_data["tow_direction"].append(direction)
                cell_data["tow_width_mm"].append(tow_width)
                cell_data["local_angle_deg"].append(0.5 * (angle0 + angle1))
                cell_data["thickness_factor"].append(0.5 * (thick0 + thick1))

    vtk_path = OUT_DIR / "type4_dome_finite_width_tows.vtk"
    write_polydata(vtk_path, points, polys, cell_data)
    return vtk_path


def write_paraview_script(shell_vtk: Path, tow_vtk: Path) -> Path:
    script_path = OUT_DIR / "open_type4_dome_paraview.py"
    state_path = OUT_DIR / "type4_dome_geodesic_state.pvsm"
    screenshot_angle = OUT_DIR / "paraview_dome_angle_thickness.png"
    screenshot_tows = OUT_DIR / "paraview_dome_finite_width_tows.png"
    script = f'''from paraview.simple import *

shell_file = r"{shell_vtk}"
tow_file = r"{tow_vtk}"
state_file = r"{state_path}"
screenshot_angle = r"{screenshot_angle}"
screenshot_tows = r"{screenshot_tows}"

_DisableFirstRenderCameraReset()
shell = LegacyVTKReader(registrationName="dome_geodesic_shell", FileNames=[shell_file])
tows = LegacyVTKReader(registrationName="finite_width_tow_bands", FileNames=[tow_file])
shell.UpdatePipeline()
tows.UpdatePipeline()

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1400, 900]
view.Background = [0.08, 0.08, 0.09]

shell_display = Show(shell, view, "GeometryRepresentation")
shell_display.Representation = "Surface With Edges"
shell_display.Opacity = 0.66
shell_display.EdgeColor = [0.02, 0.02, 0.02]
ColorBy(shell_display, ("CELLS", "local_thickness_factor"))
shell_display.SetScalarBarVisibility(view, True)

lut = GetColorTransferFunction("local_thickness_factor")
lut.ApplyPreset("Turbo", True)
lut.RescaleTransferFunction(1.0, 8.0)
pwf = GetOpacityTransferFunction("local_thickness_factor")
pwf.RescaleTransferFunction(1.0, 8.0)

tow_display = Show(tows, view, "GeometryRepresentation")
tow_display.Representation = "Surface"
tow_display.Opacity = 0.88
ColorBy(tow_display, ("CELLS", "tow_direction"))
tow_display.SetScalarBarVisibility(view, False)
tow_lut = GetColorTransferFunction("tow_direction")
tow_lut.RescaleTransferFunction(-1.0, 1.0)

view.CameraPosition = [220.0, -360.0, 260.0]
view.CameraFocalPoint = [0.0, 0.0, 45.0]
view.CameraViewUp = [-0.15, 0.42, 0.90]
view.CameraParallelScale = 135.0

Render(view)
SaveScreenshot(screenshot_angle, view)

ColorBy(shell_display, ("CELLS", "local_geodesic_angle_deg"))
angle_lut = GetColorTransferFunction("local_geodesic_angle_deg")
angle_lut.ApplyPreset("Turbo", True)
angle_lut.RescaleTransferFunction(15.0, 90.0)
Render(view)
SaveScreenshot(screenshot_tows, view)
SaveState(state_file)
print("Saved dome state:", state_file)
print("Saved dome screenshots:", screenshot_angle, screenshot_tows)
'''
    script_path.write_text(script, encoding="utf-8")
    return script_path


def write_open_cmd(script_path: Path, shell_vtk: Path, tow_vtk: Path) -> None:
    cmd_path = OUT_DIR / "open_dome_in_paraview.cmd"
    cmd_path.write_text(
        f'''@echo off
set "PV=C:\\Program Files\\ParaView 6.1.0\\bin\\paraview.exe"
if not exist "%PV%" (
  echo ParaView not found at "%PV%"
  exit /b 1
)
echo Opening dome files. In ParaView, click Apply if the Properties panel asks for it.
start "" "%PV%" "{shell_vtk}" "{tow_vtk}"
''',
        encoding="utf-8",
    )
    pvpython_cmd = OUT_DIR / "render_dome_with_pvpython.cmd"
    pvpython_cmd.write_text(
        f'''@echo off
set "PVPY=C:\\Program Files\\ParaView 6.1.0\\bin\\pvpython.exe"
if not exist "%PVPY%" (
  echo pvpython not found at "%PVPY%"
  exit /b 1
)
"%PVPY%" "{script_path}"
''',
        encoding="utf-8",
    )


def write_dome_note(data: dict) -> None:
    note_path = OUT_DIR / "DOME_MODEL_NOTE.md"
    note_path.write_text(
        f"""# Note sur le dôme et l'enroulement filamentaire

Tu avais raison: les courbes fines ne sont pas un depot physique de composite. Dans cette version, le dôme est represente avec:

- une surface hemispherique de reference;
- une loi geodesique simple `sin(alpha) = r_boss / r_local`;
- un facteur d'epaississement local qui augmente quand le rayon local diminue vers le boss;
- des bandes de tow a largeur finie, largeur de reference {data['filament']['tow_width_mm_reference']} mm, au lieu de simples lignes.

Ce que cela montre:

- pres de l'equateur du dôme, l'angle local reste proche de l'angle cylindre bas-angle;
- pres de l'ouverture polaire, l'angle augmente fortement et la densite de matiere s'accumule;
- c'est exactement la zone que `genetic` ne calcule pas encore, car son coeur est un cylindre epais multicouche.

Fichiers:

- `type4_dome_geodesic_shell.vtk`
- `type4_dome_finite_width_tows.vtk`
- `paraview_dome_angle_thickness.png`
- `paraview_dome_finite_width_tows.png`
- `open_dome_in_paraview.cmd`
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_common_input()
    shell_vtk = build_dome_shell(data)
    tow_vtk = build_tow_ribbons(data)
    script = write_paraview_script(shell_vtk, tow_vtk)
    write_open_cmd(script, shell_vtk, tow_vtk)
    write_dome_note(data)
    print(json.dumps({"shell_vtk": str(shell_vtk), "tow_vtk": str(tow_vtk), "script": str(script)}, indent=2))


if __name__ == "__main__":
    main()
