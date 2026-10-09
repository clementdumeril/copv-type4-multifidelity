#!/usr/bin/env python3
"""Render the CalculiX fibre failure index of one DOE case on the vessel surface.

Reads the raw CalculiX result (.frd) and the element orientation table of a case,
projects the stresses into each ply's material frame with the same functions as
postprocess_calculix_case.py, keeps the worst ply per shell element, writes a VTK
file with a `fiber_fi` cell field and renders it with ParaView (pvpython).

The raw .frd files (~20 MB per case) are not in this repository; point
TYPE4_OUTPUTS_DIR to a local CalculiX output tree to run it:

    TYPE4_OUTPUTS_DIR=.../outputs_11l_single_boss \
    TYPE4_CASES_FILE=data/doe_11l_single_boss_cases.json \
    python code/plotting/render_fiber_field.py --case-id 11l_0122
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "high_fidelity"))

from calibration_utils import SIM_RESULTS, active_layup, case_materials, load_cases, load_materials, local_ply_angle_deg, make_layup, ply_material_name_for_zone  # noqa: E402
from postprocess_calculix_case import (  # noqa: E402
    assign_ply_from_thickness_position,
    boundary_epsilon_mm,
    compute_ply_boundary_radii,
    hashin_mode_parts,
    load_orientation_rows,
    nearest_element_index,
    parse_frd_nodes,
    parse_stress_rows,
    project_stress,
)

PVPYTHON = Path(r"C:\Program Files\ParaView 6.1.0\bin\pvpython.exe")
FIG_DIR = ROOT.parent / "reports" / "figures"


def element_fiber_index(case: dict, case_dir: Path) -> dict[int, float]:
    """Worst-ply fibre stress ratio |sigma_11| / X per shell element (ply-mean stresses)."""
    frd = case_dir / f"{case['case_id']}_calculix_composite.frd"
    nodes = parse_frd_nodes(frd)
    stress_rows = parse_stress_rows(frd)
    elements = load_orientation_rows(case_dir / "element_orientation_thickness.csv")
    layup = make_layup(case)
    materials = case_materials(load_materials(), case)

    sums: dict[tuple[int, int], list] = {}
    last = None
    for stress in stress_rows:
        point = nodes.get(int(stress["node_id"]))
        if point is None:
            continue
        last = nearest_element_index(point, elements, last)
        element = elements[last]
        r = (point[0] ** 2 + point[1] ** 2) ** 0.5
        plies = active_layup(case, element["zone"], float(element["r"]), layup)
        eps = boundary_epsilon_mm(element, plies)
        if any(abs(r - b) < eps for b in compute_ply_boundary_radii(element, plies)):
            continue
        ply = assign_ply_from_thickness_position(r, element, plies)
        key = (element["element_id"], int(ply["ply"]))
        if key not in sums:
            sums[key] = [element, ply, {k: 0.0 for k in ("sxx", "syy", "szz", "sxy", "syz", "sxz")}, 0]
        acc = sums[key]
        for k in acc[2]:
            acc[2][k] += float(stress[k])
        acc[3] += 1

    worst: dict[int, float] = {}
    for (element_id, _), (element, ply, total, n) in sums.items():
        mean = {k: v / n for k, v in total.items()}
        angle = local_ply_angle_deg(ply, element["zone"], float(element["r"]), float(element["geodesic_angle_deg"]), case)
        s1, s2, t12 = project_stress(mean, element["meridional"], element["circumferential"], angle)
        mat = materials["composites"][ply_material_name_for_zone(case, ply, element["zone"])]
        fi = hashin_mode_parts(s1, s2, t12, mat)["fiber_stress_ratio_abs"]
        worst[element_id] = max(worst.get(element_id, 0.0), fi)
    return worst


def write_vtk(preview: Path, field: dict[int, float], out: Path) -> None:
    text = preview.read_text(encoding="utf-8")
    n_cells = int(text.split("CELL_DATA")[1].split()[0])
    # Elements that received no stress sample are written as -1 and drawn grey, not as 0.
    values = "\n".join(f"{field[i + 1]:.6f}" if i + 1 in field else "-1" for i in range(n_cells))
    out.write_text(text.rstrip() + f"\nSCALARS fiber_fi float 1\nLOOKUP_TABLE default\n{values}\n", encoding="utf-8")


PV_SCRIPT = """
from paraview.simple import *
src = LegacyVTKReader(FileNames=[r"{vtk}"])
view = CreateView("RenderView")
view.ViewSize = [1400, 900]
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
def band(lo, hi):
    t = Threshold(Input=src)
    t.Scalars = ["CELLS", "fiber_fi"]
    try:
        t.LowerThreshold = lo
        t.UpperThreshold = hi
        t.ThresholdMethod = "Between"
    except AttributeError:
        t.ThresholdRange = [lo, hi]
    return t
missing = Show(band(-2.0, -0.5), view)
missing.ColorArrayName = ["CELLS", ""]
missing.AmbientColor = missing.DiffuseColor = [0.78, 0.78, 0.78]
missing.SetRepresentationType("Surface With Edges")
missing.EdgeColor = [0.3, 0.3, 0.3]
disp = Show(band(0.0, 10.0), view)
ColorBy(disp, ("CELLS", "fiber_fi"))
lut = GetColorTransferFunction("fiber_fi")
for preset in ("Inferno", "Inferno (matplotlib)", "Viridis (matplotlib)"):
    try:
        lut.ApplyPreset(preset, True)
        break
    except RuntimeError:
        pass
lut.RescaleTransferFunction(0.0, {vmax})
disp.SetScalarBarVisibility(view, True)
bar = GetScalarBar(lut, view)
bar.Title = "fibre index |s11|/X"
bar.ComponentTitle = ""
bar.TitleColor = [0, 0, 0]
bar.LabelColor = [0, 0, 0]
bar.TitleFontSize = 20
bar.LabelFontSize = 18
bar.RangeLabelFormat = "%.2f"
bar.WindowLocation = "Any Location"
bar.Position = [0.86, 0.15]
bar.ScalarBarLength = 0.6
disp.EdgeColor = [0.3, 0.3, 0.3]
disp.SetRepresentationType("Surface With Edges")
view.CameraPosition = [-650, -900, -700]
view.CameraFocalPoint = [0, 0, -40]
view.CameraViewUp = [1, 0, 0]
view.ResetCamera(False)
GetActiveCamera().Zoom(1.15)
SaveScreenshot(r"{png}", view, ImageResolution=[1400, 900], TransparentBackground=0)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case-id", default="11l_0122")
    parser.add_argument("--out", default=str(FIG_DIR / "fiber_index_field_11l_0122.png"))
    args = parser.parse_args()

    case = {c["case_id"]: c for c in load_cases()}[args.case_id]
    case_dir = SIM_RESULTS / "calculix" / args.case_id
    field = element_fiber_index(case, case_dir)
    vmax = max(field.values())
    tmp = Path(tempfile.mkdtemp())
    vtk = tmp / f"{args.case_id}_fiber_fi.vtk"
    write_vtk(case_dir / "calculix_composite_shell_preview.vtk", field, vtk)
    script = tmp / "render.py"
    script.write_text(PV_SCRIPT.format(vtk=vtk, png=Path(args.out).resolve(), vmax=round(vmax, 2)), encoding="utf-8")
    subprocess.run([str(PVPYTHON), str(script)], check=True)
    summary = {"case_id": args.case_id, "max_element_fiber_fi": vmax, "elements": len(field)}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
