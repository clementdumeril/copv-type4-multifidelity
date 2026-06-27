#!/usr/bin/env python3
"""
Deterministic Type IV COPV comparison output generator.

Outputs are written in ./type4_copv_comparison_output so the genetic folder
contains the common input, comparison curves, CSV/JSON summaries and ParaView
visualization files.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from material import Material
from tank import Layer, Tank
from computation import Computation


GENETIC_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = GENETIC_DIR.parent
SIM_DIR = WORKSPACE_DIR / "hydrogen_type4_copv"
OUT_DIR = GENETIC_DIR / "type4_copv_comparison_output"
COMMON_INPUT = SIM_DIR / "comparison_input.json"

sys.path.insert(0, str(SIM_DIR))
from simulate_copv import analyze_laminate, expand_stack  # noqa: E402


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_common_input() -> dict:
    with COMMON_INPUT.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_material(data: dict) -> Material:
    m = data["material"]
    return Material(
        young_modulus=tuple(m["young_modulus_mpa"]),
        shear_modulus=tuple(m["shear_modulus_mpa"]),
        poisson_ratios=tuple(m["poisson_ratios"]),
        Xt=m["Xt_mpa"],
        Xc=m["Xc_mpa"],
        Yt=m["Yt_mpa"],
        Yc=m["Yc_mpa"],
        X12=m["X12_mpa"],
        name=m["name"],
    )


def run_genetic(data: dict) -> dict:
    material = make_material(data)
    layers = [
        Layer(layer["thickness_mm"], layer["angle_deg"], material, layer["name"])
        for layer in data["layers"]
    ]
    tank = Tank(
        layers=layers,
        internal_radius=data["geometry"]["internal_radius_mm"],
        length=data["geometry"]["length_mm"],
        pressure=data["pressure"]["working_pressure_mpa"],
        burst_test_pressure=data["pressure"]["calculation_pressure_mpa"],
    )
    comp = Computation(tank, data["mesh"]["points_per_layer_genetic"])
    comp.calculate_deformation_and_constraints()

    hashin = np.clip(comp.hashin_failure_criteria(), 0, np.inf)
    tsai_wu = np.clip(comp.tsai_wu_failure_criteria(), 0, np.inf)
    puck = np.clip(comp.puck_failure_criteria(), 0, np.inf)
    combined = np.clip(comp.combined_failure_criteria(), 0, np.inf)

    ppl = comp.number_of_points_by_layer
    layer_rows = []
    point_rows = []
    for i, layer in enumerate(layers):
        start = i * ppl
        end = (i + 1) * ppl
        max_tw_local = int(np.argmax(tsai_wu[start:end]))
        max_tw_idx = start + max_tw_local
        layer_rows.append(
            {
                "layer": i + 1,
                "name": layer.name,
                "angle_deg": layer.angle,
                "thickness_mm": layer.thickness,
                "r_min_mm": float(comp.computation_radius[start]),
                "r_max_mm": float(comp.computation_radius[end - 1]),
                "sigma1_mpa_at_max_tsai_wu": float(comp.Sigo[max_tw_idx, 0]),
                "sigma2_mpa_at_max_tsai_wu": float(comp.Sigo[max_tw_idx, 1]),
                "tau12_mpa_at_max_tsai_wu": float(comp.Sigo[max_tw_idx, 5]),
                "max_hashin": float(np.max(hashin[start:end])),
                "max_tsai_wu": float(np.max(tsai_wu[start:end])),
                "max_puck": float(np.max(puck[start:end])),
                "max_combined": float(np.max(combined[start:end])),
            }
        )
        for j in range(start, end):
            point_rows.append(
                {
                    "layer": i + 1,
                    "name": layer.name,
                    "angle_deg": layer.angle,
                    "radius_mm": float(comp.computation_radius[j]),
                    "normalized_radius": float(comp.normalized_radius[j]),
                    "Ur_mm": float(comp.Ur[j]),
                    "sigma1_mpa": float(comp.Sigo[j, 0]),
                    "sigma2_mpa": float(comp.Sigo[j, 1]),
                    "sigma3_radial_mpa": float(comp.Sigo[j, 2]),
                    "tau12_mpa": float(comp.Sigo[j, 5]),
                    "hashin": float(hashin[j]),
                    "tsai_wu": float(tsai_wu[j]),
                    "puck": float(puck[j]),
                    "combined": float(combined[j]),
                }
            )

    return {
        "tank": tank,
        "computation": comp,
        "layer_rows": layer_rows,
        "point_rows": point_rows,
        "summary": {
            "max_hashin": float(np.max(hashin)),
            "max_tsai_wu": float(np.max(tsai_wu)),
            "max_puck": float(np.max(puck)),
            "max_combined": float(np.max(combined)),
            "max_radial_displacement_mm": float(np.max(comp.Ur)),
            "min_radial_displacement_mm": float(np.min(comp.Ur)),
            "total_thickness_mm": float(tank.get_total_thickness()),
        },
    }


def laminate_config(data: dict) -> dict:
    m = data["material"]
    return {
        "case_name": data["case_name"],
        "geometry": {
            "inner_radius_m": data["geometry"]["internal_radius_mm"] * 1e-3,
            "cylindrical_length_m": data["geometry"]["length_mm"] * 1e-3,
            "boss_polar_opening_radius_m": 25e-3,
        },
        "materials": {
            m["name"]: {
                "kind": "orthotropic_ud",
                "density_kg_m3": 1580.0,
                "E1_GPa": m["young_modulus_mpa"][0] / 1000.0,
                "E2_GPa": m["young_modulus_mpa"][1] / 1000.0,
                "G12_GPa": m["shear_modulus_mpa"][2] / 1000.0,
                "nu12": m["poisson_ratios"][0],
                "Xt_MPa": m["Xt_mpa"],
                "Xc_MPa": m["Xc_mpa"],
                "Yt_MPa": m["Yt_mpa"],
                "Yc_MPa": m["Yc_mpa"],
                "S_MPa": m["X12_mpa"],
            }
        },
        "stacking_sequence": [
            {
                "name": layer["name"],
                "material": m["name"],
                "angle_deg": layer["angle_deg"],
                "thickness_mm": layer["thickness_mm"],
                "repeat": 1,
                "zone": "full",
                "geodesic": False,
            }
            for layer in data["layers"]
        ],
        "analysis": {"tsai_wu_interaction": -0.5},
    }


def run_laminate(data: dict) -> dict:
    config = laminate_config(data)
    layers = expand_stack(config, "cylinder")
    radius_m = data["geometry"]["internal_radius_mm"] * 1e-3
    pressure_pa = data["pressure"]["calculation_pressure_mpa"] * 1e6
    membrane = (pressure_pa * radius_m / 2.0, pressure_pa * radius_m, 0.0)
    result = analyze_laminate(
        config,
        layers,
        data["pressure"]["calculation_pressure_mpa"],
        membrane,
        "laminate_membrane_cylinder",
    )
    layer_rows = []
    for row in result["rows"]:
        layer_rows.append(
            {
                "layer": row["ply"],
                "name": row["group"],
                "angle_deg": row["angle_deg"],
                "thickness_mm": row["thickness_mm"],
                "sigma1_mpa": row["sigma1_MPa"],
                "sigma2_mpa": row["sigma2_MPa"],
                "tau12_mpa": row["tau12_MPa"],
                "max_stress_index": row["max_stress_index"],
                "tsai_wu_index": row["tsai_wu_index"],
            }
        )
    return {
        "layer_rows": layer_rows,
        "summary": {
            "max_tsai_wu": float(result["worst_tsai_wu"]["tsai_wu_index"]),
            "max_stress_index": float(result["worst_max_stress"]["max_stress_index"]),
            "ex": float(result["strain"]["ex"]),
            "ey": float(result["strain"]["ey"]),
            "gxy": float(result["strain"]["gxy"]),
            "total_thickness_mm": float(result["total_thickness_mm"]),
        },
    }


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_failure_by_layer(genetic: dict, laminate: dict) -> None:
    apply_plot_style()
    x = np.array([r["layer"] for r in genetic["layer_rows"]])
    plt.figure()
    plt.plot(x, [r["max_hashin"] for r in genetic["layer_rows"]], marker="o", label="genetic Hashin")
    plt.plot(x, [r["max_tsai_wu"] for r in genetic["layer_rows"]], marker="s", label="genetic Tsai-Wu")
    plt.plot(x, [r["max_puck"] for r in genetic["layer_rows"]], marker="^", label="genetic Puck")
    plt.plot(x, [r["max_combined"] for r in genetic["layer_rows"]], linewidth=2.5, label="genetic combined")
    plt.plot(x, [r["tsai_wu_index"] for r in laminate["layer_rows"]], linestyle="--", marker="d", label="laminate Tsai-Wu")
    plt.axhline(1.0, color="red", linestyle=":", label="failure index = 1")
    plt.xlabel("Layer index")
    plt.ylabel("Failure index")
    plt.title("Layer-wise failure index comparison")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_failure_indices_by_layer.png", dpi=220)
    plt.close()


def plot_stress_compare(genetic: dict, laminate: dict) -> None:
    apply_plot_style()
    x = np.array([r["layer"] for r in genetic["layer_rows"]])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    pairs = [
        ("sigma1_mpa_at_max_tsai_wu", "sigma1_mpa", "sigma1 fiber direction [MPa]"),
        ("sigma2_mpa_at_max_tsai_wu", "sigma2_mpa", "sigma2 transverse [MPa]"),
        ("tau12_mpa_at_max_tsai_wu", "tau12_mpa", "tau12 shear [MPa]"),
    ]
    for ax, (g_key, l_key, label) in zip(axes, pairs):
        ax.plot(x, [r[g_key] for r in genetic["layer_rows"]], marker="o", label="genetic")
        ax.plot(x, [r[l_key] for r in laminate["layer_rows"]], linestyle="--", marker="s", label="laminate")
        ax.set_ylabel(label)
        ax.legend()
    axes[-1].set_xlabel("Layer index")
    fig.suptitle("Local stress comparison by layer")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_stress_compare_by_layer.png", dpi=220)
    plt.close(fig)


def plot_radial_profile(genetic: dict) -> None:
    apply_plot_style()
    rows = genetic["point_rows"]
    r = np.array([row["radius_mm"] for row in rows])
    plt.figure()
    plt.plot(r, [row["hashin"] for row in rows], label="Hashin")
    plt.plot(r, [row["tsai_wu"] for row in rows], label="Tsai-Wu")
    plt.plot(r, [row["puck"] for row in rows], label="Puck")
    plt.plot(r, [row["combined"] for row in rows], linewidth=2.5, label="Combined")
    plt.axhline(1.0, color="red", linestyle=":", label="failure index = 1")
    for layer in genetic["layer_rows"]:
        plt.axvline(layer["r_max_mm"], color="0.75", linewidth=0.6)
    plt.xlabel("Radius [mm]")
    plt.ylabel("Failure index")
    plt.title("Genetic thick-cylinder radial failure profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_radial_failure_profile_genetic.png", dpi=220)
    plt.close()


def plot_layup(data: dict) -> None:
    apply_plot_style()
    x = np.arange(1, len(data["layers"]) + 1)
    angles = [layer["angle_deg"] for layer in data["layers"]]
    thickness = [layer["thickness_mm"] for layer in data["layers"]]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x, angles, color="#4C78A8", alpha=0.75, label="angle")
    ax1.set_ylabel("Angle [deg]")
    ax1.set_xlabel("Layer index")
    ax2 = ax1.twinx()
    ax2.plot(x, thickness, color="#F58518", marker="o", label="thickness")
    ax2.set_ylabel("Thickness [mm]")
    ax1.set_title("Common layup input: angle and thickness")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_layup_angles_thickness.png", dpi=220)
    plt.close(fig)


def plot_tsaiwu_delta(genetic: dict, laminate: dict) -> None:
    apply_plot_style()
    x = np.array([r["layer"] for r in genetic["layer_rows"]])
    g = np.array([r["max_tsai_wu"] for r in genetic["layer_rows"]])
    l = np.array([r["tsai_wu_index"] for r in laminate["layer_rows"]])
    plt.figure()
    plt.bar(x, l - g, color=np.where(l - g >= 0, "#D62728", "#2CA02C"))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Layer index")
    plt.ylabel("Laminate Tsai-Wu - genetic Tsai-Wu")
    plt.title("Tsai-Wu model delta by layer")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_tsaiwu_delta_laminate_minus_genetic.png", dpi=220)
    plt.close()


def make_vtk(genetic: dict, laminate: dict) -> Path:
    layer_lookup = {row["layer"]: row for row in laminate["layer_rows"]}
    vtk_path = OUT_DIR / "type4_cylinder_comparison_layers.vtk"
    n_theta = 96
    n_z = 36
    z_len = 420.0
    points = []
    polygons = []
    cell_values = {
        "layer_index": [],
        "angle_deg": [],
        "thickness_mm": [],
        "genetic_tsai_wu": [],
        "laminate_tsai_wu": [],
        "delta_tsai_wu": [],
        "genetic_combined": [],
    }
    for layer in genetic["layer_rows"]:
        r_mid = 0.5 * (layer["r_min_mm"] + layer["r_max_mm"])
        point_offset = len(points)
        for iz in range(n_z + 1):
            z = -z_len / 2.0 + z_len * iz / n_z
            for it in range(n_theta):
                theta = 2.0 * math.pi * it / n_theta
                points.append((r_mid * math.cos(theta), r_mid * math.sin(theta), z))
        for iz in range(n_z):
            for it in range(n_theta):
                p0 = point_offset + iz * n_theta + it
                p1 = point_offset + iz * n_theta + ((it + 1) % n_theta)
                p2 = point_offset + (iz + 1) * n_theta + ((it + 1) % n_theta)
                p3 = point_offset + (iz + 1) * n_theta + it
                polygons.append((p0, p1, p2, p3))
                lam_tw = layer_lookup[layer["layer"]]["tsai_wu_index"]
                cell_values["layer_index"].append(layer["layer"])
                cell_values["angle_deg"].append(layer["angle_deg"])
                cell_values["thickness_mm"].append(layer["thickness_mm"])
                cell_values["genetic_tsai_wu"].append(layer["max_tsai_wu"])
                cell_values["laminate_tsai_wu"].append(lam_tw)
                cell_values["delta_tsai_wu"].append(lam_tw - layer["max_tsai_wu"])
                cell_values["genetic_combined"].append(layer["max_combined"])

    with vtk_path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Type IV COPV cylinder layer comparison\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        f.write(f"POLYGONS {len(polygons)} {len(polygons) * 5}\n")
        for poly in polygons:
            f.write(f"4 {poly[0]} {poly[1]} {poly[2]} {poly[3]}\n")
        f.write(f"CELL_DATA {len(polygons)}\n")
        for name, values in cell_values.items():
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in values:
                f.write(f"{float(value):.8f}\n")
    return vtk_path


def write_paraview_script(vtk_path: Path) -> Path:
    script_path = OUT_DIR / "open_type4_comparison_paraview.py"
    screenshot_path = OUT_DIR / "paraview_type4_cylinder_tsaiwu.png"
    layer_screenshot_path = OUT_DIR / "paraview_type4_cylinder_layer_cutaway.png"
    state_path = OUT_DIR / "type4_cylinder_comparison_state.pvsm"
    script = f'''from paraview.simple import *

vtk_file = r"{vtk_path}"
state_file = r"{state_path}"
screenshot_file = r"{screenshot_path}"
layer_screenshot_file = r"{layer_screenshot_path}"

_DisableFirstRenderCameraReset()
reader = LegacyVTKReader(registrationName="type4_cylinder_layers", FileNames=[vtk_file])
reader.UpdatePipeline()

clip = Clip(registrationName="quarter_cutaway", Input=reader)
clip.ClipType = "Plane"
clip.ClipType.Origin = [0.0, 0.0, 0.0]
clip.ClipType.Normal = [0.0, -1.0, 0.0]
clip.UpdatePipeline()

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1400, 900]
view.Background = [0.08, 0.08, 0.09]

display = Show(clip, view, "GeometryRepresentation")
display.Representation = "Surface With Edges"
display.Opacity = 0.92
display.EdgeColor = [0.02, 0.02, 0.02]
ColorBy(display, ("CELLS", "genetic_tsai_wu"))
display.SetScalarBarVisibility(view, True)

lut = GetColorTransferFunction("genetic_tsai_wu")
lut.ApplyPreset("Turbo", True)
lut.RescaleTransferFunction(0.25, 0.80)
pwf = GetOpacityTransferFunction("genetic_tsai_wu")
pwf.RescaleTransferFunction(0.25, 0.80)

view.CameraPosition = [360.0, -720.0, 420.0]
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.CameraViewUp = [-0.18, 0.35, 0.92]
view.CameraParallelScale = 260.0

Render(view)
SaveState(state_file)
SaveScreenshot(screenshot_file, view)

ColorBy(display, ("CELLS", "layer_index"))
display.SetScalarBarVisibility(view, True)
layer_lut = GetColorTransferFunction("layer_index")
layer_lut.ApplyPreset("Turbo", True)
layer_lut.RescaleTransferFunction(1.0, 16.0)
Render(view)
SaveScreenshot(layer_screenshot_file, view)

print("Saved ParaView state:", state_file)
print("Saved ParaView screenshot:", screenshot_file)
print("Saved ParaView layer cutaway:", layer_screenshot_file)
'''
    script_path.write_text(script, encoding="utf-8")
    return script_path


def write_modeling_basis(data: dict, summary: dict) -> None:
    text = f"""# Base de simulation et comparaison

## Ce qui est compare

Ce run compare deux modeles avec la meme entree:

- `genetic`: solveur analytique de cylindre epais multicouche, avec gradient radial dans l'epaisseur et criteres Hashin/Tsai-Wu/Puck.
- `laminate membrane`: modele CLT de cylindre mince equivalent, sans gradient radial, avec Tsai-Wu et max-stress.

La comparaison porte donc sur la zone cylindrique du reservoir Type IV. Le modele complet dôme + boss + contact liner/composite n'est pas encore inclus dans `genetic`.

## Entree commune

- Rayon interne: {data['geometry']['internal_radius_mm']} mm
- Pression de calcul: {data['pressure']['calculation_pressure_mpa']} MPa
- Nombre de plis: {len(data['layers'])}
- Epaisseur totale: {summary['genetic']['total_thickness_mm']:.3f} mm
- Materiau: {data['material']['name']} UD

## Resultats principaux

- Max Tsai-Wu `genetic`: {summary['genetic']['max_tsai_wu']:.6f}
- Max Puck `genetic`: {summary['genetic']['max_puck']:.6f}
- Max combined `genetic`: {summary['genetic']['max_combined']:.6f}
- Max Tsai-Wu `laminate membrane`: {summary['laminate']['max_tsai_wu']:.6f}

## Bibliographie utilisee pour cadrer le modele

1. Wang et al., 2022, thickness prediction with tow redistribution for hydrogen vessel domes: https://doi.org/10.3390/polym14050902
2. Park et al., 2002, winding angle changes through thickness direction: https://doi.org/10.1016/S0263-8223(01)00137-4
3. Jois et al., 2021, variable dome contour and cylinder-dome secondary stresses: https://doi.org/10.3390/jcs5020056
4. Leh et al., 2015, progressive failure analysis of 700-bar Type IV hydrogen COPV: https://doi.org/10.1016/j.ijhydene.2015.05.061
5. Alam et al., 2020, COPV design/development with WCM, FEA and burst testing: https://doi.org/10.1016/j.jcomc.2020.100045
6. SIMULIA Abaqus filament-wound COPV brief: https://www.3ds.com/fileadmin/PRODUCTS-SERVICES/SIMULIA/RESOURCES/IE-Filament-Wound-Composite-Pressure-Vessel-Analysis-05.pdf
7. ISO 19881:2025, gaseous hydrogen land vehicle fuel containers: https://www.iso.org/standard/19881?browse=tc

## Consequence pour la visualisation

Les courbes fines de trajectoire ne representent pas des filaments physiques. Un tow reel doit etre represente par une bande avec largeur et epaisseur. Le fichier ParaView produit ici montre donc des couches cylindriques equivalentes colorees par indice de rupture, pas une trajectoire de bobinage 3D complete.
"""
    (OUT_DIR / "MODELING_BASIS.md").write_text(text, encoding="utf-8")


def write_report(data: dict, summary: dict) -> None:
    text = f"""# Resultats Type IV COPV - dossier genetic

## Run

- Entree commune: `type4_common_input.json`
- Pression de calcul: {data['pressure']['calculation_pressure_mpa']} MPa
- Rayon interne: {data['geometry']['internal_radius_mm']} mm
- Nombre de plis: {len(data['layers'])}

## Comparaison globale

| Grandeur | genetic cylindre epais | laminate membrane |
|---|---:|---:|
| Max Tsai-Wu | {summary['genetic']['max_tsai_wu']:.6f} | {summary['laminate']['max_tsai_wu']:.6f} |
| Max Hashin | {summary['genetic']['max_hashin']:.6f} | n/a |
| Max Puck | {summary['genetic']['max_puck']:.6f} | n/a |
| Max combined | {summary['genetic']['max_combined']:.6f} | n/a |
| Deplacement radial max | {summary['genetic']['max_radial_displacement_mm']:.6f} mm | n/a |
| Strain hoop/ey | n/a | {summary['laminate']['ey']:.6e} |

## Courbes generees

- `01_failure_indices_by_layer.png`
- `02_stress_compare_by_layer.png`
- `03_radial_failure_profile_genetic.png`
- `04_layup_angles_thickness.png`
- `05_tsaiwu_delta_laminate_minus_genetic.png`

## ParaView

- Donnees: `type4_cylinder_comparison_layers.vtk`
- Script d'ouverture: `open_type4_comparison_paraview.py`
- Etat sauvegarde: `type4_cylinder_comparison_state.pvsm`
- Capture: `paraview_type4_cylinder_tsaiwu.png`
"""
    (OUT_DIR / "REPORT.md").write_text(text, encoding="utf-8")


def run() -> dict:
    ensure_output_dir()
    data = load_common_input()
    shutil.copy2(COMMON_INPUT, OUT_DIR / "type4_common_input.json")
    for source in ["BIBLIOGRAPHY.md", "GENETIC_AUDIT.md"]:
        src = SIM_DIR / source
        if src.exists():
            shutil.copy2(src, OUT_DIR / source)

    genetic = run_genetic(data)
    laminate = run_laminate(data)
    write_csv(OUT_DIR / "genetic_layer_results.csv", genetic["layer_rows"])
    write_csv(OUT_DIR / "genetic_radial_points.csv", genetic["point_rows"])
    write_csv(OUT_DIR / "laminate_layer_results.csv", laminate["layer_rows"])

    summary = {"input_case": data["case_name"], "genetic": genetic["summary"], "laminate": laminate["summary"]}
    write_json(OUT_DIR / "comparison_summary.json", summary)
    write_json(OUT_DIR / "type4_common_input_expanded.json", data)

    plot_failure_by_layer(genetic, laminate)
    plot_stress_compare(genetic, laminate)
    plot_radial_profile(genetic)
    plot_layup(data)
    plot_tsaiwu_delta(genetic, laminate)

    vtk_path = make_vtk(genetic, laminate)
    write_paraview_script(vtk_path)
    write_modeling_basis(data, summary)
    write_report(data, summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
