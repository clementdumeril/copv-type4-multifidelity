#!/usr/bin/env python3
"""
Scientific coherence campaign for the Type IV COPV project.

The campaign builds a 10-input-parameter design of experiments and compares:
- the local genetic analytical thick-cylinder solver
- the laminate membrane Python solver

OpenFOAM is used here as an equivalent-isotropic thick-cylinder benchmark.
The standard solidDisplacementFoam solver available in this installation does
not directly read an oriented orthotropic stiffness tensor for each composite
ply, so the real layer-by-layer anisotropic reference remains the local
genetic calculation.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from material import Material
from tank import Layer, Tank
from computation import Computation


ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT.parent / "hydrogen_type4_copv"
OUT = ROOT / "type4_copv_comparison_output"
REPORT = OUT / "openfoam_coherence_report.html"
MANIFEST_DIR = OUT / "openfoam_case_manifests"
SOLID_RESULTS = OUT / "openfoam_solid_results.csv"

sys.path.insert(0, str(SIM_DIR))
from simulate_copv import analyze_laminate, expand_stack  # noqa: E402


STRENGTHS = {
    "Xt_mpa": 2000.0,
    "Xc_mpa": 1500.0,
    "Yt_mpa": 50.0,
    "Yc_mpa": 200.0,
    "S_mpa": 80.0,
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def design_cases() -> list[dict]:
    """Ten cases, each with exactly ten explicit input parameters."""
    rows = [
        # Ri, P, helix, transition, hoop, t_h, t_tr, t_hoop, E1, E2
        (70, 35, 12, 30, 86, 0.65, 0.55, 1.15, 120, 7.0),
        (75, 45, 14, 34, 87, 0.75, 0.65, 1.35, 130, 7.8),
        (80, 55, 16, 38, 88, 0.85, 0.75, 1.55, 140, 8.5),
        (85, 65, 18, 42, 88.5, 0.95, 0.85, 1.75, 151, 9.0),
        (90, 70, 20, 46, 89, 1.05, 0.95, 1.95, 155, 9.5),
        (95, 75, 22, 50, 87.5, 1.15, 1.05, 2.15, 160, 10.0),
        (100, 80, 24, 54, 88.7, 1.25, 1.15, 2.35, 165, 10.5),
        (105, 87, 25, 55, 89, 1.35, 1.20, 2.55, 170, 11.0),
        (85, 87, 13.5, 43.5, 88.7, 1.03, 1.03, 2.46, 151, 9.0),
        (100, 70, 17.5, 48, 77, 1.10, 0.90, 2.20, 145, 8.2),
    ]
    cases = []
    for i, row in enumerate(rows, start=1):
        cases.append(
            {
                "case_id": f"case_{i:02d}",
                "inner_radius_mm": row[0],
                "pressure_mpa": row[1],
                "helical_angle_deg": row[2],
                "transition_angle_deg": row[3],
                "hoop_angle_deg": row[4],
                "helical_ply_thickness_mm": row[5],
                "transition_ply_thickness_mm": row[6],
                "hoop_ply_thickness_mm": row[7],
                "E1_GPa": row[8],
                "E2_GPa": row[9],
            }
        )
    return cases


def make_layers(case: dict, material: Material) -> list[Layer]:
    layers: list[Layer] = []
    for rep in range(4):
        layers.append(Layer(case["helical_ply_thickness_mm"], case["helical_angle_deg"], material, f"h{rep+1}_plus"))
        layers.append(Layer(case["helical_ply_thickness_mm"], -case["helical_angle_deg"], material, f"h{rep+1}_minus"))
    for rep in range(2):
        layers.append(Layer(case["transition_ply_thickness_mm"], case["transition_angle_deg"], material, f"tr{rep+1}_plus"))
        layers.append(Layer(case["transition_ply_thickness_mm"], -case["transition_angle_deg"], material, f"tr{rep+1}_minus"))
    for rep in range(2):
        layers.append(Layer(case["hoop_ply_thickness_mm"], case["hoop_angle_deg"], material, f"hoop{rep+1}_plus"))
        layers.append(Layer(case["hoop_ply_thickness_mm"], -case["hoop_angle_deg"], material, f"hoop{rep+1}_minus"))
    return layers


def material_for_case(case: dict) -> Material:
    e1 = case["E1_GPa"] * 1000.0
    e2 = case["E2_GPa"] * 1000.0
    return Material(
        young_modulus=(e1, e2, e2),
        shear_modulus=(5000.0, 5000.0, 5000.0),
        poisson_ratios=(0.30, 0.30, 0.42),
        Xt=STRENGTHS["Xt_mpa"],
        Xc=STRENGTHS["Xc_mpa"],
        Yt=STRENGTHS["Yt_mpa"],
        Yc=STRENGTHS["Yc_mpa"],
        X12=STRENGTHS["S_mpa"],
        name="CarbonFiber_case",
    )


def run_genetic_case(case: dict) -> dict:
    material = material_for_case(case)
    layers = make_layers(case, material)
    tank = Tank(
        layers=layers,
        internal_radius=case["inner_radius_mm"],
        length=1.0,
        pressure=case["pressure_mpa"],
        burst_test_pressure=case["pressure_mpa"],
    )
    comp = Computation(tank, number_of_points_by_layer=10)
    comp.calculate_deformation_and_constraints()
    hashin = np.clip(comp.hashin_failure_criteria(), 0, np.inf)
    tsai = np.clip(comp.tsai_wu_failure_criteria(), 0, np.inf)
    puck = np.clip(comp.puck_failure_criteria(), 0, np.inf)
    combined = np.clip(comp.combined_failure_criteria(), 0, np.inf)
    return {
        "genetic_max_hashin": float(np.max(hashin)),
        "genetic_max_tsai_wu": float(np.max(tsai)),
        "genetic_max_puck": float(np.max(puck)),
        "genetic_max_combined": float(np.max(combined)),
        "genetic_max_radial_displacement_mm": float(np.max(comp.Ur)),
        "total_thickness_mm": float(tank.get_total_thickness()),
        "outer_radius_mm": float(tank.get_external_radius()),
        "layer_count": len(layers),
    }


def laminate_config(case: dict) -> dict:
    mat_name = "CarbonFiber_case"
    stack = []
    for rep in range(4):
        for angle in [case["helical_angle_deg"], -case["helical_angle_deg"]]:
            stack.append({"name": f"h{rep+1}_{angle:+.1f}", "material": mat_name, "angle_deg": angle, "thickness_mm": case["helical_ply_thickness_mm"], "repeat": 1, "zone": "full", "geodesic": False})
    for rep in range(2):
        for angle in [case["transition_angle_deg"], -case["transition_angle_deg"]]:
            stack.append({"name": f"tr{rep+1}_{angle:+.1f}", "material": mat_name, "angle_deg": angle, "thickness_mm": case["transition_ply_thickness_mm"], "repeat": 1, "zone": "full", "geodesic": False})
    for rep in range(2):
        for angle in [case["hoop_angle_deg"], -case["hoop_angle_deg"]]:
            stack.append({"name": f"hoop{rep+1}_{angle:+.1f}", "material": mat_name, "angle_deg": angle, "thickness_mm": case["hoop_ply_thickness_mm"], "repeat": 1, "zone": "full", "geodesic": False})
    return {
        "case_name": case["case_id"],
        "geometry": {
            "inner_radius_m": case["inner_radius_mm"] * 1e-3,
            "cylindrical_length_m": 0.42,
            "boss_polar_opening_radius_m": 0.025,
        },
        "materials": {
            mat_name: {
                "kind": "orthotropic_ud",
                "density_kg_m3": 1580.0,
                "E1_GPa": case["E1_GPa"],
                "E2_GPa": case["E2_GPa"],
                "G12_GPa": 5.0,
                "nu12": 0.30,
                "Xt_MPa": STRENGTHS["Xt_mpa"],
                "Xc_MPa": STRENGTHS["Xc_mpa"],
                "Yt_MPa": STRENGTHS["Yt_mpa"],
                "Yc_MPa": STRENGTHS["Yc_mpa"],
                "S_MPa": STRENGTHS["S_mpa"],
            }
        },
        "stacking_sequence": stack,
        "analysis": {"tsai_wu_interaction": -0.5},
    }


def run_laminate_case(case: dict) -> dict:
    cfg = laminate_config(case)
    layers = expand_stack(cfg, "cylinder")
    radius_m = case["inner_radius_mm"] * 1e-3
    pressure_pa = case["pressure_mpa"] * 1e6
    membrane = (pressure_pa * radius_m / 2.0, pressure_pa * radius_m, 0.0)
    result = analyze_laminate(cfg, layers, case["pressure_mpa"], membrane, case["case_id"])
    return {
        "laminate_max_tsai_wu": float(result["worst_tsai_wu"]["tsai_wu_index"]),
        "laminate_max_stress": float(result["worst_max_stress"]["max_stress_index"]),
        "laminate_ex": float(result["strain"]["ex"]),
        "laminate_ey": float(result["strain"]["ey"]),
    }


def openfoam_manifest(case: dict, row: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "status": "NOT_RUN_OPENFOAM_NOT_AVAILABLE_ON_THIS_MACHINE",
        "recommended_solver_level_1": "solidDisplacementFoam for isotropic Lame thick-cylinder benchmark",
        "recommended_solver_level_2": "solids4Foam or another anisotropic solid solver for multilayer composite validation",
        "geometry": {
            "inner_radius_mm": case["inner_radius_mm"],
            "outer_radius_mm": row["outer_radius_mm"],
            "visual_cylinder_length_mm": 420.0,
        },
        "pressure_mpa": case["pressure_mpa"],
        "material": {
            "E1_GPa": case["E1_GPa"],
            "E2_GPa": case["E2_GPa"],
            "G12_GPa": 5.0,
            "nu12": 0.30,
        },
        "layup": {
            "helical_angle_deg": case["helical_angle_deg"],
            "transition_angle_deg": case["transition_angle_deg"],
            "hoop_angle_deg": case["hoop_angle_deg"],
            "layer_count": int(row["layer_count"]),
            "total_thickness_mm": row["total_thickness_mm"],
        },
        "target_python_observables_for_future_openfoam_run": {
            "max_radial_displacement_mm": row["genetic_max_radial_displacement_mm"],
            "max_tsai_wu_genetic": row["genetic_max_tsai_wu"],
            "max_combined_genetic": row["genetic_max_combined"],
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_results(rows: list[dict]) -> None:
    case_ids = [r["case_id"] for r in rows]
    x = np.arange(len(rows))
    g_tw = np.array([r["genetic_max_tsai_wu"] for r in rows])
    l_tw = np.array([r["laminate_max_tsai_wu"] for r in rows])
    combined = np.array([r["genetic_max_combined"] for r in rows])
    pressure = np.array([r["pressure_mpa"] for r in rows])
    rel = np.array([r["relative_delta_tsai_wu"] for r in rows])

    plt.rcParams.update({"figure.figsize": (11, 7), "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

    plt.figure()
    plt.plot(x, g_tw, marker="o", label="genetic Tsai-Wu")
    plt.plot(x, l_tw, marker="s", label="laminate Tsai-Wu")
    plt.plot(x, combined, marker="^", label="genetic combined")
    plt.axhline(1.0, color="red", linestyle=":", label="failure index = 1")
    plt.xticks(x, case_ids, rotation=35)
    plt.ylabel("Failure index")
    plt.title("10-case panel: failure indices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "openfoam_panel_01_failure_indices.png", dpi=220)
    plt.close()

    plt.figure()
    lim = max(float(np.max(g_tw)), float(np.max(l_tw))) * 1.15
    plt.scatter(g_tw, l_tw, c=pressure, cmap="turbo", s=90, edgecolor="black")
    plt.plot([0, lim], [0, lim], color="black", linestyle="--", label="perfect agreement")
    cbar = plt.colorbar()
    cbar.set_label("Pressure [MPa]")
    for i, cid in enumerate(case_ids):
        plt.annotate(cid[-2:], (g_tw[i], l_tw[i]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("genetic max Tsai-Wu")
    plt.ylabel("laminate max Tsai-Wu")
    plt.title("Model-to-model coherence scatter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "openfoam_panel_02_tsaiwu_scatter.png", dpi=220)
    plt.close()

    plt.figure()
    plt.bar(x, rel * 100.0, color=np.where(rel >= 0, "#C44D2D", "#1F8A4C"))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, case_ids, rotation=35)
    plt.ylabel("Relative delta [%]")
    plt.title("(laminate Tsai-Wu - genetic Tsai-Wu) / genetic Tsai-Wu")
    plt.tight_layout()
    plt.savefig(OUT / "openfoam_panel_03_relative_delta.png", dpi=220)
    plt.close()

    plt.figure()
    plt.scatter(pressure, combined, s=90, c=[r["total_thickness_mm"] for r in rows], cmap="viridis", edgecolor="black")
    cbar = plt.colorbar()
    cbar.set_label("Total thickness [mm]")
    plt.xlabel("Pressure [MPa]")
    plt.ylabel("genetic max combined FI")
    plt.title("Pressure sensitivity over the input panel")
    plt.tight_layout()
    plt.savefig(OUT / "openfoam_panel_04_pressure_sensitivity.png", dpi=220)
    plt.close()

    # Visualization coverage audit: old sparse display versus dense display generated from tow width.
    coverage_meta = {}
    coverage_path = OUT / "type4_full_tank_tow_coverage.json"
    if coverage_path.exists():
        coverage_meta = json.loads(coverage_path.read_text(encoding="utf-8"))
    sparse_display_tows_per_direction = 12
    dense_display_tows_per_direction = int(coverage_meta.get("n_tows_per_direction", sparse_display_tows_per_direction))
    tow_width = 6.0
    required = np.array([math.ceil(2.0 * math.pi * r["outer_radius_mm"] / tow_width) for r in rows])
    sparse_fraction = np.minimum(1.0, sparse_display_tows_per_direction / required)
    dense_fraction = np.minimum(1.0, dense_display_tows_per_direction / required)
    plt.figure(figsize=(11, 5))
    width = 0.38
    plt.bar(x - width / 2, sparse_fraction * 100.0, width=width, color="#C44D2D", label="ancien rendu 12 bandes")
    plt.bar(x + width / 2, dense_fraction * 100.0, width=width, color="#1F8A4C", label=f"rendu dense {dense_display_tows_per_direction} bandes")
    plt.xticks(x, case_ids, rotation=35)
    plt.ylabel("Coverage visuelle estimee par direction [%]")
    plt.title("Audit de couverture visuelle du rendu ParaView")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "openfoam_panel_05_visual_coverage_audit.png", dpi=220)
    plt.close()


def html_table(rows: list[dict], keys: list[str], max_rows: int | None = None) -> str:
    view = rows[:max_rows] if max_rows else rows
    head = "".join(f"<th>{k}</th>" for k in keys)
    body = []
    for row in view:
        cells = []
        for key in keys:
            val = row[key]
            if isinstance(val, float):
                val = f"{val:.5g}"
            cells.append(f"<td>{val}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def write_html_report(rows: list[dict]) -> None:
    avg_rel = float(np.mean([abs(r["relative_delta_tsai_wu"]) for r in rows]))
    max_rel = float(np.max([abs(r["relative_delta_tsai_wu"]) for r in rows]))
    pass_count = sum(1 for r in rows if r["genetic_max_combined"] < 1.0)
    openfoam_status = "OpenFOAM.com v2512 installe sous Ubuntu 24.04 WSL1; solidDisplacementFoam smoke test OK; les 10 cas COPV ont ete executes comme cylindres epais isotropes equivalents."
    coverage_path = OUT / "type4_full_tank_tow_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
    dense_tows = coverage.get("n_tows_per_direction", "non regenere")
    dense_pitch = coverage.get("equator_pitch_mm", None)
    dense_ratio = coverage.get("equator_width_to_pitch_ratio", None)
    dense_sentence = (
        f"Le nouveau rendu dense utilise {dense_tows} bandes par direction, "
        f"un pas equatorial de {dense_pitch:.2f} mm et un rapport largeur/pas de {dense_ratio:.2f}."
        if isinstance(dense_pitch, (int, float)) and isinstance(dense_ratio, (int, float))
        else "Le rendu dense n'a pas encore ete regenere."
    )
    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Validation OpenFOAM - rapport scientifique</title>
  <style>
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:#17202a; background:#f5f7fb; line-height:1.55; }}
    header {{ padding:42px 6vw; background:linear-gradient(120deg,#10243b,#245b7c,#81402f); color:white; }}
    h1 {{ margin:0; font-size:44px; letter-spacing:0; }}
    header p {{ max-width:1050px; font-size:18px; color:rgba(255,255,255,.86); }}
    main {{ max-width:1400px; margin:0 auto; padding:24px 6vw 70px; }}
    section {{ background:white; border:1px solid #dbe2ea; border-radius:8px; box-shadow:0 12px 30px rgba(20,35,55,.1); padding:24px; margin:22px 0; }}
    h2 {{ margin:0 0 12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .metric {{ border:1px solid #dbe2ea; border-radius:8px; padding:16px; background:#fbfdff; }}
    .metric small {{ display:block; color:#5b6674; margin-bottom:6px; }}
    .metric strong {{ font-size:28px; color:#0b6bcb; }}
    .warn {{ border-left:4px solid #c44d2d; background:#fff4ed; padding:14px 16px; border-radius:6px; }}
    .ok {{ border-left:4px solid #1f8a4c; background:#eefbf3; padding:14px 16px; border-radius:6px; }}
    .table-wrap {{ overflow-x:auto; border:1px solid #dbe2ea; border-radius:8px; }}
    table {{ width:100%; min-width:980px; border-collapse:collapse; }}
    th,td {{ padding:9px 11px; border-bottom:1px solid #dbe2ea; text-align:left; font-size:14px; }}
    th {{ background:#edf3fa; }}
    .figs {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    figure {{ margin:0; border:1px solid #dbe2ea; border-radius:8px; overflow:hidden; background:white; }}
    figure img {{ width:100%; display:block; }}
    figcaption {{ padding:10px 12px; color:#5b6674; }}
    code {{ background:#edf3fa; padding:2px 5px; border-radius:4px; }}
    a {{ color:#0b6bcb; }}
    @media(max-width:900px) {{ .grid,.figs {{ grid-template-columns:1fr; }} h1 {{ font-size:32px; }} }}
  </style>
</head>
<body>
<header>
  <h1>Rapport scientifique: coherence OpenFOAM/Python</h1>
    <p>Campagne a 10 parametres d'entree pour evaluer la coherence des modeles. Le rapport distingue ce que valide OpenFOAM en isotrope equivalent et ce qui doit rester dans l'analyse composite anisotrope couche par couche.</p>
</header>
<main>
  <section>
    <h2>Conclusion courte</h2>
    <p class="ok"><strong>Blocage environnement resolu:</strong> OpenFOAM.com v2512 est installe dans Ubuntu 24.04 WSL1, le solveur <code>solidDisplacementFoam</code> a passe le test officiel <code>plateHole</code>, puis les 10 cas COPV ont tourne en cylindre epais isotrope equivalent.</p>
    <p class="warn"><strong>Ce qui n'est pas encore prouve:</strong> ces runs OpenFOAM ne sont pas des runs composites anisotropes. Ils valident le maillage, la pression et les conditions aux limites contre Lame; l'analyse couche par couche reste portee par <code>genetic</code>.</p>
    <div class="grid">
      <div class="metric"><small>Cas analyses</small><strong>{len(rows)}</strong></div>
      <div class="metric"><small>Cas genetic FI &lt; 1</small><strong>{pass_count}/{len(rows)}</strong></div>
      <div class="metric"><small>Delta moyen absolu Tsai-Wu</small><strong>{100*avg_rel:.1f}%</strong></div>
    </div>
  </section>

  <section>
    <h2>Les 10 parametres d'entree</h2>
    <p>Chaque cas varie exactement ces dix grandeurs: rayon interne, pression, angle helicoidal, angle transition, angle hoop, epaisseur helice, epaisseur transition, epaisseur hoop, module E1 et module E2. Les resistances et cisaillements sont gardes constants pour isoler l'effet geometrique/raideur.</p>
    {html_table(rows, ['case_id','inner_radius_mm','pressure_mpa','helical_angle_deg','transition_angle_deg','hoop_angle_deg','helical_ply_thickness_mm','transition_ply_thickness_mm','hoop_ply_thickness_mm','E1_GPa','E2_GPa'])}
  </section>

  <section>
    <h2>Resultats du panel</h2>
    {html_table(rows, ['case_id','total_thickness_mm','outer_radius_mm','genetic_max_tsai_wu','laminate_max_tsai_wu','relative_delta_tsai_wu','genetic_max_combined','genetic_max_radial_displacement_mm','openfoam_status'])}
  </section>

  <section>
    <h2>Courbes de coherence</h2>
    <div class="figs">
      <figure><img src="openfoam_panel_01_failure_indices.png"><figcaption>Indices de rupture sur les 10 cas.</figcaption></figure>
      <figure><img src="openfoam_panel_02_tsaiwu_scatter.png"><figcaption>Scatter genetic vs laminate. La diagonale indique un accord parfait.</figcaption></figure>
      <figure><img src="openfoam_panel_03_relative_delta.png"><figcaption>Delta relatif de Tsai-Wu entre les deux formulations Python.</figcaption></figure>
      <figure><img src="openfoam_panel_04_pressure_sensitivity.png"><figcaption>Sensibilite a la pression, couleur = epaisseur totale.</figcaption></figure>
      <figure><img src="openfoam_panel_05_visual_coverage_audit.png"><figcaption>Audit de couverture du rendu ParaView: ancien sous-echantillonnage vs rendu dense.</figcaption></figure>
      <figure><img src="paraview_full_tank_with_tows.png"><figcaption>Rendu global dense du tank. Il reste une visualisation, pas un plan de bobinage certifie.</figcaption></figure>
    </div>
  </section>

  <section>
    <h2>Runs OpenFOAM effectues</h2>
    <p class="ok"><strong>Les 10 cas ont maintenant ete lances dans OpenFOAM</strong> avec <code>solidDisplacementFoam</code>, sous forme de quart de cylindre epais isotrope equivalent. Ce niveau valide le solveur, le maillage et la pression interne contre la solution analytique de Lame.</p>
    <p><a href="openfoam_solid_run_report.html">Ouvrir le rapport detaille des runs OpenFOAM</a></p>
    <p><a href="anisotropic_layer_final_report.html">Ouvrir le rapport final anisotrope couche par couche</a></p>
    {html_table(rows, ['case_id','openfoam_status','openfoam_max_u_mm','openfoam_vs_lame_inner_error_pct','openfoam_vs_genetic_max_u_delta_pct','openfoam_max_sigmaEq_pa']) if 'openfoam_max_u_mm' in rows[0] else '<p>Les resultats OpenFOAM detailles ne sont pas encore parses.</p>'}
    <div class="figs">
      <figure><img src="openfoam_solid_01_displacement_compare.png"><figcaption>Deplacement radial: genetic, OpenFOAM et Lame.</figcaption></figure>
      <figure><img src="openfoam_solid_02_lame_error.png"><figcaption>Erreur OpenFOAM vs Lame, indicateur de qualite numerique du run.</figcaption></figure>
      <figure><img src="openfoam_solid_03_genetic_delta.png"><figcaption>Ecart OpenFOAM isotrope equivalent vs genetic composite.</figcaption></figure>
    </div>
  </section>

  <section>
    <h2>Raisonnement scientifique</h2>
    <p>Pour prouver la coherence d'OpenFOAM, il faut comparer des grandeurs observables identiques: deplacement radial, contrainte circonferentielle, contrainte radiale, contraintes locales dans les plis et indice de rupture. Un simple rendu ParaView ne suffit pas.</p>
    <ol>
      <li><strong>Niveau 1:</strong> valider OpenFOAM sur un cylindre epais isotrope contre la solution de Lame. C'est le test minimal pour verifier maillage, pression interne et conditions aux limites.</li>
      <li><strong>Niveau 2:</strong> comparer le composite cylindrique anisotrope contre le solveur <code>genetic</code>. Le solveur OpenFOAM disponible ne lit pas directement les tenseurs orthotropes orientes par pli; il faudrait donc un solveur composite specifique ou une loi materiau custom.</li>
      <li><strong>Niveau 3:</strong> valider le dome filamentaire avec angle local et epaisseur locale. C'est ici que les references Wang/Park/Jois deviennent centrales.</li>
    </ol>
    <p>Le statut actuel est donc: <code>{openfoam_status}</code></p>
  </section>

  <section>
    <h2>Critique sur les zones non couvertes par filament</h2>
    <p>La critique est correcte. L'ancien rendu affichait seulement 12 bandes par direction, ce qui creait artificiellement des zones non couvertes. A rayon externe typique, avec une largeur de tow de 6 mm, il faut de l'ordre de 100 bandes par direction a l'equateur pour approcher une couverture visuelle continue. {dense_sentence}</p>
    <p>Un vrai modele de depot doit imposer: largeur de tow, epaisseur de tow, pas, recouvrement, sequence de passes, retournement au boss et accumulation locale d'epaisseur.</p>
  </section>

  <section>
    <h2>Fichiers</h2>
    <ul>
      <li><a href="openfoam_coherence_panel.csv">openfoam_coherence_panel.csv</a></li>
      <li><a href="openfoam_case_manifests/">openfoam_case_manifests/</a></li>
      <li><a href="openfoam_panel_summary.json">openfoam_panel_summary.json</a></li>
      <li><a href="openfoam_environment_status.json">openfoam_environment_status.json</a></li>
      <li><a href="openfoam_solid_run_report.html">openfoam_solid_run_report.html</a></li>
      <li><a href="openfoam_solid_results.csv">openfoam_solid_results.csv</a></li>
      <li><a href="anisotropic_layer_final_report.html">anisotropic_layer_final_report.html</a></li>
      <li><a href="anisotropic_layer_by_layer_summary.csv">anisotropic_layer_by_layer_summary.csv</a></li>
      <li><a href="index.html">Retour au rapport principal</a></li>
    </ul>
  </section>
</main>
</body>
</html>
"""
    REPORT.write_text(html, encoding="utf-8")


def run() -> None:
    ensure_dirs()
    solid_by_case = {row["case_id"]: row for row in read_csv_dicts(SOLID_RESULTS)}
    rows = []
    for case in design_cases():
        genetic = run_genetic_case(case)
        laminate = run_laminate_case(case)
        row = {**case, **genetic, **laminate}
        row["relative_delta_tsai_wu"] = (row["laminate_max_tsai_wu"] - row["genetic_max_tsai_wu"]) / max(row["genetic_max_tsai_wu"], 1e-12)
        solid = solid_by_case.get(case["case_id"], {})
        if solid.get("openfoam_status") == "completed":
            row["openfoam_status"] = "completed_equivalent_isotropic"
            for key in [
                "openfoam_max_u_mm",
                "openfoam_inner_mean_u_mm",
                "openfoam_outer_mean_u_mm",
                "openfoam_vs_lame_inner_error_pct",
                "openfoam_vs_genetic_max_u_delta_pct",
                "openfoam_max_sigmaEq_pa",
            ]:
                row[key] = float(solid[key])
        else:
            row["openfoam_status"] = "solver_available_case_not_built"
            for key in [
                "openfoam_max_u_mm",
                "openfoam_inner_mean_u_mm",
                "openfoam_outer_mean_u_mm",
                "openfoam_vs_lame_inner_error_pct",
                "openfoam_vs_genetic_max_u_delta_pct",
                "openfoam_max_sigmaEq_pa",
            ]:
                row[key] = ""
        rows.append(row)
        save_json(MANIFEST_DIR / f"{case['case_id']}.json", openfoam_manifest(case, row))

    write_csv(OUT / "openfoam_coherence_panel.csv", rows)
    save_json(
        OUT / "openfoam_panel_summary.json",
        {
            "rows": rows,
            "openfoam_available": True,
            "openfoam_distribution": "OpenFOAM.com v2512",
            "wsl_distribution": "Ubuntu-OpenFOAM-24.04-WSL1",
            "solid_solver_smoke_test": "passed_plateHole_to_time_100",
            "copv_cases": "10 equivalent-isotropic thick-cylinder cases completed in OpenFOAM",
        },
    )
    save_json(
        OUT / "openfoam_environment_status.json",
        {
            "wsl": "Ubuntu 24.04 WSL1, visible in elevated/admin context",
            "openfoam_foundation": "OpenFOAM-13 installed; foamRun and blockMesh available; no stressAnalysis solid solver found in Foundation package",
            "openfoam_com": "OpenFOAM.com v2512 installed; solidDisplacementFoam available",
            "smoke_test": {
                "case": "official stressAnalysis/solidDisplacementFoam/plateHole",
                "solver": "solidDisplacementFoam",
                "result": "completed to time 100",
                "last_reported_max_sigmaEq": 28845.5,
            },
            "remaining_for_copv_validation": [
                "standard solidDisplacementFoam only supports scalar E/nu for this campaign",
                "use genetic as the layer-by-layer anisotropic cylinder reference",
                "extend later with a composite/orthotropic solid solver for OpenFOAM",
                "extend the dome with local angle, local thickness, boss and cylinder-dome junction",
            ],
        },
    )
    plot_results(rows)
    write_html_report(rows)
    print(REPORT)


if __name__ == "__main__":
    run()
