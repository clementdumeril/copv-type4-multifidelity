#!/usr/bin/env python3
"""
Build a focused report for the layer-by-layer anisotropic interpretation.

The report separates three things that were previously mixed together:
- OpenFOAM/Lame equivalent isotropic benchmark
- genetic anisotropic thick-cylinder layer-by-layer calculation
- laminate membrane CLT screening calculation
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.patches import Rectangle, Wedge

from type4_compare_outputs import run_genetic, run_laminate, load_common_input


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "type4_copv_comparison_output"
REPORT = OUT / "anisotropic_layer_final_report.html"
SUMMARY_JSON = OUT / "anisotropic_layer_summary.json"
LAYER_CSV = OUT / "anisotropic_layer_by_layer_summary.csv"
OPENFOAM_SOLID_CSV = OUT / "openfoam_solid_results.csv"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return str(value)


def html_table(rows: list[dict], columns: list[tuple[str, str]], max_rows: int | None = None) -> str:
    shown = rows[:max_rows] if max_rows else rows
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in shown:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if key not in {"name", "interpretation"}:
                value = fmt(value, 5)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def image_card(src: str, title: str, caption: str) -> str:
    return f"""
    <figure>
      <img src="{html.escape(src)}" alt="{html.escape(title)}" loading="lazy">
      <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption>
    </figure>
    """


def lame_profile(data: dict, laminate: dict) -> dict:
    ri = data["geometry"]["internal_radius_mm"] * 1e-3
    t = sum(layer["thickness_mm"] for layer in data["layers"]) * 1e-3
    ro = ri + t
    pressure = data["pressure"]["calculation_pressure_mpa"] * 1e6
    nu = 0.30
    ey = max(abs(laminate["summary"]["ey"]), 1e-12)
    nominal_hoop_stress = pressure * ri / t
    e_eq = nominal_hoop_stress / ey
    a = pressure * ri * ri / (ro * ro - ri * ri)
    b = pressure * ri * ri * ro * ro / (ro * ro - ri * ri)
    r = np.linspace(ri, ro, 260)
    u = ((1.0 - nu) * a * r + (1.0 + nu) * b / r) / e_eq
    return {
        "r_m": r,
        "r_mm": r * 1000.0,
        "u_mm": u * 1000.0,
        "u_inner_mm": float(u[0] * 1000.0),
        "u_outer_mm": float(u[-1] * 1000.0),
        "equivalent_E_GPa": float(e_eq / 1e9),
        "nu": nu,
        "nominal_hoop_stress_MPa": float(nominal_hoop_stress / 1e6),
    }


def layer_color(angle: float):
    norm = colors.Normalize(vmin=-90, vmax=90)
    return cm.coolwarm(norm(angle))


def plot_radial_response(genetic: dict, laminate: dict, data: dict, lame: dict) -> None:
    comp = genetic["computation"]
    rows = genetic["point_rows"]
    r = np.array([row["radius_mm"] for row in rows])
    angles_by_point = np.array([row["angle_deg"] for row in rows])
    ur = np.array([row["Ur_mm"] for row in rows])
    s1 = np.array([row["sigma1_mpa"] for row in rows])
    s2 = np.array([row["sigma2_mpa"] for row in rows])
    sr = np.array([row["sigma3_radial_mpa"] for row in rows])
    tau = np.array([row["tau12_mpa"] for row in rows])

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax = axes[0]
    for layer in genetic["layer_rows"]:
        ax.axvspan(layer["r_min_mm"], layer["r_max_mm"], color=layer_color(layer["angle_deg"]), alpha=0.08, linewidth=0)
    ax.plot(r, ur, color="#0b6bcb", linewidth=2.5, label="genetic anisotrope couche par couche")
    ax.plot(lame["r_mm"], lame["u_mm"], color="#c44d2d", linestyle="--", linewidth=2.2, label="Lame isotrope equivalent exact")
    ax.set_ylabel("Deplacement radial Ur [mm]")
    ax.set_title("Deplacement radial: anisotrope couche par couche vs isotrope equivalent")
    ax.legend(loc="best")

    ax = axes[1]
    for layer in genetic["layer_rows"]:
        ax.axvspan(layer["r_min_mm"], layer["r_max_mm"], color=layer_color(layer["angle_deg"]), alpha=0.08, linewidth=0)
    ax.plot(r, s1, label="sigma1 fibre", linewidth=2.0)
    ax.plot(r, s2, label="sigma2 transverse", linewidth=2.0)
    ax.plot(r, sr, label="sigma3 radial", linewidth=2.0)
    ax.plot(r, tau, label="tau12", linewidth=1.8)
    ax.set_xlabel("Rayon [mm]")
    ax.set_ylabel("Contrainte locale [MPa]")
    ax.set_title("Contraintes locales calculees par genetic dans chaque pli")
    ax.legend(ncol=2, loc="best")

    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-90, vmax=90), cmap=cm.coolwarm)
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Angle de pli [deg]")
    fig.tight_layout()
    fig.savefig(OUT / "anisotropic_01_radial_response.png", dpi=230)
    plt.close(fig)


def plot_failure_comparison(genetic: dict, laminate: dict) -> None:
    layers = genetic["layer_rows"]
    x = np.arange(1, len(layers) + 1)
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - 1.5 * width, [r["max_hashin"] for r in layers], width, label="Hashin genetic", color="#5DA5DA")
    ax.bar(x - 0.5 * width, [r["max_tsai_wu"] for r in layers], width, label="Tsai-Wu genetic", color="#F17CB0")
    ax.bar(x + 0.5 * width, [r["max_puck"] for r in layers], width, label="Puck genetic", color="#60BD68")
    ax.bar(x + 1.5 * width, [r["tsai_wu_index"] for r in laminate["layer_rows"]], width, label="Tsai-Wu CLT membrane", color="#FAA43A")
    ax.plot(x, [r["max_combined"] for r in layers], color="#111111", marker="o", linewidth=2.2, label="Combined genetic")
    ax.axhline(1.0, color="#c44d2d", linestyle=":", linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r["angle_deg"])) for r in layers], rotation=45)
    ax.set_xlabel("Pli, etiquette = angle [deg]")
    ax.set_ylabel("Indice de rupture")
    ax.set_title("Comparaison couche par couche des criteres de rupture")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "anisotropic_02_failure_layer_compare.png", dpi=230)
    plt.close(fig)


def plot_orientation_cutaway(genetic: dict, data: dict) -> None:
    layers = genetic["layer_rows"]
    ri = data["geometry"]["internal_radius_mm"]
    ro = ri + sum(layer["thickness_mm"] for layer in data["layers"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.1, 0.9]})
    ax = axes[0]
    for layer in layers:
        wedge = Wedge(
            center=(0, 0),
            r=layer["r_max_mm"],
            theta1=25,
            theta2=335,
            width=layer["r_max_mm"] - layer["r_min_mm"],
            facecolor=layer_color(layer["angle_deg"]),
            edgecolor="white",
            linewidth=0.8,
        )
        ax.add_patch(wedge)
    ax.add_patch(Wedge((0, 0), ri, 25, 335, facecolor="#f5f7fb", edgecolor="#777777", linewidth=0.7))
    ax.set_aspect("equal")
    lim = ro * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("Coupe annulaire: chaque couleur = orientation de pli")
    ax.axis("off")

    ax = axes[1]
    current = ri
    for i, layer in enumerate(layers, start=1):
        t = layer["r_max_mm"] - layer["r_min_mm"]
        ax.add_patch(Rectangle((0, current), 1.0, t, facecolor=layer_color(layer["angle_deg"]), edgecolor="white", linewidth=0.8))
        ax.text(1.04, current + 0.5 * t, f"{i:02d}  {layer['angle_deg']:+.1f} deg", va="center", fontsize=9)
        current += t
    ax.set_xlim(0, 1.75)
    ax.set_ylim(ri, ro)
    ax.set_xticks([])
    ax.set_ylabel("Rayon [mm]")
    ax.set_title("Empilement radial du liner vers l'exterieur")
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-90, vmax=90), cmap=cm.coolwarm)
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.06, pad=0.12)
    cbar.set_label("Angle de fibre [deg], signe = sens +theta/-theta")
    fig.tight_layout()
    fig.savefig(OUT / "anisotropic_03_orientation_cutaway.png", dpi=230)
    plt.close(fig)


def plot_unwrapped_orientations(genetic: dict) -> None:
    layers = genetic["layer_rows"]
    fig, ax = plt.subplots(figsize=(13, 8))
    for i, layer in enumerate(layers, start=1):
        angle = layer["angle_deg"]
        ax.add_patch(Rectangle((0, i - 0.43), 1.0, 0.86, facecolor=layer_color(angle), edgecolor="white", alpha=0.7))
        slope = math.tan(math.radians(max(-82.0, min(82.0, angle)))) * 0.16
        for offset in np.linspace(-0.3, 0.95, 6):
            x0, x1 = max(0.0, offset), min(1.0, offset + 0.32)
            y0 = i - 0.32 + slope * (x0 - offset)
            y1 = i - 0.32 + slope * (x1 - offset)
            ax.plot([x0, x1], [y0, y1], color="#111111", linewidth=1.4, alpha=0.75)
        ax.text(1.015, i, f"{i:02d}  {layer['name']}  {angle:+.1f} deg", va="center", fontsize=8.5)
    ax.set_xlim(0, 1.55)
    ax.set_ylim(0.35, len(layers) + 0.65)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(1, len(layers) + 1))
    ax.set_xlabel("Developpe cylindrique schematique: axe axial normalise")
    ax.set_ylabel("Ordre radial des plis")
    ax.set_title("Orientation des fibres dans chaque couche anisotrope")
    ax.spines[["top", "right"]].set_visible(False)
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-90, vmax=90), cmap=cm.coolwarm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Angle [deg]")
    fig.tight_layout()
    fig.savefig(OUT / "anisotropic_04_unwrapped_orientations.png", dpi=230)
    plt.close(fig)


def plot_openfoam_limit(openfoam_rows: list[dict]) -> None:
    if not openfoam_rows:
        return
    case_ids = [row["case_id"] for row in openfoam_rows]
    x = np.arange(len(case_ids))
    lame_err = np.array([abs(float(row["openfoam_vs_lame_inner_error_pct"])) for row in openfoam_rows])
    genetic_delta = np.array([abs(float(row["openfoam_vs_genetic_max_u_delta_pct"])) for row in openfoam_rows])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(x, lame_err, color="#1f8a4c")
    axes[0].set_ylabel("|Erreur| [%]")
    axes[0].set_title("OpenFOAM vs Lame: meme probleme isotrope, donc erreur numerique tres faible")
    axes[1].bar(x, genetic_delta, color="#c44d2d")
    axes[1].set_ylabel("|Delta| [%]")
    axes[1].set_title("OpenFOAM isotrope equivalent vs genetic composite anisotrope: ecart de modele")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(case_ids, rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "anisotropic_05_openfoam_limit_explained.png", dpi=230)
    plt.close(fig)


def build_layer_summary(genetic: dict, laminate: dict) -> list[dict]:
    laminate_lookup = {int(row["layer"]): row for row in laminate["layer_rows"]}
    rows = []
    comp = genetic["computation"]
    for row in genetic["layer_rows"]:
        idx = int(row["layer"]) - 1
        lam = laminate_lookup[int(row["layer"])]
        rows.append(
            {
                "layer": row["layer"],
                "name": row["name"],
                "angle_deg": row["angle_deg"],
                "thickness_mm": row["thickness_mm"],
                "r_min_mm": row["r_min_mm"],
                "r_max_mm": row["r_max_mm"],
                "anisotropic_exponent_b": float(comp.b[idx]),
                "coupling_a1": float(comp.a1[idx]),
                "coupling_a2": float(comp.a2[idx]),
                "genetic_max_hashin": row["max_hashin"],
                "genetic_max_tsai_wu": row["max_tsai_wu"],
                "genetic_max_puck": row["max_puck"],
                "genetic_max_combined": row["max_combined"],
                "clt_tsai_wu": lam["tsai_wu_index"],
            }
        )
    return rows


def make_report(data: dict, genetic: dict, laminate: dict, lame: dict, layer_summary: list[dict], openfoam_rows: list[dict]) -> str:
    g = genetic["summary"]
    l = laminate["summary"]
    genetic_max_u = g["max_radial_displacement_mm"]
    iso_outer_u = lame["u_outer_mm"]
    iso_delta = 100.0 * (iso_outer_u - genetic_max_u) / max(genetic_max_u, 1e-12)
    of_lame_avg = float("nan")
    of_genetic_avg = float("nan")
    if openfoam_rows:
        of_lame_avg = float(np.mean([abs(float(r["openfoam_vs_lame_inner_error_pct"])) for r in openfoam_rows]))
        of_genetic_avg = float(np.mean([abs(float(r["openfoam_vs_genetic_max_u_delta_pct"])) for r in openfoam_rows]))

    method_rows = [
        {
            "method": "OpenFOAM solidDisplacementFoam",
            "anisotropy": "non",
            "radial_gradient": "oui",
            "layer_angles": "non",
            "role": "benchmark numerique isotrope",
        },
        {
            "method": "Lame",
            "anisotropy": "non",
            "radial_gradient": "oui",
            "layer_angles": "non",
            "role": "solution analytique du meme benchmark isotrope",
        },
        {
            "method": "genetic",
            "anisotropy": "oui",
            "radial_gradient": "oui",
            "layer_angles": "oui",
            "role": "reference composite cylindrique couche par couche",
        },
        {
            "method": "CLT membrane",
            "anisotropy": "oui",
            "radial_gradient": "non",
            "layer_angles": "oui",
            "role": "controle rapide mince, pas une simulation epaisse",
        },
    ]

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport anisotrope couche par couche</title>
  <style>
    :root {{
      --bg:#f5f7fb;
      --panel:#ffffff;
      --text:#17202a;
      --muted:#5b6674;
      --line:#dbe2ea;
      --accent:#0b6bcb;
      --red:#c44d2d;
      --green:#1f8a4c;
      --shadow:0 12px 30px rgba(20,35,55,.1);
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:var(--text); background:var(--bg); line-height:1.55; }}
    header {{ padding:44px 6vw 36px; background:linear-gradient(120deg,#10243b,#205b78,#7b3f2f); color:white; }}
    header h1 {{ margin:0; font-size:clamp(30px,5vw,54px); line-height:1.04; letter-spacing:0; }}
    header p {{ max-width:1050px; color:rgba(255,255,255,.88); font-size:18px; }}
    nav {{ position:sticky; top:0; z-index:10; padding:10px 6vw; display:flex; gap:10px; overflow-x:auto; background:rgba(255,255,255,.94); border-bottom:1px solid var(--line); }}
    nav a {{ text-decoration:none; color:var(--text); border:1px solid var(--line); background:white; border-radius:6px; padding:8px 10px; white-space:nowrap; font-size:14px; }}
    main {{ max-width:1440px; margin:0 auto; padding:28px 6vw 70px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:var(--shadow); padding:26px; margin:22px 0; }}
    h2 {{ margin:0 0 12px; font-size:26px; }}
    h3 {{ margin:22px 0 10px; }}
    .lead {{ color:var(--muted); font-size:17px; max-width:1080px; }}
    .grid {{ display:grid; gap:16px; }}
    .two {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .three {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
    .metric {{ border:1px solid var(--line); border-radius:8px; padding:16px; background:#fbfdff; }}
    .metric small {{ display:block; color:var(--muted); margin-bottom:6px; }}
    .metric strong {{ display:block; font-size:28px; color:var(--accent); }}
    .metric.warn strong {{ color:var(--red); }}
    .metric.good strong {{ color:var(--green); }}
    .callout {{ border-left:4px solid var(--accent); background:#eef6ff; padding:14px 16px; border-radius:6px; }}
    .warn {{ border-left-color:var(--red); background:#fff4ed; }}
    .ok {{ border-left-color:var(--green); background:#eefbf3; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; min-width:850px; border-collapse:collapse; background:white; }}
    th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; font-size:14px; }}
    th {{ background:#edf3fa; }}
    figure {{ margin:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:white; }}
    figure img {{ display:block; width:100%; height:auto; }}
    figcaption {{ padding:12px 14px 14px; display:grid; gap:4px; }}
    figcaption span {{ color:var(--muted); font-size:14px; }}
    code {{ background:#edf3fa; padding:2px 5px; border-radius:4px; }}
    a {{ color:var(--accent); }}
    @media(max-width:900px) {{ .two,.three {{ grid-template-columns:1fr; }} section {{ padding:18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Analyse anisotrope couche par couche</h1>
    <p>Comparaison scientifique entre le benchmark OpenFOAM/Lame isotrope et la reference composite anisotrope calculee par <code>genetic</code>, avec les courbes Python et les orientations de plis affichees en couleur.</p>
  </header>
  <nav>
    <a href="#idee">Pourquoi 0.03%</a>
    <a href="#anisotrope">Analyse anisotrope</a>
    <a href="#methodes">Methodes</a>
    <a href="#courbes">Courbes</a>
    <a href="#orientation">Orientations</a>
    <a href="#dome">Dome</a>
    <a href="#fichiers">Fichiers</a>
  </nav>
  <main>
    <section id="idee">
      <h2>Pourquoi OpenFOAM colle a Lame a 0.03%</h2>
      <p class="lead">Le faible ecart ne prouve pas encore le composite. Il prouve que le probleme isotrope equivalent est bien pose.</p>
      <div class="grid three">
        <div class="metric good"><small>Erreur moyenne OpenFOAM vs Lame</small><strong>{of_lame_avg:.3f}%</strong></div>
        <div class="metric warn"><small>Ecart moyen OpenFOAM vs genetic</small><strong>{of_genetic_avg:.1f}%</strong></div>
        <div class="metric"><small>Module isotrope equivalent du cas commun</small><strong>{lame['equivalent_E_GPa']:.1f} GPa</strong></div>
      </div>
      <p>OpenFOAM et Lame resolvent ici le meme cylindre epais lineaire: meme rayon interne/externe, meme pression interne, meme condition exterieure libre, meme hypothese isotrope <code>E, nu</code>. L'erreur restante vient presque seulement de la discretisation du maillage et de l'extraction des valeurs aux faces.</p>
      <p class="callout warn"><strong>Donc:</strong> le 0.03% valide le solveur, le maillage, les conditions aux limites et la pression du benchmark. Il ne valide pas encore une bouteille Type IV composite avec plis orthotropes et orientation filamentaire.</p>
    </section>

    <section id="anisotrope">
      <h2>Pourquoi l'ecart de 17 a 28% apparait</h2>
      <p class="lead">L'isotrope equivalent remplace toute la stratification par un seul scalaire de raideur. Un seul <code>E_eq</code> peut approcher une compliance globale, mais il ne peut pas reproduire simultanement la direction fibre, la direction transverse, les couplages +theta/-theta, les contraintes locales et le gradient radial couche par couche.</p>
      <div class="grid three">
        <div class="metric"><small>Ur max genetic anisotrope</small><strong>{g['max_radial_displacement_mm']:.4f} mm</strong></div>
        <div class="metric warn"><small>Ur externe Lame isotrope equivalent</small><strong>{iso_outer_u:.4f} mm</strong></div>
        <div class="metric warn"><small>Delta isotrope equivalent vs genetic</small><strong>{iso_delta:.1f}%</strong></div>
      </div>
      <p>Dans <code>genetic</code>, chaque pli possede sa matrice orthotrope transformee par son angle. Les conditions imposees sont la continuite du deplacement radial et de la contrainte radiale aux interfaces, avec pression interne et surface externe libre. C'est beaucoup plus proche d'un reservoir filamentaire cylindrique qu'une seule couche isotrope.</p>
    </section>

    <section id="methodes">
      <h2>Comparaison des methodes</h2>
      {html_table(method_rows, [('method','Methode'),('anisotropy','Anisotropie'),('radial_gradient','Gradient radial'),('layer_angles','Angles de plis'),('role','Role scientifique')])}
      <p class="callout"><strong>Limite OpenFOAM actuelle:</strong> le solveur installe <code>solidDisplacementFoam</code> lit <code>E</code> et <code>nu</code> comme champs scalaires. Il peut varier l'isotrope par zone/couche, mais il ne lit pas directement un tenseur orthotrope oriente par pli. Pour une vraie FEA OpenFOAM anisotrope, il faudrait un solveur solide composite specifique ou une loi materiau tensorielle custom.</p>
    </section>

    <section id="courbes">
      <h2>Courbes Python et comparaison</h2>
      <div class="grid two">
        {image_card('anisotropic_01_radial_response.png', 'Profil radial anisotrope', 'Deplacement radial et contraintes locales calculees par genetic, comparees a Lame isotrope equivalent.')}
        {image_card('anisotropic_02_failure_layer_compare.png', 'Criteres de rupture par couche', 'Hashin, Tsai-Wu, Puck et comparaison CLT membrane.')}
        {image_card('anisotropic_05_openfoam_limit_explained.png', 'Pourquoi OpenFOAM/Lame colle mais pas OpenFOAM/genetic', 'Top: erreur numerique isotrope. Bas: ecart de modele composite.')}
        {image_card('01_failure_indices_by_layer.png', 'Courbe Python initiale: indices par pli', 'Sortie existante de la methode Python, conservee pour comparaison.')}
        {image_card('02_stress_compare_by_layer.png', 'Courbe Python initiale: contraintes par pli', 'Comparaison des contraintes locales genetic vs CLT.')}
        {image_card('03_radial_failure_profile_genetic.png', 'Courbe Python initiale: profil radial de rupture', 'Evolution des indices dans l epaisseur.')}
      </div>
    </section>

    <section id="orientation">
      <h2>Images de simulation: orientation des couches</h2>
      <p class="lead">Ces figures representent la stratification mecanique utilisee par l'analyse anisotrope. La couleur code l'angle du pli et le signe distingue les familles +theta et -theta.</p>
      <div class="grid two">
        {image_card('anisotropic_03_orientation_cutaway.png', 'Coupe coloree des plis anisotropes', 'Empilement radial du cylindre: chaque anneau correspond a une couche.')}
        {image_card('anisotropic_04_unwrapped_orientations.png', 'Developpe des orientations fibre', 'Vue schematique des directions de fibres dans chaque couche.')}
        {image_card('paraview_type4_cylinder_layer_cutaway.png', 'Vue ParaView des couches cylindriques', 'Cylindre multicouche colore par index de couche.')}
        {image_card('paraview_type4_cylinder_tsaiwu.png', 'Vue ParaView coloree par Tsai-Wu', 'Champ de rupture projete sur les couches cylindriques.')}
      </div>
    </section>

    <section id="dome">
      <h2>Dome et depot filamentaire</h2>
      <p class="lead">Le dome reste une visualisation geodesique avec bandes de tow a largeur finie. Il montre l'orientation locale et l'accumulation qualitative, mais il n'est pas encore couple au solveur mecanique couche par couche.</p>
      <div class="grid two">
        {image_card('paraview_full_tank_with_tows.png', 'Tank complet avec tows a largeur finie', 'Rendu dense corrigeant l ancien effet de zones artificiellement decouvertes.')}
        {image_card('paraview_dome_angle_thickness.png', 'Dome: angle local et epaisseur', 'Surface coloree et tows +/- pour voir la variation locale.')}
        {image_card('paraview_dome_finite_width_tows.png', 'Dome: bandes de tow', 'Les filaments sont affiches comme bandes, pas comme cordes infiniment fines.')}
        {image_card('openfoam_panel_05_visual_coverage_audit.png', 'Audit de couverture', 'Controle quantitatif simple de la densite de bandes affichees.')}
      </div>
    </section>

    <section>
      <h2>Resultats par couche</h2>
      {html_table(layer_summary, [('layer','#'),('name','Pli'),('angle_deg','Angle'),('thickness_mm','Epaisseur mm'),('anisotropic_exponent_b','b anisotrope'),('genetic_max_tsai_wu','Tsai-Wu genetic'),('genetic_max_puck','Puck'),('genetic_max_combined','Combined'),('clt_tsai_wu','Tsai-Wu CLT')], max_rows=16)}
    </section>

    <section id="fichiers">
      <h2>Fichiers produits</h2>
      <ul>
        <li><a href="anisotropic_layer_by_layer_summary.csv">anisotropic_layer_by_layer_summary.csv</a></li>
        <li><a href="anisotropic_layer_summary.json">anisotropic_layer_summary.json</a></li>
        <li><a href="genetic_radial_points.csv">genetic_radial_points.csv</a></li>
        <li><a href="openfoam_solid_results.csv">openfoam_solid_results.csv</a></li>
        <li><a href="openfoam_solid_run_report.html">Rapport OpenFOAM isotrope equivalent</a></li>
        <li><a href="index.html">Retour au rapport principal</a></li>
      </ul>
    </section>
  </main>
</body>
</html>
"""


def run() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_common_input()
    genetic = run_genetic(data)
    laminate = run_laminate(data)
    lame = lame_profile(data, laminate)
    layer_summary = build_layer_summary(genetic, laminate)
    openfoam_rows = read_csv(OPENFOAM_SOLID_CSV)

    plot_radial_response(genetic, laminate, data, lame)
    plot_failure_comparison(genetic, laminate)
    plot_orientation_cutaway(genetic, data)
    plot_unwrapped_orientations(genetic)
    plot_openfoam_limit(openfoam_rows)

    write_csv(LAYER_CSV, layer_summary)
    summary = {
        "case": data["case_name"],
        "genetic": genetic["summary"],
        "laminate": laminate["summary"],
        "lame_equivalent_isotropic": {
            "E_GPa": lame["equivalent_E_GPa"],
            "nu": lame["nu"],
            "u_inner_mm": lame["u_inner_mm"],
            "u_outer_mm": lame["u_outer_mm"],
            "delta_outer_vs_genetic_max_pct": 100.0
            * (lame["u_outer_mm"] - genetic["summary"]["max_radial_displacement_mm"])
            / max(genetic["summary"]["max_radial_displacement_mm"], 1e-12),
        },
        "openfoam_equivalent_isotropic_panel": {
            "case_count": len(openfoam_rows),
            "mean_abs_lame_error_pct": float(np.mean([abs(float(r["openfoam_vs_lame_inner_error_pct"])) for r in openfoam_rows])) if openfoam_rows else None,
            "mean_abs_genetic_delta_pct": float(np.mean([abs(float(r["openfoam_vs_genetic_max_u_delta_pct"])) for r in openfoam_rows])) if openfoam_rows else None,
            "interpretation": "OpenFOAM validates the isotropic thick-cylinder benchmark; it is not the anisotropic layer-by-layer model.",
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT.write_text(make_report(data, genetic, laminate, lame, layer_summary, openfoam_rows), encoding="utf-8")
    return REPORT


if __name__ == "__main__":
    print(run())
