#!/usr/bin/env python3
"""
Generate a static web report for the Type IV COPV comparison.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "type4_copv_comparison_output"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: object, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return str(value)


def table_from_rows(rows: list[dict], columns: list[tuple[str, str]], max_rows: int | None = None) -> str:
    shown = rows[:max_rows] if max_rows else rows
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in shown:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if key not in {"name"}:
                value = fmt(value, 4)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    note = ""
    if max_rows and len(rows) > max_rows:
        note = f"<p class=\"table-note\">Table tronquee: {max_rows} lignes affichees sur {len(rows)}. Les CSV complets sont dans le dossier.</p>"
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>{note}"


def layer_sequence(layers: list[dict]) -> str:
    rows = []
    for idx, layer in enumerate(layers, start=1):
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(layer['name'])}</td>"
            f"<td>{fmt(layer['angle_deg'], 1)}</td>"
            f"<td>{fmt(layer['thickness_mm'], 3)}</td>"
            "</tr>"
        )
    return "<div class=\"table-wrap\"><table><thead><tr><th>#</th><th>Pli</th><th>Angle deg</th><th>Epaisseur mm</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def image_card(src: str, title: str, caption: str) -> str:
    return f"""
    <figure class="figure-card">
      <img src="{html.escape(src)}" alt="{html.escape(title)}" loading="lazy">
      <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption>
    </figure>
    """


def generate() -> Path:
    summary = read_json(OUT / "comparison_summary.json")
    common = read_json(OUT / "type4_common_input_expanded.json")
    genetic_layers = read_csv(OUT / "genetic_layer_results.csv")
    laminate_layers = read_csv(OUT / "laminate_layer_results.csv")
    openfoam_rows = read_csv(OUT / "openfoam_coherence_panel.csv") if (OUT / "openfoam_coherence_panel.csv").exists() else []

    g = summary["genetic"]
    l = summary["laminate"]
    layers = common["layers"]

    html_text = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport Type IV COPV hydrogene</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5b6674;
      --line: #dbe2ea;
      --accent: #0b6bcb;
      --accent-2: #c44d2d;
      --good: #1f8a4c;
      --warn: #b7791f;
      --shadow: 0 12px 30px rgba(20, 35, 55, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.55;
    }}
    header {{
      background: linear-gradient(120deg, #10243b, #194f78 56%, #7b3f2f);
      color: white;
      padding: 46px 6vw 36px;
    }}
    header h1 {{
      margin: 0;
      font-size: clamp(30px, 5vw, 56px);
      line-height: 1.02;
      letter-spacing: 0;
    }}
    header p {{
      max-width: 980px;
      font-size: 18px;
      margin: 18px 0 0;
      color: rgba(255,255,255,0.86);
    }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(255,255,255,0.94);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
      padding: 10px 6vw;
      display: flex;
      gap: 10px;
      overflow-x: auto;
    }}
    nav a {{
      color: var(--text);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      white-space: nowrap;
      font-size: 14px;
      background: white;
    }}
    main {{
      padding: 28px 6vw 60px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 26px;
      margin: 22px 0;
    }}
    h2 {{
      font-size: 26px;
      margin: 0 0 14px;
    }}
    h3 {{
      margin: 22px 0 10px;
      font-size: 20px;
    }}
    .lead {{
      color: var(--muted);
      font-size: 17px;
      max-width: 1000px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fbfdff;
    }}
    .metric small {{
      display: block;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .metric strong {{
      display: block;
      font-size: 28px;
      color: var(--accent);
    }}
    .metric.warn strong {{ color: var(--accent-2); }}
    .metric.good strong {{ color: var(--good); }}
    .figure-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      overflow: hidden;
    }}
    .figure-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #eee;
    }}
    .figure-card figcaption {{
      padding: 12px 14px 14px;
      display: grid;
      gap: 4px;
    }}
    .figure-card figcaption span {{
      color: var(--muted);
      font-size: 14px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
      background: white;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    th {{
      background: #edf3fa;
      font-weight: 700;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{
      background: #edf3fa;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: #eef6ff;
      padding: 14px 16px;
      border-radius: 6px;
    }}
    .warning {{
      border-left-color: var(--accent-2);
      background: #fff4ed;
    }}
    .equation {{
      font-family: Consolas, monospace;
      background: #101820;
      color: white;
      border-radius: 8px;
      padding: 14px 16px;
      overflow-x: auto;
    }}
    .files a, .refs a {{
      color: var(--accent);
      overflow-wrap: anywhere;
    }}
    .table-note {{
      color: var(--muted);
      font-size: 13px;
    }}
    footer {{
      color: var(--muted);
      padding: 0 6vw 40px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    @media (max-width: 900px) {{
      .grid.two, .grid.three {{ grid-template-columns: 1fr; }}
      section {{ padding: 18px; }}
      header {{ padding-top: 34px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Rapport de simulation Type IV COPV hydrogene</h1>
    <p>Comparaison entre le code analytique <code>genetic</code>, un modele laminate membrane, et une visualisation du dome avec bandes de tow a largeur finie.</p>
  </header>
  <nav>
    <a href="#resume">Resume</a>
    <a href="#entrees">Entrees</a>
    <a href="#modeles">Modeles</a>
    <a href="#resultats">Resultats</a>
    <a href="#anisotrope">Anisotrope</a>
    <a href="#openfoam">OpenFOAM</a>
    <a href="#dome">Dome</a>
    <a href="#courbes">Courbes</a>
    <a href="#fichiers">Fichiers</a>
    <a href="#bibliographie">Bibliographie</a>
  </nav>
  <main>
    <section id="resume">
      <h2>Resume executif</h2>
      <p class="lead">La fenetre ParaView et la CAO FreeCAD ne sont pas necessaires pour lire les resultats: cette page regroupe les figures, les sorties numeriques et l'interpretation. Le point central est que le code <code>genetic</code> calcule correctement une zone cylindrique multicouche, tandis que le dome est actuellement un modele de visualisation geodesique a completer avant une vraie FEA.</p>
      <div class="grid three">
        <div class="metric good"><small>Max combined genetic</small><strong>{g['max_combined']:.3f}</strong></div>
        <div class="metric"><small>Max Tsai-Wu genetic</small><strong>{g['max_tsai_wu']:.3f}</strong></div>
        <div class="metric warn"><small>Max Tsai-Wu laminate</small><strong>{l['max_tsai_wu']:.3f}</strong></div>
      </div>
      <div class="callout warning" style="margin-top:16px">
        <strong>Attention:</strong> le resultat du dome ne doit pas etre compare directement a <code>genetic</code>, car <code>genetic</code> ne resout pas encore le dome. Il resout un cylindre epais multicouche.
      </div>
    </section>

    <section id="entrees">
      <h2>Donnees d'entree communes</h2>
      <div class="grid three">
        <div class="metric"><small>Rayon interne</small><strong>{common['geometry']['internal_radius_mm']:.1f} mm</strong></div>
        <div class="metric"><small>Pression de calcul</small><strong>{common['pressure']['calculation_pressure_mpa']:.1f} MPa</strong></div>
        <div class="metric"><small>Epaisseur totale</small><strong>{g['total_thickness_mm']:.2f} mm</strong></div>
      </div>
      <h3>Sequence de plis</h3>
      {layer_sequence(layers)}
    </section>

    <section id="modeles">
      <h2>Sur quoi la simulation est basee</h2>
      <div class="grid two">
        <div>
          <h3>Modele <code>genetic</code></h3>
          <p>Le code local calcule un cylindre epais multicouche. Il discretise chaque pli dans le rayon avec <code>{common['mesh']['points_per_layer_genetic']}</code> points et calcule les contraintes dans l'epaisseur. Il donne Hashin, Tsai-Wu, Puck et un indice combine.</p>
          <div class="equation">sigma = sigma(r), r_interne <= r <= r_externe</div>
        </div>
        <div>
          <h3>Modele laminate membrane</h3>
          <p>Le modele CLT utilise les efforts membrane du cylindre mince et une matrice laminate equivalente. Il est utile pour un premier controle, mais il ne calcule pas le gradient radial ni la contrainte radiale.</p>
          <div class="equation">Nx = pR/2 ; Ny = pR ; A epsilon = N</div>
        </div>
      </div>
      <h3>Dome geodesique</h3>
      <p>Le dome est visualise avec une loi geodesique simple et une epaisseur locale variable. Cela represente beaucoup mieux le depot filamentaire que des lignes fines, car les tows ont une largeur et une accumulation pres du boss.</p>
      <div class="equation">sin(alpha_local) = r_boss / r_local</div>
    </section>

    <section id="resultats">
      <h2>Resultats numeriques</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Grandeur</th><th>genetic cylindre epais</th><th>laminate membrane</th></tr></thead>
          <tbody>
            <tr><td>Max Tsai-Wu</td><td>{g['max_tsai_wu']:.6f}</td><td>{l['max_tsai_wu']:.6f}</td></tr>
            <tr><td>Max Hashin</td><td>{g['max_hashin']:.6f}</td><td>n/a</td></tr>
            <tr><td>Max Puck</td><td>{g['max_puck']:.6f}</td><td>n/a</td></tr>
            <tr><td>Max combined</td><td>{g['max_combined']:.6f}</td><td>n/a</td></tr>
            <tr><td>Deplacement radial max</td><td>{g['max_radial_displacement_mm']:.6f} mm</td><td>n/a</td></tr>
            <tr><td>Deformation circonferentielle ey</td><td>n/a</td><td>{l['ey']:.6e}</td></tr>
          </tbody>
        </table>
      </div>
      <p class="lead">Le laminate est plus severe ici sur Tsai-Wu. Cela ne signifie pas que le dome est plus dangereux; cela signifie surtout que les formulations cylindre epais et membrane mince ne recuperent pas les contraintes de la meme maniere.</p>
      <h3>Resultats par pli</h3>
      {table_from_rows(genetic_layers, [('layer', '#'), ('name', 'Pli'), ('angle_deg', 'Angle deg'), ('max_tsai_wu', 'Tsai-Wu genetic'), ('max_puck', 'Puck'), ('max_combined', 'Combined')], max_rows=16)}
    </section>

    <section id="anisotrope">
      <h2>Analyse anisotrope couche par couche</h2>
      <p class="lead">Le rapport final dedie explique pourquoi OpenFOAM/Lame colle tres bien en isotrope equivalent, pourquoi l'ecart avec <code>genetic</code> est un ecart de modele, puis affiche les courbes Python avec des visuels couleur de l'orientation des plis.</p>
      <p><a href="anisotropic_layer_final_report.html">Ouvrir le rapport final anisotrope couche par couche</a></p>
      <div class="grid two" style="margin-top:16px">
        {image_card('anisotropic_01_radial_response.png', 'Profil radial anisotrope', 'Deplacement radial et contraintes locales couche par couche.')}
        {image_card('anisotropic_03_orientation_cutaway.png', 'Coupe coloree des plis', 'Chaque couleur correspond a une orientation de couche.')}
        {image_card('anisotropic_04_unwrapped_orientations.png', 'Developpe des orientations', 'Representation schematique des directions fibre par pli.')}
        {image_card('anisotropic_05_openfoam_limit_explained.png', 'OpenFOAM/Lame vs genetic', 'Difference entre erreur numerique isotrope et ecart de modele composite.')}
      </div>
    </section>

    <section id="openfoam">
      <h2>Campagne de coherence OpenFOAM / Python</h2>
      <p class="lead">J'ai construit une campagne a 10 cas et 10 parametres d'entree pour comparer les modeles sur un panel plus large: rayon interne, pression, trois angles de depot, trois epaisseurs de plis et deux modules orthotropes. Les memes donnees alimentent le solveur <code>genetic</code> et le modele laminate; des manifests OpenFOAM sont generes pour chaque cas.</p>
      <div class="callout warning">
        <strong>Conclusion scientifique actuelle:</strong> OpenFOAM.com v2512 est installe et les 10 cas ont tourne comme cylindres epais isotropes equivalents. Cela valide le niveau solveur/maillage/pression contre Lame, mais pas encore une FEA composite orthotrope couche par couche.
      </div>
      <p><a href="openfoam_coherence_report.html">Ouvrir le rapport scientifique OpenFOAM/Python detaille</a></p>
      <p><a href="openfoam_solid_run_report.html">Ouvrir le rapport des 10 runs OpenFOAM solidDisplacementFoam</a></p>
      <p><a href="anisotropic_layer_final_report.html">Ouvrir le rapport final anisotrope couche par couche</a></p>
      {table_from_rows(openfoam_rows, [('case_id', 'Cas'), ('inner_radius_mm', 'R int mm'), ('pressure_mpa', 'P MPa'), ('helical_angle_deg', 'Angle helix'), ('hoop_angle_deg', 'Angle hoop'), ('total_thickness_mm', 't total mm'), ('genetic_max_tsai_wu', 'Tsai-Wu genetic'), ('laminate_max_tsai_wu', 'Tsai-Wu CLT'), ('relative_delta_tsai_wu', 'Delta rel.'), ('openfoam_status', 'Statut OpenFOAM')], max_rows=10) if openfoam_rows else '<p>La campagne OpenFOAM/Python n a pas encore ete generee.</p>'}
      <div class="grid two" style="margin-top:16px">
        {image_card('openfoam_panel_01_failure_indices.png', 'Panel de rupture 10 cas', 'Hashin, Tsai-Wu et Puck sur chaque cas du panel.')}
        {image_card('openfoam_panel_02_tsaiwu_scatter.png', 'Scatter Tsai-Wu genetic vs CLT', 'La diagonale represente un accord parfait entre modeles.')}
        {image_card('openfoam_panel_03_relative_delta.png', 'Ecart relatif par cas', 'Ecart relatif entre le cylindre epais et le laminate membrane.')}
        {image_card('openfoam_panel_05_visual_coverage_audit.png', 'Audit de couverture des tows', 'Comparaison entre ancien rendu sous-echantillonne et rendu dense.')}
        {image_card('openfoam_solid_01_displacement_compare.png', 'Runs OpenFOAM: deplacements', 'Comparaison deplacement radial entre genetic, OpenFOAM et Lame.')}
        {image_card('openfoam_solid_02_lame_error.png', 'Runs OpenFOAM: erreur Lame', 'Controle numerique du cylindre epais OpenFOAM.')}
      </div>
    </section>

    <section id="dome">
      <h2>Dome et depot filamentaire</h2>
      <p class="lead">C'est la partie interessante pour l'enroulement. La visualisation ci-dessous montre des bandes a largeur finie au lieu de fils. Le rendu complet utilise maintenant une densite de bandes estimee depuis la largeur du tow et la circonference, avec recouvrement visuel, pour eviter les zones artificiellement non couvertes.</p>
      <div class="grid">
        {image_card('paraview_full_tank_with_tows.png', 'Tank complet rendu avec ParaView', 'Cylindre, deux domes et bandes de tow a largeur finie sur toute la bouteille.')}
      </div>
      <h3>Zoom sur le dome</h3>
      <div class="grid two">
        {image_card('paraview_dome_angle_thickness.png', 'Dome geodesique avec bandes de tow', 'Bandes rouges/bleues = directions opposees; surface coloree = facteur local d epaississement.')}
        {image_card('paraview_dome_finite_width_tows.png', 'Bandes de tow a largeur finie', 'Les trajectoires ne sont plus de simples cordes: chaque tow est une bande surfacique.')}
      </div>
      <div class="callout" style="margin-top:16px">
        Pour passer de cette visualisation a une vraie simulation mecanique du dome, il faut convertir ces bandes en couches/sections locales avec orientation <code>alpha(r)</code>, epaisseur locale <code>t(r)</code>, et raffinement de maillage au boss et a la jonction cylindre-dome.
      </div>
    </section>

    <section id="courbes">
      <h2>Courbes generees</h2>
      <div class="grid two">
        {image_card('01_failure_indices_by_layer.png', 'Indices de rupture par pli', 'Comparaison Hashin, Tsai-Wu, Puck et laminate Tsai-Wu.')}
        {image_card('02_stress_compare_by_layer.png', 'Contraintes locales par pli', 'sigma1, sigma2 et tau12 pour genetic vs laminate.')}
        {image_card('03_radial_failure_profile_genetic.png', 'Profil radial genetic', 'Evolution des indices dans l epaisseur du cylindre.')}
        {image_card('04_layup_angles_thickness.png', 'Angles et epaisseurs', 'Verification visuelle de la sequence commune.')}
        {image_card('05_tsaiwu_delta_laminate_minus_genetic.png', 'Delta Tsai-Wu', 'Difference layer-wise entre laminate et genetic.')}
        {image_card('paraview_type4_cylinder_layer_cutaway.png', 'Coupe cylindrique ParaView', 'Couches equivalentes colorees par index de pli.')}
      </div>
    </section>

    <section id="fichiers" class="files">
      <h2>Fichiers produits</h2>
      <ul>
        <li><a href="type4_common_input.json">type4_common_input.json</a> - entree commune</li>
        <li><a href="comparison_summary.json">comparison_summary.json</a> - synthese numerique</li>
        <li><a href="genetic_layer_results.csv">genetic_layer_results.csv</a> - resultats par pli genetic</li>
        <li><a href="genetic_radial_points.csv">genetic_radial_points.csv</a> - points radiaux genetic</li>
        <li><a href="laminate_layer_results.csv">laminate_layer_results.csv</a> - resultats laminate par pli</li>
        <li><a href="type4_dome_geodesic_shell.vtk">type4_dome_geodesic_shell.vtk</a> - dome geodesique pour ParaView</li>
        <li><a href="type4_dome_finite_width_tows.vtk">type4_dome_finite_width_tows.vtk</a> - bandes de tow a largeur finie</li>
        <li><a href="openfoam_coherence_report.html">openfoam_coherence_report.html</a> - campagne scientifique OpenFOAM/Python</li>
        <li><a href="openfoam_solid_run_report.html">openfoam_solid_run_report.html</a> - resultats des 10 runs OpenFOAM</li>
        <li><a href="openfoam_solid_results.csv">openfoam_solid_results.csv</a> - extractions numeriques OpenFOAM</li>
        <li><a href="anisotropic_layer_final_report.html">anisotropic_layer_final_report.html</a> - rapport final anisotrope couche par couche</li>
        <li><a href="anisotropic_layer_by_layer_summary.csv">anisotropic_layer_by_layer_summary.csv</a> - synthese anisotrope par couche</li>
        <li><a href="openfoam_coherence_panel.csv">openfoam_coherence_panel.csv</a> - panel 10 cas / 10 parametres</li>
        <li><a href="openfoam_case_manifests/">openfoam_case_manifests/</a> - manifests par cas pour preparer OpenFOAM</li>
        <li><a href="type4_full_tank_tow_coverage.json">type4_full_tank_tow_coverage.json</a> - audit de densite du rendu filamentaire</li>
        <li><a href="MODELING_BASIS.md">MODELING_BASIS.md</a> - base de simulation</li>
        <li><a href="DOME_MODEL_NOTE.md">DOME_MODEL_NOTE.md</a> - note specifique dome</li>
      </ul>
    </section>

    <section id="bibliographie" class="refs">
      <h2>Bibliographie</h2>
      <ol>
        <li>Wang et al., 2022, tow redistribution for hydrogen vessel domes: <a href="https://doi.org/10.3390/polym14050902">10.3390/polym14050902</a></li>
        <li>Park et al., 2002, winding angle changes through thickness: <a href="https://doi.org/10.1016/S0263-8223(01)00137-4">10.1016/S0263-8223(01)00137-4</a></li>
        <li>Jois et al., 2021, variable dome contour: <a href="https://doi.org/10.3390/jcs5020056">10.3390/jcs5020056</a></li>
        <li>Leh et al., 2015, 700-bar Type IV progressive failure: <a href="https://doi.org/10.1016/j.ijhydene.2015.05.061">10.1016/j.ijhydene.2015.05.061</a></li>
        <li>Alam et al., 2020, COPV design/development: <a href="https://doi.org/10.1016/j.jcomc.2020.100045">10.1016/j.jcomc.2020.100045</a></li>
        <li>SIMULIA Abaqus filament-wound COPV brief: <a href="https://www.3ds.com/fileadmin/PRODUCTS-SERVICES/SIMULIA/RESOURCES/IE-Filament-Wound-Composite-Pressure-Vessel-Analysis-05.pdf">PDF</a></li>
        <li>ISO 19881:2025, gaseous hydrogen land vehicle fuel containers: <a href="https://www.iso.org/standard/19881?browse=tc">ISO page</a></li>
      </ol>
    </section>
  </main>
  <footer>
    Rapport local genere automatiquement dans <code>genetic/type4_copv_comparison_output/index.html</code>. Ce document est un support de pre-dimensionnement et de comparaison, pas une certification de reservoir hydrogene.
  </footer>
</body>
</html>
"""
    out = OUT / "index.html"
    out.write_text(html_text, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(generate())
