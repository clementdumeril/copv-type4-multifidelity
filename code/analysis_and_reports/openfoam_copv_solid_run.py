#!/usr/bin/env python3
"""
Generate, parse, and report OpenFOAM solidDisplacementFoam runs for the
10-case Type IV COPV panel.

The OpenFOAM model is a first-level structural benchmark: a quarter thick
cylinder with symmetry planes, internal pressure, and an equivalent isotropic
Young modulus derived from the laminate hoop strain. It is intentionally not
yet a layer-by-layer anisotropic composite model.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "type4_copv_comparison_output"
RUN_ROOT = OUT / "openfoam_runs"
PANEL_CSV = OUT / "openfoam_coherence_panel.csv"
RESULT_CSV = OUT / "openfoam_solid_results.csv"
REPORT = OUT / "openfoam_solid_run_report.html"
WSL_RUNNER = OUT / "run_openfoam_solid_cases.sh"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str) -> float:
    return float(row[key])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def block_mesh_dict(ri: float, ro: float) -> str:
    z = 0.01
    nr = 18
    nt = 40
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM quarter annulus mesh for COPV thick-cylinder benchmark             |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale 1;

vertices
(
    ({ri:.10g} 0 0)
    ({ro:.10g} 0 0)
    (0 {ro:.10g} 0)
    (0 {ri:.10g} 0)
    ({ri:.10g} 0 {z:.10g})
    ({ro:.10g} 0 {z:.10g})
    (0 {ro:.10g} {z:.10g})
    (0 {ri:.10g} {z:.10g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nr} {nt} 1) simpleGrading (1 1 1)
);

edges
(
    arc 0 3 origin (0 0 0)
    arc 1 2 origin (0 0 0)
    arc 4 7 origin (0 0 {z:.10g})
    arc 5 6 origin (0 0 {z:.10g})
);

boundary
(
    theta0
    {{
        type symmetryPlane;
        faces ((0 1 5 4));
    }}
    theta90
    {{
        type symmetryPlane;
        faces ((3 2 6 7));
    }}
    inner
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outer
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
"""


def control_dict() -> str:
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     solidDisplacementFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         100;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""


def fv_schemes() -> str:
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

d2dt2Schemes
{
    default         steadyState;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         leastSquares;
    grad(D)         leastSquares;
    grad(T)         leastSquares;
}

divSchemes
{
    default         none;
    div(sigmaD)     Gauss linear;
}

laplacianSchemes
{
    default         none;
    laplacian(DD,D) Gauss linear corrected;
    laplacian(DT,T) Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         none;
}
"""


def fv_solution() -> str:
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    "(D|T)"
    {
        solver          GAMG;
        tolerance       1e-08;
        relTol          0.1;
        smoother        GaussSeidel;
        nCellsInCoarsestLevel 20;
    }
}

stressAnalysis
{
    compactNormalStress yes;
    nCorrectors     1;
    D               1e-08;
}
"""


def mechanical_properties(e_pa: float, nu: float) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      mechanicalProperties;
}}

rho
{{
    type        uniform;
    value       1580;
}}

nu
{{
    type        uniform;
    value       {nu:.10g};
}}

E
{{
    type        uniform;
    value       {e_pa:.10g};
}}

planeStress     yes;
"""


def thermal_properties() -> str:
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      thermalProperties;
}

C
{
    type        uniform;
    value       1000;
}

k
{
    type        uniform;
    value       1;
}

alpha
{
    type        uniform;
    value       0;
}

thermalStress   no;
"""


def displacement_field(pressure_pa: float) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      D;
}}

dimensions      [0 1 0 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    theta0
    {{
        type            symmetryPlane;
    }}
    theta90
    {{
        type            symmetryPlane;
    }}
    inner
    {{
        type            tractionDisplacement;
        traction        uniform (0 0 0);
        pressure        uniform {pressure_pa:.10g};
        value           uniform (0 0 0);
    }}
    outer
    {{
        type            tractionDisplacement;
        traction        uniform (0 0 0);
        pressure        uniform 0;
        value           uniform (0 0 0);
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""


def temperature_field() -> str:
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}

dimensions      [0 0 0 1 0 0 0];
internalField   uniform 300;

boundaryField
{
    theta0
    {
        type            symmetryPlane;
    }
    theta90
    {
        type            symmetryPlane;
    }
    inner
    {
        type            zeroGradient;
    }
    outer
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""


def equivalent_young_modulus(row: dict) -> float:
    ri = f(row, "inner_radius_mm") * 1e-3
    pressure = f(row, "pressure_mpa") * 1e6
    thickness = f(row, "total_thickness_mm") * 1e-3
    ey = max(abs(f(row, "laminate_ey")), 1e-12)
    nominal_hoop_stress = pressure * ri / thickness
    return nominal_hoop_stress / ey


def lame_displacement(ri: float, ro: float, pressure: float, e_pa: float, nu: float) -> tuple[float, float]:
    a = pressure * ri * ri / (ro * ro - ri * ri)
    b = pressure * ri * ri * ro * ro / (ro * ro - ri * ri)
    u_inner = ((1.0 - nu) * a * ri + (1.0 + nu) * b / ri) / e_pa
    u_outer = ((1.0 - nu) * a * ro + (1.0 + nu) * b / ro) / e_pa
    return u_inner * 1000.0, u_outer * 1000.0


def generate_cases(rows: list[dict]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        case_dir = RUN_ROOT / row["case_id"]
        ri = f(row, "inner_radius_mm") * 1e-3
        ro = f(row, "outer_radius_mm") * 1e-3
        pressure = f(row, "pressure_mpa") * 1e6
        e_eq = equivalent_young_modulus(row)
        nu = 0.30
        write_text(case_dir / "system" / "blockMeshDict", block_mesh_dict(ri, ro))
        write_text(case_dir / "system" / "controlDict", control_dict())
        write_text(case_dir / "system" / "fvSchemes", fv_schemes())
        write_text(case_dir / "system" / "fvSolution", fv_solution())
        write_text(case_dir / "constant" / "mechanicalProperties", mechanical_properties(e_eq, nu))
        write_text(case_dir / "constant" / "thermalProperties", thermal_properties())
        write_text(case_dir / "0" / "D", displacement_field(pressure))
        write_text(case_dir / "0" / "T", temperature_field())
        u_in, u_out = lame_displacement(ri, ro, pressure, e_eq, nu)
        metadata = {
            "case_id": row["case_id"],
            "model": "quarter_annulus_plane_stress_equivalent_isotropic",
            "inner_radius_m": ri,
            "outer_radius_m": ro,
            "pressure_pa": pressure,
            "equivalent_E_pa": e_eq,
            "nu": nu,
            "lame_inner_displacement_mm": u_in,
            "lame_outer_displacement_mm": u_out,
            "target_genetic_max_radial_displacement_mm": f(row, "genetic_max_radial_displacement_mm"),
        }
        write_text(case_dir / "openfoam_case_metadata.json", json.dumps(metadata, indent=2))

    wsl_path = "/mnt/c/Users/PC/OneDrive - ensam.eu/Bureau/simulation/genetic/type4_copv_comparison_output/openfoam_runs"
    runner = f"""#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -eo pipefail

WIN_ROOT="{wsl_path}"
RUN_ROOT="/tmp/openfoam_copv_runs"
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
cp -r "$WIN_ROOT"/case_* "$RUN_ROOT"/

for case_dir in "$RUN_ROOT"/case_*; do
  [ -d "$case_dir" ] || continue
  echo "=== $(basename "$case_dir") ==="
  cd "$case_dir"
  rm -rf constant/polyMesh processor* [1-9]* log.blockMesh log.solidDisplacementFoam
  blockMesh > log.blockMesh 2>&1
  solidDisplacementFoam > log.solidDisplacementFoam 2>&1
  tail -5 log.solidDisplacementFoam
done

for case_dir in "$RUN_ROOT"/case_*; do
  [ -d "$case_dir" ] || continue
  name="$(basename "$case_dir")"
  rm -rf "$WIN_ROOT/$name"/constant/polyMesh "$WIN_ROOT/$name"/[1-9]* "$WIN_ROOT/$name"/log.blockMesh "$WIN_ROOT/$name"/log.solidDisplacementFoam
  cp -r "$case_dir"/constant/polyMesh "$WIN_ROOT/$name"/constant/
  cp -r "$case_dir"/[1-9]* "$WIN_ROOT/$name"/ 2>/dev/null || true
  cp "$case_dir"/log.blockMesh "$WIN_ROOT/$name"/
  cp "$case_dir"/log.solidDisplacementFoam "$WIN_ROOT/$name"/
done
"""
    write_text(WSL_RUNNER, runner)


def latest_time(case_dir: Path) -> Path | None:
    candidates = []
    for child in case_dir.iterdir():
        if child.is_dir():
            try:
                candidates.append((float(child.name), child))
            except ValueError:
                pass
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def parse_patch_vectors(field_text: str, patch: str) -> list[tuple[float, float, float]]:
    start = field_text.find(f"\n    {patch}\n")
    if start < 0:
        start = field_text.find(f"\n{patch}\n")
    if start < 0:
        return []
    value_pos = field_text.find("value", start)
    if value_pos < 0:
        return []
    list_pos = field_text.find("List<vector>", value_pos)
    if list_pos < 0:
        uniform = re.search(r"value\s+uniform\s+\(([^)]+)\)", field_text[value_pos:value_pos + 200])
        if not uniform:
            return []
        vals = tuple(float(x) for x in uniform.group(1).split())
        return [vals]  # type: ignore[list-item]
    open_paren = field_text.find("(", list_pos)
    close_marker = field_text.find("\n)\n;", open_paren)
    if open_paren < 0 or close_marker < 0:
        return []
    body = field_text[open_paren:close_marker]
    vectors = []
    for match in re.finditer(r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)", body):
        vectors.append((float(match.group(1)), float(match.group(2)), float(match.group(3))))
    return vectors


def parse_scalar_internal_max(path: Path) -> float:
    if not path.exists():
        return float("nan")
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "internalField   nonuniform"
    pos = text.find(marker)
    if pos < 0:
        uniform = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)", text)
        return float(uniform.group(1)) if uniform else float("nan")
    open_paren = text.find("(", pos)
    close_marker = text.find("\n)\n;", open_paren)
    if open_paren < 0 or close_marker < 0:
        return float("nan")
    values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text[open_paren:close_marker])]
    return max(values) if values else float("nan")


def summarize_case(case_dir: Path) -> dict:
    metadata = json.loads((case_dir / "openfoam_case_metadata.json").read_text(encoding="utf-8"))
    time_dir = latest_time(case_dir)
    if time_dir is None or not (time_dir / "D").exists():
        return {
            **metadata,
            "openfoam_status": "failed_no_time_directory",
        }
    d_text = (time_dir / "D").read_text(encoding="utf-8", errors="ignore")
    inner = parse_patch_vectors(d_text, "inner")
    outer = parse_patch_vectors(d_text, "outer")

    def stats(vectors: list[tuple[float, float, float]], key: str) -> dict:
        if not vectors:
            return {
                f"openfoam_{key}_mean_u_mm": float("nan"),
                f"openfoam_{key}_max_u_mm": float("nan"),
            }
        magnitudes = [math.sqrt(x * x + y * y + z * z) * 1000.0 for x, y, z in vectors]
        return {
            f"openfoam_{key}_mean_u_mm": float(np.mean(magnitudes)),
            f"openfoam_{key}_max_u_mm": float(np.max(magnitudes)),
        }

    log_text = (case_dir / "log.solidDisplacementFoam").read_text(encoding="utf-8", errors="ignore") if (case_dir / "log.solidDisplacementFoam").exists() else ""
    completed = "End" in log_text
    result = {
        **metadata,
        "openfoam_status": "completed" if completed else "run_incomplete",
        "latest_time": time_dir.name,
        **stats(inner, "inner"),
        **stats(outer, "outer"),
        "openfoam_max_sigmaEq_pa": parse_scalar_internal_max(time_dir / "sigmaEq"),
    }
    result["openfoam_max_u_mm"] = max(result["openfoam_inner_max_u_mm"], result["openfoam_outer_max_u_mm"])
    result["openfoam_vs_lame_inner_error_pct"] = 100.0 * (result["openfoam_inner_mean_u_mm"] - result["lame_inner_displacement_mm"]) / max(abs(result["lame_inner_displacement_mm"]), 1e-12)
    result["openfoam_vs_lame_outer_error_pct"] = 100.0 * (result["openfoam_outer_mean_u_mm"] - result["lame_outer_displacement_mm"]) / max(abs(result["lame_outer_displacement_mm"]), 1e-12)
    result["openfoam_vs_genetic_max_u_delta_pct"] = 100.0 * (result["openfoam_max_u_mm"] - result["target_genetic_max_radial_displacement_mm"]) / max(abs(result["target_genetic_max_radial_displacement_mm"]), 1e-12)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict]) -> None:
    completed = [r for r in rows if r.get("openfoam_status") == "completed"]
    if not completed:
        return
    ids = [r["case_id"] for r in completed]
    x = np.arange(len(completed))
    genetic = np.array([r["target_genetic_max_radial_displacement_mm"] for r in completed], dtype=float)
    foam = np.array([r["openfoam_max_u_mm"] for r in completed], dtype=float)
    lame = np.array([r["lame_inner_displacement_mm"] for r in completed], dtype=float)
    lame_err = np.array([r["openfoam_vs_lame_inner_error_pct"] for r in completed], dtype=float)
    gen_err = np.array([r["openfoam_vs_genetic_max_u_delta_pct"] for r in completed], dtype=float)

    plt.rcParams.update({"figure.figsize": (11, 6), "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

    plt.figure()
    plt.plot(x, genetic, marker="o", label="genetic max Ur")
    plt.plot(x, foam, marker="s", label="OpenFOAM max |D|")
    plt.plot(x, lame, marker="^", label="Lame plane-stress inner u")
    plt.xticks(x, ids, rotation=35)
    plt.ylabel("Displacement [mm]")
    plt.title("Radial displacement comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "openfoam_solid_01_displacement_compare.png", dpi=220)
    plt.close()

    plt.figure()
    plt.bar(x, lame_err, color="#1F8A4C")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, ids, rotation=35)
    plt.ylabel("Error [%]")
    plt.title("OpenFOAM vs Lame analytical benchmark")
    plt.tight_layout()
    plt.savefig(OUT / "openfoam_solid_02_lame_error.png", dpi=220)
    plt.close()

    plt.figure()
    plt.bar(x, gen_err, color="#C44D2D")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, ids, rotation=35)
    plt.ylabel("Delta [%]")
    plt.title("OpenFOAM equivalent isotropic model vs genetic composite model")
    plt.tight_layout()
    plt.savefig(OUT / "openfoam_solid_03_genetic_delta.png", dpi=220)
    plt.close()


def html_table(rows: list[dict], keys: list[str]) -> str:
    head = "".join(f"<th>{key}</th>" for key in keys)
    body = []
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.5g}"
            cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def write_report(rows: list[dict]) -> None:
    completed = [r for r in rows if r.get("openfoam_status") == "completed"]
    n = len(rows)
    n_ok = len(completed)
    mean_lame = float(np.nanmean([abs(r.get("openfoam_vs_lame_inner_error_pct", float("nan"))) for r in completed])) if completed else float("nan")
    mean_genetic = float(np.nanmean([abs(r.get("openfoam_vs_genetic_max_u_delta_pct", float("nan"))) for r in completed])) if completed else float("nan")
    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Runs OpenFOAM COPV</title>
  <style>
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:#17202a; background:#f5f7fb; line-height:1.55; }}
    header {{ padding:42px 6vw; background:linear-gradient(120deg,#10243b,#245b7c,#81402f); color:white; }}
    h1 {{ margin:0; font-size:42px; letter-spacing:0; }}
    main {{ max-width:1400px; margin:0 auto; padding:24px 6vw 70px; }}
    section {{ background:white; border:1px solid #dbe2ea; border-radius:8px; box-shadow:0 12px 30px rgba(20,35,55,.1); padding:24px; margin:22px 0; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .metric {{ border:1px solid #dbe2ea; border-radius:8px; padding:16px; background:#fbfdff; }}
    .metric small {{ display:block; color:#5b6674; }}
    .metric strong {{ font-size:28px; color:#0b6bcb; }}
    .warn {{ border-left:4px solid #c44d2d; background:#fff4ed; padding:14px 16px; border-radius:6px; }}
    .ok {{ border-left:4px solid #1f8a4c; background:#eefbf3; padding:14px 16px; border-radius:6px; }}
    .table-wrap {{ overflow-x:auto; border:1px solid #dbe2ea; border-radius:8px; }}
    table {{ width:100%; min-width:1050px; border-collapse:collapse; }}
    th,td {{ padding:9px 11px; border-bottom:1px solid #dbe2ea; text-align:left; font-size:14px; }}
    th {{ background:#edf3fa; }}
    .figs {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    figure {{ margin:0; border:1px solid #dbe2ea; border-radius:8px; overflow:hidden; }}
    figure img {{ width:100%; display:block; }}
    figcaption {{ padding:10px 12px; color:#5b6674; }}
    code {{ background:#edf3fa; padding:2px 5px; border-radius:4px; }}
  </style>
</head>
<body>
<header>
  <h1>Runs OpenFOAM solidDisplacementFoam</h1>
  <p>Execution des 10 cas COPV avec un modele cylindre epais isotrope equivalent, derive des memes entrees que <code>genetic</code>.</p>
</header>
<main>
  <section>
    <h2>Statut</h2>
    <p class="ok"><strong>OpenFOAM a tourne:</strong> {n_ok}/{n} cas termines avec <code>solidDisplacementFoam</code>.</p>
    <p class="warn"><strong>Limite du modele:</strong> ce run est un benchmark de cylindre epais isotrope equivalent. Il valide l'environnement OpenFOAM et le chargement pression/maillage, mais ne remplace pas encore un vrai modele composite anisotrope couche-par-couche.</p>
    <div class="grid">
      <div class="metric"><small>Erreur moyenne OpenFOAM vs Lame</small><strong>{mean_lame:.2f}%</strong></div>
      <div class="metric"><small>Delta moyen OpenFOAM vs genetic</small><strong>{mean_genetic:.1f}%</strong></div>
      <div class="metric"><small>Cases</small><strong>{n_ok}/{n}</strong></div>
    </div>
  </section>
  <section>
    <h2>Table des resultats</h2>
    {html_table(rows, ['case_id','openfoam_status','latest_time','equivalent_E_pa','openfoam_inner_mean_u_mm','openfoam_outer_mean_u_mm','openfoam_max_u_mm','lame_inner_displacement_mm','target_genetic_max_radial_displacement_mm','openfoam_vs_lame_inner_error_pct','openfoam_vs_genetic_max_u_delta_pct','openfoam_max_sigmaEq_pa'])}
  </section>
  <section>
    <h2>Courbes</h2>
    <div class="figs">
      <figure><img src="openfoam_solid_01_displacement_compare.png"><figcaption>Deplacement radial: genetic, OpenFOAM et Lame.</figcaption></figure>
      <figure><img src="openfoam_solid_02_lame_error.png"><figcaption>Erreur numerique OpenFOAM par rapport a Lame.</figcaption></figure>
      <figure><img src="openfoam_solid_03_genetic_delta.png"><figcaption>Ecart OpenFOAM isotrope equivalent vs genetic composite.</figcaption></figure>
    </div>
  </section>
  <section>
    <h2>Fichiers</h2>
    <ul>
      <li><a href="openfoam_solid_results.csv">openfoam_solid_results.csv</a></li>
      <li><a href="openfoam_runs/">openfoam_runs/</a></li>
      <li><a href="openfoam_coherence_report.html">Retour coherence OpenFOAM/Python</a></li>
      <li><a href="index.html">Retour rapport principal</a></li>
    </ul>
  </section>
</main>
</body>
</html>
"""
    write_text(REPORT, html)


def summarize_results() -> list[dict]:
    rows = []
    for case_dir in sorted(RUN_ROOT.glob("case_*")):
        if case_dir.is_dir() and (case_dir / "openfoam_case_metadata.json").exists():
            rows.append(summarize_case(case_dir))
    write_csv(RESULT_CSV, rows)
    plot_results(rows)
    write_report(rows)
    return rows


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        rows = summarize_results()
        print(f"Summarized {len(rows)} OpenFOAM cases into {RESULT_CSV}")
        print(REPORT)
        return
    rows = read_rows(PANEL_CSV)
    generate_cases(rows)
    print(f"Generated {len(rows)} OpenFOAM cases in {RUN_ROOT}")
    print(f"Runner: {WSL_RUNNER}")


if __name__ == "__main__":
    main()
