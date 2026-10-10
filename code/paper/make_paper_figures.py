#!/usr/bin/env python3
"""Figures and numbers for paper/main.tex, built from files in data/.

    python code/paper/make_paper_figures.py [--runs D:/.../paper_runs]

With --runs, the per-design summaries of the CalculiX campaigns made for the paper are first
collected into data/paper_mesh_study.csv and data/fiber_proxy_dataset_strict.csv (the raw
result files are too large to version). Without it, the committed CSVs are used.
Writes paper/figures/*.pdf and paper/generated/numbers.tex.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                            "pdf.fonttype": 42, "savefig.bbox": "tight"})

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
FIG = REPO / "paper" / "figures"
GEN = REPO / "paper" / "generated"
MESHES = (16, 24, 32, 48, 64)
MESH_CASES = ("11l_0122", "11l_0318", "11l_0136", "11l_0007")
SUMMARY_KEYS = ("solver_status", "max_fiber_stress_ratio_abs", "fiber_stress_ratio_p95_abs",
                "fiber_stress_ratio_p99_abs", "fiber_proxy_zone", "fiber_proxy_ply_name",
                "fiber_proxy_local_angle_deg", "failure_stress_samples")


# ----------------------------------------------------------------------------- collection
def collect(runs: Path) -> None:
    rows = []
    for variant in ("baseline", "strict"):
        for mesh in MESHES:
            calc = runs / f"mesh_{variant}_m{mesh}" / "simulation_results" / "calculix"
            log = runs / f"mesh_{variant}_m{mesh}" / f"campaign_log_mesh{mesh}.csv"
            times = pd.read_csv(log).set_index("case_id") if log.exists() else None
            for case in MESH_CASES:
                f = calc / case / "calculix_summary.json"
                if not f.exists():
                    continue
                s = json.loads(f.read_text(encoding="utf-8"))
                row = {"variant": variant, "mesh": mesh, "case_id": case, **{k: s.get(k, "") for k in SUMMARY_KEYS}}
                if times is not None and case in times.index:
                    t = times.loc[case]
                    row["wall_time_s"] = float(sum(t[c] for c in times.columns if c.endswith("_s")))
                rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(DATA / "paper_mesh_study.csv", index=False)
        print("wrote data/paper_mesh_study.csv", len(rows), "rows")
    hoop0 = runs / "hoop0_m24" / "simulation_results" / "calculix" / "11l_0122" / "calculix_summary.json"
    if hoop0.exists():
        s = json.loads(hoop0.read_text(encoding="utf-8"))
        pd.DataFrame([{"variant": "hoop_coverage_epsilon=0", "mesh": 24, "case_id": "11l_0122",
                       **{k: s.get(k, "") for k in SUMMARY_KEYS}}]).to_csv(DATA / "paper_hoop0_11l_0122.csv", index=False)
    strict = runs / "strict_m24"
    if (strict / "simulation_results").exists():
        subprocess.run([sys.executable, str(REPO / "code" / "paper" / "build_variant_dataset.py"),
                        "--runs", str(strict), "--out", str(DATA / "fiber_proxy_dataset_strict.csv")], check=True)


# ----------------------------------------------------------------------------- figures
def fig_scatter() -> None:
    p = pd.read_csv(DATA / "extended_statistical_eval_predictions.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    styles = {"holdout": ("o", "in-distribution"), "boundary": ("s", "geometric boundary"), "pressure_scale": ("^", "untrained pressure")}
    for role, (marker, label) in styles.items():
        q = p[p.doe_role == role]
        ax.scatter(q.C_calculix, q.C_pred_MLP, marker=marker, s=16, alpha=0.8, label=f"MLP, {label}")
    ax.scatter(p.C_calculix, p.C_pred_Ridge, marker=".", s=10, color="0.6", label="Ridge, all", zorder=0)
    lim = [0.9, max(p.C_calculix.max(), p.C_pred_MLP.max()) * 1.03]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"$C$ from CalculiX")
    ax.set_ylabel(r"predicted $C$")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig_scatter.pdf")
    plt.close(fig)


def fig_field_zones() -> None:
    df = pd.read_csv(DATA / "fiber_proxy_dataset.csv")
    zones = ["cylinder", "junction", "left_dome", "right_dome", "boss"]
    labels = ["cylinder", "junction", "dome\n(fixed boss)", "dome\n(sliding boss)", "boss\nedge"]
    fast = df.python_raw_critical_zone.value_counts().reindex(zones, fill_value=0)
    fe = df.calculix_fiber_proxy_zone.value_counts().reindex(zones, fill_value=0)
    img = mpimg.imread(REPO / "figures" / "fiber_index_field_11l_0122.png")
    h, w = img.shape[:2]
    img = img[int(0.14 * h): int(0.82 * h), int(0.2 * w): int(0.85 * w)]
    vmax = float(df.set_index("case_id").loc["11l_0122", "calculix_fiber_stress_ratio_max"])
    fig = plt.figure(figsize=(7.2, 2.6))
    ax0 = fig.add_axes([0.0, 0.05, 0.48, 0.9])
    ax0.imshow(img)
    ax0.axis("off")
    cax = fig.add_axes([0.49, 0.2, 0.012, 0.6])
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(0, vmax), "inferno"), cax=cax)
    bar.set_label(r"$|\sigma_{11}|/X$")
    ax1 = fig.add_axes([0.62, 0.22, 0.37, 0.7])
    x = np.arange(len(zones))
    ax1.bar(x - 0.2, fast.values, 0.4, color="#9aa5b1", label="fast model")
    ax1.bar(x + 0.2, fe.values, 0.4, color="#c2410c", label="CalculiX")
    ax1.set_xticks(x, labels, fontsize=7)
    ax1.set_ylabel("designs")
    ax1.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig_field_zones.pdf")
    plt.close(fig)


def fig_ga() -> None:
    d = pd.read_csv(DATA / "ga_statistical_validation.csv").sort_values("calculix_fibre_fi_max").reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.scatter(x, d.python_raw_fibre_fi, marker="v", color="#9aa5b1", label="fast model")
    ax.scatter(x, d.python_corrected_fibre_fi_p95, marker="o", color="#2563eb", label="fast model, corrected")
    ax.scatter(x, d.calculix_fibre_fi_p95, marker="s", facecolor="none", edgecolor="#c2410c", label="CalculiX, 95th percentile")
    ax.scatter(x, d.calculix_fibre_fi_max, marker="x", color="#7f1d1d", label="CalculiX, maximum")
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xticks(x, [f"{s}-{r}" for s, r in zip(d.seed, d["rank"])], rotation=60, fontsize=6)
    ax.set_xlabel("GA design (seed-rank)")
    ax.set_ylabel(r"$|\sigma_{11}|/X$ at 87 MPa")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.savefig(FIG / "fig_ga.pdf")
    plt.close(fig)


def fig_mesh(m: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(MESH_CASES), figsize=(7.2, 4.0), sharex=True)
    for row, variant in enumerate(("baseline", "strict")):
        for ax, case in zip(axes[row], MESH_CASES):
            q = m[(m.case_id == case) & (m.variant == variant)].sort_values("mesh")
            ax.plot(q.mesh, q.max_fiber_stress_ratio_abs, "-o", ms=3, color="#7f1d1d", label="maximum")
            ax.plot(q.mesh, q.fiber_stress_ratio_p99_abs, "--s", ms=3, color="#c2410c", mfc="none", label="99th percentile")
            ax.plot(q.mesh, q.fiber_stress_ratio_p95_abs, "-^", ms=3, color="#2563eb", label="95th percentile")
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(case, fontsize=8)
            else:
                ax.set_xlabel("elements per direction")
            ax.set_xticks(MESHES)
        axes[row][0].set_ylabel(f"{variant}\n" + r"$|\sigma_{11}|/X$")
    axes[0][0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_mesh.pdf")
    fig.savefig(REPO / "figures" / "mesh_study.png", dpi=150, facecolor="white")  # README copy
    plt.close(fig)


def fig_targets(base: pd.DataFrame, strict: pd.DataFrame) -> None:
    sets = [
        ("baseline\nmaximum", base.calculix_fiber_stress_ratio_max / base.python_fiber_stress_ratio_max),
        ("baseline\n95th pct.", base.calculix_fiber_stress_ratio_p95 / base.python_fiber_stress_ratio_max),
        ("strict\nmaximum", strict.calculix_fiber_stress_ratio_max / strict.python_fiber_stress_ratio_max),
        ("strict\n95th pct.", strict.calculix_fiber_stress_ratio_p95 / strict.python_fiber_stress_ratio_max),
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.boxplot([v.values for _, v in sets], showfliers=True, flierprops=dict(markersize=2))
    ax.set_xticks(range(1, len(sets) + 1), [k for k, _ in sets], fontsize=7)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 7])
    ax.set_yticklabels(["1", "2", "4", "7"])
    ax.set_ylabel(r"$C = \mathrm{FI}^{\mathrm{FE}} / \mathrm{FI}^{\mathrm{fast}}$")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.savefig(FIG / "fig_targets.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- numbers
def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def numbers(m: pd.DataFrame | None, strict: pd.DataFrame | None) -> None:
    met = json.loads((DATA / "extended_statistical_metrics.json").read_text(encoding="utf-8"))
    meta, cv, ev = met["dataset_metadata"], met["cross_validation_metrics"], met["evaluation_metrics"]
    df = pd.read_csv(DATA / "fiber_proxy_dataset.csv")
    ga = pd.read_csv(DATA / "ga_statistical_validation.csv")
    cases = json.loads((DATA / "doe_11l_single_boss_cases.json").read_text(encoding="utf-8"))["cases"]
    inconsistent = sum(c["geometry"]["inner_radius_mm"] * math.sin(math.radians(min(c["winding"]["helical_angles_deg"])))
                       > c["geometry"]["boss_radius_mm"] for c in cases)
    dome = df[df.calculix_fiber_proxy_zone.isin(["left_dome", "right_dome"])]
    z = df.calculix_fiber_proxy_zone.value_counts()
    C = df.fiber_correction_ratio_calculix_over_python
    models = ["Constant", "OLS", "Ridge", "MLP"]
    n = {
        "NdoeCases": len(df), "NtrainCases": meta["train_count"], "NevalCases": meta["eval_count"],
        "NholdoutCases": meta["holdout_count"], "NboundaryCases": meta["boundary_count"], "NpressureCases": meta["pressure_scale_count"],
        "RtwoCvMlp": fmt(cv["MLP"]["r2_mean"]), "RtwoCvRidge": fmt(cv["Ridge"]["r2_mean"]),
        "RtwoEvalMlp": fmt(ev["MLP"]["eval_r2"]), "RtwoEvalRidge": fmt(ev["Ridge"]["eval_r2"]),
        "DeltaRtwoCv": fmt(cv["MLP"]["r2_mean"] - cv["Ridge"]["r2_mean"]),
        "Cmin": fmt(C.min(), 2), "Cmax": fmt(C.max(), 2), "Cmedian": fmt(C.median(), 2),
        "NgaDesigns": len(ga), "NgaConsMax": int(ga.is_conservative_vs_max.sum()),
        "NgaConsPninetyfive": int(ga.is_conservative_vs_p95.sum()), "NgaMaxAboveOne": int((ga.calculix_fibre_fi_max > 1).sum()),
        "NdomeCritical": len(dome), "NdomeCriticalHoop": int(dome.calculix_fiber_proxy_ply_name.str.startswith("hoop").sum()),
        "NcylCritical": int(z.get("cylinder", 0)), "NjunctionCritical": int(z.get("junction", 0)), "NbossCritical": int(z.get("boss", 0)),
        "NdoeInconsistent": inconsistent, "NdoeConsistent": len(cases) - inconsistent,
        "MetricsTableCv": " & ".join(f"${cv[k]['r2_mean']:.3f} \\pm {cv[k]['r2_std']:.3f}$" for k in models),
    }
    for key, tag in (("MetricsTableEval", "eval"), ("MetricsTableHoldout", "holdout"), ("MetricsTableBoundary", "boundary"), ("MetricsTablePressure", "pressure")):
        n[key] = " & ".join(f"${ev[k][f'{tag}_r2']:.3f}$" for k in models)
    hz = DATA / "paper_hoop0_11l_0122.csv"
    n["HoopZeroBefore"] = fmt(float(df.set_index("case_id").loc["11l_0122", "calculix_fiber_stress_ratio_max"]), 2)
    n["HoopZeroAfter"] = fmt(float(pd.read_csv(hz).max_fiber_stress_ratio_abs.iloc[0]), 2) if hz.exists() else "?"
    if m is not None and not m.empty:
        q = m[(m.case_id == "11l_0122") & (m.variant == "baseline")].set_index("mesh")
        n["MeshPeakLow"] = fmt(q.loc[16, "max_fiber_stress_ratio_abs"], 2) if 16 in q.index else "?"
        n["MeshPeakHigh"] = fmt(q.loc[64, "max_fiber_stress_ratio_abs"], 2) if 64 in q.index else "?"
    # Peak location and the p95-based ratio (baseline campaign, all 384 designs)
    loc = pd.read_csv(DATA / "baseline_fiber_peak_location.csv")
    n["NpeakAtPole"] = int((loc.peak_radius_mm < 1.25 * loc.boss_radius_mm).sum())
    bq = pd.read_csv(DATA / "fiber_proxy_dataset_baseline_quantiles.csv")
    c95 = bq.calculix_fiber_stress_ratio_p95 / bq.python_fiber_stress_ratio_max
    n["CninetyfiveMedian"], n["CninetyfiveMin"], n["CninetyfiveMax"] = fmt(c95.median(), 2), fmt(c95.min(), 2), fmt(c95.max(), 2)
    n["NCninetyfiveWithinTen"] = int(((c95 - 1).abs() <= 0.10).sum())

    def row(fname: str) -> str:
        mm = json.loads((DATA / fname).read_text(encoding="utf-8"))
        c, e = mm["cross_validation_metrics"], mm["evaluation_metrics"]
        return " & ".join(f"${c[k]['r2_mean']:.2f}$ & ${e[k]['eval_r2']:.2f}$" for k in ("OLS", "Ridge", "MLP"))

    n["TargetRowBaselineMax"] = row("extended_statistical_metrics.json")
    n["TargetRowBaselinePninetyfive"] = row("extended_statistical_metrics_p95_target.json")
    pt = json.loads((DATA / "extended_statistical_metrics_p95_target.json").read_text(encoding="utf-8"))
    n["RtwoCvMlpPninetyfive"] = fmt(pt["cross_validation_metrics"]["MLP"]["r2_mean"], 2)
    n["RtwoEvalMlpPninetyfive"] = fmt(pt["evaluation_metrics"]["MLP"]["eval_r2"], 2)
    if strict is not None:
        n["TargetRowStrictMax"] = row("extended_statistical_metrics_strict_max.json")
        n["TargetRowStrictPninetyfive"] = row("extended_statistical_metrics_strict_p95.json")
        sm = json.loads((DATA / "extended_statistical_metrics_strict_p95.json").read_text(encoding="utf-8"))
        n["NstrictTrain"], n["NstrictEval"] = sm["dataset_metadata"]["train_count"], sm["dataset_metadata"]["eval_count"]
        n["RtwoEvalRidgeStrictPninetyfive"] = fmt(sm["evaluation_metrics"]["Ridge"]["eval_r2"], 2)
        cs = strict.calculix_fiber_stress_ratio_max / strict.python_fiber_stress_ratio_max
        n["CstrictMedian"], n["CstrictMax"] = fmt(cs.median(), 2), fmt(cs.max(), 2)
        n["CstrictNinetyfiveMedian"] = fmt((strict.calculix_fiber_stress_ratio_p95 / strict.python_fiber_stress_ratio_max).median(), 2)
        n["NstrictDome"] = int(strict.calculix_fiber_proxy_zone.isin(["left_dome", "right_dome"]).sum())
    if m is not None and not m.empty:
        def change(variant: str, a: int, b: int, col: str) -> float:
            q = m[m.variant == variant].pivot_table(index="case_id", columns="mesh", values=col)
            return float(((q[b] / q[a] - 1).abs() * 100).max())
        n["MeshBasePninetyfiveFortyEight"] = fmt(change("baseline", 48, 64, "fiber_stress_ratio_p95_abs"), 1)
        n["MeshBasePninetyfiveTwentyFour"] = fmt(change("baseline", 24, 64, "fiber_stress_ratio_p95_abs"), 1)
        n["MeshStrictPninetyfiveFortyEight"] = fmt(change("strict", 48, 64, "fiber_stress_ratio_p95_abs"), 0)
        st = m[m.variant == "strict"].max_fiber_stress_ratio_abs
        n["MeshStrictMaxLow"], n["MeshStrictMaxHigh"] = fmt(st.min(), 2), fmt(st.max(), 2)
        n["MeshTimeSixtyFour"] = int(round(m[m.mesh == 64].wall_time_s.median() / 60))
    lines = ["% Generated by code/paper/make_paper_figures.py; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items()]
    lines += ["\\providecommand{\\MeshPeakLow}{?}", "\\providecommand{\\MeshPeakHigh}{?}"]
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/generated/numbers.tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="")
    args = ap.parse_args()
    if args.runs:
        collect(Path(args.runs))
    FIG.mkdir(parents=True, exist_ok=True)
    fig_scatter()
    fig_field_zones()
    fig_ga()
    m = pd.read_csv(DATA / "paper_mesh_study.csv") if (DATA / "paper_mesh_study.csv").exists() else None
    strict = pd.read_csv(DATA / "fiber_proxy_dataset_strict.csv") if (DATA / "fiber_proxy_dataset_strict.csv").exists() else None
    if m is not None:
        fig_mesh(m)
    if strict is not None:
        fig_targets(pd.read_csv(DATA / "fiber_proxy_dataset_baseline_quantiles.csv"), strict)
    numbers(m, strict)


if __name__ == "__main__":
    main()
