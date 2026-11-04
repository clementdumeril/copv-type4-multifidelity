# Type IV dome correction: a learned multi-fidelity correction for hydrogen-vessel sizing, and an audit of its target

An MLP trained on 384 CalculiX composite-shell simulations corrects the fiber failure index of a fast, cylinder-only sizing model for 11 L Type IV vessels (cross-validated R² 0.862 on log C, Ridge 0.796); a mesh and modelling audit then shows that the peak it learns sits at the polar opening and is not mesh-converged.

[Preprint (PDF)](paper/main.pdf) · [Validation details](docs/VALIDATION.md) · [Data](data/) · [Cite](CITATION.cff)

![CalculiX fiber failure index on one 11 L vessel, and where the peak falls across the 384 cases](figures/readme_hero.png)
*Left: fiber index |σ11|/X, worst ply per element, design `11l_0122` at 70 MPa. Right: critical zone over the 384 designs.*

## Overview

The genetic-algorithm (GA) layup optimizer I started from sizes the vessel with an analytical thick-cylinder model that runs in under a millisecond and has no dome, boss or winding-angle change. I wanted to make it dome-aware without touching its code, so I built a dome-resolved shell model, solved a design of experiments (DOE) with it, and learned a multiplicative correction C that the optimizer applies to its own index.

What is mine and what is reused:

- Mine (Arts et Métiers, May to June 2026; audit in October 2026): the CalculiX model generator and post-processing, the 384-case DOE, the dataset, the correctors and their validation, the GA coupling layer ([`ga_correction.py`](code/optimization/ga_correction.py)), the mesh and boundary-condition audit, the comparison with published burst tests, and the preprint. The work started as a group assignment; the modelling, code and analysis are mine.
- Reused: the GA optimizer and its fast model come from an existing 2025 codebase that I did not write. It is not included here.

```mermaid
flowchart LR
    doe["DOE generator<br/>generate_doe_11l.py<br/>384 designs, 11 L"] -->|"geometry, layup, p (MPa)"| ccx["CalculiX 2.23 shell model<br/>S8R, 24 x 24, linear static"]
    ccx -->|".frd ply stresses"| post["Post-processing<br/>sigma_11 in ply frame<br/>FI max and p95"]
    fast["Fast cylinder model<br/>(existing GA code)"] -->|"FI_fast"| ds[("data/fiber_proxy_dataset.csv<br/>target log C")]
    post -->|"FI_FE"| ds
    ds -->|"288 train / 96 held-out"| fit["Correctors<br/>OLS, Ridge, MLP 24-12<br/>8-fold x 4 CV"]
    fit -->|"C_hat(x) exp(m95)"| ga["Existing GA optimizer<br/>17 designs re-solved<br/>in CalculiX"]
```

*Pipeline and the files behind each step. The fast model and the GA are the reused parts.*

## Formulation

**Shell model.** Eight-node reduced-integration shells (S8R) in CalculiX 2.23 cover the cylinder, both domes and the polar openings, 24 x 24 elements (meridional x circumferential, six along each dome meridian). Each element has its own composite section. Helical and transition plies follow the geodesic law sin α(r) = r_t / r with r_t = R sin α0; hoop plies keep 86.5 to 89.5°. Below its turnaround radius a ply keeps a thickness floor ε (0.20 to 0.35) at about 87°, a numerical convention rather than a winding law.

**Loads and boundary conditions.** Uniform internal pressure (`*DLOAD`), linear static. One boss is fixed axially and the other slides, so the vessel can elongate. No liner, no damage, no contact. The end-cap force p π r_boss² is off in every DOE design; its effect is measured in the audit.

**Material.** Carbon/epoxy lamina, E1 = 151 GPa, E2 = E3 = 9 GPa, G12 = 5 GPa, ν12 = 0.30, ν23 = 0.42, X_T = 2000 MPa, X_C = 1500 MPa ([`code/config/material_database.json`](code/config/material_database.json)).

**Target.** FI = max over (element, ply) of |σ11| / X_T in tension or |σ11| / X_C in compression; the 95th and 99th percentiles are also stored. The regression target is log C, with C = FI(CalculiX) / FI(fast model).

**DOE and correctors.** 384 designs at 11 L internal volume: inner radius 76 to 96 mm, three dome shapes, helical angle 12 to 26°, thickness 10.5 to 26.1 mm, 70 MPa (52.5 and 87.5 MPa for the pressure split). Roles are fixed before training: 288 train, 48 in-distribution holdout, 24 geometric boundary, 24 untrained pressure. Models: constant, OLS, Ridge (α = 10), and an MLP with 24 and 12 tanh units, written in NumPy with hand-coded backpropagation and Adam. Cross-validation uses 8 folds repeated 4 times on the training designs. The full parameter table is in [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Results

| Quantity | Value | Source and check |
|---|---|---|
| R² on log C, repeated 8-fold CV (288 designs) | MLP 0.862 ± 0.029, Ridge 0.796 ± 0.046, OLS 0.791 ± 0.047 | `data/extended_statistical_metrics.json`; re-running the script reproduces the file exactly |
| R² on log C, 96 held-out designs | MLP 0.778, Ridge 0.727 | same file |
| R² at untrained pressures (24 designs) | OLS 0.726, MLP 0.714 | same file; the MLP is not ahead here |
| Correction factor C over the DOE | 1.00 to 4.51, median 1.42 | `data/fiber_proxy_dataset.csv` |
| GA check, 10 seeds, 17 designs re-solved at 87 MPa | corrected index ≥ CalculiX p95 for 17/17, ≥ CalculiX max for 4/17 (raw fast model: 0/17) | `data/ga_statistical_validation.csv` |

![Predicted vs CalculiX correction factor on the 96 held-out designs](figures/extended_benchmark_nuage_en.png)

## Verification and audit

| Check | Finding | Source |
|---|---|---|
| Mesh study, 4 designs, 16 to 64 elements per direction | max index of design 0122 goes 0.85 → 1.11; the p95 moves at most 0.42 % between 48 and 64 | `data/paper_mesh_study.csv` |
| Peak location | 205 of 384 maxima at a radius below 1.25 boss radii, 190 of them on the axially fixed boss | `data/baseline_fiber_peak_location.csv` |
| Geometric consistency of the DOE | in 202 of 384 designs no helical ply can reach the polar opening (R sin α > r_boss) | `data/doe_11l_single_boss_cases.json` |
| End-cap force added, 4 designs, 64 x 64 | max index +27 to +54 %, p95 within 0.8 % | `data/paper_endcap.csv` |
| Published burst tests | 4 to 22 % low on five tests; the DLR 2025 test is missed by 74 % | [`docs/VALIDATION.md`](docs/VALIDATION.md); benchmark runs not versioned |

![Fiber index against mesh density for four designs](figures/mesh_study.png)
*Maximum, 99th and 95th percentiles of the fiber index against mesh density. Top: the DOE model, where only the 95th percentile settles. Bottom: plies stopped at their turnaround radius (preprint, section 6).*

The MLP fits its target well, but that target is mostly a numerical peak at the pole. The correction is usable as a conservative margin against the converged p95, not as evidence of a physical dome effect. Section 6 of the preprint has the full audit, including the coverage-floor ablation and the strict-turnaround re-analysis.

`python scripts/readme_numbers.py` recomputes every number in the two tables above from `data/` (except the burst-test row).

## How to run

Tested on Windows 10 with Python 3.11.9, NumPy 2.2.6, pandas 2.3.0 and Matplotlib 3.10.6 (November 2026: the validation script reproduced the committed metrics JSON exactly in 4.5 min).

```bash
pip install -r requirements.txt
python code/calibration/extended_statistical_validation.py   # correctors and CV, ~4.5 min
python scripts/readme_numbers.py                             # README numbers, ~15 s
bash paper/build_paper.sh                                    # paper figures, numbers.tex and PDF (needs latexmk)
```

These scripts overwrite the committed files in `figures/`, `data/` and `paper/`. To write the metrics elsewhere, set `TYPE4_METRICS_OUT=<path>` and `TYPE4_NO_FIGURES=1`. The CalculiX runs use [`code/paper/run_campaign.py`](code/paper/run_campaign.py) and [`code/high_fidelity/run_calculix_doe.py`](code/high_fidelity/run_calculix_doe.py) with CalculiX 2.23 (set `CCX`). Raw `.frd` files are not versioned.

## Reproducing figures

| Figure | Script |
|---|---|
| `figures/readme_hero.png`, `figures/readme_ga_check.png` | `code/plotting/make_readme_figures.py` |
| `figures/fiber_index_field_11l_0122.png` | `code/plotting/render_fiber_field.py` (ParaView pvpython) |
| `figures/extended_benchmark_nuage_en.png`, `figures/extended_benchmark_residus_en.png` | `code/calibration/extended_statistical_validation.py` |
| `figures/jin2022_mesh_convergence.png` | `code/plotting/plot_mesh_convergence_en.py` |
| `paper/figures/*.pdf`, `paper/generated/numbers.tex` | `code/paper/make_paper_figures.py` (called by `paper/build_paper.sh`) |

## Code structure

| Path | Content |
|---|---|
| `code/config/` | DOE generator for the 11 L family, material database |
| `code/high_fidelity/` | CalculiX input generator (geometry, winding, S8R sections, boundary conditions), runner, post-processing |
| `code/calibration/` | dataset build, correctors and validation (NumPy MLP) |
| `code/optimization/ga_correction.py` | correction layer called by the external GA |
| `code/paper/` | audit campaigns (mesh, end cap, hoop removal, strict turnaround, BC variants) and paper figures |
| `code/archive/` | earlier scripts kept for the record, not used for the results |
| `data/` | DOE case files, datasets, metrics, GA check, audit results |
| `scripts/readme_numbers.py` | recomputes the README numbers from `data/` |
| `paper/` | preprint source, generated numbers, PDF |

## Limitations

- Linear elastic shells, with no liner, progressive damage or qualified liner/boss contact. One vessel family: 11 L, single boss, geodesic winding.
- The learned maximum is not mesh-converged and depends on the coverage floor and on the missing end-cap force. Only the p95 is converged.
- No physical test of my own. The comparison with published burst tests is an order-of-magnitude check, and one test (DLR 2025) is not reproduced.
- The GA is not public, so the 17-design check cannot be re-run from this repo.

## Corrections

October 2026: the metrics are unchanged, but section 6 of the preprint replaces my earlier reading of the dome peak as a physical, hoop-driven effect.

## Citation, license, credits

Cite with [`CITATION.cff`](CITATION.cff) or the preprint. MIT license, see [LICENSE](LICENSE).

- GA optimizer and fast analytical model: an existing 2025 codebase, not written by me and not included here.
- [CalculiX](http://www.calculix.de/) 2.23 and ParaView. Published burst tests and methods are cited in [`paper/references.bib`](paper/references.bib).

Clément Dumeril · [clement.dumeril.net](https://clement.dumeril.net) · [github.com/clementdumeril](https://github.com/clementdumeril)
