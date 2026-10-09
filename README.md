# Correcting a fast pressure-vessel sizing model with 384 CalculiX simulations

A Type IV hydrogen tank optimizer used a fast analytical model that only sees the cylinder. I built a CalculiX composite-shell model of the whole vessel, ran 384 cases, and trained a small MLP that corrects the fast model's fibre failure index (cross-validated R² 0.862 on log C, against 0.796 for Ridge).

![CalculiX fibre failure index on one 11 L vessel, and where the peak falls across the 384 cases](reports/figures/readme_hero.png)
*Left: fibre index |σ11|/X (worst ply per element) for case `11l_0122` at 70 MPa, rendered from the CalculiX result. The peak (0.86) sits in the ring of elements at the polar opening of the dome on the axially fixed boss; the fast model gives 0.42 for this design. Grey: two rings of cylinder elements that no stress sample was mapped to in post-processing (missing data, not zero stress). Right: critical zone over the 384 cases. The fast model always reports the cylinder.*

Student research project, May to June 2026, cleaned up for publication over the summer. The starting point was a genetic-algorithm layup optimizer written by a team in 2025. Its mechanics are a thick multilayer anisotropic cylinder (Lekhnitskii) with Hashin, Tsai-Wu and Puck criteria. It runs in under a millisecond, but it has no dome, no polar boss and no winding-angle variation, and that is where filament-wound vessels usually fail. The question was whether the optimizer could be made aware of the dome without slowing it down or touching its code.

## Results

Correction factor C = FI(CalculiX) / FI(fast model), predicted from the design inputs. Metrics are on log C, from [`data/extended_statistical_metrics.json`](data/extended_statistical_metrics.json).

| | Ridge (α = 10) | MLP 24×12 tanh |
|---|---:|---:|
| R², repeated 8-fold CV on the 288 training cases | 0.796 ± 0.046 | 0.862 ± 0.029 |
| R², 96 held-out cases | 0.727 | 0.778 |
| R², 24 cases at the edge of the geometric domain | 0.672 | 0.800 |
| R², 24 cases at untrained pressures (52.5 and 87.5 MPa) | 0.703 | 0.714 (OLS: 0.726) |

![Predicted vs CalculiX correction factor on the 96 held-out cases](reports/figures/extended_benchmark_nuage_en.png)

I then plugged the corrected model into the optimizer and checked the designs it produced: 10 seeds gave 17 distinct layups, each rebuilt and re-run in CalculiX at the 87 MPa design burst pressure ([`data/ga_statistical_validation.csv`](data/ga_statistical_validation.csv)).

![Fibre index of the 17 GA designs: fast model, corrected model, CalculiX p95 and maximum](reports/figures/readme_ga_check.png)

The raw fast model underestimated the CalculiX fibre index on all 17 designs, even against the 95th percentile. With the correction, the prediction is above the CalculiX p95 value on 17/17 designs, but above the pointwise maximum on only 4/17, and 5 of the 17 designs have a local CalculiX peak above 1. The correction removes the systematic optimism of the fast model; it does not cover the local peaks at the polar opening (see Limitations).

## What I built

The optimizer and the fast model are not mine ([Credits](#credits)). Everything below is.

- **CalculiX vessel model**, [`code/high_fidelity/generate_calculix_composite_case.py`](code/high_fidelity/generate_calculix_composite_case.py): cylinder, dome (hemispherical, isotensoid-like or variable contour) and polar boss meshed as S8R quadratic shells. The winding angle follows Clairaut's geodesic law, sin α(r) = r_b / r, and the ply thickness builds up towards the pole. Each element gets its own `*SHELL SECTION, COMPOSITE` with local angles and thicknesses. One boss is fixed axially and the other slides, so the vessel can elongate under pressure without an artificial axial clamp. Post-processing ([`postprocess_calculix_case.py`](code/high_fidelity/postprocess_calculix_case.py)) reads the `.frd`, assigns each section point to a ply from its radial position, averages the stress per ply and per element, rotates it into the fibre frame and evaluates Hashin, Tsai-Wu, Puck and the fibre index |σ11|/X_T (X_C in compression).
- **Two DOEs.** First an exploratory 128-case campaign over a wide range of vessel sizes. After feedback that the sizes varied too much, a second one at a fixed 11 L volume, [`code/config/generate_doe_11l.py`](code/config/generate_doe_11l.py): 384 cases varying the radius (76 to 96 mm), boss radius, dome shape, helical angle (12 to 26°), number of helical, transition and hoop pairs, and total thickness (10.5 to 26.1 mm). Every case was solved in CalculiX. Case roles (train, holdout, boundary, pressure scale) are fixed in [`data/doe_11l_single_boss_cases.json`](data/doe_11l_single_boss_cases.json).
- **Choice of the target.** I first used the maximum of the combined criteria, as the fast model does. Near the polar opening the matrix and shear indices reach the hundreds at operating pressure, which made the ratio meaningless, so I switched to the fibre index: burst in Type IV vessels is fibre-driven, and C came back to between 1.0 and 4.5.
- **Correctors**, [`code/calibration/extended_statistical_validation.py`](code/calibration/extended_statistical_validation.py): constant, OLS, Ridge and an MLP (24×12, tanh, Adam, L2 4e-4, 6000 epochs), all written in NumPy with manual backpropagation. Inputs: 17 design and fast-model quantities, 6 derived ratios (P·r/t, boss-to-radius ratio and others) and the one-hot dome shape. Target: log C. The dataset is assembled by [`build_fiber_proxy_dataset.py`](code/calibration/build_fiber_proxy_dataset.py).
- **Optimizer coupling**, [`code/optimization/ga_correction.py`](code/optimization/ga_correction.py): the fitness uses FI_fast × Ĉ(x) × exp(m₉₅), where m₉₅ is the 95th percentile of the MLP's cross-validation residuals on log C. I changed three files of the optimizer to call it and left its mechanics untouched.

## How it works

![Workflow: fast model, CalculiX DOE, correction model, GA, FE re-check, literature benchmarks](reports/figures/fig_workflow_en.png)

The fast model stays in the loop because the optimizer calls it tens of thousands of times. The correction only has to learn one scalar per design, C = FI_FE / FI_fast, which is far easier than learning the stress field and keeps the fast model's own trends. Learning log C rather than C stops the large ratios near the polar opening from dominating the loss.

## How I checked it

- Baselines first. The MLP is compared with a constant, OLS and Ridge on the same splits ([table above](#results)). Its gain over Ridge is 0.066 in CV R², which is modest. It is clearly ahead only at the edge of the geometric domain, and on the pressure-extrapolation split OLS does slightly better.
- Splits and leakage. 288 training cases for 4 repeats of 8-fold CV, and 96 cases never used for training: 48 in-distribution holdouts, 24 geometric-boundary cases and 24 at untrained pressures. No duplicate designs and no overlap between training and evaluation (checked in the script).
- Reproducible numbers. Re-running `extended_statistical_validation.py` on the committed dataset reproduces every value in `extended_statistical_metrics.json` exactly (fixed seeds).
- End-to-end check. The 17 GA designs above, re-meshed and re-solved in CalculiX.
- Published burst tests, with the same shell model and fibre index:
  - Agne et al. 2025, GFRP and CFRP vessels with the layup table reconstructed from the paper: predicted burst 4 to 14 % below the measured 29.4 and 27.0 MPa.
  - Jin et al. 2022, vessels EX-A and EX-B with the thickness read from a figure of the paper: 10 % and 22 % below the measured 65.2 MPa at the finest mesh, using the p99 value.
  - Hu et al. 2021, 70 MPa vessel: the critical zone is in the dome, as in the paper, but the absolute index is not calibrated.
- Mesh study on EX-A and EX-B with 16, 24 and 32 elements per direction ([`data/jin2022_mesh_convergence.csv`](data/jin2022_mesh_convergence.csv)). The pointwise maximum does not converge; it goes up, then down. The p99 value increases steadily but still moves by 6 % (EX-A) and 14 % (EX-B) between the last two meshes.

![Mesh study on the Jin et al. 2022 vessels](reports/figures/paik2023_mesh_convergence_en.png)

## Limitations

- Linear elastic shells: no progressive damage, no liner plasticity, no liner/boss contact in the DOE. I tested contact separately on two cases of the first DOE. Changing the penalty stiffness moved the stresses by about 0.3 %, but going from tied to penalty contact moved them by 7.5 to 8 %, and frictional contact never ran reliably, so contact stayed out of the DOE.
- The local peak at the polar opening is mesh-dependent, which is why the GA check passes against p95 but not against the maximum. A finer local mesh or a solid model of the boss region would be the next step.
- The peak lands in the dome on the fixed-boss side in 198 of 384 cases and never on the sliding side, so part of the dome effect comes from the boss boundary condition.
- One vessel family: 11 L, single boss, geodesic winding, carbon/epoxy. The correction should not be used outside the sampled ranges.
- No physical test of my own. The burst comparisons rely on published data, with some geometry read from figures.
- The optimizer code is not in this repository, so the GA check cannot be re-run from here.

## Reproduce

```bash
pip install numpy pandas matplotlib
python code/calibration/extended_statistical_validation.py   # about 2 min; rewrites data/extended_statistical_metrics.json and the two benchmark figures
python code/plotting/make_readme_figures.py                  # top figure and GA check figure
```

Re-running the CalculiX cases needs CalculiX 2.23 (`ccx`) and produces about 20 MB per case; those raw `.frd` files are not in the repository. [`code/plotting/render_fiber_field.py`](code/plotting/render_fiber_field.py) renders the field on the left of the top figure from such a file with ParaView 6.1 (`pvpython`). The paths to `ccx` and to FreeCAD's Python are hard-coded for Windows in `code/high_fidelity/run_calculix_*.py`.

The full write-up, with equations and benchmark details, is in [`reports/english/rapport_public_en.pdf`](reports/english/rapport_public_en.pdf) (French version in `reports/french/`).

## Credits

- Genetic-algorithm optimizer and fast analytical model (`computation.py`, `tank.py`, `individual.py`, `population.py`, `crossover.py`, `mutation.py`): collaborative student codebase from 2025, not included here.
- Solver: [CalculiX](http://www.calculix.de/) 2.23. Rendering: ParaView.
- Published tests used for comparison: Hu, Chen & Pan, *Int. J. Hydrogen Energy* (2021); Agne et al., *Composite Structures* (2025); Jin, Cheng, Bai, Paik & Li, *Ships and Offshore Structures* (2022). Full references in [`reports/references_public.bib`](reports/references_public.bib).

Clément Dumeril · [clement.dumeril.net](https://clement.dumeril.net) · [github.com/clementdumeril](https://github.com/clementdumeril)
