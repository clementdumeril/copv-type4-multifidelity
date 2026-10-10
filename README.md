# A learned dome correction for Type IV vessel sizing, and an audit of its high-fidelity target

A Type IV hydrogen tank optimizer used a fast analytical model that only sees the cylinder. I built a CalculiX composite-shell model of the whole vessel, ran it on 384 designs, and trained a small MLP that corrects the fast model's fiber failure index (cross-validated R² 0.862 on log C, against 0.796 for Ridge). Auditing that target with over 220 further CalculiX runs showed that what the MLP learns is mostly a mesh-dependent peak at the polar opening, not a physical dome effect.

![CalculiX fiber failure index on one 11 L vessel, and where the peak falls across the 384 cases](figures/readme_hero.png)
*Left: fiber index |σ11|/X (worst ply per element) for case `11l_0122` at 70 MPa, rendered from the CalculiX result. The peak sits in the ring of elements at the polar opening of the dome on the axially fixed boss. Right: critical zone over the 384 cases. The fast model always reports the cylinder.*

Started as an Arts et Métiers group assignment (May to June 2026); I did all of the modelling, code and analysis myself, and ran the audit in October 2026. The starting point was a 2025 genetic-algorithm layup optimizer whose mechanics are a thick multilayer cylinder (Lekhnitskii) with Hashin, Tsai-Wu and Puck criteria, evaluated in under a millisecond. It models neither the dome nor the boss, and burst often starts in the dome. The question was how to make the optimizer account for the dome without slowing it down or changing its code.

Full report: [`paper/main.pdf`](paper/main.pdf) (9 pages).

Skills: CalculiX (S8R composite shells), NumPy MLP with hand-written backprop/Adam, DOE, repeated k-fold CV, mesh convergence studies, ParaView.

## Results

The MLP predicts the correction factor C = FI(CalculiX) / FI(fast model) from the design inputs. R² is computed on log C ([`data/extended_statistical_metrics.json`](data/extended_statistical_metrics.json)).

| R² on log C | Ridge | MLP |
|---|---:|---:|
| Repeated 8-fold CV, 288 training cases | 0.796 | 0.862 |
| 96 held-out cases | 0.727 | 0.778 |
| 24 cases at the edge of the geometric domain | 0.672 | 0.800 |
| 24 cases at untrained pressures | 0.703 | 0.714 |

![Predicted vs CalculiX correction factor on the 96 held-out cases](figures/extended_benchmark_nuage_en.png)

I then ran the optimizer with the corrected model. Ten seeds gave 17 distinct designs, and I re-solved each one in CalculiX at the 87 MPa design burst pressure.

![Fiber index of the 17 GA designs: fast model, corrected model, CalculiX p95 and maximum](figures/readme_ga_check.png)

Without correction, the fast model underestimates the CalculiX fiber index on all 17 designs. With the correction, it is above the CalculiX 95th percentile on all 17, but above the maximum on only 4. The audit below explains the gap.

## What the audit found

The target C uses the maximum of the shell-model fiber index. In 205 of the 384 designs that maximum sits in the ring of elements at the polar opening.

- **The maximum is not mesh-converged.** Four designs were re-solved at 16 to 64 elements per direction. The 95th percentile changes by less than 0.4 % between 48 and 64; the maximum and the 99th percentile jump around, the critical ply changes, and in one design the critical zone moves from the cylinder to the dome.
- **It depends on a coverage law with no physical counterpart.** Below its turnaround radius, each ply keeps 20 to 35 % of its thickness and becomes nearly circumferential. Removing the hoop plies from the dome raises the peak instead of lowering it (0.86 to 1.16 for design 0122).
- **Half of the DOE is geometrically inconsistent.** In 202 of 384 designs the helical turnaround radius R·sin α is larger than the boss radius, so no geodesically wound ply can reach the polar opening; the coverage floor is what covers the pole.

![Fiber index against mesh density for four designs, with the DOE coverage law (top) and with plies stopped at their turnaround radius (bottom)](figures/mesh_study.png)

Re-running the learning protocol on other targets ([`paper/main.pdf`](paper/main.pdf), Table 4):

| Target | Ridge, held-out R² | MLP, held-out R² |
|---|---:|---:|
| Maximum, original DOE (above) | 0.73 | 0.78 |
| 95th percentile, original DOE (mesh-converged) | 0.27 | 0.48 |
| Maximum, 182 consistent designs, plies stopped at turnaround | 0.21 | −1.25 |
| 95th percentile, same | 0.48 | −0.33 |

On the converged 95th percentile, C is within 10 % of 1 for 369 of the 384 designs: away from the polar opening, the fast model and the shell model agree. The consistent variant is not mesh-converged either (its 95th percentile moves by up to 18 % between 48 and 64), because dropping plies creates thickness steps.

What holds: the pipeline, the agreement of the two models away from the pole, and the correction as a conservative margin with respect to the 95th percentile. What does not: the dome peak as a converged physical quantity, and R² 0.862 as evidence of a physical dome effect. It measures how well the MLP reproduces this numerical target.

## What I built

The optimizer and fast model come from an existing 2025 codebase that I started from ([Credits](#credits)). Everything below is my own work.

- **CalculiX vessel model** ([`generate_calculix_composite_case.py`](code/high_fidelity/generate_calculix_composite_case.py)). Cylinder, dome and polar boss meshed as S8R composite shells; winding angle from Clairaut's geodesic law, sin α = r_t / r; one boss fixed axially, the other sliding. [`postprocess_calculix_case.py`](code/high_fidelity/postprocess_calculix_case.py) averages the stress per ply and rotates it into the fiber frame.
- Two DOEs: 128 cases of varying size, then 384 cases at a fixed 11 L volume ([`generate_doe_11l.py`](code/config/generate_doe_11l.py)).
- Correctors ([`extended_statistical_validation.py`](code/calibration/extended_statistical_validation.py)): constant, OLS, Ridge and a 24×12 tanh MLP, all in NumPy. Target log C, so the largest ratios do not dominate the loss.
- Optimizer coupling ([`ga_correction.py`](code/optimization/ga_correction.py)): the fitness uses FI_fast × Ĉ(x) × exp(m₉₅), m₉₅ being the 95th percentile of the MLP's cross-validation residuals. The GA maximizes the inner-to-outer radius ratio and penalizes designs above 1 at 87 MPa.
- Audit campaigns ([`code/paper/`](code/paper)): mesh study, coverage ablations, the strict-turnaround variant, and the paper's figures and numbers, all generated from files in [`data/`](data).

![Workflow: fast model and CalculiX DOE, correction model, GA, CalculiX re-check, published tests](figures/fig_workflow_en.png)

## Limitations

Against six published burst tests, the shell model is 4 to 22 % low on five (Hu 2021, Agne 2025, Jin 2022) and does not reproduce the DLR 2025 test (−74 %, failure near a helical turnaround radius, the region the audit flags); details in [docs/VALIDATION.md](docs/VALIDATION.md). Beyond that: linear elastic shells with no liner, damage or contact; the peak always lands on the fixed-boss dome, so the boundary condition contributes; one vessel family (11 L, single boss, carbon/epoxy); and the optimizer is not public, so the GA check cannot be re-run from here. A next version needs a DOE with α ≥ arcsin(r_boss/R), a tow-accumulation model at the turnaround, and a refined model of the polar region.

## Reproduce

```bash
pip install numpy pandas matplotlib
python code/calibration/extended_statistical_validation.py   # about 3 min; rewrites the metrics JSON and two figures
python code/paper/make_paper_figures.py                      # paper figures and numbers from data/
bash paper/build_paper.sh                                    # figures + LaTeX (needs latexmk)
```

CalculiX runs use [`code/paper/run_campaign.py`](code/paper/run_campaign.py) with CalculiX 2.23 (set `CCX`); about 2 min per design at 24×24 and 11 min at 64×64 with 2 solver threads. Raw `.frd` files are not versioned; per-design summaries are in `data/`. [`code/archive/`](code/archive) holds early exploration scripts not used here.

## Corrections

October 2026: earlier versions presented the dome peak as the physical effect the correction captures, and blamed hoop plies for it. The audit shows the peak is mesh-dependent, comes from the coverage law for every ply type, and that 202 of the 384 designs are geometrically inconsistent. The metrics are unchanged; their interpretation is. The old school reports were replaced by [`paper/main.pdf`](paper/main.pdf).

## Credits

- Genetic-algorithm optimizer and fast analytical model: an existing 2025 codebase, not written by me and not included here.
- [CalculiX](http://www.calculix.de/) 2.23 and ParaView.
- Published tests: Hu, Chen & Pan (2021); Agne et al. (2025); Jin, Cheng, Bai, Paik & Li (2022); Lüders, Ropte, Schmidt & Liebisch (2025). See [`paper/references.bib`](paper/references.bib).

Clément Dumeril · [clement.dumeril.net](https://clement.dumeril.net) · [github.com/clementdumeril](https://github.com/clementdumeril)
