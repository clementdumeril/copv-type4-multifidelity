# Validation details

Everything here supports the main [README](../README.md). Numbers come from files in [`data/`](../data).

## Design of experiments

The first campaign (128 cases) varied the vessel size widely. After feedback that this mixed too many effects, the second campaign fixed the internal volume at 11 L ([`code/config/generate_doe_11l.py`](../code/config/generate_doe_11l.py)). All 384 cases were solved in CalculiX.

| Parameter | Range in the 384-case DOE |
|---|---|
| Inner radius | 76 to 96 mm (cylinder length follows from the 11 L volume: 257 to 550 mm) |
| Boss radius | 21.5 to 34.1 mm |
| Dome shape | hemispherical, isotensoid-like, variable contour (128 each) |
| Helical angle | 12 to 26° |
| Helical / transition / hoop pairs | 4–5 / 1–2 / 2–3 |
| Total thickness | 10.5 to 26.1 mm |
| Pressure | 70 MPa (360 cases); 52.5 and 87.5 MPa (12 each, pressure-extrapolation split) |
| Hoop thickness kept in the dome (`coverage_epsilon`) | 0.20 to 0.35 |

### Coverage law near the pole

Below its turnaround radius (0.985 R for hoop plies, R sin α for the others), every ply keeps `coverage_epsilon` (0.20 to 0.35) of its thickness, and the code assigns it an angle near 87° (a numerical convention: a geodesic ply does not exist there). In 179 of the 198 dome-critical designs the peak is in a hoop ply at the DOE mesh, but the mechanism is not specific to hoops: removing hoop coverage from the dome moves the peak to a transition ply at the same radius and raises it, and at 64×64 the critical ply is a transition ply in all four mesh-study designs. In 202 of the 384 designs no helical ply reaches the polar opening (R sin α larger than the boss radius); of the other 182, 132 have R sin α below 0.9 r_boss, so only about 50 designs are close to windable (α ≈ arcsin(r_boss/R)). The peak always lands on the left pole because the case generator labels the two polar rings differently; removing the axial clamp (`data/paper_bc_minimal_pins.csv`) changes the peak by at most 4 %. See the audit in [`paper/main.pdf`](../paper/main.pdf), section 6, and the data in `data/paper_mesh_study.csv`, `data/fiber_proxy_dataset_strict.csv` and `data/paper_hoop0_11l_0122.csv`.

Case roles are fixed in [`data/doe_11l_single_boss_cases.json`](../data/doe_11l_single_boss_cases.json): 288 train, 48 in-distribution holdout, 24 geometric boundary, 24 pressure scale.

## Correctors

All models are in [`code/calibration/extended_statistical_validation.py`](../code/calibration/extended_statistical_validation.py), written in NumPy.

- Target: log C, with C = FI(CalculiX) / FI(fast model), the fiber index |σ11|/X of the worst ply.
- Inputs: 17 design and fast-model quantities, 6 derived ratios (P·r/t, boss-to-radius ratio, length-to-radius ratio, hoop and transition pair fractions, helical-hoop angle gap) and the one-hot dome shape, standardized on the training set.
- Models: constant, OLS, Ridge (α = 10), MLP 24×12 with tanh, trained with Adam (learning rate 0.002, L2 4e-4, 6000 full-batch epochs, seed 42), backpropagation written by hand.
- Cross-validation: 8 folds, repeated 4 times with different shuffles, on the 288 training cases only.
- Leakage check: no duplicated design and no case shared between training and evaluation sets.

| R² on log C | Constant | OLS | Ridge | MLP |
|---|---:|---:|---:|---:|
| Repeated 8-fold CV (288) | −3.91 | 0.791 | 0.796 | 0.862 |
| Held-out, all (96) | −4.01 | 0.717 | 0.727 | 0.778 |
| In-distribution holdout (48) | −5.90 | 0.770 | 0.787 | 0.789 |
| Geometric boundary (24) | −2.27 | 0.647 | 0.672 | 0.800 |
| Pressure extrapolation (24) | −4.02 | 0.726 | 0.703 | 0.714 |

Full metrics, including RMSE, MAE and per-zone scores: [`data/extended_statistical_metrics.json`](../data/extended_statistical_metrics.json).

## GA check

The optimizer was run with 10 seeds. That gave 17 distinct designs, all with 12 plies and a 15.22 mm wall. Each was rebuilt and re-solved in CalculiX at the 87 MPa design burst pressure ([`data/ga_statistical_validation.csv`](../data/ga_statistical_validation.csv)).

| Prediction is at or above CalculiX… | p95 of the fiber index | pointwise maximum |
|---|---:|---:|
| fast model, raw | 0 / 17 | 0 / 17 |
| fast model × MLP correction (p95 margin) | 17 / 17 | 4 / 17 |

In 5 of the 17 designs, the CalculiX maximum exceeds 1.

## Published burst tests

The comparisons use the same shell model and the same fiber index as the DOE. Geometry and layups were rebuilt from each paper, and some of them were read from figures.

| Source | Vessel | Measured | Model | Gap |
|---|---|---:|---:|---:|
| Hu, Chen & Pan 2021 | 70 MPa H₂ vessel, no dome reinforcement | 161 MPa (first fiber damage) | 148.7 MPa | −7.6 % |
| Agne et al. 2025 | GFRP | 29.39 MPa | 25.4 to 27.7 MPa | −6 to −14 % |
| Agne et al. 2025 | CFRP | 27.04 MPa | 25.7 to 25.9 MPa | −4 to −5 % |
| Jin et al. 2024 | EX-A, 32×32 mesh, p99 | 65.2 MPa | 58.6 MPa | −10 % |
| Jin et al. 2024 | EX-B, 32×32 mesh, p99 | 65.17 MPa | 50.8 MPa | −22 % |
| Lueders et al. 2025 (DLR) | SN03, simplified layup | 25.37 MPa | 11.8 MPa | −53 % |
| Lueders et al. 2025 (DLR) | SN03, published layup book | 25.37 MPa | 6.5 MPa | −74 % |

The ranges for Agne span the maximum, p95 and p99 of the fiber index. For Hu, the pressure is close, but in the model the first failed ply is a hoop ply in the cylinder, while the paper reports the dome region. The DLR test is not reproduced. With the exact layup, the 22.2° helical layer becomes critical near its turnaround radius. The likely causes are:

- the dome contour, which is a spherical-cap approximation instead of the measured liner contour;
- the coverage law near the turnaround;
- the absence of progressive damage.

## Mesh study

The mesh study uses vessels EX-A and EX-B of Jin et al. 2024 (online 2022), at 16, 24 and 32 elements per direction ([`data/jin2022_mesh_convergence.csv`](../data/jin2022_mesh_convergence.csv)).

![Mesh study on the Jin et al. 2024 vessels](../figures/jin2022_mesh_convergence.png)

| Predicted burst (MPa) | 16 | 24 | 32 |
|---|---:|---:|---:|
| EX-A, maximum | 41.2 | 48.3 | 43.4 |
| EX-A, p99 | 43.3 | 55.1 | 58.6 |
| EX-B, maximum | 33.4 | 44.1 | 38.0 |
| EX-B, p99 | 34.7 | 44.5 | 50.8 |

The maximum does not converge. The p99 value increases steadily, but it still moves by 6 % (EX-A) and 14 % (EX-B) between the last two meshes.

## Liner / boss contact

Contact was tested separately on two cases of the first DOE. It was never included in the 384-case DOE.

- Changing the penalty stiffness moved the stresses by about 0.3 % and the displacements by 3.6 to 4.6 %.
- Going from tied to penalty contact moved the stresses by 7.5 to 8 %.
- Frictional contact never ran reliably, and a 0.02 mm clearance did not converge.
