#!/usr/bin/env python3
"""Extended statistical validation script for the 384-case single-boss COPV dataset.

Calculates mean and standard deviation of RMSE, MAE, and R2 under repeated 8-fold cross-validation.
Generates diagnostic plots (prediction vs actual, residuals histogram) and evaluates OOD performance.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_utils import CASES_FILE, COMPARISON, FIGURES, MODELS, read_csv, write_json  # noqa: E402

DATASET = COMPARISON / "fiber_proxy_dataset.csv"
DOE_CASES = Path(os.environ.get("TYPE4_CASES_FILE", ROOT / "config" / "doe_11l_single_boss_cases.json"))
OUT_METRICS = COMPARISON / "extended_statistical_metrics.json"

NUMERIC_FEATURES = [
    "pressure_mpa", "inner_radius_mm", "cylindrical_length_mm", "boss_radius_mm",
    "dome_radius_factor", "helical_angle_min_deg", "helical_angle_max_deg",
    "transition_angle_max_deg", "hoop_angle_deg", "helical_pairs",
    "transition_pairs", "hoop_pairs", "total_thickness_mm", "layer_count",
    "python_fiber_stress_ratio_max", "python_fiber_proxy_pressure_mpa",
    "python_raw_max_combined"
]

DERIVED_FEATURES = [
    "boss_radius_ratio", "length_radius_ratio", "pressure_radius_over_thickness",
    "hoop_pair_fraction", "transition_pair_fraction", "helical_hoop_angle_gap_deg"
]

CATEGORICAL_FEATURES = ["dome_shape"]

def as_float(val: any, default: float = 0.0) -> float:
    try:
        if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default

def load_data():
    # 1. Load cases and roles
    if not DOE_CASES.exists():
        print(f"Error: Cases file not found at {DOE_CASES}")
        sys.exit(1)
    
    cases_data = json.loads(DOE_CASES.read_text(encoding="utf-8"))
    roles = {c["case_id"]: c.get("campaign", {}).get("role", "unknown") for c in cases_data.get("cases", [])}
    
    # 2. Load dataset CSV
    if not DATASET.exists():
        print(f"Error: Dataset not found at {DATASET}")
        sys.exit(1)
        
    df = pd.read_csv(DATASET)
    df["doe_role"] = df["case_id"].map(roles).fillna("unknown")
    
    # Check for duplicates or data leakage
    # We check if there are duplicate geometries (same radius, length, boss, thickness, layup)
    geom_cols = ["inner_radius_mm", "cylindrical_length_mm", "boss_radius_mm", "total_thickness_mm", "helical_angle_min_deg", "layer_count"]
    duplicates = df.duplicated(subset=geom_cols).sum()
    
    return df, duplicates

def extract_features(df: pd.DataFrame):
    rows = df.to_dict(orient="records")
    categories = {name: sorted(list(df[name].unique())) for name in CATEGORICAL_FEATURES}
    
    values = []
    for row in rows:
        x = [as_float(row.get(name)) for name in NUMERIC_FEATURES]
        radius = max(as_float(row.get("inner_radius_mm")), 1e-9)
        thickness = max(as_float(row.get("total_thickness_mm")), 1e-9)
        helical_pairs = as_float(row.get("helical_pairs"))
        transition_pairs = as_float(row.get("transition_pairs"))
        hoop_pairs = as_float(row.get("hoop_pairs"))
        pair_total = max(helical_pairs + transition_pairs + hoop_pairs, 1e-9)
        
        x.extend([
            as_float(row.get("boss_radius_mm")) / radius,
            as_float(row.get("cylindrical_length_mm")) / radius,
            as_float(row.get("pressure_mpa")) * radius / thickness,
            hoop_pairs / pair_total,
            transition_pairs / pair_total,
            abs(as_float(row.get("hoop_angle_deg")) - as_float(row.get("helical_angle_min_deg")))
        ])
        
        for name in CATEGORICAL_FEATURES:
            observed = row.get(name, "") or "unknown"
            x.extend([1.0 if observed == value else 0.0 for value in categories[name]])
            
        values.append(x)
        
    feature_names = list(NUMERIC_FEATURES) + list(DERIVED_FEATURES)
    for name in CATEGORICAL_FEATURES:
        feature_names.extend([f"{name}={val}" for val in categories[name]])
        
    X = np.array(values, dtype=float)
    y = np.array([[math.log(max(1.0, as_float(row.get("fiber_correction_ratio_calculix_over_python"), 1.0)))] for row in rows], dtype=float)
    
    return X, y, feature_names

# NumPy implementations of the models
class ConstantModel:
    def fit(self, x: np.ndarray, y: np.ndarray):
        # Predict 95th percentile of y in train
        self.c_log = float(np.quantile(y, 0.95))
        return self
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((x.shape[0], 1), self.c_log)

class OLSModel:
    def fit(self, x: np.ndarray, y: np.ndarray):
        x_bias = np.column_stack([np.ones(x.shape[0]), x])
        self.beta = np.linalg.pinv(x_bias.T @ x_bias) @ x_bias.T @ y
        return self
    def predict(self, x: np.ndarray) -> np.ndarray:
        x_bias = np.column_stack([np.ones(x.shape[0]), x])
        return x_bias @ self.beta

class RidgeModel:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
    def fit(self, x: np.ndarray, y: np.ndarray):
        x_bias = np.column_stack([np.ones(x.shape[0]), x])
        penalty = np.eye(x_bias.shape[1]) * self.alpha
        penalty[0, 0] = 0.0 # intercept is not penalized
        self.beta = np.linalg.solve(x_bias.T @ x_bias + penalty, x_bias.T @ y)
        return self
    def predict(self, x: np.ndarray) -> np.ndarray:
        x_bias = np.column_stack([np.ones(x.shape[0]), x])
        return x_bias @ self.beta

class MLPModel:
    def __init__(self, h1: int = 24, h2: int = 12, seed: int = 42):
        self.h1 = h1
        self.h2 = h2
        self.seed = seed
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 6000, lr: float = 0.002, l2: float = 4e-4):
        rng = np.random.default_rng(self.seed)
        i_dim, o_dim = x.shape[1], y.shape[1]
        self.weights = {
            "W1": rng.normal(0.0, math.sqrt(2.0 / i_dim), size=(i_dim, self.h1)),
            "b1": np.zeros(self.h1),
            "W2": rng.normal(0.0, math.sqrt(2.0 / self.h1), size=(self.h1, self.h2)),
            "b2": np.zeros(self.h2),
            "W3": rng.normal(0.0, math.sqrt(2.0 / self.h2), size=(self.h2, o_dim)),
            "b3": np.zeros(o_dim),
        }
        m = {k: np.zeros_like(v) for k, v in self.weights.items()}
        v = {k: np.zeros_like(v) for k, v in self.weights.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for epoch in range(1, epochs + 1):
            z1 = x @ self.weights["W1"] + self.weights["b1"]
            a1 = np.tanh(z1)
            z2 = a1 @ self.weights["W2"] + self.weights["b2"]
            a2 = np.tanh(z2)
            pred = a2 @ self.weights["W3"] + self.weights["b3"]
            
            err = pred - y
            grad_y = 2.0 * err / err.size
            grad = {
                "W3": a2.T @ grad_y + 2.0 * l2 * self.weights["W3"],
                "b3": grad_y.sum(axis=0),
            }
            da2 = grad_y @ self.weights["W3"].T
            dz2 = da2 * (1.0 - np.tanh(z2)**2)
            grad["W2"] = a1.T @ dz2 + 2.0 * l2 * self.weights["W2"]
            grad["b2"] = dz2.sum(axis=0)
            da1 = dz2 @ self.weights["W2"].T
            dz1 = da1 * (1.0 - np.tanh(z1)**2)
            grad["W1"] = x.T @ dz1 + 2.0 * l2 * self.weights["W1"]
            grad["b1"] = dz1.sum(axis=0)
            
            for k in self.weights:
                m[k] = beta1 * m[k] + (1.0 - beta1) * grad[k]
                v[k] = beta2 * v[k] + (1.0 - beta2) * (grad[k] * grad[k])
                m_hat = m[k] / (1.0 - beta1**epoch)
                v_hat = v[k] / (1.0 - beta2**epoch)
                self.weights[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return self
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        a1 = np.tanh(x @ self.weights["W1"] + self.weights["b1"])
        a2 = np.tanh(a1 @ self.weights["W2"] + self.weights["b2"])
        return a2 @ self.weights["W3"] + self.weights["b3"]

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))

def rmse_score(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae_score(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def main():
    print("Loading data...")
    df, duplicates = load_data()
    print(f"Dataset loaded. Row count: {len(df)}. Geometry duplicates: {duplicates}")
    
    # Check data leakage: train and holdout case lists must be disjoint
    train_cases = set(df[df["doe_role"] == "train"]["case_id"])
    eval_cases = set(df[df["doe_role"] != "train"]["case_id"])
    leakage = len(train_cases.intersection(eval_cases))
    print(f"Data leakage check: Intersection between train and eval set = {leakage} cases.")
    
    X_raw, y, feature_names = extract_features(df)
    
    # Scale variables
    train_mask = (df["doe_role"] == "train").values
    mean = X_raw[train_mask].mean(axis=0)
    std = X_raw[train_mask].std(axis=0)
    std[std < 1e-12] = 1.0
    X = (X_raw - mean) / std
    
    train_idx = np.where(df["doe_role"] == "train")[0]
    eval_idx = np.where(df["doe_role"] != "train")[0]
    holdout_idx = np.where(df["doe_role"] == "holdout")[0]
    boundary_idx = np.where(df["doe_role"] == "boundary")[0]
    pressure_scale_idx = np.where(df["doe_role"] == "pressure_scale")[0]
    
    print(f"Train count: {len(train_idx)}, Eval count: {len(eval_idx)} (Holdout: {len(holdout_idx)}, Boundary: {len(boundary_idx)}, Pressure Scale: {len(pressure_scale_idx)})")
    
    models = {
        "Constant": ConstantModel(),
        "OLS": OLSModel(),
        "Ridge": RidgeModel(alpha=10.0),
        "MLP": MLPModel(seed=42)
    }
    
    # 1. Cross-validation on Train Set
    n_folds = 8
    n_repeats = 4
    cv_metrics = {name: {"rmse": [], "mae": [], "r2": []} for name in models}
    
    order = np.argsort(y[train_idx, 0])
    n_train = len(train_idx)
    
    print("Running cross-validation...")
    for repeat in range(n_repeats):
        rng = np.random.default_rng(20260624 + repeat)
        buckets = [[] for _ in range(n_folds)]
        for start in range(0, n_train, n_folds):
            chunk = np.array(order[start:start+n_folds], copy=True)
            rng.shuffle(chunk)
            for j, idx in enumerate(chunk):
                buckets[(j + repeat) % n_folds].append(int(train_idx[idx]))
                
        for fold_id, bucket in enumerate(buckets):
            val_f = np.array(bucket, dtype=int)
            val_set = set(val_f.tolist())
            train_f = np.array([idx for idx in train_idx if idx not in val_set], dtype=int)
            
            # Local scale
            mean_f = X_raw[train_f].mean(axis=0)
            std_f = X_raw[train_f].std(axis=0)
            std_f[std_f < 1e-12] = 1.0
            x_train_f = (X_raw[train_f] - mean_f) / std_f
            x_val_f = (X_raw[val_f] - mean_f) / std_f
            
            for name, model in models.items():
                model.fit(x_train_f, y[train_f])
                pred_f = model.predict(x_val_f)
                cv_metrics[name]["rmse"].append(rmse_score(y[val_f], pred_f))
                cv_metrics[name]["mae"].append(mae_score(y[val_f], pred_f))
                cv_metrics[name]["r2"].append(r2_score(y[val_f], pred_f))
                
    summary_cv = {}
    for name in models:
        summary_cv[name] = {
            "rmse_mean": float(np.mean(cv_metrics[name]["rmse"])),
            "rmse_std": float(np.std(cv_metrics[name]["rmse"])),
            "mae_mean": float(np.mean(cv_metrics[name]["mae"])),
            "mae_std": float(np.std(cv_metrics[name]["mae"])),
            "r2_mean": float(np.mean(cv_metrics[name]["r2"])),
            "r2_std": float(np.std(cv_metrics[name]["r2"]))
        }
    
    # 2. Evaluation on official evaluation sets
    print("Evaluating final models...")
    summary_eval = {}
    predictions_on_eval = {}
    
    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        
        # Whole eval set (96 cases)
        pred_eval = model.predict(X[eval_idx])
        predictions_on_eval[name] = pred_eval
        
        # Separated sets
        pred_holdout = model.predict(X[holdout_idx])
        pred_boundary = model.predict(X[boundary_idx])
        pred_pressure = model.predict(X[pressure_scale_idx])
        
        summary_eval[name] = {
            "eval_rmse": rmse_score(y[eval_idx], pred_eval),
            "eval_mae": mae_score(y[eval_idx], pred_eval),
            "eval_r2": r2_score(y[eval_idx], pred_eval),
            "holdout_rmse": rmse_score(y[holdout_idx], pred_holdout),
            "holdout_mae": mae_score(y[holdout_idx], pred_holdout),
            "holdout_r2": r2_score(y[holdout_idx], pred_holdout),
            "boundary_rmse": rmse_score(y[boundary_idx], pred_boundary),
            "boundary_mae": mae_score(y[boundary_idx], pred_boundary),
            "boundary_r2": r2_score(y[boundary_idx], pred_boundary),
            "pressure_rmse": rmse_score(y[pressure_scale_idx], pred_pressure),
            "pressure_mae": mae_score(y[pressure_scale_idx], pred_pressure),
            "pressure_r2": r2_score(y[pressure_scale_idx], pred_pressure)
        }
        
    # 3. Zone-separated performance on Eval Set
    zones = ["left_dome", "cylinder", "junction", "boss"]
    zone_metrics = {}
    
    for zone in zones:
        zone_mask = (df["calculix_fiber_proxy_zone"] == zone).values
        zone_eval_idx = np.where((df["doe_role"] != "train") & zone_mask)[0]
        
        if len(zone_eval_idx) > 0:
            zone_metrics[zone] = {}
            for name, model in models.items():
                pred_zone = model.predict(X[zone_eval_idx])
                zone_metrics[zone][name] = {
                    "count": int(len(zone_eval_idx)),
                    "rmse": rmse_score(y[zone_eval_idx], pred_zone),
                    "mae": mae_score(y[zone_eval_idx], pred_zone),
                    "r2": r2_score(y[zone_eval_idx], pred_zone)
                }
                
    # 4. Generate plots
    # Plot 1: Scatter plot
    plt.figure(figsize=(8, 6), dpi=150)
    eval_y_actual = np.exp(y[eval_idx, 0])
    
    for name in ["OLS", "Ridge", "MLP"]:
        eval_y_pred = np.exp(predictions_on_eval[name][:, 0])
        plt.scatter(eval_y_actual, eval_y_pred, label=f"{name} ($R^2={summary_eval[name]['eval_r2']:.3f}$)", alpha=0.7, s=25)
        
    plt.plot([1.0, 4.5], [1.0, 4.5], 'k--', label="Reference $C_{\\text{CalculiX}} = C_{\\text{Python}}$")
    plt.xlabel("Actual CalculiX Correction Factor ($C_{\\text{CalculiX}}$)")
    plt.ylabel("Predicted Correction Factor ($C_{\\text{predicted}}$)")
    plt.title("Prediction vs. Reference on Evaluation Set (96 cases)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(FIGURES / "extended_benchmark_nuage_en.png", bbox_inches="tight")
    plt.close()
    
    # Plot 2: Residuals Histogram
    plt.figure(figsize=(8, 6), dpi=150)
    for name in ["OLS", "Ridge", "MLP"]:
        residuals = y[eval_idx, 0] - predictions_on_eval[name][:, 0]
        plt.hist(residuals, bins=15, alpha=0.5, label=f"{name} (MAE={summary_eval[name]['eval_mae']:.3f})", density=True)
        
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel("Prediction Residual $\\log(C_{\\text{actual}}) - \\log(C_{\\text{predicted}})$")
    plt.ylabel("Density")
    plt.title("Histogram of Log Residuals on Evaluation Set")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(FIGURES / "extended_benchmark_residus_en.png", bbox_inches="tight")
    plt.close()
    
    # Write json
    output_payload = {
        "dataset_metadata": {
            "path": str(DATASET),
            "total_rows": len(df),
            "duplicates": int(duplicates),
            "data_leakage_train_eval_intersection": leakage,
            "train_count": len(train_idx),
            "eval_count": len(eval_idx),
            "holdout_count": len(holdout_idx),
            "boundary_count": len(boundary_idx),
            "pressure_scale_count": len(pressure_scale_idx)
        },
        "cross_validation_metrics": summary_cv,
        "evaluation_metrics": summary_eval,
        "zone_metrics": zone_metrics
    }
    
    write_json(OUT_METRICS, output_payload)
    print(f"Successfully generated metrics and figures. Saved json to {OUT_METRICS}")

if __name__ == "__main__":
    main()
