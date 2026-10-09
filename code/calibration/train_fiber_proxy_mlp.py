#!/usr/bin/env python3
"""Train a small MLP correction model for the DOE-B 11 L fibre-proxy target.

The target is deliberately not the old raw `max_combined` failure index.  The
model learns the finite-element discrepancy on a burst-scale observable:

    C_fibre = max(|sigma_11|/X)_CalculiX / max(|sigma_11|/X)_Python

The saved safe prediction is conservative by adding a cross-validation residual
margin in log-space.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_utils import CASES_FILE, COMPARISON, FIGURES, MODELS, as_float, ensure_dirs, read_csv, write_csv, write_json  # noqa: E402


DATASET = COMPARISON / "fiber_proxy_dataset.csv"
MODEL_JSON = MODELS / "fiber_proxy_mlp_model.json"
DOE_CASES = CASES_FILE
MLP_LEARNING_RATE = 0.0015
MLP_EPOCHS = 8000

NUMERIC_FEATURES = [
    "pressure_mpa",
    "inner_radius_mm",
    "cylindrical_length_mm",
    "boss_radius_mm",
    "dome_radius_factor",
    "helical_angle_min_deg",
    "helical_angle_max_deg",
    "transition_angle_max_deg",
    "hoop_angle_deg",
    "helical_pairs",
    "transition_pairs",
    "hoop_pairs",
    "total_thickness_mm",
    "layer_count",
    "python_fiber_stress_ratio_max",
    "python_fiber_proxy_pressure_mpa",
    "python_raw_max_combined",
]

DERIVED_FEATURES = [
    "boss_radius_ratio",
    "length_radius_ratio",
    "pressure_radius_over_thickness",
    "hoop_pair_fraction",
    "transition_pair_fraction",
    "helical_hoop_angle_gap_deg",
]

CATEGORICAL_FEATURES = ["dome_shape"]


def finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def positive_residual_margin(residuals: np.ndarray, policy: str) -> float:
    """Return a non-negative log-space safety margin for actual-minus-predicted residuals."""
    positive = residuals[residuals > 0.0]
    if policy == "raw" or positive.size == 0:
        return 0.0
    if policy == "p90":
        value = float(np.quantile(positive, 0.90))
    elif policy == "p95":
        value = float(np.quantile(positive, 0.95))
    elif policy == "p99":
        value = float(np.quantile(positive, 0.99))
    elif policy in {"safe", "max"}:
        value = float(np.max(positive))
    else:
        raise ValueError(f"Unknown margin policy: {policy}")
    return float(max(0.025, value))


def conservative_fraction(records: list[dict[str, Any]], margin_log: float) -> float:
    if not records:
        return 0.0
    return float(
        np.mean(
            [
                math.exp(math.log(max(r["mlp_C_fibre_raw"], 1e-12)) + margin_log) >= r["actual_C_fibre"]
                for r in records
            ]
        )
    )


def load_dataset() -> list[dict[str, str]]:
    rows = []
    for row in read_csv(DATASET):
        target = as_float(row, "fiber_correction_ratio_calculix_over_python", math.nan)
        py_ratio = as_float(row, "python_fiber_stress_ratio_max", math.nan)
        calc_ratio = as_float(row, "calculix_fiber_stress_ratio_max", math.nan)
        if finite_positive(target) and finite_positive(py_ratio) and finite_positive(calc_ratio):
            rows.append(row)
    return rows


def load_case_roles() -> dict[str, str]:
    if not DOE_CASES.exists():
        return {}
    data = json.loads(DOE_CASES.read_text(encoding="utf-8"))
    return {case["case_id"]: case.get("campaign", {}).get("role", "") for case in data.get("cases", [])}


def attach_case_roles(rows: list[dict[str, str]], roles: dict[str, str]) -> None:
    for row in rows:
        row["doe_role"] = roles.get(row.get("case_id", ""), "unknown") or "unknown"


def categorical_values(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {name: sorted({row.get(name, "") or "unknown" for row in rows}) for name in CATEGORICAL_FEATURES}


def raw_features(rows: list[dict[str, str]], categories: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    values: list[list[float]] = []
    feature_names = list(NUMERIC_FEATURES) + list(DERIVED_FEATURES)
    for name in CATEGORICAL_FEATURES:
        feature_names.extend([f"{name}={value}" for value in categories[name]])

    for row in rows:
        x = [as_float(row, name) for name in NUMERIC_FEATURES]
        radius = max(as_float(row, "inner_radius_mm"), 1e-9)
        thickness = max(as_float(row, "total_thickness_mm"), 1e-9)
        helical_pairs = as_float(row, "helical_pairs")
        transition_pairs = as_float(row, "transition_pairs")
        hoop_pairs = as_float(row, "hoop_pairs")
        pair_total = max(helical_pairs + transition_pairs + hoop_pairs, 1e-9)
        x.extend(
            [
                as_float(row, "boss_radius_mm") / radius,
                as_float(row, "cylindrical_length_mm") / radius,
                as_float(row, "pressure_mpa") * radius / thickness,
                hoop_pairs / pair_total,
                transition_pairs / pair_total,
                abs(as_float(row, "hoop_angle_deg") - as_float(row, "helical_angle_min_deg")),
            ]
        )
        for name in CATEGORICAL_FEATURES:
            observed = row.get(name, "") or "unknown"
            x.extend([1.0 if observed == value else 0.0 for value in categories[name]])
        values.append(x)
    return np.array(values, dtype=float), feature_names


def target_log_c(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array(
        [[math.log(max(1.0, as_float(row, "fiber_correction_ratio_calculix_over_python", 1.0)))] for row in rows],
        dtype=float,
    )


def standardize_fit(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std < 1e-12] = 1.0
    return (raw - mean) / std, mean, std


def standardize_apply(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (raw - mean) / std


def init_model(input_dim: int, output_dim: int = 1, seed: int = 20260529) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h1 = 24
    h2 = 12
    return {
        "W1": rng.normal(0.0, math.sqrt(2.0 / input_dim), size=(input_dim, h1)),
        "b1": np.zeros(h1),
        "W2": rng.normal(0.0, math.sqrt(2.0 / h1), size=(h1, h2)),
        "b2": np.zeros(h2),
        "W3": rng.normal(0.0, math.sqrt(2.0 / h2), size=(h2, output_dim)),
        "b3": np.zeros(output_dim),
    }


def forward(model: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = x @ model["W1"] + model["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ model["W2"] + model["b2"]
    a2 = np.tanh(z2)
    y = a2 @ model["W3"] + model["b3"]
    return y, {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int = MLP_EPOCHS,
    learning_rate: float = MLP_LEARNING_RATE,
    l2: float = 4e-4,
) -> tuple[dict[str, np.ndarray], list[float]]:
    model = init_model(x.shape[1], y.shape[1], seed=seed)
    m = {key: np.zeros_like(value) for key, value in model.items()}
    v = {key: np.zeros_like(value) for key, value in model.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        pred, cache = forward(model, x)
        err = pred - y
        reg = sum(float(np.sum(model[key] * model[key])) for key in ("W1", "W2", "W3"))
        loss = float(np.mean(err * err) + l2 * reg)
        history.append(loss)

        grad_y = 2.0 * err / err.size
        grad: dict[str, np.ndarray] = {}
        grad["W3"] = cache["a2"].T @ grad_y + 2.0 * l2 * model["W3"]
        grad["b3"] = grad_y.sum(axis=0)
        da2 = grad_y @ model["W3"].T
        dz2 = da2 * (1.0 - np.tanh(cache["z2"]) ** 2)
        grad["W2"] = cache["a1"].T @ dz2 + 2.0 * l2 * model["W2"]
        grad["b2"] = dz2.sum(axis=0)
        da1 = dz2 @ model["W2"].T
        dz1 = da1 * (1.0 - np.tanh(cache["z1"]) ** 2)
        grad["W1"] = cache["x"].T @ dz1 + 2.0 * l2 * model["W1"]
        grad["b1"] = dz1.sum(axis=0)

        for key in model:
            m[key] = beta1 * m[key] + (1.0 - beta1) * grad[key]
            v[key] = beta2 * v[key] + (1.0 - beta2) * (grad[key] * grad[key])
            m_hat = m[key] / (1.0 - beta1**epoch)
            v_hat = v[key] / (1.0 - beta2**epoch)
            model[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if epoch > 600 and len(history) > 240:
            recent = float(np.mean(history[-80:]))
            older = float(np.mean(history[-240:-160]))
            if abs(older - recent) < 1e-8:
                break
    return model, history


def predict_log_c(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return forward(model, x)[0]


def balanced_folds(y: np.ndarray, folds: int = 8, repeats: int = 4, seed: int = 20260529) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    n = y.shape[0]
    folds = max(2, min(folds, n))
    order = np.argsort(y[:, 0])
    out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        buckets: list[list[int]] = [[] for _ in range(folds)]
        for start in range(0, n, folds):
            chunk = np.array(order[start : start + folds], copy=True)
            rng.shuffle(chunk)
            for j, idx in enumerate(chunk):
                buckets[(j + repeat) % folds].append(int(idx))
        for fold_id, bucket in enumerate(buckets):
            val = np.array(sorted(bucket), dtype=int)
            val_set = set(val.tolist())
            train = np.array([i for i in range(n) if i not in val_set], dtype=int)
            out.append((repeat, fold_id, train, val))
    return out


def cross_validate(raw_x: np.ndarray, y: np.ndarray, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    residuals: list[float] = []
    fold_rmses: list[float] = []
    for repeat, fold_id, train_idx, val_idx in balanced_folds(y):
        x_train, mean, std = standardize_fit(raw_x[train_idx])
        x_val = standardize_apply(raw_x[val_idx], mean, std)
        model, history = train_mlp(x_train, y[train_idx], seed=20260529 + repeat * 101 + fold_id)
        pred = predict_log_c(model, x_val)
        err = y[val_idx] - pred
        fold_rmses.append(float(np.sqrt(np.mean(err * err))))
        residuals.extend(float(value) for value in err[:, 0])
        for local, idx in enumerate(val_idx):
            actual = float(math.exp(y[idx, 0]))
            predicted = float(math.exp(pred[local, 0]))
            records.append(
                {
                    "repeat": repeat,
                    "fold": fold_id,
                    "doe_role": rows[idx].get("doe_role", "unknown"),
                    "case_id": rows[idx]["case_id"],
                    "actual_C_fibre": actual,
                    "mlp_C_fibre_raw": predicted,
                    "log_residual_actual_minus_pred": float(err[local, 0]),
                    "conservative_raw": predicted >= actual,
                    "training_loss_final": float(history[-1]),
                }
            )
    residual_arr = np.array(residuals, dtype=float)
    margin_raw = positive_residual_margin(residual_arr, "raw")
    margin_p90 = positive_residual_margin(residual_arr, "p90")
    margin_p95 = positive_residual_margin(residual_arr, "p95")
    margin_p99 = positive_residual_margin(residual_arr, "p99")
    # Keep the historical `cv_residual_margin_log` as the strict max margin.
    # It is the safest choice for GA screening, while p95 is useful when the
    # correction would otherwise become too punitive for optimization.
    margin = positive_residual_margin(residual_arr, "max")
    metrics = {
        "fold_count": len(fold_rmses),
        "rmse_log_C_cv_mean": float(np.mean(fold_rmses)),
        "rmse_log_C_cv_max": float(np.max(fold_rmses)),
        "raw_conservative_fraction_cv": float(np.mean([r["conservative_raw"] for r in records])),
        "cv_residual_margin_log_raw": margin_raw,
        "cv_residual_margin_log_p90": margin_p90,
        "cv_residual_margin_log_p95": margin_p95,
        "cv_residual_margin_log_p99": margin_p99,
        "cv_residual_margin_log_max": margin,
        "cv_residual_margin_log": margin,
    }
    return records, residual_arr, metrics


def deterministic_audit_holdout(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    holdout = idx[idx % 5 == 0]
    train = idx[idx % 5 != 0]
    return train, holdout


def audit_holdout(raw_x: np.ndarray, y: np.ndarray, rows: list[dict[str, str]], margin_log: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_idx, hold_idx = deterministic_audit_holdout(len(rows))
    x_train, mean, std = standardize_fit(raw_x[train_idx])
    model, history = train_mlp(x_train, y[train_idx], seed=20260601)
    pred = predict_log_c(model, standardize_apply(raw_x[hold_idx], mean, std))
    records = []
    for local, idx in enumerate(hold_idx):
        actual = float(math.exp(y[idx, 0]))
        raw = float(math.exp(pred[local, 0]))
        safe = float(math.exp(pred[local, 0] + margin_log))
        records.append(
            {
                "case_id": rows[idx]["case_id"],
                "doe_role": rows[idx].get("doe_role", "unknown"),
                "actual_C_fibre": actual,
                "mlp_C_fibre_raw": raw,
                "mlp_C_fibre_safe": safe,
                "conservative_raw": raw >= actual,
                "conservative_safe": safe >= actual,
            }
        )
    err = y[hold_idx, 0] - pred[:, 0]
    metrics = {
        "audit_holdout_policy": "deterministic idx % 5 == 0 inside the MLP training scope; official DOE roles are evaluated separately when available",
        "audit_holdout_count": int(len(hold_idx)),
        "audit_train_count": int(len(train_idx)),
        "audit_rmse_log_C_raw": float(np.sqrt(np.mean(err * err))),
        "audit_raw_conservative_fraction": float(np.mean([r["conservative_raw"] for r in records])),
        "audit_safe_conservative_fraction": float(np.mean([r["conservative_safe"] for r in records])),
        "audit_training_loss_final": float(history[-1]),
    }
    return records, metrics


def official_role_validation(
    raw_x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, str]],
    margin_log: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    role_list = [row.get("doe_role", "unknown") or "unknown" for row in rows]
    train_idx = np.array([idx for idx, role in enumerate(role_list) if role == "train"], dtype=int)
    eval_idx = np.array([idx for idx, role in enumerate(role_list) if role != "train"], dtype=int)
    if len(train_idx) < 20 or len(eval_idx) == 0:
        return [], {
            "official_role_validation_policy": "not_available",
            "official_train_count": int(len(train_idx)),
            "official_eval_count": int(len(eval_idx)),
        }

    x_train, mean, std = standardize_fit(raw_x[train_idx])
    model, history = train_mlp(x_train, y[train_idx], seed=20260603, epochs=MLP_EPOCHS, learning_rate=MLP_LEARNING_RATE)
    pred = predict_log_c(model, standardize_apply(raw_x[eval_idx], mean, std))
    records: list[dict[str, Any]] = []
    for local, idx in enumerate(eval_idx):
        actual = float(math.exp(y[idx, 0]))
        raw = float(math.exp(pred[local, 0]))
        safe = float(math.exp(pred[local, 0] + margin_log))
        records.append(
            {
                "case_id": rows[idx]["case_id"],
                "doe_role": role_list[idx],
                "actual_C_fibre": actual,
                "mlp_C_fibre_raw": raw,
                "mlp_C_fibre_safe": safe,
                "log_residual_actual_minus_pred": float(y[idx, 0] - pred[local, 0]),
                "conservative_raw": raw >= actual,
                "conservative_safe": safe >= actual,
            }
        )

    err = y[eval_idx, 0] - pred[:, 0]
    metrics: dict[str, Any] = {
        "official_role_validation_policy": "trained on DOE role=train only; evaluated on completed non-train DOE roles",
        "official_train_count": int(len(train_idx)),
        "official_eval_count": int(len(eval_idx)),
        "official_eval_rmse_log_C_raw": float(np.sqrt(np.mean(err * err))),
        "official_eval_raw_conservative_fraction": float(np.mean([r["conservative_raw"] for r in records])),
        "official_eval_safe_conservative_fraction": float(np.mean([r["conservative_safe"] for r in records])),
        "official_eval_training_loss_final": float(history[-1]),
    }
    for role in sorted({role_list[idx] for idx in eval_idx}):
        role_records = [record for record in records if record["doe_role"] == role]
        role_errors = np.array([record["log_residual_actual_minus_pred"] for record in role_records], dtype=float)
        metrics[f"official_{role}_count"] = int(len(role_records))
        metrics[f"official_{role}_rmse_log_C_raw"] = float(np.sqrt(np.mean(role_errors * role_errors)))
        metrics[f"official_{role}_raw_conservative_fraction"] = float(np.mean([r["conservative_raw"] for r in role_records]))
        metrics[f"official_{role}_safe_conservative_fraction"] = float(np.mean([r["conservative_safe"] for r in role_records]))
    return records, metrics


def plot_outputs(rows: list[dict[str, str]], final_rows: list[dict[str, Any]], cv_records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [r["case_id"] for r in rows]
    x = np.arange(len(rows))
    actual = np.array([r["actual_C_fibre"] for r in final_rows], dtype=float)
    raw = np.array([r["mlp_C_fibre_raw"] for r in final_rows], dtype=float)
    safe = np.array([r["mlp_C_fibre_safe"] for r in final_rows], dtype=float)
    py_fi = np.array([as_float(r, "python_fiber_stress_ratio_max") for r in rows], dtype=float)
    calc_fi = np.array([as_float(r, "calculix_fiber_stress_ratio_max") for r in rows], dtype=float)
    pred_calc_fi_safe = py_fi * safe

    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.plot(x, actual, color="#222222", marker="o", markersize=3, linewidth=1.0, label="C_fibre actual CalculiX/Python")
    ax.plot(x, raw, color="#0b6bcb", marker=".", linewidth=1.0, label="MLP raw")
    ax.plot(x, safe, color="#1f8a4c", marker=".", linewidth=1.0, label="MLP safe")
    ax.set_xticks(x[:: max(1, len(x) // 24)])
    ax.set_xticklabels([labels[i] for i in x[:: max(1, len(x) // 24)]], rotation=35, ha="right")
    ax.set_ylabel("Fibre correction factor")
    ax.set_title(f"Fibre-proxy MLP correction over {len(rows)} real CalculiX cases")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fiber_mlp_correction_factor_by_case.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    lim = max(float(np.max(actual)), float(np.max(safe)), 1.0) * 1.08
    ax.scatter(actual, raw, s=25, color="#0b6bcb", edgecolor="white", linewidth=0.35, label="raw")
    ax.scatter(actual, safe, s=25, color="#1f8a4c", edgecolor="white", linewidth=0.35, label="safe")
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="conservative boundary")
    ax.set_xlabel("Actual C_fibre")
    ax.set_ylabel("Predicted C_fibre")
    ax.set_title("MLP fibre correction: prediction vs actual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fiber_mlp_prediction_vs_actual.png", dpi=220)
    plt.close(fig)

    cv_raw = float(metrics["raw_conservative_fraction_cv"])
    cv_safe = float(np.mean([math.exp(math.log(r["mlp_C_fibre_raw"]) + metrics["cv_residual_margin_log"]) >= r["actual_C_fibre"] for r in cv_records]))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.bar(["Raw CV", "Safe CV"], [cv_raw, cv_safe], color=["#c44d2d", "#1f8a4c"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Conservative fraction")
    ax.set_title("Repeated K-fold conservatism")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fiber_mlp_cv_conservative_fraction.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    lim = max(float(np.max(calc_fi)), float(np.max(pred_calc_fi_safe)), 1.0) * 1.08
    ax.scatter(calc_fi, pred_calc_fi_safe, s=25, color="#1f8a4c", edgecolor="white", linewidth=0.35)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="conservative boundary")
    ax.set_xlabel("CalculiX fibre FI proxy")
    ax.set_ylabel("Python FI proxy x MLP safe C")
    ax.set_title("Corrected Python fibre FI vs CalculiX")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fiber_mlp_corrected_fi_vs_calculix.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rows = load_dataset()
    if len(rows) < 20:
        raise SystemExit("Need at least 20 real fibre-proxy rows to train a useful MLP.")
    roles = load_case_roles()
    attach_case_roles(rows, roles)
    categories = categorical_values(rows)
    raw_x, feature_names = raw_features(rows, categories)
    y = target_log_c(rows)

    train_scope_idx = np.array([idx for idx, row in enumerate(rows) if row.get("doe_role") == "train"], dtype=int)
    non_train_count = sum(1 for row in rows if row.get("doe_role") != "train")
    if len(train_scope_idx) >= 20 and non_train_count > 0:
        cv_idx = train_scope_idx
        cv_scope = "official DOE role=train rows only"
    else:
        cv_idx = np.arange(len(rows), dtype=int)
        cv_scope = "all completed rows"
    cv_rows = [rows[int(idx)] for idx in cv_idx]
    cv_records, cv_residuals, cv_metrics = cross_validate(raw_x[cv_idx], y[cv_idx], cv_rows)
    cv_metrics["cv_training_scope"] = cv_scope
    cv_metrics["cv_case_count"] = int(len(cv_idx))
    margin_log = float(cv_metrics["cv_residual_margin_log"])
    margin_by_policy = {
        "raw": float(cv_metrics["cv_residual_margin_log_raw"]),
        "p90": float(cv_metrics["cv_residual_margin_log_p90"]),
        "p95": float(cv_metrics["cv_residual_margin_log_p95"]),
        "p99": float(cv_metrics["cv_residual_margin_log_p99"]),
        "max": float(cv_metrics["cv_residual_margin_log_max"]),
    }
    audit_records, audit_metrics = audit_holdout(raw_x[cv_idx], y[cv_idx], cv_rows, margin_log)
    official_records, official_metrics = official_role_validation(raw_x, y, rows, margin_log)

    x_all, mean, std = standardize_fit(raw_x)
    final_model, history = train_mlp(x_all, y, seed=20260602, epochs=MLP_EPOCHS, learning_rate=MLP_LEARNING_RATE)
    final_log_raw = predict_log_c(final_model, x_all)[:, 0]
    final_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        actual = as_float(row, "fiber_correction_ratio_calculix_over_python")
        raw = float(math.exp(final_log_raw[idx]))
        p90 = float(math.exp(final_log_raw[idx] + margin_by_policy["p90"]))
        p95 = float(math.exp(final_log_raw[idx] + margin_by_policy["p95"]))
        p99 = float(math.exp(final_log_raw[idx] + margin_by_policy["p99"]))
        safe = float(math.exp(final_log_raw[idx] + margin_log))
        py_fi = as_float(row, "python_fiber_stress_ratio_max")
        final_rows.append(
            {
                "case_id": row["case_id"],
                "doe_role": row.get("doe_role", "unknown"),
                "python_fiber_stress_ratio_max": py_fi,
                "calculix_fiber_stress_ratio_max": as_float(row, "calculix_fiber_stress_ratio_max"),
                "actual_C_fibre": actual,
                "mlp_C_fibre_raw": raw,
                "mlp_C_fibre_p90": p90,
                "mlp_C_fibre_p95": p95,
                "mlp_C_fibre_p99": p99,
                "mlp_C_fibre_safe": safe,
                "python_fiber_fi_corrected_raw": py_fi * raw,
                "python_fiber_fi_corrected_p95": py_fi * p95,
                "python_fiber_fi_corrected_safe": py_fi * safe,
                "conservative_raw": raw >= actual,
                "conservative_p95": p95 >= actual,
                "conservative_safe": safe >= actual,
            }
        )

    final_actual_log = y[:, 0]
    final_err = final_actual_log - final_log_raw
    final_safe = np.array([r["mlp_C_fibre_safe"] for r in final_rows], dtype=float)
    actual_c = np.array([r["actual_C_fibre"] for r in final_rows], dtype=float)
    cv_safe_fraction = conservative_fraction(cv_records, margin_log)
    margin_policy_records: list[dict[str, Any]] = []
    for policy, policy_margin in margin_by_policy.items():
        pred_final = np.exp(final_log_raw + policy_margin)
        margin_policy_records.append(
            {
                "policy": policy,
                "margin_log": policy_margin,
                "margin_factor": float(math.exp(policy_margin)),
                "cv_conservative_fraction": conservative_fraction(cv_records, policy_margin),
                "official_eval_conservative_fraction": conservative_fraction(official_records, policy_margin)
                if official_records
                else "",
                "final_train_conservative_fraction": float(np.mean(pred_final >= actual_c)),
                "predicted_C_min": float(np.min(pred_final)),
                "predicted_C_median": float(np.median(pred_final)),
                "predicted_C_max": float(np.max(pred_final)),
                "median_predicted_over_actual": float(np.median(pred_final / actual_c)),
            }
        )
    metrics: dict[str, Any] = {
        **cv_metrics,
        **audit_metrics,
        **official_metrics,
        "case_count": len(rows),
        "target": "log(C_fibre), with C_fibre=CalculiX_fibre_ratio/Python_fibre_ratio",
        "final_model_training_scope": "all completed real CalculiX rows after official role validation",
        "final_train_rmse_log_C_raw": float(np.sqrt(np.mean(final_err * final_err))),
        "final_train_raw_conservative_fraction": float(np.mean([r["conservative_raw"] for r in final_rows])),
        "final_train_safe_conservative_fraction": float(np.mean([r["conservative_safe"] for r in final_rows])),
        "cv_safe_conservative_fraction": cv_safe_fraction,
        "C_actual_min": float(np.min(actual_c)),
        "C_actual_median": float(np.median(actual_c)),
        "C_actual_max": float(np.max(actual_c)),
        "C_safe_min": float(np.min(final_safe)),
        "C_safe_median": float(np.median(final_safe)),
        "C_safe_max": float(np.max(final_safe)),
        "margin_policy_note": "raw is diagnostic only; p95 is the recommended optimization-search mode; max/safe is the final conservative screening mode.",
        "recommended_ga_margin_policy_search": "p95",
        "recommended_ga_margin_policy_final_screening": "max",
        "warning": "Use this as a conservative pre-design correction for the fibre proxy only; it is not a certification burst model.",
    }

    write_csv(COMPARISON / "fiber_mlp_cross_validation.csv", cv_records)
    write_csv(COMPARISON / "fiber_mlp_audit_holdout.csv", audit_records)
    write_csv(COMPARISON / "fiber_mlp_official_role_validation.csv", official_records)
    write_csv(COMPARISON / "fiber_mlp_predictions.csv", final_rows)
    write_csv(COMPARISON / "fiber_mlp_margin_policy_comparison.csv", margin_policy_records)
    write_json(COMPARISON / "fiber_mlp_validation_metrics.json", metrics)

    model_payload = {
        "model_type": "tanh_mlp_fiber_proxy_correction",
        "learning_rate": MLP_LEARNING_RATE,
        "epochs": MLP_EPOCHS,
        "target": "log_C_fibre",
        "rule": "C_fibre_safe = exp(MLP(features) + cv_residual_margin_log)",
        "corrected_fi_rule": "FI_fibre_python_corrected = FI_fibre_python * C_fibre_safe",
        "feature_names": feature_names,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "categorical_values": categories,
        "weights": {key: value.tolist() for key, value in final_model.items()},
        "cv_residual_margin_log": margin_log,
        "cv_residual_margins_log": margin_by_policy,
        "recommended_margin_policy_search": "p95",
        "recommended_margin_policy_final_screening": "max",
        "cv_residuals_actual_minus_pred_log": cv_residuals.tolist(),
        "training_loss_final": float(history[-1]),
        "metrics": metrics,
    }
    write_json(MODEL_JSON, model_payload)
    plot_outputs(rows, final_rows, cv_records, metrics)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    policies = [r["policy"] for r in margin_policy_records]
    x = np.arange(len(policies))
    ax1.bar(x, [r["predicted_C_median"] for r in margin_policy_records], color="#6a9fb5", label="median predicted C")
    ax1.set_ylabel("Median predicted C_fibre")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        [r["cv_conservative_fraction"] for r in margin_policy_records],
        color="#1f8a4c",
        marker="o",
        linewidth=1.8,
        label="CV conservative fraction",
    )
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("CV conservative fraction")
    ax1.set_title("Fibre MLP margin policy trade-off")
    fig.tight_layout()
    fig.savefig(FIGURES / "fiber_mlp_margin_policy_comparison.png", dpi=220)
    plt.close(fig)
    print(MODEL_JSON)


if __name__ == "__main__":
    main()
