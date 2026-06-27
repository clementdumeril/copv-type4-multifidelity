"""External conservative correction layer for the fast GA evaluator.

This module deliberately sits outside :mod:`computation`.  The analytical
Python solver remains unchanged; the correction is applied only after raw
failure indices have been computed.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
TYPE4_ROOT = ROOT / "type4_calibration"
DEFAULT_OUTPUTS = TYPE4_ROOT / "outputs_11l_single_boss"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    return _as_float(os.environ.get(name), default)


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    return default if abs(den) < 1e-12 else num / den


class ConservativeCorrection:
    """Predict a conservative ``C_safe`` for a GA individual.

    Modes:
    - ``log_linear``: use the interpretable conservative regression.
    - ``mlp``: use the MLP plus conformal/CV margins.
    - ``max``: take the maximum of log-linear, MLP and zone guardrails.
    """

    def __init__(
        self,
        outputs_dir: str | Path | None = None,
        mode: str = "max",
        *,
        domain_policy: str = "warn",
    ) -> None:
        self.outputs_dir = Path(outputs_dir) if outputs_dir else Path(
            os.environ.get("TYPE4_GA_CORRECTION_OUTPUTS_DIR", DEFAULT_OUTPUTS)
        )
        self.mode = mode.lower()
        self.domain_policy = domain_policy.lower()
        self.models_dir = self.outputs_dir / "correction_models"
        self.comparison_dir = self.outputs_dir / "comparison_tables"
        self.log_model = self._load_optional(self.models_dir / "correction_model.json")
        self.mlp_model = self._load_optional(self.models_dir / "mlp_correction_model.json")
        self.fiber_model = self._load_optional(self.models_dir / "fiber_proxy_mlp_model.json")
        self.domain = self._load_optional(self.comparison_dir / "validation_metrics.json").get(
            "domain_of_validity", {}
        )
        self.fiber_domain = self._fiber_domain_from_dataset()
        if not self.log_model and not self.mlp_model and not self.fiber_model:
            raise FileNotFoundError(
                f"No correction model found in {self.models_dir}. "
                "Run the DOE calibration pipeline first."
            )

    @staticmethod
    def _load_optional(path: Path) -> dict[str, Any]:
        return _read_json(path) if path.exists() else {}

    @property
    def c_global(self) -> float:
        if self._is_fiber_mode() and self.fiber_model:
            metrics = self.fiber_model.get("metrics", {})
            return max(
                1.0,
                _as_float(metrics.get("C_safe_max"), _as_float(metrics.get("C_actual_max"), 1.0)),
            )
        if self.log_model:
            return max(1.0, _as_float(self.log_model.get("C_global"), 1.0))
        if self.mlp_model:
            return max(
                1.0,
                _as_float(self.mlp_model.get("C_global_by_target", {}).get("combined"), 1.0),
            )
        return 1.0

    def _is_fiber_mode(self) -> bool:
        return self.mode in {
            "fiber_proxy",
            "fibre_proxy",
            "fiber",
            "fibre",
            "fiber_proxy_raw",
            "fibre_proxy_raw",
            "fiber_proxy_p90",
            "fibre_proxy_p90",
            "fiber_proxy_p95",
            "fibre_proxy_p95",
            "fiber_proxy_p99",
            "fibre_proxy_p99",
            "fiber_proxy_max",
            "fibre_proxy_max",
            "fiber_proxy_safe",
            "fibre_proxy_safe",
        }

    def _fiber_margin_policy(self) -> str:
        env_policy = os.environ.get("TYPE4_GA_FIBER_MARGIN_POLICY") or os.environ.get(
            "TYPE4_GA_FIBER_MARGIN_MODE"
        )
        if env_policy:
            policy = env_policy.lower()
        elif self.mode.endswith("_raw"):
            policy = "raw"
        elif self.mode.endswith("_p90"):
            policy = "p90"
        elif self.mode.endswith("_p95"):
            policy = "p95"
        elif self.mode.endswith("_p99"):
            policy = "p99"
        elif self.mode.endswith("_max") or self.mode.endswith("_safe"):
            policy = "max"
        else:
            policy = "max"
        if policy == "safe":
            policy = "max"
        if policy not in {"raw", "p90", "p95", "p99", "max"}:
            policy = "max"
        return policy

    def _fiber_margin_log(self, policy: str | None = None) -> float:
        if not self.fiber_model:
            return 0.0
        resolved = policy or self._fiber_margin_policy()
        if resolved == "raw":
            return 0.0
        margins = self.fiber_model.get("cv_residual_margins_log", {})
        if isinstance(margins, dict) and resolved in margins:
            return _as_float(margins[resolved], 0.0)
        if resolved == "max":
            return _as_float(self.fiber_model.get("cv_residual_margin_log"), 0.0)
        return _as_float(self.fiber_model.get("cv_residual_margin_log"), 0.0)

    def _fiber_feature_vector(self, row: dict[str, Any]) -> np.ndarray:
        if not self.fiber_model:
            return np.array([], dtype=float)
        inner_radius = max(_as_float(row.get("inner_radius_mm")), 1e-9)
        total_thickness = max(_as_float(row.get("total_thickness_mm")), 1e-9)
        helical_pairs = _as_float(row.get("helical_pairs"))
        transition_pairs = _as_float(row.get("transition_pairs"))
        hoop_pairs = _as_float(row.get("hoop_pairs"))
        pair_total = max(helical_pairs + transition_pairs + hoop_pairs, 1e-9)
        derived = {
            "boss_radius_ratio": _as_float(row.get("boss_radius_mm")) / inner_radius,
            "length_radius_ratio": _as_float(row.get("cylindrical_length_mm")) / inner_radius,
            "pressure_radius_over_thickness": _as_float(row.get("pressure_mpa")) * inner_radius / total_thickness,
            "hoop_pair_fraction": hoop_pairs / pair_total,
            "transition_pair_fraction": transition_pairs / pair_total,
            "helical_hoop_angle_gap_deg": abs(
                _as_float(row.get("hoop_angle_deg")) - _as_float(row.get("helical_angle_min_deg"))
            ),
        }
        values: list[float] = []
        for name in self.fiber_model.get("feature_names", []):
            if "=" in name:
                key, expected = name.split("=", 1)
                values.append(1.0 if str(row.get(key, "") or "unknown") == expected else 0.0)
            elif name in derived:
                values.append(float(derived[name]))
            else:
                values.append(_as_float(row.get(name)))
        return np.array(values, dtype=float)

    def _fiber_domain_from_dataset(self) -> dict[str, dict[str, float]]:
        dataset = self.comparison_dir / "fiber_proxy_dataset.csv"
        if not self.fiber_model or not dataset.exists():
            return {}
        import csv

        rows: list[dict[str, Any]] = []
        with dataset.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return {}
        matrix = np.array([self._fiber_feature_vector(row) for row in rows], dtype=float)
        domain: dict[str, dict[str, float]] = {}
        for idx, name in enumerate(self.fiber_model.get("feature_names", [])):
            if "=" in name:
                continue
            vals = matrix[:, idx]
            domain[name] = {"min": float(np.min(vals)), "max": float(np.max(vals))}
        return domain

    def row_from_individual(self, individual: Any, metrics: dict[str, float]) -> dict[str, Any]:
        layers = list(getattr(individual.tank, "layers", []))
        angles = [abs(float(layer.angle)) for layer in layers]
        helical = [a for a in angles if a < 30.0]
        transition = [a for a in angles if 30.0 <= a < 80.0]
        hoop = [a for a in angles if a >= 80.0]
        inner_radius = float(individual.tank.internal_radius)
        total_thickness = float(individual.tank.get_total_thickness())
        pressure = float(individual.tank.burst_test_pressure)

        boss_ratio_default = _env_float("TYPE4_GA_BOSS_RADIUS_RATIO", 0.32)
        boss_radius = _env_float("TYPE4_GA_BOSS_RADIUS_MM", boss_ratio_default * inner_radius)

        # The GA cylinder solver does not carry dome-manufacturing metadata.
        # These defaults define the 11 L correction domain used for screening.
        dome_factor = _env_float("TYPE4_GA_DOME_RADIUS_FACTOR", 1.08)
        cyl_length = _env_float("TYPE4_GA_CYLINDRICAL_LENGTH_MM", float(individual.tank.length))
        dome_shape = os.environ.get("TYPE4_GA_DOME_SHAPE", "isotensoid_like")
        tow_width = _env_float("TYPE4_GA_TOW_WIDTH_MM", 6.175)
        tow_thickness = _env_float("TYPE4_GA_TOW_THICKNESS_MM", 0.125)
        overlap = _env_float("TYPE4_GA_OVERLAP_FACTOR", 1.0)

        non_hoop = helical + transition
        helical_min = min(helical or non_hoop or [0.0])
        helical_max = max(helical or non_hoop or [0.0])
        transition_max = max(transition or [helical_max])
        hoop_angle = float(np.mean(hoop)) if hoop else 90.0

        return {
            "pressure_mpa": pressure,
            "inner_radius_mm": inner_radius,
            "cylindrical_length_mm": cyl_length,
            "boss_radius_mm": boss_radius,
            "dome_radius_factor": dome_factor,
            "dome_shape": dome_shape,
            "helical_angle_min_deg": helical_min,
            "helical_angle_max_deg": helical_max,
            "transition_angle_max_deg": transition_max,
            "hoop_angle_deg": hoop_angle,
            "helical_pairs": len(helical) / 2.0,
            "transition_pairs": len(transition) / 2.0,
            "hoop_pairs": len(hoop) / 2.0,
            "tow_width_mm": tow_width,
            "tow_thickness_mm": tow_thickness,
            "overlap_factor": overlap,
            "total_thickness_mm": total_thickness,
            "layer_count": len(layers),
            "python_max_hashin": metrics.get("max_hashin", 0.0),
            "python_max_tsai_wu": metrics.get("max_tsai_wu", 0.0),
            "python_max_puck": metrics.get("max_puck", 0.0),
            "python_max_combined": metrics.get("max_combined", 0.0),
            "python_max_radial_displacement_mm": metrics.get("max_radial_displacement_mm", 0.0),
            "python_fiber_stress_ratio_max": metrics.get("max_fiber_stress_ratio", 0.0),
            "python_fiber_proxy_pressure_mpa": metrics.get("fiber_proxy_pressure_mpa", 0.0),
            "python_raw_max_combined": metrics.get("max_combined", 0.0),
        }

    def domain_violations(self, row: dict[str, Any]) -> list[str]:
        violations: list[str] = []
        for key, bounds in self.domain.items():
            if key not in row:
                continue
            value = _as_float(row[key], math.nan)
            lo = _as_float(bounds.get("min"), -math.inf)
            hi = _as_float(bounds.get("max"), math.inf)
            if math.isfinite(value) and (value < lo or value > hi):
                violations.append(f"{key}={value:.6g} outside [{lo:.6g}, {hi:.6g}]")
        return violations

    def predict_log_linear(self, row: dict[str, Any]) -> float:
        if not self.log_model:
            return 1.0
        vals = np.array([_as_float(row.get(key)) for key in self.log_model["feature_names"]])
        mean = np.array(self.log_model["feature_mean"], dtype=float)
        std = np.array(self.log_model["feature_std"], dtype=float)
        std[std < 1e-12] = 1.0
        beta = np.array(self.log_model["beta"], dtype=float)
        x = np.concatenate([[1.0], (vals - mean) / std])
        return float(max(1.0, math.exp(float(x @ beta) + float(self.log_model["residual_margin_log"]))))

    def predict_zone_guard(self, row: dict[str, Any]) -> float:
        if not self.log_model:
            return 1.0
        zones = self.log_model.get("C_by_zone", {})
        # In the fast cylinder model, the missing physics is mainly dome/junction.
        # Use the dome guardrail as the default lower bound; switch to boss/global
        # when the assumed boss ratio is outside the calibration envelope.
        inner = max(_as_float(row.get("inner_radius_mm")), 1e-9)
        boss_ratio = _safe_ratio(_as_float(row.get("boss_radius_mm")), inner, 0.32)
        if boss_ratio < 0.28 or boss_ratio > 0.36:
            return max(_as_float(zones.get("boss"), self.c_global), self.c_global)
        return max(1.0, _as_float(zones.get("dome"), self.c_global))

    def _mlp_raw_features(self, row: dict[str, Any]) -> np.ndarray:
        model = self.mlp_model
        values = [_as_float(row.get(name)) for name in model["numeric_features"]]
        inner_radius = max(_as_float(row.get("inner_radius_mm")), 1e-9)
        total_thickness = max(_as_float(row.get("total_thickness_mm")), 1e-9)
        values.extend(
            [
                _as_float(row.get("boss_radius_mm")) / inner_radius,
                _as_float(row.get("cylindrical_length_mm")) / inner_radius,
                _as_float(row.get("pressure_mpa")) * inner_radius / total_thickness,
            ]
        )
        for name in model["categorical_features"]:
            observed = row.get(name, "") or "unknown"
            values.extend([1.0 if observed == value else 0.0 for value in model["categorical_values"][name]])
        return np.array(values, dtype=float)

    @staticmethod
    def _mlp_forward(weights: dict[str, Any], x: np.ndarray) -> np.ndarray:
        w1 = np.array(weights["W1"], dtype=float)
        b1 = np.array(weights["b1"], dtype=float)
        w2 = np.array(weights["W2"], dtype=float)
        b2 = np.array(weights["b2"], dtype=float)
        w3 = np.array(weights["W3"], dtype=float)
        b3 = np.array(weights["b3"], dtype=float)
        a1 = np.tanh(x @ w1 + b1)
        a2 = np.tanh(a1 @ w2 + b2)
        return a2 @ w3 + b3

    def predict_mlp(self, row: dict[str, Any]) -> float:
        if not self.mlp_model:
            return 1.0
        raw = self._mlp_raw_features(row)
        mean = np.array(self.mlp_model["feature_mean"], dtype=float)
        std = np.array(self.mlp_model["feature_std"], dtype=float)
        std[std < 1e-12] = 1.0
        x = (raw - mean) / std
        log_ratio = self._mlp_forward(self.mlp_model["weights"], x)
        total_margin = np.array(self.mlp_model.get("total_margin_log", []), dtype=float)
        if total_margin.size == 0:
            residual = np.array(self.mlp_model.get("residual_margin_log", []), dtype=float)
            cv = np.array(self.mlp_model.get("cv_extra_margin_log", []), dtype=float)
            total_margin = residual + cv
        target_names = list(self.mlp_model["target_names"])
        combined_idx = target_names.index("combined")
        return float(max(1.0, math.exp(float(log_ratio[combined_idx] + total_margin[combined_idx]))))

    def predict_fiber_proxy_mlp(self, row: dict[str, Any], margin_policy: str | None = None) -> float:
        if not self.fiber_model:
            return 1.0
        raw = self._fiber_feature_vector(row)
        mean = np.array(self.fiber_model["feature_mean"], dtype=float)
        std = np.array(self.fiber_model["feature_std"], dtype=float)
        std[std < 1e-12] = 1.0
        x = (raw - mean) / std
        log_c = self._mlp_forward(self.fiber_model["weights"], x)
        margin = self._fiber_margin_log(margin_policy)
        return float(max(1.0, math.exp(float(log_c[0] + margin))))

    def fiber_domain_violations(self, row: dict[str, Any]) -> list[str]:
        violations: list[str] = []
        if not self.fiber_domain:
            return violations
        features = dict(zip(self.fiber_model.get("feature_names", []), self._fiber_feature_vector(row)))
        for key, bounds in self.fiber_domain.items():
            if key not in features:
                continue
            value = float(features[key])
            lo = bounds["min"]
            hi = bounds["max"]
            if value < lo or value > hi:
                violations.append(f"{key}={value:.6g} outside [{lo:.6g}, {hi:.6g}]")
        return violations

    def evaluate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if self._is_fiber_mode():
            margin_policy = self._fiber_margin_policy()
            margin_log = self._fiber_margin_log(margin_policy)
            c_fiber = self.predict_fiber_proxy_mlp(row, margin_policy)
            violations = self.fiber_domain_violations(row)
            c_safe = c_fiber
            if violations and self.domain_policy in {"global", "penalize"}:
                c_safe = max(c_safe, self.c_global)
            return {
                "C_safe": float(max(1.0, c_safe)),
                "C_log_linear": 1.0,
                "C_mlp": float(c_fiber),
                "C_zone_guard": 1.0,
                "C_global": float(self.c_global),
                "C_fiber_mlp": float(c_fiber),
                "fiber_margin_policy": margin_policy,
                "fiber_margin_log": float(margin_log),
                "domain_violations": violations,
                "mode": self.mode,
                "correction_target": "fiber_proxy",
            }
        c_log = self.predict_log_linear(row)
        c_zone = self.predict_zone_guard(row)
        c_mlp = self.predict_mlp(row)
        if self.mode == "log_linear":
            c_safe = max(c_log, c_zone)
        elif self.mode == "mlp":
            c_safe = max(c_mlp, c_zone)
        elif self.mode == "zone":
            c_safe = c_zone
        elif self.mode == "global":
            c_safe = self.c_global
        else:
            c_safe = max(c_log, c_mlp, c_zone)
        violations = self.domain_violations(row)
        if violations and self.domain_policy in {"global", "penalize"}:
            c_safe = max(c_safe, self.c_global)
        return {
            "C_safe": float(max(1.0, c_safe)),
            "C_log_linear": float(c_log),
            "C_mlp": float(c_mlp),
            "C_zone_guard": float(c_zone),
            "C_global": float(self.c_global),
            "domain_violations": violations,
            "mode": self.mode,
        }

    def evaluate_individual(self, individual: Any, metrics: dict[str, float]) -> dict[str, Any]:
        row = self.row_from_individual(individual, metrics)
        result = self.evaluate_row(row)
        result["features"] = row
        return result


def correction_from_environment() -> ConservativeCorrection | None:
    enabled = os.environ.get("TYPE4_USE_CALCULIX_CORRECTION", "0").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    mode = os.environ.get("TYPE4_GA_CORRECTION_MODE", "max")
    policy = os.environ.get("TYPE4_GA_CORRECTION_DOMAIN_POLICY", "warn")
    return ConservativeCorrection(mode=mode, domain_policy=policy)
