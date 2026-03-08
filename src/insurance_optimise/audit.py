"""
Audit trail generation for regulatory compliance.

Every optimisation run produces a JSON-serialisable dict capturing:
- Input summary (segment counts, premium totals, constraint config)
- Solver settings (method, tolerances, restarts)
- Solution summary (multiplier statistics, objective values)
- Constraint evaluation at solution (is each constraint binding?)
- Shadow prices (Lagrange multipliers)
- Convergence metadata (status, iterations, function evaluations)

The audit trail is designed to satisfy FCA Consumer Duty auditability
requirements. Pricing teams can attach it as evidence that the ENBP
constraint was enforced and that the optimisation methodology is documented.

Nothing in this module makes network calls or writes files — that is left
to the caller (result.save_audit).
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np


def build_audit_trail(
    *,
    n_policies: int,
    n_renewal: int,
    technical_price: np.ndarray,
    expected_loss_cost: np.ndarray,
    enbp: np.ndarray | None,
    prior_multiplier: np.ndarray,
    constraint_config_dict: dict[str, Any],
    demand_model_name: str,
    solver: str,
    solver_options: dict[str, Any],
    n_restarts: int,
    x0_strategy: str,
    # Solution
    multipliers: np.ndarray,
    converged: bool,
    solver_message: str,
    n_iter: int,
    n_fun_eval: int,
    # Portfolio metrics
    expected_profit: float,
    expected_gwp: float,
    expected_lr: float,
    expected_retention: float | None,
    # Constraint evaluation
    constraint_values: dict[str, float],
    shadow_prices: dict[str, float],
) -> dict[str, Any]:
    """
    Build a complete audit trail dict for one optimisation run.

    Returns
    -------
    dict
        JSON-serialisable audit record.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Input summary statistics
    input_summary = {
        "n_policies": int(n_policies),
        "n_renewal": int(n_renewal),
        "n_new_business": int(n_policies - n_renewal),
        "total_technical_premium": float(np.sum(technical_price)),
        "mean_technical_premium": float(np.mean(technical_price)),
        "total_expected_loss_cost": float(np.sum(expected_loss_cost)),
        "technical_loss_ratio": float(
            np.sum(expected_loss_cost) / max(np.sum(technical_price), 1e-10)
        ),
        "has_enbp": enbp is not None,
        "prior_multiplier_mean": float(np.mean(prior_multiplier)),
    }

    if enbp is not None:
        # Report ENBP headroom (how much room below ENBP does current pricing have)
        tc_arr = np.asarray(technical_price, dtype=float)
        enbp_arr = np.asarray(enbp, dtype=float)
        enbp_multiplier = enbp_arr / np.maximum(tc_arr, 1e-10)
        input_summary["enbp_multiplier_mean"] = float(np.mean(enbp_multiplier))
        input_summary["enbp_multiplier_min"] = float(np.min(enbp_multiplier))

    # Solution summary
    solution_summary = {
        "multiplier_mean": float(np.mean(multipliers)),
        "multiplier_p25": float(np.percentile(multipliers, 25)),
        "multiplier_median": float(np.median(multipliers)),
        "multiplier_p75": float(np.percentile(multipliers, 75)),
        "multiplier_min": float(np.min(multipliers)),
        "multiplier_max": float(np.max(multipliers)),
        "rate_change_mean_pct": float(
            np.mean((multipliers / np.maximum(prior_multiplier, 1e-10) - 1) * 100)
        ),
    }

    # Portfolio metrics
    portfolio_metrics = {
        "expected_profit": float(expected_profit),
        "expected_gwp": float(expected_gwp),
        "expected_loss_ratio": float(expected_lr),
    }
    if expected_retention is not None:
        portfolio_metrics["expected_retention"] = float(expected_retention)

    # Convergence
    convergence = {
        "converged": bool(converged),
        "solver_message": str(solver_message),
        "n_iterations": int(n_iter),
        "n_function_evaluations": int(n_fun_eval),
    }

    return {
        "library": "insurance-optimise",
        "version": "0.1.0",
        "timestamp_utc": ts,
        "inputs": input_summary,
        "constraints": constraint_config_dict,
        "solver": {
            "method": solver,
            "options": solver_options,
            "n_restarts": int(n_restarts),
            "x0_strategy": str(x0_strategy),
        },
        "demand_model": str(demand_model_name),
        "solution": solution_summary,
        "portfolio_metrics": portfolio_metrics,
        "constraint_evaluation": {
            k: float(v) for k, v in constraint_values.items()
        },
        "shadow_prices": {k: float(v) for k, v in shadow_prices.items()},
        "convergence": convergence,
    }


def extract_shadow_prices(
    scipy_result: Any,
    constraint_names: list[str],
) -> dict[str, float]:
    """
    Extract shadow prices (Lagrange multipliers) from a scipy OptimizeResult.

    SLSQP reports constraint multipliers in result.v — a list of arrays.
    The first element of each array is the Lagrange multiplier for that
    constraint. Positive value means the constraint is binding.

    Parameters
    ----------
    scipy_result:
        Return value from scipy.optimize.minimize.
    constraint_names:
        Names corresponding to the constraints list passed to minimize.
        Must have same length as the constraints list.

    Returns
    -------
    dict mapping constraint name to its Lagrange multiplier.
    """
    shadow = {}
    # scipy SLSQP stores multipliers in result.v (undocumented but consistent)
    v = getattr(scipy_result, "v", None)
    if v is not None and len(v) > 0:
        for i, name in enumerate(constraint_names):
            if i < len(v):
                val = v[i]
                # v[i] is array; take first element for ineq constraints
                if hasattr(val, "__len__") and len(val) > 0:
                    shadow[name] = float(val[0])
                else:
                    shadow[name] = float(val)
    # Fallback: report zeros if SLSQP didn't populate .v
    for name in constraint_names:
        if name not in shadow:
            shadow[name] = 0.0
    return shadow


def evaluate_constraints(
    m: np.ndarray,
    constraints: list[dict],
    constraint_names: list[str],
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """
    Evaluate each constraint at solution m and return constraint values.

    For inequality constraints (type='ineq'), positive value means satisfied.
    We store the raw value so the audit trail shows whether constraints are
    binding (near zero) or slack (positive).

    Returns
    -------
    dict mapping constraint name to constraint function value at m.
    """
    values = {}
    for name, con in zip(constraint_names, constraints):
        try:
            val = float(con["fun"](m))
        except Exception:
            val = float("nan")
        values[name] = val
    return values
