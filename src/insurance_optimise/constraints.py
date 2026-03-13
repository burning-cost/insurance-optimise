"""
Constraint definitions for insurance portfolio rate optimisation.

Constraints are categorised by tier:

Tier 1 — Hard regulatory (FCA):
  - ENBP: renewal premiums must not exceed new business equivalent price
  - Technical floor: premium >= cost (prevents actuarially unsound pricing)

Tier 2 — Portfolio-level business:
  - Loss ratio: aggregate LR within bounds
  - GWP: total written premium within bounds
  - Retention: renewal retention rate floor

Tier 3 — Individual segment guardrails:
  - Max rate change: cap per-policy year-on-year movement

All constraints are expressed in multiplier space (m_i = p_i / tc_i).
Bounds constraints (ENBP, rate change, technical floor) are folded into
scipy.optimize.Bounds — they do not appear as constraint functions.
Non-bound constraints (LR, GWP, retention) are expressed as scipy-compatible
dicts with 'type', 'fun', and 'jac' keys.

Design note: analytical gradients are provided for all constraints. Without
them SLSQP uses finite differences, which is 2*N extra function evaluations
per iteration — prohibitively slow for N=10,000.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds

if TYPE_CHECKING:
    from insurance_optimise._demand_model import LogLinearDemand, LogisticDemand


@dataclass
class ConstraintConfig:
    """
    Configuration for all portfolio constraints.

    Parameters
    ----------
    lr_max:
        Maximum allowed aggregate loss ratio (e.g. 0.70). None = unconstrained.
    lr_min:
        Minimum allowed aggregate loss ratio (prevents pricing too aggressively).
        None = unconstrained.
    gwp_min:
        Minimum gross written premium in currency units. None = unconstrained.
    gwp_max:
        Maximum gross written premium. None = unconstrained.
    retention_min:
        Minimum expected retention rate for renewal policies (0-1).
        None = unconstrained.
    max_rate_change:
        Maximum allowed |multiplier - prior_multiplier| / prior_multiplier.
        E.g. 0.25 means ±25% rate change allowed. None = unconstrained.
    enbp_buffer:
        Safety margin below ENBP for FCA compliance. The hard upper bound is
        set to ENBP * (1 - enbp_buffer). Default 0.0 (tight to ENBP).
    technical_floor:
        If True, enforce m_i >= 1.0 (price >= technical_price). This prevents
        loss-leading. Default True.
    min_multiplier:
        Absolute lower bound on multipliers. Default 0.5 (50% of technical
        price). Applied in addition to rate-change and technical-floor bounds.
    max_multiplier:
        Absolute upper bound on multipliers. Default 3.0. Applied before ENBP.
    stochastic_lr:
        If True, use Branda (2014) stochastic LR constraint:
        E[LR] + z_alpha * sigma[LR] <= lr_max.
        Requires claims_variance in the input data.
    stochastic_alpha:
        Confidence level for stochastic LR constraint (e.g. 0.90).
        z_alpha = sqrt(alpha / (1 - alpha)) via one-sided Chebyshev.
    cvar_max:
        Maximum allowed CVaR (expected shortfall) on profit across scenarios.
        Only active when scenario mode is used. None = unconstrained.
    cvar_alpha:
        Tail probability for CVaR constraint (e.g. 0.10 = worst 10%).
    """

    lr_max: float | None = None
    lr_min: float | None = None
    gwp_min: float | None = None
    gwp_max: float | None = None
    retention_min: float | None = None
    max_rate_change: float | None = None
    enbp_buffer: float = 0.0
    technical_floor: bool = True
    min_multiplier: float = 0.5
    max_multiplier: float = 3.0
    stochastic_lr: bool = False
    stochastic_alpha: float = 0.90
    cvar_max: float | None = None
    cvar_alpha: float = 0.10

    def validate(self) -> None:
        """Raise ValueError if the configuration is internally inconsistent."""
        if self.lr_min is not None and self.lr_max is not None:
            if self.lr_min >= self.lr_max:
                raise ValueError(
                    f"lr_min ({self.lr_min}) must be < lr_max ({self.lr_max})"
                )
        if self.max_rate_change is not None and self.max_rate_change <= 0:
            raise ValueError("max_rate_change must be positive")
        if not 0 < self.enbp_buffer < 1:
            if self.enbp_buffer != 0.0:
                raise ValueError("enbp_buffer must be in [0, 1)")
        if self.retention_min is not None:
            if not 0 < self.retention_min < 1:
                raise ValueError("retention_min must be in (0, 1)")
        if self.stochastic_alpha <= 0 or self.stochastic_alpha >= 1:
            raise ValueError("stochastic_alpha must be in (0, 1)")


def build_bounds(
    config: ConstraintConfig,
    n: int,
    technical_price: np.ndarray,
    prior_multiplier: np.ndarray,
    enbp: np.ndarray | None,
    renewal_flag: np.ndarray | None,
) -> Bounds:
    """
    Build scipy Bounds from constraint config.

    Handles ENBP, technical floor, rate change, and absolute multiplier
    bounds. All are folded into scipy Bounds (box constraints) rather than
    expressed as constraint functions — this is more efficient for SLSQP.

    Parameters
    ----------
    config:
        Constraint configuration.
    n:
        Number of policies.
    technical_price:
        Technical price array, shape (N,).
    prior_multiplier:
        Prior year multiplier array, shape (N,). 1.0 if first year.
    enbp:
        ENBP array, shape (N,). None if no ENBP constraint.
    renewal_flag:
        Boolean array, shape (N,). True = renewal policy.

    Returns
    -------
    scipy.optimize.Bounds
    """
    lb = np.full(n, config.min_multiplier)
    ub = np.full(n, config.max_multiplier)

    # Technical floor: m >= 1.0 (price >= cost)
    if config.technical_floor:
        lb = np.maximum(lb, 1.0)

    # Rate change bounds
    if config.max_rate_change is not None:
        delta = config.max_rate_change
        lb = np.maximum(lb, prior_multiplier * (1.0 - delta))
        ub = np.minimum(ub, prior_multiplier * (1.0 + delta))

    # ENBP: renewal price <= ENBP * (1 - buffer)
    # Expressed as multiplier bound: m_i <= ENBP_i / tc_i * (1 - buffer)
    if enbp is not None and renewal_flag is not None:
        enbp_arr = np.asarray(enbp, dtype=float)
        tc_arr = np.asarray(technical_price, dtype=float)
        enbp_ub = enbp_arr / np.maximum(tc_arr, 1e-10) * (1.0 - config.enbp_buffer)
        renewal_mask = np.asarray(renewal_flag, dtype=bool)
        ub = np.where(renewal_mask, np.minimum(ub, enbp_ub), ub)

    # Ensure lb <= ub (infeasible if ENBP < technical floor)
    feasible = lb <= ub
    if not np.all(feasible):
        n_infeasible = np.sum(~feasible)
        import warnings
        warnings.warn(
            f"{n_infeasible} policies have lb > ub after applying constraints. "
            "ENBP may be below technical floor. Clipping lb to ub for these "
            "policies — their prices will be set at ENBP.",
            stacklevel=3,
        )
        lb = np.minimum(lb, ub)

    return Bounds(lb=lb, ub=ub)


def build_scipy_constraints(
    config: ConstraintConfig,
    technical_price: np.ndarray,
    expected_loss_cost: np.ndarray,
    renewal_flag: np.ndarray | None,
    demand_model: "LogLinearDemand | LogisticDemand",
    claims_variance: np.ndarray | None = None,
) -> list[dict]:
    """
    Build scipy constraint dicts for the non-bound constraints.

    Returns a list of dicts, each with 'type', 'fun', and 'jac'.
    Only active constraints (where the config value is set) are included.

    Parameters
    ----------
    config:
        Constraint configuration.
    technical_price:
        Technical price array (cost proxy), shape (N,).
    expected_loss_cost:
        Expected claims cost per policy, shape (N,). Used for LR numerator.
    renewal_flag:
        Boolean array, shape (N,). For retention constraint.
    demand_model:
        Fitted demand model instance.
    claims_variance:
        Per-policy claims variance, shape (N,). Required for stochastic LR.

    Returns
    -------
    list of scipy constraint dicts
    """
    tc = np.asarray(technical_price, dtype=float)
    cost = np.asarray(expected_loss_cost, dtype=float)
    constraints = []

    # --- Loss ratio constraints ---
    if config.lr_max is not None or config.lr_min is not None:
        if config.stochastic_lr and claims_variance is not None:
            var_c = np.asarray(claims_variance, dtype=float)
            z_alpha = np.sqrt(
                config.stochastic_alpha / (1.0 - config.stochastic_alpha)
            )
        else:
            var_c = None
            z_alpha = 0.0

        if config.lr_max is not None:
            lr_max = config.lr_max

            def _lr_upper_fun(m: np.ndarray) -> float:
                # ineq: LR_max - LR(m) >= 0
                x = demand_model.demand(m)
                p = m * tc
                gwp = np.dot(p, x)
                claims = np.dot(cost, x)
                lr = claims / max(gwp, 1e-10)
                if var_c is not None:
                    # Branda: E[LR] + z * sigma[LR] <= lr_max
                    # sigma[LR] = sqrt(sum(var_c * x^2)) / gwp
                    sigma_lr = np.sqrt(np.dot(var_c, x**2)) / max(gwp, 1e-10)
                    lr = lr + z_alpha * sigma_lr
                return lr_max - lr

            def _lr_upper_jac(m: np.ndarray) -> np.ndarray:
                x = demand_model.demand(m)
                dx = demand_model.demand_gradient(m)
                p = m * tc
                gwp = np.dot(p, x)
                claims = np.dot(cost, x)
                if gwp < 1e-10:
                    return np.zeros_like(m)
                # d(LR)/d(m_i) = [cost_i * dx_i * gwp - claims * (tc_i*x_i + p_i*dx_i)] / gwp^2
                d_claims = cost * dx
                d_gwp = tc * x + p * dx
                d_lr = (d_claims * gwp - claims * d_gwp) / gwp**2
                if var_c is not None and np.dot(var_c, x**2) > 0:
                    sigma_lr = np.sqrt(np.dot(var_c, x**2)) / gwp
                    # d(sigma)/d(m_i) = [var_c_i * x_i * dx_i / sigma_num - sigma_num/gwp * d_gwp_i] / gwp
                    sigma_num = np.sqrt(np.dot(var_c, x**2))
                    d_sigma_num = var_c * x * dx / sigma_num
                    d_sigma = (d_sigma_num * gwp - sigma_num * d_gwp) / gwp**2
                    d_lr = d_lr + z_alpha * d_sigma
                # constraint is lr_max - lr so jac = -d_lr
                return -d_lr

            constraints.append(
                {
                    "type": "ineq",
                    "fun": _lr_upper_fun,
                    "jac": _lr_upper_jac,
                }
            )

        if config.lr_min is not None:
            lr_min = config.lr_min

            def _lr_lower_fun(m: np.ndarray) -> float:
                # ineq: LR(m) - LR_min >= 0
                x = demand_model.demand(m)
                p = m * tc
                gwp = np.dot(p, x)
                claims = np.dot(cost, x)
                return claims / max(gwp, 1e-10) - lr_min

            def _lr_lower_jac(m: np.ndarray) -> np.ndarray:
                x = demand_model.demand(m)
                dx = demand_model.demand_gradient(m)
                p = m * tc
                gwp = np.dot(p, x)
                claims = np.dot(cost, x)
                if gwp < 1e-10:
                    return np.zeros_like(m)
                d_claims = cost * dx
                d_gwp = tc * x + p * dx
                return (d_claims * gwp - claims * d_gwp) / gwp**2

            constraints.append(
                {
                    "type": "ineq",
                    "fun": _lr_lower_fun,
                    "jac": _lr_lower_jac,
                }
            )

    # --- GWP constraints ---
    if config.gwp_min is not None:
        gwp_min = float(config.gwp_min)

        def _gwp_min_fun(m: np.ndarray) -> float:
            x = demand_model.demand(m)
            p = m * tc
            return np.dot(p, x) - gwp_min

        def _gwp_min_jac(m: np.ndarray) -> np.ndarray:
            x = demand_model.demand(m)
            dx = demand_model.demand_gradient(m)
            p = m * tc
            return tc * x + p * dx

        constraints.append(
            {"type": "ineq", "fun": _gwp_min_fun, "jac": _gwp_min_jac}
        )

    if config.gwp_max is not None:
        gwp_max = float(config.gwp_max)

        def _gwp_max_fun(m: np.ndarray) -> float:
            x = demand_model.demand(m)
            p = m * tc
            return gwp_max - np.dot(p, x)

        def _gwp_max_jac(m: np.ndarray) -> np.ndarray:
            x = demand_model.demand(m)
            dx = demand_model.demand_gradient(m)
            p = m * tc
            return -(tc * x + p * dx)

        constraints.append(
            {"type": "ineq", "fun": _gwp_max_fun, "jac": _gwp_max_jac}
        )

    # --- Retention constraint ---
    if config.retention_min is not None and renewal_flag is not None:
        renewal_mask = np.asarray(renewal_flag, dtype=bool)
        n_renewal = int(np.sum(renewal_mask))
        if n_renewal > 0:
            ret_min = config.retention_min

            def _ret_fun(m: np.ndarray) -> float:
                # ineq: sum(x[renewal]) / n_renewal - ret_min >= 0
                x = demand_model.demand(m)
                return np.sum(x[renewal_mask]) / n_renewal - ret_min

            def _ret_jac(m: np.ndarray) -> np.ndarray:
                dx = demand_model.demand_gradient(m)
                grad = np.zeros_like(m)
                grad[renewal_mask] = dx[renewal_mask] / n_renewal
                return grad

            constraints.append(
                {"type": "ineq", "fun": _ret_fun, "jac": _ret_jac}
            )

    return constraints
