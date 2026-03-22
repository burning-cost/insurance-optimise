"""
PortfolioOptimiser: the main entry point for insurance portfolio rate optimisation.

Solves:
    min_m  -f(m)
    s.t.   regulatory and business constraints
    where  f(m) = sum_i [(m_i * tc_i - cost_i) * x_i(m_i)]  (profit)

Decision variables are price multipliers m_i = p_i / tc_i, where tc_i is the
technical price. Operating in multiplier space rather than price space keeps
variables O(1) in magnitude and makes the ENBP bound symmetric. Operating in
direct multiplier space (rather than log space) is simpler and works well with
SLSQP's bound handling.

The solver is SLSQP (scipy.optimize.minimize). Analytical gradients are
computed for the objective and all constraints — this is 5-10x faster than
finite difference for large portfolios. The fall-back solver trust-constr
is available for verification.

Usage
-----
>>> from insurance_optimise import PortfolioOptimiser, ConstraintConfig
>>> config = ConstraintConfig(lr_max=0.70, retention_min=0.85, max_rate_change=0.20)
>>> opt = PortfolioOptimiser(
...     technical_price=df['tc'].to_numpy(),
...     expected_loss_cost=df['cost'].to_numpy(),
...     p_demand=df['p_renew'].to_numpy(),
...     elasticity=df['elasticity'].to_numpy(),
...     renewal_flag=df['is_renewal'].to_numpy(),
...     enbp=df['enbp'].to_numpy(),
...     constraints=config,
... )
>>> result = opt.optimise()
>>> print(result)
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize, Bounds

from insurance_optimise.audit import (
    build_audit_trail,
    evaluate_constraints,
    extract_shadow_prices,
)
from insurance_optimise.constraints import (
    ConstraintConfig,
    build_bounds,
    build_scipy_constraints,
)
from insurance_optimise._demand_model import make_demand_model
from insurance_optimise.result import OptimisationResult


class PortfolioOptimiser:
    """
    Constrained portfolio rate optimiser for UK personal lines insurance.

    Parameters
    ----------
    technical_price:
        Array of technical prices (expected cost + expense loading) per
        policy, shape (N,). This is the reference price; multipliers are
        applied to it to get the final premium.
    expected_loss_cost:
        Array of expected claims costs per policy, shape (N,). Should be
        <= technical_price (the difference being the expense loading).
    p_demand:
        Baseline demand probability array, shape (N,). For renewals: renewal
        probability at current price. For new business: conversion probability.
        Values in (0, 1).
    elasticity:
        Price elasticity array, shape (N,). For log_linear model: constant
        elasticity d(log x)/d(log p). Should be negative. Typical values:
        -0.5 to -3.0 for personal lines insurance.
    renewal_flag:
        Boolean array, shape (N,). True = renewal policy (ENBP applies).
        None = treat all as new business.
    enbp:
        Equivalent new business price array, shape (N,). FCA ENBP per
        PS21/11. Required for renewal policies when ENBP constraint is
        active. None = no ENBP constraint.
    prior_multiplier:
        Prior year multiplier array, shape (N,). Used for rate-change bounds.
        Default 1.0 for all policies (first-year equivalent).
    claims_variance:
        Per-policy claims variance array, shape (N,). Required only for
        stochastic LR constraint (Branda 2014 formulation).
    constraints:
        ConstraintConfig instance. Defaults to no constraints (unconstrained
        profit maximisation).
    demand_model:
        ``'log_linear'`` (default) or ``'logistic'``.
    solver:
        ``'slsqp'`` (default) or ``'trust_constr'``.
    n_restarts:
        Number of random restarts. Best solution across restarts is returned.
        Default 1 (no restart).
    seed:
        Random seed for reproducibility of restarts. Default 42.
    ftol:
        SLSQP function tolerance. Default 1e-9 (tighter than scipy default
        1e-6 to reduce premature convergence).
    maxiter:
        Maximum solver iterations. Default 1000.
    """

    def __init__(
        self,
        technical_price: np.ndarray,
        expected_loss_cost: np.ndarray,
        p_demand: np.ndarray,
        elasticity: np.ndarray,
        renewal_flag: np.ndarray | None = None,
        enbp: np.ndarray | None = None,
        prior_multiplier: np.ndarray | None = None,
        claims_variance: np.ndarray | None = None,
        constraints: ConstraintConfig | None = None,
        demand_model: str = "log_linear",
        solver: str = "slsqp",
        n_restarts: int = 1,
        seed: int = 42,
        ftol: float = 1e-9,
        maxiter: int = 1000,
    ) -> None:
        self.tc = np.asarray(technical_price, dtype=float)
        self.cost = np.asarray(expected_loss_cost, dtype=float)
        self.x0 = np.asarray(p_demand, dtype=float)
        self.elasticity = np.asarray(elasticity, dtype=float)
        self.n = len(self.tc)

        if renewal_flag is not None:
            self.renewal_flag = np.asarray(renewal_flag, dtype=bool)
        else:
            self.renewal_flag = np.zeros(self.n, dtype=bool)

        self.enbp = (
            np.asarray(enbp, dtype=float) if enbp is not None else None
        )
        self.prior_multiplier = (
            np.asarray(prior_multiplier, dtype=float)
            if prior_multiplier is not None
            else np.ones(self.n)
        )
        self.claims_variance = (
            np.asarray(claims_variance, dtype=float)
            if claims_variance is not None
            else None
        )
        self.config = constraints if constraints is not None else ConstraintConfig()
        self.demand_model_name = demand_model
        self.solver_method = "SLSQP" if solver.lower() == "slsqp" else "trust-constr"
        self.n_restarts = max(1, n_restarts)
        self.rng = np.random.default_rng(seed)
        self.ftol = ftol
        self.maxiter = maxiter

        # Validate
        self._validate_inputs()
        self.config.validate()

        # Build demand model
        self._demand = make_demand_model(
            demand_model, self.x0, self.elasticity, self.tc
        )

        # Build bounds (box constraints folded in)
        self._bounds = build_bounds(
            config=self.config,
            n=self.n,
            technical_price=self.tc,
            prior_multiplier=self.prior_multiplier,
            enbp=self.enbp,
            renewal_flag=self.renewal_flag,
        )

        # Build scipy constraint list
        self._scipy_constraints = build_scipy_constraints(
            config=self.config,
            technical_price=self.tc,
            expected_loss_cost=self.cost,
            renewal_flag=self.renewal_flag,
            demand_model=self._demand,
            claims_variance=self.claims_variance,
        )

        # Constraint names (for audit trail)
        self._constraint_names = self._build_constraint_names()

    # ------------------------------------------------------------------
    # Objective function and gradient
    # ------------------------------------------------------------------

    def _profit(self, m: np.ndarray) -> float:
        """Expected total profit = sum((price - cost) * demand)."""
        x = self._demand.demand(m)
        p = m * self.tc
        return float(np.dot(p - self.cost, x))

    def _neg_profit(self, m: np.ndarray) -> float:
        """Negative profit (scipy minimises)."""
        return -self._profit(m)

    def _neg_profit_gradient(self, m: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of -profit w.r.t. m.

        d(profit)/d(m_i) = tc_i * x_i + (p_i - cost_i) * dx_i/dm_i
        d(-profit)/d(m_i) = -(tc_i * x_i + (p_i - cost_i) * dx_i)
        """
        x = self._demand.demand(m)
        dx = self._demand.demand_gradient(m)
        p = m * self.tc
        grad = self.tc * x + (p - self.cost) * dx
        return -grad

    def _combined_ratio(self, m: np.ndarray) -> float:
        """Expected aggregate loss ratio (for minimisation objective)."""
        x = self._demand.demand(m)
        p = m * self.tc
        gwp = np.dot(p, x)
        if gwp < 1e-10:
            return float("inf")
        return float(np.dot(self.cost, x) / gwp)

    # ------------------------------------------------------------------
    # Initial point generation
    # ------------------------------------------------------------------

    def _x0_from_bounds(self, jitter: float = 0.0) -> np.ndarray:
        """
        Build starting point as midpoint of bounds with optional jitter.

        Starting at the midpoint of [lb, ub] avoids trivial solutions at
        the boundary and gives SLSQP a sensible step to take.
        """
        lb = self._bounds.lb
        ub = self._bounds.ub
        mid = 0.5 * (lb + ub)
        if jitter > 0:
            noise = self.rng.uniform(-jitter, jitter, size=self.n)
            mid = mid * (1.0 + noise)
            mid = np.clip(mid, lb, ub)
        return mid

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def _solve_once(self, x0: np.ndarray) -> Any:
        """Run scipy.optimize.minimize once from initial point x0.

        Wraps the scipy call to handle a known SLSQP edge case where the
        Fortran subroutine returns an exit code that is not in scipy's
        exit_modes dict (e.g. 676972397 on large problems or certain
        platform/compiler combinations).  When this happens we construct a
        synthetic failure result rather than propagating a KeyError.
        """
        if self.solver_method == "SLSQP":
            options = {"ftol": self.ftol, "maxiter": self.maxiter, "disp": False}
        else:
            options = {"maxiter": self.maxiter, "verbose": 0}

        try:
            result = minimize(
                fun=self._neg_profit,
                x0=x0,
                jac=self._neg_profit_gradient,
                method=self.solver_method,
                bounds=self._bounds,
                constraints=self._scipy_constraints,
                options=options,
            )
        except KeyError as exc:
            # scipy SLSQP can raise KeyError when the Fortran exit code is not
            # in its exit_modes dict.  This happens on very large problems where
            # the internal workspace sizing overflows.  Return a synthetic
            # failure result so the outer loop can handle it gracefully.
            from scipy.optimize import OptimizeResult
            warnings.warn(
                f"SLSQP returned an unknown exit code ({exc}). "
                "This usually means the problem is too large for the internal "                "Fortran workspace. Returning failure result for this restart.",
                RuntimeWarning,
                stacklevel=3,
            )
            result = OptimizeResult(
                x=x0,
                fun=self._neg_profit(x0),
                jac=np.zeros_like(x0),
                nit=0,
                nfev=1,
                njev=0,
                status=-1,
                success=False,
                message=f"SLSQP unknown exit code: {exc}",
            )
        return result

    def _check_feasibility(self, m: np.ndarray, tol: float = 1e-4) -> bool:
        """
        Verify that all constraints are satisfied at m.

        Returns True if all inequalities are >= -tol.
        """
        lb = self._bounds.lb
        ub = self._bounds.ub
        if np.any(m < lb - tol) or np.any(m > ub + tol):
            return False
        for con in self._scipy_constraints:
            val = con["fun"](m)
            if val < -tol:
                return False
        return True

    def optimise(self) -> OptimisationResult:
        """
        Run the portfolio optimisation and return results.

        Runs up to n_restarts times from different starting points. Returns
        the best feasible solution found. If no restart yields a feasible
        solution, returns the best infeasible solution with converged=False.

        Returns
        -------
        OptimisationResult
        """
        best_result = None
        best_profit = -np.inf
        best_feasible = False

        for restart in range(self.n_restarts):
            jitter = 0.0 if restart == 0 else 0.1 * restart
            x0 = self._x0_from_bounds(jitter=jitter)
            scipy_result = self._solve_once(x0)
            m = np.clip(scipy_result.x, self._bounds.lb, self._bounds.ub)
            feasible = self._check_feasibility(m)
            profit = self._profit(m)

            # Accept if: feasible and better profit, or first feasible, or
            # no feasible found and better than previous best infeasible
            if feasible and (not best_feasible or profit > best_profit):
                best_result = scipy_result
                best_profit = profit
                best_feasible = True
                best_m = m
            elif not best_feasible and profit > best_profit:
                best_result = scipy_result
                best_profit = profit
                best_m = m

        if best_result is None:
            raise RuntimeError("Optimiser failed to produce any result.")

        m_opt = best_m
        converged = bool(best_result.success) and best_feasible

        if not converged and best_result.success and not best_feasible:
            warnings.warn(
                "SLSQP reported success but constraint violations detected. "
                "Treat result with caution. Consider relaxing constraints or "
                "increasing maxiter.",
                stacklevel=2,
            )

        return self._build_result(
            m_opt=m_opt,
            scipy_result=best_result,
            converged=converged,
        )

    def _build_result(
        self,
        m_opt: np.ndarray,
        scipy_result: Any,
        converged: bool,
    ) -> OptimisationResult:
        """Package the scipy result into an OptimisationResult."""
        x_opt = self._demand.demand(m_opt)
        p_opt = m_opt * self.tc

        profit = float(np.dot(p_opt - self.cost, x_opt))
        gwp = float(np.dot(p_opt, x_opt))
        lr = float(np.dot(self.cost, x_opt) / max(gwp, 1e-10))

        # Retention (renewal policies only)
        n_renewal = int(np.sum(self.renewal_flag))
        if n_renewal > 0:
            retention = float(np.sum(x_opt[self.renewal_flag]) / n_renewal)
        else:
            retention = None

        # Shadow prices
        shadow_prices = extract_shadow_prices(scipy_result, self._constraint_names)

        # Constraint values at solution
        constraint_values = evaluate_constraints(
            m_opt, self._scipy_constraints, self._constraint_names
        )

        # Per-policy summary DataFrame
        rate_change_pct = (p_opt / np.maximum(self.tc * self.prior_multiplier, 1e-10) - 1.0) * 100.0

        # ENBP binding: where renewal and m_opt is within 1% of ENBP bound
        if self.enbp is not None:
            enbp_ub_m = self.enbp / np.maximum(self.tc, 1e-10) * (1.0 - self.config.enbp_buffer)
            enbp_binding = self.renewal_flag & (m_opt >= enbp_ub_m - 1e-4)
        else:
            enbp_binding = np.zeros(self.n, dtype=bool)

        summary_df = pl.DataFrame(
            {
                "policy_idx": list(range(self.n)),
                "multiplier": m_opt.tolist(),
                "new_premium": p_opt.tolist(),
                "expected_demand": x_opt.tolist(),
                "contribution": ((p_opt - self.cost) * x_opt).tolist(),
                "enbp_binding": enbp_binding.tolist(),
                "rate_change_pct": rate_change_pct.tolist(),
            }
        )

        # Build audit trail
        constraint_config_dict = dataclasses.asdict(self.config)
        audit = build_audit_trail(
            n_policies=self.n,
            n_renewal=n_renewal,
            technical_price=self.tc,
            expected_loss_cost=self.cost,
            enbp=self.enbp,
            prior_multiplier=self.prior_multiplier,
            constraint_config_dict=constraint_config_dict,
            demand_model_name=self.demand_model_name,
            solver=self.solver_method,
            solver_options={"ftol": self.ftol, "maxiter": self.maxiter},
            n_restarts=self.n_restarts,
            x0_strategy="midpoint_with_jitter",
            multipliers=m_opt,
            converged=converged,
            solver_message=str(scipy_result.message),
            n_iter=int(getattr(scipy_result, "nit", 0)),
            n_fun_eval=int(getattr(scipy_result, "nfev", 0)),
            expected_profit=profit,
            expected_gwp=gwp,
            expected_lr=lr,
            expected_retention=retention,
            constraint_values=constraint_values,
            shadow_prices=shadow_prices,
        )

        return OptimisationResult(
            multipliers=m_opt,
            new_premiums=p_opt,
            expected_demand=x_opt,
            expected_profit=profit,
            expected_gwp=gwp,
            expected_loss_ratio=lr,
            expected_retention=retention,
            shadow_prices=shadow_prices,
            converged=converged,
            solver_message=str(scipy_result.message),
            n_iter=int(getattr(scipy_result, "nit", 0)),
            audit_trail=audit,
            summary_df=summary_df,
        )

    # ------------------------------------------------------------------
    # Scenario mode
    # ------------------------------------------------------------------

    def optimise_scenarios(
        self,
        elasticity_scenarios: list[np.ndarray],
        scenario_names: list[str] | None = None,
    ) -> "ScenarioResult":
        """
        Run optimisation under multiple elasticity scenarios.

        Each scenario uses the same constraint config and solver settings.
        This is the recommended approach for handling elasticity uncertainty
        (from KB entry 611): run 3-5 scenarios (pessimistic, central,
        optimistic) and report the spread.

        Parameters
        ----------
        elasticity_scenarios:
            List of elasticity arrays, each shape (N,). E.g. [beta_lower,
            beta_hat, beta_upper] from insurance-elasticity confidence
            intervals.
        scenario_names:
            Names for the scenarios. Defaults to 'scenario_0', 'scenario_1',
            etc.

        Returns
        -------
        ScenarioResult
        """
        from insurance_optimise.result import ScenarioResult
        from insurance_optimise._demand_model import make_demand_model

        if scenario_names is None:
            scenario_names = [f"scenario_{i}" for i in range(len(elasticity_scenarios))]

        results = []
        for elast in elasticity_scenarios:
            # Temporarily swap the demand model
            saved_demand = self._demand
            self._demand = make_demand_model(
                self.demand_model_name, self.x0, np.asarray(elast, dtype=float), self.tc
            )
            # Rebuild scipy constraints with new demand model
            saved_constraints = self._scipy_constraints
            self._scipy_constraints = build_scipy_constraints(
                config=self.config,
                technical_price=self.tc,
                expected_loss_cost=self.cost,
                renewal_flag=self.renewal_flag,
                demand_model=self._demand,
                claims_variance=self.claims_variance,
            )
            try:
                r = self.optimise()
            finally:
                self._demand = saved_demand
                self._scipy_constraints = saved_constraints
            results.append(r)

        multiplier_stack = np.stack([r.multipliers for r in results], axis=0)
        profits = np.array([r.expected_profit for r in results])

        return ScenarioResult(
            results=results,
            scenario_names=scenario_names,
            multiplier_mean=np.mean(multiplier_stack, axis=0),
            multiplier_p10=np.percentile(multiplier_stack, 10, axis=0),
            multiplier_p90=np.percentile(multiplier_stack, 90, axis=0),
            profit_mean=float(np.mean(profits)),
            profit_p10=float(np.percentile(profits, 10)),
            profit_p90=float(np.percentile(profits, 90)),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_constraint_names(self) -> list[str]:
        """Build constraint names matching the order in _scipy_constraints."""
        names = []
        cfg = self.config
        if cfg.lr_max is not None:
            names.append("lr_max")
        if cfg.lr_min is not None:
            names.append("lr_min")
        if cfg.gwp_min is not None:
            names.append("gwp_min")
        if cfg.gwp_max is not None:
            names.append("gwp_max")
        if cfg.retention_min is not None and np.any(self.renewal_flag):
            names.append("retention_min")
        return names

    def _validate_inputs(self) -> None:
        """Validate input array shapes and value ranges."""
        n = self.n
        for name, arr in [
            ("expected_loss_cost", self.cost),
            ("p_demand", self.x0),
            ("elasticity", self.elasticity),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"Length mismatch: technical_price has {n} elements but "
                    f"{name} has {len(arr)}."
                )
        if self.enbp is not None and len(self.enbp) != n:
            raise ValueError(
                f"enbp has {len(self.enbp)} elements but technical_price has {n}."
            )
        if np.any(self.tc <= 0):
            raise ValueError("technical_price must be positive.")
        if np.any(self.x0 <= 0) or np.any(self.x0 >= 1):
            raise ValueError("p_demand must be in (0, 1).")
        if self.enbp is not None and np.any(self.enbp <= 0):
            raise ValueError("enbp values must be positive.")

    @property
    def n_constraints(self) -> int:
        """Number of active scipy (non-bound) constraints."""
        return len(self._scipy_constraints)

    def portfolio_summary(self, m: np.ndarray | None = None) -> dict[str, float]:
        """
        Return portfolio metrics at a given multiplier array.

        If m is None, returns metrics at prior_multiplier (baseline).

        Parameters
        ----------
        m:
            Multiplier array, shape (N,). Defaults to prior_multiplier.

        Returns
        -------
        dict with keys: profit, gwp, loss_ratio, retention.
        """
        if m is None:
            m = self.prior_multiplier
        x = self._demand.demand(m)
        p = m * self.tc
        profit = float(np.dot(p - self.cost, x))
        gwp = float(np.dot(p, x))
        lr = float(np.dot(self.cost, x) / max(gwp, 1e-10))
        n_renewal = int(np.sum(self.renewal_flag))
        retention = (
            float(np.sum(x[self.renewal_flag]) / n_renewal)
            if n_renewal > 0
            else None
        )
        return {"profit": profit, "gwp": gwp, "loss_ratio": lr, "retention": retention}
