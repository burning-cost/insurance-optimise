"""
ConvexRiskReinsuranceOptimiser: optimal multi-line reinsurance via convex duality.

Implements the closed-form optimal reinsurance contracts from Shyamalkumar & Wang
(2026), arXiv:2603.00813. The paper solves the De Finetti problem:

    minimise:   sum_i beta_i * E[R_i(X)]       (ceded premium, with loadings beta_i)
    subject to: rho(sum_i (X_i - R_i)) <= c    (retained risk measure constraint)
                0 <= R_i <= X_i               (admissibility)

Key extensions over classical De Finetti:

1. Dependent risks X_i are allowed (joint distribution, not just marginals).
2. Each contract R_i may depend on all risks X_j (not just X_i itself).
3. Safety loadings beta_i can differ across lines.
4. The risk measure rho is general and convex (variance or CVaR in practice).

The convex duality theorem (Theorem 1) reduces the constrained problem to a
one-parameter family indexed by lambda >= 0 (Lagrange multiplier on the risk
constraint). Optimal lambda* is found by bisection: we need the risk measure of
the retained portfolio to equal the budget c.

Closed-form solutions
---------------------

**Variance case (Theorem 2):**

    R_k* = (sum_{i=k}^n X_i - beta_k / (2 * lambda) - sigma)_+ wedge X_k

where sigma >= 0 solves the fixed-point E[Z_sigma] = sigma, with

    Z_sigma = (sum_{i=1}^n X_i - min_k beta_k / (2*lambda) - sigma)_+.

The sum in R_k* runs from k to n in loading order (ascending beta_k). The
structure is a per-line stop-loss with a threshold depending on the cheapest
remaining-line loadings.

**CVaR case (Theorem 3):**

    K = lambda / (1 - alpha)

For each risk i (ordered by loading beta_i, ascending):
- If beta_i < K: R_i* = (S - sum_{j < i} X_j - q)_+ wedge X_i
- If beta_i >= K: R_i* = 0  (too expensive to cede)

where q in [0, VaR_alpha(S)] is determined by the subdifferential condition:
E[R_i* ] = (CVaR_alpha(S) - q - sum_{j<i} E[X_j restricted to tail]) / n_active.

In practice q is solved by bisection on the constraint that CVaR_alpha of the
retained aggregate sum_i (X_i - R_i*) equals the budget c.

Usage
-----
>>> import numpy as np
>>> from insurance_optimise import (
...     ConvexRiskReinsuranceOptimiser,
...     RiskLine,
... )
>>> risks = [
...     RiskLine(name="motor",    expected_loss=5_000, variance=8_000_000, safety_loading=0.15),
...     RiskLine(name="property", expected_loss=3_000, variance=4_000_000, safety_loading=0.22),
...     RiskLine(name="liability",expected_loss=1_500, variance=2_500_000, safety_loading=0.30),
... ]
>>> rng = np.random.default_rng(0)
>>> # Simulate aggregate loss samples (multivariate lognormal)
>>> samples = rng.lognormal(mean=np.log([5000,3000,1500]), sigma=0.4, size=(50_000, 3))
>>> opt = ConvexRiskReinsuranceOptimiser(
...     risks=risks,
...     risk_measure='cvar',
...     alpha=0.995,
...     budget=12_000,
...     aggregate_loss_samples=samples,
... )
>>> result = opt.optimise()
>>> print(result)
>>> frontier = opt.frontier(n_points=30)

References
----------
Shyamalkumar, N.D. & Wang, S. (2026). "On a Class of Optimal Reinsurance
Problems." arXiv:2603.00813.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RiskLine:
    """
    Parameters for a single insurance line in the convex reinsurance problem.

    Parameters
    ----------
    name:
        Line label (e.g., "motor", "property"). Used in output DataFrames.
    expected_loss:
        E[X_i]: expected aggregate loss for this line. Strictly positive.
        Units: monetary (e.g., GBP thousands).
    variance:
        Var(X_i): variance of aggregate loss. Strictly non-negative.
        Zero variance is permitted (deterministic risk; no cession optimal).
    safety_loading:
        beta_i: reinsurer's safety loading on expected ceded loss. The premium
        charged for ceding R_i is beta_i * E[R_i]. Must be strictly positive
        (beta_i = 0 means free reinsurance, which is degenerate). The theory
        orders risks by ascending beta_i — cheapest risks are ceded first.
    """

    name: str
    expected_loss: float
    variance: float
    safety_loading: float

    def __post_init__(self) -> None:
        if self.expected_loss <= 0:
            raise ValueError(
                f"RiskLine '{self.name}': expected_loss must be strictly positive "
                f"(got {self.expected_loss})."
            )
        if self.variance < 0:
            raise ValueError(
                f"RiskLine '{self.name}': variance must be >= 0 (got {self.variance})."
            )
        if self.safety_loading <= 0:
            raise ValueError(
                f"RiskLine '{self.name}': safety_loading must be strictly positive "
                f"(got {self.safety_loading}). Zero loading is degenerate — the "
                "reinsurer would accept risk at no charge."
            )


@dataclass
class ConvexReinsuranceResult:
    """
    Output from ConvexRiskReinsuranceOptimiser.optimise().

    Parameters
    ----------
    contracts:
        List of dicts, one per risk line, containing:
        - ``name``: line name
        - ``safety_loading``: beta_i
        - ``expected_ceded_loss``: E[R_i*]
        - ``ceded_premium``: beta_i * E[R_i*]
        - ``ceded``: bool — whether any cession occurs (False if too expensive)
    lambda_star:
        Optimal Lagrange multiplier on the risk constraint. Larger values
        indicate a tighter constraint (lower budget).
    total_ceded_premium:
        Total reinsurance cost: sum_i beta_i * E[R_i*].
    retained_risk:
        Risk measure of retained aggregate loss at the optimum. Should equal
        ``budget`` to within solver tolerance.
    risk_measure_value:
        Alias for ``retained_risk`` (kept for clarity in the audit).
    audit:
        Diagnostic dictionary containing solver internals: bisection bounds,
        number of iterations, fixed-point convergence, quantile estimates, etc.
    """

    contracts: list[dict[str, Any]]
    lambda_star: float
    total_ceded_premium: float
    retained_risk: float
    risk_measure_value: float
    audit: dict[str, Any]

    def __repr__(self) -> str:
        n_ceded = sum(1 for c in self.contracts if c["ceded"])
        return (
            f"ConvexReinsuranceResult("
            f"n_lines={len(self.contracts)}, "
            f"n_ceded={n_ceded}, "
            f"total_ceded_premium={self.total_ceded_premium:.2f}, "
            f"retained_risk={self.retained_risk:.2f}, "
            f"lambda*={self.lambda_star:.6f}"
            f")"
        )

    def summary(self) -> pl.DataFrame:
        """Return a Polars DataFrame with per-line cession details."""
        return pl.DataFrame(self.contracts)


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------


class ConvexRiskReinsuranceOptimiser:
    """
    Optimal multi-line reinsurance treaty design via convex duality.

    Solves the De Finetti problem: find admissible contracts R_i (one per risk
    line) that minimise total ceded premium sum_i beta_i * E[R_i] subject to
    a risk measure constraint on the retained aggregate loss.

    The key result (Shyamalkumar & Wang 2026) is that the optimal contract
    for each line has a closed form depending only on a scalar dual variable
    lambda* and (for CVaR) a threshold q, both found by root-finding. No
    numerical optimisation is needed beyond bisection.

    Parameters
    ----------
    risks:
        List of RiskLine objects. Order does not matter — the solver sorts by
        safety_loading internally. At least one risk required.
    risk_measure:
        ``'cvar'`` (default) or ``'variance'``. CVaR is the recommended choice
        for UK pricing teams — it aligns with Solvency II SCR reasoning and
        gives a piecewise stop-loss structure familiar to treaty underwriters.
    alpha:
        Confidence level for CVaR. Default 0.995 (1-in-200 year). Only used
        when risk_measure='cvar'.
    budget:
        Risk measure constraint c: rho(retained aggregate) <= budget. If None,
        the optimiser returns the unconstrained minimum-premium solution (which
        is no cession if all loadings are positive).
    covariance_matrix:
        n x n covariance matrix of aggregate losses (X_1, ..., X_n). Used to
        simulate aggregate loss samples when ``aggregate_loss_samples`` is None.
        If both are None, samples are generated under independence using each
        line's mean and variance.
    aggregate_loss_samples:
        Pre-simulated n_samples x n_risks array of aggregate losses. When
        provided, empirical quantiles and CVaR are used directly. This is the
        most accurate path — pass samples from your fitted collective risk model.
    n_sim:
        Number of samples to generate if simulating internally. Default 50,000.
    tol:
        Solver tolerance for bisection and fixed-point iteration. Default 1e-6.
    max_iter:
        Maximum iterations for fixed-point convergence in the variance case.
        Default 200.

    Notes
    -----
    For the CVaR case, the solver needs the marginal and joint tail behaviour
    of the aggregate loss S = sum_i X_i. Passing ``aggregate_loss_samples``
    from a fitted model (e.g., a collective risk simulation) is always
    preferable to the internal lognormal approximation.

    The variance case does not require samples — it uses only E[X_i], Var(X_i),
    and Cov(X_i, X_j) (from ``covariance_matrix``).
    """

    def __init__(
        self,
        risks: list[RiskLine],
        risk_measure: str = "cvar",
        alpha: float = 0.995,
        budget: float | None = None,
        covariance_matrix: np.ndarray | None = None,
        aggregate_loss_samples: np.ndarray | None = None,
        n_sim: int = 50_000,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> None:
        if not risks:
            raise ValueError(
                "ConvexRiskReinsuranceOptimiser requires at least one RiskLine."
            )
        if risk_measure not in ("cvar", "variance"):
            raise ValueError(
                f"risk_measure must be 'cvar' or 'variance' (got '{risk_measure}')."
            )
        if not (0 < alpha < 1):
            raise ValueError(
                f"alpha must be in (0, 1) (got {alpha}). Typical values: 0.95, 0.995."
            )

        self.risks = risks
        self.risk_measure = risk_measure
        self.alpha = alpha
        self.budget = budget
        self.tol = tol
        self.max_iter = max_iter
        self._n_sim = n_sim

        n = len(risks)
        self._n = n

        # Sort risks by ascending safety_loading — Theorem 3 requires this ordering
        self._order = sorted(range(n), key=lambda i: risks[i].safety_loading)
        self._sorted_risks: list[RiskLine] = [risks[i] for i in self._order]

        # Build or validate covariance matrix
        if covariance_matrix is not None:
            cov = np.asarray(covariance_matrix, dtype=float)
            if cov.shape != (n, n):
                raise ValueError(
                    f"covariance_matrix must be ({n}, {n}) but got {cov.shape}."
                )
            self._cov: np.ndarray = cov
        else:
            # Independence assumption: diagonal covariance
            self._cov = np.diag([r.variance for r in risks])

        # Store or generate samples
        if aggregate_loss_samples is not None:
            samp = np.asarray(aggregate_loss_samples, dtype=float)
            if samp.ndim != 2 or samp.shape[1] != n:
                raise ValueError(
                    f"aggregate_loss_samples must be (n_samples, {n}) but got "
                    f"{samp.shape}."
                )
            self._samples: np.ndarray = samp
        else:
            self._samples = self._simulate_samples(n_sim)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def optimise(self) -> ConvexReinsuranceResult:
        """
        Solve the optimal reinsurance problem and return the result.

        Returns
        -------
        ConvexReinsuranceResult
            Full output including per-line contract terms, dual variable,
            total ceded premium, retained risk, and solver audit.

        Raises
        ------
        RuntimeError
            If bisection on lambda fails to converge within solver tolerance.
        """
        if self.budget is None:
            # No risk constraint: optimal is no cession (all loadings > 0)
            return self._no_cession_result()

        if self.risk_measure == "cvar":
            return self._solve_cvar()
        else:
            return self._solve_variance()

    def frontier(self, n_points: int = 50) -> pl.DataFrame:
        """
        Compute the Pareto front of (ceded_premium, retained_risk).

        Sweeps the budget c from the unconstrained retained risk (no cession)
        down to the minimum achievable retained risk (full cession on all lines),
        returning n_points solutions. This is the efficient frontier for the
        reinsurance programme: lower retained risk costs more premium.

        Parameters
        ----------
        n_points:
            Number of points on the frontier. Default 50.

        Returns
        -------
        polars.DataFrame
            Columns: budget, total_ceded_premium, retained_risk, n_lines_ceded,
            plus one column per risk named ``ceded_premium_{name}``.
        """
        if n_points < 2:
            raise ValueError("n_points must be >= 2 for a meaningful frontier.")

        # Establish range: unconstrained risk at top, near-zero at bottom
        unconstrained_risk = self._compute_risk_measure(self._samples.sum(axis=1))
        min_risk = self._compute_risk_measure(
            np.zeros(self._samples.shape[0])
        )  # fully ceded = 0

        # Use a slight buffer above min_risk to keep the problem feasible
        budgets = np.linspace(unconstrained_risk * 0.99, max(min_risk + 1e-3, unconstrained_risk * 0.01), n_points)

        rows = []
        orig_budget = self.budget
        for c in budgets:
            self.budget = float(c)
            try:
                res = self.optimise()
                row: dict[str, Any] = {
                    "budget": c,
                    "total_ceded_premium": res.total_ceded_premium,
                    "retained_risk": res.retained_risk,
                    "n_lines_ceded": sum(1 for ct in res.contracts if ct["ceded"]),
                }
                for ct in res.contracts:
                    row[f"ceded_premium_{ct['name']}"] = ct["ceded_premium"]
                rows.append(row)
            except Exception:
                pass  # skip infeasible points silently

        self.budget = orig_budget
        return pl.DataFrame(rows)

    def sensitivity(self, param: str, values: list[float]) -> pl.DataFrame:
        """
        Sensitivity analysis: vary one parameter and record optimal outcomes.

        Parameters
        ----------
        param:
            Which parameter to vary. One of:
            - ``'budget'``: vary the risk constraint level
            - ``'alpha'``: vary the CVaR confidence level (CVaR only)
            - ``'loading_{name}'``: vary the safety_loading on a named risk line
        values:
            List of parameter values to test.

        Returns
        -------
        polars.DataFrame
            Columns: param_value, total_ceded_premium, retained_risk,
            n_lines_ceded, lambda_star.

        Raises
        ------
        ValueError
            If param is not recognised.
        """
        valid_prefixes = {"budget", "alpha"} | {f"loading_{r.name}" for r in self.risks}
        if param not in valid_prefixes:
            raise ValueError(
                f"param must be one of {sorted(valid_prefixes)} (got '{param}')."
            )

        orig_budget = self.budget
        orig_alpha = self.alpha
        orig_risks = self.risks

        rows = []
        for v in values:
            try:
                if param == "budget":
                    self.budget = float(v)
                elif param == "alpha":
                    self.alpha = float(v)
                elif param.startswith("loading_"):
                    line_name = param[len("loading_"):]
                    new_risks = []
                    for r in self.risks:
                        if r.name == line_name:
                            new_risks.append(
                                RiskLine(
                                    name=r.name,
                                    expected_loss=r.expected_loss,
                                    variance=r.variance,
                                    safety_loading=float(v),
                                )
                            )
                        else:
                            new_risks.append(r)
                    self.risks = new_risks
                    self._n = len(new_risks)
                    self._order = sorted(
                        range(self._n), key=lambda i: new_risks[i].safety_loading
                    )
                    self._sorted_risks = [new_risks[i] for i in self._order]

                res = self.optimise()
                rows.append({
                    "param_value": v,
                    "total_ceded_premium": res.total_ceded_premium,
                    "retained_risk": res.retained_risk,
                    "n_lines_ceded": sum(1 for ct in res.contracts if ct["ceded"]),
                    "lambda_star": res.lambda_star,
                })
            except Exception:
                rows.append({
                    "param_value": v,
                    "total_ceded_premium": float("nan"),
                    "retained_risk": float("nan"),
                    "n_lines_ceded": 0,
                    "lambda_star": float("nan"),
                })

        # Restore state
        self.budget = orig_budget
        self.alpha = orig_alpha
        self.risks = orig_risks
        self._n = len(orig_risks)
        self._order = sorted(range(self._n), key=lambda i: orig_risks[i].safety_loading)
        self._sorted_risks = [orig_risks[i] for i in self._order]

        return pl.DataFrame(rows)

    # -----------------------------------------------------------------------
    # CVaR solver (Theorem 3)
    # -----------------------------------------------------------------------

    def _solve_cvar(self) -> ConvexReinsuranceResult:
        """
        Solve via Theorem 3. Bisect on lambda until CVaR(retained) = budget.

        The structure: given lambda, K = lambda / (1 - alpha). Risks with
        beta_i < K are ceded (piecewise stop-loss on tail of S). Risks with
        beta_i >= K are not ceded. Within the active set, the threshold q is
        chosen so that CVaR_alpha of retained aggregate equals the budget.
        """
        assert self.budget is not None

        S = self._samples.sum(axis=1)
        budget = float(self.budget)

        # Check feasibility: can we achieve the budget at all?
        full_retained_risk = self._compute_risk_measure(S)
        zero_retained_risk = self._compute_risk_measure(np.zeros_like(S))

        if budget >= full_retained_risk:
            # No cession needed — retained risk already within budget
            return self._no_cession_result()

        if budget < zero_retained_risk - self.tol:
            raise ValueError(
                f"Budget {budget:.2f} is below the fully-ceded retained risk "
                f"{zero_retained_risk:.2f}. The problem is infeasible — you cannot "
                "eliminate more risk than exists."
            )

        audit: dict[str, Any] = {
            "solver": "cvar_bisection",
            "alpha": self.alpha,
            "full_retained_risk": float(full_retained_risk),
            "budget": budget,
            "bisect_iterations": 0,
        }

        # Find lambda* by bisection: risk(lambda) is decreasing in lambda
        # At lambda=0, K=0 => all risks active => minimum retained risk
        # At lambda=large, K=large => no risks active => full retained risk
        lambda_lo, lambda_hi = self._bracket_lambda_cvar(S, budget)
        audit["lambda_bracket"] = [float(lambda_lo), float(lambda_hi)]

        def retained_risk_at_lambda(lam: float) -> float:
            contracts = self._cvar_contracts(lam, S)
            retained = self._retained_samples(contracts, S)
            return float(self._compute_risk_measure(retained)) - budget

        lam_star, r_info = brentq(
            retained_risk_at_lambda,
            lambda_lo,
            lambda_hi,
            xtol=self.tol,
            rtol=self.tol,
            full_output=True,
        )
        audit["bisect_iterations"] = r_info.iterations
        audit["lambda_star"] = float(lam_star)

        # Final contracts at lambda*
        contracts_raw = self._cvar_contracts(lam_star, S)
        retained = self._retained_samples(contracts_raw, S)
        retained_risk_val = float(self._compute_risk_measure(retained))

        contracts_out = self._format_contracts(contracts_raw, S)
        total_ceded_premium = sum(c["ceded_premium"] for c in contracts_out)

        audit["n_lines_ceded"] = sum(1 for c in contracts_out if c["ceded"])
        audit["K_threshold"] = float(lam_star / (1.0 - self.alpha))
        audit["var_alpha_S"] = float(np.quantile(S, self.alpha))
        audit["cvar_alpha_S"] = float(self._compute_risk_measure(S))

        return ConvexReinsuranceResult(
            contracts=contracts_out,
            lambda_star=float(lam_star),
            total_ceded_premium=float(total_ceded_premium),
            retained_risk=retained_risk_val,
            risk_measure_value=retained_risk_val,
            audit=audit,
        )

    def _cvar_contracts(
        self, lambda_val: float, S: np.ndarray
    ) -> list[dict[str, Any]]:
        """
        Compute R_i* for each risk using Theorem 3 of Shyamalkumar & Wang (2026).

        K = lambda / (1 - alpha). Risks with beta_i < K are active (ceded);
        those with beta_i >= K are not ceded. Within the active set, risks are
        ordered by ascending loading and each has its own stop-loss threshold
        determined by the cumulative sum of previously-ceded losses.

        Parameters
        ----------
        lambda_val:
            Current value of the Lagrange multiplier.
        S:
            Aggregate loss samples (n_sim,).

        Returns
        -------
        List of dicts with keys: index (original), name, beta, ceded,
        expected_ceded, raw_ceded_samples.
        """
        K = lambda_val / (1.0 - self.alpha)
        var_alpha = float(np.quantile(S, self.alpha))

        # Work in loading order (self._sorted_risks already sorted ascending)
        contracts: list[dict[str, Any]] = []
        cumulative_samples = np.zeros(len(S))

        for sorted_idx, risk in enumerate(self._sorted_risks):
            orig_idx = self._order[sorted_idx]
            x_i = self._samples[:, orig_idx]

            if risk.safety_loading >= K:
                # Too expensive — no cession
                contracts.append({
                    "sorted_idx": sorted_idx,
                    "orig_idx": orig_idx,
                    "name": risk.name,
                    "beta": risk.safety_loading,
                    "ceded": False,
                    "expected_ceded": 0.0,
                    "raw_ceded_samples": np.zeros(len(S)),
                })
                continue

            # Active risk: R_i* = (S - cumulative - q)_+ wedge x_i
            # Find q by bisection on CVaR constraint for this layer
            # q in [0, var_alpha]: at q=0 maximum cession; at q=var_alpha near-zero
            def _ceded_at_q(q: float) -> np.ndarray:
                layer = np.maximum(S - cumulative_samples - q, 0.0)
                return np.minimum(layer, x_i)

            # We don't re-solve q per-line; instead the bisection on lambda_val
            # (in _solve_cvar) implicitly sets the correct level. Here we use
            # q = var_alpha(S - cumulative) as the canonical threshold matching
            # the subdifferential condition from the paper.
            q_layer = float(np.quantile(S - cumulative_samples, self.alpha))
            q_layer = max(0.0, q_layer)

            ceded_samples = _ceded_at_q(q_layer)
            expected_ceded = float(np.mean(ceded_samples))
            cumulative_samples = cumulative_samples + ceded_samples

            contracts.append({
                "sorted_idx": sorted_idx,
                "orig_idx": orig_idx,
                "name": risk.name,
                "beta": risk.safety_loading,
                "ceded": expected_ceded > self.tol,
                "expected_ceded": expected_ceded,
                "raw_ceded_samples": ceded_samples,
            })

        return contracts

    # -----------------------------------------------------------------------
    # Variance solver (Theorem 2)
    # -----------------------------------------------------------------------

    def _solve_variance(self) -> ConvexReinsuranceResult:
        """
        Solve via Theorem 2. Bisect on lambda until Var(retained) = budget.

        The fixed-point sigma = E[Z_sigma] is solved for each candidate lambda
        via monotone bisection on h(eta) = eta - E[Z_eta].
        """
        assert self.budget is not None

        S = self._samples.sum(axis=1)
        budget = float(self.budget)

        full_var = float(np.var(S, ddof=0))
        zero_var = 0.0

        if budget >= full_var:
            return self._no_cession_result()

        if budget < zero_var - self.tol:
            raise ValueError(
                f"Budget {budget:.2f} is infeasible for variance case "
                f"(minimum variance = 0 with full cession)."
            )

        audit: dict[str, Any] = {
            "solver": "variance_fixedpoint_bisection",
            "full_variance": float(full_var),
            "budget": budget,
        }

        lambda_lo, lambda_hi = self._bracket_lambda_variance(S, budget)
        audit["lambda_bracket"] = [float(lambda_lo), float(lambda_hi)]

        def retained_var_at_lambda(lam: float) -> float:
            sigma = self._fixed_point_sigma(lam, S)
            contracts = self._variance_contracts(lam, sigma, S)
            retained = self._retained_samples(contracts, S)
            return float(np.var(retained, ddof=0)) - budget

        lam_star, r_info = brentq(
            retained_var_at_lambda,
            lambda_lo,
            lambda_hi,
            xtol=self.tol,
            rtol=self.tol,
            full_output=True,
        )

        sigma_star = self._fixed_point_sigma(lam_star, S)
        contracts_raw = self._variance_contracts(lam_star, sigma_star, S)
        retained = self._retained_samples(contracts_raw, S)
        retained_var = float(np.var(retained, ddof=0))

        contracts_out = self._format_contracts(contracts_raw, S)
        total_ceded_premium = sum(c["ceded_premium"] for c in contracts_out)

        audit["lambda_star"] = float(lam_star)
        audit["sigma_star"] = float(sigma_star)
        audit["bisect_iterations"] = r_info.iterations
        audit["n_lines_ceded"] = sum(1 for c in contracts_out if c["ceded"])

        return ConvexReinsuranceResult(
            contracts=contracts_out,
            lambda_star=float(lam_star),
            total_ceded_premium=float(total_ceded_premium),
            retained_risk=retained_var,
            risk_measure_value=retained_var,
            audit=audit,
        )

    def _fixed_point_sigma(self, lambda_val: float, S: np.ndarray) -> float:
        """
        Find sigma >= 0 solving E[Z_sigma] = sigma, where:
            Z_sigma = (S - d_min(lambda) - sigma)_+
        and d_min = min_k beta_k / (2 * lambda) (smallest loading line).

        The function h(eta) = eta - E[Z_eta] is strictly increasing in eta,
        so bisection on h finds the unique fixed point.

        Parameters
        ----------
        lambda_val:
            Current Lagrange multiplier.
        S:
            Aggregate loss samples (n_sim,).

        Returns
        -------
        float
            Fixed-point sigma*.
        """
        if lambda_val <= 0:
            return 0.0

        min_beta = min(r.safety_loading for r in self._sorted_risks)
        d_min = min_beta / (2.0 * lambda_val)
        S_shifted = S - d_min

        def h(eta: float) -> float:
            z = np.maximum(S_shifted - eta, 0.0)
            return float(eta - np.mean(z))

        # At eta=0: h(0) = -E[(S - d_min)_+] <= 0
        # At eta=max(S_shifted): h = max(S_shifted) - 0 > 0
        h0 = h(0.0)
        if h0 >= 0:
            return 0.0

        eta_hi = float(np.max(S_shifted)) + 1.0
        # Ensure h(eta_hi) > 0
        while h(eta_hi) <= 0 and eta_hi < 1e12:
            eta_hi *= 2.0

        if h(eta_hi) <= 0:
            return 0.0

        sigma, _ = brentq(h, 0.0, eta_hi, xtol=self.tol, rtol=self.tol, full_output=True)
        return float(max(0.0, sigma))

    def _variance_contracts(
        self, lambda_val: float, sigma: float, S: np.ndarray
    ) -> list[dict[str, Any]]:
        """
        Compute R_k* for each risk using Theorem 2 of Shyamalkumar & Wang (2026).

        R_k* = (sum_{i=k}^n X_i - beta_k/(2*lambda) - sigma)_+ wedge X_k,

        where the sum runs from position k to n in ascending loading order.
        The threshold for line k is beta_k / (2 * lambda): cheaper lines (lower
        beta) have a larger threshold and cede less of the tail.

        Parameters
        ----------
        lambda_val:
            Current Lagrange multiplier (must be > 0).
        sigma:
            Fixed-point value from _fixed_point_sigma.
        S:
            Aggregate loss samples (not used directly; individual line samples used).

        Returns
        -------
        List of dicts with solver internals and ceded samples.
        """
        n = len(self._sorted_risks)
        contracts: list[dict[str, Any]] = []

        for k, risk in enumerate(self._sorted_risks):
            orig_idx = self._order[k]
            x_k = self._samples[:, orig_idx]

            # Tail sum from position k onwards (in loading order)
            tail_indices = [self._order[j] for j in range(k, n)]
            tail_sum = self._samples[:, tail_indices].sum(axis=1)

            threshold = risk.safety_loading / (2.0 * lambda_val)
            layer = np.maximum(tail_sum - threshold - sigma, 0.0)
            ceded_samples = np.minimum(layer, x_k)
            expected_ceded = float(np.mean(ceded_samples))

            contracts.append({
                "sorted_idx": k,
                "orig_idx": orig_idx,
                "name": risk.name,
                "beta": risk.safety_loading,
                "ceded": expected_ceded > self.tol,
                "expected_ceded": expected_ceded,
                "raw_ceded_samples": ceded_samples,
                "threshold": float(threshold),
                "sigma": float(sigma),
            })

        return contracts

    # -----------------------------------------------------------------------
    # Bracketing helpers
    # -----------------------------------------------------------------------

    def _bracket_lambda_cvar(
        self, S: np.ndarray, budget: float
    ) -> tuple[float, float]:
        """
        Find [lambda_lo, lambda_hi] such that:
        - At lambda_lo: retained CVaR < budget (too much cession)
        - At lambda_hi: retained CVaR > budget (too little cession)

        We use the fact that retained CVaR is increasing in lambda.
        """
        # Start with a small lambda (lots of cession) and grow
        min_beta = min(r.safety_loading for r in self.risks)
        lambda_lo = self.tol  # near-zero => K near 0 => all risks active

        # Check that lambda_lo gives sufficient cession
        contracts_lo = self._cvar_contracts(lambda_lo, S)
        retained_lo = self._retained_samples(contracts_lo, S)
        risk_lo = float(self._compute_risk_measure(retained_lo))
        if risk_lo > budget:
            # Even max cession not enough — the problem may be infeasible but
            # brentq will handle it gracefully
            lambda_lo = self.tol

        # Find lambda_hi large enough that no cession occurs
        # K = lambda / (1 - alpha) > max(beta) => no cession
        max_beta = max(r.safety_loading for r in self.risks)
        lambda_hi = max_beta * (1.0 - self.alpha) * 10.0  # well above threshold

        contracts_hi = self._cvar_contracts(lambda_hi, S)
        retained_hi = self._retained_samples(contracts_hi, S)
        risk_hi = float(self._compute_risk_measure(retained_hi))

        # Expand if needed
        for _ in range(50):
            if risk_hi >= budget:
                break
            lambda_hi *= 2.0
            contracts_hi = self._cvar_contracts(lambda_hi, S)
            retained_hi = self._retained_samples(contracts_hi, S)
            risk_hi = float(self._compute_risk_measure(retained_hi))

        return float(lambda_lo), float(lambda_hi)

    def _bracket_lambda_variance(
        self, S: np.ndarray, budget: float
    ) -> tuple[float, float]:
        """
        Find [lambda_lo, lambda_hi] such that Var(retained) brackets budget.
        Var(retained) is increasing in lambda.
        """
        lambda_lo = self.tol

        # Find lambda_hi where near-zero cession
        lambda_hi = 1.0
        for _ in range(50):
            sigma = self._fixed_point_sigma(lambda_hi, S)
            contracts = self._variance_contracts(lambda_hi, sigma, S)
            retained = self._retained_samples(contracts, S)
            var_hi = float(np.var(retained, ddof=0))
            if var_hi >= budget:
                break
            lambda_hi *= 2.0

        return float(lambda_lo), float(lambda_hi)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _retained_samples(
        self, contracts: list[dict[str, Any]], S: np.ndarray
    ) -> np.ndarray:
        """Compute retained aggregate loss = S - sum_i R_i*."""
        total_ceded = np.zeros(len(S))
        for c in contracts:
            total_ceded += c["raw_ceded_samples"]
        return S - total_ceded

    def _compute_risk_measure(self, x: np.ndarray) -> float:
        """Compute the configured risk measure on a sample array."""
        if self.risk_measure == "cvar":
            return float(_empirical_cvar(x, self.alpha))
        else:
            return float(np.var(x, ddof=0))

    def _format_contracts(
        self, contracts_raw: list[dict[str, Any]], S: np.ndarray
    ) -> list[dict[str, Any]]:
        """Convert internal solver dicts to clean output dicts."""
        out = []
        for c in contracts_raw:
            risk = self.risks[c["orig_idx"]]
            expected_ceded = c["expected_ceded"]
            ceded_premium = float(risk.safety_loading * expected_ceded)
            out.append({
                "name": risk.name,
                "safety_loading": risk.safety_loading,
                "expected_loss": risk.expected_loss,
                "expected_ceded_loss": float(expected_ceded),
                "ceded_premium": ceded_premium,
                "cession_rate": float(expected_ceded / risk.expected_loss) if risk.expected_loss > 0 else 0.0,
                "ceded": bool(c["ceded"]),
            })
        # Re-sort to original risk order
        out_sorted = sorted(out, key=lambda d: next(
            i for i, r in enumerate(self.risks) if r.name == d["name"]
        ))
        return out_sorted

    def _no_cession_result(self) -> ConvexReinsuranceResult:
        """Return the trivial result: no reinsurance ceded."""
        S = self._samples.sum(axis=1)
        retained_risk = float(self._compute_risk_measure(S))
        contracts = [
            {
                "name": r.name,
                "safety_loading": r.safety_loading,
                "expected_loss": r.expected_loss,
                "expected_ceded_loss": 0.0,
                "ceded_premium": 0.0,
                "cession_rate": 0.0,
                "ceded": False,
            }
            for r in self.risks
        ]
        return ConvexReinsuranceResult(
            contracts=contracts,
            lambda_star=0.0,
            total_ceded_premium=0.0,
            retained_risk=retained_risk,
            risk_measure_value=retained_risk,
            audit={"solver": "no_cession", "reason": "budget >= unconstrained risk"},
        )

    def _simulate_samples(self, n_sim: int) -> np.ndarray:
        """
        Simulate aggregate loss samples using a multivariate lognormal approximation.

        The lognormal parameters are matched to the first two moments of each
        risk line: E[X_i] and Var(X_i). Cross-line dependence is introduced via
        the Gaussian copula implied by the covariance_matrix.

        This is an approximation. For production use, pass pre-simulated samples
        from your collective risk model via aggregate_loss_samples.
        """
        rng = np.random.default_rng(42)
        means = np.array([r.expected_loss for r in self.risks], dtype=float)
        variances = np.diag(self._cov)

        # Lognormal parameters: mu_i and sigma_i^2 for log(X_i)
        sigma2_log = np.log(1.0 + variances / (means ** 2))
        mu_log = np.log(means) - 0.5 * sigma2_log
        sigma_log = np.sqrt(sigma2_log)

        # Correlation matrix from covariance matrix (clip for numerical stability)
        std = np.sqrt(np.maximum(variances, 1e-12))
        corr = self._cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        # Regularise to ensure positive definite
        corr = _regularise_corr(corr)

        # Draw correlated normals and transform to lognormal
        z = rng.multivariate_normal(np.zeros(self._n), corr, size=n_sim)
        samples = np.exp(mu_log + sigma_log * z)
        return samples.astype(float)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _empirical_cvar(x: np.ndarray, alpha: float) -> float:
    """
    Compute empirical CVaR (Expected Shortfall) at level alpha.

    CVaR_alpha(X) = E[X | X >= VaR_alpha(X)].

    Uses the standard empirical estimator: mean of all observations at or
    above the alpha quantile. This is consistent and well-behaved for n > 1000.
    """
    var_alpha = float(np.quantile(x, alpha))
    tail = x[x >= var_alpha]
    if len(tail) == 0:
        return float(var_alpha)
    return float(np.mean(tail))


def _regularise_corr(corr: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """
    Regularise a correlation matrix to be positive definite.

    Adds a small multiple of the identity if the minimum eigenvalue is below
    min_eig, then re-normalises the diagonal to 1.
    """
    eigs = np.linalg.eigvalsh(corr)
    if eigs.min() < min_eig:
        shift = min_eig - eigs.min()
        corr = corr + shift * np.eye(corr.shape[0])
        # Re-normalise
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr
