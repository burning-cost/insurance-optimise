"""
OptimisationResult and related output types.

Every solve returns an OptimisationResult. The audit_trail field is a
JSON-serialisable dict that satisfies FCA auditability requirements under
Consumer Duty — it captures exactly what the optimiser was asked to do and
what it produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass
class OptimisationResult:
    """
    Output from a single PortfolioOptimiser.optimise() call.

    Attributes
    ----------
    multipliers:
        Array of optimal price multipliers, shape (N,). Applied to
        technical_price to get the optimal premium: price = multiplier *
        technical_price.
    new_premiums:
        Array of optimal premiums, shape (N,). Equal to multipliers *
        technical_price.
    expected_demand:
        Array of expected demand at optimal prices, shape (N,). Units match
        the input exposure (policy counts, exposure years, etc.).
    expected_profit:
        Scalar total expected profit = sum((price - cost) * demand).
    expected_gwp:
        Scalar expected gross written premium = sum(price * demand).
    expected_loss_ratio:
        Scalar expected aggregate loss ratio = sum(cost * demand) / GWP.
    expected_retention:
        Scalar expected retention rate for renewal policies (0-1). None if
        no renewal policies in the dataset.
    shadow_prices:
        Dict mapping constraint name to its Lagrange multiplier (dual
        value). Positive value = constraint is binding and its marginal cost
        of tightening by 1 unit. E.g. ``{"lr_max": 0.034}`` means relaxing
        the LR ceiling by 0.001 would improve profit by 0.034 * 0.001.
    converged:
        True if the solver reported successful convergence AND all
        constraints are satisfied to tolerance.
    solver_message:
        Raw message from scipy.
    n_iter:
        Number of solver iterations taken.
    audit_trail:
        JSON-serialisable dict containing full record of inputs, constraints,
        solution, and convergence info. Write to disk for regulatory evidence.
    summary_df:
        Per-policy output as a Polars DataFrame. Columns: policy_idx,
        multiplier, new_premium, expected_demand, contribution,
        enbp_binding, rate_change_pct.
    """

    multipliers: np.ndarray
    new_premiums: np.ndarray
    expected_demand: np.ndarray
    expected_profit: float
    expected_gwp: float
    expected_loss_ratio: float
    expected_retention: float | None
    shadow_prices: dict[str, float]
    converged: bool
    solver_message: str
    n_iter: int
    audit_trail: dict[str, Any]
    summary_df: pl.DataFrame

    def to_json(self) -> str:
        """Return the audit trail as a JSON string."""
        return json.dumps(self.audit_trail, indent=2, default=_json_default)

    def save_audit(self, path: str) -> None:
        """Write audit trail to ``path`` (creates or overwrites)."""
        with open(path, "w") as f:
            f.write(self.to_json())

    def __repr__(self) -> str:  # noqa: D105
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        return (
            f"OptimisationResult({status}, N={len(self.multipliers)}, "
            f"profit={self.expected_profit:,.0f}, "
            f"gwp={self.expected_gwp:,.0f}, "
            f"lr={self.expected_loss_ratio:.3f})"
        )


@dataclass
class ScenarioResult:
    """
    Output from PortfolioOptimiser.optimise_scenarios().

    Contains one OptimisationResult per scenario plus summary statistics
    across scenarios.
    """

    results: list[OptimisationResult]
    scenario_names: list[str]
    multiplier_mean: np.ndarray
    multiplier_p10: np.ndarray
    multiplier_p90: np.ndarray
    profit_mean: float
    profit_p10: float
    profit_p90: float

    def summary(self) -> pl.DataFrame:
        """
        Return a Polars DataFrame summarising per-scenario portfolio metrics.
        """
        rows = []
        for name, r in zip(self.scenario_names, self.results):
            rows.append(
                {
                    "scenario": name,
                    "converged": r.converged,
                    "profit": r.expected_profit,
                    "gwp": r.expected_gwp,
                    "loss_ratio": r.expected_loss_ratio,
                    "retention": r.expected_retention,
                }
            )
        return pl.DataFrame(rows)


@dataclass
class FrontierPoint:
    """A single point on the efficient frontier."""

    epsilon: float
    result: OptimisationResult


@dataclass
class EfficientFrontierResult:
    """
    Output from EfficientFrontier.run().

    Contains the sequence of frontier points and a convenience DataFrame.
    """

    points: list[FrontierPoint]
    sweep_param: str
    data: pl.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        rows = []
        for p in self.points:
            rows.append(
                {
                    "epsilon": p.epsilon,
                    "converged": p.result.converged,
                    "profit": p.result.expected_profit,
                    "gwp": p.result.expected_gwp,
                    "loss_ratio": p.result.expected_loss_ratio,
                    "retention": p.result.expected_retention,
                }
            )
        self.data = pl.DataFrame(rows)

    def pareto_data(self) -> pl.DataFrame:
        """Return only the converged points on the frontier."""
        return self.data.filter(pl.col("converged"))


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
