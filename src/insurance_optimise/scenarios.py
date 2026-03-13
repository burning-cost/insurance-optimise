"""
Scenario-based objective and CVaR constraint for insurance optimisation.

When the input elasticity estimates carry uncertainty (they always do),
the simplest robust approach is to run the optimiser under K scenarios
(e.g. low/central/high elasticity) and report the spread.

For more sophisticated uncertainty handling, this module provides:
- ScenarioObjective: objective = (1/K) * sum_k profit_k over K demand scenarios
- CVaRConstraint: expected shortfall constraint on worst-alpha% profit scenarios

The scenario objective approach is directly motivated by KB entry 611:
"scenario-based approach preferred over DRO for v1".

CVaR formulation
----------------
Given K profit realisations {pi_k}, CVaR at level alpha (e.g. alpha=0.10
means worst 10%) is:

    CVaR_alpha = E[pi | pi <= VaR_alpha]
               = VaR_alpha + (1/alpha) * E[min(pi - VaR_alpha, 0)]

For the optimiser, we want CVaR_alpha >= -cvar_max (bound the worst-case loss).
The CVaR constraint is implemented using the Rockafellar-Uryasev (2000)
reformulation:

    CVaR_alpha(m) = max_{xi} {xi + (1/alpha*K) * sum_k max(-profit_k - xi, 0)}

This is non-smooth but can be approximated with a softplus. For simplicity,
we use the direct approximation: sort scenario profits, take the mean of
the worst ceil(alpha*K) scenarios.

Reference: Rockafellar & Uryasev (2000), "Optimization of Conditional
Value-at-Risk", Journal of Risk.
"""

from __future__ import annotations

import numpy as np

from insurance_optimise._demand_model import make_demand_model


class ScenarioObjective:
    """
    Objective function that averages profit over K demand scenarios.

    Parameters
    ----------
    technical_price:
        Technical price array, shape (N,).
    expected_loss_cost:
        Expected loss cost array, shape (N,).
    x0_scenarios:
        List of K baseline demand arrays, each shape (N,). If None,
        uses the single x0 array K times.
    elasticity_scenarios:
        List of K elasticity arrays, each shape (N,).
    demand_model_name:
        'log_linear' or 'logistic'.
    """

    def __init__(
        self,
        technical_price: np.ndarray,
        expected_loss_cost: np.ndarray,
        x0_scenarios: list[np.ndarray] | None,
        elasticity_scenarios: list[np.ndarray],
        demand_model_name: str = "log_linear",
    ) -> None:
        self.tc = np.asarray(technical_price, dtype=float)
        self.cost = np.asarray(expected_loss_cost, dtype=float)
        k = len(elasticity_scenarios)
        if x0_scenarios is None:
            raise ValueError("x0_scenarios must not be None")
        if len(x0_scenarios) != k:
            raise ValueError(
                f"x0_scenarios has {len(x0_scenarios)} elements but "
                f"elasticity_scenarios has {k}."
            )
        self.demand_models = [
            make_demand_model(demand_model_name, x0, elast, self.tc)
            for x0, elast in zip(x0_scenarios, elasticity_scenarios)
        ]
        self.k = k

    def profit_scenarios(self, m: np.ndarray) -> np.ndarray:
        """Compute profit for each scenario. Returns array of shape (K,)."""
        profits = np.zeros(self.k)
        p = m * self.tc
        for i, dm in enumerate(self.demand_models):
            x = dm.demand(m)
            profits[i] = float(np.dot(p - self.cost, x))
        return profits

    def mean_profit(self, m: np.ndarray) -> float:
        """Expected profit = mean across scenarios."""
        return float(np.mean(self.profit_scenarios(m)))

    def neg_mean_profit(self, m: np.ndarray) -> float:
        """Negative expected profit (for minimisation)."""
        return -self.mean_profit(m)

    def neg_mean_profit_gradient(self, m: np.ndarray) -> np.ndarray:
        """
        Gradient of negative expected profit w.r.t. m.

        d(-E[profit])/d(m_i) = -(1/K) * sum_k [tc_i * x_ik + (p_i - cost_i) * dx_ik/dm_i]
        """
        p = m * self.tc
        grad_sum = np.zeros_like(m)
        for dm in self.demand_models:
            x = dm.demand(m)
            dx = dm.demand_gradient(m)
            grad_sum += self.tc * x + (p - self.cost) * dx
        return -grad_sum / self.k

    def cvar(self, m: np.ndarray, alpha: float = 0.10) -> float:
        """
        Compute CVaR_alpha: mean profit in the worst alpha fraction of scenarios.

        Lower CVaR (more negative) = worse. Returns a negative number when
        scenarios have losses.

        Parameters
        ----------
        alpha:
            Tail probability (e.g. 0.10 = worst 10% of scenarios).
        """
        profits = self.profit_scenarios(m)
        profits_sorted = np.sort(profits)  # ascending: worst first
        k_tail = max(1, int(np.ceil(alpha * self.k)))
        return float(np.mean(profits_sorted[:k_tail]))
