"""
OptimalPrice: Find profit-maximising prices subject to demand and regulatory constraints.

This is the thin layer between the demand models and the rate-optimiser library.
It answers the question: given a demand curve, a cost structure, and a set of
constraints, what is the optimal price?

Scope:
- Single-policy or single-segment optimisation (scalar price per segment)
- Objective: maximise expected profit per policy
- Constraints: volume floor, loss ratio ceiling, ENBP ceiling (PS21/11)
- Uses scipy.optimize for the numerical solving

For portfolio-level factor optimisation (adjusting multiplicative rating factors
across many policies simultaneously), use the ``rate-optimiser`` library directly.
That library's DemandModel interface accepts the callables from DemandCurve.as_demand_callable().

Why keep this here at all? Because before handing off to rate-optimiser, teams
often want to do simple segment-level calculations: "what's the optimal price for
PCW business in London?" without the full factor optimisation infrastructure. This
module covers that use case.

Profit model:
  E[profit | price] = P(buy | price) × (price - expected_loss - expenses)

where:
- P(buy | price) is from the demand curve
- expected_loss = technical_premium (risk model output)
- expenses = fixed cost per policy (acquisition, admin, commission)

We maximise over price subject to:
- price >= min_price
- price <= max_price (and price <= enbp if renewal)
- P(buy | price) >= min_volume_rate (volume floor as conversion rate)
- (price - expected_loss - expenses) / price >= min_margin_rate (loss ratio floor)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .demand_curve import DemandCurve


@dataclass
class OptimisationResult:
    """
    Result of a single-segment price optimisation.

    Attributes
    ----------
    optimal_price : float
        The profit-maximising price.
    expected_profit : float
        Expected profit per quote at the optimal price.
    conversion_prob : float
        P(buy) at the optimal price.
    expected_margin : float
        Margin per policy bound: (price - loss - expenses) at optimal price.
    constraints_active : list of str
        Which constraints are binding at the optimum.
    converged : bool
        Whether the optimiser converged.
    """
    optimal_price: float
    expected_profit: float
    conversion_prob: float
    expected_margin: float
    constraints_active: list[str] = field(default_factory=list)
    converged: bool = True


class OptimalPrice:
    """
    Find profit-maximising price for a single segment using a demand curve.

    Parameters
    ----------
    demand_curve : DemandCurve
        Fitted or parametric demand curve for this segment.
    expected_loss : float
        Expected claims cost per policy (technical premium).
    expense_ratio : float
        Expenses as a fraction of price (commissions, admin). Default 0.15.
    fixed_expense : float
        Fixed expenses per policy regardless of price (£). Default 0.
    min_price : float
        Hard lower bound on price. Default 50.
    max_price : float
        Hard upper bound on price. Default 10,000.
    enbp : float, optional
        Equivalent New Business Price ceiling (PS21/11 renewal constraint).
        If supplied, max_price is min(max_price, enbp).
    min_conversion_rate : float, optional
        Volume floor: P(buy) must be at least this high. Default None.
    min_margin_rate : float, optional
        Minimum required margin: (price - loss - expenses) / price >= this.
        Default None.

    Examples
    --------
    >>> from insurance_demand import DemandCurve, OptimalPrice
    >>> curve = DemandCurve(
    ...     elasticity=-2.0, base_price=500.0, base_prob=0.12,
    ...     functional_form='semi_log',
    ... )
    >>> opt = OptimalPrice(
    ...     demand_curve=curve,
    ...     expected_loss=350.0,
    ...     expense_ratio=0.15,
    ...     min_price=200.0,
    ...     max_price=900.0,
    ... )
    >>> result = opt.optimise()
    >>> print(f"Optimal price: £{result.optimal_price:.2f}")
    >>> print(f"Expected profit per quote: £{result.expected_profit:.2f}")
    """

    def __init__(
        self,
        demand_curve: DemandCurve,
        expected_loss: float,
        expense_ratio: float = 0.15,
        fixed_expense: float = 0.0,
        min_price: float = 50.0,
        max_price: float = 10_000.0,
        enbp: Optional[float] = None,
        min_conversion_rate: Optional[float] = None,
        min_margin_rate: Optional[float] = None,
    ) -> None:
        self.demand_curve = demand_curve
        self.expected_loss = expected_loss
        self.expense_ratio = expense_ratio
        self.fixed_expense = fixed_expense
        self.min_price = min_price
        self.max_price = max_price if enbp is None else min(max_price, enbp)
        self.enbp = enbp
        self.min_conversion_rate = min_conversion_rate
        self.min_margin_rate = min_margin_rate

        if self.min_price >= self.max_price:
            raise ValueError(
                f"min_price ({self.min_price}) must be less than max_price/ENBP ({self.max_price}). "
                "If ENBP is binding this segment may have no feasible prices above minimum."
            )

    # ------------------------------------------------------------------
    # Core profit function
    # ------------------------------------------------------------------

    def expected_profit_at(self, price: float) -> float:
        """
        Expected profit per quote at a given price.

        E[profit | price] = P(buy | price) × (price - expected_loss - expenses)

        Parameters
        ----------
        price : float

        Returns
        -------
        float
            Expected profit per quote (can be negative).
        """
        prices_arr = np.array([price])
        _, probs = self.demand_curve.evaluate(
            price_range=(price * 0.9999, price * 1.0001),
            n_points=1,
        )
        prob = probs[0]
        expenses = price * self.expense_ratio + self.fixed_expense
        margin = price - self.expected_loss - expenses
        return float(prob * margin)

    def _prob_at(self, price: float) -> float:
        """Get demand probability at a single price point."""
        _, probs = self.demand_curve.evaluate(
            price_range=(price * 0.9999, price * 1.0001),
            n_points=1,
        )
        return float(probs[0])

    # ------------------------------------------------------------------
    # Optimise
    # ------------------------------------------------------------------

    def optimise(self) -> OptimisationResult:
        """
        Find the profit-maximising price.

        Uses scipy.optimize.minimize_scalar (Brent's method on a bounded interval).
        Constraints are enforced by narrowing the feasible price range:
        - ENBP sets the upper bound.
        - Volume floor sets a lower bound on price (higher price → lower volume,
          so min_conversion_rate is violated above some price ceiling).
        - Margin floor sets a lower bound (below cost you have negative margin).

        Returns
        -------
        OptimisationResult
        """
        lo = self.min_price
        hi = self.max_price
        constraints_active = []

        # Enforce ENBP
        if self.enbp is not None and self.max_price == self.enbp:
            constraints_active.append("ENBP")

        # Enforce margin floor: price - loss - expenses >= min_margin_rate * price
        # => price * (1 - expense_ratio - min_margin_rate) >= expected_loss + fixed_expense
        if self.min_margin_rate is not None:
            min_price_margin = (
                (self.expected_loss + self.fixed_expense)
                / (1 - self.expense_ratio - self.min_margin_rate)
            )
            if min_price_margin > lo:
                lo = min_price_margin
                constraints_active.append("margin_floor")

        # Enforce volume floor: P(buy | price) >= min_conversion_rate
        # Since P is decreasing in price, find the price ceiling.
        if self.min_conversion_rate is not None:
            price_ceiling = self._find_price_for_prob(
                self.min_conversion_rate, lo, hi
            )
            if price_ceiling is not None and price_ceiling < hi:
                hi = price_ceiling
                constraints_active.append("volume_floor")

        if lo >= hi:
            # Feasible set is empty or degenerate - return minimum feasible price
            p_opt = lo
            return OptimisationResult(
                optimal_price=p_opt,
                expected_profit=self.expected_profit_at(p_opt),
                conversion_prob=self._prob_at(p_opt),
                expected_margin=self._margin_at(p_opt),
                constraints_active=constraints_active + ["infeasible_all_binding"],
                converged=False,
            )

        # Maximise: scipy minimises, so negate the objective
        result = minimize_scalar(
            lambda p: -self.expected_profit_at(p),
            bounds=(lo, hi),
            method="bounded",
        )

        p_opt = float(result.x)
        return OptimisationResult(
            optimal_price=p_opt,
            expected_profit=self.expected_profit_at(p_opt),
            conversion_prob=self._prob_at(p_opt),
            expected_margin=self._margin_at(p_opt),
            constraints_active=constraints_active,
            converged=result.success,
        )

    def _margin_at(self, price: float) -> float:
        """Margin per policy: price - loss - expenses."""
        expenses = price * self.expense_ratio + self.fixed_expense
        return price - self.expected_loss - expenses

    def _find_price_for_prob(
        self,
        target_prob: float,
        lo: float,
        hi: float,
    ) -> Optional[float]:
        """
        Find the price at which demand probability equals target_prob.
        Returns None if the target is not achievable within [lo, hi].
        """
        # If parametric curve, use analytic inversion
        if self.demand_curve.functional_form in ("log_linear", "semi_log"):
            try:
                return self.demand_curve.price_at_prob(target_prob)
            except Exception:
                pass

        # Numerical bisection otherwise
        p_lo = self._prob_at(lo)
        p_hi = self._prob_at(hi)

        if target_prob > p_lo:
            return None  # even at min price we're below target - infeasible
        if target_prob < p_hi:
            return hi  # all prices are feasible

        # Bisect
        for _ in range(50):
            mid = (lo + hi) / 2
            p_mid = self._prob_at(mid)
            if abs(p_mid - target_prob) < 1e-6:
                return mid
            if p_mid > target_prob:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2

    # ------------------------------------------------------------------
    # Grid evaluation (for inspection)
    # ------------------------------------------------------------------

    def profit_curve(
        self,
        price_range: Optional[tuple[float, float]] = None,
        n_points: int = 100,
    ) -> "pd.DataFrame":
        """
        Evaluate expected profit over a price grid.

        Parameters
        ----------
        price_range : tuple, optional
            (min_price, max_price). Defaults to (self.min_price, self.max_price).
        n_points : int

        Returns
        -------
        pd.DataFrame
            Columns: price, conversion_prob, margin, expected_profit.
        """
        import pandas as pd

        if price_range is None:
            price_range = (self.min_price, self.max_price)

        prices_arr = np.linspace(price_range[0], price_range[1], n_points)
        _, probs = self.demand_curve.evaluate(price_range, n_points)
        margins = prices_arr - self.expected_loss - prices_arr * self.expense_ratio - self.fixed_expense
        profits = probs * margins

        return pd.DataFrame({
            "price": prices_arr,
            "conversion_prob": probs,
            "margin": margins,
            "expected_profit": profits,
        })
