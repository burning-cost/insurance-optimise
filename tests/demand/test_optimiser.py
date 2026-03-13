"""Tests for OptimalPrice."""

import numpy as np
import pytest

from insurance_optimise.demand import DemandCurve, OptimalPrice, OptimisationResult


def make_curve(elasticity=-2.0, base_price=500.0, base_prob=0.12):
    return DemandCurve(
        elasticity=elasticity,
        base_price=base_price,
        base_prob=base_prob,
        functional_form="semi_log",
    )


class TestOptimalPrice:
    def setup_method(self):
        self.curve = make_curve()
        self.opt = OptimalPrice(
            demand_curve=self.curve,
            expected_loss=350.0,
            expense_ratio=0.15,
            min_price=200.0,
            max_price=900.0,
        )

    def test_optimise_returns_result(self):
        result = self.opt.optimise()
        assert isinstance(result, OptimisationResult)

    def test_optimal_price_in_bounds(self):
        result = self.opt.optimise()
        assert 200.0 <= result.optimal_price <= 900.0

    def test_conversion_prob_in_range(self):
        result = self.opt.optimise()
        assert 0 < result.conversion_prob < 1

    def test_expected_profit_at_optimal(self):
        result = self.opt.optimise()
        # At optimal, expected profit should be positive (above break-even)
        assert result.expected_profit > 0

    def test_profit_curve_returns_dataframe(self):
        import pandas as pd
        df = self.opt.profit_curve(n_points=20)
        assert isinstance(df, pd.DataFrame)
        assert "price" in df.columns
        assert "expected_profit" in df.columns
        assert len(df) == 20

    def test_profit_is_concave(self):
        """Expected profit should peak in the interior, not at a boundary."""
        df = self.opt.profit_curve(n_points=100)
        max_idx = df["expected_profit"].idxmax()
        # Optimal should not be at the extreme boundaries
        assert 5 < max_idx < 95, "Profit peak seems to be at a boundary"

    def test_enbp_constraint_applied(self):
        opt = OptimalPrice(
            demand_curve=self.curve,
            expected_loss=350.0,
            expense_ratio=0.15,
            min_price=200.0,
            max_price=900.0,
            enbp=450.0,  # tight ENBP constraint
        )
        result = opt.optimise()
        assert result.optimal_price <= 450.0 + 1e-6
        assert "ENBP" in result.constraints_active

    def test_volume_floor_constraint(self):
        opt = OptimalPrice(
            demand_curve=self.curve,
            expected_loss=350.0,
            expense_ratio=0.15,
            min_price=200.0,
            max_price=900.0,
            min_conversion_rate=0.15,  # require at least 15% conversion
        )
        result = opt.optimise()
        # Conversion rate should meet the floor
        assert result.conversion_prob >= 0.15 - 0.01  # small tolerance for numerics

    def test_infeasible_min_price_too_high(self):
        """If min_price >= max_price (e.g. ENBP binding below min_price), should not crash."""
        with pytest.raises(ValueError, match="min_price"):
            OptimalPrice(
                demand_curve=self.curve,
                expected_loss=350.0,
                min_price=600.0,
                max_price=400.0,  # below min_price
            )

    def test_expected_profit_at(self):
        profit = self.opt.expected_profit_at(500.0)
        assert isinstance(profit, float)

    def test_higher_loss_shifts_optimal_price_up(self):
        """With higher expected loss, the optimal price should be higher."""
        result_lo = OptimalPrice(
            demand_curve=self.curve,
            expected_loss=300.0,
            expense_ratio=0.15,
            min_price=200.0,
            max_price=900.0,
        ).optimise()
        result_hi = OptimalPrice(
            demand_curve=self.curve,
            expected_loss=450.0,
            expense_ratio=0.15,
            min_price=200.0,
            max_price=900.0,
        ).optimise()
        assert result_hi.optimal_price > result_lo.optimal_price


class TestOptimalPriceMarginFloor:
    def test_margin_floor_applied(self):
        curve = make_curve()
        opt = OptimalPrice(
            demand_curve=curve,
            expected_loss=350.0,
            expense_ratio=0.15,
            min_price=200.0,
            max_price=900.0,
            min_margin_rate=0.10,  # 10% profit margin minimum
        )
        result = opt.optimise()
        margin = result.optimal_price - 350.0 - result.optimal_price * 0.15
        margin_rate = margin / result.optimal_price
        assert margin_rate >= 0.10 - 0.01
