"""
Tests for pareto.py: 3-objective Pareto surface utilities.

Covers:
- premium_disparity_ratio: basic cases, single group, zero group mean
- loss_ratio_disparity: group-level LR disparity with mock demand model
- _filter_pareto_front: non-dominated filtering correctness
- _topsis_select: selection from Pareto surface
- ParetoResult.summary(): schema and statistics
- ParetoResult.select(): TOPSIS selection, empty surface error
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from insurance_optimise.pareto import (
    ParetoResult,
    _filter_pareto_front,
    _topsis_select,
    loss_ratio_disparity,
    premium_disparity_ratio,
)
from insurance_optimise.result import OptimisationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pareto_result(
    n: int = 10,
    seed: int = 42,
    all_converged: bool = True,
) -> ParetoResult:
    """Build a synthetic ParetoResult with n grid points."""
    rng = np.random.default_rng(seed)

    # Staircase Pareto: profit and fairness (lower=better) trade off
    profits = np.linspace(8000, 15000, n)
    fairness = np.linspace(1.6, 1.0, n)  # decreasing (more fair) as profit falls
    retention = rng.uniform(0.85, 0.92, size=n)

    surface = pl.DataFrame(
        {
            "profit": profits.tolist(),
            "fairness": fairness.tolist(),
            "retention": retention.tolist(),
            "gwp": (profits * 1.2).tolist(),
            "loss_ratio": rng.uniform(0.55, 0.75, size=n).tolist(),
            "converged": [all_converged] * n,
            "eps_x": np.linspace(0.9, 1.5, n).tolist(),
            "eps_y": np.linspace(1.0, 2.0, n).tolist(),
            "n_iter": [50] * n,
            "solver_message": ["Optimization terminated successfully"] * n,
            "grid_i": list(range(n)),
            "grid_j": [0] * n,
        }
    )
    # All points are on the Pareto front (staircase)
    pareto_df = surface.clone()
    grid_multipliers = {
        (i, 0): np.ones(100) * (1.0 + 0.01 * i) for i in range(n)
    }
    return ParetoResult(
        surface=surface,
        pareto_df=pareto_df,
        metadata={
            "n_points_x": n,
            "n_points_y": 1,
            "fairness_metric_name": "premium_disparity_ratio",
        },
        _grid_multipliers=grid_multipliers,
    )


# ---------------------------------------------------------------------------
# premium_disparity_ratio
# ---------------------------------------------------------------------------


class TestPremiumDisparityRatio:

    def test_equal_groups_returns_one(self):
        """Equal mean premiums across groups => ratio = 1.0."""
        multipliers = np.array([1.0, 1.0, 1.0, 1.0])
        technical_price = np.array([500.0, 500.0, 500.0, 500.0])
        group_labels = np.array([1, 1, 2, 2])
        ratio = premium_disparity_ratio(multipliers, technical_price, group_labels)
        assert abs(ratio - 1.0) < 1e-9

    def test_known_disparity(self):
        """Group 1 pays double group 2 on average => ratio = 2.0."""
        multipliers = np.ones(4)
        technical_price = np.array([1000.0, 1000.0, 500.0, 500.0])
        group_labels = np.array([1, 1, 2, 2])
        ratio = premium_disparity_ratio(multipliers, technical_price, group_labels)
        assert abs(ratio - 2.0) < 1e-9

    def test_multiplier_scales_disparity(self):
        """Uniform multiplier scales premiums but not the ratio."""
        multipliers = np.array([2.0, 2.0, 2.0, 2.0])
        technical_price = np.array([1000.0, 1000.0, 500.0, 500.0])
        group_labels = np.array([1, 1, 2, 2])
        ratio = premium_disparity_ratio(multipliers, technical_price, group_labels)
        assert abs(ratio - 2.0) < 1e-9

    def test_single_group_returns_one(self):
        """Only one group => no inter-group disparity => ratio = 1.0."""
        multipliers = np.ones(5)
        technical_price = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        group_labels = np.array([1, 1, 1, 1, 1])
        ratio = premium_disparity_ratio(multipliers, technical_price, group_labels)
        assert abs(ratio - 1.0) < 1e-9

    def test_zero_min_mean_returns_one(self):
        """If any group mean is near zero, return 1.0 to avoid division by zero."""
        multipliers = np.array([0.0, 0.0, 1.0, 1.0])
        technical_price = np.array([100.0, 100.0, 100.0, 100.0])
        group_labels = np.array([1, 1, 2, 2])
        ratio = premium_disparity_ratio(multipliers, technical_price, group_labels)
        assert abs(ratio - 1.0) < 1e-9

    def test_ratio_is_always_ge_one(self):
        """Ratio is defined as max/min, so always >= 1.0."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            mult = rng.uniform(0.5, 2.0, size=50)
            tc = rng.uniform(100, 1000, size=50)
            labels = rng.integers(1, 5, size=50)
            ratio = premium_disparity_ratio(mult, tc, labels)
            assert ratio >= 1.0 - 1e-9

    def test_three_groups(self):
        """Ratio with three groups uses the outermost pair."""
        # Group means: 100, 200, 400 => ratio = 400/100 = 4.0
        multipliers = np.ones(6)
        technical_price = np.array([100.0, 100.0, 200.0, 200.0, 400.0, 400.0])
        group_labels = np.array([1, 1, 2, 2, 3, 3])
        ratio = premium_disparity_ratio(multipliers, technical_price, group_labels)
        assert abs(ratio - 4.0) < 1e-9


# ---------------------------------------------------------------------------
# loss_ratio_disparity
# ---------------------------------------------------------------------------


class TestLossRatioDisparity:

    def _mock_demand(self, values: np.ndarray):
        """Return a mock demand model that returns fixed values."""
        model = MagicMock()
        model.demand.return_value = values
        return model

    def test_equal_groups_returns_one(self):
        """When LRs are identical across groups, disparity = 1.0."""
        n = 6
        multipliers = np.ones(n)
        technical_price = np.full(n, 100.0)
        expected_loss = np.full(n, 60.0)  # LR = 60/100 = 0.6 for all
        demand = self._mock_demand(np.ones(n))  # no demand effect
        group_labels = np.array([1, 1, 2, 2, 3, 3])
        ratio = loss_ratio_disparity(
            multipliers, technical_price, expected_loss, demand, group_labels
        )
        assert abs(ratio - 1.0) < 1e-6

    def test_known_disparity(self):
        """Two groups: one has LR=0.6, other has LR=1.2 => ratio = 2.0."""
        multipliers = np.ones(4)
        technical_price = np.array([100.0, 100.0, 100.0, 100.0])
        # Group 1: loss = 60, Group 2: loss = 120
        expected_loss = np.array([60.0, 60.0, 120.0, 120.0])
        demand = self._mock_demand(np.ones(4))
        group_labels = np.array([1, 1, 2, 2])
        ratio = loss_ratio_disparity(
            multipliers, technical_price, expected_loss, demand, group_labels
        )
        assert abs(ratio - 2.0) < 1e-6

    def test_single_group_returns_one(self):
        """Only one group => ratio = 1.0."""
        multipliers = np.ones(4)
        technical_price = np.full(4, 100.0)
        expected_loss = np.array([60.0, 70.0, 80.0, 50.0])
        demand = self._mock_demand(np.ones(4))
        group_labels = np.array([1, 1, 1, 1])
        ratio = loss_ratio_disparity(
            multipliers, technical_price, expected_loss, demand, group_labels
        )
        assert abs(ratio - 1.0) < 1e-9

    def test_ratio_ge_one(self):
        """Loss ratio disparity is always >= 1.0."""
        rng = np.random.default_rng(1)
        n = 20
        multipliers = rng.uniform(0.8, 1.5, size=n)
        tc = rng.uniform(100, 500, size=n)
        loss = rng.uniform(50, 400, size=n)
        demand = self._mock_demand(rng.uniform(0.3, 1.0, size=n))
        labels = rng.integers(1, 4, size=n)
        ratio = loss_ratio_disparity(multipliers, tc, loss, demand, labels)
        assert ratio >= 1.0 - 1e-9


# ---------------------------------------------------------------------------
# _filter_pareto_front
# ---------------------------------------------------------------------------


class TestFilterParetoFront:

    def _make_df(self, profit, retention, fairness, converged=None):
        n = len(profit)
        if converged is None:
            converged = [True] * n
        return pl.DataFrame(
            {
                "profit": profit,
                "retention": retention,
                "fairness": fairness,
                "converged": converged,
                "eps_x": [1.0] * n,
                "eps_y": [1.0] * n,
                "gwp": [10000.0] * n,
                "loss_ratio": [0.6] * n,
                "n_iter": [50] * n,
                "solver_message": ["ok"] * n,
                "grid_i": list(range(n)),
                "grid_j": [0] * n,
            }
        )

    def test_dominated_point_removed(self):
        """A point dominated by another on all objectives is filtered out."""
        # Point 0: (profit=100, retention=0.9, fairness=1.5) dominates point 1
        # Point 2: (profit=50, retention=0.85, fairness=1.3) - distinct tradeoff
        df = self._make_df(
            profit=[100.0, 80.0, 50.0],
            retention=[0.90, 0.85, 0.85],
            fairness=[1.5, 1.6, 1.3],  # lower = better
        )
        # Point 1 is dominated by Point 0 (higher profit, higher retention, lower fairness)
        result = _filter_pareto_front(df)
        assert len(result) <= 2
        assert 80.0 not in result["profit"].to_list()

    def test_all_pareto_optimal_staircase(self):
        """Trade-off staircase: no point dominates another."""
        df = self._make_df(
            profit=[50.0, 75.0, 100.0],
            retention=[0.95, 0.90, 0.85],
            fairness=[1.1, 1.3, 1.5],
        )
        result = _filter_pareto_front(df)
        assert len(result) == 3

    def test_empty_dataframe_returns_empty(self):
        df = self._make_df(profit=[], retention=[], fairness=[])
        result = _filter_pareto_front(df)
        assert len(result) == 0

    def test_single_point_returns_itself(self):
        df = self._make_df(profit=[100.0], retention=[0.9], fairness=[1.2])
        result = _filter_pareto_front(df)
        assert len(result) == 1

    def test_identical_points_all_returned(self):
        """Identical solutions: none strictly dominates another."""
        df = self._make_df(
            profit=[100.0, 100.0, 100.0],
            retention=[0.9, 0.9, 0.9],
            fairness=[1.2, 1.2, 1.2],
        )
        result = _filter_pareto_front(df)
        assert len(result) == 3

    def test_clear_domination_chain(self):
        """Only the best point survives when all others are dominated."""
        df = self._make_df(
            profit=[100.0, 80.0, 60.0, 40.0],
            retention=[0.95, 0.90, 0.85, 0.80],
            fairness=[1.1, 1.2, 1.3, 1.4],
        )
        result = _filter_pareto_front(df)
        assert len(result) == 1
        assert result["profit"][0] == 100.0


# ---------------------------------------------------------------------------
# _topsis_select
# ---------------------------------------------------------------------------


class TestTopsisSelect:

    def _make_pareto_df(self, n: int = 5, seed: int = 0) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        return pl.DataFrame(
            {
                "profit": np.linspace(8000, 15000, n).tolist(),
                "retention": rng.uniform(0.85, 0.95, size=n).tolist(),
                "fairness": np.linspace(1.5, 1.0, n).tolist(),
                "gwp": (np.linspace(8000, 15000, n) * 1.2).tolist(),
                "loss_ratio": rng.uniform(0.5, 0.7, size=n).tolist(),
                "converged": [True] * n,
                "eps_x": [1.0] * n,
                "eps_y": [1.0] * n,
                "n_iter": [50] * n,
                "solver_message": ["ok"] * n,
                "grid_i": list(range(n)),
                "grid_j": [0] * n,
            }
        )

    def test_returns_valid_index(self):
        df = self._make_pareto_df(5)
        idx = _topsis_select(df, weights=(0.5, 0.3, 0.2))
        assert 0 <= idx < 5

    def test_profit_weight_one_selects_highest_profit(self):
        """With all weight on profit, should select highest-profit solution."""
        df = self._make_pareto_df(5)
        idx = _topsis_select(df, weights=(1.0, 0.0, 0.0))
        # Highest profit is last row in our staircase
        assert df["profit"][idx] == df["profit"].max()

    def test_fairness_weight_one_selects_most_fair(self):
        """With all weight on fairness (min), should select lowest fairness value."""
        df = self._make_pareto_df(5)
        idx = _topsis_select(df, weights=(0.0, 0.0, 1.0))
        # Fairness decreases from first to last row; lowest = last = most fair
        assert df["fairness"][idx] == df["fairness"].min()

    def test_empty_df_raises(self):
        df = pl.DataFrame(
            {"profit": [], "retention": [], "fairness": [], "gwp": [],
             "loss_ratio": [], "converged": [], "eps_x": [], "eps_y": [],
             "n_iter": [], "solver_message": [], "grid_i": [], "grid_j": []}
        )
        with pytest.raises(ValueError, match="empty"):
            _topsis_select(df, weights=(0.5, 0.3, 0.2))

    def test_zero_weights_raises(self):
        df = self._make_pareto_df(3)
        with pytest.raises(ValueError, match="weights must not all be zero"):
            _topsis_select(df, weights=(0.0, 0.0, 0.0))

    def test_single_row_returns_zero(self):
        df = self._make_pareto_df(1)
        idx = _topsis_select(df, weights=(0.5, 0.3, 0.2))
        assert idx == 0

    def test_custom_objectives(self):
        """Verify custom objective column names work."""
        df = pl.DataFrame(
            {
                "p": [8000.0, 12000.0, 15000.0],
                "r": [0.95, 0.90, 0.85],
                "f": [1.1, 1.3, 1.5],
            }
        )
        idx = _topsis_select(
            df,
            weights=(1.0, 0.0, 0.0),
            objectives=("p", "r", "f"),
        )
        assert df["p"][idx] == 15000.0


# ---------------------------------------------------------------------------
# ParetoResult.summary()
# ---------------------------------------------------------------------------


class TestParetoResultSummary:

    def test_summary_returns_dataframe(self):
        pr = _make_pareto_result(n=8)
        s = pr.summary()
        assert isinstance(s, pl.DataFrame)

    def test_summary_has_expected_metrics(self):
        pr = _make_pareto_result(n=8)
        s = pr.summary()
        metrics = s["metric"].to_list()
        for expected in [
            "grid_points_total",
            "grid_points_converged",
            "pareto_optimal_solutions",
            "profit_min",
            "profit_max",
            "retention_min",
            "retention_max",
            "fairness_disparity_min",
            "fairness_disparity_max",
        ]:
            assert expected in metrics, f"Missing metric: {expected}"

    def test_total_equals_n(self):
        pr = _make_pareto_result(n=10)
        s = pr.summary()
        total = s.filter(pl.col("metric") == "grid_points_total")["value"][0]
        assert total == 10.0

    def test_profit_range_correct(self):
        pr = _make_pareto_result(n=10)
        s = pr.summary()
        profit_min = s.filter(pl.col("metric") == "profit_min")["value"][0]
        profit_max = s.filter(pl.col("metric") == "profit_max")["value"][0]
        assert profit_min < profit_max

    def test_empty_pareto_returns_nans(self):
        """If pareto_df is empty, ranges should be NaN."""
        pr = _make_pareto_result(n=5)
        # Replace pareto_df with empty
        pr = ParetoResult(
            surface=pr.surface,
            pareto_df=pr.surface.head(0),  # empty
            metadata=pr.metadata,
            _grid_multipliers=pr._grid_multipliers,
        )
        s = pr.summary()
        profit_min = s.filter(pl.col("metric") == "profit_min")["value"][0]
        import math
        assert math.isnan(profit_min)


# ---------------------------------------------------------------------------
# ParetoResult.select()
# ---------------------------------------------------------------------------


class TestParetoResultSelect:

    def test_select_populates_selected(self):
        pr = _make_pareto_result(n=8)
        pr.select(method="topsis", weights=(0.5, 0.3, 0.2))
        assert pr.selected is not None

    def test_select_returns_self(self):
        pr = _make_pareto_result(n=8)
        returned = pr.select()
        assert returned is pr

    def test_selected_is_optimisation_result(self):
        pr = _make_pareto_result(n=8)
        pr.select()
        assert isinstance(pr.selected, OptimisationResult)

    def test_selected_audit_trail_has_method(self):
        pr = _make_pareto_result(n=8)
        pr.select(method="topsis")
        assert pr.selected.audit_trail["selected_by"] == "topsis"

    def test_select_closest_to_utopia(self):
        """closest_to_utopia is an alias for topsis — should not raise."""
        pr = _make_pareto_result(n=8)
        pr.select(method="closest_to_utopia")
        assert pr.selected is not None

    def test_select_invalid_method_raises(self):
        pr = _make_pareto_result(n=8)
        with pytest.raises(ValueError, match="Unknown selection method"):
            pr.select(method="magic")

    def test_select_empty_pareto_raises(self):
        pr = _make_pareto_result(n=5)
        pr = ParetoResult(
            surface=pr.surface,
            pareto_df=pr.surface.head(0),
            metadata=pr.metadata,
            _grid_multipliers=pr._grid_multipliers,
        )
        with pytest.raises(ValueError, match="empty"):
            pr.select()

    def test_select_profit_weight_gives_high_profit_solution(self):
        """Strong profit weighting selects a high-profit solution."""
        pr = _make_pareto_result(n=10)
        pr.select(method="topsis", weights=(1.0, 0.0, 0.0))
        assert pr.selected.expected_profit > 12000.0  # well above midpoint
