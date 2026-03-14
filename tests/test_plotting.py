"""
Tests for the plotting module (ported from rate-optimiser).

We skip all tests if matplotlib is not installed — it's an optional dependency.
Tests verify that plotting functions return Axes and don't crash on typical
inputs; they do not check pixel-level rendering.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="matplotlib not installed")


from insurance_optimise.plotting import (
    plot_frontier,
    plot_factor_adjustments,
    plot_shadow_prices,
)


@pytest.fixture
def frontier_result(small_portfolio):
    """Run a quick frontier sweep to get an EfficientFrontierResult."""
    from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier

    p = small_portfolio
    config = ConstraintConfig(
        lr_max=0.75,
        technical_floor=False,
        min_multiplier=0.5,
    )
    opt = PortfolioOptimiser(
        technical_price=p["technical_price"],
        expected_loss_cost=p["expected_loss_cost"],
        p_demand=p["p_demand"],
        elasticity=p["elasticity"],
        renewal_flag=p["renewal_flag"],
        constraints=config,
    )
    frontier = EfficientFrontier(
        optimiser=opt,
        sweep_param="lr_max",
        sweep_range=(0.70, 0.85),
        n_points=3,
    )
    return frontier.run()


class TestPlotFrontier:
    def test_returns_axes(self, frontier_result):
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_frontier(frontier_result)
        assert ax is not None

    def test_accepts_existing_axes(self, frontier_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        returned_ax = plot_frontier(frontier_result, ax=ax)
        assert returned_ax is ax

    def test_custom_metrics(self, frontier_result):
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_frontier(frontier_result, x_metric="epsilon", y_metric="profit")
        assert ax is not None

    def test_custom_title(self, frontier_result):
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_frontier(frontier_result, title="My Custom Title")
        assert ax.get_title() == "My Custom Title"


class TestPlotFactorAdjustments:
    def test_returns_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        multipliers = np.array([1.05, 0.95, 1.10, 1.02, 0.98])
        ax = plot_factor_adjustments(multipliers)
        assert ax is not None

    def test_accepts_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        multipliers = np.array([1.05, 0.95, 1.10])
        labels = ["Age 20-25", "NCB 3+", "London"]
        ax = plot_factor_adjustments(multipliers, labels=labels)
        assert ax is not None

    def test_accepts_existing_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        multipliers = np.ones(5) * 1.05
        returned = plot_factor_adjustments(multipliers, ax=ax)
        assert returned is ax

    def test_single_policy(self):
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_factor_adjustments(np.array([1.0]))
        assert ax is not None


class TestPlotShadowPrices:
    def test_returns_axes(self, frontier_result):
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_shadow_prices(frontier_result)
        assert ax is not None

    def test_accepts_existing_axes(self, frontier_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        returned = plot_shadow_prices(frontier_result, ax=ax)
        assert returned is ax
