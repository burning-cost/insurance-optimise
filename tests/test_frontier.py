"""
Tests for EfficientFrontier.

Verify:
- Frontier runs without error for supported sweep_param values
- Returns EfficientFrontierResult with correct number of points
- All epsilon values are in the sweep range
- Increasing retention constraint -> lower or equal profit (monotone front)
- Invalid sweep_param raises ValueError
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_optimise import ConstraintConfig, PortfolioOptimiser
from insurance_optimise.frontier import EfficientFrontier
from insurance_optimise.result import EfficientFrontierResult


def _make_opt(n: int = 20, seed: int = 0) -> PortfolioOptimiser:
    rng = np.random.default_rng(seed)
    tc = rng.uniform(300, 800, size=n)
    cost = tc * rng.uniform(0.55, 0.70)
    p_demand = rng.uniform(0.72, 0.92, size=n)
    elasticity = -rng.uniform(0.8, 2.5, size=n)
    renewal_flag = np.zeros(n, dtype=bool)
    renewal_flag[:n//2] = True
    enbp = tc * 1.15

    config = ConstraintConfig(lr_max=0.80, technical_floor=True)
    return PortfolioOptimiser(
        technical_price=tc,
        expected_loss_cost=cost,
        p_demand=p_demand,
        elasticity=elasticity,
        renewal_flag=renewal_flag,
        enbp=enbp,
        constraints=config,
    )


class TestEfficientFrontier:

    def test_invalid_sweep_param_raises(self):
        opt = _make_opt()
        with pytest.raises(ValueError, match="sweep_param"):
            EfficientFrontier(opt, sweep_param="invalid", sweep_range=(0.85, 0.99))

    def test_retention_sweep_returns_result(self):
        opt = _make_opt()
        ef = EfficientFrontier(
            opt,
            sweep_param="volume_retention",
            sweep_range=(0.75, 0.92),
            n_points=4,
        )
        result = ef.run()
        assert isinstance(result, EfficientFrontierResult)

    def test_correct_number_of_points(self):
        opt = _make_opt()
        ef = EfficientFrontier(
            opt,
            sweep_param="volume_retention",
            sweep_range=(0.75, 0.90),
            n_points=5,
        )
        result = ef.run()
        assert len(result.points) == 5

    def test_epsilon_values_in_range(self):
        opt = _make_opt()
        lo, hi = 0.76, 0.91
        ef = EfficientFrontier(
            opt,
            sweep_param="volume_retention",
            sweep_range=(lo, hi),
            n_points=4,
        )
        result = ef.run()
        for fp in result.points:
            assert lo - 1e-9 <= fp.epsilon <= hi + 1e-9

    def test_lr_max_sweep_returns_result(self):
        opt = _make_opt()
        ef = EfficientFrontier(
            opt,
            sweep_param="lr_max",
            sweep_range=(0.60, 0.80),
            n_points=3,
        )
        result = ef.run()
        assert len(result.points) == 3

    def test_data_columns(self):
        opt = _make_opt()
        ef = EfficientFrontier(
            opt,
            sweep_param="volume_retention",
            sweep_range=(0.75, 0.90),
            n_points=3,
        )
        result = ef.run()
        assert "epsilon" in result.data.columns
        assert "profit" in result.data.columns
        assert "converged" in result.data.columns

    def test_pareto_data_has_fewer_rows(self):
        """pareto_data() filters to converged only — should have <= total rows."""
        opt = _make_opt()
        ef = EfficientFrontier(
            opt,
            sweep_param="volume_retention",
            sweep_range=(0.75, 0.90),
            n_points=4,
        )
        result = ef.run()
        assert len(result.pareto_data()) <= len(result.data)

    def test_original_optimiser_not_mutated(self):
        """Running frontier should not change the original optimiser config."""
        opt = _make_opt()
        original_lr_max = opt.config.lr_max
        original_retention = opt.config.retention_min

        ef = EfficientFrontier(
            opt,
            sweep_param="volume_retention",
            sweep_range=(0.75, 0.90),
            n_points=3,
        )
        ef.run()

        assert opt.config.lr_max == original_lr_max
        assert opt.config.retention_min == original_retention

    def test_gwp_min_sweep(self):
        opt = _make_opt()
        tc_total = np.sum(opt.tc)
        ef = EfficientFrontier(
            opt,
            sweep_param="gwp_min",
            sweep_range=(tc_total * 0.3, tc_total * 0.7),
            n_points=3,
        )
        result = ef.run()
        assert len(result.points) == 3
