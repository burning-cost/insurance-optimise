"""
Tests for scenario-based optimisation and ScenarioObjective.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_optimise import ConstraintConfig, PortfolioOptimiser
from insurance_optimise.result import ScenarioResult
from insurance_optimise.scenarios import ScenarioObjective


def _make_basic_portfolio(n: int = 15, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    tc = rng.uniform(300, 700, size=n)
    cost = tc * rng.uniform(0.55, 0.70)
    p_demand = rng.uniform(0.70, 0.90, size=n)
    elasticity = -rng.uniform(0.5, 2.5, size=n)
    renewal_flag = np.zeros(n, dtype=bool)
    renewal_flag[:n//2] = True
    enbp = tc * 1.15
    return dict(
        technical_price=tc,
        expected_loss_cost=cost,
        p_demand=p_demand,
        elasticity=elasticity,
        renewal_flag=renewal_flag,
        enbp=enbp,
    )


class TestScenarioObjective:

    def _make_scenario_obj(self, n: int = 8, k: int = 3) -> ScenarioObjective:
        rng = np.random.default_rng(9)
        tc = rng.uniform(400, 700, size=n)
        cost = tc * 0.65
        elasticity_central = -rng.uniform(1.0, 2.0, size=n)
        elasticity_scenarios = [
            elasticity_central * 0.7,   # optimistic (less elastic)
            elasticity_central,          # central
            elasticity_central * 1.3,   # pessimistic (more elastic)
        ]
        x0 = [np.full(n, 0.82)] * k
        return ScenarioObjective(
            technical_price=tc,
            expected_loss_cost=cost,
            x0_scenarios=x0,
            elasticity_scenarios=elasticity_scenarios,
        )

    def test_profit_scenarios_shape(self):
        obj = self._make_scenario_obj(k=3)
        m = np.ones(8)
        profits = obj.profit_scenarios(m)
        assert profits.shape == (3,)

    def test_mean_profit_scalar(self):
        obj = self._make_scenario_obj(k=3)
        m = np.ones(8)
        mp = obj.mean_profit(m)
        assert isinstance(mp, float)

    def test_neg_mean_profit_negates(self):
        obj = self._make_scenario_obj(k=3)
        m = np.ones(8)
        assert obj.neg_mean_profit(m) == pytest.approx(-obj.mean_profit(m))

    def test_gradient_finite_difference(self):
        """neg_mean_profit_gradient matches finite-difference."""
        obj = self._make_scenario_obj(n=5, k=3)
        rng = np.random.default_rng(77)
        m = rng.uniform(0.9, 1.2, size=5)
        analytical = obj.neg_mean_profit_gradient(m)
        eps = 1e-6
        fd = np.zeros(5)
        for i in range(5):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd[i] = (obj.neg_mean_profit(m_p) - obj.neg_mean_profit(m_m)) / (2 * eps)
        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_cvar_is_worst_fraction(self):
        """CVaR at alpha should equal mean of worst alpha fraction of scenarios."""
        # Use K=10 scenarios with known profits
        n = 4
        k = 10
        rng = np.random.default_rng(11)
        tc = np.ones(n) * 500
        cost = np.ones(n) * 300
        m = np.ones(n)
        # Elasticity scenarios: vary to get spread of profits
        elasticity_scenarios = [np.full(n, -e) for e in np.linspace(0.5, 2.0, k)]
        x0_s = [np.full(n, 0.8)] * k
        obj = ScenarioObjective(
            technical_price=tc,
            expected_loss_cost=cost,
            x0_scenarios=x0_s,
            elasticity_scenarios=elasticity_scenarios,
        )
        alpha = 0.30
        cvar = obj.cvar(m, alpha=alpha)
        profits = obj.profit_scenarios(m)
        k_tail = max(1, int(np.ceil(alpha * k)))
        expected_cvar = np.mean(np.sort(profits)[:k_tail])
        assert cvar == pytest.approx(expected_cvar, rel=1e-6)

    def test_x0_scenarios_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            ScenarioObjective(
                technical_price=np.ones(5) * 500,
                expected_loss_cost=np.ones(5) * 300,
                x0_scenarios=None,
                elasticity_scenarios=[np.full(5, -1.0)],
            )

    def test_mismatched_x0_elasticity_raises(self):
        with pytest.raises(ValueError, match="x0_scenarios has"):
            ScenarioObjective(
                technical_price=np.ones(5) * 500,
                expected_loss_cost=np.ones(5) * 300,
                x0_scenarios=[np.full(5, 0.8)] * 2,
                elasticity_scenarios=[np.full(5, -1.0)] * 3,  # length mismatch
            )


class TestOptimiserScenarioMode:

    def test_optimise_scenarios_returns_scenario_result(self):
        p = _make_basic_portfolio(n=15, seed=5)
        opt = PortfolioOptimiser(
            **p,
            constraints=ConstraintConfig(lr_max=0.72, technical_floor=True),
        )
        elasticity_scenarios = [
            p["elasticity"] * 0.8,   # less elastic
            p["elasticity"],          # central
            p["elasticity"] * 1.2,   # more elastic
        ]
        sr = opt.optimise_scenarios(
            elasticity_scenarios=elasticity_scenarios,
            scenario_names=["pessimistic", "central", "optimistic"],
        )
        assert isinstance(sr, ScenarioResult)

    def test_scenario_result_has_correct_count(self):
        p = _make_basic_portfolio(n=15, seed=6)
        opt = PortfolioOptimiser(**p)
        elasticity_scenarios = [p["elasticity"], p["elasticity"] * 1.1]
        sr = opt.optimise_scenarios(elasticity_scenarios=elasticity_scenarios)
        assert len(sr.results) == 2

    def test_default_scenario_names(self):
        p = _make_basic_portfolio(n=10, seed=7)
        opt = PortfolioOptimiser(**p)
        sr = opt.optimise_scenarios(
            elasticity_scenarios=[p["elasticity"], p["elasticity"] * 1.1]
        )
        assert sr.scenario_names == ["scenario_0", "scenario_1"]

    def test_scenario_summary_dataframe(self):
        p = _make_basic_portfolio(n=12, seed=8)
        opt = PortfolioOptimiser(**p)
        sr = opt.optimise_scenarios(
            elasticity_scenarios=[p["elasticity"]],
            scenario_names=["base"],
        )
        df = sr.summary()
        assert len(df) == 1
        assert "scenario" in df.columns

    def test_multiplier_ci_shapes(self):
        p = _make_basic_portfolio(n=12, seed=9)
        opt = PortfolioOptimiser(**p)
        sr = opt.optimise_scenarios(
            elasticity_scenarios=[p["elasticity"] * f for f in [0.8, 1.0, 1.2]]
        )
        assert sr.multiplier_mean.shape == (opt.n,)
        assert sr.multiplier_p10.shape == (opt.n,)
        assert sr.multiplier_p90.shape == (opt.n,)

    def test_p10_le_mean_le_p90_profit(self):
        """Profit distribution percentiles should be ordered."""
        p = _make_basic_portfolio(n=12, seed=10)
        opt = PortfolioOptimiser(**p)
        sr = opt.optimise_scenarios(
            elasticity_scenarios=[p["elasticity"] * f for f in [0.7, 0.9, 1.0, 1.1, 1.3]]
        )
        assert sr.profit_p10 <= sr.profit_mean + 1.0
        assert sr.profit_mean <= sr.profit_p90 + 1.0
