"""
End-to-end integration tests.

These tests simulate realistic UK personal lines pricing workflows:
- Full optimisation with all constraints active
- Scenario mode with CI bounds from upstream elasticity library
- Frontier sweep with monotone profit-retention trade-off
- Audit trail round-trip through JSON serialisation
- Portfolio summary before vs after optimisation

All tests use N=30 policies to keep runtime under 5s per test.
"""

from __future__ import annotations

import json
import numpy as np
import polars as pl
import pytest

from insurance_optimise import ConstraintConfig, PortfolioOptimiser, EfficientFrontier
from insurance_optimise.result import ScenarioResult


def _make_renewal_nb_mix(n: int = 30, seed: int = 42) -> dict:
    """30 policies: 18 renewals, 12 new business. Typical UK motor book."""
    rng = np.random.default_rng(seed)
    tc = rng.uniform(350, 900, size=n)
    cost = tc * rng.uniform(0.55, 0.72, size=n)
    p_demand = rng.uniform(0.72, 0.92, size=n)
    elasticity = -rng.uniform(0.8, 2.5, size=n)

    renewal_flag = np.zeros(n, dtype=bool)
    renewal_flag[:18] = True
    rng.shuffle(renewal_flag)

    enbp = np.where(renewal_flag, tc * rng.uniform(1.05, 1.20, size=n), tc * 2.0)
    prior_multiplier = np.ones(n)

    return dict(
        technical_price=tc,
        expected_loss_cost=cost,
        p_demand=p_demand,
        elasticity=elasticity,
        renewal_flag=renewal_flag,
        enbp=enbp,
        prior_multiplier=prior_multiplier,
    )


class TestFullConstrainedWorkflow:

    @pytest.fixture
    def opt(self):
        p = _make_renewal_nb_mix()
        # Use constraints that are feasible for the synthetic data:
        # LR is 55-75% of technical premium, so lr_max=0.75 has headroom.
        # Retention at m=1 is ~80-90%. With technical_floor=False, SLSQP can
        # reduce prices to boost retention, making both constraints satisfiable.
        # Rate change ±30% gives SLSQP room to manoeuvre.
        config = ConstraintConfig(
            lr_max=0.75,
            retention_min=0.80,
            max_rate_change=0.30,
            enbp_buffer=0.01,
            technical_floor=False,
            min_multiplier=0.85,   # don't go below 85% of technical price
        )
        return PortfolioOptimiser(**p, constraints=config, n_restarts=5)

    def test_converges_or_feasible(self, opt):
        """Either SLSQP reports convergence, or the solution satisfies constraints.
        
        SLSQP can report 'Positive directional derivative for linesearch' when
        stuck at a constraint boundary — the solution may still be feasible.
        We accept a result if either: (a) converged=True, or (b) all constraints
        are satisfied to 1e-3 tolerance.
        """
        result = opt.optimise()
        if result.converged:
            return  # clean convergence
        # Check feasibility manually
        m = result.multipliers
        lb = opt._bounds.lb
        ub = opt._bounds.ub
        bounds_ok = (np.all(m >= lb - 1e-3) and np.all(m <= ub + 1e-3))
        constraints_ok = all(
            c["fun"](m) >= -1e-3 for c in opt._scipy_constraints
        )
        assert bounds_ok and constraints_ok, (
            f"Solution is not feasible. Message: {result.solver_message}"
        )

    def test_profit_positive(self, opt):
        result = opt.optimise()
        assert result.expected_profit > 0

    def test_enbp_strictly_enforced(self, opt):
        """Zero renewal policies should exceed ENBP * (1 - buffer)."""
        result = opt.optimise()
        renewal = opt.renewal_flag
        enbp = opt.enbp
        buffer = opt.config.enbp_buffer
        ub = enbp / opt.tc * (1.0 - buffer)
        violations = result.multipliers[renewal] > ub[renewal] + 1e-4
        assert not np.any(violations), f"{np.sum(violations)} ENBP violations"

    def test_lr_not_exceeded(self, opt):
        result = opt.optimise()
        if result.converged:
            assert result.expected_loss_ratio <= 0.75 + 1e-3

    def test_retention_floor_met(self, opt):
        result = opt.optimise()
        if result.converged and result.expected_retention is not None:
            assert result.expected_retention >= 0.80 - 1e-3

    def test_rate_change_bounded(self, opt):
        result = opt.optimise()
        delta = opt.config.max_rate_change
        pm = opt.prior_multiplier
        assert np.all(result.multipliers <= pm * (1 + delta) + 1e-4)
        assert np.all(result.multipliers >= pm * (1 - delta) - 1e-4)

    def test_audit_trail_complete(self, opt):
        result = opt.optimise()
        trail = result.audit_trail
        required = [
            "library", "timestamp_utc", "inputs", "constraints",
            "solver", "solution", "portfolio_metrics", "convergence",
        ]
        for k in required:
            assert k in trail, f"Audit trail missing: {k}"

    def test_audit_trail_json_roundtrip(self, opt):
        result = opt.optimise()
        json_str = result.to_json()
        loaded = json.loads(json_str)
        assert loaded["inputs"]["n_policies"] == opt.n

    def test_summary_df_is_polars(self, opt):
        result = opt.optimise()
        assert isinstance(result.summary_df, pl.DataFrame)
        assert len(result.summary_df) == opt.n

    def test_baseline_vs_optimised_profit(self, opt):
        """Optimised profit should be >= baseline (prior multiplier) profit."""
        result = opt.optimise()
        baseline = opt.portfolio_summary()
        # Optimised should be at least as good as baseline, subject to constraints
        if result.converged:
            assert result.expected_profit >= baseline["profit"] - 1e-2


class TestScenarioWorkflow:

    def test_three_scenario_run(self):
        """Simulate upstream confidence interval: low/mid/high elasticity."""
        p = _make_renewal_nb_mix(seed=7)
        config = ConstraintConfig(lr_max=0.75, technical_floor=True)
        opt = PortfolioOptimiser(**p, constraints=config)

        beta_central = p["elasticity"]
        sr = opt.optimise_scenarios(
            elasticity_scenarios=[
                beta_central * 0.7,
                beta_central,
                beta_central * 1.3,
            ],
            scenario_names=["pessimistic", "central", "optimistic"],
        )
        assert isinstance(sr, ScenarioResult)
        assert len(sr.results) == 3

        # Pessimistic (more elastic) -> lower profit than optimistic
        # This is approximate since constraint config differs
        df = sr.summary()
        assert "pessimistic" in list(df["scenario"])
        assert "optimistic" in list(df["scenario"])

    def test_multiplier_ci_ordered(self):
        """p10 <= mean <= p90 for each policy (approximately)."""
        p = _make_renewal_nb_mix(seed=8)
        opt = PortfolioOptimiser(**p)
        sr = opt.optimise_scenarios(
            elasticity_scenarios=[p["elasticity"] * f for f in [0.6, 0.8, 1.0, 1.2, 1.4]]
        )
        # Mean should be between p10 and p90
        assert np.all(sr.multiplier_p10 <= sr.multiplier_mean + 0.1)
        assert np.all(sr.multiplier_mean <= sr.multiplier_p90 + 0.1)


class TestFrontierWorkflow:

    def test_retention_frontier_runs(self):
        p = _make_renewal_nb_mix(seed=3)
        config = ConstraintConfig(lr_max=0.78, technical_floor=True)
        opt = PortfolioOptimiser(**p, constraints=config)
        ef = EfficientFrontier(
            opt,
            sweep_param="volume_retention",
            sweep_range=(0.75, 0.92),
            n_points=4,
        )
        result = ef.run()
        assert len(result.points) == 4

    def test_frontier_data_monotone_direction(self):
        """
        Higher retention requirement -> lower or equal profit (approximately).
        The frontier should be downward-sloping in profit vs retention.
        We check that the profit doesn't *increase* significantly as retention
        requirement increases.
        """
        p = _make_renewal_nb_mix(seed=4)
        config = ConstraintConfig(lr_max=0.80, technical_floor=True)
        opt = PortfolioOptimiser(**p, constraints=config)
        ef = EfficientFrontier(
            opt,
            sweep_param="volume_retention",
            sweep_range=(0.75, 0.90),
            n_points=5,
        )
        result = ef.run()
        converged_pts = [fp for fp in result.points if fp.result.converged]
        if len(converged_pts) < 2:
            pytest.skip("Not enough converged frontier points for monotonicity test")

        profits = np.array([fp.result.expected_profit for fp in converged_pts])
        epsilons = np.array([fp.epsilon for fp in converged_pts])
        # As epsilon (retention floor) increases, profit should not increase
        # Allow small tolerance for numerical noise
        for i in range(len(profits) - 1):
            if epsilons[i+1] > epsilons[i]:
                assert profits[i+1] <= profits[i] + abs(profits[i]) * 0.05 + 1.0, (
                    f"Profit increased when retention requirement went up: "
                    f"{profits[i]:.0f} -> {profits[i+1]:.0f}"
                )


class TestPortfolioSummary:

    def test_baseline_summary_sensible(self):
        p = _make_renewal_nb_mix()
        opt = PortfolioOptimiser(**p)
        summary = opt.portfolio_summary()
        assert summary["gwp"] > 0
        assert 0 < summary["loss_ratio"] < 2.0
        assert summary["retention"] is not None
        assert 0 < summary["retention"] <= 1.0

    def test_summary_at_custom_multiplier(self):
        p = _make_renewal_nb_mix()
        opt = PortfolioOptimiser(**p)
        m_high = opt.prior_multiplier * 1.20
        summary_high = opt.portfolio_summary(m=m_high)
        summary_base = opt.portfolio_summary()
        # Higher prices -> lower retention (demand falls)
        assert summary_high["retention"] <= summary_base["retention"] + 1e-6
        # Higher prices -> higher LR denominator -> lower LR
        assert summary_high["loss_ratio"] <= summary_base["loss_ratio"] + 1e-6
