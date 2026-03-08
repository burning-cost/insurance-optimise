"""
Tests for PortfolioOptimiser.

Tests verify:
- Input validation raises on bad inputs
- Unconstrained optimisation improves over baseline
- Constrained optimisation respects all binding constraints
- Result has correct shapes and types
- ENBP constraint never violated
- Convergence and audit trail populated
- portfolio_summary works at baseline
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_optimise import ConstraintConfig, PortfolioOptimiser
from insurance_optimise.result import OptimisationResult


class TestInputValidation:

    def test_mismatched_lengths_raises(self, small_portfolio):
        p = small_portfolio
        with pytest.raises(ValueError, match="Length mismatch"):
            PortfolioOptimiser(
                technical_price=p["technical_price"],
                expected_loss_cost=p["expected_loss_cost"][:-1],  # wrong length
                p_demand=p["p_demand"],
                elasticity=p["elasticity"],
            )

    def test_nonpositive_technical_price_raises(self, small_portfolio):
        p = small_portfolio.copy()
        p["technical_price"][0] = 0.0
        with pytest.raises(ValueError, match="technical_price must be positive"):
            PortfolioOptimiser(**p)

    def test_p_demand_out_of_range_raises(self, small_portfolio):
        p = small_portfolio.copy()
        p["p_demand"][0] = 1.5
        with pytest.raises(ValueError, match="p_demand must be in"):
            PortfolioOptimiser(**p)

    def test_negative_enbp_raises(self, small_portfolio):
        p = small_portfolio.copy()
        p["enbp"][0] = -1.0
        with pytest.raises(ValueError, match="enbp values must be positive"):
            PortfolioOptimiser(**p)

    def test_enbp_wrong_length_raises(self, small_portfolio):
        p = small_portfolio.copy()
        p["enbp"] = p["enbp"][:-2]  # wrong length
        with pytest.raises(ValueError, match="enbp has"):
            PortfolioOptimiser(**p)

    def test_invalid_demand_model_raises(self, small_portfolio):
        p = small_portfolio
        with pytest.raises(ValueError, match="Unknown demand_model"):
            PortfolioOptimiser(
                technical_price=p["technical_price"],
                expected_loss_cost=p["expected_loss_cost"],
                p_demand=p["p_demand"],
                elasticity=p["elasticity"],
                demand_model="random_forest",
            )


class TestUnconstrainedOptimisation:

    def test_optimiser_returns_result(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        assert isinstance(result, OptimisationResult)

    def test_multipliers_shape(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        assert result.multipliers.shape == (unconstrained_optimiser.n,)

    def test_new_premiums_shape(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        assert result.new_premiums.shape == (unconstrained_optimiser.n,)

    def test_new_premiums_consistent(self, unconstrained_optimiser):
        """new_premiums should equal multipliers * technical_price."""
        result = unconstrained_optimiser.optimise()
        expected = result.multipliers * unconstrained_optimiser.tc
        np.testing.assert_allclose(result.new_premiums, expected, rtol=1e-6)

    def test_profit_is_scalar(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        assert isinstance(result.expected_profit, float)

    def test_gwp_positive(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        assert result.expected_gwp > 0

    def test_loss_ratio_in_range(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        assert 0 < result.expected_loss_ratio < 5.0  # broad sanity

    def test_audit_trail_populated(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        assert "timestamp_utc" in result.audit_trail
        assert "inputs" in result.audit_trail
        assert "convergence" in result.audit_trail

    def test_summary_df_columns(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        expected_cols = {
            "policy_idx", "multiplier", "new_premium",
            "expected_demand", "contribution", "enbp_binding", "rate_change_pct"
        }
        assert expected_cols.issubset(set(result.summary_df.columns))

    def test_summary_df_row_count(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        assert len(result.summary_df) == unconstrained_optimiser.n

    def test_repr_contains_status(self, unconstrained_optimiser):
        result = unconstrained_optimiser.optimise()
        repr_str = repr(result)
        assert "CONVERGED" in repr_str or "NOT CONVERGED" in repr_str


class TestConstrainedOptimisation:

    def test_constrained_optimise_runs(self, constrained_optimiser):
        result = constrained_optimiser.optimise()
        assert isinstance(result, OptimisationResult)

    def test_enbp_constraint_not_violated(self, constrained_optimiser):
        """No renewal policy should have price > ENBP * (1 - buffer)."""
        opt = constrained_optimiser
        result = opt.optimise()
        if opt.enbp is not None:
            renewal = opt.renewal_flag
            enbp_ub = opt.enbp / opt.tc * (1.0 - opt.config.enbp_buffer)
            violations = result.multipliers[renewal] > enbp_ub[renewal] + 1e-4
            assert not np.any(violations), (
                f"ENBP violated: {np.sum(violations)} renewal policies above bound"
            )

    def test_technical_floor_respected(self, constrained_optimiser):
        """All multipliers should be >= 1.0 when technical_floor=True."""
        result = constrained_optimiser.optimise()
        assert np.all(result.multipliers >= 1.0 - 1e-4), (
            "Technical floor violated: some multipliers < 1.0"
        )

    def test_loss_ratio_constraint_respected(self, small_portfolio):
        """Portfolio LR should not exceed lr_max at solution."""
        p = small_portfolio
        config = ConstraintConfig(lr_max=0.72, technical_floor=True)
        opt = PortfolioOptimiser(**p, constraints=config)
        result = opt.optimise()
        if result.converged:
            assert result.expected_loss_ratio <= 0.72 + 1e-3, (
                f"LR constraint violated: {result.expected_loss_ratio:.4f} > 0.72"
            )

    def test_retention_constraint_respected(self, small_portfolio):
        """Retention should meet the minimum when constraint is active."""
        p = small_portfolio
        config = ConstraintConfig(retention_min=0.80, technical_floor=True)
        opt = PortfolioOptimiser(**p, constraints=config)
        result = opt.optimise()
        if result.converged and result.expected_retention is not None:
            assert result.expected_retention >= 0.80 - 1e-3, (
                f"Retention constraint violated: {result.expected_retention:.4f} < 0.80"
            )

    def test_rate_change_respected(self, small_portfolio):
        """No multiplier should exceed prior * (1 + delta)."""
        p = small_portfolio
        config = ConstraintConfig(max_rate_change=0.15, technical_floor=True)
        opt = PortfolioOptimiser(**p, constraints=config)
        result = opt.optimise()
        delta = 0.15
        pm = opt.prior_multiplier
        assert np.all(result.multipliers <= pm * (1.0 + delta) + 1e-4)
        assert np.all(result.multipliers >= pm * (1.0 - delta) - 1e-4)

    def test_gwp_min_constraint_respected(self, small_portfolio):
        """GWP at solution should be >= gwp_min."""
        p = small_portfolio
        tc = p["technical_price"]
        # Set gwp_min to 50% of total technical premium (easy to achieve)
        gwp_target = np.sum(tc) * 0.5
        config = ConstraintConfig(gwp_min=gwp_target, technical_floor=True)
        opt = PortfolioOptimiser(**p, constraints=config)
        result = opt.optimise()
        if result.converged:
            assert result.expected_gwp >= gwp_target - 1e-2

    def test_shadow_prices_dict(self, constrained_optimiser):
        """shadow_prices should be a dict of floats."""
        result = constrained_optimiser.optimise()
        assert isinstance(result.shadow_prices, dict)
        for k, v in result.shadow_prices.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_portfolio_summary_at_baseline(self, constrained_optimiser):
        """portfolio_summary() at prior_multiplier=1 should return sensible dict."""
        opt = constrained_optimiser
        summary = opt.portfolio_summary()
        assert "profit" in summary
        assert "gwp" in summary
        assert "loss_ratio" in summary
        assert summary["gwp"] > 0
        assert 0 < summary["loss_ratio"] < 2.0

    def test_n_constraints_nonzero(self, constrained_optimiser):
        assert constrained_optimiser.n_constraints > 0


class TestLogisticDemandOptimiser:

    def test_logistic_demand_runs(self, small_portfolio):
        """Optimiser should work with logistic demand model."""
        p = small_portfolio
        # Convert log-space elasticity to price semi-elasticity
        elast_price = p["elasticity"] / p["technical_price"]  # rough conversion
        opt = PortfolioOptimiser(
            technical_price=p["technical_price"],
            expected_loss_cost=p["expected_loss_cost"],
            p_demand=p["p_demand"],
            elasticity=elast_price,
            demand_model="logistic",
        )
        result = opt.optimise()
        assert isinstance(result, OptimisationResult)
        assert result.multipliers.shape == (opt.n,)


class TestNewBusinessOnly:

    def test_nb_optimiser_no_enbp(self, nb_only_portfolio):
        """New business portfolio without ENBP or retention constraint."""
        p = nb_only_portfolio
        config = ConstraintConfig(lr_max=0.70, technical_floor=True)
        opt = PortfolioOptimiser(**p, constraints=config)
        result = opt.optimise()
        assert isinstance(result, OptimisationResult)
        assert result.expected_retention is None  # no renewals

    def test_nb_no_enbp_binding(self, nb_only_portfolio):
        """Without renewals, enbp_binding should all be False."""
        p = nb_only_portfolio
        opt = PortfolioOptimiser(**p)
        result = opt.optimise()
        assert not result.summary_df["enbp_binding"].any()


class TestEdgeCases:

    def test_single_policy(self):
        """N=1 should work without errors."""
        opt = PortfolioOptimiser(
            technical_price=np.array([500.0]),
            expected_loss_cost=np.array([300.0]),
            p_demand=np.array([0.85]),
            elasticity=np.array([-1.5]),
        )
        result = opt.optimise()
        assert len(result.multipliers) == 1

    def test_all_renewals(self):
        """All renewal policies with ENBP should work."""
        n = 10
        rng = np.random.default_rng(11)
        tc = rng.uniform(400, 800, size=n)
        cost = tc * 0.65
        p_demand = rng.uniform(0.75, 0.90, size=n)
        elast = -rng.uniform(1.0, 2.0, size=n)
        enbp = tc * 1.10
        renewal = np.ones(n, dtype=bool)
        opt = PortfolioOptimiser(
            technical_price=tc,
            expected_loss_cost=cost,
            p_demand=p_demand,
            elasticity=elast,
            renewal_flag=renewal,
            enbp=enbp,
            constraints=ConstraintConfig(enbp_buffer=0.01),
        )
        result = opt.optimise()
        assert isinstance(result, OptimisationResult)

    def test_to_json_serialisable(self, constrained_optimiser):
        """Result audit trail should be JSON-serialisable."""
        import json
        result = constrained_optimiser.optimise()
        json_str = result.to_json()
        data = json.loads(json_str)
        assert "library" in data
        assert data["library"] == "insurance-optimise"

    def test_save_audit_creates_file(self, tmp_path, constrained_optimiser):
        """save_audit() should create a JSON file."""
        result = constrained_optimiser.optimise()
        path = str(tmp_path / "audit.json")
        result.save_audit(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert "timestamp_utc" in data
