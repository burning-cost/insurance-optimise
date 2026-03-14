"""
Tests for ClaimsVarianceModel (ported from rate-optimiser).

Covers construction from Tweedie GLM outputs and from a frequency-severity
decomposition. The stochastic LR constraint itself is tested end-to-end in
test_optimiser.py via ConstraintConfig(stochastic_lr=True).
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_optimise.stochastic import ClaimsVarianceModel


@pytest.fixture
def mean_claims(small_portfolio):
    """Expected loss costs from the small synthetic portfolio."""
    return small_portfolio["expected_loss_cost"]


class TestClaimsVarianceModelFromTweedie:
    def test_shapes_preserved(self, mean_claims):
        model = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=1.5)
        assert model.mean_claims.shape == mean_claims.shape
        assert model.variance_claims.shape == mean_claims.shape

    def test_variance_positive(self, mean_claims):
        model = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=1.5)
        assert (model.variance_claims > 0).all()

    def test_higher_dispersion_gives_higher_variance(self, mean_claims):
        m1 = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=1.5)
        m2 = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=2.0, power=1.5)
        assert (m2.variance_claims > m1.variance_claims).all()

    def test_higher_power_gives_steeper_scaling(self, mean_claims):
        # For means > 1, higher power gives higher variance (phi * mu^p scales faster)
        m1 = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=1.0)
        m2 = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=2.0)
        high_mean_mask = mean_claims > 1
        if high_mean_mask.any():
            assert (m2.variance_claims[high_mean_mask] > m1.variance_claims[high_mean_mask]).all()

    def test_invalid_dispersion_raises(self, mean_claims):
        with pytest.raises(ValueError, match="dispersion"):
            ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=-1.0)

    def test_known_values(self):
        # phi=1, p=2 (gamma): Var = mu^2, so mean=10 -> var=100
        mc = np.array([10.0, 20.0])
        model = ClaimsVarianceModel.from_tweedie(mc, dispersion=1.0, power=2.0)
        np.testing.assert_allclose(model.variance_claims, np.array([100.0, 400.0]))


class TestClaimsVarianceModelFromFreqSev:
    def test_shapes(self, mean_claims):
        n = len(mean_claims)
        counts = np.ones(n) * 0.1
        severity = mean_claims / 0.1
        sev_var = severity**2 * 0.5
        model = ClaimsVarianceModel.from_overdispersed_poisson(
            expected_counts=counts,
            mean_severity=severity,
            severity_variance=sev_var,
            overdispersion=1.5,
        )
        assert model.mean_claims.shape == (n,)
        assert (model.variance_claims > 0).all()

    def test_mean_claims_correct(self):
        counts = np.array([0.1, 0.2])
        severity = np.array([500.0, 700.0])
        sev_var = np.array([1000.0, 2000.0])
        model = ClaimsVarianceModel.from_overdispersed_poisson(
            expected_counts=counts, mean_severity=severity, severity_variance=sev_var
        )
        np.testing.assert_allclose(model.mean_claims, counts * severity)

    def test_higher_overdispersion_increases_variance(self):
        counts = np.array([0.5, 0.5])
        severity = np.array([400.0, 400.0])
        sev_var = np.array([500.0, 500.0])
        m1 = ClaimsVarianceModel.from_overdispersed_poisson(
            counts, severity, sev_var, overdispersion=1.0
        )
        m2 = ClaimsVarianceModel.from_overdispersed_poisson(
            counts, severity, sev_var, overdispersion=2.0
        )
        assert (m2.variance_claims > m1.variance_claims).all()


class TestClaimsVarianceModelValidation:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            ClaimsVarianceModel(
                mean_claims=np.array([100.0, 200.0]),
                variance_claims=np.array([10.0]),
            )

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ClaimsVarianceModel(
                mean_claims=np.array([100.0]),
                variance_claims=np.array([-5.0]),
            )

    def test_repr_contains_key_info(self):
        model = ClaimsVarianceModel(
            mean_claims=np.array([100.0, 200.0]),
            variance_claims=np.array([50.0, 100.0]),
        )
        r = repr(model)
        assert "ClaimsVarianceModel" in r
        assert "n_policies=2" in r


class TestClaimsVarianceModelIntegrationWithOptimiser:
    """Verify ClaimsVarianceModel integrates correctly with PortfolioOptimiser."""

    def test_stochastic_lr_runs(self, small_portfolio):
        from insurance_optimise import PortfolioOptimiser, ConstraintConfig

        p = small_portfolio
        var_model = ClaimsVarianceModel.from_tweedie(
            mean_claims=p["expected_loss_cost"],
            dispersion=1.2,
            power=1.5,
        )
        config = ConstraintConfig(
            lr_max=0.75,
            stochastic_lr=True,
            stochastic_alpha=0.90,
            technical_floor=False,
            min_multiplier=0.5,
        )
        opt = PortfolioOptimiser(
            technical_price=p["technical_price"],
            expected_loss_cost=p["expected_loss_cost"],
            p_demand=p["p_demand"],
            elasticity=p["elasticity"],
            claims_variance=var_model.variance_claims,
            constraints=config,
        )
        result = opt.optimise()
        # Should produce a result (convergence may vary, but no exception)
        assert hasattr(result, "multipliers")
        assert len(result.multipliers) == len(p["technical_price"])

    def test_stochastic_more_conservative_than_deterministic(self, small_portfolio):
        """
        With stochastic_lr=True, the effective LR constraint is tighter,
        so the optimiser must work harder to achieve the same nominal target.
        The resulting LR at solution should be <= lr_max.
        """
        from insurance_optimise import PortfolioOptimiser, ConstraintConfig

        p = small_portfolio
        var_model = ClaimsVarianceModel.from_tweedie(
            mean_claims=p["expected_loss_cost"],
            dispersion=1.5,
            power=1.5,
        )

        lr_target = 0.80  # loose target so both are likely feasible

        def make_opt(stochastic: bool) -> PortfolioOptimiser:
            config = ConstraintConfig(
                lr_max=lr_target,
                stochastic_lr=stochastic,
                stochastic_alpha=0.95,
                technical_floor=False,
                min_multiplier=0.5,
            )
            return PortfolioOptimiser(
                technical_price=p["technical_price"],
                expected_loss_cost=p["expected_loss_cost"],
                p_demand=p["p_demand"],
                elasticity=p["elasticity"],
                claims_variance=var_model.variance_claims if stochastic else None,
                constraints=config,
            )

        r_det = make_opt(stochastic=False).optimise()
        r_stoc = make_opt(stochastic=True).optimise()

        # If both converged, stochastic should produce lower or equal LR
        # (because it's constrained to be more conservative)
        if r_det.converged and r_stoc.converged:
            assert r_stoc.expected_loss_ratio <= r_det.expected_loss_ratio + 1e-3, (
                "Stochastic constraint should produce same or lower LR than deterministic"
            )
