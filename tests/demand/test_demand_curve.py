"""Tests for DemandCurve."""

import numpy as np
import pytest

from insurance_optimise.demand import DemandCurve


class TestDemandCurveParametric:
    def setup_method(self):
        self.curve = DemandCurve(
            elasticity=-2.0,
            base_price=500.0,
            base_prob=0.12,
            functional_form="semi_log",
        )

    def test_evaluate_returns_correct_shapes(self):
        prices, probs = self.curve.evaluate(price_range=(300, 800), n_points=50)
        assert len(prices) == 50
        assert len(probs) == 50

    def test_probs_in_range(self):
        prices, probs = self.curve.evaluate(price_range=(300, 800), n_points=50)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_demand_decreases_with_price(self):
        prices, probs = self.curve.evaluate(price_range=(300, 800), n_points=100)
        # Demand curve should be downward sloping (negative elasticity)
        assert probs[0] > probs[-1], "Higher price should give lower demand"

    def test_base_point_recovered(self):
        """Evaluate at base price should return approximately base_prob."""
        prices, probs = self.curve.evaluate(price_range=(499, 501), n_points=3)
        # Middle point is approximately 500
        mid_prob = probs[1]
        assert abs(mid_prob - 0.12) < 0.005, f"Base prob mismatch: {mid_prob:.4f} vs 0.12"

    def test_price_at_prob_inverse(self):
        """price_at_prob should invert the demand curve."""
        target = 0.10
        price = self.curve.price_at_prob(target)
        _, probs = self.curve.evaluate(price_range=(price * 0.99, price * 1.01), n_points=3)
        assert abs(probs[1] - target) < 0.01

    def test_as_demand_callable(self):
        fn = self.curve.as_demand_callable()
        assert callable(fn)
        prices = np.array([0.9, 1.0, 1.1])
        probs = fn(prices)
        assert len(probs) == 3
        assert probs[0] > probs[2]  # lower price → higher demand


class TestDemandCurveLogLinear:
    def setup_method(self):
        self.curve = DemandCurve(
            elasticity=-1.5,
            base_price=500.0,
            base_prob=0.15,
            functional_form="log_linear",
        )

    def test_evaluate_returns_correct_shapes(self):
        prices, probs = self.curve.evaluate(price_range=(300, 800), n_points=30)
        assert len(prices) == 30
        assert len(probs) == 30

    def test_demand_decreases_with_price(self):
        prices, probs = self.curve.evaluate(price_range=(200, 1000), n_points=50)
        assert probs[0] > probs[-1]

    def test_constant_elasticity(self):
        """Log-linear form should have approximately constant elasticity."""
        prices = np.array([400.0, 500.0, 600.0])
        log_prices = np.log(prices)

        curve = self.curve
        _, probs_400 = curve.evaluate((399, 401), n_points=3)
        _, probs_500 = curve.evaluate((499, 501), n_points=3)
        _, probs_600 = curve.evaluate((599, 601), n_points=3)

        e1 = np.log(probs_500[1] / probs_400[1]) / np.log(500 / 400)
        e2 = np.log(probs_600[1] / probs_500[1]) / np.log(600 / 500)

        # Both elasticities should be close to -1.5
        assert abs(e1 - (-1.5)) < 0.15
        assert abs(e2 - (-1.5)) < 0.15


class TestDemandCurveValidation:
    def test_no_model_no_elasticity_raises(self):
        with pytest.raises(ValueError):
            DemandCurve()

    def test_parametric_without_base_price_raises(self):
        with pytest.raises(ValueError):
            DemandCurve(elasticity=-2.0, functional_form="semi_log")

    def test_model_mode_without_model_raises(self):
        with pytest.raises(ValueError):
            DemandCurve(functional_form="model")
