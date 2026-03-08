"""
Tests for demand models (LogLinearDemand and LogisticDemand).

Core properties to verify:
- Demand at m=1 matches x0
- Demand is positive for any positive multiplier
- Demand decreases as price increases (negative elasticity)
- Gradient is correct (finite-difference check)
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_optimise.demand import LogLinearDemand, LogisticDemand, make_demand_model


# ---------------------------------------------------------------------------
# LogLinearDemand
# ---------------------------------------------------------------------------

class TestLogLinearDemand:

    def test_demand_at_baseline(self):
        """x(m=1) should equal x0."""
        x0 = np.array([0.8, 0.7, 0.9])
        elast = np.array([-1.0, -2.0, -0.5])
        model = LogLinearDemand(x0=x0, elasticity=elast)
        m = np.ones(3)
        np.testing.assert_allclose(model.demand(m), x0, rtol=1e-8)

    def test_demand_decreases_with_price(self):
        """Higher multiplier -> lower demand (negative elasticity)."""
        x0 = np.array([0.8])
        elast = np.array([-1.5])
        model = LogLinearDemand(x0=x0, elasticity=elast)
        d1 = model.demand(np.array([1.0]))
        d2 = model.demand(np.array([1.1]))
        assert d2[0] < d1[0], "Demand should fall when price rises"

    def test_demand_positive_always(self):
        """Demand must be positive for any positive multiplier."""
        x0 = np.array([0.5, 0.9])
        elast = np.array([-2.0, -0.3])
        model = LogLinearDemand(x0=x0, elasticity=elast)
        for m_val in [0.1, 0.5, 1.0, 1.5, 3.0, 10.0]:
            m = np.full(2, m_val)
            d = model.demand(m)
            assert np.all(d > 0), f"Demand should be positive at m={m_val}"

    def test_gradient_finite_difference(self):
        """Analytical gradient should match finite-difference approximation."""
        rng = np.random.default_rng(123)
        n = 5
        x0 = rng.uniform(0.5, 0.9, size=n)
        elast = -rng.uniform(0.5, 2.5, size=n)
        model = LogLinearDemand(x0=x0, elasticity=elast)
        m = rng.uniform(0.8, 1.5, size=n)

        analytical = model.demand_gradient(m)
        eps = 1e-7
        fd = np.zeros(n)
        for i in range(n):
            m_plus = m.copy()
            m_plus[i] += eps
            m_minus = m.copy()
            m_minus[i] -= eps
            fd[i] = (np.sum(model.demand(m_plus)) - np.sum(model.demand(m_minus))) / (2 * eps)

        # fd is sum-based; analytical is element-wise. Check element matching.
        fd_element = np.zeros(n)
        for i in range(n):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd_element[i] = (model.demand(m_p)[i] - model.demand(m_m)[i]) / (2 * eps)

        np.testing.assert_allclose(analytical, fd_element, rtol=1e-5, atol=1e-8)

    def test_elasticity_interpretation(self):
        """
        Verify constant elasticity: 10% price increase -> elasticity * 10% demand change.
        """
        x0 = np.array([0.8])
        elast = np.array([-2.0])
        model = LogLinearDemand(x0=x0, elasticity=elast)
        d_base = model.demand(np.array([1.0]))[0]
        d_up = model.demand(np.array([1.1]))[0]
        # log(d_up/d_base) / log(1.1) should equal elasticity
        actual_elast = np.log(d_up / d_base) / np.log(1.1)
        np.testing.assert_allclose(actual_elast, -2.0, rtol=1e-6)

    def test_positive_elasticity_warning(self):
        """Positive elasticity should trigger a warning."""
        with pytest.warns(UserWarning, match="positive"):
            model = LogLinearDemand(
                x0=np.array([0.8]),
                elasticity=np.array([1.0]),  # positive — wrong sign
            )

    def test_factory_log_linear(self):
        """make_demand_model('log_linear') returns LogLinearDemand."""
        m = make_demand_model(
            "log_linear",
            x0=np.array([0.8]),
            elasticity=np.array([-1.0]),
            technical_price=np.array([500.0]),
        )
        assert isinstance(m, LogLinearDemand)

    def test_factory_unknown_raises(self):
        """Unknown demand model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown demand_model"):
            make_demand_model(
                "neural_net",
                x0=np.array([0.8]),
                elasticity=np.array([-1.0]),
                technical_price=np.array([500.0]),
            )


# ---------------------------------------------------------------------------
# LogisticDemand
# ---------------------------------------------------------------------------

class TestLogisticDemand:

    def _make_model(self, n: int = 5) -> tuple[LogisticDemand, np.ndarray]:
        rng = np.random.default_rng(0)
        x0 = rng.uniform(0.60, 0.90, size=n)
        tc = rng.uniform(300, 800, size=n)
        # Semi-elasticity in price units (not multiplier units)
        elast_price = -rng.uniform(0.001, 0.005, size=n)
        return LogisticDemand(x0=x0, elasticity=elast_price, technical_price=tc), tc

    def test_demand_at_baseline(self):
        """x(m=1) should recover x0 approximately."""
        rng = np.random.default_rng(1)
        x0 = rng.uniform(0.60, 0.90, size=3)
        tc = rng.uniform(400, 700, size=3)
        elast = -rng.uniform(0.001, 0.003, size=3)
        model = LogisticDemand(x0=x0, elasticity=elast, technical_price=tc)
        m = np.ones(3)
        np.testing.assert_allclose(model.demand(m), x0, rtol=1e-5)

    def test_demand_in_unit_interval(self):
        """Logistic demand must stay in (0, 1)."""
        model, tc = self._make_model()
        for m_val in [0.5, 1.0, 1.5, 2.0]:
            m = np.full(5, m_val)
            d = model.demand(m)
            assert np.all(d > 0) and np.all(d < 1)

    def test_demand_decreases_with_price(self):
        """Higher multiplier -> lower demand."""
        model, tc = self._make_model(n=1)
        d1 = model.demand(np.array([1.0]))[0]
        d2 = model.demand(np.array([1.1]))[0]
        assert d2 < d1

    def test_gradient_finite_difference(self):
        """Analytical gradient matches finite-difference."""
        model, tc = self._make_model(n=4)
        rng = np.random.default_rng(77)
        m = rng.uniform(0.9, 1.3, size=4)
        analytical = model.demand_gradient(m)

        eps = 1e-6
        fd = np.zeros(4)
        for i in range(4):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd[i] = (model.demand(m_p)[i] - model.demand(m_m)[i]) / (2 * eps)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_factory_logistic(self):
        """make_demand_model('logistic') returns LogisticDemand."""
        m = make_demand_model(
            "logistic",
            x0=np.array([0.8]),
            elasticity=np.array([-0.002]),
            technical_price=np.array([500.0]),
        )
        assert isinstance(m, LogisticDemand)
