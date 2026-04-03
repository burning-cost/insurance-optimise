"""
Extended tests for constraints.py.

Targets gaps in test_constraints.py:
- gwp_max constraint: direction, jacobian
- lr_min constraint: direction, jacobian
- ConstraintConfig: zero rate change, enbp_buffer=0 valid, retention=0 raises
- build_bounds: no rate change (None) does not affect bounds
- model_quality LR adjustment warning in build_scipy_constraints
- stochastic LR with variance on jacobian is covered in main; add combined LR min+max
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_optimise._demand_model import LogLinearDemand
from insurance_optimise.constraints import (
    ConstraintConfig,
    build_bounds,
    build_scipy_constraints,
)


def _make_demand(n: int, seed: int = 0) -> LogLinearDemand:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.7, 0.9, size=n)
    elast = -rng.uniform(0.5, 2.0, size=n)
    return LogLinearDemand(x0=x0, elasticity=elast)


def _setup(n: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    tc = rng.uniform(300, 800, size=n)
    cost = tc * rng.uniform(0.55, 0.70)
    renewal = np.zeros(n, dtype=bool)
    renewal[: n // 2] = True
    demand = _make_demand(n, seed)
    return tc, cost, renewal, demand


# ---------------------------------------------------------------------------
# ConstraintConfig: additional validation cases
# ---------------------------------------------------------------------------


class TestConstraintConfigExtended:
    def test_zero_rate_change_raises(self):
        """max_rate_change=0 is not allowed (must be positive)."""
        with pytest.raises(ValueError, match="max_rate_change"):
            ConstraintConfig(max_rate_change=0.0).validate()

    def test_enbp_buffer_zero_valid(self):
        """enbp_buffer=0.0 (default) should pass validation without error."""
        ConstraintConfig(enbp_buffer=0.0).validate()  # should not raise

    def test_retention_min_zero_raises(self):
        """retention_min=0 is not in (0, 1)."""
        with pytest.raises(ValueError, match="retention_min"):
            ConstraintConfig(retention_min=0.0).validate()

    def test_lr_min_valid_below_lr_max(self):
        """lr_min < lr_max should be valid."""
        ConstraintConfig(lr_min=0.55, lr_max=0.75).validate()  # should not raise

    def test_gwp_bounds_accept_any_positive_value(self):
        """gwp_min and gwp_max accept any positive value without validation error."""
        ConstraintConfig(gwp_min=1000.0, gwp_max=5_000_000.0).validate()


# ---------------------------------------------------------------------------
# build_bounds: edge cases
# ---------------------------------------------------------------------------


class TestBuildBoundsExtended:
    def test_no_rate_change_does_not_restrict(self):
        """
        Without max_rate_change, bounds should just be min/max_multiplier
        (subject to technical floor).
        """
        n = 5
        tc = np.ones(n) * 500
        pm = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        config = ConstraintConfig(
            max_rate_change=None,
            technical_floor=False,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )
        bounds = build_bounds(config, n, tc, pm, enbp=None, renewal_flag=None)
        np.testing.assert_allclose(bounds.lb, 0.5)
        np.testing.assert_allclose(bounds.ub, 3.0)

    def test_enbp_none_no_restriction_on_new_business(self):
        """When enbp=None, new business upper bound is just max_multiplier."""
        n = 4
        tc = np.ones(n) * 400
        pm = np.ones(n)
        renewal = np.array([False, False, False, False])
        config = ConstraintConfig(enbp_buffer=0.0, technical_floor=False)
        bounds = build_bounds(config, n, tc, pm, enbp=None, renewal_flag=renewal)
        np.testing.assert_allclose(bounds.ub, 3.0)

    def test_large_rate_change_does_not_exceed_absolute_bounds(self):
        """
        Even with max_rate_change=0.5, the bound should not exceed
        max_multiplier=3.0 or go below min_multiplier=0.5.
        """
        n = 3
        tc = np.ones(n) * 500
        pm = np.array([2.5, 3.0, 0.6])  # high prior multipliers
        config = ConstraintConfig(
            max_rate_change=0.5, min_multiplier=0.5, max_multiplier=3.0, technical_floor=False
        )
        bounds = build_bounds(config, n, tc, pm, enbp=None, renewal_flag=None)
        assert np.all(bounds.ub <= 3.0 + 1e-10)
        assert np.all(bounds.lb >= 0.5 - 1e-10)


# ---------------------------------------------------------------------------
# build_scipy_constraints: gwp_max and lr_min
# ---------------------------------------------------------------------------


class TestBuildScipyConstraintsExtended:
    def test_gwp_max_constraint_direction(self):
        """
        GWP_max constraint: fun > 0 when GWP is below the maximum.
        """
        n = 10
        tc, cost, renewal, demand = _setup(n)
        # Very high max: easy to satisfy at m=1
        gwp_at_1 = float(np.dot(np.ones(n) * tc, demand.demand(np.ones(n))))
        config = ConstraintConfig(gwp_max=gwp_at_1 * 2.0)
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        # Should have one gwp_max constraint
        assert len(cons) == 1
        m = np.ones(n)
        val = cons[0]["fun"](m)
        assert val > 0, f"GWP below max -> constraint val should be > 0, got {val}"

    def test_gwp_max_violated_when_gwp_high(self):
        """GWP_max constraint is violated when GWP exceeds the cap."""
        n = 5
        tc = np.ones(n) * 500
        demand = LogLinearDemand(x0=np.full(n, 0.9), elasticity=np.full(n, -0.5))
        cost = tc * 0.60
        renewal = np.zeros(n, dtype=bool)

        # At m=1: GWP = 5 * 500 * 0.9 = 2250
        config = ConstraintConfig(gwp_max=100.0)  # tiny cap
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        m = np.ones(n)
        val = cons[0]["fun"](m)
        assert val < 0, f"GWP above max -> constraint val should be < 0, got {val}"

    def test_gwp_max_jacobian_finite_difference(self):
        """Analytical Jacobian for gwp_max matches finite-difference."""
        n = 6
        tc, cost, renewal, demand = _setup(n, seed=20)
        config = ConstraintConfig(gwp_max=1e6)  # high cap: constraint active but not binding
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        con = cons[0]
        rng = np.random.default_rng(33)
        m = rng.uniform(0.9, 1.2, size=n)

        analytical = con["jac"](m)
        eps = 1e-7
        fd = np.zeros(n)
        for i in range(n):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd[i] = (con["fun"](m_p) - con["fun"](m_m)) / (2 * eps)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_lr_min_constraint_direction(self):
        """
        LR_min constraint fun > 0 when LR is above the minimum.
        At m=1 with typical 55-70% LR, this should be satisfied if lr_min is low.
        """
        n = 10
        tc, cost, renewal, demand = _setup(n)
        config = ConstraintConfig(lr_min=0.40)  # easy to satisfy
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        assert len(cons) == 1
        m = np.ones(n)
        val = cons[0]["fun"](m)
        assert val > 0, f"LR above min -> constraint val should be > 0, got {val}"

    def test_lr_min_violated_when_margin_too_high(self):
        """
        LR_min is violated when multipliers are so high that LR drops below minimum.
        """
        n = 5
        tc = np.ones(n) * 500
        cost = tc * 0.50  # 50% LR at m=1
        demand = LogLinearDemand(x0=np.full(n, 0.8), elasticity=np.full(n, -1.0))
        renewal = np.zeros(n, dtype=bool)
        config = ConstraintConfig(lr_min=0.60)  # requires LR >= 60%, but we have 50%
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        m = np.ones(n)
        val = cons[0]["fun"](m)
        assert val < 0, f"LR below min -> constraint val should be < 0, got {val}"

    def test_lr_min_jacobian_finite_difference(self):
        """Analytical Jacobian for lr_min constraint matches finite-difference."""
        n = 6
        tc, cost, renewal, demand = _setup(n, seed=30)
        config = ConstraintConfig(lr_min=0.45)
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        con = cons[0]
        rng = np.random.default_rng(44)
        m = rng.uniform(0.9, 1.2, size=n)

        analytical = con["jac"](m)
        eps = 1e-7
        fd = np.zeros(n)
        for i in range(n):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd[i] = (con["fun"](m_p) - con["fun"](m_m)) / (2 * eps)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_lr_min_and_lr_max_both_active(self):
        """Both lr_min and lr_max can be active simultaneously."""
        n = 8
        tc, cost, renewal, demand = _setup(n, seed=50)
        config = ConstraintConfig(lr_min=0.45, lr_max=0.80)
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        assert len(cons) == 2  # one for lr_min, one for lr_max

    def test_model_quality_lr_adjustment_warns(self):
        """
        model_quality_adjusted_lr=True should emit a UserWarning describing
        the adjustment when build_scipy_constraints is called.
        """
        n = 8
        tc, cost, renewal, demand = _setup(n, seed=60)
        config = ConstraintConfig(
            lr_max=0.70,
            model_quality_adjusted_lr=True,
            model_rho=0.85,
            model_cv_lambda=1.2,
        )
        with pytest.warns(UserWarning, match="Model quality LR adjustment"):
            cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        # Should produce exactly one constraint (lr_max only)
        assert len(cons) == 1

    def test_model_quality_lr_adjustment_relaxes_constraint(self):
        """
        Model quality adjustment should relax (increase) the effective lr_max,
        producing a less binding (or equally binding) constraint.
        """
        n = 8
        tc, cost, renewal, demand = _setup(n, seed=70)
        lr_max = 0.70

        config_base = ConstraintConfig(lr_max=lr_max)
        config_adj = ConstraintConfig(
            lr_max=lr_max,
            model_quality_adjusted_lr=True,
            model_rho=0.80,
            model_cv_lambda=2.0,
        )

        cons_base = build_scipy_constraints(config_base, tc, cost, renewal, demand)
        with pytest.warns(UserWarning):
            cons_adj = build_scipy_constraints(config_adj, tc, cost, renewal, demand)

        m = np.ones(n)
        val_base = cons_base[0]["fun"](m)
        val_adj = cons_adj[0]["fun"](m)

        # Adjusted constraint has higher effective lr_max -> constraint is less tight
        # (fun = lr_max_eff - LR(m), so higher lr_max_eff -> higher fun value)
        assert val_adj >= val_base - 1e-10, (
            f"Adjusted constraint ({val_adj:.6f}) should be >= base ({val_base:.6f}): "
            "model quality adjustment should relax the lr_max constraint"
        )
