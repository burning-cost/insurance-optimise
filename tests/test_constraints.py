"""
Tests for constraint building (ConstraintConfig, build_bounds,
build_scipy_constraints).

Tests verify:
- Bounds are correctly constructed from config
- ENBP upper bound is applied only to renewal policies
- Rate-change bounds are symmetric around prior multiplier
- Technical floor raises lower bound to 1.0
- Constraint functions return correct sign/direction
- Analytical Jacobians match finite-difference
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_optimise.constraints import (
    ConstraintConfig,
    build_bounds,
    build_scipy_constraints,
)
from insurance_optimise._demand_model import LogLinearDemand


def _make_demand(n: int, seed: int = 0) -> LogLinearDemand:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.7, 0.9, size=n)
    elast = -rng.uniform(0.5, 2.0, size=n)
    return LogLinearDemand(x0=x0, elasticity=elast)


class TestConstraintConfig:

    def test_default_config_is_valid(self):
        """Default ConstraintConfig should pass validation."""
        ConstraintConfig().validate()  # should not raise

    def test_lr_min_gt_lr_max_raises(self):
        with pytest.raises(ValueError, match="lr_min"):
            ConstraintConfig(lr_min=0.80, lr_max=0.70).validate()

    def test_lr_min_equals_lr_max_raises(self):
        with pytest.raises(ValueError, match="lr_min"):
            ConstraintConfig(lr_min=0.70, lr_max=0.70).validate()

    def test_negative_rate_change_raises(self):
        with pytest.raises(ValueError, match="max_rate_change"):
            ConstraintConfig(max_rate_change=-0.10).validate()

    def test_retention_out_of_range_raises(self):
        with pytest.raises(ValueError, match="retention_min"):
            ConstraintConfig(retention_min=1.5).validate()

    def test_invalid_stochastic_alpha_raises(self):
        with pytest.raises(ValueError, match="stochastic_alpha"):
            ConstraintConfig(stochastic_alpha=1.1).validate()

    def test_cvar_max_raises_not_implemented(self):
        """P1-1: cvar_max is documented but not implemented; validate() must raise."""
        with pytest.raises(NotImplementedError, match="cvar_max"):
            ConstraintConfig(cvar_max=50000.0).validate()

    def test_cvar_max_none_does_not_raise(self):
        """cvar_max=None (default) should pass validation without error."""
        ConstraintConfig(cvar_max=None).validate()  # should not raise


class TestBuildBounds:

    def test_absolute_bounds_respected(self):
        """min_multiplier and max_multiplier form base bounds."""
        n = 5
        tc = np.ones(n) * 500
        pm = np.ones(n)
        config = ConstraintConfig(min_multiplier=0.8, max_multiplier=1.5, technical_floor=False)
        bounds = build_bounds(config, n, tc, pm, enbp=None, renewal_flag=None)
        assert np.all(bounds.lb == pytest.approx(0.8))
        assert np.all(bounds.ub == pytest.approx(1.5))

    def test_technical_floor_sets_lb_to_1(self):
        """technical_floor=True forces lb >= 1.0."""
        n = 3
        tc = np.ones(n) * 400
        pm = np.ones(n)
        config = ConstraintConfig(technical_floor=True, min_multiplier=0.5)
        bounds = build_bounds(config, n, tc, pm, enbp=None, renewal_flag=None)
        assert np.all(bounds.lb >= 1.0)

    def test_rate_change_bounds(self):
        """Rate change bounds are symmetric around prior_multiplier."""
        n = 4
        tc = np.ones(n) * 500
        pm = np.array([1.0, 1.2, 0.9, 1.1])
        config = ConstraintConfig(max_rate_change=0.15, technical_floor=False)
        bounds = build_bounds(config, n, tc, pm, enbp=None, renewal_flag=None)
        expected_lb = pm * 0.85
        expected_ub = pm * 1.15
        # lb is max(min_multiplier, expected_lb)
        np.testing.assert_allclose(
            bounds.lb, np.maximum(0.5, expected_lb), rtol=1e-6
        )
        np.testing.assert_allclose(
            bounds.ub, np.minimum(3.0, expected_ub), rtol=1e-6
        )

    def test_enbp_applies_only_to_renewals(self):
        """ENBP upper bound should only apply where renewal_flag=True."""
        n = 4
        tc = np.array([500.0, 600.0, 400.0, 700.0])
        pm = np.ones(n)
        enbp = np.array([550.0, 650.0, 450.0, 750.0])  # enbp/tc = ~1.1
        renewal = np.array([True, False, True, False])
        config = ConstraintConfig(enbp_buffer=0.0, technical_floor=False)
        bounds = build_bounds(config, n, tc, pm, enbp=enbp, renewal_flag=renewal)

        # Renewal policies: ub = enbp/tc
        np.testing.assert_allclose(bounds.ub[0], enbp[0] / tc[0], rtol=1e-6)
        np.testing.assert_allclose(bounds.ub[2], enbp[2] / tc[2], rtol=1e-6)
        # Non-renewal: ub = max_multiplier = 3.0
        assert bounds.ub[1] == pytest.approx(3.0)
        assert bounds.ub[3] == pytest.approx(3.0)

    def test_enbp_buffer_applied(self):
        """ENBP buffer reduces the effective upper bound."""
        n = 2
        tc = np.array([500.0, 600.0])
        pm = np.ones(n)
        enbp = np.array([600.0, 700.0])
        renewal = np.array([True, True])
        config = ConstraintConfig(enbp_buffer=0.02, technical_floor=False)
        bounds = build_bounds(config, n, tc, pm, enbp=enbp, renewal_flag=renewal)
        expected_ub = enbp / tc * 0.98
        np.testing.assert_allclose(bounds.ub, expected_ub, rtol=1e-6)

    def test_infeasible_clip_warning(self):
        """When ENBP < technical floor, lb > ub: warn and clip."""
        n = 2
        tc = np.array([500.0, 600.0])
        pm = np.ones(n)
        # ENBP less than technical_price -> ub_m < 1.0 < lb (technical floor)
        enbp = np.array([400.0, 500.0])  # both less than tc
        renewal = np.array([True, True])
        config = ConstraintConfig(enbp_buffer=0.0, technical_floor=True)
        with pytest.warns(UserWarning, match="lb > ub"):
            bounds = build_bounds(config, n, tc, pm, enbp=enbp, renewal_flag=renewal)
        # After clipping: lb should equal ub
        assert np.all(bounds.lb <= bounds.ub + 1e-9)


class TestBuildScipyConstraints:

    def _setup(self, n: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        tc = rng.uniform(300, 800, size=n)
        cost = tc * rng.uniform(0.55, 0.70)
        renewal = np.zeros(n, dtype=bool)
        renewal[:n//2] = True
        demand = _make_demand(n, seed)
        return tc, cost, renewal, demand

    def test_no_active_constraints_returns_empty(self):
        tc, cost, renewal, demand = self._setup()
        config = ConstraintConfig()  # no constraints set
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        assert cons == []

    def test_lr_max_constraint_at_optimum_direction(self):
        """LR constraint fun > 0 when LR is below max."""
        n = 10
        tc, cost, renewal, demand = self._setup(n)
        config = ConstraintConfig(lr_max=0.80)  # generous ceiling
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        assert len(cons) == 1
        m = np.ones(n)
        val = cons[0]["fun"](m)
        # At m=1, LR = sum(cost*x)/sum(tc*x) ≈ 0.6-0.7 < 0.80 -> val > 0
        assert val > 0, f"Expected constraint val > 0 but got {val}"

    def test_lr_max_constraint_violated_when_lr_high(self):
        """LR constraint fun < 0 when LR is above max."""
        n = 5
        rng = np.random.default_rng(10)
        tc = np.ones(n) * 100
        cost = np.ones(n) * 90  # 90% LR
        renewal = np.zeros(n, dtype=bool)
        demand = LogLinearDemand(x0=np.full(n, 0.8), elasticity=np.full(n, -1.0))
        config = ConstraintConfig(lr_max=0.70)  # strict ceiling
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        m = np.ones(n)
        val = cons[0]["fun"](m)
        assert val < 0, f"Expected constraint val < 0 but got {val}"

    def test_gwp_min_constraint(self):
        """GWP_min constraint: positive when GWP is above minimum."""
        n = 10
        tc, cost, renewal, demand = self._setup(n)
        config = ConstraintConfig(gwp_min=100.0)  # very low: easy to satisfy
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        gwp_con = next(c for c in cons if True)  # only constraint
        m = np.ones(n)
        assert gwp_con["fun"](m) > 0

    def test_retention_constraint(self):
        """Retention constraint: positive when retention is above minimum."""
        n = 10
        tc, cost, renewal, demand = self._setup(n)
        config = ConstraintConfig(retention_min=0.50)  # easy: x0 is 0.7-0.9
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        assert len(cons) >= 1
        ret_con = cons[-1]  # retention is last
        m = np.ones(n)
        assert ret_con["fun"](m) > 0

    def test_retention_no_constraint_when_no_renewals(self):
        """No retention constraint when all policies are new business."""
        n = 5
        tc, cost, _, demand = self._setup(n)
        renewal = np.zeros(n, dtype=bool)
        config = ConstraintConfig(retention_min=0.85)
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        assert len(cons) == 0

    def test_lr_max_jacobian_finite_difference(self):
        """Analytical LR Jacobian matches finite-difference."""
        n = 6
        tc, cost, renewal, demand = self._setup(n, seed=5)
        config = ConstraintConfig(lr_max=0.75)
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        con = cons[0]
        rng = np.random.default_rng(50)
        m = rng.uniform(0.9, 1.3, size=n)

        analytical = con["jac"](m)
        eps = 1e-7
        fd = np.zeros(n)
        for i in range(n):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd[i] = (con["fun"](m_p) - con["fun"](m_m)) / (2 * eps)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_gwp_min_jacobian_finite_difference(self):
        """Analytical GWP Jacobian matches finite-difference."""
        n = 6
        tc, cost, renewal, demand = self._setup(n, seed=15)
        config = ConstraintConfig(gwp_min=500.0)
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        con = cons[0]
        rng = np.random.default_rng(22)
        m = rng.uniform(0.9, 1.2, size=n)

        analytical = con["jac"](m)
        eps = 1e-7
        fd = np.zeros(n)
        for i in range(n):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd[i] = (con["fun"](m_p) - con["fun"](m_m)) / (2 * eps)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_retention_jacobian_finite_difference(self):
        """Analytical retention Jacobian matches finite-difference."""
        n = 8
        tc, cost, renewal, demand = self._setup(n, seed=7)
        config = ConstraintConfig(retention_min=0.80)
        cons = build_scipy_constraints(config, tc, cost, renewal, demand)
        # Find retention constraint
        ret_con = None
        for c in cons:
            ret_con = c
        if ret_con is None:
            pytest.skip("No retention constraint built")

        rng = np.random.default_rng(33)
        m = rng.uniform(0.9, 1.2, size=n)
        analytical = ret_con["jac"](m)
        eps = 1e-7
        fd = np.zeros(n)
        for i in range(n):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd[i] = (ret_con["fun"](m_p) - ret_con["fun"](m_m)) / (2 * eps)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_stochastic_lr_constraint(self):
        """Stochastic LR constraint is more conservative than deterministic."""
        n = 10
        rng = np.random.default_rng(5)
        tc = np.ones(n) * 500
        cost = np.ones(n) * 320  # ~64% LR
        renewal = np.zeros(n, dtype=bool)
        demand = LogLinearDemand(x0=np.full(n, 0.8), elasticity=np.full(n, -1.5))
        var_c = np.ones(n) * 10000.0  # non-trivial variance

        lr_max = 0.70
        config_det = ConstraintConfig(lr_max=lr_max)
        config_stoch = ConstraintConfig(
            lr_max=lr_max, stochastic_lr=True, stochastic_alpha=0.90
        )

        cons_det = build_scipy_constraints(config_det, tc, cost, renewal, demand)
        cons_stoch = build_scipy_constraints(
            config_stoch, tc, cost, renewal, demand, claims_variance=var_c
        )

        m = np.ones(n)
        val_det = cons_det[0]["fun"](m)
        val_stoch = cons_stoch[0]["fun"](m)
        # Stochastic is more conservative: its constraint value should be lower
        assert val_stoch <= val_det + 1e-6, (
            f"Stochastic ({val_stoch:.4f}) should be <= deterministic ({val_det:.4f})"
        )

    def test_stochastic_lr_sigma_formula_known_values(self):
        """
        P0 regression: sigma[LR] = sqrt(sum(var_c * x)) / gwp, not sqrt(sum(var_c * x^2)).

        With uniform x=x0 (at m=1), known var_c, tc, cost, we can verify
        the exact sigma value produced by the constraint function.
        """
        n = 4
        # Uniform portfolio: every policy identical
        tc = np.full(n, 500.0)
        cost = np.full(n, 300.0)   # 60% LR
        x0 = np.full(n, 0.8)
        var_c = np.full(n, 10000.0)
        demand = LogLinearDemand(x0=x0, elasticity=np.full(n, -1.5))

        z_alpha = np.sqrt(0.90 / 0.10)  # stochastic_alpha=0.90

        config = ConstraintConfig(lr_max=1.0, stochastic_lr=True, stochastic_alpha=0.90)
        cons = build_scipy_constraints(config, tc, cost, renewal_flag=None,
                                       demand_model=demand, claims_variance=var_c)

        m = np.ones(n)
        x = demand.demand(m)  # = x0 = 0.8 each
        gwp = np.dot(m * tc, x)
        claims = np.dot(cost, x)
        lr_det = claims / gwp

        # Correct formula: sigma = sqrt(sum(var_c * x)) / gwp
        sigma_correct = np.sqrt(np.dot(var_c, x)) / gwp
        # Wrong formula would be: sigma_wrong = sqrt(sum(var_c * x**2)) / gwp
        sigma_wrong = np.sqrt(np.dot(var_c, x**2)) / gwp

        # The constraint function returns lr_max - (lr_det + z * sigma)
        val = cons[0]["fun"](m)
        expected_val = 1.0 - (lr_det + z_alpha * sigma_correct)

        np.testing.assert_allclose(val, expected_val, rtol=1e-9,
                                   err_msg="sigma formula uses x, not x^2")

        # Confirm the wrong formula would give a different (larger z*sigma) answer
        wrong_val = 1.0 - (lr_det + z_alpha * sigma_wrong)
        assert abs(wrong_val - val) > 1e-6, (
            "Correct and wrong sigma formulas should differ for non-trivial variance"
        )

    def test_stochastic_lr_jacobian_with_variance(self):
        """Analytical Jacobian for stochastic LR constraint matches FD (P0 fix)."""
        n = 6
        tc, cost, renewal, demand = self._setup(n, seed=42)
        var_c = np.ones(n) * 5000.0
        config = ConstraintConfig(lr_max=0.80, stochastic_lr=True, stochastic_alpha=0.90)
        cons = build_scipy_constraints(config, tc, cost, renewal, demand,
                                       claims_variance=var_c)
        con = cons[0]
        rng = np.random.default_rng(77)
        m = rng.uniform(0.9, 1.2, size=n)

        analytical = con["jac"](m)
        eps = 1e-7
        fd = np.zeros(n)
        for i in range(n):
            m_p = m.copy(); m_p[i] += eps
            m_m = m.copy(); m_m[i] -= eps
            fd[i] = (con["fun"](m_p) - con["fun"](m_m)) / (2 * eps)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8,
                                   err_msg="Stochastic LR Jacobian should match FD")

    def test_stochastic_lr_none_variance_warns(self):
        """P1-2: stochastic_lr=True with no claims_variance must warn and fall back."""
        n = 5
        tc = np.ones(n) * 500
        cost = np.ones(n) * 300
        renewal = np.zeros(n, dtype=bool)
        demand = LogLinearDemand(x0=np.full(n, 0.8), elasticity=np.full(n, -1.5))

        config = ConstraintConfig(lr_max=0.80, stochastic_lr=True, stochastic_alpha=0.90)

        with pytest.warns(UserWarning, match="stochastic_lr=True"):
            cons = build_scipy_constraints(config, tc, cost, renewal, demand,
                                           claims_variance=None)

        # After fallback, constraint value should equal the deterministic result
        config_det = ConstraintConfig(lr_max=0.80)
        cons_det = build_scipy_constraints(config_det, tc, cost, renewal, demand)
        m = np.ones(n)
        np.testing.assert_allclose(cons[0]["fun"](m), cons_det[0]["fun"](m), rtol=1e-9,
                                   err_msg="Fallback should produce deterministic result")
