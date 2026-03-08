"""
Tests for audit trail generation.

Verify:
- build_audit_trail produces complete, correctly typed output
- extract_shadow_prices handles scipy result with and without .v attribute
- evaluate_constraints returns constraint values
- JSON serialisation round-trips correctly
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from insurance_optimise.audit import (
    build_audit_trail,
    evaluate_constraints,
    extract_shadow_prices,
)


def _make_audit_kwargs(n: int = 5) -> dict:
    rng = np.random.default_rng(0)
    tc = rng.uniform(300, 700, size=n)
    cost = tc * 0.65
    m = rng.uniform(1.0, 1.2, size=n)
    return dict(
        n_policies=n,
        n_renewal=3,
        technical_price=tc,
        expected_loss_cost=cost,
        enbp=tc * 1.1,
        prior_multiplier=np.ones(n),
        constraint_config_dict={"lr_max": 0.70, "retention_min": 0.85},
        demand_model_name="log_linear",
        solver="SLSQP",
        solver_options={"ftol": 1e-9, "maxiter": 1000},
        n_restarts=1,
        x0_strategy="midpoint_with_jitter",
        multipliers=m,
        converged=True,
        solver_message="Optimization terminated successfully",
        n_iter=42,
        n_fun_eval=300,
        expected_profit=15000.0,
        expected_gwp=80000.0,
        expected_lr=0.68,
        expected_retention=0.87,
        constraint_values={"lr_max": 0.02, "retention_min": 0.015},
        shadow_prices={"lr_max": 0.034},
    )


class TestBuildAuditTrail:

    def test_required_top_level_keys(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        for key in ("library", "version", "timestamp_utc", "inputs",
                    "constraints", "solver", "demand_model", "solution",
                    "portfolio_metrics", "constraint_evaluation",
                    "shadow_prices", "convergence"):
            assert key in trail, f"Missing key: {key}"

    def test_library_name(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        assert trail["library"] == "insurance-optimise"

    def test_version_string(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        assert trail["version"] == "0.1.0"

    def test_n_policies_correct(self):
        trail = build_audit_trail(**_make_audit_kwargs(n=8))
        assert trail["inputs"]["n_policies"] == 8

    def test_n_renewal_correct(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        assert trail["inputs"]["n_renewal"] == 3

    def test_n_new_business_derived(self):
        kwargs = _make_audit_kwargs(n=10)
        kwargs["n_renewal"] = 4
        trail = build_audit_trail(**kwargs)
        assert trail["inputs"]["n_new_business"] == 6

    def test_json_serialisable(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        s = json.dumps(trail)
        data = json.loads(s)
        assert data["library"] == "insurance-optimise"

    def test_enbp_info_present_when_enbp_provided(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        assert trail["inputs"]["has_enbp"] is True
        assert "enbp_multiplier_mean" in trail["inputs"]

    def test_enbp_info_absent_when_no_enbp(self):
        kwargs = _make_audit_kwargs()
        kwargs["enbp"] = None
        trail = build_audit_trail(**kwargs)
        assert trail["inputs"]["has_enbp"] is False
        assert "enbp_multiplier_mean" not in trail["inputs"]

    def test_converged_flag(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        assert trail["convergence"]["converged"] is True

    def test_portfolio_metrics_present(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        pm = trail["portfolio_metrics"]
        assert "expected_profit" in pm
        assert "expected_gwp" in pm
        assert "expected_loss_ratio" in pm
        assert pm["expected_profit"] == pytest.approx(15000.0)

    def test_retention_included_when_present(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        assert "expected_retention" in trail["portfolio_metrics"]

    def test_retention_excluded_when_none(self):
        kwargs = _make_audit_kwargs()
        kwargs["expected_retention"] = None
        trail = build_audit_trail(**kwargs)
        assert "expected_retention" not in trail["portfolio_metrics"]

    def test_multiplier_statistics_present(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        sol = trail["solution"]
        for k in ("multiplier_mean", "multiplier_median", "multiplier_min",
                  "multiplier_max", "rate_change_mean_pct"):
            assert k in sol, f"Missing solution stat: {k}"

    def test_solver_section(self):
        trail = build_audit_trail(**_make_audit_kwargs())
        assert trail["solver"]["method"] == "SLSQP"
        assert trail["solver"]["n_restarts"] == 1


class TestExtractShadowPrices:

    def test_returns_dict(self):
        scipy_result = SimpleNamespace(v=None)
        shadow = extract_shadow_prices(scipy_result, ["lr_max"])
        assert isinstance(shadow, dict)

    def test_fallback_zeros_when_no_v(self):
        scipy_result = SimpleNamespace(v=None)
        shadow = extract_shadow_prices(scipy_result, ["lr_max", "gwp_min"])
        assert shadow["lr_max"] == 0.0
        assert shadow["gwp_min"] == 0.0

    def test_extracts_array_v(self):
        scipy_result = SimpleNamespace(v=[np.array([0.034]), np.array([0.0])])
        shadow = extract_shadow_prices(scipy_result, ["lr_max", "gwp_min"])
        assert shadow["lr_max"] == pytest.approx(0.034)

    def test_extracts_scalar_v(self):
        scipy_result = SimpleNamespace(v=[0.021])
        shadow = extract_shadow_prices(scipy_result, ["lr_max"])
        assert shadow["lr_max"] == pytest.approx(0.021)

    def test_names_longer_than_v_fallback(self):
        """More constraint names than v entries: fallback to zeros."""
        scipy_result = SimpleNamespace(v=[0.01])
        shadow = extract_shadow_prices(scipy_result, ["a", "b", "c"])
        assert "a" in shadow
        assert "b" in shadow
        assert "c" in shadow


class TestEvaluateConstraints:

    def test_returns_dict_of_floats(self):
        m = np.ones(4)
        constraints = [
            {"fun": lambda m: float(np.sum(m) - 3.0), "type": "ineq"},
            {"fun": lambda m: float(2.0 - np.sum(m**2)), "type": "ineq"},
        ]
        vals = evaluate_constraints(m, constraints, ["c1", "c2"])
        assert isinstance(vals, dict)
        assert set(vals.keys()) == {"c1", "c2"}
        assert isinstance(vals["c1"], float)

    def test_constraint_value_correct(self):
        m = np.array([1.0, 1.0, 1.0])
        constraints = [{"fun": lambda m: float(np.sum(m)), "type": "ineq"}]
        vals = evaluate_constraints(m, constraints, ["sum"])
        assert vals["sum"] == pytest.approx(3.0)

    def test_handles_exception_gracefully(self):
        """If a constraint function raises, return NaN for that constraint."""
        def bad_fun(m):
            raise RuntimeError("bad constraint")

        constraints = [{"fun": bad_fun, "type": "ineq"}]
        vals = evaluate_constraints(np.ones(3), constraints, ["bad"])
        assert vals["bad"] != vals["bad"]  # NaN check
