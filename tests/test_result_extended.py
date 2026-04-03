"""
Extended tests for result.py.

Targets gaps in test_result.py:
- OptimisationResult.profit property (alias for expected_profit)
- to_json with numpy arrays in audit_trail
- ScenarioResult with a single result
- EfficientFrontierResult with all converged vs all failed
- FrontierPoint stores epsilon correctly
- OptimisationResult model quality fields (model_lre, lr_constraint_used)
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from insurance_optimise.result import (
    EfficientFrontierResult,
    FrontierPoint,
    OptimisationResult,
    ScenarioResult,
    _json_default,
)


def _make_dummy_result(
    n: int = 5,
    converged: bool = True,
    model_lre: float | None = None,
    lr_constraint_used: float | None = None,
) -> OptimisationResult:
    rng = np.random.default_rng(0)
    tc = rng.uniform(300, 600, size=n)
    m = rng.uniform(1.0, 1.3, size=n)
    p = m * tc
    x = rng.uniform(0.7, 0.9, size=n)

    summary_df = pl.DataFrame({
        "policy_idx": list(range(n)),
        "multiplier": m.tolist(),
        "new_premium": p.tolist(),
        "expected_demand": x.tolist(),
        "contribution": ((p - tc * 0.65) * x).tolist(),
        "enbp_binding": [False] * n,
        "rate_change_pct": ((m - 1.0) * 100).tolist(),
    })

    audit = {
        "library": "insurance-optimise",
        "version": "0.1.0",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "inputs": {"n_policies": n},
        "convergence": {"converged": converged},
    }

    return OptimisationResult(
        multipliers=m,
        new_premiums=p,
        expected_demand=x,
        expected_profit=float(np.dot(p - tc * 0.65, x)),
        expected_gwp=float(np.dot(p, x)),
        expected_loss_ratio=0.65,
        expected_retention=0.88,
        shadow_prices={"lr_max": 0.02},
        converged=converged,
        solver_message="Optimization terminated successfully",
        n_iter=45,
        audit_trail=audit,
        summary_df=summary_df,
        model_lre=model_lre,
        lr_constraint_used=lr_constraint_used,
    )


# ---------------------------------------------------------------------------
# OptimisationResult: profit alias
# ---------------------------------------------------------------------------


class TestOptimisationResultExtended:
    def test_profit_alias_equals_expected_profit(self):
        """result.profit should be identical to result.expected_profit."""
        result = _make_dummy_result()
        assert result.profit == result.expected_profit

    def test_profit_alias_is_float(self):
        result = _make_dummy_result()
        assert isinstance(result.profit, float)

    def test_model_lre_default_none(self):
        """model_lre defaults to None when not provided."""
        result = _make_dummy_result()
        assert result.model_lre is None

    def test_model_lre_populated(self):
        """model_lre can be stored and retrieved."""
        result = _make_dummy_result(model_lre=0.023, lr_constraint_used=0.723)
        assert result.model_lre == pytest.approx(0.023)
        assert result.lr_constraint_used == pytest.approx(0.723)

    def test_to_json_handles_numpy_arrays_in_audit(self):
        """
        When audit_trail contains numpy arrays, to_json should succeed
        via the _json_default serialiser.
        """
        result = _make_dummy_result()
        # Insert a numpy array into the audit trail
        result.audit_trail["inputs"]["multipliers"] = np.array([1.0, 1.1, 1.2])
        json_str = result.to_json()
        data = json.loads(json_str)
        # Should have been serialised to a list
        assert isinstance(data["inputs"]["multipliers"], list)

    def test_new_premiums_are_positive(self):
        """New premiums should all be positive."""
        result = _make_dummy_result()
        assert np.all(result.new_premiums > 0)

    def test_repr_shows_n_policies(self):
        """Repr should show N= (number of policies)."""
        result = _make_dummy_result(n=8)
        assert "N=8" in repr(result)


# ---------------------------------------------------------------------------
# ScenarioResult: edge cases
# ---------------------------------------------------------------------------


class TestScenarioResultExtended:
    def test_single_scenario(self):
        """ScenarioResult with a single scenario should work fine."""
        results = [_make_dummy_result(5)]
        sr = ScenarioResult(
            results=results,
            scenario_names=["central"],
            multiplier_mean=np.ones(5),
            multiplier_p10=np.ones(5),
            multiplier_p90=np.ones(5),
            profit_mean=10000.0,
            profit_p10=9000.0,
            profit_p90=11000.0,
        )
        df = sr.summary()
        assert len(df) == 1
        assert list(df["scenario"]) == ["central"]

    def test_summary_has_all_required_columns(self):
        """Summary DataFrame should have scenario, profit, gwp, loss_ratio, retention."""
        results = [_make_dummy_result(5) for _ in range(3)]
        sr = ScenarioResult(
            results=results,
            scenario_names=["a", "b", "c"],
            multiplier_mean=np.ones(5),
            multiplier_p10=np.zeros(5),
            multiplier_p90=np.ones(5) * 2,
            profit_mean=5000.0,
            profit_p10=4000.0,
            profit_p90=6000.0,
        )
        df = sr.summary()
        required_cols = {"scenario", "profit", "gwp", "loss_ratio", "retention"}
        assert required_cols.issubset(set(df.columns))

    def test_summary_converged_column(self):
        """Summary should include converged column."""
        results = [
            _make_dummy_result(converged=True),
            _make_dummy_result(converged=False),
        ]
        sr = ScenarioResult(
            results=results,
            scenario_names=["ok", "failed"],
            multiplier_mean=np.ones(5),
            multiplier_p10=np.ones(5),
            multiplier_p90=np.ones(5),
            profit_mean=0.0,
            profit_p10=0.0,
            profit_p90=0.0,
        )
        df = sr.summary()
        assert "converged" in df.columns
        assert list(df["converged"]) == [True, False]


# ---------------------------------------------------------------------------
# EfficientFrontierResult: all converged and all failed
# ---------------------------------------------------------------------------


class TestEfficientFrontierResultExtended:
    def _make_fp(self, epsilon: float, converged: bool = True) -> FrontierPoint:
        r = _make_dummy_result(converged=converged)
        return FrontierPoint(epsilon=epsilon, result=r)

    def test_all_converged_pareto_data_has_all_rows(self):
        """When all points converged, pareto_data() should return all rows."""
        points = [self._make_fp(eps) for eps in [0.85, 0.90, 0.95]]
        fr = EfficientFrontierResult(points=points, sweep_param="test")
        assert len(fr.pareto_data()) == 3

    def test_all_failed_pareto_data_empty(self):
        """When all points failed, pareto_data() should return an empty DataFrame."""
        points = [self._make_fp(eps, converged=False) for eps in [0.85, 0.90]]
        fr = EfficientFrontierResult(points=points, sweep_param="test")
        pareto = fr.pareto_data()
        assert len(pareto) == 0

    def test_frontier_point_stores_epsilon(self):
        """FrontierPoint.epsilon should store the value passed."""
        fp = self._make_fp(epsilon=0.723)
        assert fp.epsilon == pytest.approx(0.723)

    def test_data_epsilon_column_values(self):
        """data.epsilon column should match the epsilons of the frontier points."""
        epsilons = [0.80, 0.85, 0.90, 0.95]
        points = [self._make_fp(eps) for eps in epsilons]
        fr = EfficientFrontierResult(points=points, sweep_param="lr_range")
        np.testing.assert_allclose(
            fr.data["epsilon"].to_list(), epsilons, rtol=1e-10
        )

    def test_sweep_param_stored(self):
        """EfficientFrontierResult stores sweep_param."""
        points = [self._make_fp(0.9)]
        fr = EfficientFrontierResult(points=points, sweep_param="volume_retention")
        assert fr.sweep_param == "volume_retention"


# ---------------------------------------------------------------------------
# _json_default: edge cases
# ---------------------------------------------------------------------------


class TestJsonDefaultExtended:
    def test_numpy_int32(self):
        """np.int32 should be converted to Python int."""
        val = np.int32(99)
        assert _json_default(val) == 99
        assert isinstance(_json_default(val), int)

    def test_numpy_float32(self):
        """np.float32 should be converted to Python float."""
        val = np.float32(1.5)
        result = _json_default(val)
        assert isinstance(result, float)
        assert abs(result - 1.5) < 1e-5

    def test_nested_numpy_not_handled(self):
        """_json_default does not handle non-numpy objects — should raise."""
        with pytest.raises(TypeError):
            _json_default({"key": "value"})
