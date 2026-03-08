"""
Tests for result types (OptimisationResult, ScenarioResult,
EfficientFrontierResult).
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


def _make_dummy_result(n: int = 5) -> OptimisationResult:
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
        "convergence": {"converged": True},
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
        converged=True,
        solver_message="Optimization terminated successfully",
        n_iter=45,
        audit_trail=audit,
        summary_df=summary_df,
    )


class TestOptimisationResult:

    def test_repr_converged(self):
        result = _make_dummy_result()
        assert "CONVERGED" in repr(result)

    def test_repr_not_converged(self):
        result = _make_dummy_result()
        result = OptimisationResult(
            **{**result.__dict__, "converged": False}
        )
        assert "NOT CONVERGED" in repr(result)

    def test_to_json_valid(self):
        result = _make_dummy_result()
        s = result.to_json()
        data = json.loads(s)
        assert "library" in data

    def test_save_audit_creates_file(self, tmp_path):
        result = _make_dummy_result()
        path = str(tmp_path / "test_audit.json")
        result.save_audit(path)
        with open(path) as f:
            data = json.load(f)
        assert "convergence" in data

    def test_multipliers_array(self):
        result = _make_dummy_result()
        assert isinstance(result.multipliers, np.ndarray)

    def test_summary_df_is_polars(self):
        result = _make_dummy_result()
        assert isinstance(result.summary_df, pl.DataFrame)


class TestScenarioResult:

    def test_summary_dataframe(self):
        results = [_make_dummy_result(5) for _ in range(3)]
        sr = ScenarioResult(
            results=results,
            scenario_names=["low", "mid", "high"],
            multiplier_mean=np.ones(5),
            multiplier_p10=np.ones(5) * 0.9,
            multiplier_p90=np.ones(5) * 1.1,
            profit_mean=10000.0,
            profit_p10=8000.0,
            profit_p90=12000.0,
        )
        df = sr.summary()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert "scenario" in df.columns
        assert "profit" in df.columns

    def test_scenario_names_in_summary(self):
        results = [_make_dummy_result() for _ in range(2)]
        sr = ScenarioResult(
            results=results,
            scenario_names=["pessimistic", "optimistic"],
            multiplier_mean=np.ones(5),
            multiplier_p10=np.ones(5),
            multiplier_p90=np.ones(5),
            profit_mean=0.0,
            profit_p10=0.0,
            profit_p90=0.0,
        )
        df = sr.summary()
        assert list(df["scenario"]) == ["pessimistic", "optimistic"]


class TestEfficientFrontierResult:

    def _make_fp(self, epsilon: float, converged: bool = True) -> FrontierPoint:
        r = _make_dummy_result()
        if not converged:
            r = OptimisationResult(**{**r.__dict__, "converged": False})
        return FrontierPoint(epsilon=epsilon, result=r)

    def test_data_columns(self):
        points = [self._make_fp(eps) for eps in [0.85, 0.90, 0.95]]
        fr = EfficientFrontierResult(points=points, sweep_param="volume_retention")
        assert "epsilon" in fr.data.columns
        assert "profit" in fr.data.columns
        assert "loss_ratio" in fr.data.columns

    def test_pareto_data_filters_converged(self):
        points = [
            self._make_fp(0.85, converged=True),
            self._make_fp(0.90, converged=False),
            self._make_fp(0.95, converged=True),
        ]
        fr = EfficientFrontierResult(points=points, sweep_param="volume_retention")
        pareto = fr.pareto_data()
        assert len(pareto) == 2

    def test_frontier_row_count_matches_points(self):
        points = [self._make_fp(eps) for eps in np.linspace(0.80, 0.99, 7)]
        fr = EfficientFrontierResult(points=points, sweep_param="volume_retention")
        assert len(fr.data) == 7


class TestJsonDefault:

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _json_default(arr) == [1.0, 2.0, 3.0]

    def test_numpy_int(self):
        val = np.int64(42)
        assert _json_default(val) == 42

    def test_numpy_float(self):
        val = np.float64(3.14)
        assert abs(_json_default(val) - 3.14) < 1e-10

    def test_numpy_bool(self):
        val = np.bool_(True)
        assert _json_default(val) is True

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            _json_default(object())
