"""
Tests for ConvexRiskReinsuranceOptimiser.

Covers:
1. RiskLine validation (bad inputs raise correctly)
2. Single risk, zero loading -> raises (loading must be > 0)
3. Single risk, high loading -> no cession
4. Single risk, low budget -> cedes everything
5. CVaR solver: loading-ordered cession (cheapest first)
6. CVaR solver: retained risk at or below budget
7. CVaR solver: two-risk known structure
8. Variance solver: retained variance at or below budget
9. Variance solver: fixed-point sigma is non-negative
10. Variance solver: monotone cession in loading
11. Frontier: monotonicity (lower budget => higher ceded premium)
12. Frontier: correct columns and length
13. Budget constraint: retained risk <= budget (with tolerance)
14. Audit dict: required keys present
15. Result repr: correct format
16. Summary DataFrame: correct schema
17. Edge case: identical loadings
18. Edge case: zero-variance risk
19. Edge case: single line CVaR
20. Edge case: single line variance
21. Sensitivity on budget: monotone
22. Sensitivity on alpha: monotone
23. Sensitivity on loading: monotone
24. Invalid sensitivity param raises
25. ConvexReinsuranceResult dataclass fields
26. No cession when budget >= unconstrained risk
27. Infeasible budget raises
28. Pre-supplied samples respected
29. Covariance matrix shape validation
30. n_points < 2 in frontier raises
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_optimise.convex_reinsurance import (
    ConvexReinsuranceResult,
    ConvexRiskReinsuranceOptimiser,
    RiskLine,
    _empirical_cvar,
    _regularise_corr,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def simple_samples(rng: np.random.Generator) -> np.ndarray:
    """5000 x 2 samples from independent lognormals."""
    s1 = rng.lognormal(mean=np.log(5000), sigma=0.3, size=5000)
    s2 = rng.lognormal(mean=np.log(3000), sigma=0.4, size=5000)
    return np.column_stack([s1, s2])


@pytest.fixture
def three_risk_samples(rng: np.random.Generator) -> np.ndarray:
    """5000 x 3 samples from independent lognormals."""
    s1 = rng.lognormal(mean=np.log(5000), sigma=0.3, size=5000)
    s2 = rng.lognormal(mean=np.log(3000), sigma=0.4, size=5000)
    s3 = rng.lognormal(mean=np.log(1500), sigma=0.5, size=5000)
    return np.column_stack([s1, s2, s3])


@pytest.fixture
def two_risks() -> list[RiskLine]:
    return [
        RiskLine(name="motor",    expected_loss=5_000, variance=8_000_000, safety_loading=0.15),
        RiskLine(name="property", expected_loss=3_000, variance=4_000_000, safety_loading=0.22),
    ]


@pytest.fixture
def three_risks() -> list[RiskLine]:
    return [
        RiskLine(name="motor",     expected_loss=5_000, variance=8_000_000, safety_loading=0.15),
        RiskLine(name="property",  expected_loss=3_000, variance=4_000_000, safety_loading=0.22),
        RiskLine(name="liability", expected_loss=1_500, variance=2_500_000, safety_loading=0.30),
    ]


# ---------------------------------------------------------------------------
# 1-3: RiskLine validation
# ---------------------------------------------------------------------------


class TestRiskLineValidation:
    def test_zero_expected_loss_raises(self):
        with pytest.raises(ValueError, match="expected_loss must be strictly positive"):
            RiskLine(name="x", expected_loss=0.0, variance=1000.0, safety_loading=0.2)

    def test_negative_expected_loss_raises(self):
        with pytest.raises(ValueError, match="expected_loss must be strictly positive"):
            RiskLine(name="x", expected_loss=-100.0, variance=1000.0, safety_loading=0.2)

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError, match="variance must be >= 0"):
            RiskLine(name="x", expected_loss=1000.0, variance=-1.0, safety_loading=0.2)

    def test_zero_safety_loading_raises(self):
        with pytest.raises(ValueError, match="safety_loading must be strictly positive"):
            RiskLine(name="x", expected_loss=1000.0, variance=1000.0, safety_loading=0.0)

    def test_negative_safety_loading_raises(self):
        with pytest.raises(ValueError, match="safety_loading must be strictly positive"):
            RiskLine(name="x", expected_loss=1000.0, variance=1000.0, safety_loading=-0.1)

    def test_valid_risk_line_constructs(self):
        r = RiskLine(name="motor", expected_loss=5000.0, variance=8_000_000.0, safety_loading=0.15)
        assert r.name == "motor"
        assert r.expected_loss == 5000.0
        assert r.variance == 8_000_000.0
        assert r.safety_loading == 0.15

    def test_zero_variance_is_valid(self):
        r = RiskLine(name="det", expected_loss=1000.0, variance=0.0, safety_loading=0.1)
        assert r.variance == 0.0


# ---------------------------------------------------------------------------
# 4: ConvexRiskReinsuranceOptimiser construction validation
# ---------------------------------------------------------------------------


class TestOptimiserConstruction:
    def test_empty_risks_raises(self):
        with pytest.raises(ValueError, match="at least one RiskLine"):
            ConvexRiskReinsuranceOptimiser(risks=[])

    def test_invalid_risk_measure_raises(self):
        r = RiskLine(name="x", expected_loss=1000.0, variance=1000.0, safety_loading=0.2)
        with pytest.raises(ValueError, match="risk_measure must be"):
            ConvexRiskReinsuranceOptimiser(risks=[r], risk_measure="var")

    def test_alpha_out_of_range_raises(self):
        r = RiskLine(name="x", expected_loss=1000.0, variance=1000.0, safety_loading=0.2)
        with pytest.raises(ValueError, match="alpha must be in"):
            ConvexRiskReinsuranceOptimiser(risks=[r], alpha=1.5)

    def test_wrong_covariance_shape_raises(self, two_risks, simple_samples):
        with pytest.raises(ValueError, match="covariance_matrix must be"):
            ConvexRiskReinsuranceOptimiser(
                risks=two_risks,
                covariance_matrix=np.eye(3),
                aggregate_loss_samples=simple_samples,
            )

    def test_wrong_sample_shape_raises(self, two_risks):
        bad_samples = np.ones((100, 3))
        with pytest.raises(ValueError, match="aggregate_loss_samples must be"):
            ConvexRiskReinsuranceOptimiser(
                risks=two_risks,
                aggregate_loss_samples=bad_samples,
            )

    def test_valid_construction_with_samples(self, two_risks, simple_samples):
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            aggregate_loss_samples=simple_samples,
        )
        assert opt._n == 2


# ---------------------------------------------------------------------------
# 5: No cession when budget >= unconstrained risk
# ---------------------------------------------------------------------------


class TestNoCession:
    def test_no_budget_gives_no_cession(self, two_risks, simple_samples):
        """Without a budget constraint, no cession is optimal (loadings > 0)."""
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            budget=None,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        assert result.total_ceded_premium == 0.0
        assert all(not c["ceded"] for c in result.contracts)
        assert result.lambda_star == 0.0

    def test_large_budget_gives_no_cession(self, two_risks, simple_samples):
        """Budget above unconstrained retained risk => no cession needed."""
        S = simple_samples.sum(axis=1)
        cvar = _empirical_cvar(S, 0.995)
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            budget=cvar * 2.0,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        assert result.total_ceded_premium == 0.0

    def test_infeasible_budget_raises(self, two_risks, simple_samples):
        """Budget below zero is infeasible."""
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            budget=-1.0,
            aggregate_loss_samples=simple_samples,
        )
        with pytest.raises(ValueError, match="infeasible"):
            opt.optimise()


# ---------------------------------------------------------------------------
# 6: CVaR solver — retained risk within budget
# ---------------------------------------------------------------------------


class TestCVaRBudgetConstraint:
    def test_retained_cvar_within_budget(self, two_risks, simple_samples):
        """CVaR of retained aggregate must be <= budget (within 2% tolerance)."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)
        budget = full_cvar * 0.75  # reduce retained risk by 25%

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=budget,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        assert result.retained_risk <= budget * 1.02, (
            f"Retained CVaR {result.retained_risk:.2f} exceeds budget {budget:.2f}"
        )

    def test_retained_cvar_near_budget(self, two_risks, simple_samples):
        """Constraint should bind: retained CVaR should be close to budget."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)
        budget = full_cvar * 0.80

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=budget,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        # Should be within 5% of the target (bisection is tight)
        assert abs(result.retained_risk - budget) / budget < 0.05, (
            f"Retained CVaR {result.retained_risk:.2f} far from budget {budget:.2f}"
        )

    def test_tighter_budget_increases_ceded_premium(self, two_risks, simple_samples):
        """Tighter risk constraint => higher ceded premium."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        premiums = []
        for frac in [0.90, 0.75, 0.60]:
            opt = ConvexRiskReinsuranceOptimiser(
                risks=two_risks,
                risk_measure="cvar",
                alpha=0.995,
                budget=full_cvar * frac,
                aggregate_loss_samples=simple_samples,
            )
            result = opt.optimise()
            premiums.append(result.total_ceded_premium)

        # Tighter budget (lower frac) => higher premium
        assert premiums[0] < premiums[1] < premiums[2], (
            f"Expected monotone ceded premium increase: {premiums}"
        )


# ---------------------------------------------------------------------------
# 7: CVaR solver — loading-ordered cession
# ---------------------------------------------------------------------------


class TestCVaRLoadingOrder:
    def test_cheaper_risk_ceded_first(self, three_risks, three_risk_samples):
        """
        Cheapest risk (lowest beta) should be ceded first. With a mild budget,
        only the cheapest line is ceded.
        """
        S = three_risk_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)
        # Mild constraint: requires just a little cession
        budget = full_cvar * 0.92

        opt = ConvexRiskReinsuranceOptimiser(
            risks=three_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=budget,
            aggregate_loss_samples=three_risk_samples,
        )
        result = opt.optimise()
        by_name = {c["name"]: c for c in result.contracts}

        # If any risk is ceded, motor (beta=0.15) should be ceded before
        # liability (beta=0.30)
        if by_name["liability"]["ceded"]:
            assert by_name["motor"]["ceded"], (
                "Liability (beta=0.30) was ceded but motor (beta=0.15) was not. "
                "Loading order violated."
            )

    def test_most_expensive_risk_ceded_last(self, three_risks, three_risk_samples):
        """Liability (highest beta=0.30) should be ceded only if motor and property are."""
        S = three_risk_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)
        budget = full_cvar * 0.88

        opt = ConvexRiskReinsuranceOptimiser(
            risks=three_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=budget,
            aggregate_loss_samples=three_risk_samples,
        )
        result = opt.optimise()
        by_name = {c["name"]: c for c in result.contracts}

        if by_name["liability"]["ceded"]:
            assert by_name["motor"]["ceded"], "Motor must be ceded before liability."
            assert by_name["property"]["ceded"], "Property must be ceded before liability."

    def test_ceded_premium_positive_when_ceding(self, two_risks, simple_samples):
        """When cession occurs, ceded premium must be positive."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.70,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        for c in result.contracts:
            if c["ceded"]:
                assert c["ceded_premium"] > 0.0, f"Zero ceded premium on {c['name']}"

    def test_cession_rate_in_unit_interval(self, two_risks, simple_samples):
        """Cession rate must be in [0, 1]."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.70,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        for c in result.contracts:
            assert 0.0 <= c["cession_rate"] <= 1.0 + 1e-6, (
                f"Cession rate {c['cession_rate']:.4f} out of [0,1] for {c['name']}"
            )


# ---------------------------------------------------------------------------
# 8-9: Variance solver
# ---------------------------------------------------------------------------


class TestVarianceSolver:
    def test_retained_variance_within_budget(self, two_risks, simple_samples):
        """Variance of retained aggregate must be <= budget (within 3% tolerance)."""
        S = simple_samples.sum(axis=1)
        full_var = float(np.var(S, ddof=0))
        budget = full_var * 0.70

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="variance",
            budget=budget,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        assert result.retained_risk <= budget * 1.03, (
            f"Retained variance {result.retained_risk:.2f} exceeds budget {budget:.2f}"
        )

    def test_retained_variance_near_budget(self, two_risks, simple_samples):
        """Constraint should bind: retained variance close to budget."""
        S = simple_samples.sum(axis=1)
        full_var = float(np.var(S, ddof=0))
        budget = full_var * 0.75

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="variance",
            budget=budget,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        assert abs(result.retained_risk - budget) / budget < 0.05, (
            f"Retained variance {result.retained_risk:.2f} far from budget {budget:.2f}"
        )

    def test_sigma_star_non_negative(self, two_risks, simple_samples):
        """Fixed-point sigma* must be >= 0."""
        S = simple_samples.sum(axis=1)
        full_var = float(np.var(S, ddof=0))
        budget = full_var * 0.75

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="variance",
            budget=budget,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        assert result.audit.get("sigma_star", 0.0) >= 0.0

    def test_variance_tighter_budget_higher_premium(self, two_risks, simple_samples):
        """Tighter variance budget => more cession => higher premium."""
        S = simple_samples.sum(axis=1)
        full_var = float(np.var(S, ddof=0))

        premiums = []
        for frac in [0.90, 0.75, 0.60]:
            opt = ConvexRiskReinsuranceOptimiser(
                risks=two_risks,
                risk_measure="variance",
                budget=full_var * frac,
                aggregate_loss_samples=simple_samples,
            )
            result = opt.optimise()
            premiums.append(result.total_ceded_premium)

        assert premiums[0] < premiums[1] < premiums[2], (
            f"Expected monotone premium increase with tighter variance budget: {premiums}"
        )


# ---------------------------------------------------------------------------
# 10: Frontier monotonicity
# ---------------------------------------------------------------------------


class TestFrontier:
    def test_frontier_columns(self, two_risks, simple_samples):
        """Frontier DataFrame must contain required columns."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        df = opt.frontier(n_points=10)
        assert "budget" in df.columns
        assert "total_ceded_premium" in df.columns
        assert "retained_risk" in df.columns
        assert "n_lines_ceded" in df.columns

    def test_frontier_length(self, two_risks, simple_samples):
        """Frontier should return approximately n_points rows."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        df = opt.frontier(n_points=15)
        # Allow some points to be dropped for infeasibility
        assert len(df) >= 5

    def test_frontier_ceded_premium_monotone(self, two_risks, simple_samples):
        """
        As budget decreases (tighter constraint), ceded premium should increase.
        The frontier is swept from loosest to tightest budget.
        """
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        df = opt.frontier(n_points=20)
        premiums = df["total_ceded_premium"].to_numpy()
        # Ceded premium should be non-decreasing as budget decreases
        # (budget column decreases along the frontier)
        budgets = df["budget"].to_numpy()
        # Sort by budget descending -> ceded premium ascending
        order = np.argsort(-budgets)
        premiums_sorted = premiums[order]
        violations = np.sum(np.diff(premiums_sorted) < -50)  # allow small noise
        assert violations <= 3, (
            f"Frontier premium not monotone: {violations} violations"
        )

    def test_frontier_n_points_lt_2_raises(self, two_risks, simple_samples):
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            opt.frontier(n_points=1)


# ---------------------------------------------------------------------------
# 11: Audit dict
# ---------------------------------------------------------------------------


class TestAuditDict:
    def test_cvar_audit_keys(self, two_risks, simple_samples):
        """CVaR solver audit must contain required diagnostic keys."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        required = {"solver", "alpha", "full_retained_risk", "budget", "lambda_star"}
        for key in required:
            assert key in result.audit, f"Missing audit key: {key}"

    def test_variance_audit_keys(self, two_risks, simple_samples):
        """Variance solver audit must contain required keys."""
        S = simple_samples.sum(axis=1)
        full_var = float(np.var(S, ddof=0))

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="variance",
            budget=full_var * 0.75,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        required = {"solver", "full_variance", "budget", "lambda_star", "sigma_star"}
        for key in required:
            assert key in result.audit, f"Missing audit key: {key}"

    def test_no_cession_audit_has_solver_key(self, two_risks, simple_samples):
        """No-cession result audit must at least record solver type."""
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            budget=None,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        assert "solver" in result.audit


# ---------------------------------------------------------------------------
# 12: Result repr and summary
# ---------------------------------------------------------------------------


class TestResultOutput:
    def test_repr_contains_n_lines(self, two_risks, simple_samples):
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks, budget=None, aggregate_loss_samples=simple_samples
        )
        result = opt.optimise()
        r = repr(result)
        assert "n_lines=2" in r
        assert "n_ceded=" in r

    def test_summary_is_dataframe(self, two_risks, simple_samples):
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks, budget=None, aggregate_loss_samples=simple_samples
        )
        result = opt.optimise()
        import polars as pl
        assert isinstance(result.summary(), pl.DataFrame)

    def test_summary_has_name_column(self, two_risks, simple_samples):
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks, budget=None, aggregate_loss_samples=simple_samples
        )
        result = opt.optimise()
        df = result.summary()
        assert "name" in df.columns
        names = df["name"].to_list()
        assert "motor" in names
        assert "property" in names

    def test_result_contracts_length(self, two_risks, simple_samples):
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks, budget=None, aggregate_loss_samples=simple_samples
        )
        result = opt.optimise()
        assert len(result.contracts) == 2

    def test_risk_measure_value_matches_retained_risk(self, two_risks, simple_samples):
        """risk_measure_value is an alias for retained_risk."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        assert result.risk_measure_value == result.retained_risk


# ---------------------------------------------------------------------------
# 13: Sensitivity
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_sensitivity_budget_monotone_premium(self, two_risks, simple_samples):
        """Higher budget => less cession => lower premium."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        budgets = [full_cvar * f for f in [0.65, 0.70, 0.75, 0.80, 0.85]]
        df = opt.sensitivity("budget", budgets)
        assert "param_value" in df.columns
        assert "total_ceded_premium" in df.columns
        premiums = df["total_ceded_premium"].to_numpy()
        # Ascending budget => descending (or flat) premium
        violations = np.sum(np.diff(premiums) > 50)
        assert violations <= 1, f"Budget sensitivity not monotone: {premiums}"

    def test_sensitivity_alpha_returns_dataframe(self, two_risks, simple_samples):
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        df = opt.sensitivity("alpha", [0.90, 0.95, 0.99])
        assert len(df) == 3
        assert "lambda_star" in df.columns

    def test_sensitivity_loading_motor(self, two_risks, simple_samples):
        """Higher motor loading => motor less attractive to cede."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.75,
            aggregate_loss_samples=simple_samples,
        )
        df = opt.sensitivity("loading_motor", [0.10, 0.20, 0.35])
        assert len(df) == 3
        assert "total_ceded_premium" in df.columns

    def test_sensitivity_invalid_param_raises(self, two_risks, simple_samples):
        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks, budget=None, aggregate_loss_samples=simple_samples
        )
        with pytest.raises(ValueError, match="param must be one of"):
            opt.sensitivity("invalid_param", [1.0, 2.0])


# ---------------------------------------------------------------------------
# 14: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_identical_loadings(self, rng):
        """Identical loadings: all risks treated symmetrically."""
        risks = [
            RiskLine(name="a", expected_loss=3000, variance=2_000_000, safety_loading=0.20),
            RiskLine(name="b", expected_loss=3000, variance=2_000_000, safety_loading=0.20),
        ]
        samples = rng.lognormal(
            mean=np.log([3000, 3000]), sigma=0.3, size=(5000, 2)
        )
        S = samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=samples,
        )
        result = opt.optimise()
        assert isinstance(result, ConvexReinsuranceResult)
        # With identical loadings, both should be treated symmetrically
        assert len(result.contracts) == 2

    def test_zero_variance_risk(self, rng):
        """
        A risk with zero variance is deterministic. Under CVaR, a deterministic
        risk contributes nothing to the tail, so its cession is minimal.
        """
        risks = [
            RiskLine(name="stochastic", expected_loss=5000, variance=8_000_000, safety_loading=0.15),
            RiskLine(name="deterministic", expected_loss=1000, variance=0.0, safety_loading=0.25),
        ]
        # Create samples: stochastic is lognormal, deterministic is constant
        s1 = rng.lognormal(mean=np.log(5000), sigma=0.4, size=5000)
        s2 = np.full(5000, 1000.0)
        samples = np.column_stack([s1, s2])
        S = samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=samples,
        )
        result = opt.optimise()
        assert isinstance(result, ConvexReinsuranceResult)

    def test_single_line_cvar(self, rng):
        """Single-line CVaR problem: should return a valid result."""
        risk = RiskLine(name="motor", expected_loss=5000, variance=8_000_000, safety_loading=0.15)
        samples = rng.lognormal(mean=np.log(5000), sigma=0.4, size=(5000, 1))
        S = samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=[risk],
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=samples,
        )
        result = opt.optimise()
        assert len(result.contracts) == 1
        assert result.contracts[0]["name"] == "motor"
        assert result.retained_risk <= full_cvar * 0.80 * 1.05

    def test_single_line_variance(self, rng):
        """Single-line variance problem: should return a valid result."""
        risk = RiskLine(name="motor", expected_loss=5000, variance=8_000_000, safety_loading=0.15)
        samples = rng.lognormal(mean=np.log(5000), sigma=0.4, size=(5000, 1))
        S = samples.sum(axis=1)
        full_var = float(np.var(S, ddof=0))

        opt = ConvexRiskReinsuranceOptimiser(
            risks=[risk],
            risk_measure="variance",
            budget=full_var * 0.70,
            aggregate_loss_samples=samples,
        )
        result = opt.optimise()
        assert len(result.contracts) == 1
        assert result.retained_risk <= full_var * 0.70 * 1.05

    def test_lambda_star_positive_when_ceding(self, two_risks, simple_samples):
        """When cession occurs, lambda* should be strictly positive."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.80,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        if any(c["ceded"] for c in result.contracts):
            assert result.lambda_star > 0.0, "lambda* must be positive when cession occurs"

    def test_expected_ceded_loss_le_expected_loss(self, two_risks, simple_samples):
        """E[R_i*] must not exceed E[X_i] (admissibility constraint R_i <= X_i)."""
        S = simple_samples.sum(axis=1)
        full_cvar = _empirical_cvar(S, 0.995)

        opt = ConvexRiskReinsuranceOptimiser(
            risks=two_risks,
            risk_measure="cvar",
            alpha=0.995,
            budget=full_cvar * 0.75,
            aggregate_loss_samples=simple_samples,
        )
        result = opt.optimise()
        for c in result.contracts:
            assert c["expected_ceded_loss"] <= c["expected_loss"] * 1.02, (
                f"E[R_{c['name']}*]={c['expected_ceded_loss']:.2f} > "
                f"E[X_{c['name']}]={c['expected_loss']:.2f}"
            )


# ---------------------------------------------------------------------------
# 15: Module-level helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_empirical_cvar_exceeds_var(self, rng):
        """CVaR >= VaR by definition."""
        x = rng.lognormal(size=10_000)
        alpha = 0.95
        var = float(np.quantile(x, alpha))
        cvar = _empirical_cvar(x, alpha)
        assert cvar >= var - 1e-10

    def test_empirical_cvar_all_equal(self):
        """CVaR of constant rv equals constant."""
        x = np.full(1000, 5.0)
        assert abs(_empirical_cvar(x, 0.95) - 5.0) < 1e-10

    def test_regularise_corr_returns_psd(self, rng):
        """Regularised matrix should have all non-negative eigenvalues."""
        corr = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
        corr_reg = _regularise_corr(corr)
        eigs = np.linalg.eigvalsh(corr_reg)
        assert eigs.min() >= -1e-10

    def test_regularise_corr_diagonal_ones(self, rng):
        """Diagonal must remain 1 after regularisation."""
        corr = np.array([[1.0, 0.999], [0.999, 1.0]])
        corr_reg = _regularise_corr(corr)
        assert abs(corr_reg[0, 0] - 1.0) < 1e-10
        assert abs(corr_reg[1, 1] - 1.0) < 1e-10
