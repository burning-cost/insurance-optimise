"""
Tests for LinearRiskSharingPool.

Covers:
- Construction and validation (direct and mean_proportional)
- Input validation errors
- validate_conditions: budget balance, actuarial fairness, capacity
- ruin_comparison: exponential exact, lognormal simulation, values in range
- ruin_comparison: pooled <= standalone under valid conditions (mean-proportional)
- simulate: output schema, ruin probabilities in [0,1], PerformanceWarning
- optimal_allocation: objective 'min_max_ruin' and 'max_min_improvement'
- audit_trail: schema, JSON-serialisable, scale family warning
- Edge cases: n=1, all-equal participants, identity matrix
- Properties: n_participants, allocation_matrix, premium_rates
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from insurance_optimise.risk_sharing import (
    LinearRiskSharingPool,
    PerformanceWarning,
    RuinResult,
    SimulationResult,
    ValidationResult,
    _cl_ruin_exponential,
    _draw_claim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paper_example():
    """n=3 example from paper (Section 3.4.1): exponential severities."""
    return LinearRiskSharingPool.mean_proportional(
        claim_intensities=np.array([2.0, 1.0, 3.0]),
        claim_means=np.array([2.0, 0.5, 1.0]),
        safety_loadings=np.array([0.4, 0.4, 0.4]),
    )


@pytest.fixture
def homogeneous_pool():
    """n=4 pool with identical participants."""
    n = 4
    lam = np.full(n, 1.0)
    b = np.full(n, 1.0)
    eta = np.full(n, 0.5)
    return LinearRiskSharingPool.mean_proportional(
        claim_intensities=lam,
        claim_means=b,
        safety_loadings=eta,
    )


@pytest.fixture
def custom_matrix_pool():
    """n=2 pool with a manually specified allocation matrix."""
    A = np.array([[0.6, 0.4], [0.4, 0.6]])
    return LinearRiskSharingPool(
        allocation_matrix=A,
        claim_intensities=np.array([1.0, 1.0]),
        claim_means=np.array([1.0, 1.0]),
        safety_loadings=np.array([0.5, 0.5]),
    )


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_mean_proportional_returns_pool(self, paper_example):
        assert isinstance(paper_example, LinearRiskSharingPool)
        assert paper_example.n_participants == 3

    def test_direct_construction(self, custom_matrix_pool):
        assert isinstance(custom_matrix_pool, LinearRiskSharingPool)
        assert custom_matrix_pool.n_participants == 2

    def test_mean_proportional_column_sums(self, paper_example):
        A = paper_example.allocation_matrix
        col_sums = A.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(3), atol=1e-12)

    def test_mean_proportional_all_columns_equal(self, paper_example):
        A = paper_example.allocation_matrix
        # Under mean-proportional, every column is the same
        for j in range(3):
            np.testing.assert_allclose(A[:, 0], A[:, j], atol=1e-12)

    def test_initial_capital_default(self, paper_example):
        np.testing.assert_array_equal(paper_example.initial_capital, np.ones(3))

    def test_initial_capital_custom(self):
        pool = LinearRiskSharingPool.mean_proportional(
            claim_intensities=np.array([1.0, 2.0]),
            claim_means=np.array([1.0, 1.0]),
            safety_loadings=np.array([0.3, 0.3]),
            initial_capital=np.array([2.0, 3.0]),
        )
        np.testing.assert_array_equal(pool.initial_capital, [2.0, 3.0])

    def test_premium_rates(self, paper_example):
        # c_i = (1 + eta) * lambda_i * b_i
        expected = np.array([1.4 * 2.0 * 2.0, 1.4 * 1.0 * 0.5, 1.4 * 3.0 * 1.0])
        np.testing.assert_allclose(paper_example.premium_rates, expected, rtol=1e-10)

    def test_repr_contains_n(self, paper_example):
        r = repr(paper_example)
        assert "n=3" in r

    def test_n1_pool(self):
        pool = LinearRiskSharingPool(
            allocation_matrix=np.array([[1.0]]),
            claim_intensities=np.array([2.0]),
            claim_means=np.array([1.0]),
            safety_loadings=np.array([0.3]),
        )
        assert pool.n_participants == 1


# ---------------------------------------------------------------------------
# 2. Input validation errors
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_allocation_shape(self):
        with pytest.raises(ValueError, match="allocation_matrix must be"):
            LinearRiskSharingPool(
                allocation_matrix=np.eye(3),
                claim_intensities=np.array([1.0, 2.0]),
                claim_means=np.array([1.0, 1.0]),
                safety_loadings=np.array([0.3, 0.3]),
            )

    def test_negative_intensity(self):
        with pytest.raises(ValueError, match="claim_intensities"):
            LinearRiskSharingPool.mean_proportional(
                claim_intensities=np.array([-1.0, 1.0]),
                claim_means=np.array([1.0, 1.0]),
                safety_loadings=np.array([0.3, 0.3]),
            )

    def test_negative_mean(self):
        with pytest.raises(ValueError, match="claim_means"):
            LinearRiskSharingPool.mean_proportional(
                claim_intensities=np.array([1.0, 1.0]),
                claim_means=np.array([1.0, -1.0]),
                safety_loadings=np.array([0.3, 0.3]),
            )

    def test_zero_loading(self):
        with pytest.raises(ValueError, match="safety_loadings"):
            LinearRiskSharingPool.mean_proportional(
                claim_intensities=np.array([1.0, 1.0]),
                claim_means=np.array([1.0, 1.0]),
                safety_loadings=np.array([0.0, 0.3]),
            )

    def test_negative_loading(self):
        with pytest.raises(ValueError, match="safety_loadings"):
            LinearRiskSharingPool.mean_proportional(
                claim_intensities=np.array([1.0, 1.0]),
                claim_means=np.array([1.0, 1.0]),
                safety_loadings=np.array([-0.1, 0.3]),
            )

    def test_negative_capital(self):
        with pytest.raises(ValueError, match="initial_capital"):
            LinearRiskSharingPool.mean_proportional(
                claim_intensities=np.array([1.0, 1.0]),
                claim_means=np.array([1.0, 1.0]),
                safety_loadings=np.array([0.3, 0.3]),
                initial_capital=np.array([-1.0, 1.0]),
            )

    def test_negative_allocation_entry(self):
        with pytest.raises(ValueError, match="non-negative"):
            LinearRiskSharingPool(
                allocation_matrix=np.array([[1.5, -0.5], [0.5, 0.5]]),  # negative entry
                claim_intensities=np.array([1.0, 1.0]),
                claim_means=np.array([1.0, 1.0]),
                safety_loadings=np.array([0.3, 0.3]),
            )


# ---------------------------------------------------------------------------
# 3. validate_conditions
# ---------------------------------------------------------------------------


class TestValidateConditions:
    def test_mean_proportional_budget_balance(self, paper_example):
        result = paper_example.validate_conditions()
        assert result.budget_balance_ok

    def test_mean_proportional_actuarial_fairness(self, paper_example):
        result = paper_example.validate_conditions()
        assert result.actuarial_fairness_ok

    def test_all_ok_for_valid_pool(self, paper_example):
        result = paper_example.validate_conditions()
        assert result.all_ok

    def test_returns_validation_result_type(self, paper_example):
        result = paper_example.validate_conditions()
        assert isinstance(result, ValidationResult)

    def test_budget_balance_violation_detected(self):
        # Column sums to 1.1 instead of 1.0
        A = np.array([[0.6, 0.4], [0.5, 0.7]])  # column 1 sums to 1.1
        pool = LinearRiskSharingPool(
            allocation_matrix=A,
            claim_intensities=np.array([1.0, 1.0]),
            claim_means=np.array([1.0, 1.0]),
            safety_loadings=np.array([0.5, 0.5]),
        )
        result = pool.validate_conditions()
        assert not result.budget_balance_ok
        assert not result.all_ok

    def test_actuarial_fairness_violation_detected(self):
        # Correct column sums but wrong row sums (not actuarially fair)
        A = np.array([[0.8, 0.2], [0.2, 0.8]])
        # lambda*b = [2, 0.5], so fairness requires row-weighted sums to match
        # With asymmetric A, this will fail unless lambda*b are equal
        pool = LinearRiskSharingPool(
            allocation_matrix=A,
            claim_intensities=np.array([2.0, 1.0]),
            claim_means=np.array([1.0, 1.0]),
            safety_loadings=np.array([0.5, 0.5]),
        )
        result = pool.validate_conditions()
        # A @ (lam*b) = [[0.8*2+0.2*1],[0.2*2+0.8*1]] = [1.8, 1.2]
        # lam*b = [2, 1] — not equal so fairness should fail
        assert not result.actuarial_fairness_ok

    def test_capacity_violation_detected(self):
        # a_{i,j} * b_j > b_i: participant 1 gets 80% of participant 2's claims
        # b_1 = 0.5, b_2 = 5.0; a_{0,1} * b_2 = 0.8 * 5 = 4 > b_0 = 0.5
        A = np.array([[0.2, 0.8], [0.8, 0.2]])
        pool = LinearRiskSharingPool(
            allocation_matrix=A,
            claim_intensities=np.array([1.0, 1.0]),
            claim_means=np.array([0.5, 5.0]),
            safety_loadings=np.array([0.5, 0.5]),
        )
        result = pool.validate_conditions()
        assert not result.capacity_ok

    def test_validation_result_repr(self, paper_example):
        result = paper_example.validate_conditions()
        r = repr(result)
        assert "VALID" in r or "INVALID" in r

    def test_identity_matrix_budget_balance(self):
        A = np.eye(3)
        pool = LinearRiskSharingPool(
            allocation_matrix=A,
            claim_intensities=np.array([1.0, 2.0, 3.0]),
            claim_means=np.array([1.0, 1.0, 1.0]),
            safety_loadings=np.array([0.3, 0.3, 0.3]),
        )
        result = pool.validate_conditions()
        assert result.budget_balance_ok
        # Identity is actuarially fair only if lambda*b are equal — they're not here
        # but check capacity: a_{i,j}*b_j for i != j is 0 <= b_i, so capacity ok
        assert result.capacity_ok


# ---------------------------------------------------------------------------
# 4. Cramér-Lundberg formula
# ---------------------------------------------------------------------------


class TestCramerLundberg:
    def test_npc_violated_returns_1(self):
        # c = lambda * b (no profit) => certain ruin
        assert _cl_ruin_exponential(lam=1.0, b=1.0, c=1.0, u=0.0) == 1.0

    def test_u0_formula(self):
        # psi(0) = lambda * b / c = 1 / (1 + eta)
        lam, b, eta = 2.0, 1.0, 0.4
        c = (1.0 + eta) * lam * b
        psi0 = _cl_ruin_exponential(lam, b, c, 0.0)
        expected = 1.0 / (1.0 + eta)
        assert abs(psi0 - expected) < 1e-10

    def test_ruin_decreases_with_capital(self):
        lam, b, eta = 1.0, 1.0, 0.5
        c = (1.0 + eta) * lam * b
        psi_0 = _cl_ruin_exponential(lam, b, c, 0.0)
        psi_1 = _cl_ruin_exponential(lam, b, c, 1.0)
        psi_5 = _cl_ruin_exponential(lam, b, c, 5.0)
        assert psi_0 > psi_1 > psi_5

    def test_ruin_in_0_1(self):
        lam, b, eta = 1.0, 2.0, 0.3
        c = (1.0 + eta) * lam * b
        for u in [0.0, 0.5, 2.0, 10.0]:
            psi = _cl_ruin_exponential(lam, b, c, u)
            assert 0.0 <= psi <= 1.0


# ---------------------------------------------------------------------------
# 5. ruin_comparison
# ---------------------------------------------------------------------------


class TestRuinComparison:
    def test_returns_ruin_result(self, paper_example):
        result = paper_example.ruin_comparison()
        assert isinstance(result, RuinResult)

    def test_shapes(self, paper_example):
        result = paper_example.ruin_comparison()
        assert result.pooled.shape == (3,)
        assert result.standalone.shape == (3,)
        assert result.improvement.shape == (3,)

    def test_probabilities_in_range(self, paper_example):
        result = paper_example.ruin_comparison()
        assert np.all(result.pooled >= 0.0)
        assert np.all(result.pooled <= 1.0)
        assert np.all(result.standalone >= 0.0)
        assert np.all(result.standalone <= 1.0)

    def test_improvement_positive_for_valid_pool(self, paper_example):
        """Under mean-proportional with valid conditions, pooling should not worsen ruin."""
        result = paper_example.ruin_comparison()
        # improvement = standalone - pooled; should be >= 0 for all i
        # (weakly positive by Proposition 3.8)
        assert np.all(result.improvement >= -1e-6)

    def test_method_cramerlundberg(self, paper_example):
        result = paper_example.ruin_comparison(method="cramerlundberg")
        assert result.method == "cramerlundberg"

    def test_cramerlundberg_rejects_lognormal(self, paper_example):
        with pytest.raises(ValueError, match="exponential"):
            paper_example.ruin_comparison(method="cramerlundberg", claim_dist="lognormal")

    def test_simulation_method(self, paper_example):
        result = paper_example.ruin_comparison(
            method="simulation", n_sim=500, time_horizon=20.0, seed=1
        )
        assert result.method == "simulation"
        assert result.pooled.shape == (3,)

    def test_improvement_sign(self, paper_example):
        result = paper_example.ruin_comparison()
        # improvement = standalone - pooled, element-wise
        np.testing.assert_allclose(
            result.improvement, result.standalone - result.pooled, atol=1e-12
        )

    def test_n1_standalone_equals_pooled(self):
        """n=1 pool: allocation matrix is [[1]], pooled should equal standalone."""
        pool = LinearRiskSharingPool(
            allocation_matrix=np.array([[1.0]]),
            claim_intensities=np.array([2.0]),
            claim_means=np.array([1.0]),
            safety_loadings=np.array([0.5]),
            initial_capital=np.array([1.0]),
        )
        result = pool.ruin_comparison()
        np.testing.assert_allclose(result.pooled, result.standalone, atol=1e-10)

    def test_ruin_result_repr(self, paper_example):
        result = paper_example.ruin_comparison()
        r = repr(result)
        assert "RuinResult" in r


# ---------------------------------------------------------------------------
# 6. simulate
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_returns_simulation_result(self, paper_example):
        sim = paper_example.simulate(T=10.0, n_paths=50, seed=0)
        assert isinstance(sim, SimulationResult)

    def test_output_shapes(self, paper_example):
        sim = paper_example.simulate(T=10.0, n_paths=50, seed=0)
        assert sim.empirical_ruin_probability.shape == (3,)
        assert sim.mean_ruin_time.shape == (3,)
        assert sim.ruin_count.shape == (3,)

    def test_probabilities_in_range(self, paper_example):
        sim = paper_example.simulate(T=20.0, n_paths=100, seed=42)
        assert np.all(sim.empirical_ruin_probability >= 0.0)
        assert np.all(sim.empirical_ruin_probability <= 1.0)

    def test_metadata(self, paper_example):
        sim = paper_example.simulate(T=30.0, n_paths=200, seed=7)
        assert sim.n_paths == 200
        assert sim.T == 30.0
        assert sim.claim_dist == "exponential"
        assert sim.n_participants == 3

    def test_lognormal_claim_dist(self, paper_example):
        sim = paper_example.simulate(
            T=10.0, n_paths=50, claim_dist="lognormal",
            claim_dist_params={"sigma": 0.5}, seed=1
        )
        assert sim.claim_dist == "lognormal"
        assert np.all(sim.empirical_ruin_probability >= 0.0)

    def test_reproducible_with_seed(self, paper_example):
        sim1 = paper_example.simulate(T=20.0, n_paths=100, seed=42)
        sim2 = paper_example.simulate(T=20.0, n_paths=100, seed=42)
        np.testing.assert_array_equal(
            sim1.empirical_ruin_probability, sim2.empirical_ruin_probability
        )

    def test_performance_warning(self, paper_example):
        # n=3, n_paths=10000, T=100, lambda_total=6 => 3*10000*6*100 = 18M > 1e7
        with pytest.warns(PerformanceWarning):
            paper_example.simulate(T=100.0, n_paths=10000, seed=0)

    def test_simulation_result_repr(self, paper_example):
        sim = paper_example.simulate(T=5.0, n_paths=10, seed=0)
        r = repr(sim)
        assert "SimulationResult" in r


# ---------------------------------------------------------------------------
# 7. optimal_allocation
# ---------------------------------------------------------------------------


class TestOptimalAllocation:
    def test_returns_pool(self, paper_example):
        opt = paper_example.optimal_allocation(objective="min_max_ruin")
        assert isinstance(opt, LinearRiskSharingPool)

    def test_does_not_mutate_original(self, paper_example):
        A_before = paper_example.allocation_matrix.copy()
        _ = paper_example.optimal_allocation(objective="min_max_ruin")
        np.testing.assert_array_equal(paper_example.allocation_matrix, A_before)

    def test_optimised_conditions_valid(self, paper_example):
        opt = paper_example.optimal_allocation(objective="min_max_ruin")
        val = opt.validate_conditions()
        # Budget balance and actuarial fairness should be approximately satisfied
        assert val.budget_balance_ok or np.max(np.abs(val.budget_balance_violations)) < 1e-4
        assert val.actuarial_fairness_ok or np.max(np.abs(val.actuarial_fairness_violations)) < 1e-4

    def test_max_min_improvement_objective(self, paper_example):
        opt = paper_example.optimal_allocation(objective="max_min_improvement")
        assert isinstance(opt, LinearRiskSharingPool)

    def test_invalid_objective_raises(self, paper_example):
        with pytest.raises(ValueError, match="Unknown objective"):
            paper_example.optimal_allocation(objective="bad_objective")

    def test_performance_warning_large_n(self):
        n = 31
        lam = np.ones(n)
        b = np.ones(n)
        eta = np.full(n, 0.5)
        pool = LinearRiskSharingPool.mean_proportional(lam, b, eta)
        with pytest.warns(PerformanceWarning):
            pool.optimal_allocation(objective="min_max_ruin")

    def test_optimised_same_params(self, paper_example):
        opt = paper_example.optimal_allocation(objective="min_max_ruin")
        np.testing.assert_array_equal(opt.claim_intensities, paper_example.claim_intensities)
        np.testing.assert_array_equal(opt.claim_means, paper_example.claim_means)
        np.testing.assert_array_equal(opt.safety_loadings, paper_example.safety_loadings)


# ---------------------------------------------------------------------------
# 8. audit_trail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_returns_dict(self, paper_example):
        trail = paper_example.audit_trail()
        assert isinstance(trail, dict)

    def test_json_serialisable(self, paper_example):
        trail = paper_example.audit_trail()
        # Should not raise
        s = json.dumps(trail)
        assert len(s) > 0

    def test_required_keys(self, paper_example):
        trail = paper_example.audit_trail()
        for key in ["library", "version", "timestamp_utc", "model_reference",
                    "n_participants", "parameters", "allocation_matrix", "validation",
                    "scale_family_warning"]:
            assert key in trail, f"Missing key: {key}"

    def test_n_participants_correct(self, paper_example):
        trail = paper_example.audit_trail()
        assert trail["n_participants"] == 3

    def test_allocation_matrix_shape(self, paper_example):
        trail = paper_example.audit_trail()
        assert trail["allocation_matrix_shape"] == [3, 3]

    def test_validation_section(self, paper_example):
        trail = paper_example.audit_trail()
        val = trail["validation"]
        assert "budget_balance_ok" in val
        assert "actuarial_fairness_ok" in val
        assert "capacity_ok" in val
        assert "all_conditions_ok" in val

    def test_scale_family_warning_present(self, paper_example):
        trail = paper_example.audit_trail()
        assert "scale family" in trail["scale_family_warning"].lower()

    def test_library_field(self, paper_example):
        trail = paper_example.audit_trail()
        assert trail["library"] == "insurance-optimise"

    def test_parameters_present(self, paper_example):
        trail = paper_example.audit_trail()
        params = trail["parameters"]
        assert "claim_intensities" in params
        assert "claim_means" in params
        assert "safety_loadings" in params
        assert "initial_capital" in params


# ---------------------------------------------------------------------------
# 9. draw_claim helper
# ---------------------------------------------------------------------------


class TestDrawClaim:
    def test_exponential_positive(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            v = _draw_claim(2.0, "exponential", {}, rng)
            assert v > 0

    def test_lognormal_positive(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            v = _draw_claim(1.0, "lognormal", {"sigma": 0.5}, rng)
            assert v > 0

    def test_gamma_positive(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            v = _draw_claim(1.5, "gamma", {"shape": 2.0}, rng)
            assert v > 0

    def test_unknown_dist_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Unknown claim distribution"):
            _draw_claim(1.0, "pareto", {}, rng)


# ---------------------------------------------------------------------------
# 10. Homogeneous pool sanity checks
# ---------------------------------------------------------------------------


class TestHomogeneousPool:
    def test_all_equal_ruin_probs(self, homogeneous_pool):
        result = homogeneous_pool.ruin_comparison()
        # All participants identical => all pooled probs equal
        np.testing.assert_allclose(
            result.pooled, result.pooled[0] * np.ones(4), atol=1e-10
        )

    def test_no_ruin_improvement_for_identity(self):
        """Identity allocation = no sharing; pooled should equal standalone."""
        A = np.eye(3)
        pool = LinearRiskSharingPool(
            allocation_matrix=A,
            claim_intensities=np.ones(3),
            claim_means=np.ones(3),
            safety_loadings=np.full(3, 0.5),
        )
        result = pool.ruin_comparison()
        np.testing.assert_allclose(result.improvement, np.zeros(3), atol=1e-10)
