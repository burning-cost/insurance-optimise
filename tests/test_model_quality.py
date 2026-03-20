"""
Tests for the model_quality module (Hedges 2025, arXiv:2512.03242).

These tests verify:
- The Theorem 1 closed-form formula behaves correctly at boundary and interior points
- Monotonicity and diminishing-returns properties
- Frequency-severity product structure (Theorem 2)
- Calibration roundtrip (recover eta from LR)
- ModelQualityReport field population
- ConstraintConfig validation for model quality fields
"""

from __future__ import annotations

import pytest
import numpy as np

from insurance_optimise.model_quality import (
    loss_ratio_formula,
    loss_ratio_error,
    lr_improvement_ratio,
    frequency_severity_lr,
    model_quality_report,
    calibrate_elasticity_from_data,
    ModelQualityReport,
)
from insurance_optimise.constraints import ConstraintConfig


# ---------------------------------------------------------------------------
# 1. Perfect model: rho=1 gives LR = 1/M
# ---------------------------------------------------------------------------


def test_perfect_model_lr_equals_1_over_M():
    """With rho=1.0, the formula must return exactly 1/M regardless of CV or eta."""
    M = 1.15
    cv_lambda = 2.5
    eta = 1.8
    lr = loss_ratio_formula(rho=1.0, cv_lambda=cv_lambda, eta=eta, M=M)
    assert abs(lr - 1.0 / M) < 1e-10, f"Expected {1/M:.6f}, got {lr:.6f}"


def test_perfect_model_various_params():
    """Spot-check perfect model for several (CV, eta, M) combinations."""
    for M in [1.0, 1.25, 1.4286]:
        for cv in [0.5, 1.5, 3.0]:
            for eta in [0.8, 1.5, 3.0]:
                lr = loss_ratio_formula(rho=1.0, cv_lambda=cv, eta=eta, M=M)
                assert abs(lr - 1.0 / M) < 1e-9


# ---------------------------------------------------------------------------
# 2. LRE is zero at rho=1
# ---------------------------------------------------------------------------


def test_lre_is_zero_at_rho_one():
    """loss_ratio_error must return 0.0 when rho=1."""
    lre = loss_ratio_error(rho=1.0, cv_lambda=2.5, eta=1.5)
    assert lre == 0.0


def test_lre_positive_for_imperfect_model():
    """For rho < 1 and typical eta > 0.5, E_LR > 0 (LR > perfect-model LR)."""
    lre = loss_ratio_error(rho=0.80, cv_lambda=1.5, eta=1.5)
    assert lre > 0.0, f"Expected LRE > 0, got {lre}"


# ---------------------------------------------------------------------------
# 3. Diminishing returns: marginal gain from rho improvement decreases
# ---------------------------------------------------------------------------


def test_diminishing_returns_monotone():
    """
    For a fixed delta_rho=0.10, the absolute LR improvement strictly decreases
    as rho_old increases from 0.5 to 0.8.

    That is: improvement(0.5->0.6) > improvement(0.6->0.7) > improvement(0.7->0.8).
    """
    cv_lambda = 1.5
    eta = 1.5
    delta = 0.10
    rho_starts = [0.50, 0.60, 0.70, 0.80]
    improvements = []
    for rho_old in rho_starts:
        rho_new = rho_old + delta
        lr_old = loss_ratio_formula(rho_old, cv_lambda, eta)
        lr_new = loss_ratio_formula(rho_new, cv_lambda, eta)
        improvements.append(lr_old - lr_new)  # positive = better LR

    # Check strict monotone decrease: each improvement is smaller than the previous
    for i in range(1, len(improvements)):
        assert improvements[i] < improvements[i - 1], (
            f"Expected diminishing returns: improvement[{i}]={improvements[i]:.6f} "
            f"should be < improvement[{i-1}]={improvements[i-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Frequency-severity product structure
# ---------------------------------------------------------------------------


def test_frequency_severity_product_structure():
    """
    When rho_f = rho_s = rho and CV_f = CV_s = CV, the frequency-severity LR
    should equal (1/M) * [F(rho, CV, eta)]^2, where F is the Theorem 1 factor.

    Equivalently: frequency_severity_lr = loss_ratio_formula(rho, CV, eta, M=1)^2 / M
    """
    rho = 0.85
    cv = 1.2
    eta = 1.5
    M = 1.4286

    lr_freq_sev = frequency_severity_lr(
        rho_f=rho, rho_s=rho, cv_f=cv, cv_s=cv, eta=eta, M=M
    )

    # F(rho, CV, eta) with M=1
    factor = loss_ratio_formula(rho, cv, eta, M=1.0)
    expected = (1.0 / M) * factor * factor

    assert abs(lr_freq_sev - expected) < 1e-12, (
        f"frequency_severity_lr={lr_freq_sev:.8f}, expected={expected:.8f}"
    )


def test_frequency_severity_worse_than_single():
    """
    The frequency-severity LR (both models imperfect) should be higher
    (worse) than a single-model LR with the same rho.

    LR_freq_sev = (1/M) * F_f * F_s  vs  LR_single = (1/M) * F
    Both factors > 1 for rho < 1, so product > single factor.
    """
    rho = 0.85
    cv = 1.2
    eta = 1.5
    M = 1.0

    lr_single = loss_ratio_formula(rho, cv, eta, M=M)
    lr_fs = frequency_severity_lr(rho, rho, cv, cv, eta, M=M)
    assert lr_fs > lr_single, (
        f"Expected freq-sev LR {lr_fs:.4f} > single LR {lr_single:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Calibrate elasticity roundtrip
# ---------------------------------------------------------------------------


def test_calibrate_elasticity_roundtrip():
    """
    Set true eta=1.8, compute the implied LR, then recover eta via
    calibrate_elasticity_from_data. Should recover to within 1e-4.
    """
    true_eta = 1.8
    rho = 0.82
    cv_lambda = 1.3
    M = 1.0

    lr_true = loss_ratio_formula(rho, cv_lambda, true_eta, M=M)
    recovered_eta = calibrate_elasticity_from_data(
        rho_observed=rho,
        lr_observed=lr_true,
        cv_lambda=cv_lambda,
        M=M,
        eta_bounds=(0.5, 5.0),
    )

    assert recovered_eta is not None, "calibrate_elasticity_from_data returned None"
    assert abs(recovered_eta - true_eta) < 1e-4, (
        f"Roundtrip failed: true eta={true_eta}, recovered={recovered_eta:.6f}"
    )


def test_calibrate_returns_none_for_impossible_lr():
    """
    If the observed LR is outside the achievable range in eta_bounds,
    the function should return None (not raise).
    """
    # rho=0.9, CV=1.0, eta in [0.5,5.0]: LR will be in a bounded range
    # Setting lr_observed extremely high (e.g. 100) should return None
    result = calibrate_elasticity_from_data(
        rho_observed=0.9,
        lr_observed=100.0,
        cv_lambda=1.0,
        M=1.0,
        eta_bounds=(0.5, 5.0),
    )
    assert result is None


# ---------------------------------------------------------------------------
# 6. lr_improvement_ratio < 1 when rho_new > rho_old
# ---------------------------------------------------------------------------


def test_lr_improvement_ratio_less_than_one():
    """
    Moving from a weaker to a stronger model (rho_new > rho_old) should
    give a ratio < 1.0 — the new model achieves a lower expected LR.
    """
    cv_lambda = 1.5
    eta = 1.5
    rho_old = 0.75
    rho_new = 0.90

    ratio = lr_improvement_ratio(rho_old, rho_new, cv_lambda, eta)
    assert ratio < 1.0, f"Expected ratio < 1.0, got {ratio:.4f}"


def test_lr_improvement_ratio_equals_one_same_rho():
    """Same rho => ratio = 1.0 exactly."""
    ratio = lr_improvement_ratio(0.85, 0.85, 1.5, 1.5)
    assert abs(ratio - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 7. model_quality_report fields
# ---------------------------------------------------------------------------


def test_model_quality_report_fields():
    """
    model_quality_report should return a ModelQualityReport with all fields
    populated and of the correct type.
    """
    rho = 0.85
    cv_lambda = 1.2
    eta = 1.5
    M = 1.4286  # 70% target LR

    report = model_quality_report(rho=rho, cv_lambda=cv_lambda, eta=eta, M=M)

    assert isinstance(report, ModelQualityReport)
    assert isinstance(report.rho, float)
    assert isinstance(report.cv_lambda, float)
    assert isinstance(report.eta, float)
    assert isinstance(report.M, float)
    assert isinstance(report.lr_perfect, float)
    assert isinstance(report.lr_expected, float)
    assert isinstance(report.lre, float)
    assert isinstance(report.lr_adjustment_bps, float)
    assert isinstance(report.improvement_table, np.ndarray)

    # Check internal consistency
    assert abs(report.lr_perfect - 1.0 / M) < 1e-10
    assert abs(report.lre - (report.lr_expected - report.lr_perfect)) < 1e-12
    assert abs(report.lr_adjustment_bps - report.lre * 10_000) < 1e-6

    # Improvement table: shape (G, 3), columns [rho, lr, lre]
    assert report.improvement_table.ndim == 2
    assert report.improvement_table.shape[1] == 3

    # All rows should have rho column in [0, 1]
    assert np.all(report.improvement_table[:, 0] <= 1.0)
    assert np.all(report.improvement_table[:, 0] > 0.0)


def test_model_quality_report_custom_grid():
    """Custom rho_grid is respected in improvement_table."""
    grid = np.array([0.6, 0.7, 0.8, 0.9])
    report = model_quality_report(rho=0.85, cv_lambda=1.0, eta=1.5, M=1.0, rho_grid=grid)
    assert report.improvement_table.shape == (4, 3)
    np.testing.assert_allclose(report.improvement_table[:, 0], grid, atol=1e-10)


# ---------------------------------------------------------------------------
# 8. ConstraintConfig validation for model quality fields
# ---------------------------------------------------------------------------


def test_constraint_config_model_quality_requires_rho():
    """
    Setting model_quality_adjusted_lr=True without model_rho should raise ValueError.
    """
    config = ConstraintConfig(
        lr_max=0.70,
        model_quality_adjusted_lr=True,
        model_rho=None,
    )
    with pytest.raises(ValueError, match="model_rho"):
        config.validate()


def test_constraint_config_model_quality_valid():
    """
    Setting model_quality_adjusted_lr=True with model_rho should pass validation.
    """
    config = ConstraintConfig(
        lr_max=0.70,
        model_quality_adjusted_lr=True,
        model_rho=0.85,
        model_cv_lambda=1.2,
    )
    config.validate()  # should not raise


def test_constraint_config_model_quality_default_off():
    """By default, model quality adjustment is off and no rho is needed."""
    config = ConstraintConfig(lr_max=0.70)
    config.validate()  # should not raise
    assert config.model_quality_adjusted_lr is False
    assert config.model_rho is None


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------


def test_invalid_rho_raises():
    with pytest.raises(ValueError, match="rho"):
        loss_ratio_formula(rho=0.0, cv_lambda=1.0, eta=1.5)


def test_invalid_cv_raises():
    with pytest.raises(ValueError, match="cv_lambda"):
        loss_ratio_formula(rho=0.8, cv_lambda=-0.5, eta=1.5)


def test_invalid_eta_raises():
    with pytest.raises(ValueError, match="eta"):
        loss_ratio_formula(rho=0.8, cv_lambda=1.0, eta=-1.0)


def test_invalid_M_raises():
    with pytest.raises(ValueError, match="M"):
        loss_ratio_formula(rho=0.8, cv_lambda=1.0, eta=1.5, M=0.0)
