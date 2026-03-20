"""
Model quality adjustment for insurance portfolio loss ratios.

Implements the closed-form relationship between Pearson correlation (rho) and
expected portfolio loss ratio from Hedges (2025), arXiv:2512.03242.

The key insight is that no pricing model is perfect. A model with Pearson
correlation rho < 1 between predicted and actual loss cost will systematically
produce a portfolio loss ratio above the perfect-model target 1/M. Theorem 1
gives a closed-form expression for how much worse.

This is important for rate setting: if you target a 70% LR but your model
has rho = 0.85, you should expect to achieve something higher. The LR
constraint in the optimiser should be adjusted upward to reflect the model's
realistic capability, not an idealistic perfect-model target.

Reference
---------
Hedges, T. (2025). "On the relationship between the Pearson correlation
coefficient and the expected loss ratio in insurance pricing."
arXiv:2512.03242.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Core formula (Theorem 1)
# ---------------------------------------------------------------------------


def loss_ratio_formula(
    rho: float,
    cv_lambda: float,
    eta: float,
    M: float = 1.0,
) -> float:
    """
    Expected portfolio loss ratio at Pearson correlation rho (Theorem 1).

    Under the Hedges (2025) framework, when a pricing model has Pearson
    correlation rho with the true loss cost, the expected portfolio loss ratio
    is:

        LR = (1/M) * ((1 + rho^2 * CV^{-2}) / (rho^2 * (1 + CV^{-2})))^{(2*eta - 1)/2}

    where CV = cv_lambda is the coefficient of variation of the loss cost
    distribution, eta is the demand elasticity parameter, and M is the target
    loss ratio multiplier (e.g. M = 1/0.70 = 1.4286 for a 70% target LR).

    Parameters
    ----------
    rho:
        Pearson correlation between model predictions and true loss cost.
        Must be in (0, 1].
    cv_lambda:
        Coefficient of variation of the loss cost distribution. Must be > 0.
        Typical values: 0.5 (homogeneous book) to 3.0 (heavy-tailed).
    eta:
        Demand elasticity parameter (price sensitivity exponent). Must be > 0.
        The formula is only well-behaved for eta > 0.5; a warning is issued
        for eta <= 0.5.
    M:
        Target loss ratio multiplier. The perfect-model LR is 1/M. Default
        1.0 means the perfect-model LR is 1.0 (break-even pricing).
        For a 70% target: M = 1/0.70 ≈ 1.4286.

    Returns
    -------
    float
        Expected portfolio loss ratio.

    Raises
    ------
    ValueError
        If rho <= 0 or cv_lambda <= 0 or eta <= 0 or M <= 0.
    """
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if cv_lambda <= 0.0:
        raise ValueError(f"cv_lambda must be > 0, got {cv_lambda}")
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0, got {eta}")
    if M <= 0.0:
        raise ValueError(f"M must be > 0, got {M}")

    if eta <= 0.5:
        warnings.warn(
            f"eta={eta} <= 0.5. The Hedges (2025) formula behaves poorly in this "
            "regime — demand is very inelastic and the power exponent (2*eta-1)/2 "
            "is negative, which may produce counterintuitive results. "
            "Typical insurance elasticity values are in (0.5, 3.0).",
            UserWarning,
            stacklevel=2,
        )

    # Special case: perfect model
    if rho >= 0.999:
        return 1.0 / M

    cv2_inv = 1.0 / (cv_lambda**2)  # CV^{-2}
    rho2 = rho**2

    # Ratio inside the power: (1 + rho^2 * CV^{-2}) / (rho^2 * (1 + CV^{-2}))
    numerator = 1.0 + rho2 * cv2_inv
    denominator = rho2 * (1.0 + cv2_inv)
    ratio = numerator / denominator

    exponent = (2.0 * eta - 1.0) / 2.0

    return (1.0 / M) * (ratio**exponent)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def loss_ratio_error(rho: float, cv_lambda: float, eta: float) -> float:
    """
    Expected loss ratio error at correlation rho versus a perfect model.

    E_LR = LR(rho) - LR(rho=1) = LR(rho) - 1.0

    This is the systematic upward bias in the portfolio LR due to model
    imperfection. With M=1.0 (break-even target), a perfect model achieves
    LR=1.0, and any rho < 1 gives LR > 1.0 — meaning the book loses money.

    Parameters
    ----------
    rho:
        Pearson correlation, in (0, 1].
    cv_lambda:
        Coefficient of variation of the loss cost distribution.
    eta:
        Demand elasticity parameter.

    Returns
    -------
    float
        Loss ratio error. Returns 0.0 when rho=1.
    """
    if rho >= 0.999:
        return 0.0
    return loss_ratio_formula(rho, cv_lambda, eta, M=1.0) - 1.0


def lr_improvement_ratio(
    rho_old: float,
    rho_new: float,
    cv_lambda: float,
    eta: float,
) -> float:
    """
    Ratio of new portfolio LR to old, quantifying model improvement benefit.

    Returns LR(rho_new) / LR(rho_old). A value < 1.0 means the new model
    achieves a better (lower) expected loss ratio.

    This is useful for quantifying the financial value of a model upgrade.
    If your book writes £100M GWP and the ratio is 0.97, the upgrade is worth
    roughly £3M/year in improved underwriting performance.

    Parameters
    ----------
    rho_old:
        Pearson correlation of the current model, in (0, 1].
    rho_new:
        Pearson correlation of the candidate model, in (0, 1].
    cv_lambda:
        Coefficient of variation of the loss cost distribution.
    eta:
        Demand elasticity parameter.

    Returns
    -------
    float
        LR_new / LR_old. Less than 1.0 if rho_new > rho_old.
    """
    lr_old = loss_ratio_formula(rho_old, cv_lambda, eta, M=1.0)
    lr_new = loss_ratio_formula(rho_new, cv_lambda, eta, M=1.0)
    if lr_old < 1e-12:
        raise ValueError("LR_old is effectively zero — check inputs.")
    return lr_new / lr_old


def frequency_severity_lr(
    rho_f: float,
    rho_s: float,
    cv_f: float,
    cv_s: float,
    eta: float,
    M: float = 1.0,
) -> float:
    """
    Expected portfolio LR for a frequency-severity model (Theorem 2).

    When frequency and severity are modelled independently (the standard
    actuarial approach), Theorem 2 states:

        LR = (1/M) * F(rho_f, CV_f, eta) * F(rho_s, CV_s, eta)

    where F(rho, CV, eta) = ((1 + rho^2*CV^{-2}) / (rho^2*(1 + CV^{-2})))^{(2*eta-1)/2}
    is the factor from Theorem 1 (with M=1).

    The product structure means errors compound: a mediocre frequency model
    AND a mediocre severity model together produce worse results than either
    alone would suggest.

    Parameters
    ----------
    rho_f:
        Pearson correlation of frequency model predictions, in (0, 1].
    rho_s:
        Pearson correlation of severity model predictions, in (0, 1].
    cv_f:
        Coefficient of variation of claim frequency.
    cv_s:
        Coefficient of variation of claim severity.
    eta:
        Demand elasticity parameter (shared for frequency and severity).
    M:
        Target loss ratio multiplier. Default 1.0.

    Returns
    -------
    float
        Expected portfolio loss ratio.
    """
    # F(rho, CV, eta) with M=1 is just the factor; divide by M once at the end
    factor_f = loss_ratio_formula(rho_f, cv_f, eta, M=1.0)
    factor_s = loss_ratio_formula(rho_s, cv_s, eta, M=1.0)
    return (1.0 / M) * factor_f * factor_s


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelQualityReport:
    """
    Summary of model quality impact on expected portfolio loss ratio.

    Generated by ``model_quality_report()``. All fields are populated.

    Attributes
    ----------
    rho:
        Pearson correlation used.
    cv_lambda:
        Coefficient of variation of loss cost.
    eta:
        Demand elasticity parameter.
    M:
        Target loss ratio multiplier.
    lr_perfect:
        Expected LR if the model were perfect (= 1/M).
    lr_expected:
        Expected LR at the given rho (Theorem 1).
    lre:
        Loss ratio error = lr_expected - lr_perfect.
    lr_adjustment_bps:
        LRE expressed in basis points (lre * 10000). Useful for discussing
        the practical magnitude of the adjustment.
    improvement_table:
        Numpy array of shape (G, 3) where G is the number of rho grid points.
        Columns: [rho_grid, lr_at_rho, lre_at_rho].
    """

    rho: float
    cv_lambda: float
    eta: float
    M: float
    lr_perfect: float
    lr_expected: float
    lre: float
    lr_adjustment_bps: float
    improvement_table: np.ndarray

    def __repr__(self) -> str:
        return (
            f"ModelQualityReport("
            f"rho={self.rho:.3f}, "
            f"cv_lambda={self.cv_lambda:.3f}, "
            f"eta={self.eta:.3f}, "
            f"M={self.M:.4f}, "
            f"lr_expected={self.lr_expected:.4f}, "
            f"lre={self.lre:+.4f}, "
            f"lr_adj={self.lr_adjustment_bps:+.1f}bps"
            f")"
        )


def model_quality_report(
    rho: float,
    cv_lambda: float,
    eta: float,
    M: float = 1.0,
    rho_grid: Optional[np.ndarray] = None,
) -> ModelQualityReport:
    """
    Generate a ModelQualityReport for the given model quality parameters.

    Parameters
    ----------
    rho:
        Pearson correlation of the pricing model, in (0, 1].
    cv_lambda:
        Coefficient of variation of the loss cost distribution.
    eta:
        Demand elasticity parameter.
    M:
        Target loss ratio multiplier. Default 1.0 (break-even).
    rho_grid:
        Array of rho values for the improvement table. Defaults to
        np.linspace(0.5, 1.0, 11) if None.

    Returns
    -------
    ModelQualityReport
    """
    if rho_grid is None:
        rho_grid = np.linspace(0.5, 1.0, 11)

    rho_grid = np.asarray(rho_grid, dtype=float)
    # Clamp to valid range
    rho_grid = np.clip(rho_grid, 1e-6, 1.0)

    lr_perfect = 1.0 / M
    lr_expected = loss_ratio_formula(rho, cv_lambda, eta, M=M)
    lre = lr_expected - lr_perfect
    lr_adjustment_bps = lre * 10_000.0

    # Build improvement table
    table_rows = []
    for r in rho_grid:
        lr_r = loss_ratio_formula(r, cv_lambda, eta, M=M)
        lre_r = lr_r - lr_perfect
        table_rows.append([r, lr_r, lre_r])
    improvement_table = np.array(table_rows)

    return ModelQualityReport(
        rho=rho,
        cv_lambda=cv_lambda,
        eta=eta,
        M=M,
        lr_perfect=lr_perfect,
        lr_expected=lr_expected,
        lre=lre,
        lr_adjustment_bps=lr_adjustment_bps,
        improvement_table=improvement_table,
    )


# ---------------------------------------------------------------------------
# Calibration: recover eta from observed (rho, LR)
# ---------------------------------------------------------------------------


def calibrate_elasticity_from_data(
    rho_observed: float,
    lr_observed: float,
    cv_lambda: float,
    M: float = 1.0,
    eta_bounds: tuple[float, float] = (0.5, 5.0),
) -> Optional[float]:
    """
    Invert Theorem 1 to recover the implied demand elasticity eta.

    Given an observed portfolio loss ratio and the known (or estimated) model
    correlation and loss cost CV, this solves for the eta that makes Theorem 1
    consistent with the observation. Useful for calibrating the elasticity
    parameter from historical data when a direct demand model estimate is not
    available.

    Uses scipy.optimize.brentq for reliable root-finding. Returns None if no
    solution exists within eta_bounds.

    Parameters
    ----------
    rho_observed:
        Observed Pearson correlation of the pricing model, in (0, 1].
    lr_observed:
        Observed portfolio loss ratio (e.g. 0.74).
    cv_lambda:
        Coefficient of variation of the loss cost distribution.
    M:
        Target loss ratio multiplier used when writing the business.
        Default 1.0.
    eta_bounds:
        (eta_low, eta_high) search interval for brentq. Default (0.5, 5.0).

    Returns
    -------
    float or None
        Estimated eta, or None if the observed LR is outside the range
        achievable within eta_bounds.
    """
    eta_low, eta_high = eta_bounds

    def _residual(eta: float) -> float:
        return loss_ratio_formula(rho_observed, cv_lambda, eta, M=M) - lr_observed

    # Check bracket
    try:
        f_low = _residual(eta_low)
        f_high = _residual(eta_high)
    except (ValueError, ZeroDivisionError):
        return None

    if f_low * f_high > 0:
        # No sign change in the interval — observed LR not achievable
        warnings.warn(
            f"No eta in {eta_bounds} reproduces lr_observed={lr_observed:.4f} "
            f"at rho={rho_observed:.3f}, cv_lambda={cv_lambda:.3f}, M={M:.4f}. "
            "Returning None. Try widening eta_bounds.",
            UserWarning,
            stacklevel=2,
        )
        return None

    eta_star = brentq(_residual, eta_low, eta_high, xtol=1e-8, rtol=1e-8)
    return float(eta_star)


# ---------------------------------------------------------------------------
# Helper: CV estimator from expected loss costs
# ---------------------------------------------------------------------------


def _estimate_cv(expected_loss_cost: np.ndarray) -> float:
    """
    Estimate the coefficient of variation from a portfolio of expected loss costs.

    This is a rough heuristic — it uses the cross-sectional variation in
    expected costs as a proxy for the true CV of the loss distribution. This
    systematically underestimates the true CV because it ignores within-cell
    variance (the variance around each cell mean). The returned estimate should
    be treated as a lower bound.

    Parameters
    ----------
    expected_loss_cost:
        Array of per-policy expected loss costs, shape (N,).

    Returns
    -------
    float
        Estimated CV (std / mean of expected_loss_cost). At least 0.1.
    """
    cost = np.asarray(expected_loss_cost, dtype=float)
    mean = np.mean(cost)
    if mean < 1e-10:
        warnings.warn(
            "expected_loss_cost has near-zero mean. CV estimate unreliable.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0
    cv = np.std(cost) / mean
    if cv < 0.1:
        cv = 0.1
    warnings.warn(
        f"CV estimated from cross-sectional spread of expected_loss_cost: {cv:.3f}. "
        "This underestimates the true CV because within-cell variance is ignored. "
        "Supply model_cv_lambda explicitly for more accurate model quality adjustment.",
        UserWarning,
        stacklevel=2,
    )
    return float(cv)
