"""
Claims variance models for stochastic loss ratio constraints.

The stochastic LR constraint (Branda 2014) replaces the deterministic
E[LR] <= target with a chance constraint:

    P(portfolio LR <= target) >= alpha

Under the normal approximation (CLT; reasonable for large books), this
reformulates to:

    E[LR] + z_alpha * sigma[LR] <= target

where z_alpha is the Chebyshev-style or normal quantile, and sigma[LR] is
the standard deviation of the portfolio loss ratio.

This module provides ``ClaimsVarianceModel`` — a helper for constructing
per-policy variance estimates from GLM outputs, which you then pass to
``PortfolioOptimiser`` via ``claims_variance``. Enable the stochastic
constraint by setting ``ConstraintConfig(stochastic_lr=True)`` alongside
``stochastic_alpha``.

The constraint itself is implemented inside ``constraints.py`` and is
activated automatically when ``ConstraintConfig.stochastic_lr=True`` and
``claims_variance`` is supplied to the optimiser.

References
----------
Branda, M. (2013). "Optimization Approaches to Multiplicative Tariff of Rates."
ASTIN Colloquium, Hague.

Charnes, A. and Cooper, W. W. (1959). "Chance-Constrained Programming."
Management Science 6(1):73-79.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClaimsVarianceModel:
    """
    Per-policy variance estimates for expected claims.

    Provides the ``variance_claims`` array that ``PortfolioOptimiser`` needs
    when ``ConstraintConfig.stochastic_lr=True``.

    In a Tweedie GLM, the variance of the claims amount Y_i is:

        Var(Y_i) = phi * mu_i^p

    where mu_i is the fitted mean (expected loss cost), phi is the dispersion
    parameter, and p is the Tweedie power parameter (typically 1.5 for combined
    frequency-severity, or 1.0 for Poisson frequency).

    For a portfolio, the aggregate loss variance is approximately:

        Var(L) = sum_i x_i * Var(Y_i) + sum_i x_i * (1 - x_i) * mu_i^2

    The second term captures the variance from uncertainty in whether each
    policy actually renews (Bernoulli variance on the selection process).
    The constraint implementation in ``constraints.py`` uses a simplified form
    (the first term only) for tractability with analytical gradients.

    Parameters
    ----------
    mean_claims : np.ndarray
        Shape (n_policies,). Expected claims per policy (from GLM fitted means).
    variance_claims : np.ndarray
        Shape (n_policies,). Per-policy claims variance from the GLM.
    """

    mean_claims: np.ndarray
    variance_claims: np.ndarray

    def __post_init__(self) -> None:
        self.mean_claims = np.asarray(self.mean_claims, dtype=float)
        self.variance_claims = np.asarray(self.variance_claims, dtype=float)
        if self.mean_claims.shape != self.variance_claims.shape:
            raise ValueError(
                f"mean_claims and variance_claims must have the same shape. "
                f"Got {self.mean_claims.shape} and {self.variance_claims.shape}."
            )
        if np.any(self.variance_claims < 0):
            raise ValueError("variance_claims must be non-negative.")

    @classmethod
    def from_tweedie(
        cls,
        mean_claims: np.ndarray,
        dispersion: float,
        power: float = 1.5,
    ) -> "ClaimsVarianceModel":
        """
        Construct from Tweedie GLM outputs.

        The Tweedie family is the standard actuarial GLM for aggregate
        claims. Power 1.5 is the compound Poisson-gamma (frequency-severity)
        model; power 1.0 is Poisson; power 2.0 is gamma.

        Parameters
        ----------
        mean_claims : np.ndarray
            Fitted means from the Tweedie GLM (expected loss cost per policy).
        dispersion : float
            Dispersion parameter phi from the GLM summary. Available in R via
            ``summary(model)$dispersion`` or Python statsmodels.
        power : float
            Tweedie power parameter. Default 1.5 (insurance compound model).
        """
        mean_claims = np.asarray(mean_claims, dtype=float)
        if dispersion <= 0:
            raise ValueError(f"dispersion must be positive, got {dispersion}.")
        if power < 1 or power > 2:
            import warnings
            warnings.warn(
                f"Tweedie power={power} is outside the [1, 2] range typical for "
                "insurance. Power=1 is Poisson, power=2 is gamma. Power=1.5 is "
                "the compound Poisson-gamma (Tweedie) model.",
                stacklevel=2,
            )
        variance = dispersion * np.power(mean_claims, power)
        return cls(mean_claims=mean_claims, variance_claims=variance)

    @classmethod
    def from_overdispersed_poisson(
        cls,
        expected_counts: np.ndarray,
        mean_severity: np.ndarray,
        severity_variance: np.ndarray,
        overdispersion: float = 1.0,
    ) -> "ClaimsVarianceModel":
        """
        Construct from a frequency-severity decomposition.

        For a frequency-severity model:

            E[Y_i] = expected_count_i * mean_severity_i
            Var[Y_i] = expected_count_i * severity_variance_i
                      + mean_severity_i^2 * expected_count_i * overdispersion

        This is the law of total variance applied to the compound distribution.

        Parameters
        ----------
        expected_counts : np.ndarray
            Expected claim counts per policy (from frequency model).
        mean_severity : np.ndarray
            Expected severity per claim (from severity model).
        severity_variance : np.ndarray
            Variance of severity per claim.
        overdispersion : float
            Overdispersion relative to Poisson. 1.0 = Poisson (no overdispersion).
            Quasi-Poisson models estimate this from residual deviance / df.
        """
        expected_counts = np.asarray(expected_counts, dtype=float)
        mean_severity = np.asarray(mean_severity, dtype=float)
        severity_variance = np.asarray(severity_variance, dtype=float)

        mean_claims = expected_counts * mean_severity
        var_claims = (
            expected_counts * severity_variance
            + mean_severity**2 * expected_counts * overdispersion
        )
        return cls(mean_claims=mean_claims, variance_claims=var_claims)

    def __repr__(self) -> str:
        return (
            f"ClaimsVarianceModel("
            f"n_policies={len(self.mean_claims)}, "
            f"mean_range=[{self.mean_claims.min():.1f}, {self.mean_claims.max():.1f}], "
            f"var_range=[{self.variance_claims.min():.1f}, {self.variance_claims.max():.1f}])"
        )
