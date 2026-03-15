"""
Demand models for insurance pricing optimisation.

Two built-in models, both parameterised by the constant-elasticity assumption
from upstream insurance-elasticity output:

LogLinearDemand (default):
    x(m) = x0 * m^epsilon
    where m is the price multiplier, epsilon < 0 is the price elasticity.
    d(log x)/d(log m) = epsilon (constant elasticity).
    This is the workhorse model: simple, interpretable, fast gradients.

LogisticDemand:
    x(m) = 1 / (1 + exp(alpha_i + beta_i * m * tc_i))
    For renewal: x(p) is the probability of renewing at price p.
    Requires converting the elasticity estimate to the logistic parameter
    beta: beta_i = elasticity_i / (p0_i * (1 - x0_i)).
    More theoretically grounded for renewal modelling but adds numerical
    complexity.

Both models are used inside PortfolioOptimiser — the user selects via
``demand_model='log_linear'`` or ``demand_model='logistic'``.
"""

from __future__ import annotations

import numpy as np


class LogLinearDemand:
    """
    Log-linear (constant elasticity) demand model.

    Demand at multiplier m:
        x(m) = x0 * m^epsilon

    This model has the desirable property that demand is always positive for
    any positive multiplier — no clipping needed.

    Parameters
    ----------
    x0:
        Baseline demand array, shape (N,). Demand at m=1 (current price).
    elasticity:
        Price elasticity array, shape (N,). Should be negative (higher price
        -> lower demand). If positive values are passed a warning is issued
        but values are used as-is.
    """

    def __init__(self, x0: np.ndarray, elasticity: np.ndarray) -> None:
        self.x0 = np.asarray(x0, dtype=float)
        self.elasticity = np.asarray(elasticity, dtype=float)
        if np.any(self.elasticity > 0):
            import warnings
            warnings.warn(
                "Some elasticity values are positive. Log-linear demand assumes "
                "negative elasticity (higher price = lower demand). Check inputs.",
                stacklevel=3,
            )

    def demand(self, m: np.ndarray) -> np.ndarray:
        """
        Compute demand at multipliers m.

        Parameters
        ----------
        m:
            Price multiplier array, shape (N,). Must be positive.

        Returns
        -------
        np.ndarray
            Expected demand, shape (N,).
        """
        m = np.asarray(m, dtype=float)
        # x(m) = x0 * m^epsilon
        # Use exp(epsilon * log(m)) for numerical stability
        log_m = np.log(np.maximum(m, 1e-10))
        return self.x0 * np.exp(self.elasticity * log_m)

    def demand_gradient(self, m: np.ndarray) -> np.ndarray:
        """
        Compute dx/dm — gradient of demand with respect to each multiplier.

        dx_i/dm_i = x0_i * epsilon_i * m_i^(epsilon_i - 1)
                  = demand_i * epsilon_i / m_i

        Returns
        -------
        np.ndarray
            Gradient array, shape (N,). Diagonal of the full Jacobian
            (demand_i depends only on m_i).
        """
        m = np.asarray(m, dtype=float)
        d = self.demand(m)
        return d * self.elasticity / np.maximum(m, 1e-10)


class LogisticDemand:
    """
    Logistic (sigmoid) demand model.

    Demand at multiplier m (hence price p = m * tc):
        x(m) = 1 / (1 + exp(alpha_i + beta_i * m_i * tc_i))

    Parameters alpha and beta are derived from (x0, elasticity, p0):
        beta_i = elasticity_i / (p0_i * (1 - x0_i))
        alpha_i = log(1/x0_i - 1) - beta_i * p0_i

    where p0_i = current premium = tc_i * m0_i (m0=1 at baseline).

    Parameters
    ----------
    x0:
        Baseline demand probabilities, shape (N,). Values in (0, 1).
    elasticity:
        Log-log price elasticity d(log x)/d(log p), shape (N,). Negative.
        This is the same elasticity concept used by LogLinearDemand — not a
        semi-elasticity. The conversion to the logistic beta parameter is:
        beta_i = -elasticity_i / (p0_i * (1 - x0_i)).
    technical_price:
        Technical price (baseline m=1 price), shape (N,).
    """

    def __init__(
        self,
        x0: np.ndarray,
        elasticity: np.ndarray,
        technical_price: np.ndarray,
    ) -> None:
        self.x0 = np.asarray(x0, dtype=float)
        self.elasticity = np.asarray(elasticity, dtype=float)
        self.tc = np.asarray(technical_price, dtype=float)

        # Derive logistic parameters from (x0, elasticity, p0)
        # p0 = technical_price (at multiplier m=1)
        p0 = self.tc
        # elasticity = d(log x)/d(log p) = p * d(log x)/dp = p * (-(1-x)*beta)
        # At baseline: elasticity = p0 * (-(1-x0)*beta)
        # => beta = -elasticity / (p0 * (1 - x0))  [positive, since elasticity < 0]
        x0_clipped = np.clip(self.x0, 1e-6, 1 - 1e-6)
        self.beta = -self.elasticity / (p0 * (1.0 - x0_clipped))
        # alpha: x0 = 1/(1+exp(alpha + beta*p0))
        # => alpha = log(1/x0 - 1) - beta * p0
        self.alpha = np.log(1.0 / x0_clipped - 1.0) - self.beta * p0

    def demand(self, m: np.ndarray) -> np.ndarray:
        """Compute demand at multipliers m."""
        m = np.asarray(m, dtype=float)
        p = m * self.tc
        logit = self.alpha + self.beta * p
        # Clip to prevent overflow
        logit = np.clip(logit, -500, 500)
        return 1.0 / (1.0 + np.exp(logit))

    def demand_gradient(self, m: np.ndarray) -> np.ndarray:
        """
        Gradient dx/dm.

        dx/dm_i = dx/dp_i * dp_i/dm_i = dx/dp_i * tc_i

        dx/dp = -beta * x * (1 - x)
        so dx/dm = -beta_i * x_i * (1 - x_i) * tc_i
        """
        m = np.asarray(m, dtype=float)
        x = self.demand(m)
        return -self.beta * x * (1.0 - x) * self.tc


def make_demand_model(
    model_name: str,
    x0: np.ndarray,
    elasticity: np.ndarray,
    technical_price: np.ndarray,
) -> LogLinearDemand | LogisticDemand:
    """
    Factory function for demand models.

    Parameters
    ----------
    model_name:
        ``'log_linear'`` or ``'logistic'``.
    x0:
        Baseline demand, shape (N,).
    elasticity:
        Elasticity array, shape (N,).
    technical_price:
        Technical price array, shape (N,).

    Returns
    -------
    LogLinearDemand or LogisticDemand
    """
    if model_name == "log_linear":
        return LogLinearDemand(x0=x0, elasticity=elasticity)
    elif model_name == "logistic":
        return LogisticDemand(
            x0=x0, elasticity=elasticity, technical_price=technical_price
        )
    else:
        raise ValueError(
            f"Unknown demand_model '{model_name}'. "
            "Choose 'log_linear' or 'logistic'."
        )
