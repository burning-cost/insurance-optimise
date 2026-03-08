"""
Shared fixtures for insurance-optimise tests.

All fixtures produce small but realistic synthetic insurance datasets.
Segment counts are kept small (N <= 50) so tests run fast in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_optimise import ConstraintConfig, PortfolioOptimiser


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_portfolio(
    n: int = 20,
    n_renewal: int = 12,
    seed: int = 0,
    include_enbp: bool = True,
) -> dict:
    """
    Generate a small synthetic insurance portfolio.

    Returns a dict with arrays:
    - technical_price: technical premium per policy
    - expected_loss_cost: expected claims cost
    - p_demand: baseline demand probability
    - elasticity: price elasticity (negative)
    - renewal_flag: boolean
    - enbp: equivalent new business price
    - prior_multiplier: 1.0 for all
    """
    rng = np.random.default_rng(seed)

    # Technical premiums: uniform £200 to £1000
    tc = rng.uniform(200, 1000, size=n)

    # Loss costs: 55% to 75% of technical premium
    loss_ratio_true = rng.uniform(0.55, 0.75, size=n)
    cost = tc * loss_ratio_true

    # Demand: 70% to 95% baseline renewal/conversion probability
    p_demand = rng.uniform(0.70, 0.95, size=n)

    # Elasticity: -0.5 to -3.0 (typical personal lines range)
    elasticity = -rng.uniform(0.5, 3.0, size=n)

    # Renewal flag
    renewal_flag = np.zeros(n, dtype=bool)
    if n_renewal > 0:
        renewal_idx = rng.choice(n, size=min(n_renewal, n), replace=False)
        renewal_flag[renewal_idx] = True

    # ENBP: typically 5-20% above technical price for renewals
    if include_enbp:
        enbp = tc * rng.uniform(1.05, 1.20, size=n)
    else:
        enbp = None

    prior_multiplier = np.ones(n)

    return {
        "technical_price": tc,
        "expected_loss_cost": cost,
        "p_demand": p_demand,
        "elasticity": elasticity,
        "renewal_flag": renewal_flag,
        "enbp": enbp,
        "prior_multiplier": prior_multiplier,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_portfolio():
    """N=20 portfolio with renewals and ENBP."""
    return make_portfolio(n=20, n_renewal=12, seed=42)


@pytest.fixture
def medium_portfolio():
    """N=100 portfolio with renewals and ENBP."""
    return make_portfolio(n=100, n_renewal=60, seed=99)


@pytest.fixture
def nb_only_portfolio():
    """N=20 portfolio with no renewals (new business only)."""
    return make_portfolio(n=20, n_renewal=0, seed=7, include_enbp=False)


@pytest.fixture
def unconstrained_optimiser(small_portfolio):
    """PortfolioOptimiser with no constraints — for testing objective."""
    p = small_portfolio
    return PortfolioOptimiser(
        technical_price=p["technical_price"],
        expected_loss_cost=p["expected_loss_cost"],
        p_demand=p["p_demand"],
        elasticity=p["elasticity"],
        renewal_flag=p["renewal_flag"],
        enbp=p["enbp"],
        constraints=ConstraintConfig(technical_floor=False, min_multiplier=0.1),
    )


@pytest.fixture
def constrained_optimiser(small_portfolio):
    """PortfolioOptimiser with typical real-world constraints."""
    p = small_portfolio
    config = ConstraintConfig(
        lr_max=0.70,
        retention_min=0.85,
        max_rate_change=0.20,
        enbp_buffer=0.01,
        technical_floor=True,
    )
    return PortfolioOptimiser(
        technical_price=p["technical_price"],
        expected_loss_cost=p["expected_loss_cost"],
        p_demand=p["p_demand"],
        elasticity=p["elasticity"],
        renewal_flag=p["renewal_flag"],
        enbp=p["enbp"],
        constraints=config,
    )
