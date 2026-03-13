"""
Synthetic datasets for insurance demand modelling.

Two data generating processes are implemented here:

1. ``generate_conversion_data``: Simulates a UK motor PCW new business quote
   panel. The DGP has genuine confounding - high-risk customers receive higher
   technical premiums and are also less price-sensitive because they have fewer
   alternatives. A naive logistic regression on price will overestimate
   elasticity. DML corrects this.

2. ``generate_retention_data``: Simulates a UK motor renewal portfolio.
   Lapse probability is driven by price change, tenure, NCD, and payment
   method. The DGP embeds a known price elasticity so you can verify whether
   your model recovers it.

Both functions return Polars DataFrames. The true elasticity parameter is
documented in the function signature so you know what the right answer is
before fitting.

Design choices:
- Realistic distributions (not uniform random): age follows a skewed
  distribution weighted towards younger drivers for PCW; NCD concentrates
  at 5+ years for renewals because churn filters out low-NCD policies.
- Confounding is explicit, not incidental. We want to demonstrate the
  bias of naive approaches, not hide it.
- Sizes are realistic: 150k quotes per year for a medium PCW book.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl


# ------------------------------------------------------------------
# Conversion data (new business)
# ------------------------------------------------------------------

def generate_conversion_data(
    n_quotes: int = 150_000,
    true_price_elasticity: float = -2.0,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate synthetic new business quote data with known price elasticity.

    The data generating process:

    - Technical premium is set by a GLM (simulated) based on risk features.
    - Commercial loading (price / technical_premium ratio) varies by time period
      and by a small random per-quote component, creating quasi-exogenous
      variation in the price ratio.
    - Conversion probability is a logistic function of log(price_ratio) and
      log(price), with PCW rank position as an additional driver.
    - Confounding: high-risk customers (young age, high vehicle group) have
      BOTH higher technical premiums AND lower price elasticity (|β| is
      smaller for them - fewer alternatives means sticking with whoever quotes).

    The true population-average elasticity is ``true_price_elasticity``
    (default -2.0, meaning a 1% price increase → 2% conversion drop). This
    is at the lower end of published UK PCW estimates (-1.5 to -3.0).

    Parameters
    ----------
    n_quotes : int
        Number of quotes to generate. Default 150,000 (realistic annual PCW
        volume for a mid-tier UK motor insurer).
    true_price_elasticity : float
        Population average log-price elasticity. Should be negative.
        Default -2.0.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        Columns:
        - quote_id : str
        - quote_date : date
        - channel : str  ('pcw_confused', 'pcw_msm', 'pcw_ctm', 'pcw_go', 'direct')
        - age : int  (17–80)
        - vehicle_group : int  (1–4; higher = sportier/higher risk)
        - ncd_years : int  (0–9)
        - area : str  ('london', 'south_east', 'midlands', 'north', 'scotland')
        - annual_mileage : float
        - technical_premium : float  (risk model output at quote time)
        - quoted_price : float
        - price_ratio : float  (quoted_price / technical_premium)
        - log_price_ratio : float
        - competitor_price_min : float  (cheapest competitor on PCW)
        - price_to_market : float  (quoted_price / competitor_price_min)
        - rank_position : int  (1 = cheapest)
        - converted : int  (1 = bound, 0 = not bound)
        - true_elasticity : float  (per-customer true elasticity, for validation)

    Notes
    -----
    The confounding structure:
    High vehicle_group customers receive higher technical premiums. They also
    have a lower (in absolute value) price elasticity because they face fewer
    alternative quotes. This means that in a portfolio that prices high-risk
    customers higher, naive regression of conversion on price will see that
    higher prices are associated with lower conversion rates, but the estimated
    slope will be biased because risk class is a confounder.
    """
    rng = np.random.default_rng(seed)
    n = n_quotes

    # --- Risk features ---
    age = rng.integers(17, 81, size=n)
    vehicle_group = rng.choice([1, 2, 3, 4], size=n, p=[0.35, 0.30, 0.25, 0.10])
    ncd_years = rng.integers(0, 10, size=n)
    area = rng.choice(
        ["london", "south_east", "midlands", "north", "scotland"],
        size=n,
        p=[0.18, 0.22, 0.25, 0.22, 0.13],
    )
    annual_mileage = rng.lognormal(mean=9.6, sigma=0.5, size=n)  # ~15k miles mean
    channel = rng.choice(
        ["pcw_confused", "pcw_msm", "pcw_ctm", "pcw_go", "direct"],
        size=n,
        p=[0.28, 0.22, 0.18, 0.14, 0.18],
    )

    # --- Quote dates: spread over 2 years, with January/September peaks ---
    day_of_year = _sample_seasonal_days(rng, n)
    year_offset = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    base_date = np.datetime64("2023-01-01")
    quote_dates = base_date + (year_offset * 365 + day_of_year).astype("timedelta64[D]")

    # --- Technical premium: GLM-like simulation ---
    area_effect = {"london": 0.30, "south_east": 0.10, "midlands": 0.0, "north": -0.05, "scotland": -0.10}
    age_effect = np.where(age < 25, 0.60, np.where(age < 30, 0.20, np.where(age > 70, 0.25, 0.0)))
    veh_effect = (vehicle_group - 1) * 0.18
    ncd_effect = -0.10 * np.minimum(ncd_years, 5)
    area_eff_arr = np.array([area_effect[a] for a in area])
    mileage_effect = 0.15 * np.log(annual_mileage / 15000)

    log_tech = (
        6.2  # base: exp(6.2) ≈ £492
        + age_effect
        + veh_effect
        + ncd_effect
        + area_eff_arr
        + mileage_effect
        + rng.normal(0, 0.08, size=n)  # idiosyncratic risk noise
    )
    technical_premium = np.exp(log_tech)

    # --- Commercial loading: varies by quarter + small per-quote noise ---
    # This is the source of quasi-exogenous price variation for DML.
    quarter = (day_of_year // 91).astype(int) + year_offset * 4
    quarter_loading = {0: 1.05, 1: 1.00, 2: 0.98, 3: 1.02,
                       4: 1.06, 5: 1.01, 6: 0.97, 7: 1.03}
    base_loading = np.array([quarter_loading.get(int(q), 1.0) for q in quarter])
    per_quote_noise = rng.lognormal(0, 0.03, size=n)
    loading = base_loading * per_quote_noise
    quoted_price = technical_premium * loading

    # --- PCW competitor prices ---
    # Competitors have their own pricing: correlated with our technical premium
    # but with independent noise. This creates rank variation.
    n_competitors = 5
    competitor_prices = np.zeros((n, n_competitors))
    for c in range(n_competitors):
        comp_loading = rng.lognormal(0, 0.12, size=n)
        competitor_prices[:, c] = technical_premium * comp_loading

    competitor_price_min = np.min(competitor_prices, axis=1)
    price_to_market = quoted_price / competitor_price_min

    # Rank: 1 = cheapest overall (including us)
    all_prices = np.column_stack([quoted_price, competitor_prices])
    rank_position = np.sum(all_prices <= quoted_price[:, np.newaxis], axis=1).astype(int)
    rank_position = np.maximum(rank_position, 1)

    # --- Per-customer true elasticity: varies by risk class ---
    # High-risk customers (high vehicle_group, young age) are LESS price sensitive
    # (fewer alternatives). The true elasticity is more negative for standard risk.
    risk_score = (vehicle_group - 2.5) * 0.2 + np.where(age < 25, 0.3, 0.0)
    true_elasticity = true_price_elasticity * (1.0 - 0.25 * risk_score)
    # Direct customers are less price sensitive than PCW
    is_direct = (channel == "direct")
    true_elasticity = np.where(is_direct, true_elasticity * 0.7, true_elasticity)

    # --- Conversion probability ---
    price_ratio = quoted_price / technical_premium
    log_price_ratio = np.log(price_ratio)

    # Log-odds of conversion
    base_logit = (
        0.8  # baseline: ~69% conversion at ratio=1, rank=1
        + true_elasticity * log_price_ratio  # price effect (core elasticity)
        - 0.6 * np.log(np.maximum(price_to_market, 0.5))  # relative to cheapest
        - 0.3 * np.log(rank_position)  # rank penalty
        + 0.2 * ncd_years / 9  # high NCD customers are more loyal
        + rng.normal(0, 0.15, size=n)  # unexplained noise
    )

    # PCW vs direct baseline shift
    base_logit = np.where(is_direct, base_logit + 0.4, base_logit)

    conv_prob = 1.0 / (1.0 + np.exp(-base_logit))
    converted = rng.binomial(1, conv_prob)

    # --- Assemble DataFrame ---
    quote_id = [f"Q{i:08d}" for i in range(n)]

    return pl.DataFrame({
        "quote_id": quote_id,
        "quote_date": quote_dates,
        "channel": channel,
        "age": age.astype(np.int32),
        "vehicle_group": vehicle_group.astype(np.int32),
        "ncd_years": ncd_years.astype(np.int32),
        "area": area,
        "annual_mileage": annual_mileage,
        "technical_premium": technical_premium,
        "quoted_price": quoted_price,
        "price_ratio": price_ratio,
        "log_price_ratio": log_price_ratio,
        "competitor_price_min": competitor_price_min,
        "price_to_market": price_to_market,
        "rank_position": rank_position.astype(np.int32),
        "converted": converted.astype(np.int32),
        "true_elasticity": true_elasticity,
    })


# ------------------------------------------------------------------
# Retention data (renewals)
# ------------------------------------------------------------------

def generate_retention_data(
    n_policies: int = 80_000,
    true_price_change_elasticity: float = 3.5,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate synthetic renewal portfolio data with known price change elasticity.

    The data generating process:

    - Each row is one policy offered a renewal.
    - Lapse probability is driven by: renewal price change (primary), tenure
      (longer = stickier), NCD (higher = stickier), payment method (DD = stickier),
      and channel of original acquisition.
    - The true elasticity of lapse probability w.r.t. price change is
      ``true_price_change_elasticity`` (default 3.5). This means a 10% price
      increase raises the log-odds of lapsing by 3.5 × log(1.10) ≈ 0.33,
      which translates to roughly a 3–5pp increase in lapse rate depending on
      the base rate.

    Post-PS21/11 note: This dataset does NOT embed price-walking. All renewal
    prices are at or below ENBP. The variation in price change is driven by
    portfolio-level rate movements (simulated quarterly index changes).

    Parameters
    ----------
    n_policies : int
        Number of renewal offers to generate. Default 80,000.
    true_price_change_elasticity : float
        Log-odds effect of a 1-unit increase in log(renewal_price / prior_price).
        Should be positive (higher price increase → higher lapse log-odds → more lapses).
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        Columns:
        - policy_id : str
        - renewal_date : date
        - channel : str
        - tenure_years : float  (time with insurer at renewal date)
        - ncd_years : int  (0–9)
        - payment_method : str  ('dd', 'card', 'cheque')
        - age : int
        - area : str
        - vehicle_group : int
        - claim_last_3yr : int  (claims in last 3 years)
        - renewal_price : float
        - prior_year_price : float
        - price_change_pct : float  (% change: (renewal - prior) / prior)
        - log_price_change : float  (log(renewal_price / prior_year_price))
        - technical_premium : float  (at renewal date)
        - nb_equivalent_price : float  (ENBP - new business price for same risk)
        - enbp_compliant : bool  (renewal_price <= nb_equivalent_price)
        - lapsed : int  (1 = lapsed, 0 = renewed)
        - true_lapse_prob : float  (for validation)
    """
    rng = np.random.default_rng(seed)
    n = n_policies

    # --- Policy features ---
    # Renewal portfolios skew older and higher NCD than quote panels
    age = rng.integers(25, 80, size=n)
    vehicle_group = rng.choice([1, 2, 3, 4], size=n, p=[0.40, 0.32, 0.22, 0.06])
    ncd_years = rng.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        size=n,
        p=[0.03, 0.04, 0.05, 0.07, 0.09, 0.15, 0.18, 0.16, 0.13, 0.10],
    )
    tenure_years = rng.exponential(scale=4.0, size=n) + 1.0  # min 1 year
    tenure_years = np.minimum(tenure_years, 20.0)
    payment_method = rng.choice(["dd", "card", "cheque"], size=n, p=[0.62, 0.30, 0.08])
    area = rng.choice(
        ["london", "south_east", "midlands", "north", "scotland"],
        size=n,
        p=[0.18, 0.22, 0.25, 0.22, 0.13],
    )
    channel = rng.choice(
        ["pcw_confused", "pcw_msm", "pcw_ctm", "pcw_go", "direct"],
        size=n,
        p=[0.22, 0.18, 0.14, 0.12, 0.34],  # more direct for renewals
    )
    claim_last_3yr = rng.choice([0, 1, 2], size=n, p=[0.75, 0.21, 0.04])

    # --- Technical premiums ---
    area_effect = {"london": 0.30, "south_east": 0.10, "midlands": 0.0, "north": -0.05, "scotland": -0.10}
    age_effect = np.where(age > 70, 0.20, 0.0)
    veh_effect = (vehicle_group - 1) * 0.18
    ncd_effect = -0.10 * np.minimum(ncd_years, 5)
    area_eff_arr = np.array([area_effect[a] for a in area])
    claim_effect = claim_last_3yr * 0.25

    log_tech = (
        6.2
        + age_effect
        + veh_effect
        + ncd_effect
        + area_eff_arr
        + claim_effect
        + rng.normal(0, 0.08, size=n)
    )
    technical_premium = np.exp(log_tech)

    # --- Renewal pricing ---
    # Quarterly portfolio rate changes drive systematic price movements.
    # These are the same across all policies in a quarter (actuarial index movement).
    day_of_year = _sample_seasonal_days(rng, n)
    quarter = (day_of_year // 91).astype(int)
    quarter_rate_change = {0: 1.08, 1: 1.05, 2: 1.06, 3: 1.10}  # market inflation 2023
    portfolio_change = np.array([quarter_rate_change[int(q) % 4] for q in quarter])
    per_policy_noise = rng.lognormal(0, 0.02, size=n)
    price_change_ratio = portfolio_change * per_policy_noise

    prior_year_price = technical_premium / price_change_ratio * rng.lognormal(0, 0.05, size=n)
    renewal_price = prior_year_price * price_change_ratio

    # --- ENBP: new business equivalent price ---
    # The new business price for the same risk through the same channel.
    # In this simulation, NB prices = technical premium * (loading slightly above tech)
    # PS21/11: renewal must be <= NB price.
    nb_loading = rng.lognormal(0, 0.04, size=n)  # NB pricing is competitive
    nb_equivalent_price = technical_premium * np.exp(np.log(1.12) + 0.15 * np.log(portfolio_change) + np.log(nb_loading))
    # Ensure compliance: cap renewal_price at ENBP
    renewal_price = np.minimum(renewal_price, nb_equivalent_price)
    enbp_compliant = renewal_price <= nb_equivalent_price

    price_change_pct = (renewal_price - prior_year_price) / prior_year_price
    log_price_change = np.log(renewal_price / prior_year_price)

    # --- Lapse probability ---
    dd_effect = np.where(payment_method == "dd", -0.8, 0.0)
    cheque_effect = np.where(payment_method == "cheque", 0.3, 0.0)
    tenure_effect = -0.06 * np.log1p(tenure_years)
    ncd_stickiness = -0.10 * ncd_years
    pcw_sensitivity = np.where(channel == "direct", -0.4, 0.0)

    lapse_logit = (
        -1.3  # baseline: ~12% lapse at no price change, average features
        + true_price_change_elasticity * log_price_change
        + dd_effect
        + cheque_effect
        + tenure_effect
        + ncd_stickiness
        + pcw_sensitivity
        + 0.3 * claim_last_3yr  # claims shock makes customers shop
        + rng.normal(0, 0.15, size=n)
    )

    true_lapse_prob = 1.0 / (1.0 + np.exp(-lapse_logit))
    lapsed = rng.binomial(1, true_lapse_prob)

    # --- Renewal dates ---
    base_date = np.datetime64("2023-01-01")
    renewal_dates = base_date + day_of_year.astype("timedelta64[D]")

    policy_id = [f"P{i:08d}" for i in range(n)]

    return pl.DataFrame({
        "policy_id": policy_id,
        "renewal_date": renewal_dates,
        "channel": channel,
        "age": age.astype(np.int32),
        "vehicle_group": vehicle_group.astype(np.int32),
        "ncd_years": ncd_years.astype(np.int32),
        "tenure_years": tenure_years,
        "payment_method": payment_method,
        "area": area,
        "claim_last_3yr": claim_last_3yr.astype(np.int32),
        "technical_premium": technical_premium,
        "prior_year_price": prior_year_price,
        "renewal_price": renewal_price,
        "price_change_pct": price_change_pct,
        "log_price_change": log_price_change,
        "nb_equivalent_price": nb_equivalent_price,
        "enbp_compliant": enbp_compliant,
        "lapsed": lapsed.astype(np.int32),
        "true_lapse_prob": true_lapse_prob,
    })


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _sample_seasonal_days(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample days of year with January and September peaks (UK motor pattern)."""
    # Mixture: uniform background + January peak + September peak
    base = rng.integers(0, 365, size=n)
    jan_peak = rng.integers(0, 31, size=n)
    sep_peak = rng.integers(244, 274, size=n)  # September
    which = rng.choice([0, 1, 2], size=n, p=[0.65, 0.20, 0.15])
    return np.where(which == 0, base, np.where(which == 1, jan_peak, sep_peak))
