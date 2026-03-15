"""
Benchmark: Constrained rate optimisation vs uniform rate change (insurance-optimise).

The question: can we achieve the same (or better) premium income with a
constrained portfolio optimiser that prices each policy individually, compared to
applying a flat uniform rate change to all policies?

The uniform approach is the default in many pricing teams: apply a flat +X%
to all renewal premiums. This ignores that:
  - High-elasticity customers leave when you apply a high uplift to them
  - Low-elasticity customers would accept a larger increase
  - Some segments are approaching the ENBP (FCA PS21/11) cap

The optimiser redistributes the rate change to maximise expected profit subject
to: loss ratio cap, retention floor, ENBP compliance, and ±25% rate change limit.

Setup:
- 2,000 renewal policies with heterogeneous elasticities
- Technical premiums and elasticities drawn from realistic distributions
- Target: +7% average premium increase (cost inflation)
- Uniform: apply flat 1.07 multiplier to all
- Optimised: PortfolioOptimiser with LR and retention constraints

Expected output:
- Optimised achieves target GWP with higher expected profit
- Retention under optimised is better (price sensitive customers get smaller increases)
- Loss ratio under optimised is closer to target

Run:
    python benchmarks/benchmark_optimise.py
"""

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: Constrained optimiser vs uniform rate change (insurance-optimise)")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from insurance_optimise import PortfolioOptimiser, ConstraintConfig
    print("insurance-optimise imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-optimise: {e}")
    print("Install with: pip install insurance-optimise")
    sys.exit(1)

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Synthetic portfolio
# ---------------------------------------------------------------------------

N = 2_000
SEED = 42
rng = np.random.default_rng(SEED)

print(f"\nGenerating {N:,} synthetic renewal policies...")
print()

# Technical premiums (realistic UK motor range)
log_tc = rng.normal(6.3, 0.35, N)   # exp(6.3) ~ £545, SD ~ 35%
technical_price = np.exp(log_tc)

# Expected loss costs ~ 65% of technical price (standard motor loss ratio)
loss_ratio_factor = rng.normal(0.65, 0.05, N).clip(0.40, 0.90)
expected_loss_cost = technical_price * loss_ratio_factor

# Elasticities: vary by customer segment
# PCW-acquired customers: more elastic (-1.5 to -2.5)
# Direct customers: less elastic (-0.8 to -1.5)
segment = rng.choice(["pcw", "direct"], N, p=[0.55, 0.45])
elast_mean = np.where(segment == "pcw", -2.0, -1.2)
elasticity = elast_mean + rng.normal(0, 0.3, N)
elasticity = np.clip(elasticity, -4.0, -0.3)

# Current renewal probability (base retention at current price)
p_demand = rng.beta(8, 2, N).clip(0.55, 0.95)  # mean ~80% retention

# All policies are renewals
renewal_flag = np.ones(N, dtype=bool)

# ENBP: NB price for same risk (post-PS21/11)
# Simulate as technical_price * market_loading (new business is competitive)
enbp_loading = rng.lognormal(0.08, 0.06, N)  # typically 5-15% above technical
enbp = technical_price * enbp_loading

# Prior multiplier (last year's commercial loading, ~1.0-1.1)
prior_multiplier = np.ones(N) * rng.lognormal(0.0, 0.03, N)
prior_multiplier = prior_multiplier.clip(0.9, 1.2)

print(f"Portfolio statistics:")
print(f"  Mean technical price:  £{technical_price.mean():.0f}")
print(f"  Mean expected cost:    £{expected_loss_cost.mean():.0f}")
print(f"  Mean loss ratio:       {loss_ratio_factor.mean():.2%}")
print(f"  Mean elasticity:       {elasticity.mean():.2f}")
print(f"  Mean base retention:   {p_demand.mean():.1%}")
print(f"  Mean ENBP:             £{enbp.mean():.0f}")
print(f"  PCW share:             {(segment == 'pcw').mean():.0%}")
print()

# ---------------------------------------------------------------------------
# Baseline: uniform +7% rate change
# ---------------------------------------------------------------------------

UNIFORM_INCREASE = 1.07

print("BASELINE: Uniform +7% rate change")
print("-" * 50)

# Uniform multiplier applied to all
m_uniform = np.clip(prior_multiplier * UNIFORM_INCREASE, 0.5, 3.0)
# Clip to ENBP for compliance
m_uniform_enbp_capped = np.minimum(m_uniform, enbp / technical_price)

p_uniform = m_uniform_enbp_capped * technical_price

# Demand response under uniform pricing
# log-linear: demand = p0 * exp(elasticity * (log(m_new) - log(m_old)))
x_uniform = p_demand * np.exp(
    elasticity * np.log(m_uniform_enbp_capped / prior_multiplier)
)
x_uniform = x_uniform.clip(0.01, 1.0)

uniform_gwp = float(np.dot(p_uniform, x_uniform))
uniform_profit = float(np.dot(p_uniform - expected_loss_cost, x_uniform))
uniform_lr = float(np.dot(expected_loss_cost, x_uniform) / max(uniform_gwp, 1.0))
uniform_retention = float(x_uniform.mean())
uniform_avg_rate_change = float(
    np.mean((m_uniform_enbp_capped / prior_multiplier - 1.0) * 100)
)

print(f"  GWP:              £{uniform_gwp:>12,.0f}")
print(f"  Expected profit:  £{uniform_profit:>12,.0f}")
print(f"  Loss ratio:       {uniform_lr:>12.2%}")
print(f"  Retention rate:   {uniform_retention:>12.1%}")
print(f"  Avg rate change:  {uniform_avg_rate_change:>12.1f}%")
print()

# ---------------------------------------------------------------------------
# Optimised: PortfolioOptimiser
# ---------------------------------------------------------------------------

print("OPTIMISED: PortfolioOptimiser with constraints")
print("-" * 50)
print()
print("  Constraints:")
print(f"    LR cap: 68%  (current LR {loss_ratio_factor.mean():.1%})")
print(f"    Retention floor: 78%")
print(f"    Max rate change: ±25%")
print(f"    ENBP compliance: enabled")
print(f"    Technical floor: price >= technical price")
print()

config = ConstraintConfig(
    lr_max=0.68,
    retention_min=0.78,
    max_rate_change=0.25,
    enbp_buffer=0.01,
    technical_floor=True,
)

opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    prior_multiplier=prior_multiplier,
    constraints=config,
    demand_model="log_linear",
    n_restarts=1,
    seed=SEED,
)

t0 = time.time()
result = opt.optimise()
opt_time = time.time() - t0

print(f"  Converged: {result.converged}")
print(f"  Solver message: {result.solver_message[:60]}")
print(f"  Iterations: {result.n_iter}")
print(f"  Optimisation time: {opt_time:.2f}s")
print()

opt_gwp = result.expected_gwp
opt_profit = result.expected_profit
opt_lr = result.expected_loss_ratio
opt_retention = result.expected_retention

# Average rate change under optimised solution
m_opt = result.multipliers
opt_avg_rate_change = float(np.mean((m_opt / prior_multiplier - 1.0) * 100))

print(f"  GWP:              £{opt_gwp:>12,.0f}")
print(f"  Expected profit:  £{opt_profit:>12,.0f}")
print(f"  Loss ratio:       {opt_lr:>12.2%}")
print(f"  Retention rate:   {opt_retention:>12.1%}")
print(f"  Avg rate change:  {opt_avg_rate_change:>12.1f}%")
print()

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Metric':<35} {'Uniform +7%':>15} {'Optimised':>15} {'Change':>10}")
print("-" * 70)

def _pct_change(a, b):
    if abs(a) < 1e-9:
        return float("nan")
    return (b - a) / abs(a) * 100

rows = [
    ("GWP (£000)", uniform_gwp / 1000, opt_gwp / 1000, "£"),
    ("Expected profit (£000)", uniform_profit / 1000, opt_profit / 1000, "£"),
    ("Loss ratio", uniform_lr * 100, opt_lr * 100, "%pt"),
    ("Retention rate", uniform_retention * 100, opt_retention * 100, "%pt"),
    ("Avg rate change (%)", uniform_avg_rate_change, opt_avg_rate_change, "%pt"),
]

for label, unif_val, opt_val, unit in rows:
    delta = opt_val - unif_val
    if unit == "£":
        print(f"  {label:<33} {unif_val:>12,.0f}K {opt_val:>12,.0f}K {delta:>+8,.0f}K")
    else:
        print(f"  {label:<33} {unif_val:>14.2f} {opt_val:>14.2f} {delta:>+9.2f}")

print()

# Profit uplift
profit_uplift = opt_profit - uniform_profit
profit_uplift_pct = _pct_change(uniform_profit, opt_profit)

print("KEY FINDINGS")
print(f"  Profit uplift:     £{profit_uplift:,.0f}  ({profit_uplift_pct:+.1f}%)")
print(f"  Retention gain:    {(opt_retention - uniform_retention)*100:+.1f}pp")
print(f"  LR improvement:    {(opt_lr - uniform_lr)*100:+.1f}pp")
print()
print("  The optimiser achieves higher profit by:")
print("  - Applying larger increases to price-inelastic customers (direct channel)")
print("  - Applying smaller increases to price-elastic customers (PCW) to retain them")
print("  - Ensuring all renewals comply with ENBP cap")
print("  - Staying within the LR and retention constraints")
print()
print("  The uniform +7% approach is blunt: it loses elastic customers who don't")
print("  need to be priced aggressively, and leaves margin on the table from")
print("  inelastic customers who would accept higher prices.")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
