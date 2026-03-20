"""
Benchmark: ParetoFrontier vs single-objective SLSQP (insurance-optimise v0.4.0).

The question this benchmark answers:

    Does the 3-objective Pareto surface reveal trade-offs that a single-
    objective optimiser misses?

The answer is yes, and for a concrete reason: SLSQP maximises profit subject
to a retention floor, but it ignores fairness entirely. The pricing team sees
one number (the profit-maximising solution) without knowing that a slightly
different solution offers the same profit with materially lower premium
disparity across deprivation groups. The Pareto surface maps this out.

Portfolio design notes
----------------------
Baseline retention is deliberately set high (~90%) via p_demand from Beta(12,2).
This ensures the optimizer can find solutions across a range of retention floors
(70%-87%) by pricing up the most inelastic customers while keeping elastic
customers. Without high base retention, a retention floor above the unconstrained
optimum is infeasible, and the Pareto sweep collapses.

The fairness disparity sweep is set between 1.1 and the natural unconstrained
value (~1.2-1.5), so the sweep explores meaningful fairness trade-offs.

Run
---
    cd /home/ralph/repos/insurance-optimise
    uv run python benchmarks/benchmark_pareto.py
"""

from __future__ import annotations

import sys
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: ParetoFrontier vs single-objective SLSQP")
print("insurance-optimise v0.4.0")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from insurance_optimise import PortfolioOptimiser, ConstraintConfig
    from insurance_optimise.pareto import (
        ParetoFrontier,
        premium_disparity_ratio,
    )
    from insurance_optimise import __version__
    print(f"insurance-optimise {__version__} imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-optimise: {e}")
    sys.exit(1)

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic portfolio
# ---------------------------------------------------------------------------

# N is moderate so the benchmark completes in <2 minutes on modest hardware.
# The Databricks notebook version uses N=1,500 with the same logic.
N = 150  # reduced for local Pi; Databricks notebook uses 1500
SEED = 42
rng = np.random.default_rng(SEED)

print(f"Generating {N:,} synthetic renewal policies...")
print()

# Technical premiums: realistic UK motor range (~£250-£1,200)
log_tc = rng.normal(6.2, 0.38, N)
technical_price = np.exp(log_tc)

# Expected loss costs: ~55-75% of technical price
loss_ratio_factor = rng.beta(5, 3.5, N).clip(0.40, 0.85)
expected_loss_cost = technical_price * loss_ratio_factor

# Elasticities: PCW customers more elastic than direct
segment = rng.choice(["pcw", "direct"], N, p=[0.55, 0.45])
elast_mean = np.where(segment == "pcw", -1.9, -1.1)
elasticity = np.clip(elast_mean + rng.normal(0, 0.25, N), -3.5, -0.3)

# High baseline renewal probability — ensures retention floor is achievable.
# Beta(12,2) gives mean ~86%. The unconstrained optimum will drop this as the
# optimizer prices up, creating space for the retention sweep.
p_demand = rng.beta(12, 2, N).clip(0.70, 0.97)

# All are renewals
renewal_flag = np.ones(N, dtype=bool)

# ENBP: new-business equivalent pricing (PS21/11)
enbp_loading = rng.lognormal(0.09, 0.055, N)
enbp = technical_price * enbp_loading

# Prior multiplier (last year's commercial loading)
prior_multiplier = np.clip(rng.lognormal(0.0, 0.022, N), 0.92, 1.10)

# Deprivation quintile (1 = most deprived, 5 = least deprived).
# PCW customers are mildly more likely to be in lower deprivation quintiles
# (they are price-savvy comparison shoppers). Direct customers skew deprived.
# Assignment is purely probabilistic to avoid the normalization complexity
# of a per-policy loop.
_pcw_mask = segment == 'pcw'
deprivation_quintile = np.where(
    _pcw_mask,
    rng.choice([1, 2, 3, 4, 5], N, p=[0.10, 0.15, 0.20, 0.30, 0.25]),
    rng.choice([1, 2, 3, 4, 5], N, p=[0.30, 0.30, 0.20, 0.12, 0.08]),
)

# Fairness metric
fairness_fn = partial(
    premium_disparity_ratio,
    technical_price=technical_price,
    group_labels=deprivation_quintile,
)

# Baseline assessment at prior multiplier
baseline_disparity = fairness_fn(prior_multiplier)

# Quick unconstrained run to understand the feasible space
_config_uncon = ConstraintConfig(max_rate_change=0.30, enbp_buffer=0.01)
_opt_uncon = PortfolioOptimiser(
    technical_price, expected_loss_cost, p_demand, elasticity,
    renewal_flag, enbp, prior_multiplier,
    constraints=_config_uncon, demand_model="log_linear", seed=SEED,
)
_baseline = _opt_uncon.portfolio_summary()
_uncon_result = _opt_uncon.optimise()
_uncon_disparity = fairness_fn(_uncon_result.multipliers)
_uncon_retention = _uncon_result.expected_retention or 0.0

print(f"Portfolio statistics:")
print(f"  Policies:           {N:,}")
print(f"  Mean tech price:    £{technical_price.mean():.0f}")
print(f"  Mean loss ratio:    {loss_ratio_factor.mean():.2%}")
print(f"  Mean elasticity:    {elasticity.mean():.2f}")
print(f"  Mean base retention:{p_demand.mean():.1%}")
print(f"  PCW share:          {(segment == 'pcw').mean():.0%}")
print()
print(f"  Baseline (no rate change):")
print(f"    Retention:        {_baseline['retention']:.1%}")
print(f"    LR:               {_baseline['loss_ratio']:.2%}")
print(f"    Disparity ratio:  {baseline_disparity:.3f}")
print()
print(f"  Unconstrained optimum:")
print(f"    Profit:           £{_uncon_result.expected_profit:,.0f}")
print(f"    Retention:        {_uncon_retention:.1%}")
print(f"    LR:               {_uncon_result.expected_loss_ratio:.2%}")
print(f"    Disparity ratio:  {_uncon_disparity:.3f}")
print()

# Set sweep ranges based on what we know is feasible
# Retention sweep: between unconstrained optimum and baseline (with margin)
_ret_lo = round(max(_uncon_retention + 0.03, 0.72), 2)
_ret_hi = round(min(_baseline["retention"] - 0.02, 0.90), 2)
if _ret_lo >= _ret_hi:
    _ret_lo = round(_uncon_retention + 0.01, 2)
    _ret_hi = round(_baseline["retention"], 2)

# Fairness sweep: from 10% below natural disparity to 2.0
_fair_lo = round(max(_uncon_disparity * 0.90, 1.05), 2)
_fair_hi = round(min(_uncon_disparity * 1.20, 2.50), 2)
if _fair_lo >= _fair_hi:
    _fair_lo = round(_uncon_disparity - 0.05, 2)
    _fair_hi = round(_uncon_disparity + 0.20, 2)

print(f"  Sweep ranges (auto-calibrated from unconstrained optimum):")
print(f"    Retention:        {_ret_lo:.2f} - {_ret_hi:.2f}")
print(f"    Fairness cap:     {_fair_lo:.2f} - {_fair_hi:.2f}")
print()

# ---------------------------------------------------------------------------
# PART 1: Single-objective SLSQP
# ---------------------------------------------------------------------------

print("=" * 70)
print("PART 1: Single-objective SLSQP (max profit, no fairness constraint)")
print("=" * 70)
print()

# Retention floor: midpoint of the sweep range
_ret_floor = round((_ret_lo + _ret_hi) / 2, 2)
print(f"  Constraints: retention >= {_ret_floor:.0%}, LR <= 70%, rate change <= 25%")
print()

config_single = ConstraintConfig(
    lr_max=0.70,
    retention_min=_ret_floor,
    max_rate_change=0.25,
    enbp_buffer=0.01,
    technical_floor=True,
)

opt_single = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    prior_multiplier=prior_multiplier,
    constraints=config_single,
    demand_model="log_linear",
    n_restarts=1,
    seed=SEED,
)

t0 = time.time()
single_result = opt_single.optimise()
single_time = time.time() - t0

# Compute fairness at the single-objective optimum — the team normally skips this
single_fairness = fairness_fn(single_result.multipliers)
single_profit = single_result.expected_profit
single_retention = single_result.expected_retention or float("nan")
single_lr = single_result.expected_loss_ratio
single_gwp = single_result.expected_gwp

print(f"  Converged:           {single_result.converged}")
if not single_result.converged:
    print(f"  Solver message:      {single_result.solver_message[:55]}")
print(f"  Iterations:          {single_result.n_iter}")
print(f"  Time:                {single_time:.2f}s")
print()
print(f"  Profit:              £{single_profit:>12,.0f}")
print(f"  GWP:                 £{single_gwp:>12,.0f}")
print(f"  Loss ratio:          {single_lr:>12.2%}")
print(f"  Retention:           {single_retention:>12.1%}")
print(f"  Fairness disparity:  {single_fairness:>12.3f}  <-- blind spot")
print()
print("  The single-objective solution does not account for fairness.")
print("  The disparity ratio above is what it produces by accident.")
print()

# ---------------------------------------------------------------------------
# PART 2: ParetoFrontier 2D sweep
# ---------------------------------------------------------------------------

print("=" * 70)
print("PART 2: ParetoFrontier 2D sweep (profit vs retention vs fairness)")
N_POINTS = 3  # 3x3=9 solves; Databricks notebook uses 6x6=36
print(f"        {N_POINTS}x{N_POINTS} = {N_POINTS*N_POINTS} solves (local validation; Databricks version uses 6x6=36 solves).")
print("=" * 70)
print()

print(f"  x-axis: retention {_ret_lo:.0%} - {_ret_hi:.0%}")
print(f"  y-axis: fairness cap {_fair_lo:.2f} - {_fair_hi:.2f}")
print()

# Base config: no retention floor (swept by ParetoFrontier)
config_pareto = ConstraintConfig(
    lr_max=0.72,
    max_rate_change=0.25,
    enbp_buffer=0.01,
    technical_floor=True,
)

opt_pareto = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    prior_multiplier=prior_multiplier,
    constraints=config_pareto,
    demand_model="log_linear",
    n_restarts=1,
    seed=SEED,
)

pf = ParetoFrontier(
    optimiser=opt_pareto,
    fairness_metric=fairness_fn,
    sweep_x="volume_retention",
    sweep_x_range=(_ret_lo, _ret_hi),
    sweep_y="fairness_max",
    sweep_y_range=(_fair_lo, _fair_hi),
    n_points_x=N_POINTS,
    n_points_y=N_POINTS,
    n_jobs=1,
)

t0 = time.time()
pareto_result = pf.run()
pareto_time = time.time() - t0

print(f"  Pareto sweep completed in {pareto_time:.1f}s")
print(f"  ({pareto_time / (N_POINTS*N_POINTS):.2f}s per solve average)")
print()

# Summary
summary = pareto_result.summary()
n_total = int(summary.filter(summary["metric"] == "grid_points_total")["value"][0])
n_converged = int(summary.filter(summary["metric"] == "grid_points_converged")["value"][0])
n_pareto = int(summary.filter(summary["metric"] == "pareto_optimal_solutions")["value"][0])
profit_min_p = float(summary.filter(summary["metric"] == "profit_min")["value"][0])
profit_max_p = float(summary.filter(summary["metric"] == "profit_max")["value"][0])
ret_min_p = float(summary.filter(summary["metric"] == "retention_min")["value"][0])
ret_max_p = float(summary.filter(summary["metric"] == "retention_max")["value"][0])
fair_min_p = float(summary.filter(summary["metric"] == "fairness_disparity_min")["value"][0])
fair_max_p = float(summary.filter(summary["metric"] == "fairness_disparity_max")["value"][0])

print(f"  Grid points total:         {n_total}")
print(f"  Grid points converged:     {n_converged}")
print(f"  Non-dominated (Pareto):    {n_pareto}")
print()

if n_pareto > 0:
    print(f"  Pareto front ranges:")
    print(f"    Profit:     £{profit_min_p:,.0f} - £{profit_max_p:,.0f}")
    print(f"    Retention:  {ret_min_p:.1%} - {ret_max_p:.1%}")
    print(f"    Fairness:   {fair_min_p:.3f} - {fair_max_p:.3f}")
    print()

# ---------------------------------------------------------------------------
# PART 3: TOPSIS selection
# ---------------------------------------------------------------------------

print("=" * 70)
print("PART 3: TOPSIS-selected solution from Pareto surface")
print("        Weights: profit=0.50, retention=0.30, fairness=0.20")
print("=" * 70)
print()

topsis_ok = False
topsis_profit = float("nan")
topsis_retention = float("nan")
topsis_lr = float("nan")
topsis_gwp = float("nan")
topsis_fairness = float("nan")

if n_pareto > 0:
    try:
        pareto_result = pareto_result.select(
            method="topsis",
            weights=(0.50, 0.30, 0.20),
        )
        selected = pareto_result.selected

        topsis_profit = selected.expected_profit
        topsis_retention = selected.expected_retention
        topsis_lr = selected.expected_loss_ratio
        topsis_gwp = selected.expected_gwp
        topsis_fairness = float(selected.audit_trail.get("fairness", float("nan")))

        print(f"  TOPSIS selected grid point: "
              f"({selected.audit_trail['grid_i']}, {selected.audit_trail['grid_j']})")
        print(f"  eps_x (retention floor):   {selected.audit_trail['eps_x']:.3f}")
        print(f"  eps_y (fairness cap):      {selected.audit_trail['eps_y']:.3f}")
        print()
        print(f"  Profit:                    £{topsis_profit:>12,.0f}")
        print(f"  GWP:                       £{topsis_gwp:>12,.0f}")
        print(f"  Loss ratio:                {topsis_lr:>12.2%}")
        if topsis_retention is not None:
            print(f"  Retention:                 {topsis_retention:>12.1%}")
        print(f"  Fairness disparity:        {topsis_fairness:>12.3f}")
        topsis_ok = True

    except Exception as e:
        print(f"  TOPSIS selection failed: {e}")
else:
    print("  No non-dominated solutions — Pareto selection skipped.")
    print("  Check that sweep ranges are within the feasible region.")

print()

# ---------------------------------------------------------------------------
# PART 4: Comparison
# ---------------------------------------------------------------------------

print("=" * 70)
print("COMPARISON: Single-objective vs TOPSIS selection from Pareto surface")
print("=" * 70)
print()

def _fmt(v, fmt="f"):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a".rjust(15)
    if fmt == "gbp":
        return f"£{v:,.0f}".rjust(15)
    if fmt == "pct":
        return f"{v:.1%}".rjust(15)
    if fmt == "ratio":
        return f"{v:.3f}".rjust(15)
    return f"{v:.2f}".rjust(15)

print(f"{'Metric':<35} {'Single-obj':>15} {'Pareto TOPSIS':>15}")
print("-" * 70)
print(f"  {'Profit':<33} {_fmt(single_profit, 'gbp')} {_fmt(topsis_profit, 'gbp')}")
print(f"  {'GWP':<33} {_fmt(single_gwp, 'gbp')} {_fmt(topsis_gwp, 'gbp')}")
print(f"  {'Loss ratio':<33} {_fmt(single_lr, 'pct')} {_fmt(topsis_lr, 'pct')}")
print(f"  {'Retention':<33} {_fmt(single_retention, 'pct')} {_fmt(topsis_retention, 'pct')}")
print(f"  {'Fairness disparity ratio':<33} {_fmt(single_fairness, 'ratio')} {_fmt(topsis_fairness, 'ratio')}")
print()

if topsis_ok and single_profit > 0 and topsis_profit == topsis_profit:
    profit_gap_pct = (topsis_profit - single_profit) / single_profit * 100
    fairness_gain = single_fairness - topsis_fairness

    print("KEY FINDINGS")
    print(f"  Single-objective fairness disparity:  {single_fairness:.3f}")
    print(f"  TOPSIS-selected fairness disparity:   {topsis_fairness:.3f}")
    print(f"  Disparity change:                     {fairness_gain:+.3f} "
          f"({fairness_gain / max(single_fairness, 1e-9) * 100:+.1f}%)")
    print(f"  Profit trade-off:                     {profit_gap_pct:+.1f}%")
    print()
    print(f"  Pareto surface: {n_pareto} non-dominated solutions from {n_total} grid points.")
    print(f"  Single-objective SLSQP found 1 solution without trade-off visibility.")
    print()
    print("  The Pareto surface hands the trade-off decision to the right people:")
    print("  the board can see exactly what fairness costs in profit terms.")
    print()

# ---------------------------------------------------------------------------
# PART 5: Pareto surface detail
# ---------------------------------------------------------------------------

print("=" * 70)
print("PARETO SURFACE DETAIL (non-dominated solutions)")
print("=" * 70)
print()

pareto_df = pareto_result.pareto_df
if len(pareto_df) > 0:
    print(f"  {'Retention':>12}  {'Fairness':>10}  {'Profit':>12}  {'LR':>8}")
    print("  " + "-" * 52)
    for row in pareto_df.sort("retention").iter_rows(named=True):
        print(
            f"  {row['retention']:>12.1%}  "
            f"{row['fairness']:>10.3f}  "
            f"£{row['profit']:>10,.0f}  "
            f"{row['loss_ratio']:>7.2%}"
        )
else:
    print("  No non-dominated solutions (all grid points failed to converge).")
    print("  This means the fairness cap is too tight for the portfolio.")
print()

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

elapsed = time.time() - BENCHMARK_START
print("=" * 70)
print("TIMING")
print("=" * 70)
print(f"  Single-objective SLSQP:    {single_time:.2f}s (1 solve)")
print(f"  ParetoFrontier sweep:      {pareto_time:.1f}s ({N_POINTS*N_POINTS} solves)")
print(f"  Per-solve average:         {pareto_time/(N_POINTS*N_POINTS):.2f}s")
print(f"  Total benchmark:           {elapsed:.1f}s")
print()
print("Benchmark complete.")
