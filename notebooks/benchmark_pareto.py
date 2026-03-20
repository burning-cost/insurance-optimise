# Databricks notebook source
# MAGIC %md
# MAGIC # ParetoFrontier vs Single-Objective SLSQP
# MAGIC
# MAGIC **insurance-optimise v0.4.0**
# MAGIC
# MAGIC This notebook benchmarks the new `ParetoFrontier` class against the existing
# MAGIC single-objective SLSQP optimiser. The central question: does a 3-objective
# MAGIC Pareto surface (profit, retention, fairness) reveal trade-offs that the
# MAGIC single-objective approach misses?
# MAGIC
# MAGIC It does. SLSQP maximises profit subject to a retention floor. It ignores
# MAGIC fairness entirely. The Pareto surface maps the full trade-off space — giving
# MAGIC the pricing team, actuarial function, and board a governed view of what
# MAGIC fairness costs in profit terms.
# MAGIC
# MAGIC Run time on a single-node cluster: approximately 30-60 seconds (6x6 grid,
# MAGIC N=1,500 policies, sequential SLSQP).

# COMMAND ----------

# MAGIC %pip install insurance-optimise>=0.4.0

# COMMAND ----------

import warnings
import time
from functools import partial

import numpy as np

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md ## 1. Import library

# COMMAND ----------

from insurance_optimise import PortfolioOptimiser, ConstraintConfig, __version__
from insurance_optimise.pareto import ParetoFrontier, premium_disparity_ratio

print(f"insurance-optimise {__version__}")

# COMMAND ----------

# MAGIC %md ## 2. Synthetic portfolio (1,500 UK motor renewal policies)

# COMMAND ----------

N = 1_500
SEED = 42
rng = np.random.default_rng(SEED)

# Technical premiums: realistic UK motor range
log_tc = rng.normal(6.2, 0.38, N)
technical_price = np.exp(log_tc)

# Expected loss costs
loss_ratio_factor = rng.beta(6, 3.5, N).clip(0.40, 0.88)
expected_loss_cost = technical_price * loss_ratio_factor

# Elasticities: PCW customers more elastic than direct
segment = rng.choice(["pcw", "direct"], N, p=[0.58, 0.42])
elast_mean = np.where(segment == "pcw", -2.1, -1.25)
elasticity = np.clip(elast_mean + rng.normal(0, 0.28, N), -4.0, -0.3)

# Baseline renewal probability
p_demand = rng.beta(12, 2, N).clip(0.70, 0.97)  # high baseline ~86% so retention floor is achievable

# All renewals
renewal_flag = np.ones(N, dtype=bool)

# ENBP: PS21/11 new-business equivalent price
enbp = technical_price * rng.lognormal(0.09, 0.055, N)

# Prior multiplier
prior_multiplier = np.clip(np.ones(N) * rng.lognormal(0.0, 0.025, N), 0.90, 1.15)

# Deprivation quintile (1=most deprived, 5=least deprived)
# PCW customers are mildly more likely to be in higher quintiles (less deprived)
# because they are comparison-shopping. Vectorized assignment avoids per-policy loop.
_pcw_mask = segment == "pcw"
deprivation_quintile = np.where(
    _pcw_mask,
    rng.choice([1, 2, 3, 4, 5], N, p=[0.10, 0.15, 0.20, 0.30, 0.25]),
    rng.choice([1, 2, 3, 4, 5], N, p=[0.30, 0.30, 0.20, 0.12, 0.08]),
)

print(f"Policies:           {N:,}")
print(f"Mean tech price:    £{technical_price.mean():.0f}")
print(f"Mean loss ratio:    {loss_ratio_factor.mean():.2%}")
print(f"Mean elasticity:    {elasticity.mean():.2f}")
print(f"Mean base retention:{p_demand.mean():.1%}")

# COMMAND ----------

# MAGIC %md ## 3. Define fairness metric

# COMMAND ----------

fairness_fn = partial(
    premium_disparity_ratio,
    technical_price=technical_price,
    group_labels=deprivation_quintile,
)

# Baseline fairness at no rate change
baseline_disparity = fairness_fn(prior_multiplier)
print(f"Baseline disparity ratio (no rate change): {baseline_disparity:.3f}")
print("Interpretation: highest-deprivation group pays {:.1f}x the mean premium of lowest".format(
    baseline_disparity
))

# COMMAND ----------

# MAGIC %md ## 4. Single-objective SLSQP (profit maximisation only)
# MAGIC
# MAGIC This is how most UK pricing teams run the optimiser today: max profit,
# MAGIC retention floor, LR cap. No fairness objective or constraint.

# COMMAND ----------

config_single = ConstraintConfig(
    lr_max=0.68,
    retention_min=0.82,
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
    constraints=config_single,
    demand_model="log_linear",
    n_restarts=1,
    seed=SEED,
)

t0 = time.time()
single_result = opt.optimise()
single_time = time.time() - t0

# Compute fairness at this solution — the team would not normally do this
single_fairness = fairness_fn(single_result.multipliers)

print(f"Converged:          {single_result.converged}")
print(f"Time:               {single_time:.2f}s")
print()
print(f"Profit:             £{single_result.expected_profit:,.0f}")
print(f"GWP:                £{single_result.expected_gwp:,.0f}")
print(f"Loss ratio:         {single_result.expected_loss_ratio:.2%}")
print(f"Retention:          {single_result.expected_retention:.1%}")
print(f"Fairness disparity: {single_fairness:.3f}  <- blind spot for this approach")

# COMMAND ----------

# MAGIC %md ## 5. ParetoFrontier 2D sweep (6x6 grid)
# MAGIC
# MAGIC 36 independent SLSQP solves. x-axis sweeps retention (0.78-0.92).
# MAGIC y-axis sweeps fairness disparity cap (1.05-2.00). Every point on the
# MAGIC surface is Pareto-optimal at its (eps_x, eps_y) constraint values.

# COMMAND ----------

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

N_POINTS = 6

pf = ParetoFrontier(
    optimiser=opt_pareto,
    fairness_metric=fairness_fn,
    sweep_x="volume_retention",
    sweep_x_range=(0.78, 0.92),
    sweep_y="fairness_max",
    sweep_y_range=(1.05, 2.00),
    n_points_x=N_POINTS,
    n_points_y=N_POINTS,
    n_jobs=1,
)

print(f"Grid: {N_POINTS}x{N_POINTS} = {N_POINTS*N_POINTS} solves")
t0 = time.time()
pareto_result = pf.run()
pareto_time = time.time() - t0
print(f"Completed in {pareto_time:.1f}s ({pareto_time/(N_POINTS*N_POINTS):.2f}s/solve)")

# COMMAND ----------

# MAGIC %md ## 6. Pareto surface summary

# COMMAND ----------

summary = pareto_result.summary()
print(summary)

# COMMAND ----------

# MAGIC %md ## 7. Non-dominated solutions

# COMMAND ----------

pareto_df = pareto_result.pareto_df
pareto_sorted = pareto_df.sort("retention")
print(f"Non-dominated solutions: {len(pareto_df)}")
print()
print(f"{'Retention':>12}  {'Fairness':>10}  {'Profit':>12}  {'LR':>8}")
print("-" * 50)
for row in pareto_sorted.iter_rows(named=True):
    print(
        f"{row['retention']:>12.1%}  "
        f"{row['fairness']:>10.3f}  "
        f"£{row['profit']:>10,.0f}  "
        f"{row['loss_ratio']:>7.2%}"
    )

# COMMAND ----------

# MAGIC %md ## 8. TOPSIS selection (profit=50%, retention=30%, fairness=20%)

# COMMAND ----------

pareto_result = pareto_result.select(
    method="topsis",
    weights=(0.50, 0.30, 0.20),
)
selected = pareto_result.selected

print(f"TOPSIS selected grid ({selected.audit_trail['grid_i']}, {selected.audit_trail['grid_j']})")
print(f"eps_x (retention floor): {selected.audit_trail['eps_x']:.3f}")
print(f"eps_y (fairness cap):    {selected.audit_trail['eps_y']:.3f}")
print()
print(f"Profit:              £{selected.expected_profit:,.0f}")
print(f"Loss ratio:          {selected.expected_loss_ratio:.2%}")
print(f"Retention:           {selected.expected_retention:.1%}" if selected.expected_retention else "")
print(f"Fairness disparity:  {selected.audit_trail['fairness']:.3f}")

# COMMAND ----------

# MAGIC %md ## 9. Comparison: single-objective vs TOPSIS selection

# COMMAND ----------

so_profit = single_result.expected_profit
to_profit = selected.expected_profit
so_fair = single_fairness
to_fair = float(selected.audit_trail.get("fairness", float("nan")))

print(f"{'Metric':<35} {'Single-obj':>14} {'Pareto TOPSIS':>14}")
print("-" * 65)
print(f"  {'Profit':<33} £{so_profit:>10,.0f} £{to_profit:>10,.0f}")
print(f"  {'Loss ratio':<33} {single_result.expected_loss_ratio:>14.2%} {selected.expected_loss_ratio:>14.2%}")
so_ret = single_result.expected_retention or 0.0
to_ret = selected.expected_retention or 0.0
print(f"  {'Retention':<33} {so_ret:>14.1%} {to_ret:>14.1%}")
print(f"  {'Fairness disparity ratio':<33} {so_fair:>14.3f} {to_fair:>14.3f}")
print()

profit_gap = (to_profit - so_profit) / abs(so_profit) * 100 if so_profit != 0 else 0.0
fairness_gain = so_fair - to_fair

print(f"Profit trade-off:          {profit_gap:+.1f}%")
print(f"Fairness improvement:      {fairness_gain:+.3f} ({fairness_gain/so_fair*100:+.0f}%)")
print()
print("The Pareto surface reveals what the single-objective approach hides:")
print(f"  Accepting {abs(profit_gap):.1f}% less profit reduces premium disparity")
print(f"  across deprivation quintiles from {so_fair:.3f} to {to_fair:.3f}.")
print("  With single-objective SLSQP, this trade-off is completely invisible.")

# COMMAND ----------

# MAGIC %md ## 10. Timing

# COMMAND ----------

print(f"Single-objective SLSQP: {single_time:.2f}s (1 solve)")
print(f"ParetoFrontier sweep:   {pareto_time:.1f}s ({N_POINTS*N_POINTS} solves)")
print(f"Per-solve average:      {pareto_time/(N_POINTS*N_POINTS):.2f}s")
