# Databricks notebook source
# MAGIC %md
# MAGIC # Convex Reinsurance Optimisation: De Finetti Problem
# MAGIC
# MAGIC This notebook demonstrates `ConvexRiskReinsuranceOptimiser` from the
# MAGIC `insurance-optimise` library.
# MAGIC
# MAGIC **Paper:** Shyamalkumar & Wang (2026), "On a Class of Optimal Reinsurance
# MAGIC Problems", arXiv:2603.00813.
# MAGIC
# MAGIC **Problem:** Find reinsurance contracts R_i (one per line) that minimise
# MAGIC total ceded premium subject to a risk measure constraint on retained risk.
# MAGIC
# MAGIC Unlike classical stop-loss optimisation, this solver handles:
# MAGIC - Dependent risks (joint loss distribution)
# MAGIC - Heterogeneous safety loadings (different beta_i per line)
# MAGIC - Both CVaR and variance constraints
# MAGIC - Closed-form solutions via convex duality — no numerical optimisation

# COMMAND ----------

# %pip install insurance-optimise>=0.7.0

# COMMAND ----------

import numpy as np
import polars as pl

# If running from workspace upload rather than PyPI:
import sys
sys.path.insert(0, "/Workspace/insurance-optimise/src")

from insurance_optimise import (
    ConvexRiskReinsuranceOptimiser,
    RiskLine,
    ConvexReinsuranceResult,
)
from insurance_optimise.convex_reinsurance import _empirical_cvar

print("insurance-optimise loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario: Three-line UK commercial portfolio
# MAGIC
# MAGIC Motor fleet, commercial property, and employers' liability — a typical
# MAGIC mid-market commercial account. The three lines have heterogeneous
# MAGIC volatility and safety loadings. The reinsurer charges more for liability
# MAGIC (less liquid, harder to model) than for motor.

# COMMAND ----------

# Define the three risk lines
risks = [
    RiskLine(name="motor_fleet",   expected_loss=5_200, variance=9_100_000,  safety_loading=0.14),
    RiskLine(name="comm_property", expected_loss=3_800, variance=6_300_000,  safety_loading=0.21),
    RiskLine(name="employers_li",  expected_loss=1_600, variance=2_800_000,  safety_loading=0.32),
]

print("Risk lines:")
for r in risks:
    cv = (r.variance ** 0.5) / r.expected_loss
    print(f"  {r.name:20s}  E[X]={r.expected_loss:6.0f}  CV={cv:.2f}  beta={r.safety_loading:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate aggregate loss samples
# MAGIC
# MAGIC In practice you'd pass samples from your fitted collective risk model
# MAGIC (e.g., a frequency-severity simulation from your GLMs). Here we use a
# MAGIC correlated lognormal to illustrate the effect of dependence.

# COMMAND ----------

rng = np.random.default_rng(2026)
n_sim = 100_000

# Build a correlated scenario: motor and property mildly correlated (weather),
# liability independent
means = np.array([r.expected_loss for r in risks], dtype=float)
variances = np.array([r.variance for r in risks], dtype=float)

# Lognormal parameters matched to moments
sigma2_log = np.log(1.0 + variances / means**2)
mu_log = np.log(means) - 0.5 * sigma2_log

# Correlation structure: motor-property 0.25, rest 0
corr = np.array([
    [1.00, 0.25, 0.00],
    [0.25, 1.00, 0.00],
    [0.00, 0.00, 1.00],
])

# Gaussian copula
z = rng.multivariate_normal(np.zeros(3), corr, size=n_sim)
samples = np.exp(mu_log + np.sqrt(sigma2_log) * z)

S = samples.sum(axis=1)
print(f"Aggregate loss: mean={S.mean():.0f}, std={S.std():.0f}")
print(f"CVaR(99.5%) = {_empirical_cvar(S, 0.995):.0f}")
print(f"VaR(99.5%)  = {np.quantile(S, 0.995):.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solve the CVaR-constrained problem
# MAGIC
# MAGIC Target: reduce CVaR(99.5%) of retained aggregate by 20% relative to
# MAGIC no-reinsurance.

# COMMAND ----------

full_cvar = _empirical_cvar(S, 0.995)
budget = full_cvar * 0.80
print(f"Budget (80% of unconstrained CVaR): {budget:.0f}")

opt = ConvexRiskReinsuranceOptimiser(
    risks=risks,
    risk_measure="cvar",
    alpha=0.995,
    budget=budget,
    aggregate_loss_samples=samples,
)

result = opt.optimise()
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Contract details

# COMMAND ----------

summary = result.summary()
print(summary)

# Show which lines were ceded and at what cost
for c in result.contracts:
    status = "CEDED" if c["ceded"] else "RETAINED"
    print(
        f"  {c['name']:20s} [{status}]  "
        f"cession_rate={c['cession_rate']:.1%}  "
        f"ceded_premium={c['ceded_premium']:.0f}"
    )

print(f"\nTotal ceded premium: {result.total_ceded_premium:.0f}")
print(f"Retained CVaR:       {result.retained_risk:.0f}")
print(f"lambda*:             {result.lambda_star:.6f}")
print(f"\nAudit: {result.audit}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key insight: loading-ordered cession
# MAGIC
# MAGIC Motor fleet (beta=0.14) is ceded first — it's the cheapest to transfer.
# MAGIC Employers' liability (beta=0.32) is ceded last or not at all. This is
# MAGIC exactly the Theorem 3 ordering: cheapest risks go to the reinsurer first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Efficient frontier: ceded premium vs retained risk

# COMMAND ----------

frontier = opt.frontier(n_points=30)
print(frontier.head(10))

# The frontier shows the trade-off: lower retained risk costs more premium
# Each point is an optimal programme at a different budget level
print(f"\nFrontier range:")
print(f"  Min retained risk: {frontier['retained_risk'].min():.0f}")
print(f"  Max retained risk: {frontier['retained_risk'].max():.0f}")
print(f"  Min ceded premium: {frontier['total_ceded_premium'].min():.0f}")
print(f"  Max ceded premium: {frontier['total_ceded_premium'].max():.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sensitivity: how does the programme change with budget?

# COMMAND ----------

budgets = [full_cvar * f for f in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65]]
sens_budget = opt.sensitivity("budget", budgets)
print("Budget sensitivity:")
print(sens_budget)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sensitivity: motor loading

# COMMAND ----------

motor_loadings = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30]
sens_motor = opt.sensitivity("loading_motor_fleet", motor_loadings)
print("Motor fleet loading sensitivity:")
print(sens_motor)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare: CVaR vs Variance constraint
# MAGIC
# MAGIC CVaR focuses on the tail — it's the natural constraint for Solvency II
# MAGIC thinking. Variance penalises all deviations symmetrically. For a
# MAGIC right-skewed loss distribution the two give different programmes.

# COMMAND ----------

# Variance-constrained problem at same relative reduction
full_var = float(np.var(S, ddof=0))
budget_var = full_var * 0.80

opt_var = ConvexRiskReinsuranceOptimiser(
    risks=risks,
    risk_measure="variance",
    budget=budget_var,
    aggregate_loss_samples=samples,
)
result_var = opt_var.optimise()

print("CVaR-constrained result:")
for c in result.contracts:
    print(f"  {c['name']:20s}: cession_rate={c['cession_rate']:.1%}, ceded_premium={c['ceded_premium']:.0f}")
print(f"  Total ceded premium: {result.total_ceded_premium:.0f}")

print("\nVariance-constrained result:")
for c in result_var.contracts:
    print(f"  {c['name']:20s}: cession_rate={c['cession_rate']:.1%}, ceded_premium={c['ceded_premium']:.0f}")
print(f"  Total ceded premium: {result_var.total_ceded_premium:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Two-line analytical check
# MAGIC
# MAGIC Single-line CVaR problem has a known structure: if there's only one risk,
# MAGIC the optimal contract is a stop-loss (X - q)_+. Verify that the solver
# MAGIC finds this limit.

# COMMAND ----------

single_risk = [RiskLine(name="motor", expected_loss=5_200, variance=9_100_000, safety_loading=0.14)]
single_samples = samples[:, :1]
S_single = single_samples.sum(axis=1)
full_cvar_single = _empirical_cvar(S_single, 0.995)

opt_single = ConvexRiskReinsuranceOptimiser(
    risks=single_risk,
    risk_measure="cvar",
    alpha=0.995,
    budget=full_cvar_single * 0.80,
    aggregate_loss_samples=single_samples,
)
result_single = opt_single.optimise()
print("Single-line CVaR result:")
print(result_single)
print(f"Cession rate: {result_single.contracts[0]['cession_rate']:.1%}")
print(f"Ceded premium: {result_single.contracts[0]['ceded_premium']:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC `ConvexRiskReinsuranceOptimiser` solves the De Finetti multi-line reinsurance
# MAGIC problem in milliseconds via bisection + fixed-point iteration. The key
# MAGIC practical outputs for a pricing team:
# MAGIC
# MAGIC 1. **Which lines to cede** — ordered by loading (cheapest first).
# MAGIC 2. **Expected ceded loss and premium per line** — ready for quota share or
# MAGIC    stop-loss treaty pricing.
# MAGIC 3. **lambda*** — the shadow price of the risk constraint. Higher lambda* means
# MAGIC    the constraint is binding harder — you're paying more per unit of risk
# MAGIC    reduction.
# MAGIC 4. **Efficient frontier** — the trade-off between reinsurance cost and
# MAGIC    retained risk. Use this to have a structured conversation with a reinsurer
# MAGIC    about where on the curve the programme should sit.
# MAGIC
# MAGIC The solver works directly with samples from your fitted models — no moment
# MAGIC matching or distributional assumptions beyond what your GLMs already deliver.
