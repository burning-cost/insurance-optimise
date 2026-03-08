# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-optimise: Portfolio Rate Optimisation Demo
# MAGIC
# MAGIC This notebook demonstrates the full workflow for constrained insurance portfolio
# MAGIC rate optimisation using the `insurance-optimise` library.
# MAGIC
# MAGIC **What we cover:**
# MAGIC 1. Generate a synthetic UK personal lines renewal book (1,000 policies)
# MAGIC 2. Run constrained optimisation with all typical UK constraints (ENBP, LR, retention, rate change)
# MAGIC 3. Inspect shadow prices to understand which constraints are binding
# MAGIC 4. Generate the efficient frontier (profit vs retention trade-off)
# MAGIC 5. Run scenario analysis with elasticity uncertainty
# MAGIC 6. Save and inspect the FCA audit trail
# MAGIC
# MAGIC **Pipeline context:**
# MAGIC This library sits downstream of `insurance-elasticity`. In production, the
# MAGIC `technical_price`, `expected_loss_cost`, `p_demand`, `elasticity`, and `enbp`
# MAGIC columns come from fitted GLM/GBM models. Here we generate them synthetically.

# COMMAND ----------

# MAGIC %pip install insurance-optimise polars matplotlib --quiet

# COMMAND ----------

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import json

from insurance_optimise import (
    PortfolioOptimiser,
    ConstraintConfig,
    EfficientFrontier,
)

print(f"insurance-optimise ready")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic UK Motor Renewal Book
# MAGIC
# MAGIC We generate 1,000 policies: 700 renewals, 300 new business.
# MAGIC Typical UK motor profile:
# MAGIC - Technical premium: £350 to £1,200
# MAGIC - Loss ratio: 55% to 75% of technical premium
# MAGIC - Renewal probability: 72% to 93% (at current price)
# MAGIC - Price elasticity: -0.8 to -2.5 (typical personal lines range)
# MAGIC - ENBP: 5% to 18% above technical premium (renewal book)

# COMMAND ----------

N = 1000
N_RENEWAL = 700
rng = np.random.default_rng(42)

# Technical premiums: mix of standard and high-risk
technical_price = rng.uniform(350, 1200, size=N)

# Loss costs: 55-75% of technical premium
loss_ratio_true = rng.uniform(0.55, 0.75, size=N)
expected_loss_cost = technical_price * loss_ratio_true

# Demand probabilities: higher-value customers tend to shop more (lower retention)
# Add a slight negative correlation between premium and retention
p_base = rng.uniform(0.72, 0.93, size=N)
p_demand = np.clip(p_base - (technical_price - 700) / 5000, 0.55, 0.97)

# Price elasticity: customers with lower premiums are more elastic (more price-sensitive)
# Typical range for UK motor: -0.8 to -2.5
elasticity = -(0.8 + 1.7 * (1 - (technical_price - 350) / 850))
elasticity += rng.normal(0, 0.2, size=N)
elasticity = np.clip(elasticity, -3.0, -0.3)

# Renewal flag
renewal_flag = np.zeros(N, dtype=bool)
renewal_idx = rng.choice(N, size=N_RENEWAL, replace=False)
renewal_flag[renewal_idx] = True

# ENBP: for renewals, what new business quote would they get?
# Typically 5-18% above technical premium
enbp = np.where(
    renewal_flag,
    technical_price * rng.uniform(1.05, 1.18, size=N),
    technical_price * 2.0,  # irrelevant for new business
)

# Prior multiplier: assume current pricing is near 1.0
prior_multiplier = rng.uniform(0.97, 1.03, size=N)

# Build a Polars DataFrame
df = pl.DataFrame({
    "policy_id": [f"POL{i:05d}" for i in range(N)],
    "is_renewal": renewal_flag.tolist(),
    "technical_price": technical_price.tolist(),
    "expected_loss_cost": expected_loss_cost.tolist(),
    "p_demand": p_demand.tolist(),
    "elasticity": elasticity.tolist(),
    "enbp": enbp.tolist(),
    "prior_multiplier": prior_multiplier.tolist(),
})

print(f"Portfolio: {N} policies, {N_RENEWAL} renewals, {N - N_RENEWAL} new business")
print(f"\nTechnical premium stats:")
print(f"  Mean: £{technical_price.mean():.0f}")
print(f"  P25-P75: £{np.percentile(technical_price, 25):.0f} - £{np.percentile(technical_price, 75):.0f}")
print(f"\nTrue LR distribution:")
print(f"  Mean: {loss_ratio_true.mean():.3f}")
print(f"  P25-P75: {np.percentile(loss_ratio_true, 25):.3f} - {np.percentile(loss_ratio_true, 75):.3f}")
print(f"\nElasticity distribution:")
print(f"  Mean: {elasticity.mean():.3f}")
print(f"  P25-P75: {np.percentile(elasticity, 25):.3f} - {np.percentile(elasticity, 75):.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Baseline Portfolio Metrics (Before Optimisation)
# MAGIC
# MAGIC What does the book look like at the current price (multiplier = 1.0)?

# COMMAND ----------

# Build optimiser (no constraint run first, just for baseline)
opt_baseline = PortfolioOptimiser(
    technical_price=df["technical_price"].to_numpy(),
    expected_loss_cost=df["expected_loss_cost"].to_numpy(),
    p_demand=df["p_demand"].to_numpy(),
    elasticity=df["elasticity"].to_numpy(),
    renewal_flag=df["is_renewal"].to_numpy(),
    enbp=df["enbp"].to_numpy(),
    prior_multiplier=df["prior_multiplier"].to_numpy(),
    constraints=ConstraintConfig(technical_floor=False),
)

baseline = opt_baseline.portfolio_summary()
print("=== Baseline Portfolio Metrics (current pricing) ===")
print(f"  Expected GWP:       £{baseline['gwp']:>12,.0f}")
print(f"  Expected Profit:    £{baseline['profit']:>12,.0f}")
print(f"  Loss Ratio:          {baseline['loss_ratio']:.3f}")
print(f"  Renewal Retention:   {baseline['retention']:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Constrained Optimisation
# MAGIC
# MAGIC Apply real-world UK constraints:
# MAGIC - **ENBP**: 1% safety buffer below FCA mandated ceiling
# MAGIC - **Loss ratio**: max 70% (typical commercial target)
# MAGIC - **Retention**: min 85% of renewal book
# MAGIC - **Rate change**: max ±20% per policy year
# MAGIC - **Technical floor**: price >= cost

# COMMAND ----------

config = ConstraintConfig(
    lr_max=0.70,
    retention_min=0.85,
    max_rate_change=0.20,
    enbp_buffer=0.01,     # 1% safety margin below ENBP
    technical_floor=True,
)

opt = PortfolioOptimiser(
    technical_price=df["technical_price"].to_numpy(),
    expected_loss_cost=df["expected_loss_cost"].to_numpy(),
    p_demand=df["p_demand"].to_numpy(),
    elasticity=df["elasticity"].to_numpy(),
    renewal_flag=df["is_renewal"].to_numpy(),
    enbp=df["enbp"].to_numpy(),
    prior_multiplier=df["prior_multiplier"].to_numpy(),
    constraints=config,
    n_restarts=3,  # 3 restarts to avoid local minima
    seed=42,
)

print(f"Active scipy constraints: {opt.n_constraints}")
print("Running optimisation...")
result = opt.optimise()

print(f"\nConverged: {result.converged}")
print(f"Solver: {result.solver_message}")
print(f"Iterations: {result.n_iter}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Results

# COMMAND ----------

print("=== Optimisation Results ===")
print(f"  Expected GWP:       £{result.expected_gwp:>12,.0f}   (was £{baseline['gwp']:,.0f})")
print(f"  Expected Profit:    £{result.expected_profit:>12,.0f}   (was £{baseline['profit']:,.0f})")
print(f"  Loss Ratio:          {result.expected_loss_ratio:.4f}   (was {baseline['loss_ratio']:.4f})")
print(f"  Renewal Retention:   {result.expected_retention:.4f}   (was {baseline['retention']:.4f})")
print(f"\n  Profit improvement: £{result.expected_profit - baseline['profit']:+,.0f}")
print(f"  LR improvement:     {(result.expected_loss_ratio - baseline['loss_ratio'])*100:+.2f} pp")

# Multiplier distribution
m = result.multipliers
print(f"\n=== Multiplier Distribution ===")
print(f"  Mean:    {m.mean():.4f}")
print(f"  Median:  {np.median(m):.4f}")
print(f"  P25-P75: {np.percentile(m, 25):.4f} - {np.percentile(m, 75):.4f}")
print(f"  Min-Max: {m.min():.4f} - {m.max():.4f}")

# ENBP binding
print(f"\n  ENBP binding: {result.summary_df['enbp_binding'].sum()} renewal policies")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Shadow Prices
# MAGIC
# MAGIC Shadow prices (Lagrange multipliers) tell you which constraints are binding
# MAGIC and what the marginal cost of each constraint is.
# MAGIC
# MAGIC A large shadow price means that constraint is expensive — relaxing it slightly
# MAGIC would improve profit noticeably.

# COMMAND ----------

print("=== Shadow Prices (Lagrange multipliers) ===")
for name, val in result.shadow_prices.items():
    status = "BINDING" if abs(val) > 1e-4 else "slack"
    print(f"  {name:<20s}: {val:+.6f}  [{status}]")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Per-Policy Output

# COMMAND ----------

# Join optimal prices back to the original DataFrame
output_df = df.with_columns([
    pl.Series("optimal_multiplier", result.multipliers),
    pl.Series("optimal_premium", result.new_premiums),
    pl.Series("expected_demand", result.expected_demand),
    pl.Series("rate_change_pct", result.summary_df["rate_change_pct"]),
    pl.Series("enbp_binding", result.summary_df["enbp_binding"]),
    pl.Series("contribution", result.summary_df["contribution"]),
])

print("Sample output (10 rows):")
display(output_df.select([
    "policy_id", "is_renewal", "technical_price", "optimal_premium",
    "optimal_multiplier", "rate_change_pct", "enbp_binding"
]).head(10))

# Rate change distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

rate_changes = result.summary_df["rate_change_pct"].to_numpy()
axes[0].hist(rate_changes, bins=40, color="#1f77b4", alpha=0.7)
axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_xlabel("Rate Change (%)")
axes[0].set_ylabel("Number of Policies")
axes[0].set_title("Distribution of Rate Changes")
axes[0].set_xlim(-25, 25)

# Premium distribution: current vs optimal
axes[1].scatter(
    technical_price[:200],
    result.new_premiums[:200],
    alpha=0.3, s=10, color="#1f77b4"
)
max_p = max(technical_price[:200].max(), result.new_premiums[:200].max())
axes[1].plot([0, max_p], [0, max_p], "k--", linewidth=1, label="No change")
axes[1].set_xlabel("Technical Price (£)")
axes[1].set_ylabel("Optimal Premium (£)")
axes[1].set_title("Technical vs Optimal Premium (sample 200 policies)")
axes[1].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Efficient Frontier
# MAGIC
# MAGIC Sweep the retention constraint from 78% to 93% and trace the
# MAGIC profit-retention Pareto frontier.
# MAGIC
# MAGIC This is the chart you bring to the pricing committee.

# COMMAND ----------

print("Generating efficient frontier (15 points)...")
ef = EfficientFrontier(
    opt,
    sweep_param="volume_retention",
    sweep_range=(0.78, 0.93),
    n_points=15,
)
frontier_result = ef.run()

print(f"\nFrontier points: {len(frontier_result.points)}")
print(f"Converged: {frontier_result.pareto_data()['converged'].sum()} / {len(frontier_result.data)}")
display(frontier_result.data)

# COMMAND ----------

# Plot the frontier
pareto = frontier_result.pareto_data()
if len(pareto) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Profit vs retention
    x = pareto["retention"].to_numpy()
    y = pareto["profit"].to_numpy() / 1e6  # millions

    axes[0].plot(x * 100, y, "o-", color="#1f77b4", linewidth=2, markersize=7)
    axes[0].axvline(85, color="red", linestyle="--", linewidth=1, label="Min retention (85%)")
    axes[0].set_xlabel("Expected Retention (%)")
    axes[0].set_ylabel("Expected Profit (£M)")
    axes[0].set_title("Efficient Frontier: Profit vs Retention")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss ratio vs retention
    lr = pareto["loss_ratio"].to_numpy()
    axes[1].plot(x * 100, lr * 100, "o-", color="#ff7f0e", linewidth=2, markersize=7)
    axes[1].set_xlabel("Expected Retention (%)")
    axes[1].set_ylabel("Loss Ratio (%)")
    axes[1].set_title("Loss Ratio vs Retention")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Profit-Retention Trade-off: The Efficient Frontier",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.show()

    # Marginal cost of retention
    if len(x) > 1:
        marginal_profit_per_retention_pp = np.diff(y * 1e6) / np.diff(x * 100)
        print("\nMarginal cost of 1 pp additional retention:")
        for i, (ret, cost_pp) in enumerate(zip(x[1:] * 100, marginal_profit_per_retention_pp)):
            print(f"  At retention {ret:.1f}%: £{-cost_pp:,.0f} profit per pp")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Scenario Analysis: Elasticity Uncertainty
# MAGIC
# MAGIC The elasticity estimates from `insurance-elasticity` have confidence intervals.
# MAGIC We run three scenarios: optimistic (less price-sensitive customers),
# MAGIC central, and pessimistic (more price-sensitive).

# COMMAND ----------

print("Running 3-scenario analysis (optimistic / central / pessimistic)...")
elasticity_central = df["elasticity"].to_numpy()

scenario_result = opt.optimise_scenarios(
    elasticity_scenarios=[
        elasticity_central * 0.70,   # optimistic: customers less price-sensitive
        elasticity_central,           # central estimate
        elasticity_central * 1.30,   # pessimistic: customers more price-sensitive
    ],
    scenario_names=["optimistic", "central", "pessimistic"],
)

print("\n=== Scenario Summary ===")
display(scenario_result.summary())

print(f"\nProfit range:")
print(f"  P10:  £{scenario_result.profit_p10:>12,.0f}")
print(f"  Mean: £{scenario_result.profit_mean:>12,.0f}")
print(f"  P90:  £{scenario_result.profit_p90:>12,.0f}")

# Multiplier uncertainty
m_mean = scenario_result.multiplier_mean
m_p10 = scenario_result.multiplier_p10
m_p90 = scenario_result.multiplier_p90
print(f"\nMultiplier uncertainty (portfolio average):")
print(f"  Mean multiplier: {m_mean.mean():.4f}")
print(f"  90% CI: [{m_p10.mean():.4f}, {m_p90.mean():.4f}]")

# COMMAND ----------

# Scenario multiplier comparison plot
fig, ax = plt.subplots(figsize=(10, 5))

sorted_idx = np.argsort(m_mean)[:100]  # first 100 policies for clarity
x_pos = np.arange(100)

ax.fill_between(
    x_pos, m_p10[sorted_idx], m_p90[sorted_idx],
    alpha=0.25, color="#1f77b4", label="P10-P90 range"
)
ax.plot(x_pos, m_mean[sorted_idx], color="#1f77b4", linewidth=1.5, label="Mean multiplier")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="No change")
ax.set_xlabel("Policy (sorted by mean multiplier)")
ax.set_ylabel("Optimal Price Multiplier")
ax.set_title("Multiplier Uncertainty Across Elasticity Scenarios (100 policies)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. FCA Audit Trail
# MAGIC
# MAGIC Every optimisation run produces a JSON audit trail. This is the evidence
# MAGIC that the ENBP constraint was enforced and the methodology is documented.
# MAGIC
# MAGIC Under Consumer Duty, you need to be able to explain your pricing to the FCA.
# MAGIC This is the start of that explanation.

# COMMAND ----------

audit = result.audit_trail

print("=== Audit Trail Summary ===")
print(f"\nLibrary:   {audit['library']} v{audit['version']}")
print(f"Timestamp: {audit['timestamp_utc']}")
print(f"\nInput summary:")
for k, v in audit["inputs"].items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

print(f"\nConstraints applied:")
for k, v in audit["constraints"].items():
    if v is not None and v is not False:
        print(f"  {k}: {v}")

print(f"\nSolution:")
sol = audit["solution"]
print(f"  Mean multiplier:    {sol['multiplier_mean']:.4f}")
print(f"  Mean rate change:   {sol['rate_change_mean_pct']:+.2f}%")

print(f"\nConstraint evaluation at solution:")
for k, v in audit["constraint_evaluation"].items():
    slack_status = "BINDING" if abs(v) < 0.01 else "slack"
    print(f"  {k:<20s}: {v:+.6f}  [{slack_status}]")

print(f"\nConvergence: {audit['convergence']['converged']}")
print(f"Iterations:  {audit['convergence']['n_iterations']}")
print(f"Fun evals:   {audit['convergence']['n_function_evaluations']}")

# COMMAND ----------

# Save audit trail to JSON
audit_path = "/tmp/renewal_optimisation_audit.json"
result.save_audit(audit_path)
print(f"Audit trail saved to: {audit_path}")

# Read back and verify
with open(audit_path) as f:
    loaded = json.load(f)
assert loaded["library"] == "insurance-optimise"
print("Audit trail JSON round-trip: OK")

# Show first 50 lines of JSON
json_str = json.dumps(audit, indent=2)
for line in json_str.split("\n")[:50]:
    print(line)
print("...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `insurance-optimise` library solves the constrained portfolio rate
# MAGIC optimisation problem with:
# MAGIC
# MAGIC - FCA PS21/11 ENBP constraint enforced at code level (not just policy)
# MAGIC - Analytical gradients for fast SLSQP convergence (N=1,000 in <2 seconds)
# MAGIC - Efficient frontier generation for stakeholder communication
# MAGIC - Scenario mode for elasticity uncertainty quantification
# MAGIC - JSON audit trail for regulatory evidence
# MAGIC
# MAGIC **Next steps in a real deployment:**
# MAGIC 1. Replace synthetic data with outputs from `insurance-elasticity`
# MAGIC 2. Aggregate optimal multipliers back to factor level for ratebook update
# MAGIC 3. Run A/B test: champion (current) vs challenger (optimal) on a random 10% of book
# MAGIC 4. Track uplift: (optimal_premium - current_premium) * realised_demand

# COMMAND ----------

print("Demo complete.")
print(f"Final result: {result}")
