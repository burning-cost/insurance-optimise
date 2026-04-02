# Databricks notebook source
# MAGIC %md
# MAGIC # RobustReinsuranceOptimiser Demo
# MAGIC
# MAGIC ## The problem it solves
# MAGIC
# MAGIC Every insurer buying proportional reinsurance faces two questions:
# MAGIC 1. **How much to cede?** The standard de Finetti answer: cede enough that
# MAGIC    the net surplus volatility is tolerable. But this ignores uncertainty
# MAGIC    in the drift estimate.
# MAGIC 2. **What to do when you don't trust your own model?** If your loss model
# MAGIC    might be wrong — the drift mu is uncertain — the optimal cession is
# MAGIC    higher than the classical solution.
# MAGIC
# MAGIC `RobustReinsuranceOptimiser` solves the robust dividend-reinsurance
# MAGIC problem from Boonen, Dela Vega, and Garces (2026, arXiv:2603.25350).
# MAGIC The insurer maximises expected discounted dividends against an adversary
# MAGIC who can shift the drift within an ambiguity set (DRO-style uncertainty).
# MAGIC
# MAGIC **Two solvers:**
# MAGIC - Symmetric closed-form (ODE shooting): identical lines, no correlation
# MAGIC - Asymmetric numerical PDE: different line parameters or nonzero correlation

# COMMAND ----------
# MAGIC %pip install insurance-optimise>=0.5.0 polars matplotlib --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from insurance_optimise import (
    RobustReinsuranceOptimiser,
    RobustReinsuranceResult,
    ReinsuranceLine,
)

print("insurance-optimise imported successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Single line — motor book
# MAGIC
# MAGIC A UK motor book with:
# MAGIC - mu = 2.0: expected surplus generation (£M per year)
# MAGIC - sigma = 3.0: surplus volatility
# MAGIC - reins_loading = 3.5: reinsurer charges 3.5x the expected ceded loss
# MAGIC - ambiguity = 0.0: start with full confidence in the model
# MAGIC
# MAGIC We solve for the optimal cession fraction pi*(x) as a function of
# MAGIC the current surplus x.

# COMMAND ----------

motor = ReinsuranceLine(
    name="motor",
    mu=2.0,
    sigma=3.0,
    reins_loading=3.5,
    ambiguity=0.0,    # no model uncertainty (classical de Finetti)
)

opt_classical = RobustReinsuranceOptimiser(
    lines=[motor],
    delta=0.05,      # 5% discount rate
    surplus_max=40.0,
    n_grid=300,
)

result_classical = opt_classical.optimise()
print(result_classical)
print(f"\nSolver:            {result_classical.solver}")
print(f"Converged:         {result_classical.converged}")
print(f"Dividend barrier b*: {result_classical.dividend_barrier:.2f}")
print(f"Cession at surplus=0:  {result_classical.pi_at_zero[0]:.3f}")
print(f"Cession at barrier:    {result_classical.pi_at_barrier[0]:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Effect of model uncertainty (ambiguity) on cession
# MAGIC
# MAGIC Higher ambiguity (theta) means the insurer distrusts its own drift estimate
# MAGIC and acts as if the true drift could be lower. This raises the optimal
# MAGIC cession fraction — a more conservative reinsurance strategy.
# MAGIC
# MAGIC This is the DRO interpretation: the insurer robustifies against the
# MAGIC worst-case model within an ambiguity set.

# COMMAND ----------

ambiguity_levels = [0.0, 0.1, 0.3, 0.6, 1.0]
surplus_query = 10.0  # surplus at which we compare cession

print(f"Optimal cession at surplus x={surplus_query} for different ambiguity levels:")
print(f"{'Ambiguity (theta)':>20} {'Cession pi*(x)':>16} {'Barrier b*':>12}")
print("-" * 52)

for theta in ambiguity_levels:
    line = ReinsuranceLine(name="motor", mu=2.0, sigma=3.0,
                           reins_loading=3.5, ambiguity=theta)
    opt = RobustReinsuranceOptimiser(
        lines=[line], delta=0.05, surplus_max=40.0, n_grid=300
    )
    result = opt.optimise()
    # Interpolate from schedule
    sched = result.cession_schedule
    x_grid = sched["x"].to_numpy()
    pi_grid = sched["pi"].to_numpy()
    pi_at_x = float(np.interp(surplus_query, x_grid, pi_grid))
    print(f"{theta:>20.1f} {pi_at_x:>16.4f} {result.dividend_barrier:>12.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Cession schedule — pi*(x) over the surplus range

# COMMAND ----------

# Compare two ambiguity levels
fig, ax = plt.subplots(figsize=(10, 5))

colors = {"classical (theta=0)": "#1f77b4", "robust (theta=0.5)": "#d62728"}

for theta, label in [(0.0, "classical (theta=0)"), (0.5, "robust (theta=0.5)")]:
    line = ReinsuranceLine(name="motor", mu=2.0, sigma=3.0,
                           reins_loading=3.5, ambiguity=theta)
    opt = RobustReinsuranceOptimiser(lines=[line], delta=0.05,
                                      surplus_max=40.0, n_grid=300)
    result = opt.optimise()
    sched = result.cession_schedule
    ax.plot(sched["x"].to_numpy(), sched["pi"].to_numpy(),
            lw=2, color=colors[label], label=f"{label}: b*={result.dividend_barrier:.1f}")
    ax.axvline(result.dividend_barrier, color=colors[label], ls="--", lw=1, alpha=0.6)

ax.set_xlabel("Aggregate surplus x (£M)")
ax.set_ylabel("Optimal cession fraction pi*(x)")
ax.set_title("Classical vs Robust Reinsurance: Cession Schedule")
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Sensitivity analysis
# MAGIC
# MAGIC How does the optimal cession change as we vary ambiguity (theta) or
# MAGIC reinsurance loading? The sensitivity() method sweeps one parameter while
# MAGIC holding all others fixed.

# COMMAND ----------

motor_baseline = ReinsuranceLine(
    name="motor", mu=2.0, sigma=3.0, reins_loading=3.5, ambiguity=0.2
)
opt_sens = RobustReinsuranceOptimiser(
    lines=[motor_baseline], delta=0.05, surplus_max=40.0, n_grid=300
)

# Sensitivity to ambiguity
df_ambig = opt_sens.sensitivity(param="ambiguity", n_points=12)
print("Sensitivity to ambiguity (theta):")
print(df_ambig)

# COMMAND ----------

# Sensitivity to reinsurance loading
df_load = opt_sens.sensitivity(param="loading", n_points=10)
print("\nSensitivity to reinsurance loading (c):")
print(df_load)

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Ambiguity sensitivity
ax = axes[0]
x = df_ambig["param_value"].to_numpy()
y_cession = df_ambig["cession_fraction"].to_numpy()
y_barrier = df_ambig["dividend_barrier"].to_numpy()
ax.plot(x, y_cession, "o-", color="#1f77b4", lw=2, label="Cession at baseline surplus")
ax2 = ax.twinx()
ax2.plot(x, y_barrier, "s--", color="#ff7f0e", lw=1.5, label="Dividend barrier b*")
ax.set_xlabel("Ambiguity theta")
ax.set_ylabel("Cession fraction pi*(x)", color="#1f77b4")
ax2.set_ylabel("Dividend barrier b*", color="#ff7f0e")
ax.set_title("Effect of Model Uncertainty on Reinsurance Strategy")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Loading sensitivity
ax = axes[1]
x2 = df_load["param_value"].to_numpy()
y2 = df_load["cession_fraction"].to_numpy()
y2b = df_load["dividend_barrier"].to_numpy()
ax.plot(x2, y2, "o-", color="#2ca02c", lw=2)
ax.set_xlabel("Reinsurance loading c")
ax.set_ylabel("Cession fraction pi*(x)")
ax.set_title("Higher Loading => Less Reinsurance (More Expensive)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Two-line asymmetric problem — motor and property
# MAGIC
# MAGIC When the two lines have different parameters, the symmetric closed-form
# MAGIC no longer applies. The optimiser falls back to the 2D value-iteration PDE
# MAGIC solver on a grid [0, surplus_max]^2.

# COMMAND ----------

motor_line = ReinsuranceLine(
    name="motor",
    mu=2.0,
    sigma=3.0,
    reins_loading=3.5,
    ambiguity=0.15,
)
property_line = ReinsuranceLine(
    name="property",
    mu=1.5,
    sigma=2.5,
    reins_loading=2.8,
    ambiguity=0.10,
)

opt_2line = RobustReinsuranceOptimiser(
    lines=[motor_line, property_line],
    delta=0.05,
    surplus_max=30.0,
    n_grid=50,    # 50x50 grid for the 2D problem
    tol=1e-5,
)

print("Solving 2-line asymmetric problem (PDE value iteration)...")
result_2line = opt_2line.optimise()

print(result_2line)
print(f"\nSolver:   {result_2line.solver}")
print(f"Converged:{result_2line.converged}")
print(f"n_iter:   {result_2line.n_iter}")
print(f"\nDividend barrier b*: {result_2line.dividend_barrier:.2f}")
print(f"Motor cession at zero surplus:    {result_2line.pi_at_zero[0]:.3f}")
print(f"Property cession at zero surplus: {result_2line.pi_at_zero[1]:.3f}")
print(f"Motor cession at barrier:         {result_2line.pi_at_barrier[0]:.3f}")
print(f"Property cession at barrier:      {result_2line.pi_at_barrier[1]:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cession schedule for the two-line problem

# COMMAND ----------

sched_2 = result_2line.cession_schedule
print(f"Cession schedule shape: {sched_2.shape}")
print(f"Columns: {sched_2.columns}")
print(sched_2.head(10))

# COMMAND ----------

# Plot the cession schedule as a heatmap
ax = result_2line.plot_cession_schedule()
if ax is not None:
    plt.tight_layout()
    plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Audit trail
# MAGIC
# MAGIC Every optimisation result carries a JSON-serialisable audit trail.
# MAGIC For Lloyd's syndicates and PRA-regulated firms, this documents the
# MAGIC methodology and parameterisation of the reinsurance decision.

# COMMAND ----------

import json

audit = result_2line.audit_trail
print("=== Audit Trail ===")
print(f"Timestamp: {audit.get('timestamp_utc', 'N/A')}")
print(f"\nLines:")
for ln in audit.get("lines", []):
    print(f"  {ln['name']}: mu={ln['mu']}, sigma={ln['sigma']}, "
          f"loading={ln['reins_loading']}, theta={ln['ambiguity']}")

print(f"\nSolution:")
sol = audit.get("solution", {})
print(f"  Dividend barrier b*: {sol.get('dividend_barrier', 'N/A')}")
print(f"  Cession at zero:     {sol.get('pi_at_zero', 'N/A')}")
print(f"  Cession at barrier:  {sol.get('pi_at_barrier', 'N/A')}")
print(f"  Solver:              {sol.get('solver', 'N/A')}")
print(f"  Converged:           {sol.get('converged', 'N/A')}")

# Save to disk (in practice: write to MLflow or a governance database)
audit_path = "/tmp/reinsurance_audit.json"
result_2line.save_audit(audit_path)
print(f"\nAudit trail saved to: {audit_path}")

# Verify round-trip
with open(audit_path) as f:
    loaded = json.load(f)
print(f"JSON round-trip: {len(loaded)} top-level keys")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Practical interpretation
# MAGIC
# MAGIC **pi*(x) is decreasing in surplus.** When the insurer holds ample surplus,
# MAGIC it retains more risk (less reinsurance). At the brink of ruin (x near 0),
# MAGIC full cession protects against immediate bankruptcy.
# MAGIC
# MAGIC **Higher ambiguity => higher cession.** If the insurer is uncertain whether
# MAGIC its drift estimate mu is correct, it buys more reinsurance as insurance
# MAGIC against the model being wrong. This is the key DRO result.
# MAGIC
# MAGIC **Higher reinsurance loading => less cession + higher barrier.** Expensive
# MAGIC reinsurance pushes the insurer to self-insure more, but it holds more
# MAGIC surplus as a buffer before paying dividends.
# MAGIC
# MAGIC **Dividend barrier b*.** All dividends are paid when the surplus hits b*.
# MAGIC A higher loading makes b* larger — the insurer wants to hold more before
# MAGIC rewarding shareholders.
# MAGIC
# MAGIC **Solvency II ORSA use.** The ambiguity parameter quantifies parameter
# MAGIC uncertainty in your loss model. Running this optimiser across a range of
# MAGIC theta values gives you a robustness interval for the optimal cession decision.

# COMMAND ----------

print("Demo complete.")
dbutils.notebook.exit("RobustReinsuranceOptimiser demo completed successfully.")
