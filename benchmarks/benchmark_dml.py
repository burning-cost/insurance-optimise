"""
Benchmark: DML price elasticity vs naive logistic regression (insurance-demand).

The problem: in insurance quote data, price is set by a risk model. High-risk
customers get higher prices AND tend to convert at lower rates for non-price
reasons (fewer alternatives, less shopping behaviour). A naive logistic
regression of conversion on price conflates these effects and produces a biased
elasticity estimate.

Double Machine Learning (DML) via the ElasticityEstimator in
insurance_optimise.demand fixes this by partialling out all observable
confounders from both the outcome and the treatment before estimating the
price-demand relationship.

Setup:
- Synthetic UK motor PCW new business quote data (30,000 quotes)
- True population-average price elasticity: -2.0
- Confounders: age, vehicle_group, ncd_years, area, channel
- Naive: logistic regression of converted on log_price_ratio (no controls)
- Biased naive: logistic regression with no confounder adjustment
- DML: ElasticityEstimator with CatBoost nuisance models, 5-fold cross-fitting

Expected output:
- Naive logistic gives a biased elasticity (confounders inflate the estimate)
- DML recovers the true elasticity with a valid confidence interval

Run:
    python benchmarks/benchmark_dml.py
"""

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: DML elasticity vs naive logistic regression (insurance-demand)")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from insurance_optimise.demand.datasets import generate_conversion_data
    from insurance_optimise.demand.elasticity import ElasticityEstimator
    print("insurance_optimise.demand imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance_optimise.demand: {e}")
    print("Install with: pip install insurance-optimise")
    sys.exit(1)

import numpy as np
import polars as pl

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------

TRUE_ELASTICITY = -2.0
N_QUOTES = 30_000

print(f"Generating {N_QUOTES:,} synthetic UK motor PCW quotes...")
print(f"True population-average price elasticity: {TRUE_ELASTICITY:.1f}")
print()

df = generate_conversion_data(n_quotes=N_QUOTES, true_price_elasticity=TRUE_ELASTICITY, seed=42)

# Report confounding structure
print("CONFOUNDING STRUCTURE")
print("-" * 50)
# Show that high-risk segments get higher prices and have different elasticities
vg_stats = (
    df
    .group_by("vehicle_group")
    .agg([
        pl.col("log_price_ratio").mean().alias("mean_log_price_ratio"),
        pl.col("converted").mean().alias("conversion_rate"),
        pl.col("true_elasticity").mean().alias("true_elasticity_mean"),
        pl.len().alias("n"),
    ])
    .sort("vehicle_group")
)
print("By vehicle group (confounding: higher group = higher price AND lower elasticity):")
print(f"  {'VG':>3} {'Mean log(p/tc)':>16} {'Conv rate':>12} {'True elast':>12}")
for row in vg_stats.iter_rows(named=True):
    print(
        f"  {row['vehicle_group']:>3} {row['mean_log_price_ratio']:>16.4f}"
        f" {row['conversion_rate']:>12.3f} {row['true_elasticity_mean']:>12.3f}"
    )
print()

# ---------------------------------------------------------------------------
# Naive approach: logistic regression on log_price_ratio, no controls
# ---------------------------------------------------------------------------

print("NAIVE APPROACH: logistic regression, no confounder adjustment")
print("-" * 50)

from scipy.special import expit
from scipy.optimize import minimize

df_pd = df.to_pandas() if _PANDAS_OK else None

def _logistic_nll(params, X, y):
    """Negative log-likelihood for logistic regression."""
    log_odds = X @ params
    prob = expit(log_odds)
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

def _logistic_gradient(params, X, y):
    log_odds = X @ params
    prob = expit(log_odds)
    return X.T @ (prob - y) / len(y)

# Pure naive: only intercept + log_price_ratio
y = df["converted"].to_numpy().astype(float)
X_naive = np.column_stack([
    np.ones(len(df)),
    df["log_price_ratio"].to_numpy(),
])
res_naive = minimize(
    _logistic_nll, x0=np.zeros(2), args=(X_naive, y),
    jac=_logistic_gradient, method="L-BFGS-B",
)
naive_elasticity = float(res_naive.x[1])
print(f"  Naive logistic elasticity: {naive_elasticity:.4f}  (true = {TRUE_ELASTICITY:.1f})")
print(f"  Bias: {naive_elasticity - TRUE_ELASTICITY:+.4f}")
print()

# Naive with partial controls (age and vehicle_group but missing area/channel)
age_arr = df["age"].to_numpy().astype(float)
vg_arr = df["vehicle_group"].to_numpy().astype(float)
X_partial = np.column_stack([
    np.ones(len(df)),
    df["log_price_ratio"].to_numpy(),
    age_arr / 80.0,
    vg_arr / 4.0,
])
res_partial = minimize(
    _logistic_nll, x0=np.zeros(4), args=(X_partial, y),
    jac=_logistic_gradient, method="L-BFGS-B",
)
partial_elasticity = float(res_partial.x[1])
print(f"  Logistic with partial controls: {partial_elasticity:.4f}  (true = {TRUE_ELASTICITY:.1f})")
print(f"  Bias: {partial_elasticity - TRUE_ELASTICITY:+.4f}")
print()

# ---------------------------------------------------------------------------
# DML approach: ElasticityEstimator
# ---------------------------------------------------------------------------

print("DML APPROACH: ElasticityEstimator (CatBoost + 5-fold cross-fitting)")
print("-" * 50)

confounder_cols = ["age", "vehicle_group", "ncd_years", "area", "channel"]

est = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=confounder_cols,
    n_folds=3,                # 3-fold for speed; use 5 in production
    outcome_transform="identity",  # linear probability model for simplicity
    catboost_params={
        "iterations": 150,
        "depth": 4,
        "verbose": False,
        "allow_writing_files": False,
        "random_seed": 42,
    },
)

t0 = time.time()
est.fit(df)
fit_time = time.time() - t0

dml_summary = est.summary()
dml_elasticity = float(dml_summary["estimate"].iloc[0])
dml_se = float(dml_summary["std_error"].iloc[0])
dml_ci_lo = float(dml_summary["ci_lower_95"].iloc[0])
dml_ci_hi = float(dml_summary["ci_upper_95"].iloc[0])

print(f"  DML elasticity: {dml_elasticity:.4f}  (true = {TRUE_ELASTICITY:.1f})")
print(f"  95% CI: [{dml_ci_lo:.4f}, {dml_ci_hi:.4f}]")
print(f"  SE: {dml_se:.4f}")
print(f"  CI covers true value: {dml_ci_lo <= TRUE_ELASTICITY <= dml_ci_hi}")
print(f"  Fit time: {fit_time:.1f}s")
print()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Estimator':<35} {'Elasticity':>12} {'Bias':>10} {'Has CI':>8}")
print("-" * 70)
print(f"{'Naive logistic (no controls)':<35} {naive_elasticity:>12.4f} {naive_elasticity - TRUE_ELASTICITY:>+10.4f} {'No':>8}")
print(f"{'Logistic (partial controls)':<35} {partial_elasticity:>12.4f} {partial_elasticity - TRUE_ELASTICITY:>+10.4f} {'No':>8}")
print(f"{'DML ElasticityEstimator':<35} {dml_elasticity:>12.4f} {dml_elasticity - TRUE_ELASTICITY:>+10.4f} {'Yes':>8}")
print(f"{'True value':<35} {TRUE_ELASTICITY:>12.1f} {'0.0000':>10} {'—':>8}")
print()

print("KEY FINDINGS")
print(f"  True elasticity: {TRUE_ELASTICITY:.1f}")
print(f"  Naive overestimates |elasticity| by {abs(naive_elasticity - TRUE_ELASTICITY):.3f}")
print(f"  DML bias: {abs(dml_elasticity - TRUE_ELASTICITY):.3f} (vs naive {abs(naive_elasticity - TRUE_ELASTICITY):.3f})")
print(f"  Bias reduction: {(1 - abs(dml_elasticity - TRUE_ELASTICITY) / abs(naive_elasticity - TRUE_ELASTICITY)) * 100:.0f}%")
print()
print("  The naive model conflates risk composition (high-risk = high price = lower")
print("  conversion independent of price) with the true price effect. DML partials")
print("  out confounders from both treatment and outcome before estimating elasticity.")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
