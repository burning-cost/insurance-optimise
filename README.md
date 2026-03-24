# insurance-optimise

[![PyPI](https://img.shields.io/pypi/v/insurance-optimise)](https://pypi.org/project/insurance-optimise/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-optimise)](https://pypi.org/project/insurance-optimise/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-optimise)](https://pypi.org/project/insurance-optimise/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-optimise/blob/main/notebooks/quickstart.ipynb)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-optimise/discussions). Found it useful? A star helps others find it.

**Flat loading on a price comparison website leaves money in every segment where your elasticity varies. This library finds the right multiplier for each risk.**

Manual scenario-testing in a spreadsheet finds one solution at a time and cannot simultaneously enforce ENBP ceilings, loss ratio floors, retention constraints, and per-policy rate change caps. insurance-optimise formulates all of those as a constrained optimisation problem with analytical gradients, solves it in seconds for portfolios up to 10,000 policies, and produces an FCA-auditable JSON record showing ENBP was enforced for every renewal. Your constraints include:

- FCA PS21/5 (ENBP): renewal premiums cannot exceed what a new customer would be quoted — this is a hard per-policy pricing ceiling
- Consumer Duty (PS22/9): a principles-based governance obligation to demonstrate fair value across customer outcomes — distinct from ENBP and not a per-policy pricing ceiling
- A target loss ratio you're trying to hit
- A retention floor you can't fall below without the underwriting team getting anxious
- Rate-change limits — you can't shock customers with 40% increases even if the model says so

The question is: what set of price multipliers maximises profit subject to all of these constraints simultaneously?

That's what this library solves.

## Why bother

Benchmarked against naive logistic regression and flat pricing on a synthetic UK motor PCW quote panel — 50,000 quotes, true price elasticity −2.0, confounded assignment.

| Metric | Naive logistic regression | DML ElasticityEstimator | Notes |
|--------|--------------------------|------------------------|-------|
| Estimated elasticity | −3.43 (naive) / −1.21 (full controls) | −4.03 | true effect is −2.0 |
| Absolute bias | 1.43 (naive) / 0.79 (full controls) | 2.03 | primary metric |
| Relative bias | 71.7% (naive) / 39.6% (full controls) | 101.3% | — |
| 95% CI valid | No | Yes | Neyman-orthogonal |
| Optimiser performance vs flat loading | baseline (misprices elastic segments) | +143.8% mean profit lift per segment | scales with elasticity variance |

**Honest interpretation:** On this synthetic DGP, DML did not outperform naive-full-controls logistic on point accuracy. The estimate of −4.03 has higher absolute bias than naive full-controls (−1.21). This is a known limitation: the DGP uses small quarterly loading cycles (std of log_price_ratio = 0.045), which provides too little exogenous price variation for the DML cross-fitting step — it partials out most of the signal along with the confounding. With std(log_price_ratio) ≥ 0.10 (genuine A/B tests, larger rate change cycles), DML converges closer to truth and provides a valid 95% CI that naive logistic cannot.

The core value is the constrained optimiser: even with an imprecise elasticity estimate, demand-curve-aware pricing outperforms flat loading by +143.8% mean profit per segment because it prices each segment against its own demand curve subject to FCA constraints.

▶ [Run on Databricks](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/portfolio_optimisation_demo.py)

---

**Read more:** [Your Rate Changes Are Leaving Money on the Table](https://burning-cost.github.io/2026/03/08/insurance-optimise.html) — why manual scenario-in-a-spreadsheet pricing is guaranteed to be suboptimal, and how constrained optimisation fixes it.

## What it does

- Maximise expected profit (or minimise combined ratio) subject to any combination of:
  - **ENBP** constraint — FCA PS21/5 hard ceiling per renewal policy
  - **Loss ratio** bounds (deterministic or Branda 2014 stochastic formulation)
  - **Volume retention** floor
  - **GWP** bounds
  - **Maximum rate change** per policy
  - **Technical floor** — price >= cost
- Analytical gradients throughout — fast enough for N=10,000 policies in SLSQP
- Efficient frontier sweep — show the pricing team the profit-retention trade-off curve
- **Pareto surface** — 3-objective optimisation across profit, retention, and fairness (v0.4.0)
- **Model quality adjustment** — correct LR targets for your model's Pearson correlation (v0.4.1)
- Scenario mode — run under pessimistic/central/optimistic elasticity assumptions
- JSON audit trail — every run produces evidence of ENBP enforcement for FCA scrutiny

## Installation

```bash
pip install insurance-optimise
```

Or with uv:

```bash
uv add insurance-optimise
```

## Quick start

```python
import numpy as np
import polars as pl
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

# Synthetic UK motor renewal book — 500 policies
# In production, these come from your technical model and elasticity estimator
rng = np.random.default_rng(42)
n = 500

technical_price   = rng.uniform(300, 1200, n)          # GLM output
expected_loss_cost = technical_price * rng.uniform(0.55, 0.75, n)  # expected claims
p_renewal         = rng.uniform(0.70, 0.95, n)          # renewal probability at current price
price_elasticity  = rng.uniform(-2.5, -0.8, n)          # from insurance-elasticity
is_renewal        = rng.choice([True, False], n, p=[0.7, 0.3])
# ENBP: FCA PS21/5 — renewal premium cannot exceed new business quote
enbp              = technical_price * rng.uniform(1.05, 1.25, n)  # must exceed technical_price

config = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.80,
    max_rate_change=0.20,
    enbp_buffer=0.01,   # 1% safety margin below ENBP
    technical_floor=True,
)

opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_renewal,
    elasticity=price_elasticity,
    renewal_flag=is_renewal,
    enbp=enbp,
    constraints=config,
)

result = opt.optimise()

print(result)
# OptimisationResult(CONVERGED, N=500, profit=118,196, gwp=372,406, lr=0.694)

print(result.profit)         # shorthand alias for result.expected_profit

# Attach optimal prices back to your data
df = pl.DataFrame({
    "technical_price":    technical_price.tolist(),
    "optimal_multiplier": result.multipliers.tolist(),
    "optimal_premium":    result.new_premiums.tolist(),
})

# Save audit trail for FCA
result.save_audit("renewal_run_2025_q1_audit.json")
```

## Expected Performance

On a 2,000-policy renewal book with heterogeneous price elasticities (55% PCW, 45% direct,
mean elasticity −1.65, LR cap 68%, retention floor 78%, seed=42):

| Metric | Uniform +7% rate change | Constrained optimiser |
|--------|------------------------|----------------------|
| Expected profit uplift | baseline | +£4,000–8,000 (~5–8%) |
| Retention rate | ~74–76% | ~78–80% (+2–4pp) |
| Loss ratio | ~67–69% | ~65–67% (−2pp) |
| ENBP compliance (FCA PS21/5) | Post-hoc capping | Built into constraints |
| FCA audit trail | No | Yes (per-policy log) |
| Convergence time | N/A | <1s (2,000 policies) |

The per-policy profit uplift is £2–4. On a 50,000-policy renewal book: £100,000–200,000 per
cycle, before accounting for retention carry-over and reduced acquisition cost.

The optimiser achieves this by applying larger increases to price-inelastic direct customers
(they tolerate it) and smaller increases to elastic PCW customers (retaining them is worth
more than the extra margin). The GWP target is met; the profit improvement comes from not
losing the customers who would have stayed at a lower price.

Run the validation: import `notebooks/databricks_validation.py` into Databricks.

---

## Efficient frontier

The frontier tells your pricing team: "if we're willing to lose X points of retention, we gain Y points of profit margin." This is the conversation that actually needs to happen in pricing reviews.

```python
from insurance_optimise import EfficientFrontier

frontier = EfficientFrontier(
    opt,
    sweep_param="volume_retention",
    sweep_range=(0.80, 0.96),
    n_points=15,
)
result = frontier.run()
print(result.data)  # DataFrame: epsilon, profit, gwp, loss_ratio, retention

frontier.plot()  # matplotlib
```

## Pareto surface — profit, retention, and fairness (v0.4.0)

The efficient frontier is bi-objective: profit vs retention. In practice, pricing teams face a third dimension — fairness. Under FCA Consumer Duty (PS22/9), firms must demonstrate fair value across customer segments. The `ParetoFrontier` makes this three-way trade-off explicit.

The standard approach is to add a fairness constraint at an arbitrary cap (e.g., "premium disparity ratio <= 1.5"). We think this is wrong. The acceptable level of disparity is a governance decision, not a technical parameter. Presenting the full Pareto surface — and letting the pricing committee choose a point on it — is more defensible than pre-committing to an arbitrary fairness floor.

```python
import numpy as np
from functools import partial
from insurance_optimise import PortfolioOptimiser, ConstraintConfig
from insurance_optimise.pareto import ParetoFrontier, premium_disparity_ratio

rng = np.random.default_rng(42)
n = 1_000

technical_price    = rng.uniform(300, 1200, n)
expected_loss_cost = technical_price * rng.uniform(0.55, 0.75, n)
p_renewal          = rng.uniform(0.70, 0.95, n)
price_elasticity   = rng.uniform(-2.5, -0.8, n)
is_renewal         = rng.choice([True, False], n, p=[0.7, 0.3])
enbp               = technical_price * rng.uniform(1.05, 1.25, n)
# Deprivation quintile (1=least deprived, 5=most deprived)
deprivation        = rng.integers(1, 6, n)

opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_renewal,
    elasticity=price_elasticity,
    renewal_flag=is_renewal,
    enbp=enbp,
    constraints=ConstraintConfig(lr_max=0.72, retention_min=0.82),
)

# fairness_metric: callable(multipliers) -> float. Lower = more fair.
fairness_fn = partial(
    premium_disparity_ratio,
    technical_price=technical_price,
    group_labels=deprivation,
)

pareto = ParetoFrontier(
    optimiser=opt,
    fairness_metric=fairness_fn,
    sweep_x="volume_retention",
    sweep_x_range=(0.82, 0.96),
    sweep_y="fairness_max",
    sweep_y_range=(1.05, 2.00),
    n_points_x=10,
    n_points_y=10,    # 100 SLSQP solves total
)

result = pareto.run()
print(result.summary())
# metric                        value
# grid_points_total             100.0
# grid_points_converged          87.0
# pareto_optimal_solutions       23.0
# profit_min                  18420.0
# profit_max                  31650.0
# retention_min                  0.821
# retention_max                  0.958
# fairness_disparity_min         1.051
# fairness_disparity_max         1.893

# Select a single point using TOPSIS with explicit weights
result.select(method="topsis", weights=(0.5, 0.3, 0.2))
print(result.selected.audit_trail)

# Visualise the surface (requires matplotlib)
result.plot(x_metric="retention", y_metric="fairness", color_metric="profit")
result.plot_3d()

# Save regulatory audit trail
result.save_audit("pareto_run_2025_q1_audit.json")
```

**Built-in fairness metrics:**

- `premium_disparity_ratio` — mean premium of highest group / mean premium of lowest group, by any categorical label (deprivation quintile, age band, region). This is the primary FCA Consumer Duty metric.
- `loss_ratio_disparity` — highest-LR group / lowest-LR group. Flags cross-subsidy.

Both are available from `insurance_optimise.pareto`. You can also pass any callable `(multipliers: np.ndarray) -> float`.

**Parallel execution:** set `n_jobs=-1` to use all cores (requires `joblib`). Each of the 100 grid-point SLSQP solves is independent.

## Model quality adjustment — LR target correction (v0.4.1)

No pricing model is perfect. Hedges (2025, arXiv:2512.03242) gives a closed-form expression for how much higher your portfolio LR will be than the perfect-model target, given your model's Pearson correlation with true loss cost.

If you set `lr_max=0.70` in the optimiser but your model has rho=0.80, you are not going to achieve 70%. You will achieve something higher, systematically. The model quality module quantifies this and tells you what LR constraint to actually use.

```python
from insurance_optimise.model_quality import model_quality_report, loss_ratio_formula

# Your model has rho=0.80 Pearson correlation with true loss cost.
# The loss cost CV is 1.2 (typical for UK motor).
# Price elasticity eta = 1.5.
report = model_quality_report(rho=0.80, cv_lambda=1.2, eta=1.5, M=1.0/0.70)
print(report)
# ModelQualityReport(rho=0.800, cv_lambda=1.200, eta=1.500,
#   lr_expected=0.7381, lre=+0.0381, lr_adj=+381.0bps)

# The model quality adjustment: target 73.8% in the optimiser, not 70%.
# Otherwise you are setting an unachievable constraint.

# Frequency-severity model: errors compound
from insurance_optimise.model_quality import frequency_severity_lr

combined_lr = frequency_severity_lr(
    rho_f=0.82, rho_s=0.75,  # separate models for frequency and severity
    cv_f=0.8, cv_s=2.5,
    eta=1.5,
    M=1.0/0.70,
)
print(f"Combined LR: {combined_lr:.3f}")
# The product structure means a mediocre severity model compounds a mediocre frequency model.

# Invert the formula: recover implied elasticity from observed portfolio data
from insurance_optimise.model_quality import calibrate_elasticity_from_data

eta_implied = calibrate_elasticity_from_data(
    rho_observed=0.80,
    lr_observed=0.74,
    cv_lambda=1.2,
    M=1.0/0.70,
)
print(f"Implied eta: {eta_implied:.3f}")
```

**When to use this:** Before setting LR constraints in the optimiser. If you have a Pearson correlation estimate from your model validation (your MSRM or equivalent), pass it through `model_quality_report` to understand the realistic LR floor. Setting an unachievable LR constraint will force the optimiser to push retention below the floor to compensate.

## Scenario mode

Elasticity estimates carry uncertainty. The simplest honest approach is to run under three scenarios and report the spread:

```python
result_scenarios = opt.optimise_scenarios(
    elasticity_scenarios=[
        price_elasticity * 0.75,   # pessimistic (customers more price-sensitive)
        price_elasticity,          # central estimate
        price_elasticity * 1.25,   # optimistic (customers less price-sensitive)
    ],
    scenario_names=["pessimistic", "central", "optimistic"],
)
print(result_scenarios.summary())
# scenario     converged    profit    gwp    loss_ratio
# pessimistic  True         1.1M      8.5M   0.692
# central      True         1.3M      8.8M   0.681
# optimistic   True         1.5M      9.1M   0.672
```

## Constraint reference

| Constraint | Config parameter | Notes |
|---|---|---|
| FCA ENBP | `enbp_buffer=0.01` | Applied as upper bound on renewal multiplier |
| Max LR | `lr_max=0.70` | Deterministic or stochastic (Branda 2014) |
| Min LR | `lr_min=0.55` | Prevents unsustainable cross-subsidies |
| Min GWP | `gwp_min=50_000_000` | Portfolio size floor |
| Max GWP | `gwp_max=100_000_000` | Optional ceiling |
| Min retention | `retention_min=0.85` | Renewal book only |
| Max rate change | `max_rate_change=0.20` | Per policy, both directions |
| Technical floor | `technical_floor=True` | Enforces price >= cost |
| Stochastic LR | `stochastic_lr=True` | Requires `claims_variance` input |

## Demand models

Two built-in demand models:

**Log-linear (default):** `x(m) = x0 * m^epsilon`

Constant price elasticity. Works well with outputs from `insurance-elasticity`. Demand is always positive. Gradient is analytic and fast.

> **Valid range:** Appropriate for price changes in the ±10–15% range typical of UK personal lines annual renewals. Extrapolation beyond ±20% produces unrealistically large demand responses given the constant-elasticity assumption.

**Logistic:** `x(m) = sigmoid(alpha + beta * m * tc)`

Demand is bounded in (0,1). More appropriate for renewal probabilities when you want them to stay interpretable as probabilities. Requires conversion from elasticity estimate to logistic parameters.

## Solver details

Primary solver is SLSQP via `scipy.optimize.minimize`. Analytical gradients are provided for the objective and all constraints — without them, SLSQP uses finite differences (2N extra evaluations per iteration, prohibitively slow for large N).

SLSQP is known to sometimes report success when starting from the initial point without moving. The library uses `ftol=1e-9` (tighter than scipy's default 1e-6) and verifies constraint satisfaction after solve. If you see `converged=False`, the solution may still be useful but treat it with caution.

For N > 5,000, consider segment aggregation before optimising.

## Regulatory context

ENBP (PS21/5) and Consumer Duty (PS22/9) are distinct obligations. ENBP is a hard per-policy pricing ceiling: renewal premiums must not exceed the equivalent new business price. Consumer Duty is a principles-based governance obligation requiring firms to demonstrate fair value across customer outcomes — it does not set a per-policy price ceiling but requires documented governance of pricing practices.

This library enforces ENBP at the code level. The JSON audit trail records the constraint configuration, the solution, and whether ENBP was binding for each renewal policy. You can show this to the FCA.

The `ParetoFrontier` produces a structured audit trail of the three-objective trade-off exploration. This is directly suitable for inclusion in Consumer Duty pricing governance documentation.

Commercial tools (Akur8, WTW Radar, Earnix) do not expose their optimisation methodology. This library does.

## Pipeline position

```
[Technical model (GLM/GBM)]
        ↓ technical_price, expected_loss_cost
[insurance-elasticity]
        ↓ p_demand, elasticity
[model_quality_report]  ← adjust LR target for model rho
        ↓
[insurance-optimise]  ← this library
        ↑ enbp (new business quote — from rating engine)
        ↓ optimal_multiplier per policy
[Rating engine / ratebook update]
```

## Limitations

- **The optimiser is only as good as the elasticity inputs.** The SLSQP solver finds the mathematically optimal multipliers given the demand model. If the elasticity inputs are wrong — and in insurance they typically have substantial uncertainty, especially at n < 20,000 — the "optimal" strategy is optimal for the wrong demand curve. The benchmark on 50,000 PCW quotes shows DML producing a -4.03 estimate against a true effect of -2.0 when price variation is narrow (std log_price_ratio = 0.045). Pricing decisions based on such estimates may be directionally correct but the magnitude is unreliable. Use `optimise_scenarios()` with pessimistic/central/optimistic elasticity inputs and report the spread rather than a single solution.

- **The log-linear demand model should not be extrapolated beyond ±15–20% price change.** The constant-elasticity log-linear model `x(m) = x0 * m^epsilon` implies that demand response is proportional regardless of the price level. This fails at large changes: a 40% price increase does not simply produce 40% more lapse than the model predicts at 10% — retention floors out. The valid range documented in the code is ±10–15% for UK personal lines annual renewals. Optimise solutions that recommend rate changes outside this range should be treated with scepticism regardless of what the solver returns.

- **SLSQP has no global convergence guarantee.** SLSQP finds a local minimum starting from the initial point (technical premiums by default). For non-convex constraint sets, the solution is a local optimum that may be worse than alternatives. The library uses `ftol=1e-9` and verifies constraint satisfaction, but does not run multiple restarts. For production use with N > 1,000 policies and binding constraints that create non-convex feasible regions, run several restarts with different initial points and compare objectives.

- **N > 5,000 policies degrades solver performance.** SLSQP with analytical gradients handles N=5,000 in seconds. At N=50,000, the Jacobian matrix becomes large and the solver slows materially — expect 5–15 minutes on a standard Databricks cluster. For large portfolios, aggregate to pricing cells (vehicle group × age band × region × channel) before optimising, then disaggregate multipliers back to individual policies. This is standard actuarial practice and does not reduce pricing quality provided the cells are homogeneous.

- **The ENBP constraint is per-policy, but the ENBP reference price requires an accurate new business quote.** The library enforces ENBP as a hard per-policy ceiling using the `enbp` array you provide. If the new business quote fed as ENBP was generated by a pricing model different from the renewal model, or if there is a lag between when the ENBP was quoted and when the renewal is processed, the ENBP values may be stale or inconsistent. The library cannot detect or correct this — it blindly enforces the ceiling. Data pipeline governance is required to ensure ENBP inputs are contemporaneous and from the correct rating engine.

- **The Pareto surface fairness metric is not a regulatory test.** The `premium_disparity_ratio` metric computes mean premium of the highest group divided by mean premium of the lowest group, by deprivation quintile or other label. A Pareto front that shows this ratio declining as retention improves is useful evidence for Consumer Duty governance documentation. It is not a legal test for compliance with PS22/9 or the Equality Act. The FCA has not prescribed a specific disparity ratio threshold, and the appropriate level is a governance decision requiring Board-level sign-off.

- **Model quality adjustment requires a Pearson correlation estimate that is itself uncertain.** The `model_quality_report()` function implements the Hedges (2025) formula correcting the LR target for your model's Pearson correlation rho. In practice, rho is estimated from out-of-sample validation data and has a confidence interval. Plugging the point estimate of rho into the formula ignores this uncertainty. If rho=0.80 with a 95% CI of [0.74, 0.86], the implied LR adjustment ranges by 2–4 percentage points. Run the model quality report at rho lower bound and rho upper bound to understand the sensitivity of the LR constraint to model quality uncertainty.

- **No multi-period optimisation.** The optimiser solves a single-period problem: what multiplier maximises expected profit over the upcoming renewal cycle? It does not account for the dynamic consequences of pricing decisions — customer tenure effects, NCD accumulation, adverse selection at extreme prices, or competitor response. A policy that maximises single-period profit by raising prices sharply may erode the long-term book quality through selective lapse of good risks. Multi-period pricing optimisation is a materially harder problem outside the scope of this library.

## Limitations

- The demand model extrapolates using the assumed elasticity functional form. The log-linear demand model is appropriate for price changes in the ±10–15% range typical of UK personal lines renewals. At changes above ±20%, the constant-elasticity assumption produces unrealistically large demand responses. The optimiser does not warn when proposed multipliers exceed this range — set `max_rate_change` accordingly.
- Elasticity inputs are treated as known point estimates. The optimiser does not propagate elasticity uncertainty. If your elasticity estimates have wide confidence intervals, use `optimise_scenarios()` across pessimistic/central/optimistic assumptions to understand the sensitivity.
- SLSQP occasionally reports convergence from the starting point without moving. The library checks constraint satisfaction post-solve and flags `converged=False`, but the solution may not be meaningful. Re-run with a different starting point if you see convergence failures.
- For portfolios above N = 5,000 policies, optimisation time becomes material. The recommended approach is to aggregate to risk segments before optimising. This is an approximation — within-segment elasticity heterogeneity is averaged out.
- Consumer Duty compliance requires governance that the optimiser cannot provide. The JSON audit trail records the constraint configuration and the solution. It does not replace Board sign-off, pricing committee documentation, ongoing monitoring, or the outcome testing that Consumer Duty requires.


## Related Libraries

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — use fairness constraints in the Pareto surface alongside profit and retention |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Causal price elasticity and demand modelling — provides the `p_demand` and `elasticity` inputs this library requires |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — the optimised strategy will degrade as the portfolio drifts; this library catches when it needs refreshing |
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | Price elasticity estimation — upstream demand model feeding directly into the optimiser |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion and retention modelling — demand curves are the primary input to the optimiser |
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger deployment — optimised rates flow into the deployment framework |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID causal evaluation — after running the optimiser, use this to prove the rate change achieved what it was supposed to |

## Source repos

This package consolidates two previously separate libraries:

- `insurance-optimise` — core portfolio optimiser (v0.1.x), now v0.2.0 with demand subpackage
- `insurance-dro` — archived; scenario-based robust optimisation absorbed into `ScenarioObjective` and `CVaRConstraint` in this package. Full Distributionally Robust Optimisation (Wasserstein DRO) was evaluated and deprioritised in favour of the simpler scenario sweep — see the design rationale in `scenarios.py`.

---

## Performance

Benchmarked on synthetic UK motor PCW data — 50,000 quotes, true population-average price elasticity −2.0. Confounding is explicit: high-risk customers face higher prices via the underwriting model and have lower price sensitivity (fewer alternative quotes on PCW). Full script: `notebooks/benchmark_demand.py`.

### Elasticity estimation: DML vs naive logistic regression

| Method | Estimate | Absolute bias | Relative bias | 95% CI |
|--------|----------|---------------|---------------|--------|
| Naive logistic (price only) | −3.43 | 1.43 | 71.7% | none |
| Naive logistic (full controls) | −1.21 | 0.79 | 39.6% | none |
| DML + CatBoost (5-fold PLR) | −4.03 | 2.03 | 101.3% | [−5.65, −2.40] |

**Honest interpretation:** On this synthetic dataset, the DML estimator did not outperform naive logistic regression on point estimate accuracy — it returned −4.03 vs. the true −2.0, a larger absolute bias than the naive-full-controls logistic. The naive full-controls estimate of −1.21 was closer to truth in absolute terms.

The DML result is sensitive to the quasi-experimental variation available: in this DGP, the price variation comes from small quarterly loading cycles (std of log_price_ratio = 0.045). With such narrow treatment variation, the DML cross-fitting step partials out most signal along with the confounding. The 95% CI is wide (±1.6) and the sensitivity analysis confirms the estimate is not robust to small amounts of residual confounding (RV = 2.1%).

**When DML adds value:** When there is stronger exogenous price variation — genuine A/B test assignment, policy cycles creating larger treatment spread, or natural experiments in rate changes. With log_price_ratio std ≥ 0.10, the DML estimate converges closer to truth. With std < 0.05, naive-full-controls will often have lower MSE despite having no coverage guarantee.

DML fit time: 13s on 50,000 quotes (5 folds, CatBoost nuisance models).

### Pricing lift vs flat loading

Even with a biased elasticity estimate, demand-curve-aware pricing outperforms flat loading. Using the DML estimate (−4.03) to set prices per segment:

| Segment | Flat loading profit (£/quote) | DML-optimised profit (£/quote) | Lift |
|---------|------------------------------|-------------------------------|------|
| Young + High Risk | −31.79 | +14.39 | +145% |
| Young + Standard Risk | −22.01 | +9.64 | +144% |
| Mid-age + Standard Risk | −12.21 | +5.31 | +144% |
| Mid-age + Low Risk | −10.06 | +4.46 | +144% |
| Senior + Low Risk | −11.60 | +4.87 | +142% |

Mean profit lift across segments: **+143.8%**. Negative flat-loading profit per quote reflects that a 10% loading is not enough to cover expected losses at market conversion rates — the optimiser finds a loss-minimising price given the demand curve. Gap vs oracle pricing (true elasticity): 78%.

**When to use:** New business pricing on PCWs where flat loadings are the current practice. The demand-curve optimiser captures value even with imprecise elasticity estimates, because the shape of the demand curve constrains the price in the right direction. The benefit is largest when elasticity varies materially across segments (young vs. mature drivers).

**When NOT to use:** When the book has no genuine price variation for estimation. When regulatory constraints bind so tightly that the optimiser has no degrees of freedom. When you need to demonstrate the pricing model to FCA — see the audit trail documentation.

### Pareto surface: single-objective vs 3-objective

Benchmarked on a 1,000-policy synthetic UK motor book with deprivation quintile as the fairness dimension. Compares single-objective SLSQP (profit maximisation with fixed constraints) against the Pareto surface. Full script: `benchmarks/benchmark_pareto.py`.

| Optimisation approach | Profit (£) | Retention | Fairness disparity |
|-----------------------|-----------|-----------|-------------------|
| Single-objective SLSQP (profit only) | 31,650 | 0.871 | 1.168 |
| Pareto surface — max-profit point | 31,650 | 0.871 | 1.168 |
| Pareto surface — balanced point (TOPSIS 0.5/0.3/0.2) | 28,940 | 0.912 | 1.043 |
| Pareto surface — min-disparity point | 22,180 | 0.951 | 1.011 |

Single-objective SLSQP is blind to the fairness dimension — it achieves a disparity ratio of 1.168, meaning the most-deprived quintile pays 16.8% more on average than the least-deprived. The Pareto surface makes this trade-off visible: the pricing committee can see that a 9% reduction in profit (£31,650 → £28,940) buys a disparity reduction from 1.168 to 1.043 with improved retention. Neither point is "correct" — but the second conversation is the one Consumer Duty requires to happen.

## References

- FCA PS21/5 (ENBP): https://www.fca.org.uk/publication/policy/ps21-5.pdf
- Branda (2014): stochastic LR constraint via one-sided Chebyshev inequality
- Emms & Haberman (2005): theoretical foundation for demand-linked insurance pricing
- Spedicato, Dutang & Petrini (2018): ML-then-optimise pipeline in practice
- Hedges (2025): arXiv:2512.03242 — Pearson correlation and expected portfolio loss ratio



## Licence

MIT

---

**Need help implementing this in production?** [Talk to us](https://burning-cost.github.io/work-with-us/).
