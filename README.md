# insurance-optimise

[![Tests](https://github.com/burning-cost/insurance-optimise/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-optimise/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-optimise)](https://pypi.org/project/insurance-optimise/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Constrained portfolio rate optimisation for UK personal lines insurance.

## The problem

You have a pricing model. It tells you the right technical price for each risk. But "technically correct" isn't the only constraint. You also have:

- FCA PS21/11: renewal premiums cannot exceed what a new customer would be quoted (ENBP)
- Consumer Duty: you need to demonstrate fair value, not just set prices actuarially
- A target loss ratio you're trying to hit
- A retention floor you can't fall below without the underwriting team getting anxious
- Rate-change limits — you can't shock customers with 40% increases even if the model says so

The question is: what set of price multipliers maximises profit subject to all of these constraints simultaneously?

That's what this library solves.

## What it does

- Maximise expected profit (or minimise combined ratio) subject to any combination of:
  - **ENBP** constraint — FCA PS21/11 hard ceiling per renewal policy
  - **Loss ratio** bounds (deterministic or Branda 2014 stochastic formulation)
  - **Volume retention** floor
  - **GWP** bounds
  - **Maximum rate change** per policy
  - **Technical floor** — price >= cost
- Analytical gradients throughout — fast enough for N=10,000 policies in SLSQP
- Efficient frontier sweep — show the pricing team the profit-retention trade-off curve
- Scenario mode — run under pessimistic/central/optimistic elasticity assumptions
- JSON audit trail — every run produces evidence of ENBP enforcement for FCA scrutiny

## Install

```bash
pip install insurance-optimise
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
# ENBP: FCA PS21/11 — renewal premium cannot exceed new business quote
enbp              = technical_price * rng.uniform(0.95, 1.10, n)

config = ConstraintConfig(
    lr_max=0.70,
    retention_min=0.85,
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
# OptimisationResult(converged=True, N=500, profit=..., gwp=..., lr=0.681)

# Attach optimal prices back to your data
df = pl.DataFrame({
    "technical_price":    technical_price.tolist(),
    "optimal_multiplier": result.multipliers.tolist(),
    "optimal_premium":    result.new_premiums.tolist(),
})

# Save audit trail for FCA
result.save_audit("renewal_run_2025_q1_audit.json")
```

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

**Logistic:** `x(m) = sigmoid(alpha + beta * m * tc)`

Demand is bounded in (0,1). More appropriate for renewal probabilities when you want them to stay interpretable as probabilities. Requires conversion from elasticity estimate to logistic parameters.

## Solver details

Primary solver is SLSQP via `scipy.optimize.minimize`. Analytical gradients are provided for the objective and all constraints — without them, SLSQP uses finite differences (2N extra evaluations per iteration, prohibitively slow for large N).

SLSQP is known to sometimes report success when starting from the initial point without moving. The library uses `ftol=1e-9` (tighter than scipy's default 1e-6) and verifies constraint satisfaction after solve. If you see `converged=False`, the solution may still be useful but treat it with caution.

For N > 5,000, consider segment aggregation before optimising.

## Regulatory context

Under FCA Consumer Duty (effective July 2023), firms must demonstrate that pricing practices deliver fair value. Under PS21/11, renewal premiums must not exceed the ENBP — this is not a soft target, it is enforceable.

This library enforces ENBP at the code level. The JSON audit trail records the constraint configuration, the solution, and whether ENBP was binding for each renewal policy. You can show this to the FCA.

Commercial tools (Akur8, WTW Radar, Earnix) do not expose their optimisation methodology. This library does.

## Pipeline position

```
[Technical model (GLM/GBM)]
        ↓ technical_price, expected_loss_cost
[insurance-elasticity]
        ↓ p_demand, elasticity, enbp
[insurance-optimise]  ← this library
        ↓ optimal_multiplier per policy
[Rating engine / ratebook update]
```

## Read more

[Your Rate Changes Are Leaving Money on the Table](https://burning-cost.github.io/2026/03/08/insurance-optimise.html) — why manual scenario-in-a-spreadsheet pricing is guaranteed to be suboptimal, and how constrained optimisation fixes it.

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | Price elasticity and demand modelling — provides the `p_demand` and `elasticity` inputs this library requires |
| [insurance-survival](https://github.com/burning-cost/insurance-survival) | Survival-adjusted CLV — use CLV outputs to inform retention constraints rather than setting them arbitrarily |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID causal evaluation — after running the optimiser, use this to prove the rate change achieved what it was supposed to |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — the optimised strategy will degrade as the portfolio drifts; this library catches when it needs refreshing |

[All Burning Cost libraries →](https://burning-cost.github.io)

## Source repos

This package consolidates two previously separate libraries:

- `insurance-optimise` — core portfolio optimiser (v0.1.x), now v0.2.0 with demand subpackage
- `insurance-dro` — archived; scenario-based robust optimisation absorbed into `ScenarioObjective` and `CVaRConstraint` in this package. Full Distributionally Robust Optimisation (Wasserstein DRO) was evaluated and deprioritised in favour of the simpler scenario sweep — see the design rationale in `scenarios.py`.

---

## Performance

Benchmarked against **naive logistic regression** (for elasticity estimation) and **flat pricing** (for commercial impact) on synthetic UK motor PCW quote panel — 50,000 quotes, true price elasticity −2.0, confounded assignment (high-risk customers face higher prices and have lower sensitivity). Full notebook: `notebooks/benchmark_demand.py`.

| Metric | Naive logistic regression | DML ElasticityEstimator | Notes |
|--------|--------------------------|------------------------|-------|
| Estimated elasticity | biased (conflates risk and price effects) | near −2.0 | true effect is −2.0 |
| Absolute bias | substantial (direction: overestimates sensitivity) | near zero | primary metric |
| 95% CI valid | no | yes | Neyman-orthogonal |

The benchmark then uses the estimated elasticities to compare revenue per quote under demand-curve-aware pricing against flat loading across all segments. Segments with heterogeneous elasticities (young drivers vs. mature drivers on PCWs, for example) are systematically mispriced by flat loading — the optimiser captures revenue by pricing to each segment's actual demand curve.

**When to use:** New business pricing on price comparison websites where some segments are highly elastic and others are captive. The combination of DML elasticity estimation and constrained optimisation is justified when elasticity varies materially across the book and the ENBP constraint is binding.

**When NOT to use:** When price is randomly assigned (genuine A/B test) — naive regression is unbiased and DML adds no value. When the book is small or the treatment variation is thin, the DML confidence intervals will be wide and the optimiser will produce near-flat recommendations anyway.


## References

- FCA PS21/11 (ENBP): https://www.fca.org.uk/publication/policy/ps21-11.pdf
- Branda (2014): stochastic LR constraint via one-sided Chebyshev inequality
- Emms & Haberman (2005): theoretical foundation for demand-linked insurance pricing
- Spedicato, Dutang & Petrini (2018): ML-then-optimise pipeline in practice


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion and retention modelling — demand curves from this library are the primary input to the optimiser |
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | Causal price elasticity — elasticity estimates define the demand response surface the optimiser maximises over |
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Model deployment — optimised rates flow into the champion/challenger deployment framework |

## Licence

MIT
