# Benchmarks — insurance-optimise

**Headline:** Constrained portfolio optimiser achieves the same GWP target as a uniform +7% rate change with ~4–8% higher expected profit and ~2–4pp better retention, by applying larger increases to price-inelastic direct customers and smaller increases to elastic PCW customers.

---

## Comparison table

2,000 synthetic UK motor renewals. 55% PCW-acquired (mean elasticity −2.0), 45% direct (mean elasticity −1.2). Constraints: LR cap 68%, retention floor 78%, ±25% rate change limit, ENBP compliance (FCA PS21/11).

| Metric | Uniform +7% rate change | PortfolioOptimiser |
|---|---|---|
| Expected profit uplift | Baseline | +£4,000–8,000 (~5–8%) |
| Expected retention | ~74–76% | ~78–80% |
| Loss ratio | ~67–69% | ~65–67% |
| ENBP compliance | Partial (capped post-hoc) | Built into constraint set |
| Segment-aware pricing | No | Yes (by elasticity profile) |
| FCA PS21/11 audit trail | No | Yes (per-policy multiplier log) |
| Convergence time | N/A | <1s (2,000 policies) |

The uniform approach applies a flat multiplier to all policies. PCW customers who are highly price-sensitive get the same increase as direct customers who would accept more — the former leave, the latter subsidise the loss. The optimiser avoids this by redistributing the rate budget towards inelastic customers.

### DML elasticity benchmark (`benchmark_dml.py`)

The optimiser depends on accurate price elasticity estimates. This benchmark compares naive logistic regression vs DML for recovering true elasticity from confounded quote data.

| Estimator | Elasticity estimate | Bias | Has CI |
|---|---|---|---|
| Naive logistic (no controls) | ~−2.4 to −2.7 | +0.4–0.7 vs true −2.0 | No |
| Logistic with partial controls | ~−2.2 to −2.5 | +0.2–0.5 | No |
| DML (ElasticityEstimator) | ~−1.95 to −2.05 | <0.1 | Yes |
| True elasticity | −2.0 | — | — |

Naive logistic conflates risk composition with price sensitivity (higher-risk quotes get higher prices and lower conversion for non-price reasons). DML partials out observable confounders before estimating the price-demand relationship. Bias reduction is typically 70–85%.

---

## How to run

```bash
uv run python benchmarks/benchmark.py         # optimiser vs uniform
uv run python benchmarks/benchmark_dml.py     # DML elasticity vs naive logistic
uv run python benchmarks/benchmark_pareto.py  # Pareto frontier vs single-objective SLSQP
```

### Databricks

```bash
databricks workspace import-dir benchmarks /Workspace/insurance-optimise/benchmarks
```

Dependencies: `insurance-optimise`, `numpy`, `polars`, `scipy`.
