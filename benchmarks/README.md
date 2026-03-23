# insurance-optimise: Benchmark

## Headline result

**The constrained portfolio optimiser achieves the same GWP target as a uniform +7% rate change with £4,000–8,000 higher expected profit (~5–8%) and 2–4pp better retention on a 2,000-policy book — by applying larger increases to price-inelastic direct customers and smaller increases to elastic PCW customers.**

That is roughly £2–4 per policy per year. On a 50,000-policy renewal book, the optimiser advantage compounds: £100,000–200,000 per cycle before accounting for the retention benefit (retained customers continue generating profit; lost ones require replacement acquisition spend).

Run the benchmarks yourself:

```bash
uv run python benchmarks/benchmark.py         # optimiser vs uniform rate change
uv run python benchmarks/benchmark_dml.py     # DML elasticity vs naive logistic
uv run python benchmarks/benchmark_pareto.py  # Pareto frontier vs single-objective SLSQP
```

---

## Benchmark 1: Constrained optimiser vs uniform +7%

**Data:** 2,000 synthetic UK motor renewals. 55% PCW-acquired (mean elasticity −2.0), 45% direct (mean elasticity −1.2). Mean technical premium £545. Constraints: LR cap 68%, retention floor 78%, ±25% rate change limit, ENBP compliance (FCA PS21/5).

| Metric | Uniform +7% rate change | PortfolioOptimiser |
|---|---|---|
| Expected profit uplift | Baseline | +£4,000–8,000 (~5–8%) |
| Expected retention | ~74–76% | ~78–80% |
| Loss ratio | ~67–69% | ~65–67% |
| ENBP compliance | Partial (capped post-hoc) | Built into constraint set |
| Segment-aware pricing | No | Yes (by elasticity profile) |
| FCA PS21/5 audit trail | No | Yes (per-policy multiplier log) |
| Convergence time | N/A | <1s (2,000 policies) |

The uniform approach applies a flat multiplier to all policies. PCW customers who are highly price-sensitive get the same increase as direct customers who would accept more — the former leave, the latter subsidise the loss. The optimiser avoids this by redistributing the rate budget towards inelastic customers.

**Why the optimiser wins:**

The uniform +7% applies a blunt instrument. A PCW-acquired driver with elasticity −2.0 facing a +7% increase sees an expected renewal probability drop of roughly 14 percentage points. A direct driver with elasticity −1.2 facing the same increase drops only 8 points — but could accept a +10% increase with only a 12-point drop. The optimiser finds the per-policy multiplier that maximises expected profit. It charges direct customers more (they are less sensitive) and PCW customers less (retaining them is worth more than the extra margin).

The ENBP ceiling is binding for a small fraction of customers — typically 5–15% of the book — where the technical rerate would push the premium above the equivalent new business quote. The optimiser treats this as a hard per-policy upper bound.

---

## Benchmark 2: DML elasticity vs naive logistic regression

The optimiser depends on accurate price elasticity estimates. This benchmark compares naive logistic regression vs DML for recovering true elasticity from confounded quote data.

**Data:** 50,000 synthetic UK motor PCW quotes. True population-average price elasticity −2.0. Confounders: high-risk customers face higher prices via the underwriting model and have lower price sensitivity.

| Estimator | Estimate | Bias | Has valid CI |
|---|---|---|---|
| Naive logistic (no controls) | ~−3.43 | 1.43 (72%) | No |
| Logistic (full controls) | ~−1.21 | 0.79 (40%) | No |
| DML (ElasticityEstimator) | ~−4.03 | 2.03 (101%) | Yes |
| True elasticity | −2.0 | — | — |

**Honest interpretation:** On this DGP, DML did not outperform naive full-controls logistic on point accuracy. The DGP uses narrow quarterly loading cycles (std of log_price_ratio = 0.045), which provides too little exogenous price variation for the DML cross-fitting step. With std(log_price_ratio) ≥ 0.10 — genuine A/B tests, larger rate change cycles — DML converges closer to truth and provides a valid 95% CI that naive logistic cannot.

**The key point is still the pricing lift.** Even with a biased elasticity estimate, demand-curve-aware pricing outperforms flat loading by **+143.8% mean profit per segment**:

| Segment | Flat loading profit (£/quote) | DML-optimised (£/quote) | Lift |
|---------|------------------------------|------------------------|------|
| Young + High Risk | −31.79 | +14.39 | +145% |
| Young + Standard Risk | −22.01 | +9.64 | +144% |
| Mid-age + Standard Risk | −12.21 | +5.31 | +144% |
| Mid-age + Low Risk | −10.06 | +4.46 | +144% |
| Senior + Low Risk | −11.60 | +4.87 | +142% |

Negative flat-loading profit per quote reflects that a 10% loading is not enough to cover expected losses at market conversion rates. The optimiser finds the loss-minimising price given the demand curve.

Full script: `benchmarks/benchmark_dml.py`.

---

## Benchmark 3: Pareto surface — profit vs retention vs fairness

**Data:** 1,000 synthetic policies. Deprivation quintile (1–5) as the fairness dimension. Pareto sweep: 10 retention levels (82%–96%), 10 fairness caps (1.05–2.00). 100 SLSQP solves total.

| Optimisation | Profit (£) | Retention | Fairness disparity | Notes |
|---|---|---|---|---|
| Single-objective SLSQP (profit only) | 31,650 | 0.871 | 1.168 | Blind to fairness |
| Pareto — max-profit point | 31,650 | 0.871 | 1.168 | Same solution |
| Pareto — balanced (TOPSIS 0.5/0.3/0.2) | 28,940 | 0.912 | 1.043 | 9% profit cost |
| Pareto — min-disparity point | 22,180 | 0.951 | 1.011 | 30% profit cost |

**Consumer Duty framing:** The single-objective optimiser produces a disparity ratio of 1.168 — the most-deprived quintile pays 16.8% more on average. The Pareto surface makes this trade-off explicit: moving to the balanced point costs 9% of profit and cuts the disparity from 1.168 to 1.043. Whether that trade is acceptable is a governance decision, not a technical one. The value of the surface is that it forces the conversation to happen with concrete numbers.

Full script: `benchmarks/benchmark_pareto.py`.
