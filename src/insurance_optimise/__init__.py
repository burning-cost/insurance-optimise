"""
insurance-optimise: constrained portfolio rate optimisation for UK personal lines.

Solves the insurance pricing optimisation problem:
- Maximise expected profit subject to FCA regulatory constraints
- Handles ENBP (PS21/11), loss ratio bounds, volume retention, rate change limits
- Analytical gradients for SLSQP — fast enough for N=10,000 policies
- Efficient frontier via epsilon-constraint sweep (bi-objective)
- 3-objective Pareto surface via 2D epsilon-constraint grid (ParetoFrontier)
- Bi-objective Pareto front visualiser for any two objectives (ParetoFront)
- JSON audit trail for FCA regulatory evidence

The ``demand`` subpackage (``insurance_optimise.demand``) is the full demand
modelling suite absorbed from insurance-demand:
- ConversionModel: P(buy | price, features) for new business quotes
- RetentionModel: P(renew | features, price_change) for existing customers
- ElasticityEstimator: DML-based causal price elasticity from observational data
- DemandCurve: price => demand probability curves
- ENBPChecker: PS21/11 compliance diagnostic

The ``model_quality`` module (Hedges 2025, arXiv:2512.03242) provides:
- Closed-form relationship between Pearson correlation and expected LR
- Model quality report for pricing teams
- Integration into ConstraintConfig via model_quality_adjusted_lr

The ``reinsurance`` module (Boonen, Dela Vega, Garces 2026, arXiv:2603.25350) provides:
- RobustReinsuranceOptimiser: multi-line proportional cession under model uncertainty
- Closed-form ODE shooting for symmetric lines (identical parameters)
- Numerical PDE value iteration for asymmetric multi-line cases
- Sensitivity analysis for ambiguity and reinsurance loading parameters

The ``risk_sharing`` module (Denuit, Flores-Contró, Robert 2026, arXiv:2603.29530) provides:
- LinearRiskSharingPool: n-participant community insurance pool with allocation matrix
- Mean-proportional allocation rule (auditable, FCA-friendly)
- Exact Cramér-Lundberg ruin probabilities (exponential severity)
- Event-driven Monte Carlo simulation of surplus paths
- SLSQP optimisation of the allocation matrix (min_max_ruin / max_min_improvement)
- JSON audit trail for regulatory documentation

The ``convex_reinsurance`` module (Shyamalkumar & Wang 2026, arXiv:2603.00813) provides:
- ConvexRiskReinsuranceOptimiser: optimal multi-line treaty design via convex duality
- Closed-form CVaR and variance solutions (Theorems 2 and 3)
- Bisection on Lagrange multiplier + fixed-point iteration for the variance case
- Efficient frontier: (ceded_premium, retained_risk) Pareto front
- Sensitivity analysis on budget, confidence level, and per-line loadings

Typical workflow
----------------
>>> import numpy as np
>>> from insurance_optimise import PortfolioOptimiser, ConstraintConfig

>>> config = ConstraintConfig(
...     lr_max=0.70,
...     retention_min=0.85,
...     max_rate_change=0.20,
...     enbp_buffer=0.01,
... )
>>> opt = PortfolioOptimiser(
...     technical_price=np.array([500.0, 750.0, 300.0]),
...     expected_loss_cost=np.array([350.0, 500.0, 200.0]),
...     p_demand=np.array([0.85, 0.90, 0.75]),
...     elasticity=np.array([-1.5, -2.0, -1.0]),
...     renewal_flag=np.array([True, True, False]),
...     enbp=np.array([600.0, 800.0, 0.0]),
...     constraints=config,
... )
>>> result = opt.optimise()
>>> print(result)
>>> print(result.profit)   # shorthand alias for result.expected_profit

Stochastic LR constraint
------------------------
To use the Branda (2014) chance-constrained LR, pass ``claims_variance`` to
the optimiser and set ``stochastic_lr=True`` in the config. Use
``ClaimsVarianceModel`` to build per-policy variance estimates from GLM outputs:

>>> from insurance_optimise import ClaimsVarianceModel
>>> var_model = ClaimsVarianceModel.from_tweedie(
...     mean_claims=expected_loss_cost,
...     dispersion=1.2,
...     power=1.5,
... )
>>> config = ConstraintConfig(lr_max=0.70, stochastic_lr=True, stochastic_alpha=0.90)
>>> opt = PortfolioOptimiser(..., claims_variance=var_model.variance_claims, constraints=config)

Model quality LR adjustment
---------------------------
To account for model imperfection (Hedges 2025), set ``model_quality_adjusted_lr``
in the config. The effective lr_max is relaxed by the expected loss ratio error
at the given Pearson correlation rho:

>>> from insurance_optimise import model_quality_report
>>> report = model_quality_report(rho=0.85, cv_lambda=1.2, eta=1.5, M=1/0.70)
>>> print(report)
>>> config = ConstraintConfig(
...     lr_max=0.70,
...     model_quality_adjusted_lr=True,
...     model_rho=0.85,
...     model_cv_lambda=1.2,
... )

3-objective Pareto surface
--------------------------
>>> from functools import partial
>>> from insurance_optimise.pareto import ParetoFrontier, premium_disparity_ratio
>>> fairness_fn = partial(
...     premium_disparity_ratio,
...     technical_price=tc,
...     group_labels=deprivation_quintile,
... )
>>> pf = ParetoFrontier(
...     optimiser=opt,
...     fairness_metric=fairness_fn,
...     sweep_x='volume_retention',
...     sweep_x_range=(0.80, 0.98),
...     sweep_y='fairness_max',
...     sweep_y_range=(1.05, 2.00),
...     n_points_x=10,
...     n_points_y=10,
... )
>>> result = pf.run()
>>> result = result.select(method='topsis', weights=(0.5, 0.3, 0.2))
>>> print(result.selected)

Bi-objective Pareto front visualiser
-------------------------------------
>>> from insurance_optimise import ParetoFront
>>> profits = np.array([10_000, 11_500, 12_000, 13_000, 11_000])
>>> disparity = np.array([1.10, 1.25, 1.40, 1.60, 1.20])
>>> pf = ParetoFront(
...     obj1=profits,
...     obj2=disparity,
...     maximize1=True,
...     maximize2=False,
...     obj1_name="Profit (GBP)",
...     obj2_name="Fairness Disparity Ratio",
... )
>>> ax = pf.plot()
>>> summary = pf.summary()
>>> print(summary)

Robust reinsurance optimisation
---------------------------------
>>> from insurance_optimise import RobustReinsuranceOptimiser, ReinsuranceLine
>>> line_mot = ReinsuranceLine(name="motor", mu=2.0, sigma=3.0, reins_loading=3.5, ambiguity=0.1)
>>> line_prop = ReinsuranceLine(name="property", mu=1.5, sigma=2.5, reins_loading=2.8, ambiguity=0.08)
>>> opt = RobustReinsuranceOptimiser(lines=[line_mot, line_prop])
>>> result = opt.optimise()
>>> print(result)
>>> sched = result.cession_schedule
>>> sens = opt.sensitivity(param='ambiguity', n_points=10)

Linear risk sharing
--------------------
>>> from insurance_optimise import LinearRiskSharingPool
>>> pool = LinearRiskSharingPool.mean_proportional(
...     claim_intensities=np.array([2.0, 1.0, 3.0]),
...     claim_means=np.array([2.0, 0.5, 1.0]),
...     safety_loadings=np.array([0.4, 0.4, 0.4]),
... )
>>> val = pool.validate_conditions()
>>> print(val)
>>> ruin = pool.ruin_comparison()
>>> print(ruin.improvement)  # positive = pool reduced ruin probability

Convex reinsurance optimisation (De Finetti problem)
------------------------------------------------------
>>> from insurance_optimise import ConvexRiskReinsuranceOptimiser, RiskLine
>>> risks = [
...     RiskLine(name="motor",    expected_loss=5_000, variance=8_000_000, safety_loading=0.15),
...     RiskLine(name="property", expected_loss=3_000, variance=4_000_000, safety_loading=0.22),
...     RiskLine(name="liability",expected_loss=1_500, variance=2_500_000, safety_loading=0.30),
... ]
>>> opt = ConvexRiskReinsuranceOptimiser(risks=risks, risk_measure='cvar', alpha=0.995, budget=12_000)
>>> result = opt.optimise()
>>> print(result)
>>> frontier = opt.frontier(n_points=30)
>>> sens = opt.sensitivity('budget', [10_000, 11_000, 12_000, 13_000, 14_000])

References
----------
- FCA PS21/11 (ENBP constraint): https://www.fca.org.uk/publication/policy/ps21-11.pdf
- FCA PS22/9 (Consumer Duty): fairness outcomes across customer segments
- Branda (2014): stochastic LR constraint via Chebyshev
- Emms & Haberman (2005): theoretical foundation for demand-linked pricing
- Hedges (2025): model quality and loss ratio; arXiv:2512.03242
- Boonen, Dela Vega, Garces (2026): robust dividend-reinsurance; arXiv:2603.25350
- Denuit, Flores-Contró, Robert (2026): linear risk sharing; arXiv:2603.29530
- Shyamalkumar & Wang (2026): convex reinsurance optimisation; arXiv:2603.00813
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-optimise")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

from insurance_optimise.constraints import ConstraintConfig
from insurance_optimise._demand_model import LogLinearDemand, LogisticDemand, make_demand_model
from insurance_optimise.convex_reinsurance import (
    ConvexReinsuranceResult,
    ConvexRiskReinsuranceOptimiser,
    RiskLine,
)
from insurance_optimise.frontier import EfficientFrontier
from insurance_optimise.model_quality import ModelQualityReport, model_quality_report
from insurance_optimise.optimiser import PortfolioOptimiser
from insurance_optimise.pareto import (
    ParetoFrontier,
    ParetoResult,
    premium_disparity_ratio,
    loss_ratio_disparity,
)
from insurance_optimise.pareto_front import ParetoFront, ParetoFrontSummary
from insurance_optimise.reinsurance import (
    ReinsuranceLine,
    RobustReinsuranceOptimiser,
    RobustReinsuranceResult,
)
from insurance_optimise.result import (
    EfficientFrontierResult,
    FrontierPoint,
    OptimisationResult,
    ScenarioResult,
)
from insurance_optimise.risk_sharing import (
    LinearRiskSharingPool,
    RuinResult,
    SimulationResult,
    ValidationResult,
    PerformanceWarning,
)
from insurance_optimise.scenarios import ScenarioObjective
from insurance_optimise.stochastic import ClaimsVarianceModel
from insurance_optimise import demand

__all__ = [
    "PortfolioOptimiser",
    "ConstraintConfig",
    "OptimisationResult",
    "ScenarioResult",
    "EfficientFrontier",
    "EfficientFrontierResult",
    "FrontierPoint",
    "ParetoFrontier",
    "ParetoResult",
    "ParetoFront",
    "ParetoFrontSummary",
    "premium_disparity_ratio",
    "loss_ratio_disparity",
    "LogLinearDemand",
    "LogisticDemand",
    "make_demand_model",
    "ScenarioObjective",
    "ClaimsVarianceModel",
    "ModelQualityReport",
    "model_quality_report",
    "ReinsuranceLine",
    "RobustReinsuranceOptimiser",
    "RobustReinsuranceResult",
    "LinearRiskSharingPool",
    "RuinResult",
    "SimulationResult",
    "ValidationResult",
    "PerformanceWarning",
    "ConvexRiskReinsuranceOptimiser",
    "ConvexReinsuranceResult",
    "RiskLine",
    "demand",
    "__version__",
]
