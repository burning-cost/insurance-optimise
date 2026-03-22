"""
insurance-optimise: constrained portfolio rate optimisation for UK personal lines.

Solves the insurance pricing optimisation problem:
- Maximise expected profit subject to FCA regulatory constraints
- Handles ENBP (PS21/11), loss ratio bounds, volume retention, rate change limits
- Analytical gradients for SLSQP — fast enough for N=10,000 policies
- Efficient frontier via epsilon-constraint sweep (bi-objective)
- 3-objective Pareto surface via 2D epsilon-constraint grid (ParetoFrontier)
- JSON audit trail for FCA regulatory evidence

The ``demand`` subpackage (``insurance_optimise.demand``) is the full demand
modelling suite absorbed from insurance-demand:
- ConversionModel: P(buy | price, features) for new business quotes
- RetentionModel: P(renew | features, price_change) for existing customers
- ElasticityEstimator: DML-based causal price elasticity from observational data
- DemandCurve: price → demand probability curves
- ENBPChecker: PS21/11 compliance diagnostic

The ``model_quality`` module (Hedges 2025, arXiv:2512.03242) provides:
- Closed-form relationship between Pearson correlation and expected LR
- Model quality report for pricing teams
- Integration into ConstraintConfig via model_quality_adjusted_lr

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

References
----------
- FCA PS21/11 (ENBP constraint): https://www.fca.org.uk/publication/policy/ps21-11.pdf
- FCA PS22/9 (Consumer Duty): fairness outcomes across customer segments
- Branda (2014): stochastic LR constraint via Chebyshev
- Emms & Haberman (2005): theoretical foundation for demand-linked pricing
- Hedges (2025): model quality and loss ratio; arXiv:2512.03242
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-optimise")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

from insurance_optimise.constraints import ConstraintConfig
from insurance_optimise._demand_model import LogLinearDemand, LogisticDemand, make_demand_model
from insurance_optimise.frontier import EfficientFrontier
from insurance_optimise.model_quality import ModelQualityReport, model_quality_report
from insurance_optimise.optimiser import PortfolioOptimiser
from insurance_optimise.pareto import (
    ParetoFrontier,
    ParetoResult,
    premium_disparity_ratio,
    loss_ratio_disparity,
)
from insurance_optimise.result import (
    EfficientFrontierResult,
    FrontierPoint,
    OptimisationResult,
    ScenarioResult,
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
    "premium_disparity_ratio",
    "loss_ratio_disparity",
    "LogLinearDemand",
    "LogisticDemand",
    "make_demand_model",
    "ScenarioObjective",
    "ClaimsVarianceModel",
    "ModelQualityReport",
    "model_quality_report",
    "demand",
    "__version__",
]
