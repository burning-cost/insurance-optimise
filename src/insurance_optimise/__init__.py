"""
insurance-optimise: constrained portfolio rate optimisation for UK personal lines.

Solves the insurance pricing optimisation problem:
- Maximise expected profit subject to FCA regulatory constraints
- Handles ENBP (PS21/11), loss ratio bounds, volume retention, rate change limits
- Analytical gradients for SLSQP — fast enough for N=10,000 policies
- Efficient frontier via epsilon-constraint sweep
- JSON audit trail for FCA regulatory evidence

Typical workflow
----------------
>>> import numpy as np
>>> from insurance_optimise import PortfolioOptimiser, ConstraintConfig
>>>
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

References
----------
- FCA PS21/11 (ENBP constraint): https://www.fca.org.uk/publication/policy/ps21-11.pdf
- Branda (2014): stochastic LR constraint via Chebyshev
- Emms & Haberman (2005): theoretical foundation for demand-linked pricing
"""

from insurance_optimise.constraints import ConstraintConfig
from insurance_optimise.demand import LogLinearDemand, LogisticDemand, make_demand_model
from insurance_optimise.frontier import EfficientFrontier
from insurance_optimise.optimiser import PortfolioOptimiser
from insurance_optimise.result import (
    EfficientFrontierResult,
    FrontierPoint,
    OptimisationResult,
    ScenarioResult,
)
from insurance_optimise.scenarios import ScenarioObjective

__version__ = "0.1.0"

__all__ = [
    "PortfolioOptimiser",
    "ConstraintConfig",
    "OptimisationResult",
    "ScenarioResult",
    "EfficientFrontier",
    "EfficientFrontierResult",
    "FrontierPoint",
    "LogLinearDemand",
    "LogisticDemand",
    "make_demand_model",
    "ScenarioObjective",
    "__version__",
]
