"""
insurance-demand: Conversion, retention, and price elasticity modelling
for UK personal lines insurance.

The problem: UK personal lines insurers set prices using risk models (pure
premium = frequency × severity) but the market outcome depends on demand -
the probability that a customer accepts the quoted price. Risk models do not
tell you how to set commercial loadings. Demand models do.

The regulatory context: FCA PS21/11 (effective January 2022) banned renewal
price-walking - charging renewing customers more than new customers for
equivalent risk. Demand modelling is not banned; it is constrained. You can
use demand models for new business optimisation freely. For renewals, you
can use them to identify who merits a retention discount; you cannot use
inertia estimates to justify surcharging.

This library covers:
- ConversionModel: P(buy | price, features) for new business quotes
- RetentionModel: P(renew | features, price_change) for existing customers
- ElasticityEstimator: DML-based causal price elasticity from observational data
- DemandCurve: price → demand probability curves with multiple functional forms
- OptimalPrice: simple constrained price optimisation for a single segment
- ENBPChecker: PS21/11 compliance diagnostic

Usage
-----
    from insurance_demand import ConversionModel, RetentionModel
    from insurance_demand import ElasticityEstimator, DemandCurve, OptimalPrice
    from insurance_demand.compliance import ENBPChecker
    from insurance_demand.datasets import generate_conversion_data, generate_retention_data

The library follows sklearn's fit/predict pattern throughout. Models accept
Polars or pandas DataFrames; internal computation uses pandas at model
boundaries (sklearn compatibility) and Polars elsewhere.

Dependencies
------------
Core: scikit-learn, numpy, polars, pandas, scipy, statsmodels
Optional extras:
  catboost   -- CatBoost GBM backend for ConversionModel, RetentionModel,
                and as nuisance estimators in ElasticityEstimator
  dml        -- doubleml for causal elasticity estimation (DoubleMLPLR)
  causal     -- econml for heterogeneous treatment effects (CATE)
  survival   -- lifelines for Cox and Weibull retention models
  plot       -- matplotlib for DemandCurve.plot()
"""

from __future__ import annotations

from .conversion import ConversionModel
from .retention import RetentionModel
from .elasticity import ElasticityEstimator
from .demand_curve import DemandCurve
from .optimiser import OptimalPrice, OptimisationResult

__version__ = "0.1.0"

__all__ = [
    "ConversionModel",
    "RetentionModel",
    "ElasticityEstimator",
    "DemandCurve",
    "OptimalPrice",
    "OptimisationResult",
]
