"""
Shared type definitions for insurance-demand.

Everything is kept minimal. We don't invent abstractions before we need them.
"""

from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd

# Model backend choices
BaseEstimatorType = Literal["logistic", "catboost"]
RetentionModelType = Literal["logistic", "catboost", "cox", "weibull"]

# Data that can be passed to predict methods: either a Polars or pandas DataFrame.
# We accept both; the bridge to pandas happens at model boundaries.
try:
    import polars as pl
    DataFrameLike = Union[pd.DataFrame, "pl.DataFrame"]
except ImportError:
    DataFrameLike = pd.DataFrame

ArrayLike = Union[np.ndarray, pd.Series]
