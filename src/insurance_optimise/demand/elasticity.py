"""
ElasticityEstimator: DML-based causal price elasticity estimation.

This is the core methodology that separates insurance-demand from a naive
logistic regression wrapper. The problem: in insurance observational data,
price is correlated with risk characteristics, which are also correlated with
conversion. Naive regression of conversion on price conflates the price effect
with the risk composition effect.

Double Machine Learning (Chernozhukov et al. 2018, Econometrics Journal)
fixes this by:
1. Regressing the outcome (log-conversion-rate, or logit of conversion)
   on all confounders X - call the residual Ỹ.
2. Regressing the treatment (log-price-ratio) on all confounders X - call
   the residual D̃.
3. Regressing Ỹ on D̃ to get θ = the causal price elasticity.

Cross-fitting (k-fold sample splitting) ensures the nuisance model estimates
don't overfit and bias θ. This is the KEY methodological step.

Two modes:
- ``heterogeneous=False``: Uses doubleml.DoubleMLPLR. Estimates a single
  global average treatment effect (ATE). Provides SE, CI, and sensitivity
  analysis. Best for audit/reporting.
- ``heterogeneous=True``: Uses econml.dml.LinearDML or CausalForestDML.
  Estimates a per-customer CATE (conditional ATE). Elasticity varies by
  segment. Better for targeting optimisation decisions.

Both modes use CatBoost as the nuisance estimator by default - it handles
categorical features natively, which matters for insurance data where many
key confounders (area, vehicle_group, channel) are categorical.

The treatment variable:
The standard treatment is log(quoted_price / technical_premium), i.e., the
log of the commercial loading. This is the "excess price" above the risk-
based rate. Variation in this ratio (driven by rate review cycle, not by
individual risk) is the quasi-exogenous treatment we want to identify. See
the research report for the full identification argument.

If technical_premium is not available, log(quoted_price) can be used as
treatment, but the identification assumption is weaker.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._types import DataFrameLike
from .conversion import _to_pandas, _is_categorical


class ElasticityEstimator:
    """
    DML-based causal price elasticity estimator.

    Wraps ``doubleml`` (for global ATE) or ``econml`` (for CATE / heterogeneous
    effects) with insurance-specific defaults.

    Parameters
    ----------
    outcome_col : str
        Column name for the outcome. For conversion: 'converted' (binary 0/1).
        DML works on the logit transform internally.
    treatment_col : str
        Column for the price treatment. Should be log(price/technical_premium).
        Default 'log_price_ratio'.
    feature_cols : sequence of str
        Confounder columns: age, vehicle_group, ncd_years, area, channel, etc.
        These are the X variables in the DML PLR model. Include everything
        that affects both price (via underwriting) and conversion.
    instrument_col : str, optional
        Column for an instrumental variable (e.g., competitor price shift or
        quarterly rate change index). If supplied, uses DoubleMLPLIV instead
        of PLR. This strengthens the identification assumption significantly.
    n_folds : int
        Number of cross-fitting folds. Default 5. Minimum 3.
    outcome_model : str or sklearn estimator
        Nuisance model for the outcome equation. 'catboost' or any sklearn
        regressor. Default 'catboost'.
    treatment_model : str or sklearn estimator
        Nuisance model for the treatment equation. 'catboost' or any sklearn
        regressor. Default 'catboost'.
    heterogeneous : bool
        If False (default): estimate a single global elasticity via doubleml PLR.
        If True: estimate per-customer CATE via econml LinearDML.
    catboost_params : dict, optional
        Override CatBoost nuisance model parameters.
    outcome_transform : str
        How to transform the binary outcome for DML.
        - 'logit': logit(clipped outcome mean in each fold). Matches the
          log-odds interpretation of elasticity.
        - 'identity': use the raw binary outcome (linear probability model).
          Simpler interpretation but less appropriate for binary Y.
        Default 'logit'.

    Notes
    -----
    Data requirements:
    - At minimum 20,000 observations for stable PLR estimates.
    - The treatment must vary within confounder groups - if all your pricing
      variation is between segments (not within), DML cannot identify the
      within-segment price effect.
    - Include time effects (month or quarter dummies) in feature_cols to
      remove seasonal confounding.

    Examples
    --------
    >>> from insurance_demand import ElasticityEstimator
    >>> from insurance_demand.datasets import generate_conversion_data
    >>> df = generate_conversion_data(n_quotes=100_000)
    >>> est = ElasticityEstimator(
    ...     feature_cols=['age', 'vehicle_group', 'ncd_years', 'area', 'channel'],
    ...     n_folds=5,
    ... )
    >>> est.fit(df)
    >>> print(est.summary())
    """

    def __init__(
        self,
        outcome_col: str = "converted",
        treatment_col: str = "log_price_ratio",
        feature_cols: Sequence[str] = (),
        instrument_col: Optional[str] = None,
        n_folds: int = 5,
        outcome_model: str = "catboost",
        treatment_model: str = "catboost",
        heterogeneous: bool = False,
        catboost_params: Optional[dict] = None,
        outcome_transform: str = "logit",
    ) -> None:
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.feature_cols = list(feature_cols)
        self.instrument_col = instrument_col
        self.n_folds = n_folds
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.heterogeneous = heterogeneous
        self.catboost_params = catboost_params
        self.outcome_transform = outcome_transform

        self._dml_model = None
        self._fitted = False
        self._elasticity_: Optional[float] = None
        self._elasticity_se_: Optional[float] = None
        self._elasticity_ci_: Optional[tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: DataFrameLike) -> "ElasticityEstimator":
        """
        Fit the DML elasticity estimator.

        Parameters
        ----------
        data : DataFrame
            Must contain outcome_col, treatment_col, and all feature_cols.

        Returns
        -------
        self
        """
        df = _to_pandas(data)

        if self.heterogeneous:
            self._fit_econml(df)
        else:
            self._fit_doubleml(df)

        self._fitted = True
        return self

    def _fit_doubleml(self, df: pd.DataFrame) -> None:
        """Fit DoubleMLPLR (or PLIV) for global elasticity estimate."""
        try:
            import doubleml as dml
        except ImportError:
            raise ImportError(
                "doubleml is required for ElasticityEstimator. "
                "Install with: uv pip install insurance-demand[dml]"
            )

        X_df, y, d, z = self._prepare_data(df)

        # DoubleML needs numpy arrays or a DoubleMLData object
        dml_data = dml.DoubleMLData(
            X_df.assign(**{self.outcome_col: y, self.treatment_col: d}),
            y_col=self.outcome_col,
            d_cols=self.treatment_col,
            x_cols=X_df.columns.tolist(),
            z_cols=self.instrument_col if z is not None else None,
        )

        ml_l = self._build_nuisance_model(self.outcome_model, task="regression")
        ml_m = self._build_nuisance_model(self.treatment_model, task="regression")

        if z is not None:
            # Instrumental variable: use PLIV
            ml_r = self._build_nuisance_model(self.treatment_model, task="regression")
            self._dml_model = dml.DoubleMLPLIV(
                dml_data, ml_l, ml_m, ml_r,
                n_folds=self.n_folds,
                score="partialling out",
            )
        else:
            self._dml_model = dml.DoubleMLPLR(
                dml_data, ml_l, ml_m,
                n_folds=self.n_folds,
                score="partialling out",
            )

        self._dml_model.fit()

        # Extract results
        coef = self._dml_model.coef[0]
        se = self._dml_model.se[0]
        self._elasticity_ = float(coef)
        self._elasticity_se_ = float(se)
        ci = self._dml_model.confint(level=0.95)
        self._elasticity_ci_ = (float(ci.iloc[0, 0]), float(ci.iloc[0, 1]))

    def _fit_econml(self, df: pd.DataFrame) -> None:
        """Fit LinearDML for heterogeneous (CATE) elasticity."""
        try:
            from econml.dml import LinearDML
        except ImportError:
            raise ImportError(
                "econml is required for heterogeneous=True. "
                "Install with: uv pip install insurance-demand[causal]"
            )

        X_df, y, d, z = self._prepare_data(df)

        ml_y = self._build_nuisance_model(self.outcome_model, task="regression")
        ml_t = self._build_nuisance_model(self.treatment_model, task="regression")

        # X_df for heterogeneity features; W=None means same as confounders
        self._dml_model = LinearDML(
            model_y=ml_y,
            model_t=ml_t,
            cv=self.n_folds,
            random_state=42,
        )
        self._dml_model.fit(Y=y, T=d, X=X_df.values, W=None)

        # Global ATE for summary
        ate = self._dml_model.ate(X=X_df.values)
        ate_inf = self._dml_model.ate_inference(X=X_df.values)
        self._elasticity_ = float(ate[0])
        self._elasticity_se_ = float(ate_inf.stderr_mean[0])
        ci = ate_inf.conf_int_mean()
        self._elasticity_ci_ = (float(ci[0][0]), float(ci[1][0]))
        self._X_train_ = X_df  # store for predict

    # ------------------------------------------------------------------
    # Predict / effect
    # ------------------------------------------------------------------

    def effect(self, data: DataFrameLike) -> pd.Series:
        """
        Per-customer CATE (heterogeneous mode only).

        Returns the estimated price elasticity for each individual observation.
        Only available when ``heterogeneous=True``.

        Returns
        -------
        pd.Series
            Per-customer elasticity estimates.
        """
        self._check_fitted()
        if not self.heterogeneous:
            raise RuntimeError(
                "effect() returns per-customer CATE, which requires "
                "heterogeneous=True. For global elasticity, use summary()."
            )
        df = _to_pandas(data)
        X_df, _, _, _ = self._prepare_data(df, include_outcome=False)
        effects = self._dml_model.effect(X=X_df.values)
        return pd.Series(effects.ravel(), index=df.index, name="cate_elasticity")

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Global elasticity estimate with confidence interval.

        Returns
        -------
        pd.DataFrame
            Columns: estimate, std_error, ci_lower, ci_upper, interpretation.
        """
        self._check_fitted()
        ci_lo, ci_hi = self._elasticity_ci_
        return pd.DataFrame({
            "parameter": ["price_elasticity"],
            "estimate": [self._elasticity_],
            "std_error": [self._elasticity_se_],
            "ci_lower_95": [ci_lo],
            "ci_upper_95": [ci_hi],
            "treatment": [self.treatment_col],
            "outcome": [self.outcome_col],
            "n_folds": [self.n_folds],
        })

    def sensitivity_analysis(self) -> Optional[pd.DataFrame]:
        """
        Sensitivity to unobserved confounding (doubleml only).

        Reports how large unobserved confounding would need to be to overturn
        the elasticity estimate. Uses the doubleml sensitivity framework.

        Returns None for econml mode (not yet implemented).
        """
        self._check_fitted()
        if self.heterogeneous:
            warnings.warn(
                "Sensitivity analysis is not available in heterogeneous (econml) mode.",
                stacklevel=2,
            )
            return None

        try:
            # DoubleML >= 0.7 has sensitivity_analysis()
            self._dml_model.sensitivity_analysis()
            return self._dml_model.sensitivity_summary
        except AttributeError:
            warnings.warn(
                "Sensitivity analysis requires doubleml >= 0.7.",
                stacklevel=2,
            )
            return None

    @property
    def elasticity_(self) -> float:
        """Global price elasticity point estimate."""
        self._check_fitted()
        return self._elasticity_

    @property
    def elasticity_se_(self) -> float:
        """Standard error of the elasticity estimate."""
        self._check_fitted()
        return self._elasticity_se_

    @property
    def elasticity_ci_(self) -> tuple[float, float]:
        """95% confidence interval for the elasticity."""
        self._check_fitted()
        return self._elasticity_ci_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        df: pd.DataFrame,
        include_outcome: bool = True,
    ) -> tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """Prepare X, y, d, z arrays for DML fitting."""

        # Treatment
        if self.treatment_col not in df.columns:
            raise ValueError(f"treatment_col '{self.treatment_col}' not found in data.")
        d = df[self.treatment_col].values.astype(float)

        # Outcome (with logit transform for binary Y)
        y = None
        if include_outcome:
            if self.outcome_col not in df.columns:
                raise ValueError(f"outcome_col '{self.outcome_col}' not found in data.")
            raw_y = df[self.outcome_col].values.astype(float)
            if self.outcome_transform == "logit":
                # Clip to avoid infinite logit values
                raw_y_clipped = np.clip(raw_y, 0.001, 0.999)
                y = np.log(raw_y_clipped / (1 - raw_y_clipped))
            else:
                y = raw_y

        # Instrument
        z = None
        if self.instrument_col and self.instrument_col in df.columns:
            z = df[self.instrument_col].values.astype(float)

        # Features (confounders)
        feat_cols = [c for c in self.feature_cols if c in df.columns]
        X_df = df[feat_cols].copy()

        # One-hot encode object columns (doubleml needs numeric)
        cat_cols = [c for c in X_df.columns if _is_categorical(X_df[c])]
        if cat_cols:
            X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=float)

        # Fill any NaN
        X_df = X_df.fillna(X_df.median(numeric_only=True))

        return X_df, y, d, z

    def _build_nuisance_model(self, spec, task: str = "regression"):
        """Build a nuisance model from a string spec or return as-is."""
        if spec == "catboost":
            return self._make_catboost(task)
        elif hasattr(spec, "fit"):
            # Already an sklearn estimator
            return spec
        else:
            raise ValueError(
                f"Unknown nuisance model spec: {spec!r}. "
                "Use 'catboost' or an sklearn-compatible estimator."
            )

    def _make_catboost(self, task: str):
        """Construct a CatBoost nuisance model."""
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
        except ImportError:
            # Fall back to sklearn GBM if CatBoost not available
            warnings.warn(
                "catboost not installed; falling back to sklearn GradientBoostingRegressor "
                "for DML nuisance models. Install catboost for better performance.",
                stacklevel=3,
            )
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
            )

        default_params = {
            "iterations": 300,
            "depth": 5,
            "learning_rate": 0.05,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }
        params = {**default_params, **(self.catboost_params or {})}

        # DML nuisance for binary Y uses regression (we've applied logit transform)
        return CatBoostRegressor(**params)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ElasticityEstimator is not fitted. Call .fit(data) first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"ElasticityEstimator("
            f"treatment='{self.treatment_col}', "
            f"outcome='{self.outcome_col}', "
            f"n_folds={self.n_folds}, "
            f"heterogeneous={self.heterogeneous}, "
            f"status={status})"
        )
