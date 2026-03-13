"""
RetentionModel: P(renew | features, price_change) for existing customers.

Two backends with meaningfully different trade-offs:

1. ``'logistic'``: Logistic GLM (or CatBoost) on a binary outcome (lapsed/renewed
   at next anniversary). Fast to fit, easy to interpret, standard in the industry.
   The price_change_col is the key treatment variable - but the coefficient will
   be biased if price changes are correlated with unobserved customer characteristics.

2. ``'cox'`` / ``'weibull'``: Survival model on time-to-lapse, handling mid-term
   censoring correctly. Useful when you want to model the full tenure distribution
   rather than just next-renewal lapse probability. Required for CLV calculations.

The choice of model should depend on your objective:
- Volume/retention forecast at next renewal → logistic is fine.
- CLV model integrating over multiple future renewals → survival model.
- Regulatory audit trail → logistic (easier to explain to the FCA).
- Complex non-linear interactions → CatBoost logistic.

Post-PS21/11 note:
This model tells you who will lapse at a given price. Under PS21/11, you cannot
use lapse propensity to SET a higher renewal price (that's the loyalty penalty).
You CAN use it to identify which customers benefit from a targeted retention
discount (a discount is permitted - you're moving below ENBP, not above it).
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ._types import DataFrameLike, RetentionModelType
from .conversion import _to_pandas


class RetentionModel:
    """
    Renewal/retention model: P(renew | features, price_change).

    Parameters
    ----------
    model_type : {'logistic', 'catboost', 'cox', 'weibull'}
        Model backend. Default 'logistic'.
        - 'logistic': sklearn LogisticRegression
        - 'catboost': CatBoost classifier (requires catboost extra)
        - 'cox': Cox proportional hazards via lifelines (requires survival extra)
        - 'weibull': Weibull AFT via lifelines (requires survival extra)
    outcome_col : str
        Column for binary lapse indicator (1 = lapsed, 0 = renewed).
        For survival models, this is the event indicator.
    duration_col : str, optional
        Column for tenure/time-at-risk (years). Required for survival models.
        For logistic models, tenure can be included as a feature.
    price_change_col : str
        Column for log(renewal_price / prior_year_price). This is the price
        treatment variable. Default 'log_price_change'.
    feature_cols : sequence of str
        Additional feature columns: tenure_years, ncd_years, payment_method,
        claim_last_3yr, channel, area, etc.
    cat_features : sequence of str
        Categorical feature columns (for CatBoost backend).
    catboost_params : dict, optional
        Override CatBoost parameters.
    logistic_C : float
        Regularisation for logistic. Default 1.0.

    Examples
    --------
    >>> from insurance_demand import RetentionModel
    >>> from insurance_demand.datasets import generate_retention_data
    >>> df = generate_retention_data(n_policies=50_000)
    >>> model = RetentionModel(
    ...     model_type='logistic',
    ...     feature_cols=['tenure_years', 'ncd_years', 'payment_method', 'claim_last_3yr'],
    ... )
    >>> model.fit(df)
    >>> lapse_probs = model.predict_proba(df)
    >>> print(f"Portfolio lapse rate: {lapse_probs.mean():.1%}")
    """

    def __init__(
        self,
        model_type: RetentionModelType = "logistic",
        outcome_col: str = "lapsed",
        duration_col: Optional[str] = "tenure_years",
        price_change_col: str = "log_price_change",
        feature_cols: Sequence[str] = (),
        cat_features: Sequence[str] = (),
        catboost_params: Optional[dict] = None,
        logistic_C: float = 1.0,
    ) -> None:
        self.model_type = model_type
        self.outcome_col = outcome_col
        self.duration_col = duration_col
        self.price_change_col = price_change_col
        self.feature_cols = list(feature_cols)
        self.cat_features = list(cat_features)
        self.catboost_params = catboost_params
        self.logistic_C = logistic_C

        self._model = None
        self._fitted = False
        self._feature_names_in: list[str] = []
        self._is_survival = model_type in ("cox", "weibull")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: DataFrameLike) -> "RetentionModel":
        """
        Fit the retention model.

        Parameters
        ----------
        data : DataFrame
            Policy-level renewal data. Must contain outcome_col, price_change_col,
            and all feature_cols. For survival models, must also contain duration_col.

        Returns
        -------
        self
        """
        df = _to_pandas(data)

        if self._is_survival:
            self._fit_survival(df)
        else:
            X, y = self._build_features(df, training=True)
            if self.model_type == "logistic":
                self._fit_logistic(X, y)
            elif self.model_type == "catboost":
                self._fit_catboost(X, y)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type!r}")

        self._fitted = True
        return self

    def _fit_logistic(self, X: pd.DataFrame, y: np.ndarray) -> None:
        cat_cols = [c for c in X.columns if _is_categorical(X[c])]
        self._logistic_cat_cols = cat_cols  # store for predict-time encoding
        X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
        self._encoded_columns = X_enc.columns.tolist()

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(
                C=self.logistic_C,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        pipeline.fit(X_enc, y)
        self._model = pipeline
        self._feature_names_in = self._encoded_columns

    def _fit_catboost(self, X: pd.DataFrame, y: np.ndarray) -> None:
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError(
                "catboost is required for model_type='catboost'. "
                "Install with: uv pip install insurance-demand[catboost]"
            )
        default_params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }
        params = {**default_params, **(self.catboost_params or {})}
        cat_cols = [c for c in X.columns if _is_categorical(X[c])]
        cat_indices = [X.columns.tolist().index(c) for c in cat_cols]
        self._model = CatBoostClassifier(**params)
        self._model.fit(X, y, cat_features=cat_indices)
        self._feature_names_in = X.columns.tolist()

    def _fit_survival(self, df: pd.DataFrame) -> None:
        """Fit Cox or Weibull survival model using lifelines."""
        try:
            if self.model_type == "cox":
                from lifelines import CoxPHFitter
            else:
                from lifelines import WeibullAFTFitter
        except ImportError:
            raise ImportError(
                f"lifelines is required for model_type='{self.model_type}'. "
                "Install with: uv pip install insurance-demand[survival]"
            )

        if self.duration_col is None:
            raise ValueError(
                "duration_col is required for survival models. "
                "Supply the column containing time-at-risk (e.g. tenure_years)."
            )

        # Build feature set for survival model
        all_cols = [self.duration_col, self.outcome_col, self.price_change_col] + self.feature_cols
        available = [c for c in all_cols if c in df.columns]
        df_surv = df[available].copy()

        # One-hot encode categoricals (lifelines expects numeric)
        cat_cols = [c for c in df_surv.columns
                    if _is_categorical(df_surv[c])
                    and c not in (self.duration_col, self.outcome_col)]
        if cat_cols:
            df_surv = pd.get_dummies(df_surv, columns=cat_cols, drop_first=True, dtype=float)

        if self.model_type == "cox":
            self._model = CoxPHFitter(penalizer=0.1)
            self._model.fit(
                df_surv,
                duration_col=self.duration_col,
                event_col=self.outcome_col,
            )
        else:
            self._model = WeibullAFTFitter(penalizer=0.1)
            self._model.fit(
                df_surv,
                duration_col=self.duration_col,
                event_col=self.outcome_col,
            )

        self._survival_feature_cols = [
            c for c in df_surv.columns
            if c not in (self.duration_col, self.outcome_col)
        ]
        self._feature_names_in = self._survival_feature_cols

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, data: DataFrameLike) -> pd.Series:
        """
        Predict lapse probability at current renewal prices.

        For logistic/CatBoost: returns P(lapsed=1).
        For survival models: returns P(event before median survival time),
        i.e., the probability of lapsing within the next policy year.

        Returns
        -------
        pd.Series
            Lapse probability for each policy.
        """
        self._check_fitted()
        df = _to_pandas(data)

        if self._is_survival:
            return self._predict_survival_proba(df)

        X, _ = self._build_features(df, training=False)

        if self.model_type == "logistic":
            X_enc = _encode_categoricals(
                X, self._logistic_cat_cols, self._feature_names_in
            )
            probs = self._model.predict_proba(X_enc)[:, 1]
        else:
            probs = self._model.predict_proba(X)[:, 1]

        return pd.Series(probs, index=df.index, name="lapse_prob")

    def predict_renewal_proba(self, data: DataFrameLike) -> pd.Series:
        """
        Predict renewal (retention) probability. Convenience: 1 - lapse_prob.

        Returns
        -------
        pd.Series
            Renewal probability in [0, 1].
        """
        return (1 - self.predict_proba(data)).rename("renewal_prob")

    def _predict_survival_proba(self, df: pd.DataFrame) -> pd.Series:
        """Predict 1-year lapse probability from a fitted survival model."""
        df_feats = df[self._survival_feature_cols].copy() if self._survival_feature_cols else pd.DataFrame(index=df.index)
        # Re-encode categoricals
        cat_cols = [c for c in df_feats.columns if _is_categorical(df_feats[c])]
        if cat_cols:
            df_feats = pd.get_dummies(df_feats, columns=cat_cols, drop_first=True, dtype=float)
        df_feats = df_feats.reindex(columns=self._survival_feature_cols, fill_value=0)

        # S(1) = survival probability at 1 year
        surv_at_1yr = self._model.predict_survival_function(df_feats, times=[1.0])
        # surv_at_1yr has shape (1, n): first row is S(1)
        s1 = surv_at_1yr.iloc[0].values
        lapse_prob = 1.0 - s1
        return pd.Series(lapse_prob, index=df.index, name="lapse_prob")

    def predict_survival(
        self, data: DataFrameLike, times: Sequence[float] = (1, 2, 3, 5)
    ) -> pd.DataFrame:
        """
        Predict survival curve at specified time points (survival models only).

        Returns
        -------
        pd.DataFrame
            Shape (n_policies, len(times)). Each column is S(t) for time t.
            Values represent P(still a customer at time t).
        """
        self._check_fitted()
        if not self._is_survival:
            raise RuntimeError(
                "predict_survival is only available for survival model types "
                "(model_type='cox' or 'weibull')."
            )
        df = _to_pandas(data)
        df_feats = df[self._survival_feature_cols].copy()
        cat_cols = [c for c in df_feats.columns if _is_categorical(df_feats[c])]
        if cat_cols:
            df_feats = pd.get_dummies(df_feats, columns=cat_cols, drop_first=True, dtype=float)
        df_feats = df_feats.reindex(columns=self._survival_feature_cols, fill_value=0)

        surv = self._model.predict_survival_function(df_feats, times=list(times))
        # Transpose: from (times, n) to (n, times)
        result = surv.T.copy()
        result.columns = [f"S_t{t}" for t in times]
        result.index = df.index
        return result

    # ------------------------------------------------------------------
    # Price sensitivity
    # ------------------------------------------------------------------

    def price_sensitivity(
        self,
        data: DataFrameLike,
        price_change_delta: float = 0.01,
    ) -> pd.Series:
        """
        Marginal effect of price change on lapse probability.

        dP(lapse) / d(log_price_change), evaluated numerically at the
        current log_price_change values.

        Parameters
        ----------
        data : DataFrame
        price_change_delta : float
            Step size for central finite differences. Default 0.01.

        Returns
        -------
        pd.Series
            Marginal effect. Positive values mean higher price increase
            → higher lapse probability (as expected).
        """
        self._check_fitted()
        if self._is_survival:
            raise NotImplementedError(
                "price_sensitivity is not yet implemented for survival models. "
                "Use predict_proba at different price levels instead."
            )
        df = _to_pandas(data)
        delta = price_change_delta

        df_up = df.copy()
        df_up[self.price_change_col] = df[self.price_change_col] + delta

        df_dn = df.copy()
        df_dn[self.price_change_col] = df[self.price_change_col] - delta

        p_up = self.predict_proba(df_up).values
        p_dn = self.predict_proba(df_dn).values
        me = (p_up - p_dn) / (2 * delta)

        return pd.Series(me, index=df.index, name="price_sensitivity")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def oneway(
        self,
        data: DataFrameLike,
        factor: str,
        bins: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Observed vs. fitted lapse rate by factor level.

        Parameters
        ----------
        data : DataFrame
        factor : str
            Column to group by.
        bins : int, optional
            Bin a numeric factor into quantile groups.

        Returns
        -------
        pd.DataFrame
            factor_level, n_policies, observed_lapse_rate, fitted_lapse_rate, lift
        """
        self._check_fitted()
        df = _to_pandas(data)
        probs = self.predict_proba(data).values
        observed = df[self.outcome_col].values

        if bins is not None and pd.api.types.is_numeric_dtype(df[factor]):
            group_col = pd.qcut(df[factor], q=bins, duplicates="drop").astype(str)
        else:
            group_col = df[factor].astype(str)

        result = pd.DataFrame({
            "factor_level": group_col,
            "observed": observed,
            "fitted": probs,
        })

        summary = (
            result.groupby("factor_level")
            .agg(
                n_policies=("observed", "count"),
                observed_lapse_rate=("observed", "mean"),
                fitted_lapse_rate=("fitted", "mean"),
            )
            .reset_index()
        )
        summary["lift"] = summary["observed_lapse_rate"] / summary["fitted_lapse_rate"].clip(lower=1e-8)
        return summary.sort_values("factor_level")

    def summary(self) -> pd.DataFrame:
        """
        Model summary.

        For logistic: coefficient table with odds ratios.
        For CatBoost: feature importances.
        For survival: coefficient table from lifelines.

        Returns
        -------
        pd.DataFrame
        """
        self._check_fitted()
        if self._is_survival:
            return self._model.summary

        if self.model_type == "logistic":
            logit = self._model.named_steps["logit"]
            coefs = logit.coef_[0]
            return pd.DataFrame({
                "feature": self._feature_names_in,
                "coefficient": coefs,
                "odds_ratio": np.exp(coefs),
            }).sort_values("coefficient", key=abs, ascending=False)
        else:
            importances = self._model.get_feature_importance()
            return pd.DataFrame({
                "feature": self._feature_names_in,
                "importance": importances,
            }).sort_values("importance", ascending=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(
        self, df: pd.DataFrame, training: bool
    ) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
        cols = {}

        # Price change treatment
        if self.price_change_col in df.columns:
            cols[self.price_change_col] = df[self.price_change_col]
        else:
            raise ValueError(
                f"price_change_col '{self.price_change_col}' not found in data."
            )

        # Other features
        for col in self.feature_cols:
            if col in df.columns:
                cols[col] = df[col]

        X = pd.DataFrame(cols, index=df.index)
        y = None
        if training:
            if self.outcome_col not in df.columns:
                raise ValueError(f"outcome_col '{self.outcome_col}' not in data.")
            y = df[self.outcome_col].values.astype(int)

        return X, y

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit(data) first.")


# Re-export the shared encoding helper (same logic as in conversion.py)
from .conversion import _encode_categoricals, _is_categorical
