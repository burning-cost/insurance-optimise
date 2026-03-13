"""
ConversionModel: P(buy | price, features) for new business quotes.

This is the "static" demand model in the Akur8/commercial-platform sense.
It tells you the expected conversion rate at the current quoted price. It
does NOT give you unbiased price elasticity - for that, use ElasticityEstimator.

What it does:
- Fits a logistic GLM or CatBoost classifier on quote-level data
- Predicts P(converted | price, features) at any new price
- Computes naive marginal effect of price: dP/dP evaluated at current price
- Produces one-way observed-vs-fitted plots by factor level
- Exports as a DemandModel-compatible callable for rate-optimiser integration

The price treatment:
The model uses log(quoted_price / technical_premium) as the price input, not
the raw quoted price. This follows industry practice (Guven & McPhail 2013):
it centres the price effect around the risk-adjusted price rather than the
absolute pound value, so the coefficient is comparable across risk segments.

If you don't have technical_premium, you can pass price_transform='log_price'
to use log(quoted_price) directly. The interpretation changes but the model
is still valid.

PCW rank position:
On PCW, being ranked 1st (cheapest) has a discrete demand effect that is not
captured by the price_ratio alone. Include rank_position_col to model this.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ._types import BaseEstimatorType, DataFrameLike


class ConversionModel:
    """
    Static conversion model: P(buy | price, features).

    Fits on quote-level data and predicts conversion probability at any price.
    Two backends: ``'logistic'`` (GLM via sklearn LogisticRegression) and
    ``'catboost'`` (CatBoost classifier, requires catboost extra).

    The logistic backend is interpretable and produces analytical marginal effects.
    CatBoost captures non-linear interactions between price, channel, and risk
    features - typically a better fit on real data.

    Parameters
    ----------
    base_estimator : {'logistic', 'catboost'}
        Model backend. Default 'logistic'.
    outcome_col : str
        Column name for the binary conversion indicator (1 = converted).
    quoted_price_col : str
        Column name for the quoted price.
    technical_premium_col : str, optional
        Column name for the risk model technical premium. If supplied, the
        price treatment is log(quoted_price / technical_premium). If None,
        the treatment is log(quoted_price).
    feature_cols : sequence of str
        Additional feature columns (age, vehicle_group, ncd_years, channel,
        area, etc.). These are the confounders for elasticity estimation.
    rank_position_col : str, optional
        Column for PCW rank position (1 = cheapest). Recommended for PCW data.
        Included as a log-transformed feature.
    price_to_market_col : str, optional
        Column for price / cheapest_competitor ratio. Complements rank_position
        by capturing the magnitude of the price gap, not just the rank.
    catboost_params : dict, optional
        Override CatBoost parameters. Defaults use 500 iterations, depth 6,
        learning rate 0.05.
    logistic_C : float
        Regularisation strength for LogisticRegression (inverse of lambda).
        Default 1.0 (mild regularisation).
    cat_features : sequence of str, optional
        Column names that are categorical. For CatBoost, passed as
        ``cat_features`` parameter. For logistic, these are one-hot encoded.

    Examples
    --------
    >>> from insurance_demand import ConversionModel
    >>> from insurance_demand.datasets import generate_conversion_data
    >>> df = generate_conversion_data(n_quotes=50_000)
    >>> model = ConversionModel(
    ...     base_estimator='logistic',
    ...     feature_cols=['age', 'vehicle_group', 'ncd_years', 'area', 'channel'],
    ...     rank_position_col='rank_position',
    ... )
    >>> model.fit(df)
    >>> probs = model.predict_proba(df)
    >>> print(probs.mean())   # overall conversion rate
    """

    def __init__(
        self,
        base_estimator: BaseEstimatorType = "logistic",
        outcome_col: str = "converted",
        quoted_price_col: str = "quoted_price",
        technical_premium_col: Optional[str] = "technical_premium",
        feature_cols: Sequence[str] = (),
        rank_position_col: Optional[str] = None,
        price_to_market_col: Optional[str] = None,
        catboost_params: Optional[dict] = None,
        logistic_C: float = 1.0,
        cat_features: Sequence[str] = (),
    ) -> None:
        self.base_estimator = base_estimator
        self.outcome_col = outcome_col
        self.quoted_price_col = quoted_price_col
        self.technical_premium_col = technical_premium_col
        self.feature_cols = list(feature_cols)
        self.rank_position_col = rank_position_col
        self.price_to_market_col = price_to_market_col
        self.catboost_params = catboost_params
        self.logistic_C = logistic_C
        self.cat_features = list(cat_features)

        self._model = None
        self._fitted = False
        self._feature_names_in: list[str] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: DataFrameLike) -> "ConversionModel":
        """
        Fit the conversion model.

        Parameters
        ----------
        data : DataFrame (Polars or pandas)
            Must contain ``outcome_col``, ``quoted_price_col``, and all
            columns in ``feature_cols``.

        Returns
        -------
        self
        """
        df = _to_pandas(data)
        X, y = self._build_features(df, training=True)

        if self.base_estimator == "logistic":
            self._fit_logistic(X, y)
        elif self.base_estimator == "catboost":
            self._fit_catboost(X, y, df)
        else:
            raise ValueError(f"Unknown base_estimator: {self.base_estimator!r}")

        self._fitted = True
        return self

    def _fit_logistic(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit sklearn LogisticRegression with standard scaling."""
        # One-hot encode categoricals for logistic
        cat_mask = [c for c in X.columns if _is_categorical(X[c])]
        self._logistic_cat_cols = cat_mask  # store for predict-time encoding
        X_encoded = pd.get_dummies(X, columns=cat_mask, drop_first=True, dtype=float)
        self._encoded_columns = X_encoded.columns.tolist()

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(
                C=self.logistic_C,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        pipeline.fit(X_encoded, y)
        self._model = pipeline
        self._feature_names_in = self._encoded_columns

    def _fit_catboost(self, X: pd.DataFrame, y: np.ndarray, df_orig: pd.DataFrame) -> None:
        """Fit CatBoost classifier."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError(
                "catboost is required for base_estimator='catboost'. "
                "Install it with: uv pip install insurance-demand[catboost]"
            )

        default_params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }
        params = {**default_params, **(self.catboost_params or {})}

        # Identify categorical feature indices
        cat_cols = [c for c in X.columns if _is_categorical(X[c])]
        cat_indices = [X.columns.tolist().index(c) for c in cat_cols]

        self._model = CatBoostClassifier(**params)
        self._model.fit(X, y, cat_features=cat_indices)
        self._feature_names_in = X.columns.tolist()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, data: DataFrameLike) -> pd.Series:
        """
        Predict conversion probability at current quoted prices.

        Parameters
        ----------
        data : DataFrame
            Must contain the same columns as the training data.

        Returns
        -------
        pd.Series
            P(converted=1) for each row.
        """
        self._check_fitted()
        df = _to_pandas(data)
        X, _ = self._build_features(df, training=False)
        return pd.Series(self._raw_predict_proba(X), index=df.index, name="conversion_prob")

    def _raw_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Run model prediction; handles logistic vs catboost differences."""
        if self.base_estimator == "logistic":
            X_enc = pd.get_dummies(X, drop_first=True, dtype=float)
            X_enc = X_enc.reindex(columns=self._feature_names_in, fill_value=0)
            return self._model.predict_proba(X_enc)[:, 1]
        else:
            return self._model.predict_proba(X)[:, 1]

    def predict(self, data: DataFrameLike, threshold: float = 0.5) -> pd.Series:
        """Binary conversion prediction at given probability threshold."""
        probs = self.predict_proba(data)
        return (probs >= threshold).astype(int).rename("converted_pred")

    # ------------------------------------------------------------------
    # Elasticity / marginal effects
    # ------------------------------------------------------------------

    def marginal_effect(
        self,
        data: DataFrameLike,
        price_delta_pct: float = 1.0,
    ) -> pd.Series:
        """
        Numeric marginal effect of price on conversion probability.

        Computes (P(buy | price × 1.01) - P(buy | price × 0.99)) / (0.02 × price)
        i.e., dP/dPrice evaluated via central finite differences.

        This is the naive marginal effect - it includes confounding from risk
        features. For the debiased price elasticity, use ElasticityEstimator.

        Parameters
        ----------
        data : DataFrame
        price_delta_pct : float
            Step size as percentage. Default 1.0 means ±0.5% perturbation.

        Returns
        -------
        pd.Series
            dP/dPrice for each row (negative: higher price → lower conversion).
        """
        self._check_fitted()
        df = _to_pandas(data)
        delta = price_delta_pct / 100.0 / 2

        df_up = df.copy()
        df_up[self.quoted_price_col] = df[self.quoted_price_col] * (1 + delta)

        df_dn = df.copy()
        df_dn[self.quoted_price_col] = df[self.quoted_price_col] * (1 - delta)

        p_up = self.predict_proba(df_up).values
        p_dn = self.predict_proba(df_dn).values
        price = df[self.quoted_price_col].values
        me = (p_up - p_dn) / (2 * delta * price)

        return pd.Series(me, index=df.index, name="marginal_effect")

    def price_elasticity(self, data: DataFrameLike, price_delta_pct: float = 1.0) -> pd.Series:
        """
        Naive log-log price elasticity: d log(P(buy)) / d log(price).

        Uses numeric differentiation. Note: this is NOT the causal elasticity.
        It is the elasticity implied by the fitted model, which may be biased
        if the model does not fully control for confounders. Use
        ElasticityEstimator for causal estimates.

        Returns
        -------
        pd.Series
            Elasticity values (negative for normal goods).
        """
        self._check_fitted()
        df = _to_pandas(data)
        delta = price_delta_pct / 100.0 / 2

        df_up = df.copy()
        df_up[self.quoted_price_col] = df[self.quoted_price_col] * (1 + delta)

        df_dn = df.copy()
        df_dn[self.quoted_price_col] = df[self.quoted_price_col] * (1 - delta)

        p_up = self.predict_proba(df_up).values
        p_dn = self.predict_proba(df_dn).values
        p_base = self.predict_proba(data).values
        p_base_safe = np.where(p_base > 1e-8, p_base, 1e-8)

        # d log(p) / d log(price) = (p_up - p_dn) / (2*delta) / p_base
        elasticity = (p_up - p_dn) / (2 * delta) / p_base_safe

        return pd.Series(elasticity, index=df.index, name="price_elasticity")

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
        Observed vs. fitted conversion rate by factor level.

        A standard pricing diagnostic: does the model track the observed rate
        by each rating factor level? Large gaps indicate missing features or
        model misspecification.

        Parameters
        ----------
        data : DataFrame
        factor : str
            Column name to group by.
        bins : int, optional
            If the factor is numeric, bin into this many quantile groups.

        Returns
        -------
        pd.DataFrame
            Columns: factor level, n_quotes, observed_rate, fitted_rate, lift.
        """
        self._check_fitted()
        df = _to_pandas(data)
        probs = self.predict_proba(data).values
        observed = df[self.outcome_col].values

        if bins is not None and pd.api.types.is_numeric_dtype(df[factor]):
            labels = pd.qcut(df[factor], q=bins, duplicates="drop")
            group_col = labels.astype(str)
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
                n_quotes=("observed", "count"),
                observed_rate=("observed", "mean"),
                fitted_rate=("fitted", "mean"),
            )
            .reset_index()
        )
        summary["lift"] = summary["observed_rate"] / summary["fitted_rate"].clip(lower=1e-8)
        return summary.sort_values("factor_level")

    def summary(self) -> pd.DataFrame:
        """
        Model summary table.

        For logistic: returns coefficient table with odds ratios.
        For CatBoost: returns feature importances.

        Returns
        -------
        pd.DataFrame
        """
        self._check_fitted()
        if self.base_estimator == "logistic":
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
    # rate-optimiser integration
    # ------------------------------------------------------------------

    def as_demand_callable(self) -> "callable":
        """
        Export as a callable compatible with rate-optimiser's DemandModel.

        Returns a function with signature:
        ``f(price_ratio: np.ndarray, features: pl.DataFrame) -> np.ndarray``

        The caller is responsible for constructing an appropriate features
        DataFrame with the same columns used during training (excluding
        price-derived features which are recomputed from price_ratio).

        Returns
        -------
        callable
        """
        self._check_fitted()
        tech_col = self.technical_premium_col

        def _demand_fn(price_ratio: np.ndarray, features: pl.DataFrame) -> np.ndarray:
            df = features.to_pandas()
            if tech_col and tech_col in df.columns:
                df[self.quoted_price_col] = price_ratio * df[tech_col]
            else:
                # Treat price_ratio as the quoted price directly
                df[self.quoted_price_col] = price_ratio
            probs = self.predict_proba(df)
            return probs.values

        return _demand_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(
        self, df: pd.DataFrame, training: bool
    ) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Construct the feature matrix X and outcome y from a DataFrame."""
        # Price treatment
        if self.technical_premium_col and self.technical_premium_col in df.columns:
            price_ratio = df[self.quoted_price_col] / df[self.technical_premium_col].clip(lower=1.0)
            log_price = np.log(price_ratio.clip(lower=0.1))
            feature_col_name = "log_price_ratio"
        else:
            log_price = np.log(df[self.quoted_price_col].clip(lower=1.0))
            feature_col_name = "log_price"

        cols = {feature_col_name: log_price}

        # Additional features
        for col in self.feature_cols:
            if col in df.columns:
                cols[col] = df[col]

        # Rank position (log-transformed: from rank 1 to 6 the effect diminishes)
        if self.rank_position_col and self.rank_position_col in df.columns:
            cols["log_rank"] = np.log(df[self.rank_position_col].clip(lower=1))

        # Price to market ratio (log)
        if self.price_to_market_col and self.price_to_market_col in df.columns:
            cols["log_price_to_market"] = np.log(df[self.price_to_market_col].clip(lower=0.5))

        X = pd.DataFrame(cols, index=df.index)

        y = None
        if training:
            if self.outcome_col not in df.columns:
                raise ValueError(
                    f"outcome_col '{self.outcome_col}' not found in data. "
                    f"Available columns: {df.columns.tolist()}"
                )
            y = df[self.outcome_col].values.astype(int)

        return X, y

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Model is not fitted. Call .fit(data) before predicting."
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _is_categorical(s: pd.Series) -> bool:
    """Check if a pandas Series contains categorical/string data.

    Handles object, category, StringDtype, and ArrowDtype — all of which
    can appear when converting Polars DataFrames to pandas depending on
    the Polars + pyarrow version combination.
    """
    return (
        s.dtype == object
        or s.dtype.name == "category"
        or pd.api.types.is_string_dtype(s)
    )


def _to_pandas(data: DataFrameLike) -> pd.DataFrame:
    """Convert Polars DataFrame to pandas if necessary."""
    if isinstance(data, pd.DataFrame):
        return data
    try:
        import polars as pl
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
    except ImportError:
        pass
    raise TypeError(f"Expected pandas or Polars DataFrame, got {type(data)}")


def _encode_categoricals(
    X: pd.DataFrame,
    cat_columns: list[str],
    training_dummies: list[str],
) -> pd.DataFrame:
    """
    One-hot encode categorical columns using the training-time column list.

    Unlike pd.get_dummies(drop_first=True), this function correctly handles
    test data where only a single category is present (which causes get_dummies
    to produce 0 columns, breaking the downstream reindex).

    Encodes using the same reference category (alphabetically first) that
    drop_first=True would have used at training time, but explicitly constructs
    each indicator column from the training_dummies list.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (may contain categorical columns).
    cat_columns : list of str
        Categorical columns to encode.
    training_dummies : list of str
        Expected output column names (from pd.get_dummies at training time).

    Returns
    -------
    pd.DataFrame
        Fully encoded DataFrame with columns matching training_dummies.
    """
    X_out = X.drop(columns=[c for c in cat_columns if c in X.columns]).copy()

    for col in cat_columns:
        if col not in X.columns:
            continue
        # Add indicator columns for each dummy in training_dummies that
        # corresponds to this column
        prefix = col + "_"
        for dummy_col in training_dummies:
            if dummy_col.startswith(prefix):
                category = dummy_col[len(prefix):]
                X_out[dummy_col] = (X[col].astype(str) == category).astype(float)

    # Reindex to match training columns exactly (fills missing with 0)
    return X_out.reindex(columns=training_dummies, fill_value=0.0)
