"""
RiskInformedRetentionModel: RetentionModel extended with insurance-specific
feature engineering for UK motor renewal classification.

The base RetentionModel treats price_change_col as the sole price signal.
That is correct when the only question is "how sensitive is this customer to
price movement?" But renewal classification has a second dimension: given the
technical price, is the renewal offer fair value? A customer who is offered
£900 when the technical price is £600 (loading ratio 1.5) behaves differently
to one offered £900 when the technical price is £880 (loading ratio 1.02).
The raw price change does not distinguish these.

This class extends RetentionModel by constructing three additional features
before fitting:

1. ``loading_ratio`` = renewal_price / technical_price
   How much commercial loading is embedded in the renewal offer. This is the
   primary insight from Boonkrong et al. (2025): when the classifier can see
   the deviation from technical price, predictive accuracy improves materially.
   A loading_ratio of 1.0 is a break-even renewal; ratios above 1.2 are
   associated with elevated lapse in UK motor data.

2. ``enbp_proximity`` = renewal_price / enbp_price
   How close the renewal offer is to the FCA PS21/11 ENBP ceiling. Values
   near 1.0 mean the insurer has little room to discount further. This is a
   UK-specific feature: the Thai paper (Boonkrong et al.) does not model
   it because Thailand has no ENBP equivalent. Insurers offering at near-ENBP
   tend to retain fewer customers who shop on PCW.

3. ``ncb_years`` (passthrough, with standardised name)
   No-claims bonus years. The base generate_retention_data dataset calls this
   ``ncd_years``; real insurer systems often call it ``ncb_years``. This class
   accepts either name via ``ncb_col``.

4. ``claims_last_3yr`` (passthrough, with standardised name)
   Claim count or binary flag for claims in the past three years. Claims
   increase search activity at renewal.

The feature engineering is applied inside fit() and predict_proba() before
delegating to the parent class. All other RetentionModel functionality
(survival models, oneway diagnostics, price_sensitivity, summary) is
inherited unchanged.

Post-PS21/11 compliance note:
------------------------------
This class adds a warning if enbp_proximity > 1.0 for more than 5% of
records. Renewal prices above ENBP are a PS21/11 violation (FCA PS22/9
Consumer Duty also applies). The warning does not block fitting — you may
be intentionally modelling a pre-compliance portfolio — but it should be
investigated before production use.

Reference
---------
Boonkrong, W., Siripanich, P., & Wonglorsaichon, P. (2025). Risk-Informed
Motor Insurance Renewal Classification. MDPI Risks, 14(3), 57.
https://doi.org/10.3390/risks14030057
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .retention import RetentionModel
from ._types import DataFrameLike, RetentionModelType
from .conversion import _to_pandas


# Threshold for loading_ratio below which we suspect the columns are swapped.
# A loading_ratio of 0.5 means renewal_price is half the technical_price, which
# is implausible in normal motor pricing and suggests the arguments were inverted.
_LOADING_RATIO_SWAP_THRESHOLD = 0.5

# Fraction of records with enbp_proximity > 1.0 above which we warn.
_ENBP_VIOLATION_WARN_FRACTION = 0.05


class RiskInformedRetentionModel(RetentionModel):
    """
    Retention model with automatic risk-informed feature engineering.

    Extends :class:`RetentionModel` by constructing loading_ratio,
    enbp_proximity, and passthrough columns for NCD and claims before
    fitting. All base class functionality is preserved.

    Parameters
    ----------
    model_type : {'logistic', 'catboost', 'cox', 'weibull'}
        Model backend. Passed through to RetentionModel. Default 'logistic'.
    technical_price_col : str
        Column containing the technical (risk) premium at renewal date.
        Default 'technical_premium'.
    renewal_price_col : str
        Column containing the quoted renewal price. Default 'renewal_price'.
    enbp_price_col : str, optional
        Column containing the FCA ENBP (new business equivalent price).
        If supplied, enbp_proximity = renewal_price / enbp_price is added
        as a feature. Default 'nb_equivalent_price'. Set to None to omit.
    ncb_col : str, optional
        Column for no-claims bonus years. Added as a feature when present.
        Default 'ncd_years'. Set to None to omit.
    claims_col : str, optional
        Column for claims in the last three years. Added as a feature when
        present. Default 'claim_last_3yr'. Set to None to omit.
    **retention_kwargs
        All other keyword arguments are passed to RetentionModel.__init__.
        Common ones: outcome_col, duration_col, price_change_col,
        feature_cols, cat_features, catboost_params, logistic_C.

    Notes
    -----
    The engineered features (loading_ratio, enbp_proximity) are appended to
    feature_cols automatically. If you explicitly include them in feature_cols
    they will not be duplicated.

    Examples
    --------
    >>> from insurance_optimise.demand import RiskInformedRetentionModel
    >>> from insurance_optimise.demand.datasets import generate_retention_data
    >>> df = generate_retention_data(n_policies=50_000)
    >>> model = RiskInformedRetentionModel(
    ...     model_type='logistic',
    ...     technical_price_col='technical_premium',
    ...     renewal_price_col='renewal_price',
    ...     enbp_price_col='nb_equivalent_price',
    ...     ncb_col='ncd_years',
    ...     claims_col='claim_last_3yr',
    ... )
    >>> model.fit(df)
    >>> lapse_probs = model.predict_proba(df)
    >>> report = model.feature_importance_report()
    >>> print(report)
    """

    def __init__(
        self,
        model_type: RetentionModelType = "logistic",
        technical_price_col: str = "technical_premium",
        renewal_price_col: str = "renewal_price",
        enbp_price_col: Optional[str] = "nb_equivalent_price",
        ncb_col: Optional[str] = "ncd_years",
        claims_col: Optional[str] = "claim_last_3yr",
        **retention_kwargs,
    ) -> None:
        self.technical_price_col = technical_price_col
        self.renewal_price_col = renewal_price_col
        self.enbp_price_col = enbp_price_col
        self.ncb_col = ncb_col
        self.claims_col = claims_col

        # The engineered feature names we may add. We record which ones are
        # actually active after seeing the first dataset in fit().
        self._engineered_features: list[str] = []

        super().__init__(model_type=model_type, **retention_kwargs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: DataFrameLike) -> "RiskInformedRetentionModel":
        """
        Engineer risk-informed features then fit the retention model.

        Parameters
        ----------
        data : DataFrame
            Policy-level renewal data. Must contain outcome_col,
            price_change_col, technical_price_col, and renewal_price_col.
            Optional: enbp_price_col, ncb_col, claims_col.

        Returns
        -------
        self
        """
        df = _to_pandas(data)
        df_eng = self._engineer_features(df, training=True)
        return super().fit(df_eng)

    def predict_proba(self, data: DataFrameLike) -> pd.Series:
        """
        Engineer risk-informed features then predict lapse probability.

        Returns
        -------
        pd.Series
            Lapse probability in [0, 1] for each policy.
        """
        df = _to_pandas(data)
        df_eng = self._engineer_features(df, training=False)
        return super().predict_proba(df_eng)

    def predict_renewal_proba(self, data: DataFrameLike) -> pd.Series:
        """
        Predict renewal (retention) probability. Convenience: 1 - lapse_prob.

        Returns
        -------
        pd.Series
            Renewal probability in [0, 1].
        """
        return (1 - self.predict_proba(data)).rename("renewal_prob")

    def feature_importance_report(self) -> pd.DataFrame:
        """
        Feature importance with labels distinguishing risk-informed features.

        Wraps summary() and adds a ``feature_type`` column marking whether
        each feature is an engineered risk-informed feature or a base feature
        passed in from the original data.

        Returns
        -------
        pd.DataFrame
            summary() columns plus ``feature_type``
            ('risk_informed' or 'base').

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        RuntimeError
            If model_type is 'cox' or 'weibull' (survival models do not
            expose feature importances in the same format).
        """
        self._check_fitted()
        if self._is_survival:
            raise RuntimeError(
                "feature_importance_report is not available for survival models. "
                "Use summary() to inspect the lifelines coefficient table directly."
            )

        df_summary = self.summary()

        ri_features = set(self._engineered_features)
        # For logistic, features may be one-hot encoded: check by prefix
        df_summary = df_summary.copy()
        df_summary["feature_type"] = df_summary["feature"].apply(
            lambda f: "risk_informed" if _feature_is_risk_informed(f, ri_features)
            else "base"
        )
        return df_summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame, training: bool) -> pd.DataFrame:
        """
        Return a copy of df with engineered features added and feature_cols
        updated to include them.

        This is called from fit() and predict_proba(). The method is
        idempotent: if the engineered columns already exist in df, they
        are overwritten with freshly computed values (not duplicated).
        """
        df = df.copy()
        added: list[str] = []

        # --- loading_ratio ---
        if (self.technical_price_col in df.columns
                and self.renewal_price_col in df.columns):
            tech = df[self.technical_price_col].values.astype(float)
            ren = df[self.renewal_price_col].values.astype(float)
            _validate_loading_ratio(ren, tech)
            df["loading_ratio"] = ren / np.maximum(tech, 1e-8)
            added.append("loading_ratio")

        # --- enbp_proximity ---
        if (self.enbp_price_col is not None
                and self.enbp_price_col in df.columns
                and self.renewal_price_col in df.columns):
            enbp = df[self.enbp_price_col].values.astype(float)
            ren = df[self.renewal_price_col].values.astype(float)
            df["enbp_proximity"] = ren / np.maximum(enbp, 1e-8)
            _validate_enbp_proximity(df["enbp_proximity"].values)
            added.append("enbp_proximity")

        # --- ncb passthrough ---
        if self.ncb_col is not None and self.ncb_col in df.columns:
            # If the column is already named ncb_years leave it; otherwise
            # alias it so the model always sees the same feature name.
            if self.ncb_col != "ncb_years":
                df["ncb_years"] = df[self.ncb_col]
                added.append("ncb_years")
            else:
                added.append("ncb_years")

        # --- claims passthrough ---
        if self.claims_col is not None and self.claims_col in df.columns:
            if self.claims_col != "claims_last_3yr":
                df["claims_last_3yr"] = df[self.claims_col]
                added.append("claims_last_3yr")
            else:
                added.append("claims_last_3yr")

        # Store the engineered feature names on first call (training=True).
        if training:
            self._engineered_features = added
            # Extend feature_cols with any new engineered columns not already listed.
            existing = set(self.feature_cols)
            for col in added:
                if col not in existing:
                    self.feature_cols.append(col)

        return df


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------

def _validate_loading_ratio(renewal: np.ndarray, technical: np.ndarray) -> None:
    """Warn if loading_ratio values suggest the columns are swapped."""
    ratio = renewal / np.maximum(technical, 1e-8)
    median_ratio = float(np.median(ratio))
    if median_ratio < _LOADING_RATIO_SWAP_THRESHOLD:
        warnings.warn(
            f"Median loading_ratio is {median_ratio:.3f}, which is below "
            f"{_LOADING_RATIO_SWAP_THRESHOLD}. This may indicate that "
            "technical_price_col and renewal_price_col are swapped. "
            "Check column assignments. "
            f"loading_ratio = renewal_price / technical_price.",
            UserWarning,
            stacklevel=4,
        )


def _validate_enbp_proximity(proximity: np.ndarray) -> None:
    """Warn if a material fraction of records has renewal > ENBP (PS21/11 violation)."""
    violation_rate = float(np.mean(proximity > 1.0))
    if violation_rate > _ENBP_VIOLATION_WARN_FRACTION:
        warnings.warn(
            f"{violation_rate:.1%} of records have renewal_price > enbp_price "
            f"(enbp_proximity > 1.0). Renewal prices above ENBP are a potential "
            "FCA PS21/11 violation. Review the data before using this model in "
            "production.",
            UserWarning,
            stacklevel=4,
        )


def _feature_is_risk_informed(feature_name: str, ri_features: set) -> bool:
    """
    Return True if feature_name corresponds to a risk-informed engineered feature.

    Handles the case where logistic encoding creates derived columns such as
    ``loading_ratio_0.9_1.0`` from a binned or one-hot encoded source.
    """
    for ri in ri_features:
        if feature_name == ri or feature_name.startswith(ri + "_"):
            return True
    return False
