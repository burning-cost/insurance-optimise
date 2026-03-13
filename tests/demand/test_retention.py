"""Tests for RetentionModel."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_optimise.demand import RetentionModel
from insurance_optimise.demand.datasets import generate_retention_data


FEATURE_COLS = ["tenure_years", "ncd_years", "payment_method", "claim_last_3yr", "channel"]

_DF = generate_retention_data(n_policies=4_000, seed=42)
_DF_PD = _DF.to_pandas()


class TestRetentionModelLogistic:
    def setup_method(self):
        self.model = RetentionModel(
            model_type="logistic",
            feature_cols=FEATURE_COLS,
        )

    def test_fit_returns_self(self):
        result = self.model.fit(_DF)
        assert result is self.model

    def test_predict_proba_shape(self):
        self.model.fit(_DF)
        probs = self.model.predict_proba(_DF)
        assert isinstance(probs, pd.Series)
        assert len(probs) == len(_DF)

    def test_predict_proba_range(self):
        self.model.fit(_DF)
        probs = self.model.predict_proba(_DF)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_proba_accepts_pandas(self):
        self.model.fit(_DF)
        probs = self.model.predict_proba(_DF_PD)
        assert len(probs) == len(_DF_PD)

    def test_predict_renewal_proba_is_complement(self):
        self.model.fit(_DF)
        lapse = self.model.predict_proba(_DF)
        renewal = self.model.predict_renewal_proba(_DF)
        np.testing.assert_allclose(lapse.values + renewal.values, 1.0, atol=1e-8)

    def test_not_fitted_raises(self):
        model = RetentionModel(feature_cols=FEATURE_COLS)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(_DF)

    def test_price_sensitivity_positive(self):
        """Higher price change should increase lapse probability."""
        self.model.fit(_DF)
        sens = self.model.price_sensitivity(_DF)
        # Average sensitivity should be positive (more price increase → more lapses)
        assert sens.mean() > 0, "Expected positive price sensitivity (higher price → more lapses)"

    def test_oneway_structure(self):
        self.model.fit(_DF)
        result = self.model.oneway(_DF, "payment_method")
        assert "factor_level" in result.columns
        assert "n_policies" in result.columns
        assert "observed_lapse_rate" in result.columns
        assert "fitted_lapse_rate" in result.columns

    def test_oneway_numeric_bins(self):
        self.model.fit(_DF)
        result = self.model.oneway(_DF, "tenure_years", bins=5)
        assert len(result) <= 5

    def test_summary_structure(self):
        self.model.fit(_DF)
        summary = self.model.summary()
        assert "feature" in summary.columns
        assert "coefficient" in summary.columns

    def test_higher_price_change_higher_lapse(self):
        """Sanity: portfolio with higher price changes should have higher lapse rate."""
        self.model.fit(_DF)
        df_lo = _DF_PD.copy()
        df_lo["log_price_change"] = -0.05  # price cut → low lapse
        df_hi = _DF_PD.copy()
        df_hi["log_price_change"] = 0.20   # large increase → high lapse
        lapse_lo = self.model.predict_proba(df_lo).mean()
        lapse_hi = self.model.predict_proba(df_hi).mean()
        assert lapse_hi > lapse_lo

    def test_dd_vs_card_retention(self):
        """DD payers should have higher predicted retention than card payers."""
        self.model.fit(_DF)
        df_dd = _DF_PD.copy()
        df_dd["payment_method"] = "dd"
        df_card = _DF_PD.copy()
        df_card["payment_method"] = "card"
        lapse_dd = self.model.predict_proba(df_dd).mean()
        lapse_card = self.model.predict_proba(df_card).mean()
        assert lapse_dd < lapse_card, "DD payers should lapse less than card payers"

    def test_survival_not_available_for_logistic(self):
        self.model.fit(_DF)
        with pytest.raises(RuntimeError, match="survival model"):
            self.model.predict_survival(_DF)


class TestRetentionModelMissingPriceChangeCol:
    def test_raises_on_missing_price_col(self):
        model = RetentionModel(
            model_type="logistic",
            price_change_col="nonexistent_col",
            feature_cols=FEATURE_COLS,
        )
        with pytest.raises(ValueError, match="nonexistent_col"):
            model.fit(_DF)
