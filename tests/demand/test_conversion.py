"""Tests for ConversionModel."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_optimise.demand import ConversionModel
from insurance_optimise.demand.datasets import generate_conversion_data


FEATURE_COLS = ["age", "vehicle_group", "ncd_years", "area", "channel"]

# Generate data once for the module
_DF = generate_conversion_data(n_quotes=5_000, seed=42)
_DF_PD = _DF.to_pandas()


class TestConversionModelLogistic:
    def setup_method(self):
        self.model = ConversionModel(
            base_estimator="logistic",
            feature_cols=FEATURE_COLS,
            rank_position_col="rank_position",
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

    def test_predict_returns_binary(self):
        self.model.fit(_DF)
        preds = self.model.predict(_DF)
        assert set(preds.unique()).issubset({0, 1})

    def test_marginal_effect_negative_on_average(self):
        """Marginal effect of price should be negative (higher price → lower conversion)."""
        self.model.fit(_DF)
        me = self.model.marginal_effect(_DF)
        assert me.mean() < 0, "Expected negative average marginal effect"

    def test_price_elasticity_negative(self):
        self.model.fit(_DF)
        elast = self.model.price_elasticity(_DF)
        assert elast.mean() < 0

    def test_not_fitted_raises(self):
        model = ConversionModel(feature_cols=FEATURE_COLS)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(_DF)

    def test_oneway_structure(self):
        self.model.fit(_DF)
        result = self.model.oneway(_DF, "channel")
        assert "factor_level" in result.columns
        assert "n_quotes" in result.columns
        assert "observed_rate" in result.columns
        assert "fitted_rate" in result.columns
        assert "lift" in result.columns

    def test_oneway_numeric_with_bins(self):
        self.model.fit(_DF)
        result = self.model.oneway(_DF, "age", bins=5)
        assert len(result) <= 5

    def test_summary_structure(self):
        self.model.fit(_DF)
        summary = self.model.summary()
        assert "feature" in summary.columns
        assert "coefficient" in summary.columns
        assert "odds_ratio" in summary.columns

    def test_as_demand_callable_returns_callable(self):
        self.model.fit(_DF)
        fn = self.model.as_demand_callable()
        assert callable(fn)

    def test_as_demand_callable_output_shape(self):
        self.model.fit(_DF)
        fn = self.model.as_demand_callable()
        import polars as pl
        price_ratio = np.ones(100)
        features = _DF.head(100)
        result = fn(price_ratio, features)
        assert len(result) == 100
        assert (result >= 0).all() and (result <= 1).all()

    def test_higher_price_lower_conversion(self):
        """Sanity check: higher price should predict lower conversion rate."""
        self.model.fit(_DF)
        df_lo = _DF_PD.copy()
        df_lo["quoted_price"] = df_lo["quoted_price"] * 0.8
        df_hi = _DF_PD.copy()
        df_hi["quoted_price"] = df_hi["quoted_price"] * 1.2
        probs_lo = self.model.predict_proba(df_lo).mean()
        probs_hi = self.model.predict_proba(df_hi).mean()
        assert probs_lo > probs_hi, "Lower price should give higher conversion rate"


class TestConversionModelWithoutTechnicalPremium:
    """Test that ConversionModel works without a technical premium column."""

    def test_fit_without_tech_premium(self):
        model = ConversionModel(
            base_estimator="logistic",
            technical_premium_col=None,
            feature_cols=FEATURE_COLS,
        )
        model.fit(_DF)
        probs = model.predict_proba(_DF)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestConversionModelPriceToMarket:
    def test_fit_with_price_to_market(self):
        model = ConversionModel(
            base_estimator="logistic",
            feature_cols=FEATURE_COLS,
            rank_position_col="rank_position",
            price_to_market_col="price_to_market",
        )
        model.fit(_DF)
        probs = model.predict_proba(_DF)
        assert len(probs) == len(_DF)
