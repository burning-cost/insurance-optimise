"""Tests for RiskInformedRetentionModel."""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_optimise.demand import RiskInformedRetentionModel
from insurance_optimise.demand.datasets import generate_retention_data


# Single shared dataset for the majority of tests.
_DF = generate_retention_data(n_policies=3_000, seed=99)
_DF_PD = _DF.to_pandas()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_model(**kwargs) -> RiskInformedRetentionModel:
    defaults = dict(
        model_type="logistic",
        technical_price_col="technical_premium",
        renewal_price_col="renewal_price",
        enbp_price_col="nb_equivalent_price",
        ncb_col="ncd_years",
        claims_col="claim_last_3yr",
    )
    defaults.update(kwargs)
    return RiskInformedRetentionModel(**defaults)


# ------------------------------------------------------------------
# Feature engineering correctness
# ------------------------------------------------------------------

class TestFeatureEngineering:
    def test_loading_ratio_values(self):
        """loading_ratio = renewal_price / technical_premium."""
        model = _make_model()
        df_eng = model._engineer_features(_DF_PD.copy(), training=True)
        expected = _DF_PD["renewal_price"] / _DF_PD["technical_premium"]
        np.testing.assert_allclose(
            df_eng["loading_ratio"].values,
            expected.values,
            rtol=1e-6,
        )

    def test_enbp_proximity_values(self):
        """enbp_proximity = renewal_price / nb_equivalent_price."""
        model = _make_model()
        df_eng = model._engineer_features(_DF_PD.copy(), training=True)
        expected = _DF_PD["renewal_price"] / _DF_PD["nb_equivalent_price"]
        np.testing.assert_allclose(
            df_eng["enbp_proximity"].values,
            expected.values,
            rtol=1e-6,
        )

    def test_engineered_features_added_to_feature_cols(self):
        """After _engineer_features(training=True), feature_cols includes engineered names."""
        model = _make_model()
        model._engineer_features(_DF_PD.copy(), training=True)
        assert "loading_ratio" in model.feature_cols
        assert "enbp_proximity" in model.feature_cols

    def test_no_duplicates_in_feature_cols(self):
        """Calling _engineer_features twice does not duplicate feature_cols entries."""
        model = _make_model()
        model._engineer_features(_DF_PD.copy(), training=True)
        # Call again with training=False (as predict_proba does)
        model._engineer_features(_DF_PD.copy(), training=False)
        assert model.feature_cols.count("loading_ratio") == 1


# ------------------------------------------------------------------
# Fit / predict roundtrip — logistic
# ------------------------------------------------------------------

class TestFitPredictLogistic:
    def setup_method(self):
        self.model = _make_model(model_type="logistic")

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
        model = _make_model()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(_DF)

    def test_feature_importance_report_is_dataframe(self):
        self.model.fit(_DF)
        report = self.model.feature_importance_report()
        assert isinstance(report, pd.DataFrame)
        assert "feature_type" in report.columns

    def test_feature_importance_report_has_risk_informed_rows(self):
        self.model.fit(_DF)
        report = self.model.feature_importance_report()
        ri_rows = report[report["feature_type"] == "risk_informed"]
        assert len(ri_rows) > 0, "Expected at least one risk_informed feature in report"

    def test_feature_importance_report_has_base_rows(self):
        self.model.fit(_DF)
        report = self.model.feature_importance_report()
        base_rows = report[report["feature_type"] == "base"]
        assert len(base_rows) > 0, "Expected at least one base feature in report"


# ------------------------------------------------------------------
# Fit / predict roundtrip — catboost (skip if not installed)
# ------------------------------------------------------------------

_has_catboost = pytest.importorskip is not None  # dummy; real check below
try:
    import catboost as _catboost_mod
    _has_catboost = True
except ImportError:
    _has_catboost = False


@pytest.mark.skipif(not _has_catboost, reason="catboost not installed")
class TestFitPredictCatBoost:
    def setup_method(self):
        self.model = _make_model(model_type="catboost")

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

    def test_feature_importance_report_is_dataframe(self):
        self.model.fit(_DF)
        report = self.model.feature_importance_report()
        assert isinstance(report, pd.DataFrame)
        assert "feature_type" in report.columns


# ------------------------------------------------------------------
# Validation warnings
# ------------------------------------------------------------------

class TestValidationWarnings:
    def test_swapped_columns_warning_fires(self):
        """
        If loading_ratio median is below 0.5 a warning should fire
        (likely swapped columns or wrong data).
        """
        # Create data where renewal_price << technical_premium to trigger warning.
        df_bad = _DF_PD.copy()
        df_bad["renewal_price"] = df_bad["technical_premium"] * 0.3  # ratio = 0.3
        model = _make_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._engineer_features(df_bad, training=True)

        messages = [str(w.message) for w in caught]
        assert any("loading_ratio" in m or "swapped" in m.lower() for m in messages), (
            "Expected a UserWarning about low loading_ratio (likely swapped columns)"
        )

    def test_enbp_violation_warning_fires(self):
        """
        When renewal prices exceed ENBP for more than 5% of records a
        PS21/11 warning should fire.
        """
        df_bad = _DF_PD.copy()
        # Make most renewal prices exceed ENBP.
        df_bad["nb_equivalent_price"] = df_bad["renewal_price"] * 0.80  # ENBP < renewal
        model = _make_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._engineer_features(df_bad, training=True)

        messages = [str(w.message) for w in caught]
        assert any("PS21/11" in m or "enbp" in m.lower() or "ENBP" in m for m in messages), (
            "Expected a UserWarning about ENBP proximity violations"
        )

    def test_low_loading_ratio_warning_threshold(self):
        """
        A median loading_ratio of exactly 0.4 (below threshold 0.5) fires a warning.
        """
        df_low = _DF_PD.copy()
        # Force renewal_price to be 40% of technical_premium.
        df_low["renewal_price"] = df_low["technical_premium"] * 0.40
        model = _make_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._engineer_features(df_low, training=True)

        messages = [str(w.message) for w in caught]
        assert any("loading_ratio" in m or "swapped" in m.lower() for m in messages)

    def test_no_warning_when_data_is_clean(self):
        """No spurious UserWarning on clean synthetic data."""
        model = _make_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._engineer_features(_DF_PD.copy(), training=True)

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0, (
            f"Unexpected warnings on clean data: {[str(w.message) for w in user_warnings]}"
        )


# ------------------------------------------------------------------
# Optional columns
# ------------------------------------------------------------------

class TestOptionalColumns:
    def test_works_without_enbp_col(self):
        """enbp_proximity should be omitted when enbp_price_col=None."""
        model = RiskInformedRetentionModel(
            model_type="logistic",
            technical_price_col="technical_premium",
            renewal_price_col="renewal_price",
            enbp_price_col=None,
            ncb_col="ncd_years",
            claims_col="claim_last_3yr",
        )
        model.fit(_DF)
        assert "enbp_proximity" not in model.feature_cols
        probs = model.predict_proba(_DF)
        assert len(probs) == len(_DF)

    def test_works_without_ncb_col(self):
        """NCB feature should be omitted when ncb_col=None."""
        model = RiskInformedRetentionModel(
            model_type="logistic",
            technical_price_col="technical_premium",
            renewal_price_col="renewal_price",
            enbp_price_col="nb_equivalent_price",
            ncb_col=None,
            claims_col="claim_last_3yr",
        )
        model.fit(_DF)
        assert "ncb_years" not in model.feature_cols
        probs = model.predict_proba(_DF)
        assert len(probs) == len(_DF)

    def test_works_without_claims_col(self):
        """Claims feature should be omitted when claims_col=None."""
        model = RiskInformedRetentionModel(
            model_type="logistic",
            technical_price_col="technical_premium",
            renewal_price_col="renewal_price",
            enbp_price_col="nb_equivalent_price",
            ncb_col="ncd_years",
            claims_col=None,
        )
        model.fit(_DF)
        assert "claims_last_3yr" not in model.feature_cols
        probs = model.predict_proba(_DF)
        assert len(probs) == len(_DF)

    def test_works_with_only_loading_ratio(self):
        """Minimal configuration: only loading_ratio, nothing else."""
        model = RiskInformedRetentionModel(
            model_type="logistic",
            technical_price_col="technical_premium",
            renewal_price_col="renewal_price",
            enbp_price_col=None,
            ncb_col=None,
            claims_col=None,
        )
        model.fit(_DF)
        assert "loading_ratio" in model.feature_cols
        probs = model.predict_proba(_DF)
        assert len(probs) == len(_DF)

    def test_works_when_enbp_col_absent_from_data(self):
        """
        If enbp_price_col is set but the column is not in the data,
        enbp_proximity should be silently omitted.
        """
        df_no_enbp = _DF_PD.drop(columns=["nb_equivalent_price"])
        model = _make_model()
        model.fit(df_no_enbp)
        assert "enbp_proximity" not in model.feature_cols
        probs = model.predict_proba(df_no_enbp)
        assert len(probs) == len(df_no_enbp)

    def test_ncb_col_aliasing(self):
        """
        When ncb_col points to a column with a non-standard name, the
        feature is aliased to 'ncb_years' in feature_cols.
        """
        df_alias = _DF_PD.copy()
        df_alias = df_alias.rename(columns={"ncd_years": "no_claims_bonus"})
        model = RiskInformedRetentionModel(
            model_type="logistic",
            technical_price_col="technical_premium",
            renewal_price_col="renewal_price",
            enbp_price_col=None,
            ncb_col="no_claims_bonus",
            claims_col=None,
        )
        model.fit(df_alias)
        assert "ncb_years" in model.feature_cols


# ------------------------------------------------------------------
# Inheritance / base class methods still work
# ------------------------------------------------------------------

class TestInheritance:
    def test_price_sensitivity_available(self):
        model = _make_model()
        model.fit(_DF)
        sens = model.price_sensitivity(_DF)
        assert isinstance(sens, pd.Series)
        assert len(sens) == len(_DF)

    def test_oneway_available(self):
        model = _make_model()
        model.fit(_DF)
        result = model.oneway(_DF, "payment_method")
        assert "factor_level" in result.columns

    def test_summary_available(self):
        model = _make_model()
        model.fit(_DF)
        s = model.summary()
        assert isinstance(s, pd.DataFrame)
