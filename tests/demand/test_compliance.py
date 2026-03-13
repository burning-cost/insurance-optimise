"""Tests for ENBP compliance utilities."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_optimise.demand.compliance import ENBPChecker, enbp_compliant, price_walking_report
from insurance_optimise.demand.datasets import generate_retention_data


_DF = generate_retention_data(n_policies=2_000, seed=42)
_DF_PD = _DF.to_pandas()


class TestENBPChecker:
    def setup_method(self):
        self.checker = ENBPChecker()

    def test_check_returns_report(self):
        from insurance_optimise.demand.compliance import ENBPBreachReport
        report = self.checker.check(_DF)
        assert isinstance(report, ENBPBreachReport)

    def test_compliant_data_has_no_breaches(self):
        """The synthetic data generator enforces ENBP - should have zero breaches."""
        report = self.checker.check(_DF)
        assert report.n_breaches == 0
        assert report.breach_rate == 0.0

    def test_n_policies_correct(self):
        report = self.checker.check(_DF)
        assert report.n_policies == len(_DF)

    def test_by_channel_has_expected_structure(self):
        report = self.checker.check(_DF)
        assert "channel" in report.by_channel.columns
        assert "n_policies" in report.by_channel.columns
        assert "breach_rate" in report.by_channel.columns

    def test_detects_breaches_in_modified_data(self):
        df = _DF_PD.copy()
        # Manually introduce breaches: set renewal_price 20% above ENBP for first 100 rows
        df.loc[:99, "renewal_price"] = df.loc[:99, "nb_equivalent_price"] * 1.20
        report = self.checker.check(df)
        assert report.n_breaches == 100
        assert report.breach_rate == pytest.approx(100 / len(df))

    def test_breach_detail_contains_breaching_policies(self):
        df = _DF_PD.copy()
        df.loc[:49, "renewal_price"] = df.loc[:49, "nb_equivalent_price"] * 1.10
        report = self.checker.check(df)
        assert len(report.breach_detail) == 50

    def test_tolerance_reduces_breach_count(self):
        df = _DF_PD.copy()
        # Introduce small breach of £5
        df.loc[:99, "renewal_price"] = df.loc[:99, "nb_equivalent_price"] + 5
        checker_strict = ENBPChecker(tolerance=0.0)
        checker_lax = ENBPChecker(tolerance=10.0)
        report_strict = checker_strict.check(df)
        report_lax = checker_lax.check(df)
        assert report_strict.n_breaches == 100
        assert report_lax.n_breaches == 0

    def test_accepts_polars_dataframe(self):
        report = self.checker.check(_DF)
        assert report.n_policies == len(_DF)

    def test_missing_required_column_raises(self):
        df_bad = _DF_PD.drop(columns=["renewal_price"])
        with pytest.raises(ValueError, match="renewal_price"):
            self.checker.check(df_bad)

    def test_repr_includes_key_info(self):
        report = self.checker.check(_DF)
        r = repr(report)
        assert "n_policies" in r
        assert "n_breaches" in r


class TestENBPCompliantScalar:
    def test_compliant_case(self):
        assert enbp_compliant(500.0, 520.0) is True

    def test_breach_case(self):
        assert enbp_compliant(530.0, 520.0) is False

    def test_equal_is_compliant(self):
        assert enbp_compliant(500.0, 500.0) is True

    def test_tolerance_applied(self):
        # £505 vs £500 ENBP: breach without tolerance, compliant with £10 tolerance
        assert enbp_compliant(505.0, 500.0, tolerance=0.0) is False
        assert enbp_compliant(505.0, 500.0, tolerance=10.0) is True


class TestPriceWalkingReport:
    def test_returns_dataframe(self):
        result = price_walking_report(_DF)
        assert isinstance(result, pd.DataFrame)

    def test_has_tenure_band_column(self):
        result = price_walking_report(_DF)
        assert "tenure_band" in result.columns

    def test_has_price_columns(self):
        result = price_walking_report(_DF)
        assert "mean_renewal_price" in result.columns

    def test_with_nb_price_col(self):
        result = price_walking_report(
            _DF,
            nb_price_col="nb_equivalent_price",
        )
        assert "mean_price_to_enbp" in result.columns

    def test_by_channel(self):
        result = price_walking_report(_DF, channel_col="channel")
        # Should have channel column
        assert "channel" in result.columns

    def test_n_tenure_bins(self):
        result = price_walking_report(_DF, channel_col=None, n_tenure_bins=4)
        # Should have at most 4 tenure bands
        assert result["tenure_band"].nunique() <= 4
