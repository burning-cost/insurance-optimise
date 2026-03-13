"""Tests for synthetic dataset generation."""

import polars as pl
import numpy as np
import pytest

from insurance_optimise.demand.datasets import generate_conversion_data, generate_retention_data


class TestGenerateConversionData:
    def test_returns_polars_dataframe(self):
        df = generate_conversion_data(n_quotes=1000, seed=0)
        assert isinstance(df, pl.DataFrame)

    def test_expected_columns_present(self):
        df = generate_conversion_data(n_quotes=1000, seed=0)
        required = [
            "quote_id", "quote_date", "channel", "age", "vehicle_group",
            "ncd_years", "area", "annual_mileage", "technical_premium",
            "quoted_price", "price_ratio", "log_price_ratio",
            "competitor_price_min", "price_to_market", "rank_position",
            "converted", "true_elasticity",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count(self):
        df = generate_conversion_data(n_quotes=5000, seed=0)
        assert len(df) == 5000

    def test_converted_is_binary(self):
        df = generate_conversion_data(n_quotes=1000, seed=0)
        assert set(df["converted"].unique().to_list()).issubset({0, 1})

    def test_conversion_rate_plausible(self):
        df = generate_conversion_data(n_quotes=10000, seed=0)
        conv_rate = df["converted"].mean()
        assert 0.05 < conv_rate < 0.80, f"Conversion rate {conv_rate:.2%} looks wrong"

    def test_price_ratio_positive(self):
        df = generate_conversion_data(n_quotes=1000, seed=0)
        assert (df["price_ratio"] > 0).all()

    def test_rank_position_at_least_1(self):
        df = generate_conversion_data(n_quotes=1000, seed=0)
        assert (df["rank_position"] >= 1).all()

    def test_true_elasticity_negative(self):
        df = generate_conversion_data(n_quotes=1000, seed=0, true_price_elasticity=-2.0)
        # Most elasticities should be negative (some may vary slightly)
        assert (df["true_elasticity"] < 0).all()

    def test_reproducible_with_seed(self):
        df1 = generate_conversion_data(n_quotes=500, seed=99)
        df2 = generate_conversion_data(n_quotes=500, seed=99)
        assert df1["converted"].to_list() == df2["converted"].to_list()

    def test_different_seeds_differ(self):
        df1 = generate_conversion_data(n_quotes=500, seed=1)
        df2 = generate_conversion_data(n_quotes=500, seed=2)
        assert df1["converted"].to_list() != df2["converted"].to_list()

    def test_channels_valid(self):
        df = generate_conversion_data(n_quotes=2000, seed=0)
        valid = {"pcw_confused", "pcw_msm", "pcw_ctm", "pcw_go", "direct"}
        assert set(df["channel"].unique().to_list()).issubset(valid)

    def test_technical_premium_positive(self):
        df = generate_conversion_data(n_quotes=1000, seed=0)
        assert (df["technical_premium"] > 0).all()


class TestGenerateRetentionData:
    def test_returns_polars_dataframe(self):
        df = generate_retention_data(n_policies=500, seed=0)
        assert isinstance(df, pl.DataFrame)

    def test_expected_columns_present(self):
        df = generate_retention_data(n_policies=500, seed=0)
        required = [
            "policy_id", "renewal_date", "channel", "age", "vehicle_group",
            "ncd_years", "tenure_years", "payment_method", "area",
            "claim_last_3yr", "technical_premium", "prior_year_price",
            "renewal_price", "price_change_pct", "log_price_change",
            "nb_equivalent_price", "enbp_compliant", "lapsed", "true_lapse_prob",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_lapsed_is_binary(self):
        df = generate_retention_data(n_policies=500, seed=0)
        assert set(df["lapsed"].unique().to_list()).issubset({0, 1})

    def test_lapse_rate_plausible(self):
        df = generate_retention_data(n_policies=5000, seed=0)
        lapse_rate = df["lapsed"].mean()
        assert 0.05 < lapse_rate < 0.60, f"Lapse rate {lapse_rate:.2%} looks wrong"

    def test_enbp_compliance_flag_correct(self):
        df = generate_retention_data(n_policies=2000, seed=0)
        pd_df = df.to_pandas()
        # Check the flag matches the actual column comparison
        expected = pd_df["renewal_price"] <= pd_df["nb_equivalent_price"]
        actual = pd_df["enbp_compliant"]
        assert (expected == actual).all()

    def test_all_renewal_prices_enbp_compliant(self):
        """Generator enforces ENBP compliance (renewal_price capped at nb_price)."""
        df = generate_retention_data(n_policies=2000, seed=0)
        assert df["enbp_compliant"].all()

    def test_tenure_positive(self):
        df = generate_retention_data(n_policies=500, seed=0)
        assert (df["tenure_years"] > 0).all()

    def test_payment_methods_valid(self):
        df = generate_retention_data(n_policies=1000, seed=0)
        valid = {"dd", "card", "cheque"}
        assert set(df["payment_method"].unique().to_list()).issubset(valid)

    def test_price_change_distribution(self):
        df = generate_retention_data(n_policies=5000, seed=0)
        pct_change = df["price_change_pct"]
        # In the simulated data, portfolio is inflating (all price changes positive)
        mean_change = pct_change.mean()
        assert mean_change > 0, "Expected positive average price change in inflationary market"
