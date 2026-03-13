"""
ENBP compliance utilities for PS21/11 (GIPP).

The core rule from ICOBS 6B.3: at renewal, the price offered to an existing
customer must be no higher than the Equivalent New Business Price (ENBP) -
the price that would be offered to the same customer as a new customer through
the same channel.

This module does NOT enforce ENBP automatically in any model or optimiser.
That would be presumptuous - your pricing system may have legitimate reasons
for a slight discrepancy at the reporting date, and ENBP calculation requires
accessing your actual NB rating output (which this library doesn't control).

What this module DOES:
1. ENBPChecker: Given a renewal portfolio with renewal_price and nb_price columns,
   identify breaches, quantify the extent, and produce a regulatory-style audit table.
2. enbp_compliant: Simple boolean check per row.
3. price_walking_report: Detect systematic patterns in price vs. tenure that
   might indicate lingering price-walking (a PS21/11 violation audit tool).

References:
- FCA PS21/11: https://www.fca.org.uk/publications/policy-statements/ps21-11-general-insurance-pricing-practices-amendments
- FCA PS21/5: https://www.fca.org.uk/publication/policy/ps21-5.pdf
- FCA EP25/2: Evaluation report July 2025

Practical note on ENBP calculation:
The ENBP is the NB price through the same channel, including the value of any
cash-equivalent NB incentives (cashback, first-year discounts). If your NB
pricing has a PCW cashback of £50 on a £500 policy, the ENBP for a renewing
customer in that channel is £450, not £500. Make sure your nb_price column
accounts for this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from .conversion import _to_pandas
from ._types import DataFrameLike


@dataclass
class ENBPBreachReport:
    """
    Summary of ENBP compliance analysis.

    Attributes
    ----------
    n_policies : int
        Total renewal policies checked.
    n_breaches : int
        Policies where renewal_price > nb_price.
    breach_rate : float
        Fraction of policies in breach.
    mean_breach_amount : float
        Mean (renewal_price - nb_price) for breaching policies.
    max_breach_amount : float
        Maximum breach amount.
    total_breach_amount : float
        Sum of breach amounts across all breaching policies.
    by_channel : pd.DataFrame
        Breach summary broken down by channel.
    breach_detail : pd.DataFrame
        Row-level detail of breaching policies.
    """
    n_policies: int
    n_breaches: int
    breach_rate: float
    mean_breach_amount: float
    max_breach_amount: float
    total_breach_amount: float
    by_channel: pd.DataFrame
    breach_detail: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"ENBPBreachReport(\n"
            f"  n_policies={self.n_policies:,},\n"
            f"  n_breaches={self.n_breaches:,} ({self.breach_rate:.2%}),\n"
            f"  mean_breach=£{self.mean_breach_amount:.2f},\n"
            f"  max_breach=£{self.max_breach_amount:.2f},\n"
            f"  total_overpayment=£{self.total_breach_amount:,.2f}\n"
            f")"
        )


class ENBPChecker:
    """
    ENBP compliance checker for renewal portfolios.

    Parameters
    ----------
    renewal_price_col : str
        Column name for the renewal price offered. Default 'renewal_price'.
    nb_price_col : str
        Column name for the new business equivalent price. Default 'nb_equivalent_price'.
    channel_col : str, optional
        Column for the sales channel. Used in by-channel reporting.
    policy_id_col : str, optional
        Column for policy identifier. Used in breach detail report.
    tolerance : float
        Amount (£) by which renewal_price may exceed nb_price without being
        flagged as a breach. Default 0.0 (strict: any overprice is flagged).
        The FCA does not specify a tolerance; use 0 for full compliance.

    Examples
    --------
    >>> from insurance_demand.compliance import ENBPChecker
    >>> checker = ENBPChecker()
    >>> report = checker.check(df_renewals)
    >>> print(report)
    >>> report.by_channel  # breach rate by channel
    """

    def __init__(
        self,
        renewal_price_col: str = "renewal_price",
        nb_price_col: str = "nb_equivalent_price",
        channel_col: Optional[str] = "channel",
        policy_id_col: Optional[str] = "policy_id",
        tolerance: float = 0.0,
    ) -> None:
        self.renewal_price_col = renewal_price_col
        self.nb_price_col = nb_price_col
        self.channel_col = channel_col
        self.policy_id_col = policy_id_col
        self.tolerance = tolerance

    def check(self, data: DataFrameLike) -> ENBPBreachReport:
        """
        Run ENBP compliance check on a renewal portfolio.

        Parameters
        ----------
        data : DataFrame
            Must contain renewal_price_col and nb_price_col.

        Returns
        -------
        ENBPBreachReport
        """
        df = _to_pandas(data)

        required = [self.renewal_price_col, self.nb_price_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available: {df.columns.tolist()}"
            )

        renewal = df[self.renewal_price_col].values
        nb = df[self.nb_price_col].values
        breach_amount = renewal - nb - self.tolerance
        is_breach = breach_amount > 0

        n_policies = len(df)
        n_breaches = int(is_breach.sum())
        breach_rate = n_breaches / max(n_policies, 1)

        breach_amounts = breach_amount[is_breach]
        mean_breach = float(breach_amounts.mean()) if n_breaches > 0 else 0.0
        max_breach = float(breach_amounts.max()) if n_breaches > 0 else 0.0
        total_breach = float(breach_amounts.sum()) if n_breaches > 0 else 0.0

        # By-channel summary
        by_channel = self._by_channel_summary(df, is_breach, breach_amount)

        # Breach detail
        breach_df = df[is_breach].copy()
        breach_df["breach_amount"] = breach_amount[is_breach]
        cols = []
        if self.policy_id_col and self.policy_id_col in breach_df.columns:
            cols.append(self.policy_id_col)
        if self.channel_col and self.channel_col in breach_df.columns:
            cols.append(self.channel_col)
        cols += [self.renewal_price_col, self.nb_price_col, "breach_amount"]
        breach_detail = breach_df[[c for c in cols if c in breach_df.columns]].copy()

        return ENBPBreachReport(
            n_policies=n_policies,
            n_breaches=n_breaches,
            breach_rate=breach_rate,
            mean_breach_amount=mean_breach,
            max_breach_amount=max_breach,
            total_breach_amount=total_breach,
            by_channel=by_channel,
            breach_detail=breach_detail.reset_index(drop=True),
        )

    def _by_channel_summary(
        self,
        df: pd.DataFrame,
        is_breach: np.ndarray,
        breach_amount: np.ndarray,
    ) -> pd.DataFrame:
        """Breach summary by channel."""
        if self.channel_col is None or self.channel_col not in df.columns:
            return pd.DataFrame({
                "channel": ["all"],
                "n_policies": [len(df)],
                "n_breaches": [int(is_breach.sum())],
                "breach_rate": [is_breach.mean()],
                "mean_breach_amount": [float(breach_amount[is_breach].mean()) if is_breach.any() else 0.0],
            })

        channels = df[self.channel_col].values
        unique_channels = sorted(set(channels))
        rows = []
        for ch in unique_channels:
            mask = channels == ch
            is_breach_ch = is_breach[mask]
            breach_ch = breach_amount[mask]
            rows.append({
                "channel": ch,
                "n_policies": int(mask.sum()),
                "n_breaches": int(is_breach_ch.sum()),
                "breach_rate": float(is_breach_ch.mean()),
                "mean_breach_amount": float(breach_ch[is_breach_ch].mean()) if is_breach_ch.any() else 0.0,
                "total_breach_amount": float(breach_ch[is_breach_ch].sum()) if is_breach_ch.any() else 0.0,
            })

        return pd.DataFrame(rows)


def enbp_compliant(
    renewal_price: float,
    nb_equivalent_price: float,
    tolerance: float = 0.0,
) -> bool:
    """
    Check if a single renewal price is ENBP-compliant.

    Parameters
    ----------
    renewal_price : float
    nb_equivalent_price : float
    tolerance : float
        Permitted excess (£). Default 0.

    Returns
    -------
    bool
        True if renewal_price <= nb_equivalent_price + tolerance.
    """
    return float(renewal_price) <= float(nb_equivalent_price) + tolerance


def price_walking_report(
    data: DataFrameLike,
    renewal_price_col: str = "renewal_price",
    tenure_col: str = "tenure_years",
    channel_col: Optional[str] = "channel",
    nb_price_col: Optional[str] = None,
    n_tenure_bins: int = 6,
) -> pd.DataFrame:
    """
    Detect systematic tenure-based price patterns (price-walking diagnostic).

    Post-PS21/11, renewal prices must not systematically increase with tenure
    when controlling for risk. This function shows the mean renewal price (and
    price-to-ENBP ratio if nb_price_col is supplied) by tenure band.

    A rising price-vs-tenure pattern after controlling for channel is a signal
    of potential non-compliance. This is a diagnostic tool - not a definitive
    compliance assessment.

    Parameters
    ----------
    data : DataFrame
    renewal_price_col : str
    tenure_col : str
    channel_col : str, optional
    nb_price_col : str, optional
        If supplied, also shows mean(renewal_price / nb_price) by tenure band.
    n_tenure_bins : int
        Number of quantile bins for tenure. Default 6.

    Returns
    -------
    pd.DataFrame
        Tenure bands with mean prices and renewal-to-ENBP ratios.
    """
    df = _to_pandas(data)

    tenure_bins = pd.qcut(df[tenure_col], q=n_tenure_bins, duplicates="drop")
    df = df.copy()
    df["tenure_band"] = tenure_bins.astype(str)

    group_cols = ["tenure_band"]
    if channel_col and channel_col in df.columns:
        group_cols = [channel_col] + group_cols

    agg_dict = {
        "n_policies": (renewal_price_col, "count"),
        "mean_renewal_price": (renewal_price_col, "mean"),
        "median_renewal_price": (renewal_price_col, "median"),
    }

    if nb_price_col and nb_price_col in df.columns:
        df["price_to_enbp"] = df[renewal_price_col] / df[nb_price_col].clip(lower=1.0)
        agg_dict["mean_price_to_enbp"] = ("price_to_enbp", "mean")

    result = (
        df.groupby(group_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    return result
