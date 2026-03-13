"""
DemandCurve: Generate demand curves from fitted models or elasticity estimates.

A demand curve maps price → expected volume (or probability of purchase).
In insurance, "volume" is usually either:
- Expected conversions from N quotes: volume = N × P(buy | price)
- Expected renewals from M in-force policies: volume = M × P(renew | price)

The DemandCurve class takes a fitted ConversionModel or RetentionModel (or
a raw elasticity + base point) and generates the curve over a specified
price range.

Two functional forms are supported:

1. ``'log_linear'``: log(p) = α + ε × log(price), where p is conversion/renewal
   probability. This is the standard log-log demand model. Elasticity ε is
   constant (percentage demand change is fixed percentage of price change).

2. ``'semi_log'``: logit(p) = α + β × log(price). This uses the logistic link.
   Elasticity varies with the probability level - it is highest at p=0.5 and
   approaches zero near the boundaries. More appropriate for binary outcomes.

The model predictions take precedence over the parametric forms when a fitted
model is supplied. The parametric forms are used when only a point elasticity
estimate is available (e.g., from DML).

Integration with rate-optimiser:
The DemandCurve can export a callable compatible with rate-optimiser's
DemandModel interface. This is the standard handoff: insurance-demand fits
the model; rate-optimiser queries the curve at candidate prices.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._types import DataFrameLike
from .conversion import _to_pandas


FunctionalForm = Literal["log_linear", "semi_log", "model"]


class DemandCurve:
    """
    Demand curve: price → probability of purchase/renewal.

    Can be constructed from:
    1. A fitted ConversionModel or RetentionModel directly (recommended).
    2. An elasticity + base probability (useful when DML provides the
       elasticity estimate but you want a parametric extrapolation).
    3. A raw callable.

    Parameters
    ----------
    model : fitted model or callable, optional
        A fitted ConversionModel or RetentionModel. If supplied, the curve
        uses model.predict_proba() at each price point.
    elasticity : float, optional
        Price elasticity (d log P / d log price). Used with functional_form
        when no model is supplied. Should be negative.
    base_price : float, optional
        Reference price at which base_prob is observed. Used to anchor the
        parametric form.
    base_prob : float, optional
        Probability at base_price. Together with elasticity, determines the
        intercept of the parametric curve.
    functional_form : {'log_linear', 'semi_log', 'model'}
        Which functional form to use.
        - 'model': use model.predict_proba directly (requires model).
        - 'log_linear': constant elasticity log-log curve.
        - 'semi_log': logistic link with log price treatment.
        Default: 'model' if model supplied, 'semi_log' otherwise.
    price_col : str
        Column name for the quoted price when calling model.predict_proba.

    Examples
    --------
    From a fitted model:

    >>> from insurance_demand import ConversionModel, DemandCurve
    >>> from insurance_demand.datasets import generate_conversion_data
    >>> df = generate_conversion_data(n_quotes=50_000)
    >>> conv_model = ConversionModel(...)
    >>> conv_model.fit(df)
    >>> curve = DemandCurve(model=conv_model, price_col='quoted_price')
    >>> prices, probs = curve.evaluate(price_range=(300, 800), n_points=50)

    From elasticity estimate only:

    >>> curve = DemandCurve(
    ...     elasticity=-2.0,
    ...     base_price=500.0,
    ...     base_prob=0.12,
    ...     functional_form='semi_log',
    ... )
    >>> prices, probs = curve.evaluate(price_range=(300, 800), n_points=50)
    """

    def __init__(
        self,
        model=None,
        elasticity: Optional[float] = None,
        base_price: Optional[float] = None,
        base_prob: Optional[float] = None,
        functional_form: FunctionalForm = "model",
        price_col: str = "quoted_price",
    ) -> None:
        self.model = model
        self.elasticity = elasticity
        self.base_price = base_price
        self.base_prob = base_prob
        self.price_col = price_col

        # Resolve functional form
        if functional_form == "model" and model is None:
            if elasticity is not None:
                self.functional_form = "semi_log"
            else:
                raise ValueError(
                    "Either model or elasticity must be supplied."
                )
        else:
            self.functional_form = functional_form

        # Validate parametric mode inputs
        if self.functional_form in ("log_linear", "semi_log"):
            if elasticity is None:
                raise ValueError(
                    f"elasticity must be supplied for functional_form='{self.functional_form}'."
                )
            if base_price is None or base_prob is None:
                raise ValueError(
                    "base_price and base_prob must be supplied for parametric functional forms."
                )
            self._fit_parametric()

    def _fit_parametric(self) -> None:
        """Compute intercept from base point and elasticity."""
        log_p0 = np.log(self.base_price)
        if self.functional_form == "log_linear":
            # log(prob) = alpha + epsilon * log(price)
            # => alpha = log(base_prob) - epsilon * log(base_price)
            self._alpha = np.log(np.clip(self.base_prob, 1e-8, 1 - 1e-8)) - self.elasticity * log_p0
        else:
            # logit(prob) = alpha + beta * log(price)
            # beta is not the elasticity directly; convert:
            # d log(p) / d log(price) = beta * p * (1-p) / p = beta * (1-p)
            # At base point: epsilon = beta * (1 - base_prob)
            # => beta = epsilon / (1 - base_prob)
            self._beta = self.elasticity / (1 - self.base_prob)
            logit_p0 = np.log(self.base_prob / (1 - self.base_prob))
            self._alpha = logit_p0 - self._beta * log_p0

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        price_range: tuple[float, float],
        n_points: int = 100,
        context: Optional[DataFrameLike] = None,
        aggregation: str = "mean",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the demand curve over a price range.

        Parameters
        ----------
        price_range : (min_price, max_price)
            Price range to evaluate over.
        n_points : int
            Number of equally-spaced price points. Default 100.
        context : DataFrame, optional
            For model-based evaluation: a reference portfolio of policies.
            The model is evaluated at each price for each policy, then
            aggregated. If None, uses parametric form.
        aggregation : str
            How to aggregate across policies when context is supplied.
            'mean' (default) or 'sum'.

        Returns
        -------
        prices : np.ndarray
            Shape (n_points,). Price grid.
        probabilities : np.ndarray
            Shape (n_points,). Demand probability at each price.
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)

        if self.functional_form == "model" and self.model is not None and context is not None:
            probs = self._evaluate_model(prices, context, aggregation)
        else:
            probs = self._evaluate_parametric(prices)

        return prices, probs

    def _evaluate_model(
        self,
        prices: np.ndarray,
        context: DataFrameLike,
        aggregation: str,
    ) -> np.ndarray:
        """Evaluate model at each price, averaging over context portfolio."""
        df = _to_pandas(context)
        result = []
        for price in prices:
            df_p = df.copy()
            df_p[self.price_col] = price
            probs = self.model.predict_proba(df_p).values
            if aggregation == "mean":
                result.append(probs.mean())
            else:
                result.append(probs.sum())
        return np.array(result)

    def _evaluate_parametric(self, prices: np.ndarray) -> np.ndarray:
        """Evaluate parametric demand curve at given prices."""
        log_prices = np.log(prices)
        if self.functional_form == "log_linear":
            log_probs = self._alpha + self.elasticity * log_prices
            return np.exp(log_probs)
        else:
            logit_probs = self._alpha + self._beta * log_prices
            return 1.0 / (1.0 + np.exp(-logit_probs))

    # ------------------------------------------------------------------
    # Optimal price finding
    # ------------------------------------------------------------------

    def price_at_prob(self, target_prob: float) -> float:
        """
        Invert the demand curve: find the price that gives the target probability.

        Only available for parametric functional forms.

        Parameters
        ----------
        target_prob : float
            Target conversion/renewal probability.

        Returns
        -------
        float
            Price that achieves the target probability.
        """
        if self.functional_form == "model":
            raise NotImplementedError(
                "price_at_prob is not available for model-based curves. "
                "Use evaluate() and interpolate instead."
            )

        if self.functional_form == "log_linear":
            log_prob = np.log(np.clip(target_prob, 1e-8, 1 - 1e-8))
            log_price = (log_prob - self._alpha) / self.elasticity
        else:
            logit_prob = np.log(target_prob / (1 - target_prob))
            log_price = (logit_prob - self._alpha) / self._beta

        return float(np.exp(log_price))

    # ------------------------------------------------------------------
    # Rate-optimiser integration
    # ------------------------------------------------------------------

    def as_demand_callable(
        self,
        reference_data: Optional[DataFrameLike] = None,
        tech_premium_col: str = "technical_premium",
    ) -> Callable:
        """
        Export as a callable compatible with rate-optimiser's DemandModel.

        The returned function has signature:
        ``f(price_ratio: np.ndarray) -> np.ndarray``

        where price_ratio = quoted_price / technical_premium.

        Parameters
        ----------
        reference_data : DataFrame, optional
            Reference portfolio. Required for model-based curves.
        tech_premium_col : str
            Column for technical premium. Used to convert price_ratio to
            quoted price for model evaluation.

        Returns
        -------
        callable
        """
        curve = self  # closure

        if self.functional_form in ("log_linear", "semi_log"):
            def _fn(price_ratio: np.ndarray, **kwargs) -> np.ndarray:
                # Treat price_ratio as the "price" for parametric curves
                log_prices = np.log(np.clip(price_ratio, 0.1, 10.0))
                if curve.functional_form == "log_linear":
                    log_probs = curve._alpha + curve.elasticity * log_prices
                    return np.exp(log_probs)
                else:
                    logit_probs = curve._alpha + curve._beta * log_prices
                    return 1.0 / (1.0 + np.exp(-logit_probs))
            return _fn

        elif self.model is not None and reference_data is not None:
            ref_df = _to_pandas(reference_data)

            def _fn(price_ratio: np.ndarray, features=None) -> np.ndarray:
                if features is not None:
                    df = _to_pandas(features)
                else:
                    df = ref_df.copy()
                if tech_premium_col in df.columns:
                    df[curve.price_col] = price_ratio * df[tech_premium_col]
                else:
                    df[curve.price_col] = price_ratio
                return curve.model.predict_proba(df).values
            return _fn

        else:
            raise ValueError(
                "Cannot create demand callable: need either a parametric form "
                "(elasticity + base_price + base_prob) or a fitted model + reference_data."
            )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        price_range: tuple[float, float],
        n_points: int = 100,
        context: Optional[DataFrameLike] = None,
        ax=None,
        title: str = "Demand Curve",
        xlabel: str = "Price (£)",
        ylabel: str = "P(purchase)",
    ):
        """
        Plot the demand curve.

        Parameters
        ----------
        price_range : tuple
        n_points : int
        context : DataFrame, optional
        ax : matplotlib Axes, optional
        title, xlabel, ylabel : str

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: uv pip install insurance-demand[plot]"
            )

        prices, probs = self.evaluate(price_range, n_points, context)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        ax.plot(prices, probs, "b-", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Mark base point if parametric
        if self.base_price is not None and self.base_prob is not None:
            ax.scatter([self.base_price], [self.base_prob],
                      color="red", zorder=5, s=80, label="Base point")
            ax.legend()

        return ax
