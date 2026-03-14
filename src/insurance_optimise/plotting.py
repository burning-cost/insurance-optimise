"""
Plotting utilities for insurance portfolio optimisation outputs.

All functions return matplotlib Axes objects so callers can further customise.
No inline plt.show() calls — that is the caller's responsibility.

matplotlib is an optional dependency. Install it with:
    pip install matplotlib
or:
    pip install insurance-optimise[plot]

Functions
---------
plot_frontier(result, ...)
    Plot profit vs retention (or any two metrics) from an EfficientFrontierResult.
plot_factor_adjustments(multipliers, ...)
    Horizontal bar chart of per-policy or per-segment price adjustments.
plot_shadow_prices(result, ...)
    How the shadow price on the swept constraint varies across the frontier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import matplotlib.axes


def plot_frontier(
    result: "EfficientFrontierResult",
    x_metric: str = "retention",
    y_metric: str = "profit",
    ax: "matplotlib.axes.Axes | None" = None,
    feasible_color: str = "#1f77b4",
    infeasible_color: str = "#d62728",
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
) -> "matplotlib.axes.Axes":
    """
    Plot the efficient frontier from an EfficientFrontierResult.

    Converged points are plotted as a connected line in blue; non-converged
    points (solver failures or infeasible problems) are scattered in red.

    Parameters
    ----------
    result : EfficientFrontierResult
        Output of EfficientFrontier.run(). Contains a ``.data`` DataFrame with
        columns: epsilon, converged, profit, gwp, loss_ratio, retention.
    x_metric : str
        Column from result.data to use on the x-axis. Default ``'retention'``.
        Choose from: ``'retention'``, ``'gwp'``, ``'loss_ratio'``, ``'epsilon'``,
        ``'profit'``.
    y_metric : str
        Column for the y-axis. Default ``'profit'``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. Creates a new figure if not supplied.
    feasible_color : str
        Colour for converged frontier points. Default blue.
    infeasible_color : str
        Colour for non-converged points. Default red.
    figsize : tuple
        Figure size (width, height) in inches. Only used when ax is None.
    title : str, optional
        Plot title. Auto-generated from metric names if not supplied.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    data = result.data
    feasible = data.filter(pl.col("converged"))
    infeasible = data.filter(~pl.col("converged"))

    if len(feasible) > 0:
        ax.plot(
            feasible[x_metric].to_numpy(),
            feasible[y_metric].to_numpy(),
            "-o",
            color=feasible_color,
            linewidth=2,
            markersize=6,
            label="Converged",
            zorder=3,
        )

    if len(infeasible) > 0:
        ax.scatter(
            infeasible[x_metric].to_numpy(),
            infeasible[y_metric].to_numpy(),
            marker="x",
            color=infeasible_color,
            s=50,
            linewidths=1.5,
            label="Not converged",
            zorder=2,
        )

    xlabel = x_metric.replace("_", " ").title()
    ylabel = y_metric.replace("_", " ").title()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Efficient Frontier: {ylabel} vs {xlabel}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return ax


def plot_factor_adjustments(
    multipliers: np.ndarray,
    labels: list[str] | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    color_increase: str = "#d62728",
    color_decrease: str = "#1f77b4",
    figsize: tuple[float, float] | None = None,
    title: str = "Price Adjustments vs Technical Price",
) -> "matplotlib.axes.Axes":
    """
    Horizontal bar chart of price multipliers relative to 1.0 (no change).

    Designed for showing the per-policy or per-segment output of
    PortfolioOptimiser. Values above 1.0 (increases) are shown in red;
    values below 1.0 (reductions) in blue.

    Parameters
    ----------
    multipliers : np.ndarray
        Array of price multipliers, shape (N,). Output from
        ``OptimisationResult.multipliers``.
    labels : list of str, optional
        Labels for each element. E.g. segment names or policy indices.
        Defaults to integers 0..N-1 if not supplied.
    ax : matplotlib.axes.Axes, optional
        Existing axes. Creates a new figure if not supplied.
    color_increase : str
        Bar colour for multipliers > 1.0 (rate increase). Default red.
    color_decrease : str
        Bar colour for multipliers <= 1.0 (rate reduction or hold). Default blue.
    figsize : tuple, optional
        Figure size. Defaults to (7, max(3, N * 0.3)).
    title : str
        Chart title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    multipliers = np.asarray(multipliers, dtype=float)
    n = len(multipliers)

    if labels is None:
        labels = [str(i) for i in range(n)]

    # Show deviation from 1.0 (no change)
    deviations = multipliers - 1.0
    colours = [color_increase if v > 0 else color_decrease for v in deviations]

    if figsize is None:
        figsize = (7, max(3, n * 0.3))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(labels, deviations, color=colours, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Annotate bars if there are few enough to read
    if n <= 30:
        for bar, val in zip(bars, deviations):
            x = bar.get_width()
            ax.text(
                x + (0.001 if x >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1%}",
                va="center",
                ha="left" if x >= 0 else "right",
                fontsize=7,
            )

    ax.set_xlabel("Adjustment relative to technical price (0 = no change)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    return ax


def plot_shadow_prices(
    result: "EfficientFrontierResult",
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (7, 4),
    title: str | None = None,
) -> "matplotlib.axes.Axes":
    """
    Plot how shadow prices on the swept constraint vary across the frontier.

    The shadow price tells you: at this constraint level, what is the marginal
    cost (in profit units) of tightening the constraint by one unit? A steeply
    rising shadow price signals a hard part of the frontier — further improvement
    becomes increasingly costly.

    This requires that OptimisationResult.shadow_prices was populated, which
    happens automatically when the constraint is active in ConstraintConfig.

    Parameters
    ----------
    result : EfficientFrontierResult
        Output of EfficientFrontier.run(). The first constraint's shadow price
        is extracted from each frontier point's OptimisationResult.
    ax : matplotlib.axes.Axes, optional
        Existing axes. Creates new figure if not supplied.
    figsize : tuple
        Figure size. Only used when ax is None.
    title : str, optional
        Plot title. Auto-generated if not supplied.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Extract shadow prices from each frontier point
    epsilons = []
    shadow_vals = []

    for point in result.points:
        r = point.result
        if r.converged and r.shadow_prices:
            # Take the first shadow price (the swept constraint)
            prices = r.shadow_prices
            if prices:
                key = next(iter(prices))
                epsilons.append(point.epsilon)
                shadow_vals.append(abs(prices[key]))

    if not epsilons:
        ax.text(
            0.5,
            0.5,
            "No shadow price data available.\n"
            "Ensure constraints are active and the solver converged.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return ax

    ax.plot(
        epsilons,
        shadow_vals,
        "-o",
        color="#ff7f0e",
        linewidth=2,
        markersize=5,
    )
    ax.set_xlabel("Constraint level (epsilon)")
    ax.set_ylabel("|Shadow price| on swept constraint")
    ax.set_title(title or "Shadow Price Across Frontier")
    ax.grid(True, alpha=0.3)

    return ax
