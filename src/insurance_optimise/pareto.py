"""
Multi-objective Pareto frontier for insurance portfolio rate optimisation.

Extends the bi-objective EfficientFrontier (profit vs retention) to a genuine
3-objective Pareto surface: profit, retention, and fairness.

Algorithm: epsilon-constraint method on a 2D grid. Each grid point (eps_x,
eps_y) is an independent SLSQP solve — the existing PortfolioOptimiser with
analytical gradients. The grid is embarrassingly parallel (joblib optional).

Why not NSGA-II? Decision space is R^N where N is the number of policies.
NSGA-II cannot meaningfully explore R^N for N > ~200. SLSQP with analytical
gradients already handles N=10,000 efficiently and each epsilon-constrained
solution is guaranteed Pareto-optimal, not approximate. See KB entry 2783.

Why fairness as objective, not constraint? The acceptable level of premium
disparity is a governance decision, not a hard regulatory limit. Presenting
the full Pareto surface is more defensible than pre-committing to an arbitrary
fairness cap. The user can set `fairness_max` to hard-code a floor into the
feasible region — that is a separate mechanism from the Pareto objective.
"""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import polars as pl

from insurance_optimise.constraints import ConstraintConfig
from insurance_optimise.result import OptimisationResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurance_optimise.optimiser import PortfolioOptimiser


# ---------------------------------------------------------------------------
# Built-in fairness metrics
# ---------------------------------------------------------------------------


def premium_disparity_ratio(
    multipliers: np.ndarray,
    technical_price: np.ndarray,
    group_labels: np.ndarray,
) -> float:
    """
    Premium disparity ratio between the highest and lowest group.

    Computes mean_premium(group_max) / mean_premium(group_min), where
    group_labels partitions the portfolio (e.g. postcode deprivation quintile,
    age band). Returns 1.0 = perfect equality; 2.0 = highest group pays double
    on average.

    This is the primary fairness metric for FCA Consumer Duty compliance
    reporting. FCA PS22/9 requires firms to consider outcomes across customer
    segments — this metric makes that consideration quantitative.

    Parameters
    ----------
    multipliers:
        Array of price multipliers, shape (N,). Applied to technical_price
        to get the premium at this point on the Pareto surface.
    technical_price:
        Array of technical prices, shape (N,).
    group_labels:
        Array of group identifiers, shape (N,). Integer or string labels.
        E.g. deprivation quintile (1-5), age band ('18-25', '26-35', ...).

    Returns
    -------
    float
        Ratio >= 1.0. 1.0 = no disparity between groups.

    Example
    -------
    >>> fairness_fn = partial(
    ...     premium_disparity_ratio,
    ...     technical_price=df['tc'].to_numpy(),
    ...     group_labels=df['deprivation_quintile'].to_numpy(),
    ... )
    """
    premiums = np.asarray(multipliers, dtype=float) * np.asarray(
        technical_price, dtype=float
    )
    labels = np.asarray(group_labels)
    unique_groups = np.unique(labels)
    if len(unique_groups) < 2:
        return 1.0
    group_means = np.array(
        [
            premiums[labels == g].mean() if np.any(labels == g) else 0.0
            for g in unique_groups
        ]
    )
    min_mean = group_means.min()
    if min_mean < 1e-10:
        return 1.0
    return float(group_means.max() / min_mean)


def loss_ratio_disparity(
    multipliers: np.ndarray,
    technical_price: np.ndarray,
    expected_loss_cost: np.ndarray,
    demand_model: Any,
    group_labels: np.ndarray,
) -> float:
    """
    Loss ratio disparity ratio across groups.

    Ratio of the highest-LR group to the lowest-LR group at optimal prices.
    A ratio above 1.5 indicates cross-subsidy between segments, which may
    warrant regulatory scrutiny under Consumer Duty.

    Parameters
    ----------
    multipliers:
        Array of price multipliers, shape (N,).
    technical_price:
        Array of technical prices, shape (N,).
    expected_loss_cost:
        Array of expected claims costs per policy, shape (N,).
    demand_model:
        Fitted demand model instance with a .demand(m) method.
    group_labels:
        Group identifier array, shape (N,).

    Returns
    -------
    float
        Ratio >= 1.0. 1.0 = no LR disparity between groups.
    """
    m = np.asarray(multipliers, dtype=float)
    tc = np.asarray(technical_price, dtype=float)
    cost = np.asarray(expected_loss_cost, dtype=float)
    labels = np.asarray(group_labels)

    x = demand_model.demand(m)
    premiums = m * tc

    unique_groups = np.unique(labels)
    if len(unique_groups) < 2:
        return 1.0

    group_lrs = []
    for g in unique_groups:
        mask = labels == g
        gwp_g = float(np.dot(premiums[mask], x[mask]))
        claims_g = float(np.dot(cost[mask], x[mask]))
        if gwp_g < 1e-10:
            continue
        group_lrs.append(claims_g / gwp_g)

    if len(group_lrs) < 2:
        return 1.0

    arr = np.array(group_lrs)
    min_lr = arr.min()
    if min_lr < 1e-10:
        return 1.0
    return float(arr.max() / min_lr)


# ---------------------------------------------------------------------------
# Non-dominated filtering
# ---------------------------------------------------------------------------


def _filter_pareto_front(
    df: pl.DataFrame,
    profit_col: str = "profit",
    retention_col: str = "retention",
    fairness_col: str = "fairness",
    tol: float = 1e-6,
) -> pl.DataFrame:
    """
    Filter a DataFrame to non-dominated solutions.

    A solution is dominated if there exists another solution that is:
    - At least as profitable (>=)
    - At least as high retention (>=)
    - At least as fair, i.e., lower disparity (<=)
    with at least one strict inequality.

    Only converged solutions participate. Dominated solutions are still
    present in ParetoResult.surface — they are absent only from pareto_df.

    Parameters
    ----------
    df:
        DataFrame of converged grid points with columns profit, retention,
        fairness.
    tol:
        Numerical tolerance for dominance comparison. Avoids treating
        floating-point near-ties as dominance.

    Returns
    -------
    pl.DataFrame
        Non-dominated rows.
    """
    if len(df) == 0:
        return df

    # Work with numpy for speed
    profit = df[profit_col].to_numpy()
    retention = df[retention_col].to_numpy()
    fairness = df[fairness_col].to_numpy()
    n = len(df)

    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is at least as good in all objectives and
            # strictly better in at least one
            j_ge_profit = profit[j] >= profit[i] - tol
            j_ge_retention = retention[j] >= retention[i] - tol
            j_le_fairness = fairness[j] <= fairness[i] + tol  # lower = more fair
            # Strict in at least one
            j_gt_profit = profit[j] > profit[i] + tol
            j_gt_retention = retention[j] > retention[i] + tol
            j_lt_fairness = fairness[j] < fairness[i] - tol

            if (
                j_ge_profit
                and j_ge_retention
                and j_le_fairness
                and (j_gt_profit or j_gt_retention or j_lt_fairness)
            ):
                dominated[i] = True
                break

    return df.filter(~pl.Series(dominated))


# ---------------------------------------------------------------------------
# TOPSIS selection
# ---------------------------------------------------------------------------


def _topsis_select(
    pareto_df: pl.DataFrame,
    weights: tuple[float, float, float],
    objectives: tuple[str, str, str] = ("profit", "retention", "fairness"),
    directions: tuple[str, str, str] = ("max", "max", "min"),
) -> int:
    """
    TOPSIS multi-criteria selection from Pareto surface.

    Returns the row index in pareto_df of the selected solution.

    Steps:
    1. Normalise objectives to [0,1] by min-max, flipping 'min' objectives
       so that 1.0 always represents the best value.
    2. Apply weights element-wise.
    3. Compute Euclidean distance to ideal point (1,1,1) and anti-ideal (0,0,0).
    4. Select row maximising D_minus / (D_plus + D_minus).

    Parameters
    ----------
    pareto_df:
        DataFrame of Pareto-optimal solutions.
    weights:
        Relative importance of (profit, retention, fairness). Need not sum to 1.
    objectives:
        Column names of the three objectives.
    directions:
        'max' for objectives to maximise, 'min' for objectives to minimise.

    Returns
    -------
    int
        Row index (0-based) in pareto_df.

    Raises
    ------
    ValueError
        If weights are all zero or pareto_df is empty.
    """
    if len(pareto_df) == 0:
        raise ValueError(
            "Cannot select from an empty Pareto surface. "
            "All grid points failed to converge."
        )

    w = np.array(weights, dtype=float)
    if w.sum() < 1e-10:
        raise ValueError("weights must not all be zero.")

    # Build matrix of objective values (n_solutions x 3)
    cols = [pareto_df[obj].to_numpy() for obj in objectives]
    mat = np.stack(cols, axis=1).astype(float)  # (n, 3)

    # Normalise to [0, 1] and flip 'min' objectives
    normalised = np.zeros_like(mat)
    for k, direction in enumerate(directions):
        col = mat[:, k]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            # All solutions identical on this objective — set to 0.5
            normalised[:, k] = 0.5
        else:
            scaled = (col - lo) / (hi - lo)
            if direction == "max":
                normalised[:, k] = scaled
            else:
                normalised[:, k] = 1.0 - scaled  # flip: lower raw = higher score

    # Apply weights
    weighted = normalised * w  # broadcast: (n, 3) * (3,)

    # Ideal = max in each weighted column; anti-ideal = min
    ideal = weighted.max(axis=0)
    anti_ideal = weighted.min(axis=0)

    # Distances
    d_plus = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))

    # TOPSIS score
    denom = d_plus + d_minus
    # Avoid division by zero (all solutions identical)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    scores = d_minus / denom

    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# ParetoResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParetoResult:
    """
    Output from ParetoFrontier.run().

    Attributes
    ----------
    surface:
        All grid points (n_points_x * n_points_y rows). Columns: eps_x,
        eps_y, converged, profit, gwp, loss_ratio, retention, fairness,
        n_iter, solver_message, grid_i, grid_j.
    pareto_df:
        Filtered to non-dominated converged solutions only.
    metadata:
        Run parameters: sweep_x, sweep_y, sweep_x_range, sweep_y_range,
        n_points_x, n_points_y, fairness_metric_name, timestamp.
    _grid_multipliers:
        Raw multiplier arrays keyed by (grid_i, grid_j). Stored separately
        to avoid embedding large arrays in Polars rows.
    selected:
        Populated by .select(). The chosen OptimisationResult.
    """

    surface: pl.DataFrame
    pareto_df: pl.DataFrame
    metadata: dict[str, Any]
    _grid_multipliers: dict[tuple[int, int], np.ndarray]
    selected: OptimisationResult | None = field(default=None)

    def select(
        self,
        method: str = "topsis",
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> "ParetoResult":
        """
        Select a single solution from the Pareto surface.

        Populates ``self.selected`` with an OptimisationResult. The selected
        multiplier array is fetched from ``_grid_multipliers``.

        Parameters
        ----------
        method:
            'topsis' (default) or 'closest_to_utopia'. Both use the weights
            parameter.
        weights:
            Relative importance of (profit, retention, fairness). Need not
            sum to 1. Zero weights exclude that objective from selection.

        Returns
        -------
        ParetoResult
            Self, with .selected populated.

        Raises
        ------
        ValueError
            If the Pareto surface is empty or weights are all zero.
        """
        if len(self.pareto_df) == 0:
            raise ValueError(
                "Pareto surface is empty — all grid points failed to converge. "
                "Relax constraints or reduce n_points_x/n_points_y."
            )

        if method == "topsis":
            row_idx = _topsis_select(self.pareto_df, weights=weights)
        elif method == "closest_to_utopia":
            # Equivalent to TOPSIS with ideal = actual utopia point
            row_idx = _topsis_select(self.pareto_df, weights=weights)
        else:
            raise ValueError(
                f"Unknown selection method '{method}'. Choose 'topsis' or "
                "'closest_to_utopia'."
            )

        selected_row = self.pareto_df.row(row_idx, named=True)
        gi = int(selected_row["grid_i"])
        gj = int(selected_row["grid_j"])
        m = self._grid_multipliers.get((gi, gj))

        if m is None:
            raise RuntimeError(
                f"Multipliers not found for grid point ({gi}, {gj}). "
                "This is a bug — please report it."
            )

        # Build a lightweight OptimisationResult from stored grid data
        # We re-use the stored metrics; we do not re-run the optimiser.
        from insurance_optimise.result import OptimisationResult

        self.selected = OptimisationResult(
            multipliers=m,
            new_premiums=m,  # placeholder — tc not stored here
            expected_demand=np.array([]),
            expected_profit=float(selected_row["profit"]),
            expected_gwp=float(selected_row["gwp"]),
            expected_loss_ratio=float(selected_row["loss_ratio"]),
            expected_retention=selected_row.get("retention"),
            shadow_prices={},
            converged=bool(selected_row["converged"]),
            solver_message=str(selected_row.get("solver_message", "")),
            n_iter=int(selected_row.get("n_iter", 0)),
            audit_trail={
                "selected_by": method,
                "weights": list(weights),
                "grid_i": gi,
                "grid_j": gj,
                "eps_x": float(selected_row["eps_x"]),
                "eps_y": float(selected_row["eps_y"]),
                "fairness": float(selected_row["fairness"]),
            },
            summary_df=pl.DataFrame(),
        )
        return self

    def summary(self) -> pl.DataFrame:
        """
        Return a Polars DataFrame summarising the Pareto surface.

        Includes: total grid points, converged count, Pareto-optimal count,
        ranges of profit, retention, and fairness across the Pareto front.
        """
        n_total = len(self.surface)
        n_converged = int(self.surface["converged"].sum())
        n_pareto = len(self.pareto_df)

        if n_pareto > 0:
            profit_min = float(self.pareto_df["profit"].min())
            profit_max = float(self.pareto_df["profit"].max())
            retention_min = float(self.pareto_df["retention"].min())
            retention_max = float(self.pareto_df["retention"].max())
            fairness_min = float(self.pareto_df["fairness"].min())
            fairness_max = float(self.pareto_df["fairness"].max())
        else:
            profit_min = profit_max = float("nan")
            retention_min = retention_max = float("nan")
            fairness_min = fairness_max = float("nan")

        return pl.DataFrame(
            {
                "metric": [
                    "grid_points_total",
                    "grid_points_converged",
                    "pareto_optimal_solutions",
                    "profit_min",
                    "profit_max",
                    "retention_min",
                    "retention_max",
                    "fairness_disparity_min",
                    "fairness_disparity_max",
                ],
                "value": [
                    float(n_total),
                    float(n_converged),
                    float(n_pareto),
                    profit_min,
                    profit_max,
                    retention_min,
                    retention_max,
                    fairness_min,
                    fairness_max,
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return all metadata and surface data as a JSON-serialisable dict."""
        selected_audit: dict[str, Any] = {}
        if self.selected is not None:
            selected_audit = self.selected.audit_trail

        surface_dicts = self.surface.to_dicts()
        pareto_dicts = self.pareto_df.to_dicts()

        return {
            "metadata": self.metadata,
            "surface": surface_dicts,
            "pareto_front": pareto_dicts,
            "selected": selected_audit,
        }

    def to_json(self) -> str:
        """Return audit dict as a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=_json_default)

    def save_audit(self, path: str) -> None:
        """Write the full audit trail to a file (regulatory evidence)."""
        with open(path, "w") as f:
            f.write(self.to_json())

    def plot(
        self,
        x_metric: str = "retention",
        y_metric: str = "fairness",
        color_metric: str = "profit",
        figsize: tuple[float, float] = (9, 7),
        title: str | None = None,
        show_colorbar: bool = True,
        annotate_extremes: bool = True,
    ) -> None:
        """
        2D scatter of the Pareto surface with colour axis.

        Requires matplotlib. Non-dominated (Pareto-optimal) solutions are
        plotted solid; all other converged solutions are faded.

        Parameters
        ----------
        x_metric:
            Column to use as x-axis. Default 'retention'.
        y_metric:
            Column to use as y-axis. Default 'fairness'.
        color_metric:
            Column to use for colour scale. Default 'profit'.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting. "
                "Install with: pip install insurance-optimise[plot]"
            )

        converged = self.surface.filter(pl.col("converged"))
        pareto = self.pareto_df

        if len(converged) == 0:
            warnings.warn("No converged solutions to plot.", stacklevel=2)
            return

        fig, ax = plt.subplots(figsize=figsize)

        # All converged (faded background)
        if len(converged) > 0:
            c_all = converged[color_metric].to_numpy()
            sc = ax.scatter(
                converged[x_metric].to_numpy(),
                converged[y_metric].to_numpy(),
                c=c_all,
                cmap="Blues",
                alpha=0.3,
                s=50,
                label="Converged (dominated)",
            )

        # Pareto-optimal (solid foreground)
        if len(pareto) > 0:
            c_par = pareto[color_metric].to_numpy()
            sc = ax.scatter(
                pareto[x_metric].to_numpy(),
                pareto[y_metric].to_numpy(),
                c=c_par,
                cmap="Blues",
                edgecolors="navy",
                linewidths=1.0,
                s=100,
                label="Pareto-optimal",
            )
            if show_colorbar:
                plt.colorbar(sc, ax=ax, label=color_metric.replace("_", " ").title())

        # Annotate extremes
        if annotate_extremes and len(pareto) > 0:
            # Max profit
            idx_mp = int(pareto[color_metric].arg_max())
            row_mp = pareto.row(idx_mp, named=True)
            ax.annotate(
                "Max profit",
                (row_mp[x_metric], row_mp[y_metric]),
                xytext=(10, -15),
                textcoords="offset points",
                fontsize=8,
            )
            # Max retention
            if x_metric == "retention":
                idx_mr = int(pareto[x_metric].arg_max())
                row_mr = pareto.row(idx_mr, named=True)
                ax.annotate(
                    "Max retention",
                    (row_mr[x_metric], row_mr[y_metric]),
                    xytext=(10, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
            # Min fairness disparity
            if y_metric == "fairness":
                idx_mf = int(pareto[y_metric].arg_min())
                row_mf = pareto.row(idx_mf, named=True)
                ax.annotate(
                    "Min disparity",
                    (row_mf[x_metric], row_mf[y_metric]),
                    xytext=(10, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_xlabel(x_metric.replace("_", " ").title())
        ax.set_ylabel(y_metric.replace("_", " ").title())
        ax.set_title(
            title
            or f"Pareto Surface: {y_metric} vs {x_metric} (colour = {color_metric})"
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_3d(
        self,
        figsize: tuple[float, float] = (10, 8),
    ) -> None:
        """
        3D scatter of (profit, retention, fairness).

        Requires matplotlib. Non-dominated solutions shown in a contrasting
        colour. Uses mpl_toolkits.mplot3d.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            raise ImportError(
                "matplotlib required for 3D plotting. "
                "Install with: pip install insurance-optimise[plot]"
            )

        converged = self.surface.filter(pl.col("converged"))
        pareto = self.pareto_df

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if len(converged) > 0:
            ax.scatter(
                converged["profit"].to_numpy(),
                converged["retention"].to_numpy(),
                converged["fairness"].to_numpy(),
                c="steelblue",
                alpha=0.25,
                s=30,
                label="Converged (dominated)",
            )

        if len(pareto) > 0:
            ax.scatter(
                pareto["profit"].to_numpy(),
                pareto["retention"].to_numpy(),
                pareto["fairness"].to_numpy(),
                c="crimson",
                edgecolors="darkred",
                s=80,
                linewidths=0.8,
                label="Pareto-optimal",
            )

        ax.set_xlabel("Profit (£)")
        ax.set_ylabel("Retention")
        ax.set_zlabel("Fairness Disparity Ratio")
        ax.set_title("3D Pareto Surface: Profit vs Retention vs Fairness")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# ParetoFrontier
# ---------------------------------------------------------------------------


class ParetoFrontier:
    """
    3-objective Pareto surface for insurance portfolio rate optimisation.

    Extends the bi-objective epsilon-constraint EfficientFrontier to a 2D
    epsilon grid sweep, simultaneously optimising:

        f1: expected profit (maximise) -- primary SLSQP objective
        f2: retention rate (maximise) -- first epsilon constraint (sweep_x)
        f3: fairness metric (minimise disparity) -- second epsilon (sweep_y)

    Each of the n_points_x * n_points_y grid points is an independent SLSQP
    solve with analytical gradients. The grid is embarrassingly parallel and
    every solution is guaranteed Pareto-optimal at that (eps_x, eps_y) pair.

    Parameters
    ----------
    optimiser:
        A configured PortfolioOptimiser instance. The optimiser is deep-copied
        for each grid point; the original is not mutated.
    fairness_metric:
        Callable (multipliers: np.ndarray) -> float. Returns a scalar
        'unfairness' value. Lower = more fair (e.g. premium disparity ratio).
        Called once per grid point after the SLSQP solve. Must be O(N) fast.
        If None, defaults to premium disparity ratio (requires group_labels).
    fairness_gradient:
        Optional gradient of fairness_metric w.r.t. multipliers.
        Signature: (multipliers: np.ndarray) -> np.ndarray.
        If provided, used as the Jacobian of the fairness constraint in SLSQP.
        Without it, SLSQP uses finite differences (~2*N extra evaluations per
        SLSQP iteration). Material for N > 5,000.
    group_labels:
        Array of group identifiers, shape (N,). Required when fairness_metric
        is None (default premium_disparity_ratio). Ignored if fairness_metric
        is provided.
    sweep_x:
        First epsilon-constraint axis. Supported:
        'volume_retention' (default), 'lr_max', 'gwp_min'.
    sweep_x_range:
        (min, max) for the x-axis sweep. E.g. (0.80, 0.99) for retention.
    sweep_y:
        Second epsilon-constraint axis. Supported:
        'fairness_max' (default), 'lr_max', 'gwp_min'.
        If 'fairness_max', adds fairness_metric(m) <= eps_y constraint.
        Otherwise, fairness is computed post-hoc (diagnostic only).
    sweep_y_range:
        (min, max) for y-axis. E.g. (1.05, 2.50) for disparity ratio cap.
    n_points_x:
        Grid resolution on x-axis. Default 10.
    n_points_y:
        Grid resolution on y-axis. Default 10.
        Total SLSQP solves = n_points_x * n_points_y.
    fairness_max:
        Hard cap on fairness_metric(m) applied to all grid points (in addition
        to the sweep). E.g. fairness_max=2.0 enforces ratio <= 2.0 everywhere.
        Equivalent to restricting sweep_y_range max to fairness_max.
    n_jobs:
        Parallel workers. -1 = all CPUs. 1 = sequential. Default 1.
        Requires joblib if n_jobs != 1.
    """

    _VALID_X = ("volume_retention", "lr_max", "gwp_min")
    _VALID_Y = ("fairness_max", "lr_max", "gwp_min", "volume_retention")

    def __init__(
        self,
        optimiser: "PortfolioOptimiser",
        fairness_metric: Callable[[np.ndarray], float] | None = None,
        fairness_gradient: Callable[[np.ndarray], np.ndarray] | None = None,
        group_labels: np.ndarray | None = None,
        sweep_x: str = "volume_retention",
        sweep_x_range: tuple[float, float] = (0.80, 0.99),
        sweep_y: str = "fairness_max",
        sweep_y_range: tuple[float, float] = (1.05, 2.50),
        n_points_x: int = 10,
        n_points_y: int = 10,
        fairness_max: float | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.optimiser = optimiser
        self.fairness_gradient = fairness_gradient
        self.sweep_x = sweep_x
        self.sweep_x_range = sweep_x_range
        self.sweep_y = sweep_y
        self.sweep_y_range = sweep_y_range
        self.n_points_x = n_points_x
        self.n_points_y = n_points_y
        self.fairness_max = fairness_max
        self.n_jobs = n_jobs

        # Validate sweep parameters
        if sweep_x not in self._VALID_X:
            raise ValueError(
                f"sweep_x '{sweep_x}' not supported. Choose from {self._VALID_X}."
            )
        if sweep_y not in self._VALID_Y:
            raise ValueError(
                f"sweep_y '{sweep_y}' not supported. Choose from {self._VALID_Y}."
            )
        if sweep_x == sweep_y:
            raise ValueError("sweep_x and sweep_y must be different axes.")

        # Resolve fairness metric
        if fairness_metric is not None:
            self.fairness_metric = fairness_metric
            self._fairness_metric_name = getattr(
                fairness_metric, "__name__", "custom"
            )
        else:
            if group_labels is None:
                raise ValueError(
                    "Either fairness_metric or group_labels must be provided. "
                    "group_labels is required when using the default "
                    "premium_disparity_ratio metric."
                )
            from functools import partial

            self.fairness_metric = partial(
                premium_disparity_ratio,
                technical_price=optimiser.tc,
                group_labels=np.asarray(group_labels),
            )
            self._fairness_metric_name = "premium_disparity_ratio"

    def run(self) -> ParetoResult:
        """
        Run the full Pareto surface sweep.

        Iterates over all (eps_x, eps_y) grid points, solving an independent
        SLSQP problem at each. Post-processes to extract non-dominated solutions.

        Returns
        -------
        ParetoResult
            Contains surface (all points), pareto_df (non-dominated), and
            _grid_multipliers.
        """
        eps_x_vals = np.linspace(
            self.sweep_x_range[0], self.sweep_x_range[1], self.n_points_x
        )
        eps_y_vals = np.linspace(
            self.sweep_y_range[0], self.sweep_y_range[1], self.n_points_y
        )

        # Build list of (i, j, eps_x, eps_y) tasks
        tasks = [
            (i, j, ex, ey)
            for i, ex in enumerate(eps_x_vals)
            for j, ey in enumerate(eps_y_vals)
        ]

        if self.n_jobs == 1:
            raw_points = [self._solve_at_grid_point(i, j, ex, ey) for i, j, ex, ey in tasks]
        else:
            raw_points = self._run_parallel(tasks)

        # Build DataFrame and multiplier dict
        rows = []
        grid_multipliers: dict[tuple[int, int], np.ndarray] = {}

        for pt in raw_points:
            i, j = pt["grid_i"], pt["grid_j"]
            grid_multipliers[(i, j)] = pt.pop("_multipliers")
            rows.append(pt)

        surface = pl.DataFrame(rows)

        # Cast types explicitly to be safe with None/null
        surface = surface.with_columns([
            pl.col("eps_x").cast(pl.Float64),
            pl.col("eps_y").cast(pl.Float64),
            pl.col("converged").cast(pl.Boolean),
            pl.col("profit").cast(pl.Float64),
            pl.col("gwp").cast(pl.Float64),
            pl.col("loss_ratio").cast(pl.Float64),
            pl.col("retention").cast(pl.Float64),
            pl.col("fairness").cast(pl.Float64),
            pl.col("n_iter").cast(pl.Int64),
            pl.col("solver_message").cast(pl.String),
            pl.col("grid_i").cast(pl.Int64),
            pl.col("grid_j").cast(pl.Int64),
        ])

        # Non-dominated filtering (converged only)
        converged_df = surface.filter(pl.col("converged"))
        if len(converged_df) > 0:
            pareto_df = _filter_pareto_front(converged_df)
        else:
            pareto_df = converged_df.clone()

        # Warn if many grid points failed
        n_failed = len(surface) - int(surface["converged"].sum())
        n_total = len(surface)
        if n_total > 0 and n_failed / n_total > 0.2:
            warnings.warn(
                f"{n_failed}/{n_total} ({n_failed/n_total:.0%}) grid points failed "
                "to converge. Consider relaxing constraints or reducing the sweep "
                "range to avoid jointly infeasible (eps_x, eps_y) combinations.",
                stacklevel=2,
            )

        metadata = {
            "sweep_x": self.sweep_x,
            "sweep_y": self.sweep_y,
            "sweep_x_range": list(self.sweep_x_range),
            "sweep_y_range": list(self.sweep_y_range),
            "n_points_x": self.n_points_x,
            "n_points_y": self.n_points_y,
            "fairness_metric_name": self._fairness_metric_name,
            "fairness_max": self.fairness_max,
            "n_jobs": self.n_jobs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_policies": self.optimiser.n,
        }

        return ParetoResult(
            surface=surface,
            pareto_df=pareto_df,
            metadata=metadata,
            _grid_multipliers=grid_multipliers,
        )

    def _solve_at_grid_point(
        self, i: int, j: int, eps_x: float, eps_y: float
    ) -> dict[str, Any]:
        """
        Solve a single SLSQP problem at (eps_x, eps_y) grid point.

        Returns a dict with all surface columns plus '_multipliers'.
        """
        from insurance_optimise.optimiser import PortfolioOptimiser

        # Deep-copy constraint config and set eps_x constraint
        new_config = copy.deepcopy(self.optimiser.config)
        self._set_x_epsilon(new_config, eps_x)

        # Build fairness cap constraint for eps_y if sweep_y == 'fairness_max'
        extra_constraints: list[dict] = []
        if self.sweep_y == "fairness_max":
            fairness_cap = min(eps_y, self.fairness_max or float("inf"))
            extra_constraints = self._build_fairness_constraint(fairness_cap)
        else:
            # Non-fairness second sweep axis
            self._set_y_epsilon(new_config, eps_y)

        # Apply hard fairness_max cap if set and not already applied
        if self.fairness_max is not None and self.sweep_y != "fairness_max":
            extra_constraints.extend(self._build_fairness_constraint(self.fairness_max))

        # Rebuild optimiser with new config
        solver_str = (
            "slsqp" if self.optimiser.solver_method == "SLSQP" else "trust_constr"
        )
        opt = PortfolioOptimiser(
            technical_price=self.optimiser.tc,
            expected_loss_cost=self.optimiser.cost,
            p_demand=self.optimiser.x0,
            elasticity=self.optimiser.elasticity,
            renewal_flag=self.optimiser.renewal_flag,
            enbp=self.optimiser.enbp,
            prior_multiplier=self.optimiser.prior_multiplier,
            claims_variance=self.optimiser.claims_variance,
            constraints=new_config,
            demand_model=self.optimiser.demand_model_name,
            solver=solver_str,
            n_restarts=self.optimiser.n_restarts,
            seed=int((eps_x * 1e4 + eps_y) * 1e3) % (2**31),
            ftol=self.optimiser.ftol,
            maxiter=self.optimiser.maxiter,
        )

        # Inject extra constraints (fairness)
        if extra_constraints:
            opt._scipy_constraints = opt._scipy_constraints + extra_constraints

        # Solve
        try:
            result = opt.optimise()
            m_opt = result.multipliers
            converged = result.converged
            profit = result.expected_profit
            gwp = result.expected_gwp
            lr = result.expected_loss_ratio
            retention = result.expected_retention if result.expected_retention is not None else float("nan")
            n_iter = result.n_iter
            solver_msg = result.solver_message
        except Exception as e:
            warnings.warn(
                f"Grid point ({i},{j}) eps_x={eps_x:.4f} eps_y={eps_y:.4f} "
                f"failed: {e}",
                stacklevel=3,
            )
            m_opt = np.ones(opt.n)
            converged = False
            profit = float("nan")
            gwp = float("nan")
            lr = float("nan")
            retention = float("nan")
            n_iter = 0
            solver_msg = str(e)

        # Compute fairness at optimal solution
        if converged:
            try:
                fairness_val = float(self.fairness_metric(m_opt))
            except Exception as e:
                warnings.warn(
                    f"fairness_metric failed at grid ({i},{j}): {e}", stacklevel=3
                )
                fairness_val = float("nan")
        else:
            fairness_val = float("nan")

        return {
            "eps_x": float(eps_x),
            "eps_y": float(eps_y),
            "converged": converged,
            "profit": profit,
            "gwp": gwp,
            "loss_ratio": lr,
            "retention": retention,
            "fairness": fairness_val,
            "n_iter": n_iter,
            "solver_message": solver_msg,
            "grid_i": i,
            "grid_j": j,
            "_multipliers": m_opt,
        }

    def _build_fairness_constraint(self, eps_y: float) -> list[dict]:
        """
        Build a scipy inequality constraint: eps_y - fairness_metric(m) >= 0.

        The Jacobian is provided if fairness_gradient is set, otherwise scipy
        will use finite differences for this one constraint.
        """
        fm = self.fairness_metric
        fg = self.fairness_gradient
        cap = float(eps_y)

        def _fair_fun(m: np.ndarray) -> float:
            return cap - fm(m)

        if fg is not None:
            def _fair_jac(m: np.ndarray) -> np.ndarray:
                return -fg(m)

            return [{"type": "ineq", "fun": _fair_fun, "jac": _fair_jac}]
        else:
            return [{"type": "ineq", "fun": _fair_fun}]

    def _set_x_epsilon(self, config: ConstraintConfig, eps_x: float) -> None:
        """Set the x-axis swept constraint in config."""
        if self.sweep_x == "volume_retention":
            config.retention_min = eps_x
        elif self.sweep_x == "gwp_min":
            config.gwp_min = eps_x
        elif self.sweep_x == "lr_max":
            config.lr_max = eps_x

    def _set_y_epsilon(self, config: ConstraintConfig, eps_y: float) -> None:
        """Set the y-axis swept constraint in config (non-fairness case)."""
        if self.sweep_y == "lr_max":
            config.lr_max = eps_y
        elif self.sweep_y == "gwp_min":
            config.gwp_min = eps_y
        elif self.sweep_y == "volume_retention":
            config.retention_min = eps_y

    def _run_parallel(
        self, tasks: list[tuple[int, int, float, float]]
    ) -> list[dict[str, Any]]:
        """Run grid sweep in parallel using joblib."""
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warnings.warn(
                "joblib not installed — falling back to sequential sweep. "
                "Install joblib for parallel execution: pip install joblib",
                stacklevel=2,
            )
            return [self._solve_at_grid_point(i, j, ex, ey) for i, j, ex, ey in tasks]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._solve_at_grid_point)(i, j, ex, ey)
            for i, j, ex, ey in tasks
        )
        return list(results)


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
