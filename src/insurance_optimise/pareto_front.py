"""
ParetoFront: lightweight bi-objective Pareto front visualiser.

The existing ``ParetoFrontier`` / ``ParetoResult`` types run the full
3-objective epsilon-constraint sweep and are tightly coupled to
``PortfolioOptimiser``. This module is deliberately decoupled: it takes
any two arrays of objective values, identifies the non-dominated subset,
and provides a plot and a summary dataclass.

Typical uses:

- Visualising tradeoffs from a completed ``ParetoResult`` (e.g. plot
  profit vs fairness after the sweep).
- Comparing solutions from different modelling approaches (e.g. GAM vs
  GBM predictions) without re-running the optimiser.
- Quick sanity-check during model development: is my Pareto front
  sensible before committing to the full 100-point grid?

Why a separate class rather than extending ``ParetoResult``?
``ParetoResult`` requires a full sweep. A pricing actuary often has a
handful of scenario outputs (perhaps 5–20 hand-picked operating points)
and wants to know which ones are non-dominated. Coupling that workflow to
the sweep infrastructure is overkill.

The hypervolume indicator uses the WFG algorithm for correctness but with
a 2-objective specialisation: for two objectives, HV reduces to a simple
area calculation that is O(n log n) — no approximation needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes


# ---------------------------------------------------------------------------
# Non-dominated filter (bi-objective)
# ---------------------------------------------------------------------------


def _pareto_mask_2d(
    obj1: np.ndarray,
    obj2: np.ndarray,
    maximize1: bool,
    maximize2: bool,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Return a boolean mask of non-dominated solutions.

    A solution i is dominated if there exists j such that j is at least
    as good on both objectives and strictly better on at least one.

    Parameters
    ----------
    obj1:
        First objective values, shape (N,).
    obj2:
        Second objective values, shape (N,).
    maximize1:
        True if obj1 should be maximised. False if minimised.
    maximize2:
        True if obj2 should be maximised. False if minimised.
    tol:
        Numerical tolerance for dominance comparison.

    Returns
    -------
    np.ndarray
        Boolean mask, shape (N,). True = non-dominated.
    """
    n = len(obj1)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Flip sign of minimisation objectives so we always maximise.
    a = obj1.copy() if maximize1 else -obj1.copy()
    b = obj2.copy() if maximize2 else -obj2.copy()

    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # j dominates i: j >= i on both, strict on at least one
            j_ge_a = a[j] >= a[i] - tol
            j_ge_b = b[j] >= b[i] - tol
            j_gt_a = a[j] > a[i] + tol
            j_gt_b = b[j] > b[i] + tol
            if j_ge_a and j_ge_b and (j_gt_a or j_gt_b):
                dominated[i] = True
                break

    return ~dominated


# ---------------------------------------------------------------------------
# Hypervolume (2-objective, exact)
# ---------------------------------------------------------------------------


def _hypervolume_2d(
    obj1: np.ndarray,
    obj2: np.ndarray,
    ref1: float,
    ref2: float,
    maximize1: bool,
    maximize2: bool,
) -> float:
    """
    Exact 2-objective hypervolume relative to a reference point.

    For two objectives, HV is the area of the space dominated by the
    Pareto front and bounded above/below by the reference point. This
    is O(n log n): sort by first objective, accumulate dominated area.

    The reference point should be worse than all front points (i.e. below
    the nadir). A common choice is the nadir point with a small offset.

    Parameters
    ----------
    obj1, obj2:
        Front points (non-dominated), shape (N,).
    ref1, ref2:
        Reference point coordinates. Must be dominated by every front
        point (i.e. worse than all of them).
    maximize1, maximize2:
        Optimisation directions.

    Returns
    -------
    float
        Hypervolume indicator. Higher is better.
    """
    if len(obj1) == 0:
        return 0.0

    # Normalise: flip to maximisation, subtract reference (so ref = origin).
    a = obj1.copy() if maximize1 else -obj1.copy()
    b = obj2.copy() if maximize2 else -obj2.copy()
    r1 = ref1 if maximize1 else -ref1
    r2 = ref2 if maximize2 else -ref2

    a = a - r1
    b = b - r2

    # Discard points that do not dominate the reference (shouldn't happen
    # if caller supplies a proper reference, but be defensive).
    valid = (a > 0) & (b > 0)
    a, b = a[valid], b[valid]
    if len(a) == 0:
        return 0.0

    # Sort descending by a; then accumulate staircase area.
    order = np.argsort(-a)
    a_s, b_s = a[order], b[order]

    hv = 0.0
    prev_b = 0.0
    for k in range(len(a_s)):
        width = a_s[k] - (a_s[k + 1] if k + 1 < len(a_s) else 0.0)
        height = max(b_s[k], prev_b)
        prev_b = max(prev_b, b_s[k])
        hv += width * height

    return float(hv)


# ---------------------------------------------------------------------------
# ParetoFrontSummary dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParetoFrontSummary:
    """
    Summary statistics from a ``ParetoFront`` computation.

    Attributes
    ----------
    frontier_obj1:
        First-objective values of the Pareto-optimal points, sorted by
        obj1 ascending.
    frontier_obj2:
        Second-objective values of the Pareto-optimal points, same order.
    frontier_indices:
        Original array indices of the Pareto-optimal points.
    n_total:
        Total number of input solutions.
    n_dominated:
        Number of dominated solutions.
    n_frontier:
        Number of Pareto-optimal solutions.
    ideal_point:
        Best achievable value on each objective independently (may not be
        simultaneously achievable): (best_obj1, best_obj2).
    nadir_point:
        Worst value on each objective among Pareto-optimal solutions:
        (worst_obj1_on_front, worst_obj2_on_front).
    hypervolume:
        Hypervolume indicator relative to a reference point just outside
        the nadir. Larger = better spread across the front.
    """

    frontier_obj1: np.ndarray
    frontier_obj2: np.ndarray
    frontier_indices: np.ndarray
    n_total: int
    n_dominated: int
    n_frontier: int
    ideal_point: tuple[float, float]
    nadir_point: tuple[float, float]
    hypervolume: float = field(default=0.0)

    def __repr__(self) -> str:
        return (
            f"ParetoFrontSummary("
            f"n_frontier={self.n_frontier}, "
            f"n_total={self.n_total}, "
            f"ideal={self.ideal_point[0]:.4g}, {self.ideal_point[1]:.4g}), "
            f"nadir=({self.nadir_point[0]:.4g}, {self.nadir_point[1]:.4g}), "
            f"hypervolume={self.hypervolume:.4g}"
            f")"
        )


# ---------------------------------------------------------------------------
# ParetoFront
# ---------------------------------------------------------------------------


class ParetoFront:
    """
    Bi-objective Pareto front visualiser.

    Takes any two arrays of objective values (e.g. profit and fairness_gap
    from a scenario sweep, or accuracy and interpretability from a model
    comparison) and computes the non-dominated subset.

    The ``plot()`` method produces a publication-ready scatter showing
    dominated vs non-dominated points and the staircase frontier line.
    The ``summary()`` method returns a ``ParetoFrontSummary`` dataclass
    with ideal/nadir points and the hypervolume indicator.

    Parameters
    ----------
    obj1:
        First objective values, shape (N,). E.g. expected profit.
    obj2:
        Second objective values, shape (N,). E.g. fairness disparity ratio.
    maximize1:
        True (default) if obj1 should be maximised. False to minimise.
    maximize2:
        True (default) if obj2 should be maximised. False to minimise.
        Set to False for a disparity ratio (lower = more fair).
    labels:
        Optional sequence of string labels for each solution, length N.
        Used to annotate points in ``plot()``.
    obj1_name:
        Human-readable name for obj1. Used as axis label. Default
        ``'Objective 1'``.
    obj2_name:
        Human-readable name for obj2. Default ``'Objective 2'``.

    Raises
    ------
    ValueError
        If obj1 and obj2 have different lengths or fewer than 2 points.

    Examples
    --------
    Profit vs fairness tradeoff from a ParetoResult:

    >>> pf = ParetoFront.from_pareto_result(result, obj1="profit", obj2="fairness")
    >>> ax = pf.plot()
    >>> summary = pf.summary()
    >>> print(summary.hypervolume)

    Ad-hoc comparison of 5 pricing scenarios:

    >>> profits = np.array([10_000, 11_500, 12_000, 13_000, 11_000])
    >>> disparity = np.array([1.10, 1.25, 1.40, 1.60, 1.20])
    >>> pf = ParetoFront(
    ...     obj1=profits,
    ...     obj2=disparity,
    ...     maximize1=True,
    ...     maximize2=False,
    ...     obj1_name="Profit (£)",
    ...     obj2_name="Fairness Disparity Ratio",
    ... )
    >>> ax = pf.plot(annotate_extremes=True)
    """

    def __init__(
        self,
        obj1: np.ndarray,
        obj2: np.ndarray,
        maximize1: bool = True,
        maximize2: bool = True,
        labels: list[str] | None = None,
        obj1_name: str = "Objective 1",
        obj2_name: str = "Objective 2",
    ) -> None:
        obj1 = np.asarray(obj1, dtype=float)
        obj2 = np.asarray(obj2, dtype=float)

        if obj1.ndim != 1 or obj2.ndim != 1:
            raise ValueError(
                f"obj1 and obj2 must be 1-D arrays. Got shapes "
                f"{obj1.shape} and {obj2.shape}."
            )
        if len(obj1) != len(obj2):
            raise ValueError(
                f"obj1 and obj2 must have the same length. Got "
                f"{len(obj1)} and {len(obj2)}."
            )
        if len(obj1) < 2:
            raise ValueError(
                "ParetoFront requires at least 2 solutions. "
                f"Got {len(obj1)}."
            )
        if labels is not None and len(labels) != len(obj1):
            raise ValueError(
                f"labels must have length {len(obj1)}, got {len(labels)}."
            )

        self.obj1 = obj1
        self.obj2 = obj2
        self.maximize1 = maximize1
        self.maximize2 = maximize2
        self.labels = list(labels) if labels is not None else None
        self.obj1_name = obj1_name
        self.obj2_name = obj2_name

        # Compute the non-dominated mask once at construction.
        self._mask: np.ndarray = _pareto_mask_2d(
            obj1, obj2, maximize1, maximize2
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frontier_indices(self) -> np.ndarray:
        """Indices of the Pareto-optimal solutions."""
        return np.where(self._mask)[0]

    @property
    def dominated_indices(self) -> np.ndarray:
        """Indices of the dominated solutions."""
        return np.where(~self._mask)[0]

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_optimiser(
        cls,
        result: "OptimisationResult",  # noqa: F821
        obj2_values: np.ndarray,
        obj2_name: str = "Objective 2",
        maximize2: bool = False,
    ) -> "ParetoFront":
        """
        Build a ``ParetoFront`` from a single ``OptimisationResult`` and
        a companion array of second-objective values.

        This is most useful when you have run a manual sweep (e.g. varying
        ``lr_max`` from 0.60 to 0.80 in 10 steps) and collected a list of
        results, then evaluated a second metric (e.g. fairness disparity)
        at each point.

        Parameters
        ----------
        result:
            A list of ``OptimisationResult`` objects, one per scenario.
            Expected profit is extracted as obj1.
        obj2_values:
            Array of second-objective values, length = len(result).
        obj2_name:
            Human-readable name for the second objective.
        maximize2:
            Whether to maximise obj2. Default False (for fairness
            disparity ratio: lower is better).

        Returns
        -------
        ParetoFront

        Raises
        ------
        TypeError
            If ``result`` is not a list of ``OptimisationResult`` instances.
        ValueError
            If ``obj2_values`` length does not match ``result`` length.
        """
        # Accept either a single result or a list
        from insurance_optimise.result import OptimisationResult

        if isinstance(result, OptimisationResult):
            results = [result]
        else:
            results = list(result)
            if not all(isinstance(r, OptimisationResult) for r in results):
                raise TypeError(
                    "result must be an OptimisationResult or a list of them."
                )

        obj2_values = np.asarray(obj2_values, dtype=float)
        if len(obj2_values) != len(results):
            raise ValueError(
                f"obj2_values length ({len(obj2_values)}) must match "
                f"number of results ({len(results)})."
            )

        profits = np.array([r.expected_profit for r in results], dtype=float)
        return cls(
            obj1=profits,
            obj2=obj2_values,
            maximize1=True,
            maximize2=maximize2,
            obj1_name="Expected Profit (£)",
            obj2_name=obj2_name,
        )

    @classmethod
    def from_pareto_result(
        cls,
        pareto_result: "ParetoResult",  # noqa: F821
        obj1: str = "profit",
        obj2: str = "fairness",
        maximize1: bool = True,
        maximize2: bool = False,
        pareto_only: bool = True,
    ) -> "ParetoFront":
        """
        Build a ``ParetoFront`` from a completed ``ParetoResult``.

        Extracts two scalar columns from the Pareto surface and recomputes
        the bi-objective non-dominated front. Use this to visualise a
        specific 2D slice of a 3-objective surface, or to recompute the
        front after dropping an objective.

        Parameters
        ----------
        pareto_result:
            Output of ``ParetoFrontier.run()``.
        obj1:
            Column name from ``pareto_result.surface`` (or
            ``pareto_result.pareto_df`` when ``pareto_only=True``).
            Default ``'profit'``.
        obj2:
            Column name. Default ``'fairness'``.
        maximize1:
            Whether to maximise obj1. Default True.
        maximize2:
            Whether to maximise obj2. Default False (fairness disparity:
            lower = more fair).
        pareto_only:
            If True (default), use ``pareto_result.pareto_df`` — only
            the 3-objective Pareto-optimal points. If False, use all
            converged points from ``pareto_result.surface``.

        Returns
        -------
        ParetoFront

        Raises
        ------
        ValueError
            If the requested columns are not present in the data.
        """
        from insurance_optimise.pareto import ParetoResult

        if not isinstance(pareto_result, ParetoResult):
            raise TypeError(
                f"pareto_result must be a ParetoResult, got {type(pareto_result)}."
            )

        df = pareto_result.pareto_df if pareto_only else pareto_result.surface.filter(
            pareto_result.surface["converged"]
        )

        for col in (obj1, obj2):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in ParetoResult. "
                    f"Available columns: {df.columns}"
                )

        if len(df) < 2:
            raise ValueError(
                "ParetoResult has fewer than 2 converged solutions. "
                "Cannot build a meaningful front."
            )

        return cls(
            obj1=df[obj1].to_numpy(),
            obj2=df[obj2].to_numpy(),
            maximize1=maximize1,
            maximize2=maximize2,
            obj1_name=obj1.replace("_", " ").title(),
            obj2_name=obj2.replace("_", " ").title(),
        )

    # ------------------------------------------------------------------
    # summary()
    # ------------------------------------------------------------------

    def summary(self) -> ParetoFrontSummary:
        """
        Compute summary statistics for the Pareto front.

        Returns
        -------
        ParetoFrontSummary
            Contains ideal/nadir points, hypervolume, and the frontier
            point arrays sorted by obj1.
        """
        idx = self.frontier_indices
        f1 = self.obj1[idx]
        f2 = self.obj2[idx]

        # Sort by obj1 for consistent ordering
        order = np.argsort(f1)
        f1_sorted = f1[order]
        f2_sorted = f2[order]
        idx_sorted = idx[order]

        # Ideal: best on each objective independently (from all points)
        ideal1 = float(self.obj1.max() if self.maximize1 else self.obj1.min())
        ideal2 = float(self.obj2.max() if self.maximize2 else self.obj2.min())

        # Nadir: worst on each objective restricted to the Pareto front
        nadir1 = float(f1_sorted.min() if self.maximize1 else f1_sorted.max())
        nadir2 = float(f2_sorted.min() if self.maximize2 else f2_sorted.max())

        # Reference point for hypervolume: just outside the nadir.
        # Offset is 5% of the objective range (at least 1e-6 to handle
        # degenerate cases where all front points are identical).
        range1 = float(self.obj1.max() - self.obj1.min())
        range2 = float(self.obj2.max() - self.obj2.min())
        offset1 = max(range1 * 0.05, 1e-6)
        offset2 = max(range2 * 0.05, 1e-6)

        ref1 = nadir1 - offset1 if self.maximize1 else nadir1 + offset1
        ref2 = nadir2 - offset2 if self.maximize2 else nadir2 + offset2

        hv = _hypervolume_2d(f1_sorted, f2_sorted, ref1, ref2, self.maximize1, self.maximize2)

        return ParetoFrontSummary(
            frontier_obj1=f1_sorted,
            frontier_obj2=f2_sorted,
            frontier_indices=idx_sorted,
            n_total=len(self.obj1),
            n_dominated=int((~self._mask).sum()),
            n_frontier=len(idx),
            ideal_point=(ideal1, ideal2),
            nadir_point=(nadir1, nadir2),
            hypervolume=hv,
        )

    # ------------------------------------------------------------------
    # plot()
    # ------------------------------------------------------------------

    def plot(
        self,
        ax: "matplotlib.axes.Axes | None" = None,
        figsize: tuple[float, float] = (8, 5),
        title: str | None = None,
        dominated_color: str = "#aec7e8",
        frontier_color: str = "#1f77b4",
        frontier_edge: str = "navy",
        show_frontier_line: bool = True,
        annotate_extremes: bool = True,
        annotate_labels: bool = False,
    ) -> "matplotlib.axes.Axes":
        """
        Scatter plot of dominated vs non-dominated solutions with the
        Pareto frontier line.

        Returns the matplotlib Axes object so callers can further
        customise (titles, legends, saving) without any inline plt.show()
        calls.

        Parameters
        ----------
        ax:
            Existing Axes to plot into. If None, a new figure is created.
        figsize:
            Figure size in inches. Ignored when ``ax`` is provided.
        title:
            Plot title. Defaults to
            ``"{obj2_name} vs {obj1_name}: Pareto Front"``.
        dominated_color:
            Colour for dominated points. Default light blue.
        frontier_color:
            Fill colour for Pareto-optimal points. Default blue.
        frontier_edge:
            Edge colour for Pareto-optimal points.
        show_frontier_line:
            If True, draw the staircase frontier line connecting the
            non-dominated points in order of obj1.
        annotate_extremes:
            If True, annotate the extreme points on the frontier
            (highest obj1, best obj2).
        annotate_labels:
            If True and ``self.labels`` is set, annotate every Pareto-
            optimal point with its label. Keep off for large N.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting. "
                "Install with: pip install insurance-optimise[plot]"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        dom_idx = self.dominated_indices
        front_idx = self.frontier_indices

        # Dominated points
        if len(dom_idx) > 0:
            ax.scatter(
                self.obj1[dom_idx],
                self.obj2[dom_idx],
                color=dominated_color,
                s=50,
                alpha=0.7,
                zorder=2,
                label=f"Dominated ({len(dom_idx)})",
            )

        # Pareto-optimal points
        if len(front_idx) > 0:
            ax.scatter(
                self.obj1[front_idx],
                self.obj2[front_idx],
                color=frontier_color,
                edgecolors=frontier_edge,
                linewidths=1.0,
                s=90,
                zorder=3,
                label=f"Pareto-optimal ({len(front_idx)})",
            )

        # Frontier staircase line
        if show_frontier_line and len(front_idx) > 1:
            order = np.argsort(self.obj1[front_idx])
            fx = self.obj1[front_idx[order]]
            fy = self.obj2[front_idx[order]]
            ax.plot(fx, fy, color=frontier_edge, linewidth=1.2,
                    alpha=0.6, zorder=2, linestyle="--")

        # Annotate extremes
        if annotate_extremes and len(front_idx) > 0:
            # Best obj1 (highest if maximise, lowest if minimise)
            if self.maximize1:
                best1_local = int(np.argmax(self.obj1[front_idx]))
            else:
                best1_local = int(np.argmin(self.obj1[front_idx]))
            best1_idx = front_idx[best1_local]
            ax.annotate(
                f"Best {self.obj1_name.split('(')[0].strip()}",
                (self.obj1[best1_idx], self.obj2[best1_idx]),
                xytext=(8, -14),
                textcoords="offset points",
                fontsize=8,
                color=frontier_edge,
            )

            # Best obj2 (only annotate if different from best1)
            if self.maximize2:
                best2_local = int(np.argmax(self.obj2[front_idx]))
            else:
                best2_local = int(np.argmin(self.obj2[front_idx]))
            best2_idx = front_idx[best2_local]
            if best2_idx != best1_idx:
                ax.annotate(
                    f"Best {self.obj2_name.split('(')[0].strip()}",
                    (self.obj1[best2_idx], self.obj2[best2_idx]),
                    xytext=(8, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color=frontier_edge,
                )

        # Label individual frontier points
        if annotate_labels and self.labels is not None and len(front_idx) > 0:
            for i in front_idx:
                ax.annotate(
                    self.labels[i],
                    (self.obj1[i], self.obj2[i]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

        ax.set_xlabel(self.obj1_name)
        ax.set_ylabel(self.obj2_name)
        ax.set_title(
            title or f"{self.obj2_name} vs {self.obj1_name}: Pareto Front"
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

        return ax
