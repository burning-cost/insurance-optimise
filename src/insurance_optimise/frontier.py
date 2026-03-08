"""
Efficient frontier generation for insurance portfolio rate optimisation.

The EfficientFrontier class implements the epsilon-constraint method:
fix a primary objective (profit maximisation) and sweep a secondary
constraint (e.g. volume retention from 85% to 99%) across N_points values.
Each point on the sweep is an independent optimisation problem.

This produces the Pareto frontier: the set of solutions where you cannot
improve profit without sacrificing retention (or vice versa). The frontier
tells the pricing team the *cost* of each retention target in profit terms.

Parallelisation: each sweep point is independent, so they can run in
parallel via joblib. The library does not require joblib (it's optional),
falling back to sequential execution.

Reference: epsilon-constraint method — Laumanns et al. (2012) EJOR;
practical formulation from KB entry 610.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl

from insurance_optimise.constraints import ConstraintConfig
from insurance_optimise.result import EfficientFrontierResult, FrontierPoint

# Type alias — avoid circular import at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurance_optimise.optimiser import PortfolioOptimiser


class EfficientFrontier:
    """
    Efficient frontier via epsilon-constraint sweep.

    Sweeps one constraint parameter across a range and solves the
    optimisation at each point, tracing the profit-retention (or
    profit-GWP) Pareto frontier.

    Parameters
    ----------
    optimiser:
        A configured PortfolioOptimiser instance. The optimiser's
        constraint config will be modified for each sweep point.
    sweep_param:
        Which constraint to sweep. Supported:
        ``'volume_retention'``, ``'gwp_min'``, ``'lr_max'``.
    sweep_range:
        (min_value, max_value) for the sweep. E.g. (0.85, 0.99) for
        retention. The range is inclusive.
    n_points:
        Number of points on the frontier. Default 15.
    n_jobs:
        Number of parallel workers. -1 = all CPUs. 1 = sequential (default).
        Requires joblib if n_jobs != 1.
    """

    def __init__(
        self,
        optimiser: "PortfolioOptimiser",
        sweep_param: str,
        sweep_range: tuple[float, float],
        n_points: int = 15,
        n_jobs: int = 1,
    ) -> None:
        self.optimiser = optimiser
        self.sweep_param = sweep_param
        self.sweep_range = sweep_range
        self.n_points = n_points
        self.n_jobs = n_jobs

        _VALID_PARAMS = ("volume_retention", "gwp_min", "lr_max")
        if sweep_param not in _VALID_PARAMS:
            raise ValueError(
                f"sweep_param '{sweep_param}' not supported. "
                f"Choose from {_VALID_PARAMS}."
            )

    def run(self) -> EfficientFrontierResult:
        """
        Run the full frontier sweep.

        Returns
        -------
        EfficientFrontierResult
            Contains all frontier points and a summary DataFrame.
        """
        epsilons = np.linspace(
            self.sweep_range[0], self.sweep_range[1], self.n_points
        )

        if self.n_jobs == 1:
            points = [self._solve_at_epsilon(eps) for eps in epsilons]
        else:
            points = self._run_parallel(epsilons)

        return EfficientFrontierResult(points=points, sweep_param=self.sweep_param)

    def _solve_at_epsilon(self, epsilon: float) -> FrontierPoint:
        """Solve optimisation with the sweep constraint set to epsilon."""
        # Deep copy the config to avoid mutating the original
        new_config = copy.deepcopy(self.optimiser.config)
        self._set_epsilon(new_config, epsilon)

        # Rebuild the optimiser with the new config
        from insurance_optimise.optimiser import PortfolioOptimiser

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
            solver=(
                "slsqp"
                if self.optimiser.solver_method == "SLSQP"
                else "trust_constr"
            ),
            n_restarts=self.optimiser.n_restarts,
            seed=int(epsilon * 1e6) % (2**31),  # deterministic but varied seed
            ftol=self.optimiser.ftol,
            maxiter=self.optimiser.maxiter,
        )

        try:
            result = opt.optimise()
        except Exception as e:
            warnings.warn(
                f"Frontier point epsilon={epsilon:.4f} failed: {e}", stacklevel=2
            )
            # Return a dummy non-converged result
            from insurance_optimise.result import OptimisationResult

            dummy_m = np.ones(opt.n)
            result = OptimisationResult(
                multipliers=dummy_m,
                new_premiums=dummy_m * opt.tc,
                expected_demand=opt.x0,
                expected_profit=float("nan"),
                expected_gwp=float("nan"),
                expected_loss_ratio=float("nan"),
                expected_retention=None,
                shadow_prices={},
                converged=False,
                solver_message=str(e),
                n_iter=0,
                audit_trail={},
                summary_df=pl.DataFrame(),
            )

        return FrontierPoint(epsilon=epsilon, result=result)

    def _set_epsilon(self, config: ConstraintConfig, epsilon: float) -> None:
        """Set the swept constraint in config to epsilon."""
        if self.sweep_param == "volume_retention":
            config.retention_min = epsilon
        elif self.sweep_param == "gwp_min":
            config.gwp_min = epsilon
        elif self.sweep_param == "lr_max":
            config.lr_max = epsilon

    def _run_parallel(self, epsilons: np.ndarray) -> list[FrontierPoint]:
        """Run frontier sweep in parallel using joblib."""
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warnings.warn(
                "joblib not installed — falling back to sequential frontier sweep. "
                "Install joblib for parallel execution: pip install joblib",
                stacklevel=2,
            )
            return [self._solve_at_epsilon(eps) for eps in epsilons]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._solve_at_epsilon)(eps) for eps in epsilons
        )
        return list(results)

    def plot(
        self,
        x_metric: str = "retention",
        y_metric: str = "profit",
        figsize: tuple[float, float] = (8, 5),
        title: str | None = None,
    ) -> None:
        """
        Plot the efficient frontier.

        Requires matplotlib. Shows only converged frontier points.

        Parameters
        ----------
        x_metric:
            Column from frontier data to use as x-axis.
            One of: 'retention', 'gwp', 'loss_ratio', 'epsilon'.
        y_metric:
            Column for y-axis. Default 'profit'.
        figsize:
            Figure size tuple.
        title:
            Plot title. Auto-generated if None.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting. "
                "Install with: pip install matplotlib"
            )

        result = self.run()
        data = result.pareto_data()

        if len(data) == 0:
            warnings.warn("No converged frontier points to plot.", stacklevel=2)
            return

        x = data[x_metric].to_numpy()
        y = data[y_metric].to_numpy()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, "o-", color="#1f77b4", linewidth=2, markersize=6)
        ax.set_xlabel(x_metric.replace("_", " ").title())
        ax.set_ylabel(y_metric.replace("_", " ").title())
        ax.set_title(
            title
            or f"Efficient Frontier: {y_metric} vs {x_metric} "
            f"(sweep={self.sweep_param})"
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
