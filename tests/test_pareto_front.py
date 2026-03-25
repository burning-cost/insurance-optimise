"""
Tests for ParetoFront: bi-objective Pareto front visualiser.

Tests are structured to avoid running the full ParetoFrontier sweep
(which requires heavy computation). We test the class directly with
synthetic objective arrays, and test the classmethods with mock data.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_optimise.pareto_front import (
    ParetoFront,
    ParetoFrontSummary,
    _pareto_mask_2d,
    _hypervolume_2d,
)


# ---------------------------------------------------------------------------
# _pareto_mask_2d
# ---------------------------------------------------------------------------


class TestParetoMask2D:

    def test_two_points_dominated(self):
        # Point (1, 1) is dominated by (2, 2) when both are maximised
        obj1 = np.array([1.0, 2.0])
        obj2 = np.array([1.0, 2.0])
        mask = _pareto_mask_2d(obj1, obj2, maximize1=True, maximize2=True)
        assert mask[0] == False
        assert mask[1] == True

    def test_two_points_neither_dominates(self):
        # (2, 1) vs (1, 2) — neither dominates when both maximised
        obj1 = np.array([2.0, 1.0])
        obj2 = np.array([1.0, 2.0])
        mask = _pareto_mask_2d(obj1, obj2, maximize1=True, maximize2=True)
        assert mask[0] == True
        assert mask[1] == True

    def test_minimize_second_objective(self):
        # For obj2 minimised: (1.0, 0.5) dominates (0.5, 0.8) if maximize1=True
        # Point (1.0, 0.5): higher obj1, lower obj2 -> best
        # Point (0.5, 0.8): lower obj1, higher obj2 -> dominated
        obj1 = np.array([1.0, 0.5])
        obj2 = np.array([0.5, 0.8])
        mask = _pareto_mask_2d(obj1, obj2, maximize1=True, maximize2=False)
        assert mask[0] == True
        assert mask[1] == False

    def test_empty_array(self):
        mask = _pareto_mask_2d(np.array([]), np.array([]), True, True)
        assert len(mask) == 0

    def test_all_equal_points(self):
        # All identical: none dominated (no one is *strictly* better)
        obj1 = np.array([1.0, 1.0, 1.0])
        obj2 = np.array([2.0, 2.0, 2.0])
        mask = _pareto_mask_2d(obj1, obj2, maximize1=True, maximize2=True)
        assert mask.all()

    def test_clear_pareto_front(self):
        # Staircase front: (1,4), (2,3), (3,2), (4,1) are all non-dominated
        # (0,0) is dominated
        obj1 = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        obj2 = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
        mask = _pareto_mask_2d(obj1, obj2, maximize1=True, maximize2=True)
        assert mask[:4].all()
        assert not mask[4]

    def test_single_pareto_point(self):
        # One point dominates all others
        obj1 = np.array([5.0, 1.0, 2.0, 3.0])
        obj2 = np.array([5.0, 1.0, 2.0, 3.0])
        mask = _pareto_mask_2d(obj1, obj2, maximize1=True, maximize2=True)
        assert mask[0] == True
        assert not mask[1:].any()


# ---------------------------------------------------------------------------
# _hypervolume_2d
# ---------------------------------------------------------------------------


class TestHypervolume2D:

    def test_single_point(self):
        # Single point at (2, 3), reference at (0, 0), both maximised
        hv = _hypervolume_2d(
            np.array([2.0]), np.array([3.0]),
            ref1=0.0, ref2=0.0,
            maximize1=True, maximize2=True,
        )
        assert abs(hv - 6.0) < 1e-9  # 2 * 3

    def test_two_points(self):
        # (3, 1) and (1, 3), ref at (0, 0), both maximised
        # HV = 3*1 + 1*(3-1) = 3 + 2 = 5? Let's compute manually:
        # Sort by obj1 desc: (3,1), (1,3)
        # width(3): 3 - 1 = 2, height = max(1, 0) = 1 -> 2
        # width(1): 1 - 0 = 1, height = max(3, 1) = 3 -> 3
        # total = 5
        hv = _hypervolume_2d(
            np.array([3.0, 1.0]), np.array([1.0, 3.0]),
            ref1=0.0, ref2=0.0,
            maximize1=True, maximize2=True,
        )
        assert abs(hv - 5.0) < 1e-9

    def test_empty_front(self):
        hv = _hypervolume_2d(
            np.array([]), np.array([]),
            ref1=0.0, ref2=0.0,
            maximize1=True, maximize2=True,
        )
        assert hv == 0.0

    def test_point_below_reference_excluded(self):
        # Point (0.5, 0.5) is below reference (1.0, 1.0) -> HV = 0
        hv = _hypervolume_2d(
            np.array([0.5]), np.array([0.5]),
            ref1=1.0, ref2=1.0,
            maximize1=True, maximize2=True,
        )
        assert hv == 0.0

    def test_minimisation_direction(self):
        # Minimise both: point at (2, 3), ref at (5, 6).
        # After flip: a = -2 - (-5) = 3, b = -3 - (-6) = 3. HV = 9.
        hv = _hypervolume_2d(
            np.array([2.0]), np.array([3.0]),
            ref1=5.0, ref2=6.0,
            maximize1=False, maximize2=False,
        )
        assert abs(hv - 9.0) < 1e-9


# ---------------------------------------------------------------------------
# ParetoFront construction and validation
# ---------------------------------------------------------------------------


class TestParetoFrontInit:

    def test_basic_construction(self):
        pf = ParetoFront(
            obj1=np.array([1.0, 2.0, 3.0]),
            obj2=np.array([3.0, 2.0, 1.0]),
        )
        assert len(pf.obj1) == 3

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            ParetoFront(
                obj1=np.array([1.0, 2.0]),
                obj2=np.array([1.0]),
            )

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ParetoFront(
                obj1=np.array([1.0]),
                obj2=np.array([1.0]),
            )

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            ParetoFront(
                obj1=np.array([[1.0, 2.0]]),
                obj2=np.array([1.0, 2.0]),
            )

    def test_labels_wrong_length_raises(self):
        with pytest.raises(ValueError, match="labels"):
            ParetoFront(
                obj1=np.array([1.0, 2.0]),
                obj2=np.array([1.0, 2.0]),
                labels=["a"],
            )

    def test_frontier_indices_correct(self):
        # Point (3, 3) dominates everything else (both maximised)
        pf = ParetoFront(
            obj1=np.array([3.0, 1.0, 2.0]),
            obj2=np.array([3.0, 1.0, 2.0]),
        )
        assert 0 in pf.frontier_indices
        assert 1 not in pf.frontier_indices
        assert 2 not in pf.frontier_indices

    def test_staircase_front_all_non_dominated(self):
        obj1 = np.array([1.0, 2.0, 3.0, 4.0])
        obj2 = np.array([4.0, 3.0, 2.0, 1.0])
        pf = ParetoFront(obj1, obj2, maximize1=True, maximize2=True)
        assert len(pf.frontier_indices) == 4

    def test_profit_vs_fairness_minimize_fairness(self):
        # Profit (max) vs disparity ratio (min)
        profit = np.array([10000.0, 12000.0, 11000.0, 9000.0])
        disparity = np.array([1.10, 1.50, 1.20, 1.05])
        # (9000, 1.05): very fair but lowest profit
        # (12000, 1.50): highest profit but least fair
        # Both should be on the Pareto front (different tradeoffs)
        pf = ParetoFront(profit, disparity, maximize1=True, maximize2=False)
        fi = set(pf.frontier_indices.tolist())
        assert 1 in fi  # (12000, 1.50) — best profit
        assert 3 in fi  # (9000, 1.05) — best fairness


# ---------------------------------------------------------------------------
# ParetoFront.summary()
# ---------------------------------------------------------------------------


class TestParetoFrontSummary:

    def _make_pf(self):
        obj1 = np.array([1.0, 2.0, 3.0, 4.0, 1.5])
        obj2 = np.array([4.0, 3.0, 2.0, 1.0, 0.5])  # (1.5, 0.5) is dominated
        return ParetoFront(obj1, obj2, maximize1=True, maximize2=True)

    def test_returns_summary_dataclass(self):
        s = self._make_pf().summary()
        assert isinstance(s, ParetoFrontSummary)

    def test_n_total(self):
        s = self._make_pf().summary()
        assert s.n_total == 5

    def test_n_frontier_plus_dominated_equals_total(self):
        s = self._make_pf().summary()
        assert s.n_frontier + s.n_dominated == s.n_total

    def test_ideal_point(self):
        # Ideal: best independently — max obj1 = 4, max obj2 = 4
        pf = ParetoFront(
            obj1=np.array([4.0, 1.0, 2.0]),
            obj2=np.array([1.0, 4.0, 2.0]),
            maximize1=True, maximize2=True,
        )
        s = pf.summary()
        assert abs(s.ideal_point[0] - 4.0) < 1e-9
        assert abs(s.ideal_point[1] - 4.0) < 1e-9

    def test_hypervolume_positive(self):
        s = self._make_pf().summary()
        assert s.hypervolume > 0

    def test_frontier_sorted_by_obj1(self):
        s = self._make_pf().summary()
        assert all(s.frontier_obj1[i] <= s.frontier_obj1[i + 1]
                   for i in range(len(s.frontier_obj1) - 1))

    def test_repr_does_not_crash(self):
        s = self._make_pf().summary()
        r = repr(s)
        assert "ParetoFrontSummary" in r
        assert "n_frontier" in r


# ---------------------------------------------------------------------------
# ParetoFront.plot()
# ---------------------------------------------------------------------------

pytest.importorskip("matplotlib", reason="matplotlib not installed")


class TestParetoFrontPlot:

    def _make_pf(self):
        rng = np.random.default_rng(0)
        n = 20
        obj1 = rng.uniform(5000, 15000, size=n)
        obj2 = rng.uniform(1.0, 2.0, size=n)
        return ParetoFront(obj1, obj2, maximize1=True, maximize2=False,
                           obj1_name="Profit (£)", obj2_name="Disparity Ratio")

    def test_plot_returns_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        ax = self._make_pf().plot()
        assert ax is not None

    def test_plot_accepts_existing_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        returned = self._make_pf().plot(ax=ax)
        assert returned is ax

    def test_custom_title(self):
        import matplotlib
        matplotlib.use("Agg")
        ax = self._make_pf().plot(title="My Pareto Plot")
        assert ax.get_title() == "My Pareto Plot"

    def test_no_frontier_line(self):
        import matplotlib
        matplotlib.use("Agg")
        ax = self._make_pf().plot(show_frontier_line=False)
        assert ax is not None

    def test_annotate_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        obj1 = np.array([10000.0, 12000.0, 8000.0])
        obj2 = np.array([1.1, 1.5, 1.0])
        pf = ParetoFront(obj1, obj2, maximize1=True, maximize2=False,
                         labels=["A", "B", "C"])
        ax = pf.plot(annotate_labels=True)
        assert ax is not None


# ---------------------------------------------------------------------------
# ParetoFront.from_optimiser()
# ---------------------------------------------------------------------------


class TestFromOptimiser:

    def _make_results(self, n: int = 5):
        """Minimal mock OptimisationResult objects."""
        from unittest.mock import MagicMock
        from insurance_optimise.result import OptimisationResult

        results = []
        for i in range(n):
            r = MagicMock(spec=OptimisationResult)
            r.expected_profit = float(10000 + i * 1000)
            results.append(r)
        return results

    def test_returns_pareto_front(self):
        results = self._make_results(5)
        disparity = np.array([1.5, 1.4, 1.3, 1.2, 1.1])
        pf = ParetoFront.from_optimiser(results, obj2_values=disparity,
                                        obj2_name="Disparity", maximize2=False)
        assert isinstance(pf, ParetoFront)

    def test_length_mismatch_raises(self):
        results = self._make_results(3)
        with pytest.raises(ValueError, match="obj2_values length"):
            ParetoFront.from_optimiser(results, obj2_values=np.array([1.0, 2.0]))

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            ParetoFront.from_optimiser("not_a_result", obj2_values=np.array([1.0]))

    def test_single_result_raises_too_few(self):
        results = self._make_results(1)
        with pytest.raises(ValueError, match="at least 2"):
            ParetoFront.from_optimiser(results, obj2_values=np.array([1.0]))


# ---------------------------------------------------------------------------
# ParetoFront.from_pareto_result()
# ---------------------------------------------------------------------------


class TestFromParetoResult:

    def _make_pareto_result(self, n: int = 10):
        """Create a minimal ParetoResult with a synthetic pareto_df."""
        import polars as pl
        from unittest.mock import MagicMock
        from insurance_optimise.pareto import ParetoResult

        rng = np.random.default_rng(1)
        data = {
            "profit": rng.uniform(5000, 15000, size=n).tolist(),
            "retention": rng.uniform(0.8, 0.95, size=n).tolist(),
            "fairness": rng.uniform(1.0, 2.0, size=n).tolist(),
            "gwp": rng.uniform(8000, 20000, size=n).tolist(),
            "loss_ratio": rng.uniform(0.5, 0.8, size=n).tolist(),
            "converged": [True] * n,
            "eps_x": [0.9] * n,
            "eps_y": [1.5] * n,
            "n_iter": [50] * n,
            "solver_message": ["Optimization terminated successfully"] * n,
            "grid_i": list(range(n)),
            "grid_j": [0] * n,
        }
        surface = pl.DataFrame(data)
        pareto_df = surface.clone()

        result = MagicMock(spec=ParetoResult)
        result.pareto_df = pareto_df
        result.surface = surface
        return result

    def test_returns_pareto_front(self):
        pr = self._make_pareto_result()
        pf = ParetoFront.from_pareto_result(pr, obj1="profit", obj2="fairness")
        assert isinstance(pf, ParetoFront)

    def test_wrong_column_raises(self):
        pr = self._make_pareto_result()
        with pytest.raises(ValueError, match="Column"):
            ParetoFront.from_pareto_result(pr, obj1="profit", obj2="nonexistent")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            ParetoFront.from_pareto_result("not_a_result")

    def test_obj_names_set_from_columns(self):
        pr = self._make_pareto_result()
        pf = ParetoFront.from_pareto_result(pr, obj1="profit", obj2="fairness")
        assert "Profit" in pf.obj1_name
        assert "Fairness" in pf.obj2_name
