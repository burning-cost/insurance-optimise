# Changelog

## [0.4.5] - 2026-03-25

### Added
- `ParetoFront`: lightweight bi-objective Pareto front visualiser in `pareto_front.py`.
  Takes any two arrays of objective values (e.g. profit vs fairness disparity ratio),
  computes the non-dominated subset, and provides:
  - `plot()` — scatter of dominated vs non-dominated points with staircase frontier line;
    returns `matplotlib.axes.Axes` for further customisation, no inline `plt.show()`
  - `summary()` — returns `ParetoFrontSummary` dataclass with frontier points, ideal/nadir
    points, and exact 2-objective hypervolume indicator (O(n log n) staircase algorithm)
  - `from_optimiser()` classmethod — builds a front from a list of `OptimisationResult`
    objects and a companion array of second-objective values
  - `from_pareto_result()` classmethod — extracts a 2D slice from a completed `ParetoResult`
    (e.g. visualise profit vs fairness from a 3-objective sweep without rerunning)
- Exported `ParetoFront` and `ParetoFrontSummary` from top-level `__init__.py`.
- Tests in `tests/test_pareto_front.py`: 35 tests covering mask logic, hypervolume,
  construction validation, summary, plot, and both classmethods.


## [0.4.4] - 2026-03-23

### Fixed
- docs: update README quickstart output comment to match current `OptimisationResult.__repr__` format — old format showed `converged=True, profit=47821.34, retention=0.812`; actual format is `CONVERGED, profit=118,196` (no decimal, comma-grouped, no retention field)


## [0.4.3] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.4.1 (2026-03-22) [unreleased]
- Fix licence badge (BSD-3 -> MIT) and footer; remove emoji from discussion CTA
- fix(readme): correct factually wrong Why bother table for DML elasticity
- docs: fix README review issues

## v0.4.1 (2026-03-21)
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- docs: add ParetoFrontier and model_quality module documentation (v0.4.0-0.4.1)
- feat: add ParetoFrontier for 3-objective insurance pricing optimisation (v0.4.0)
- Add model_quality module: Hedges (2025) LR-correlation formula + constraint integration
- Add pdoc API documentation with GitHub Pages
- Add Google Colab quickstart notebook and Open-in-Colab badge
- docs: standardise Installation heading
- Add quickstart notebook
- fix: README technical errors from quality review
- Add MIT license
- Add discussions link and star CTA
- docs: update Performance section with actual benchmark numbers from Databricks run
- fix: QA audit fixes — v0.3.3
- fix: P0/P1 bugs in stochastic LR constraint, cvar_max, fallback warning, docstring (v0.3.3)
- Add consulting CTA to README
- Add benchmarks: DML demand elasticity and constrained rate optimisation
- pin statsmodels>=0.14.5 for scipy compat
- Fix SLSQP KeyError on large problems — wrap unknown exit codes
- Polish flagship README: badges, benchmark table, problem statement
- docs: add Databricks notebook link
- fix: add result.profit alias and correct ENBP synthetic data in README; bump to 0.2.1
- Add Related Libraries section to README
- fix: update cross-references to consolidated repos
- Merge rate-optimiser: add ClaimsVarianceModel and plotting utilities
- fix: remove stray broken line introduced by merge
- fix: quick-start was missing data setup and polars import
