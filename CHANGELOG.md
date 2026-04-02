# Changelog

## [0.6.0] - 2026-04-02

### Added
- `LinearRiskSharingPool`: community-based insurance pool with linear allocation matrix,
  implementing Denuit, Flores-Contró, Robert (2026), arXiv:2603.29530.
  - `mean_proportional()` classmethod: allocation proportional to expected claim volume;
    satisfies budget balance and actuarial fairness by construction
  - `validate_conditions()`: checks budget balance, actuarial fairness, capacity;
    returns `ValidationResult` with per-condition booleans and violation magnitudes
  - `ruin_comparison()`: exact Cramér-Lundberg ruin probabilities (method='cramerlundberg',
    exponential severity) or simulation estimate (method='simulation');
    returns `RuinResult` with pooled, standalone, and improvement arrays
  - `simulate()`: Gillespie event-driven Monte Carlo of all n surplus paths; supports
    exponential, lognormal, gamma severity; returns `SimulationResult`
  - `optimal_allocation()`: SLSQP optimisation of the allocation matrix under budget
    balance and actuarial fairness equality constraints; objectives: min_max_ruin,
    max_min_improvement; returns new pool instance
  - `audit_trail()`: JSON-serialisable dict for FCA/regulatory documentation; includes
    full parameter record, validation results, scale family warning
  - `PerformanceWarning`: raised when simulation complexity or n > 30 (optimal_allocation)
    is likely to be slow
- New module `src/insurance_optimise/risk_sharing.py` (~450 LOC)
- Exported `LinearRiskSharingPool`, `RuinResult`, `SimulationResult`, `ValidationResult`,
  `PerformanceWarning` from top-level `__init__.py`
- Tests in `tests/test_risk_sharing.py`: 50 tests covering construction, input validation,
  condition checking, ruin comparison (exact and simulation), simulate, optimal allocation,
  audit trail, and edge cases

### Changed
- Version bumped from 0.5.0 to 0.6.0
- Added risk-sharing and community insurance keywords to PyPI metadata

## [0.5.0] - 2026-04-02

### Added
- `RobustReinsuranceOptimiser`: multi-line proportional cession optimisation under
  model uncertainty, implementing Boonen, Dela Vega, Garces (2026) arXiv:2603.25350.
  - `ReinsuranceLine` dataclass: parameters for one insurance line (mu, sigma,
    reins_loading, ambiguity theta, correlation)
  - `RobustReinsuranceResult` dataclass: output including dividend barrier b*,
    per-surplus cession schedule (Polars DataFrame), audit trail, and plot method
  - Symmetric closed-form solver (`_solve_symmetric`): ODE shooting via
    `scipy.integrate.solve_ivp` + `scipy.optimize.brentq` for lines with identical
    parameters; returns `solver='symmetric_closed_form'`
  - Asymmetric numerical solver (`_solve_asymmetric`): PDE value iteration on a
    2D grid `[0, surplus_max]^2` for two lines with different parameters; returns
    `solver='asymmetric_pde'`
  - `cession_at(x)`: evaluate optimal cession fraction at any surplus value(s)
  - `sensitivity(param, n_points)`: sweep ambiguity or loading and return a Polars
    DataFrame of cession fraction vs parameter value
  - `to_json()` / `save_audit()`: JSON-serialisable audit trail with full parameter
    record; consistent with `OptimisationResult.to_json()` format
  - `plot_cession_schedule()`: matplotlib visualisation — 1D line plot for symmetric
    case, 2D heatmap for asymmetric case; graceful degradation if matplotlib absent
  - `force_numerical=True` flag: bypass closed-form and use PDE solver (for testing)
- New module `src/insurance_optimise/reinsurance.py` (~380 LOC)
- Exported `ReinsuranceLine`, `RobustReinsuranceOptimiser`, `RobustReinsuranceResult`
  from top-level `__init__.py`
- Tests in `tests/test_reinsurance.py`: 30 tests covering validation, closed-form
  behaviour (theta=0 vs theta>0), monotonicity, barrier sensitivity, sensitivity
  analysis, PDE convergence, output schema, plotting, and edge cases

### Changed
- Version bumped from 0.4.5 to 0.5.0
- Added reinsurance-related keywords to PyPI metadata


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
