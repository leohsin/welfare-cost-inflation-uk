# Changelog

All notable changes to this project are documented here.

---

## [Unreleased] — 2026-08-13

### Added
- Bundled Bank of England M1, Bank Rate, and merged-panel files in `data/` so the app can run on Streamlit Community Cloud without the sibling thesis folder.
- Public Streamlit URL: [https://welfare-cost-inflation-uk-5p4hsscevl37q8hiefazzz.streamlit.app/](https://welfare-cost-inflation-uk-5p4hsscevl37q8hiefazzz.streamlit.app/)
- `runtime.txt` pins Python 3.12 for Streamlit Community Cloud.

### Changed
- `data.py` now loads BoE files from in-repo `data/` first, then falls back to `Welfare_Cost_of_Inflation_UK/resources/`.
- `.gitignore` now tracks `data/*.xlsx` and `data/*.csv`.

---

## [1.1.0] — 2026-07-25

### Added
- **Multi-frequency analysis**: data frequency selector in the sidebar (Annual / Quarterly / Monthly).
- `data.py` — new raw-series loaders:
  - `_load_m1_monthly()` returns the full monthly M1 time series (DatetimeIndex) instead of pre-aggregating to annual.
  - `_load_rate_daily()` returns the full daily Bank Rate series (DatetimeIndex).
  - `_fetch_ons_gdp_raw()` returns both an annual and a quarterly GDP series from the ONS API in one call.
- `build_panel(frequency)` accepts `"annual"`, `"quarterly"`, or `"monthly"` and aggregates accordingly:
  - **Quarterly**: end-of-quarter M1 stock, quarterly-average Bank Rate, ONS quarterly GDP × 4 (annualised).
  - **Monthly**: monthly M1 stock, monthly-average Bank Rate, quarterly GDP interpolated to monthly via cubic spline × 12 (annualised).
  - In all cases `z = M1 / annualised GDP` so the money-income ratio is comparable across frequencies.
- `app.py` — frequency radio button at top of sidebar; panel, OLS estimation, and all charts update on change.
- Hover labels on the money-demand scatter now show period-appropriate text ("2024 Q3", "Dec 2024", "2024").
- Informational banner in quarterly/monthly mode noting that Bailey-Lucas is a long-run framework.
- `_period_label()` helper and `_ts_fig()` helper to reduce chart-building boilerplate.

### Changed
- `_get_full_panel()` in `app.py` now accepts a `freq` argument so Streamlit caches one panel per frequency.
- Time-series overview charts use `panel["period"]` (Timestamp) on the x-axis instead of integer year.
- GDP chart title changes to "Nominal GDP — annualised (£bn)" when frequency is sub-annual.
- Bank Rate chart title reflects the averaging window (annual / quarterly / monthly).
- Observation marker size scales down at higher frequencies (monthly: 2 px, quarterly: 3 px, annual: 5 px).
- Data-source badge now shows separate GDP and M1/Rate provenance.
- Footer updated to remove the "Annual UK data" description.

### Fixed
- `_fetch_ons_gdp_raw()` previously used `dict.values()` directly in `pd.to_datetime()`; replaced with a list comprehension to avoid a `TypeError: len() of unsized object` in pandas.

---

## [1.0.0] — 2026-07-23

Initial commit: interactive Streamlit app for the UCL MSc thesis
"Welfare Cost of Anticipated Inflation — United Kingdom".

### Added
- `app.py` — Streamlit dashboard with four sections:
  1. **Data overview** — time-series charts for M1, GDP, z = M1/Y, and Bank Rate.
  2. **Money-demand estimation** — OLS regression table, estimated parameters, and scatter plot with fitted log-log and semi-log curves.
  3. **Welfare cost explorer** — interactive W(i) curve, per-benchmark welfare table, and disinflation gains table.
  4. **International comparison** — horizontal bar charts benchmarking UK estimates against Lucas (2000) and subsequent literature.
- `data.py` — data acquisition module:
  - Live GDP fetch from ONS Timeseries API (series YBHA).
  - M1 loaded from local BoE Excel export (`LPMAVAA`) using the `calamine` engine (resolves `openpyxl` `Fill()` `TypeError`).
  - Bank Rate loaded from local BoE CSV export (`IUDBEDR`); date format `%d %b %y` parsed correctly.
  - Pre-built merged CSV as final fallback.
  - `build_panel()` assembles annual panel; `filter_panel()` slices by year range.
- `models.py` — econometric models and welfare calculations:
  - `fit_loglog()` / `fit_semilog()` — OLS estimation via statsmodels.
  - `welfare_loglog()` / `welfare_semilog()` — Bailey-Lucas consumer-surplus integrals.
  - `welfare_table()` / `marginal_gains_table()` — tabular results at benchmark rates.
  - `INTERNATIONAL` — hardcoded DataFrame of international welfare-cost estimates from the thesis.
- `requirements.txt` — `streamlit`, `pandas`, `numpy`, `statsmodels`, `plotly`, `requests`, `python-calamine`.
- `README.md` — project overview, methodology summary, data sources, and run instructions.
- `.gitignore` — excludes Python build artefacts, Streamlit cache, and large local data files.

### Infrastructure
- Git repository initialised; code pushed to GitHub.

---

## Prior work (pre-commit) — 2026-07-22 to 2026-07-23

### Investigation & design
- Thesis PDF (`inflation_cost_of_welfare_UK.pdf`) read to extract model specifications, estimated parameters, and international comparison tables.
- Decided to build a Streamlit app (rather than a static Canvas) to support live data re-fetching and dynamic year-range re-estimation.

### Data debugging
- **ONS API** — identified that the old endpoint returned 404; located and confirmed the correct endpoint (`/timeseries/ybha/data`).
- **BoE API** — direct CSV downloads via `requests` returned HTML error pages (session/cookie requirement); pivoted to reading user-supplied local Excel/CSV files from `Welfare_Cost_of_Inflation_UK/resources/`.
- **M1 Excel parsing** — `openpyxl` raised `TypeError: Fill() takes no arguments` on the BoE file; fixed by switching to `engine="calamine"` (`python-calamine` added to dependencies).
- **Bank Rate CSV parsing** — date format was `DD Mon YY` (two-digit year), not `DD Mon YYYY`; updated `pd.to_datetime` format string to `%d %b %y`.
- Temporary `_test_data.py` created for debugging; deleted after issue resolved.

### Git / GitHub setup
- `.gitignore` and `README.md` written.
- GitHub CLI (`gh`) installed via `winget`; PATH refreshed in terminal session.
- `gh auth login` completed via browser OAuth flow.
- New public repository created and code pushed.
