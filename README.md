# Welfare Cost of Anticipated Inflation — UK Interactive App

An interactive Streamlit dashboard for the MSc dissertation:
**"The Welfare Cost of Anticipated Inflation in the United Kingdom"**
(Ruo-Hao Hsin, UCL Data Science, 2025)

## What it does

- Fetches **live GDP data** from the ONS Timeseries API (series YBHA)
- Reads **M1 and Bank Rate** from Bank of England exports
- Re-runs **OLS money-demand estimation** (log-log and semi-log) on any sub-sample you choose
- Computes **Bailey-Lucas welfare costs** W(i) = area under the money-demand curve
- Shows an **international comparison** against US, Italy, Canada, China, Turkey, India and Brazil benchmarks

## Methodology

Bailey (1956) / Lucas (2000) consumer-surplus framework:

- **Log-log:**   `W(i) = -g/(1+g) · A · i^(1+g)`
- **Semi-log:**  `W(i) = B/ε · [1 - (1 + ε·i)·exp(-ε·i)]`

where `z(i) = M1/Y` is the money-income ratio and `i` is the Bank Rate (user cost proxy).

## Live demo

**Open the app:** [https://welfare-cost-inflation-uk-5p4hsscevl37q8hiefazzz.streamlit.app/](https://welfare-cost-inflation-uk-5p4hsscevl37q8hiefazzz.streamlit.app/)

Visitors do not need to install anything. GitHub: [leohsin/welfare-cost-inflation-uk](https://github.com/leohsin/welfare-cost-inflation-uk)

To redeploy: [share.streamlit.io](https://share.streamlit.io/deploy?repository=leohsin/welfare-cost-inflation-uk&branch=master&mainModule=app.py) — repository `leohsin/welfare-cost-inflation-uk`, branch `master`, main file `app.py`.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

| Series | Source | Update |
|--------|--------|--------|
| Nominal GDP (YBHA) | ONS Timeseries API | Live (hourly cache) |
| M1 (LPMAVAA) | Bank of England database export | Manual refresh |
| Bank Rate (IUDBEDR) | Bank of England database export | Manual refresh |

To refresh BoE data: export fresh files from
[bankofengland.co.uk/boeapps/database](https://www.bankofengland.co.uk/boeapps/database/)
and save them to `data/` (the app also falls back to `Welfare_Cost_of_Inflation_UK/resources/` locally).

## Project structure

```
app.py            # Streamlit UI
data.py           # Data fetching (ONS API + bundled BoE files)
models.py         # OLS estimation and welfare formulas
data/             # Bundled M1, Bank Rate, and annual fallback CSV
requirements.txt
```
