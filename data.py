"""
data.py — Data fetching for the Welfare Cost of Inflation app.

Strategy (in priority order):
  1. ONS Timeseries API  — GDP (YBHA)           live JSON, no auth required
  2. Bundled BoE files   — M1 (LPMAVAA .xlsx) in data/ (repo, for Streamlit Cloud)
                         — Bank Rate (IUDBEDR .csv)
                         — sibling Welfare_Cost_of_Inflation_UK/resources/ as local fallback
  3. Merged panel CSV    — annual fallback if individual files not found

Supported frequencies: "annual" | "quarterly" | "monthly"

For quarterly / monthly, GDP is aggregated / interpolated from the ONS
quarterly series. The money-income ratio z is always computed as:
    z = M1_stock / annualised_GDP
so that z is comparable across frequencies.

To refresh BoE data manually:
  1. Visit https://www.bankofengland.co.uk/boeapps/database/
  2. Search for LPMAVAA (M1) and IUDBEDR (Bank Rate)
  3. Export as CSV/Excel and save to the in-repo data/ folder
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_RES = _HERE / "data"
_RES_FALLBACK = _HERE.parent / "Welfare_Cost_of_Inflation_UK" / "resources"

_BOE_M1_NAME = "LPMAVAA  Bank of England  Database.xlsx"
_BOE_RATE_NAME = "IUDBEDR  Bank of England  Database.csv"
_MERGED_NAME = "uk_merged_data_for_lucas.csv"

_ONS_GDP_URL = (
    "https://www.ons.gov.uk/economy/grossdomesticproductgdp"
    "/timeseries/ybha/data"
)


def _resolve_data_file(name: str) -> Path:
    """Prefer in-repo data/, then the sibling thesis resources folder."""
    primary = _RES / name
    if primary.exists():
        return primary
    return _RES_FALLBACK / name


_BOE_M1_XLSX = _resolve_data_file(_BOE_M1_NAME)
_BOE_RATE_CSV = _resolve_data_file(_BOE_RATE_NAME)
_MERGED_CSV = _resolve_data_file(_MERGED_NAME)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

Frequency = Literal["annual", "quarterly", "monthly"]


# ---------------------------------------------------------------------------
# Raw series loaders  (DatetimeIndex, un-aggregated)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_m1_monthly() -> pd.Series | None:
    """
    BoE LPMAVAA → monthly end-of-month M1 stock, £ millions.
    DatetimeIndex of month-end dates.
    """
    if not _BOE_M1_XLSX.exists():
        return None
    try:
        raw = pd.read_excel(
            _BOE_M1_XLSX, header=None, skiprows=2, engine="calamine",
            names=["date", "value"], usecols=[0, 1],
        )
        raw["date"] = pd.to_datetime(
            raw["date"].astype(str).str.strip(), format="%d %b %y", errors="coerce"
        )
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        s = raw.dropna().set_index("date")["value"].sort_index()
        # Snap to month-end
        s.index = s.index + pd.offsets.MonthEnd(0)
        s.name = "m1_m"
        return s
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_rate_daily() -> pd.Series | None:
    """
    BoE IUDBEDR → daily Bank Rate, % (not yet in decimal).
    DatetimeIndex.
    """
    if not _BOE_RATE_CSV.exists():
        return None
    try:
        df = pd.read_csv(
            _BOE_RATE_CSV, header=0, names=["date", "value"],
            quotechar='"', skipinitialspace=True,
        )
        df["date"] = pd.to_datetime(
            df["date"].astype(str).str.strip(), format="%d %b %y", errors="coerce"
        )
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.dropna().set_index("date")["value"].sort_index()
        s.name = "rate_pct"
        return s
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ons_gdp_raw() -> tuple[pd.Series, pd.Series, bool]:
    """
    Returns (annual_gdp, quarterly_gdp, is_live).
    Both in £ millions with DatetimeIndex (annual: Dec 31; quarterly: quarter-end).
    """
    try:
        r = requests.get(_ONS_GDP_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        payload = r.json()

        # Annual
        annual = pd.Series(
            {
                int(row["year"]): float(row["value"])
                for row in payload.get("years", [])
                if row.get("value") not in ("", None, ".")
            },
            name="gdp_m",
        ).sort_index()
        ann_idx = pd.to_datetime([f"{y}-12-31" for y in annual.index])
        annual_ts = pd.Series(annual.values, index=ann_idx, name="gdp_m")

        # Quarterly
        q_records = []
        for row in payload.get("quarters", []):
            if row.get("value") in ("", None, "."):
                continue
            yr  = int(row["year"])
            qtr = row["quarter"]  # "Q1" … "Q4"
            q_month = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}[qtr]
            dt = pd.Timestamp(yr, q_month, 1) + pd.offsets.MonthEnd(0)
            q_records.append((dt, float(row["value"])))
        qts = pd.Series(
            [v for _, v in q_records],
            index=pd.DatetimeIndex([d for d, _ in q_records]),
            name="gdp_m",
        ).sort_index()

        return annual_ts, qts, True
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float), False


# ---------------------------------------------------------------------------
# Annual fallback from pre-built merged CSV
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_merged_panel() -> pd.DataFrame | None:
    if not _MERGED_CSV.exists():
        return None
    try:
        df = pd.read_csv(_MERGED_CSV)[["year", "z", "i"]].dropna()
        df["year"] = df["year"].astype(int)
        df["z"] = pd.to_numeric(df["z"], errors="coerce")
        df["i"] = pd.to_numeric(df["i"], errors="coerce")
        df = df.dropna().query("z > 0 and i > 0").sort_values("year")
        df["period"] = pd.to_datetime(df["year"].astype(str) + "-12-31")
        return df.reset_index(drop=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

def build_panel(
    frequency: Frequency = "annual",
) -> tuple[pd.DataFrame, dict]:
    """
    Build the estimation panel at the requested frequency.

    Returns
    -------
    panel : pd.DataFrame
        Columns: period (Timestamp), year (int), z (float), i (float),
                 m1_bn (float, optional), gdp_bn (float, optional)
    sources : dict  {"gdp": "live"|"local"|"none", "m1": ..., "rate": ...}

    z is always M1_stock / annualised_GDP so it is comparable across
    frequencies (quarterly GDP × 4, monthly GDP × 12).
    """
    sources: dict[str, str] = {"gdp": "none", "m1": "none", "rate": "none"}

    m1_monthly  = _load_m1_monthly()
    rate_daily  = _load_rate_daily()
    gdp_ann_ts, gdp_qts, gdp_live = _fetch_ons_gdp_raw()

    if m1_monthly is not None:
        sources["m1"] = "local"
    if rate_daily is not None:
        sources["rate"] = "local"
    if gdp_live and len(gdp_ann_ts) > 0:
        sources["gdp"] = "live"

    # ------------------------------------------------------------------ annual
    if frequency == "annual":
        if m1_monthly is not None and rate_daily is not None and len(gdp_ann_ts) > 0:
            m1_ann  = m1_monthly.resample("YE").last()   # end-Dec stock, £m
            rate_ann = rate_daily.resample("YE").mean() / 100  # decimal
            gdp_ann  = gdp_ann_ts.resample("YE").last()  # ONS already annual

            common = m1_ann.index.intersection(rate_ann.index).intersection(gdp_ann.index)
            panel = pd.DataFrame({
                "period": common,
                "year":   common.year,
                "m1_bn":  m1_ann.reindex(common).values / 1000,
                "gdp_bn": gdp_ann.reindex(common).values / 1000,
                "i":      rate_ann.reindex(common).values,
            })
            panel["z"] = panel["m1_bn"] / panel["gdp_bn"]
            panel = panel.dropna(subset=["z","i"]).query("z>0 and i>0")
            if len(panel) >= 10:
                return panel.reset_index(drop=True), sources

        # fallback
        merged = _load_merged_panel()
        if merged is not None:
            sources = {k: "local" for k in sources}
            return merged, sources

    # --------------------------------------------------------------- quarterly
    elif frequency == "quarterly":
        if m1_monthly is not None and rate_daily is not None and len(gdp_qts) > 0:
            m1_q   = m1_monthly.resample("QE").last()          # end-quarter stock
            rate_q = rate_daily.resample("QE").mean() / 100    # quarterly avg → decimal
            gdp_q  = gdp_qts.resample("QE").last()             # quarterly GDP flow, £m

            common = m1_q.index.intersection(rate_q.index).intersection(gdp_q.index)
            # Annualise GDP: quarterly flow × 4 keeps z on the same scale as annual
            panel = pd.DataFrame({
                "period": common,
                "year":   common.year,
                "m1_bn":  m1_q.reindex(common).values / 1000,
                "gdp_bn": gdp_q.reindex(common).values * 4 / 1000,  # annualised
                "i":      rate_q.reindex(common).values,
            })
            panel["z"] = panel["m1_bn"] / panel["gdp_bn"]
            panel = panel.dropna(subset=["z","i"]).query("z>0 and i>0")
            if len(panel) >= 10:
                return panel.reset_index(drop=True), sources

    # ---------------------------------------------------------------- monthly
    elif frequency == "monthly":
        if m1_monthly is not None and rate_daily is not None and len(gdp_qts) > 0:
            rate_m = rate_daily.resample("ME").mean() / 100   # monthly avg → decimal

            # Interpolate quarterly GDP to monthly using cubic spline
            # Reindex to month-end frequency first, then interpolate
            gdp_full_idx = pd.date_range(
                start=gdp_qts.index.min(),
                end=gdp_qts.index.max(),
                freq="ME",
            )
            gdp_m_interp = (
                gdp_qts
                .reindex(gdp_full_idx.union(gdp_qts.index))
                .sort_index()
                .interpolate(method="cubic")
                .reindex(gdp_full_idx)
            )

            common = (
                m1_monthly.index
                .intersection(rate_m.index)
                .intersection(gdp_m_interp.index)
            )
            # Annualise GDP: monthly flow × 12
            panel = pd.DataFrame({
                "period": common,
                "year":   common.year,
                "m1_bn":  m1_monthly.reindex(common).values / 1000,
                "gdp_bn": gdp_m_interp.reindex(common).values * 12 / 1000,
                "i":      rate_m.reindex(common).values,
            })
            panel["z"] = panel["m1_bn"] / panel["gdp_bn"]
            panel = panel.dropna(subset=["z","i"]).query("z>0 and i>0")
            if len(panel) >= 10:
                return panel.reset_index(drop=True), sources

    raise RuntimeError(
        f"Could not build the {frequency} panel.\n\n"
        "Expected data files in:\n"
        f"  {_RES}\n"
        f"  or {_RES_FALLBACK}\n\n"
        "Required:\n"
        f"  • {_BOE_M1_NAME}  (M1 monthly)\n"
        f"  • {_BOE_RATE_NAME}   (Bank Rate daily)\n\n"
        "GDP is fetched live from the ONS API."
    )


def filter_panel(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Slice panel to [start_year, end_year], requiring i > 0 and z > 0."""
    mask = (df["year"] >= start_year) & (df["year"] <= end_year)
    return df.loc[mask].query("i > 0 and z > 0").reset_index(drop=True)
