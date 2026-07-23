"""
data.py — Data fetching for the Welfare Cost of Inflation app.

Strategy (in priority order):
  1. ONS Timeseries API  — GDP (YBHA)           live JSON, no auth required
  2. Local BoE files     — M1 (LPMAVAA .xlsx)   pre-downloaded from BoE database
                         — Bank Rate (IUDBEDR .csv)
  3. Merged panel CSV    — fallback if individual files not found

The BoE Statistics Database does not support unauthenticated CSV downloads;
all three series are read from local files kept in the sibling
Welfare_Cost_of_Inflation_UK/resources/ folder.

To refresh BoE data manually:
  1. Visit https://www.bankofengland.co.uk/boeapps/database/
  2. Search for LPMAVAA (M1) and IUDBEDR (Bank Rate)
  3. Export as CSV/Excel and save to the resources folder
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_RES = _HERE.parent / "Welfare_Cost_of_Inflation_UK" / "resources"

# BoE source files (pre-downloaded)
_BOE_M1_XLSX = _RES / "LPMAVAA  Bank of England  Database.xlsx"
_BOE_RATE_CSV = _RES / "IUDBEDR  Bank of England  Database.csv"

# ONS UK Economic Accounts — GDP (YBHA)
_ONS_GDP_URL = (
    "https://www.ons.gov.uk/economy/grossdomesticproductgdp"
    "/timeseries/ybha/data"
)

# Pre-built merged panel (final fallback)
_MERGED_CSV = _RES / "uk_merged_data_for_lucas.csv"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# ONS GDP — live fetch
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ons_gdp() -> tuple[pd.Series, bool]:
    """
    Returns (series, is_live) where series is indexed by integer year
    and values are nominal GDP in £ millions.
    """
    try:
        r = requests.get(_ONS_GDP_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        payload = r.json()
        years_data = payload.get("years", [])
        if not years_data:
            raise ValueError("Empty 'years' in ONS response.")
        records = {
            int(row["year"]): float(row["value"])
            for row in years_data
            if row.get("value") not in ("", None, ".")
        }
        s = pd.Series(records, name="gdp_m").sort_index()
        return s, True
    except Exception:
        return pd.Series(dtype=float), False


# ---------------------------------------------------------------------------
# BoE M1 — local file
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_boe_m1() -> pd.Series | None:
    """
    Read BoE LPMAVAA Excel export (calamine engine) → monthly M1 in £ millions.
    Format: row 0 = title, row 1 = column headers, row 2+ = "DD Mon YY", value.
    Returns annual end-December series indexed by integer year (£ millions).
    """
    if not _BOE_M1_XLSX.exists():
        return None
    try:
        raw = pd.read_excel(
            _BOE_M1_XLSX, header=None, skiprows=2, engine="calamine",
            names=["date", "value"],
            usecols=[0, 1],
        )
        raw["date"] = pd.to_datetime(
            raw["date"].astype(str).str.strip(),
            format="%d %b %y",
            errors="coerce",
        )
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw = raw.dropna().set_index("date")["value"].sort_index()
        # Take end-of-month (last) value each calendar year (end-December stock)
        ann = raw.resample("YE").last()
        ann.index = ann.index.year
        ann.name = "m1_m"
        return ann
    except Exception:
        return None


# ---------------------------------------------------------------------------
# BoE Bank Rate — local file
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_boe_rate() -> pd.Series | None:
    """
    Read BoE IUDBEDR CSV export → daily Bank Rate in %, averaged to annual decimal.
    Format: header row, then "DD Mon YY", value — all fields quoted.
    Returns annual series indexed by integer year (decimal, e.g. 0.05).
    """
    if not _BOE_RATE_CSV.exists():
        return None
    try:
        df = pd.read_csv(
            _BOE_RATE_CSV,
            header=0,
            names=["date", "value"],
            quotechar='"',
            skipinitialspace=True,
        )
        df["date"] = pd.to_datetime(
            df["date"].astype(str).str.strip(),
            format="%d %b %y",
            errors="coerce",
        )
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().set_index("date")["value"].sort_index()
        # Annual calendar-year average
        ann = df.resample("YE").mean()
        ann.index = ann.index.year
        # Values are in %; convert to decimal
        ann = ann / 100
        ann.name = "i"
        return ann
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Merged-panel fallback
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_merged_panel() -> pd.DataFrame | None:
    if not _MERGED_CSV.exists():
        return None
    try:
        df = pd.read_csv(_MERGED_CSV)
        df = df[["year","z","i"]].dropna()
        df["year"] = df["year"].astype(int)
        df["z"] = pd.to_numeric(df["z"], errors="coerce")
        df["i"] = pd.to_numeric(df["i"], errors="coerce")
        return df.dropna().query("z > 0 and i > 0").sort_values("year").reset_index(drop=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_panel() -> tuple[pd.DataFrame, dict]:
    """
    Build the annual estimation panel.

    Returns
    -------
    panel : pd.DataFrame
        Columns: year (int), z (float), i (float),
                 m1_bn (float, optional), gdp_bn (float, optional)
    sources : dict
        {"gdp": "live"|"local"|"none",
         "m1":  "live"|"local"|"none",
         "rate":"live"|"local"|"none"}
    """
    sources: dict[str, str] = {"gdp": "none", "m1": "none", "rate": "none"}

    # --- GDP from ONS API ---
    gdp_series, gdp_live = _fetch_ons_gdp()
    if gdp_live and len(gdp_series) > 0:
        sources["gdp"] = "live"
    else:
        gdp_series = pd.Series(dtype=float)

    # --- M1 from local BoE file ---
    m1_series = _load_boe_m1()
    if m1_series is not None and len(m1_series) > 0:
        sources["m1"] = "local"

    # --- Bank Rate from local BoE file ---
    rate_series = _load_boe_rate()
    if rate_series is not None and len(rate_series) > 0:
        sources["rate"] = "local"

    # --- Assemble panel if we have all three ---
    if (
        len(gdp_series) > 0
        and m1_series is not None
        and rate_series is not None
    ):
        years = sorted(
            set(gdp_series.index)
            & set(m1_series.index)
            & set(rate_series.index)
        )
        if len(years) >= 10:
            gdp_bn = gdp_series.reindex(years) / 1000   # £m → £bn
            m1_bn = m1_series.reindex(years) / 1000     # £m → £bn
            i_dec = rate_series.reindex(years)

            panel = pd.DataFrame({
                "year":   years,
                "m1_bn":  m1_bn.values,
                "gdp_bn": gdp_bn.values,
                "i":      i_dec.values,
            })
            panel["z"] = panel["m1_bn"] / panel["gdp_bn"]
            panel = (
                panel.dropna(subset=["z","i"])
                .query("z > 0 and i > 0")
                .reset_index(drop=True)
            )
            return panel, sources

    # --- If M1 or rate missing, try merged panel fallback ---
    merged = _load_merged_panel()
    if merged is not None:
        sources = {k: "local" for k in sources}
        return merged, sources

    raise RuntimeError(
        "Could not build the estimation panel.\n\n"
        "Expected data files in:\n"
        f"  {_RES}\n\n"
        "Required:\n"
        "  • LPMAVAA  Bank of England  Database.xlsx  (M1 monthly)\n"
        "  • IUDBEDR  Bank of England  Database.csv   (Bank Rate daily)\n"
        "  • uk_merged_data_for_lucas.csv              (pre-built fallback)\n\n"
        "GDP is fetched live from the ONS API."
    )


def filter_panel(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Slice panel to [start_year, end_year], ensuring i > 0 and z > 0."""
    mask = (df["year"] >= start_year) & (df["year"] <= end_year)
    sub = df.loc[mask].copy()
    return sub.query("i > 0 and z > 0").reset_index(drop=True)
