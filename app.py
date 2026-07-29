"""
app.py — Streamlit interactive dashboard for the Welfare Cost of Inflation thesis.

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import Frequency, build_panel, filter_panel
from models import (
    BENCHMARK_RATES,
    INTERNATIONAL,
    LogLogResult,
    SemiLogResult,
    fit_loglog,
    fit_semilog,
    marginal_gains_table,
    welfare_loglog,
    welfare_semilog,
    welfare_table,
    z_loglog,
    z_semilog,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Welfare Cost of Inflation — UK",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

C_LL  = "#1f77b4"   # blue  — log-log
C_SL  = "#d62728"   # red   — semi-log
C_REF = "#888888"   # grey  — reference / axis

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Controls")

    if st.button("Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # -- Frequency --
    st.subheader("Data frequency")
    freq_label = st.radio(
        "Observation frequency",
        options=["Annual", "Quarterly", "Monthly"],
        index=0,
        help=(
            "Annual: classic Bailey-Lucas long-run approach.\n"
            "Quarterly / Monthly: more observations; z is annualised "
            "(GDP × 4 or × 12) so it stays comparable across frequencies."
        ),
    )
    frequency: Frequency = freq_label.lower()  # type: ignore[assignment]

    st.divider()

    # -- Year range --
    st.subheader("Sample period")
    year_range = st.slider(
        "Year range",
        min_value=1963,
        max_value=2026,
        value=(1980, 2024),
        step=1,
        help="OLS estimation is re-run on this sub-sample.",
    )
    start_yr, end_yr = year_range

    st.divider()

    # -- Welfare calculator --
    st.subheader("Welfare calculator")
    i_pct = st.slider(
        "Nominal user cost  i  (%)",
        min_value=0.1,
        max_value=15.0,
        value=2.0,
        step=0.1,
        format="%.1f%%",
    )
    i_val = i_pct / 100.0

    st.divider()

    # -- Model display --
    st.subheader("Specifications shown")
    show_ll = st.checkbox("Log-log",  value=True)
    show_sl = st.checkbox("Semi-log", value=True)

    st.divider()
    st.caption(
        "Data: Bank of England (M1, Bank Rate) · ONS (GDP YBHA)\n\n"
        "Methodology: Bailey (1956), Lucas (2000)"
    )

# ---------------------------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _get_full_panel(freq: str):
    return build_panel(freq)  # type: ignore[arg-type]

with st.spinner("Fetching data…"):
    try:
        full_panel, sources = _get_full_panel(frequency)
    except RuntimeError as err:
        st.error(str(err))
        st.stop()

panel = filter_panel(full_panel, start_yr, end_yr)

if len(panel) < 5:
    st.warning(
        f"Only {len(panel)} observations in {start_yr}–{end_yr}. "
        "Widen the year range to get a meaningful estimate."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Fit models
# ---------------------------------------------------------------------------
try:
    ll_fit = fit_loglog(panel)
    sl_fit = fit_semilog(panel)
except Exception as e:
    st.error(f"Estimation failed: {e}")
    st.stop()

ll_valid = -1.0 < ll_fit.g < 0.0
sl_valid = sl_fit.epsilon > 0.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_w(val: float) -> str:
    return f"{val*100:.5f}%" if abs(val) < 0.0001 else f"{val*100:.4f}%"

def _period_label(ts: pd.Timestamp, freq: Frequency) -> str:
    if freq == "quarterly":
        q = (ts.month - 1) // 3 + 1
        return f"{ts.year} Q{q}"
    if freq == "monthly":
        return ts.strftime("%b %Y")
    return str(ts.year)

# Human-readable period labels for the current panel
period_labels = panel["period"].apply(lambda t: _period_label(t, frequency))

# ---------------------------------------------------------------------------
# Title & badges
# ---------------------------------------------------------------------------
st.title("Welfare Cost of Anticipated Inflation — United Kingdom")

col_badge, col_n, col_yr, col_freq = st.columns([2, 1, 2, 1])
with col_badge:
    gdp_src = sources.get("gdp", "none")
    if gdp_src == "live":
        st.success("GDP: live · ONS API   |   M1 & Rate: local BoE files")
    else:
        st.warning("Local fallback data  ·  pre-downloaded CSVs")
with col_n:
    st.metric("Observations", f"N = {len(panel)}")
with col_yr:
    st.metric("Sample", f"{int(panel['year'].min())} – {int(panel['year'].max())}")
with col_freq:
    st.metric("Frequency", freq_label)

# Caveat for sub-annual frequencies
if frequency != "annual":
    st.info(
        f"**{freq_label} mode** — The Bailey-Lucas framework is a long-run "
        "steady-state model. At {freq_label.lower()} frequency, estimates "
        "reflect short-run noise and seasonality in money demand; interpret "
        "elasticities and welfare costs as indicative rather than structural.",
        icon="ℹ️",
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 1 — Data overview
# ---------------------------------------------------------------------------
st.subheader("1 · Data overview")

has_extras = {"m1_bn", "gdp_bn"}.issubset(panel.columns)
marker_size = 2 if frequency == "monthly" else (3 if frequency == "quarterly" else 5)

def _ts_fig(x, y, color, title, ytitle, height=260):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color=color, width=1.5),
        marker=dict(size=marker_size),
    ))
    fig.update_layout(
        title=title, xaxis_title="Period", yaxis_title=ytitle,
        height=height, margin=dict(l=40, r=10, t=40, b=30),
        showlegend=False,
    )
    return fig

if has_extras:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(
            _ts_fig(panel["period"], panel["m1_bn"], C_LL,
                    "UK M1 — end-of-period stock (£bn)", "M1 (£bn)"),
            use_container_width=True,
        )
    with c2:
        gdp_title = (
            "Nominal GDP — annualised (£bn)"
            if frequency != "annual"
            else "Nominal GDP — market prices (£bn)"
        )
        st.plotly_chart(
            _ts_fig(panel["period"], panel["gdp_bn"], C_SL, gdp_title, "GDP (£bn)"),
            use_container_width=True,
        )
    with c3:
        st.plotly_chart(
            _ts_fig(panel["period"], panel["z"], "#2ca02c",
                    "Money-income ratio  z = M1 / annualised GDP", "z"),
            use_container_width=True,
        )
else:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _ts_fig(panel["period"], panel["z"], "#2ca02c",
                    "Money-income ratio  z = M1 / Y", "z", height=280),
            use_container_width=True,
        )
    with c2:
        rate_title = {
            "annual": "Bank Rate — annual average (%)",
            "quarterly": "Bank Rate — quarterly average (%)",
            "monthly": "Bank Rate — monthly average (%)",
        }[frequency]
        st.plotly_chart(
            _ts_fig(panel["period"], panel["i"] * 100, C_LL,
                    rate_title, "Bank Rate (%)", height=280),
            use_container_width=True,
        )

with st.expander("Summary statistics"):
    stats = panel[["z", "i"]].copy()
    stats["i_pct"] = stats["i"] * 100
    desc = stats[["z", "i_pct"]].rename(columns={"i_pct": "i (%)"}).describe().T
    desc.index.name = "Variable"
    st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Money-demand estimation
# ---------------------------------------------------------------------------
st.subheader("2 · Money-demand estimation")

ec1, ec2, ec3, ec4 = st.columns(4)
if show_ll and ll_valid:
    with ec1:
        st.metric("Log-log  g", f"{ll_fit.g:.4f}", help="Interest elasticity; (−1, 0)")
    with ec2:
        st.metric("Log-log  A", f"{ll_fit.A:.4f}")
if show_sl and sl_valid:
    with ec3:
        st.metric("Semi-log  ε", f"{sl_fit.epsilon:.4f}", help="Semi-elasticity; > 0")
    with ec4:
        st.metric("Semi-log  B", f"{sl_fit.B:.4f}")

model_rows = []
if show_ll:
    model_rows.append({
        "Specification": "Log-log",
        "Intercept": f"{ll_fit.alpha:.4f}",
        "Elasticity / Semi-elast.": f"{ll_fit.g:.4f}",
        "SE": f"{ll_fit.se_g:.4f}",
        "t": f"{ll_fit.t_g:.2f}",
        "p": f"{ll_fit.p_g:.5f}",
        "R²": f"{ll_fit.r2:.3f}",
        "Adj. R²": f"{ll_fit.r2_adj:.3f}",
        "N": ll_fit.n,
    })
if show_sl:
    model_rows.append({
        "Specification": "Semi-log",
        "Intercept": f"{sl_fit.beta:.4f}",
        "Elasticity / Semi-elast.": f"{sl_fit.epsilon:.4f}",
        "SE": f"{sl_fit.se_eps:.4f}",
        "t": f"{sl_fit.t_eps:.2f}",
        "p": f"{sl_fit.p_eps:.5f}",
        "R²": f"{sl_fit.r2:.3f}",
        "Adj. R²": f"{sl_fit.r2_adj:.3f}",
        "N": sl_fit.n,
    })
if model_rows:
    st.dataframe(pd.DataFrame(model_rows).set_index("Specification"), use_container_width=True)

# Money-demand scatter + fitted curves
actual_min = int(panel["year"].min())
actual_max = int(panel["year"].max())
i_grid = np.linspace(0.001, 0.16, 300)
fig_md = go.Figure()
fig_md.add_trace(go.Scatter(
    x=panel["i"] * 100, y=panel["z"],
    mode="markers",
    marker=dict(color="#555", size=max(2, marker_size), symbol="circle", opacity=0.6),
    name="Observed  z = M1/Y",
    hovertemplate="Period: %{text}<br>i: %{x:.2f}%<br>z: %{y:.4f}",
    text=period_labels,
))
if show_ll and ll_valid:
    fig_md.add_trace(go.Scatter(
        x=i_grid * 100, y=z_loglog(i_grid, ll_fit),
        mode="lines", line=dict(color=C_LL, width=2),
        name=f"Log-log  (g={ll_fit.g:.4f}, R²={ll_fit.r2:.3f})",
    ))
if show_sl and sl_valid:
    fig_md.add_trace(go.Scatter(
        x=i_grid * 100, y=z_semilog(i_grid, sl_fit),
        mode="lines", line=dict(color=C_SL, width=2, dash="dash"),
        name=f"Semi-log  (ε={sl_fit.epsilon:.4f}, R²={sl_fit.r2:.3f})",
    ))
fig_md.add_vline(
    x=i_pct, line_dash="dot", line_color=C_REF, line_width=1.5,
    annotation_text=f"i = {i_pct:.1f}%",
    annotation_position="top right",
)
fig_md.update_layout(
    title=f"UK money demand  z = M1/Y  vs  Bank Rate  ({actual_min}–{actual_max}, {freq_label})",
    xaxis_title="User cost  i  (%)",
    yaxis_title="Money-income ratio  z = M1/Y",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="left", x=0),
    margin=dict(l=50, r=20, t=50, b=80),
)
st.plotly_chart(fig_md, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Welfare cost explorer
# ---------------------------------------------------------------------------
st.subheader("3 · Welfare cost explorer")

w_ll = welfare_loglog(i_val, ll_fit) if (show_ll and ll_valid) else None
w_sl = welfare_semilog(i_val, sl_fit) if (show_sl and sl_valid) else None

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("User cost  i", f"{i_pct:.1f}%")
if w_ll is not None:
    with metric_cols[1]:
        st.metric("W(i) — Log-log", fmt_w(w_ll), help="Welfare cost as % of income")
if w_sl is not None:
    with metric_cols[2]:
        st.metric("W(i) — Semi-log", fmt_w(w_sl), help="Welfare cost as % of income")
if w_ll is not None and w_sl is not None and w_sl > 0:
    with metric_cols[3]:
        st.metric("Log-log / Semi-log ratio", f"{w_ll/w_sl:.1f}×")

i_plot = np.linspace(0.001, 0.15, 500)
fig_w = go.Figure()
if show_ll and ll_valid:
    fig_w.add_trace(go.Scatter(
        x=i_plot * 100, y=welfare_loglog(i_plot, ll_fit) * 100,
        mode="lines", line=dict(color=C_LL, width=2), name="Log-log W(i)",
    ))
if show_sl and sl_valid:
    fig_w.add_trace(go.Scatter(
        x=i_plot * 100, y=welfare_semilog(i_plot, sl_fit) * 100,
        mode="lines", line=dict(color=C_SL, width=2, dash="dash"), name="Semi-log W(i)",
    ))
for bm in BENCHMARK_RATES:
    if show_ll and ll_valid:
        fig_w.add_trace(go.Scatter(
            x=[bm * 100], y=[welfare_loglog(bm, ll_fit) * 100],
            mode="markers", marker=dict(color=C_LL, size=8), showlegend=False,
            hovertemplate=f"i={bm*100:.0f}%<br>W={welfare_loglog(bm, ll_fit)*100:.5f}%",
        ))
    if show_sl and sl_valid:
        fig_w.add_trace(go.Scatter(
            x=[bm * 100], y=[welfare_semilog(bm, sl_fit) * 100],
            mode="markers", marker=dict(color=C_SL, size=8, symbol="circle-open"),
            showlegend=False,
            hovertemplate=f"i={bm*100:.0f}%<br>W={welfare_semilog(bm, sl_fit)*100:.5f}%",
        ))
fig_w.add_vline(
    x=i_pct, line_dash="dot", line_color=C_REF, line_width=1.5,
    annotation_text=f"i = {i_pct:.1f}%", annotation_position="top right",
)
fig_w.update_layout(
    title="Bailey-Lucas welfare cost  W(i)  — share of income",
    xaxis_title="Nominal user cost  i  (%)",
    yaxis_title="W(i)  (% of income)",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="left", x=0),
    margin=dict(l=50, r=20, t=50, b=70),
)
st.plotly_chart(fig_w, use_container_width=True)

with st.expander("Welfare costs at benchmark rates"):
    tbl = welfare_table(ll_fit, sl_fit)
    cols_show = ["User cost i (%)"]
    if show_ll and ll_valid: cols_show.append("Log-log W(i) % income")
    if show_sl and sl_valid: cols_show.append("Semi-log W(i) % income")
    st.dataframe(
        tbl[cols_show].style.format({c: "{:.5f}" for c in cols_show if "W(" in c}),
        use_container_width=True, hide_index=True,
    )

with st.expander("Welfare gains from disinflation"):
    gain_tbl = marginal_gains_table(ll_fit, sl_fit)
    cols_gain = ["Reduction"]
    if show_ll and ll_valid: cols_gain.append("Log-log gain (% income)")
    if show_sl and sl_valid: cols_gain.append("Semi-log gain (% income)")
    st.dataframe(
        gain_tbl[cols_gain].style.format({c: "{:.5f}" for c in cols_gain if "gain" in c}),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — International comparison
# ---------------------------------------------------------------------------
st.subheader("4 · International comparison")
st.caption("Benchmarks use annual estimates from the literature. UK values below reflect the currently selected frequency and year range.")

def _intl_with_uk(ll: LogLogResult, sl: SemiLogResult) -> pd.DataFrame:
    df = INTERNATIONAL.copy()
    for idx, row in df.iterrows():
        if row["note"] == "live":
            iv = row["i_pct"] / 100.0
            if "log-log" in row["Country"].lower() and ll_valid:
                df.at[idx, "W_pct"] = welfare_loglog(iv, ll) * 100
            elif "semi-log" in row["Country"].lower() and sl_valid:
                df.at[idx, "W_pct"] = welfare_semilog(iv, sl) * 100
    return df.dropna(subset=["W_pct"])

intl_df = _intl_with_uk(ll_fit, sl_fit)

for i_ref in [2, 10]:
    sub = intl_df[intl_df["i_pct"] == i_ref].sort_values("W_pct", ascending=True)
    if sub.empty:
        continue
    colors = []
    for _, row in sub.iterrows():
        if "this study" in row["Country"] and "log-log" in row["Country"].lower():
            colors.append(C_LL)
        elif "this study" in row["Country"] and "semi-log" in row["Country"].lower():
            colors.append(C_SL)
        elif row["Group"] == "Advanced":
            colors.append("#aec7e8")
        else:
            colors.append("#ffbb78")
    fig_int = go.Figure(go.Bar(
        x=sub["W_pct"], y=sub["Country"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>W(%{customdata}%) = %{x:.4f}% of income",
        customdata=sub["i_pct"],
    ))
    fig_int.update_layout(
        title=f"Welfare cost at i = {i_ref}%  (% of income)",
        xaxis_title="W(i)  (% of income)", yaxis_title="",
        height=360, margin=dict(l=220, r=20, t=50, b=40),
        xaxis=dict(type="log" if i_ref == 10 else "linear"),
    )
    st.plotly_chart(fig_int, use_container_width=True)
    st.caption(
        "Sources: Lucas (2000), Ireland (2009), Attanasio et al. (2002), "
        "Serletis & Yavari (2004), Tümtürk (2017), Shah et al. (2019), "
        "Campos & Cysne (2019), Chen & Ma (2007)."
        + ("  · x-axis: log scale" if i_ref == 10 else "")
    )

st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    "**Ruo-Hao Hsin** · UCL Data Science MSc · Supervisor: Dr. Cemil Selcuk  \n"
    "Bailey-Lucas consumer-surplus framework · Bank of England + ONS  \n"
    "W(i) = net area under inverse money-demand curve between 0 and i"
)
