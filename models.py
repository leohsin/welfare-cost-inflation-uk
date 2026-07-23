"""
models.py — OLS money-demand estimation and Bailey-Lucas welfare calculations.

Two specifications (Bailey 1956, Lucas 2000):
  Log-log:   log z = α + g·log(i)     →  z(i) = A·i^g
  Semi-log:  log z = β + γ·i          →  z(i) = B·exp(-ε·i)   where ε = -γ

Welfare cost (share of income):
  Log-log:   W(i) = -g/(1+g) · A · i^(1+g)
  Semi-log:  W(i) = B/ε · [1 - (1 + ε·i)·exp(-ε·i)]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class LogLogResult:
    alpha: float    # intercept in log-log
    g: float        # interest elasticity (negative)
    A: float        # scale = exp(alpha)
    r2: float
    r2_adj: float
    n: int
    se_g: float     # std error on g
    t_g: float
    p_g: float
    resid_std: float

    @property
    def label(self) -> str:
        return "Log-log"


@dataclass
class SemiLogResult:
    beta: float     # intercept in semi-log
    gamma: float    # coefficient on i (negative)
    epsilon: float  # semi-elasticity = -gamma (positive)
    B: float        # scale = exp(beta)
    r2: float
    r2_adj: float
    n: int
    se_eps: float
    t_eps: float
    p_eps: float
    resid_std: float

    @property
    def label(self) -> str:
        return "Semi-log"


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def fit_loglog(df: pd.DataFrame) -> LogLogResult:
    """
    OLS: log(z) = alpha + g*log(i)
    Requires columns 'z' and 'i' (both > 0).
    """
    sub = df[["z", "i"]].dropna().query("z > 0 and i > 0").copy()
    log_z = np.log(sub["z"])
    log_i = np.log(sub["i"])
    X = sm.add_constant(log_i)
    res = sm.OLS(log_z, X).fit()

    alpha = float(res.params["const"])
    g = float(res.params["i"])
    return LogLogResult(
        alpha=alpha,
        g=g,
        A=float(np.exp(alpha)),
        r2=float(res.rsquared),
        r2_adj=float(res.rsquared_adj),
        n=int(res.nobs),
        se_g=float(res.bse["i"]),
        t_g=float(res.tvalues["i"]),
        p_g=float(res.pvalues["i"]),
        resid_std=float(np.sqrt(res.mse_resid)),
    )


def fit_semilog(df: pd.DataFrame) -> SemiLogResult:
    """
    OLS: log(z) = beta + gamma*i
    Requires columns 'z' and 'i' (z > 0, i >= 0).
    """
    sub = df[["z", "i"]].dropna().query("z > 0 and i >= 0").copy()
    log_z = np.log(sub["z"])
    X = sm.add_constant(sub["i"])
    res = sm.OLS(log_z, X).fit()

    beta = float(res.params["const"])
    gamma = float(res.params["i"])
    epsilon = float(-gamma)
    return SemiLogResult(
        beta=beta,
        gamma=gamma,
        epsilon=epsilon,
        B=float(np.exp(beta)),
        r2=float(res.rsquared),
        r2_adj=float(res.rsquared_adj),
        n=int(res.nobs),
        se_eps=float(res.bse["i"]),
        t_eps=float(res.tvalues["i"]),
        p_eps=float(res.pvalues["i"]),
        resid_std=float(np.sqrt(res.mse_resid)),
    )


# ---------------------------------------------------------------------------
# Welfare formulas
# ---------------------------------------------------------------------------

def welfare_loglog(i: float | np.ndarray, fit: LogLogResult) -> float | np.ndarray:
    """
    W(i) = -g/(1+g) * A * i^(1+g),  valid for g in (-1, 0) and i >= 0.
    Returns 0 when i = 0.
    """
    g, A = fit.g, fit.A
    if not (-1.0 < g < 0.0):
        raise ValueError(f"Log-log elasticity g={g:.4f} outside valid range (-1, 0).")
    i_arr = np.atleast_1d(np.asarray(i, dtype=float))
    out = (-g / (1.0 + g)) * A * np.where(i_arr > 0, i_arr ** (1.0 + g), 0.0)
    return float(out[0]) if np.ndim(i) == 0 else out


def welfare_semilog(i: float | np.ndarray, fit: SemiLogResult) -> float | np.ndarray:
    """
    W(i) = B/ε * [1 - (1 + ε·i)·exp(-ε·i)],  valid for ε > 0, i >= 0.
    """
    eps, B = fit.epsilon, fit.B
    if eps <= 0:
        raise ValueError(f"Semi-elasticity ε={eps:.4f} must be positive.")
    i_arr = np.atleast_1d(np.asarray(i, dtype=float))
    out = (B / eps) * (1.0 - (1.0 + eps * i_arr) * np.exp(-eps * i_arr))
    out = np.where(i_arr >= 0, out, 0.0)
    return float(out[0]) if np.ndim(i) == 0 else out


# ---------------------------------------------------------------------------
# Demand curve helpers
# ---------------------------------------------------------------------------

def z_loglog(i: np.ndarray, fit: LogLogResult) -> np.ndarray:
    """Fitted log-log money demand: z(i) = A * i^g."""
    return fit.A * np.where(i > 0, i ** fit.g, np.nan)


def z_semilog(i: np.ndarray, fit: SemiLogResult) -> np.ndarray:
    """Fitted semi-log money demand: z(i) = B * exp(-ε·i)."""
    return fit.B * np.exp(-fit.epsilon * i)


# ---------------------------------------------------------------------------
# Welfare table at benchmark rates
# ---------------------------------------------------------------------------

BENCHMARK_RATES = [0.01, 0.02, 0.05, 0.08, 0.10]


def welfare_table(ll: LogLogResult, sl: SemiLogResult) -> pd.DataFrame:
    rows = []
    for rate in BENCHMARK_RATES:
        rows.append({
            "User cost i (%)": f"{rate*100:.0f}%",
            "Log-log W(i) % income": welfare_loglog(rate, ll) * 100,
            "Semi-log W(i) % income": welfare_semilog(rate, sl) * 100,
        })
    return pd.DataFrame(rows)


def marginal_gains_table(ll: LogLogResult, sl: SemiLogResult) -> pd.DataFrame:
    """Welfare gains from reducing i to the next lower benchmark."""
    pairs = [
        ("5% → 2%", 0.05, 0.02),
        ("2% → 1%", 0.02, 0.01),
        ("2% → 0%", 0.02, 0.00),
        ("8% → 2%", 0.08, 0.02),
        ("10% → 2%", 0.10, 0.02),
    ]
    rows = []
    for label, i_high, i_low in pairs:
        ll_gain = (welfare_loglog(i_high, ll) - welfare_loglog(i_low, ll)) * 100
        sl_gain = (welfare_semilog(i_high, sl) - welfare_semilog(i_low, sl)) * 100
        rows.append({
            "Reduction": label,
            "Log-log gain (% income)": ll_gain,
            "Semi-log gain (% income)": sl_gain,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# International comparison (hardcoded from thesis Tables 6 & 7)
# ---------------------------------------------------------------------------

INTERNATIONAL = pd.DataFrame([
    # Advanced economies
    {"Country": "UK (log-log, this study)",  "Group": "Advanced", "i_pct": 2,  "W_pct": None, "note": "live"},
    {"Country": "UK (semi-log, this study)", "Group": "Advanced", "i_pct": 2,  "W_pct": None, "note": "live"},
    {"Country": "US — Ireland (2009)",        "Group": "Advanced", "i_pct": 2,  "W_pct": 0.037, "note": ""},
    {"Country": "US — Ireland (2009)",        "Group": "Advanced", "i_pct": 10, "W_pct": 0.228, "note": ""},
    {"Country": "Italy — Attanasio (2002)",   "Group": "Advanced", "i_pct": 2,  "W_pct": 0.050, "note": "approx"},
    {"Country": "Canada — Serletis (2004)",   "Group": "Advanced", "i_pct": 3,  "W_pct": 0.150, "note": "3%→0%"},
    # Emerging
    {"Country": "Turkey — Tümtürk (2017)",    "Group": "Emerging", "i_pct": 10, "W_pct": 0.530, "note": "GDP share"},
    {"Country": "India — Shah (2019)",         "Group": "Emerging", "i_pct": 10, "W_pct": 0.530, "note": "GDP share"},
    {"Country": "Brazil — Campos (2019)",      "Group": "Emerging", "i_pct": 10, "W_pct": 0.375, "note": ""},
    {"Country": "China — Chen & Ma (2007)",    "Group": "Emerging", "i_pct": 10, "W_pct": 9.690, "note": "semi-log"},
])
